from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .python_import_paths import compatible_engine_site_packages
from .runtime.config import RuntimeConfig

LOCAL_MOJO_WARMUP_PAYLOAD = {
    "engine_type": "monte_carlo",
    "runs": 100,
    "seed": 1,
    "variables": [
        {"name": "x", "distribution": "fixed", "params": {"value": 100.0}},
    ],
    "objective_function": "x",
    "correlations": [],
    "scoring": {"expected_value": 1.0, "downside_risk": 0.0},
}

LOCAL_MOJO_POOL_LOCK = threading.Lock()
LOCAL_MOJO_POOL_STATE: LocalMojoPoolState | None = None
LOCAL_MOJO_POOL_ATTEMPTED = False
LOCAL_MOJO_POOL_START_EVENT: threading.Event | None = None
LOCAL_MOJO_POOL_START_ERROR: BaseException | None = None
LOCAL_MOJO_WORKER_STATE_SCHEMA_VERSION = 1
LEGACY_GLOBAL_MOJO_WORKER_MARKER = "/algenta-mojo/worker-"
LEGACY_GLOBAL_MOJO_WORKER_CLEANUP_COMMAND = (
    "pids=$(lsof -nP -U | awk '$1==\"simulate\" && $0 ~ "
    "/\\/algenta-mojo\\/worker-/ {print $2}' | sort -u); "
    '[ -n "$pids" ] && printf \'%s\\n\' "$pids" | xargs kill'
)


class LocalMojoPoolError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class LocalMojoPoolState:
    pool: Any
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    set_pool: Any
    config: RuntimeConfig


def env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def positive_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise LocalMojoPoolError(
            "invalid_local_mojo_pool_config",
            f"{name} must be a positive integer.",
            {"name": name, "value": raw},
        ) from None
    if value <= 0:
        raise LocalMojoPoolError(
            "invalid_local_mojo_pool_config",
            f"{name} must be a positive integer.",
            {"name": name, "value": raw},
        )
    return value


def local_mojo_pool_size() -> int:
    if os.environ.get("CODNA_LOCAL_MOJO_POOL_SIZE"):
        return positive_int_env("CODNA_LOCAL_MOJO_POOL_SIZE", default=1)
    return positive_int_env("MOJO_POOL_SIZE", default=1)


def local_mojo_pool_start_timeout_seconds() -> float:
    raw = os.environ.get("CODNA_LOCAL_MOJO_POOL_START_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return 45.0
    return positive_float_env(
        "CODNA_LOCAL_MOJO_POOL_START_TIMEOUT_SECONDS",
        raw,
    )


def local_mojo_pool_warmup_timeout_seconds() -> float:
    raw = os.environ.get("CODNA_LOCAL_MOJO_POOL_WARMUP_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return 5.0
    return positive_float_env(
        "CODNA_LOCAL_MOJO_POOL_WARMUP_TIMEOUT_SECONDS",
        raw,
    )


def positive_float_env(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise LocalMojoPoolError(
            "invalid_local_mojo_pool_config",
            f"{name} must be a positive number.",
            {"value": raw},
        ) from None
    if value <= 0:
        raise LocalMojoPoolError(
            "invalid_local_mojo_pool_config",
            f"{name} must be a positive number.",
            {"value": raw},
        )
    return value


def local_mojo_pool_prewarm_enabled() -> bool:
    return env_bool("CODNA_LOCAL_MOJO_POOL_PREWARM", default=True)


def local_mojo_pool_compute_warmup_enabled() -> bool:
    return env_bool("CODNA_LOCAL_MOJO_POOL_COMPUTE_WARMUP", default=True)


def local_mojo_pool_stale_cleanup_enabled() -> bool:
    return env_bool("CODNA_LOCAL_MOJO_CLEANUP_STALE_WORKERS", default=True)


def run_local_mojo_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()


def stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    if loop.is_closed():
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)


def insert_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def install_local_privacy_stub() -> None:
    module_name = "apps.api_server.services.privacy_service"
    if sys.modules.get(module_name) is not None:
        return
    module = ModuleType(module_name)

    async def record_service_egress_event(**_kwargs: Any) -> None:
        return None

    async def get_egress_policy_payload() -> dict[str, Any]:
        return {"deployment_mode": "codna_local_offline", "audit": "local_noop"}

    async def list_egress_events(**_kwargs: Any) -> dict[str, Any]:
        return {"entries": [], "total": 0, "page": 1, "limit": 0}

    async def get_privacy_report(**_kwargs: Any) -> dict[str, Any]:
        return {"entries": [], "policy": await get_egress_policy_payload()}

    module.record_service_egress_event = record_service_egress_event
    module.get_egress_policy_payload = get_egress_policy_payload
    module.list_egress_events = list_egress_events
    module.get_privacy_report = get_privacy_report
    module.CODNA_LOCAL_PRIVACY_STUB = True
    sys.modules[module_name] = module


def ensure_decision_engine_imports(config: RuntimeConfig) -> None:
    engine_dir = config.engine_dir.expanduser().resolve()
    insert_import_path(engine_dir)
    insert_import_path(engine_dir / "packages" / "algenta-core")
    insert_import_path(engine_dir / "packages" / "python-sdk")
    for site_packages in compatible_engine_site_packages(engine_dir):
        insert_import_path(site_packages)
    install_local_privacy_stub()


def local_mojo_worker_root(config: RuntimeConfig) -> Path:
    return config.paths.root / "mojo-workers"


def local_mojo_socket_base_dir(config: RuntimeConfig) -> Path:
    socket_root = Path(os.environ.get("CODNA_LOCAL_MOJO_SOCKET_ROOT") or "/tmp")
    return socket_root.expanduser().resolve(strict=False) / f"codna-mojo-{config.runtime_config_hash[:8]}"


def local_mojo_socket_dir(config: RuntimeConfig) -> Path:
    return local_mojo_socket_base_dir(config) / f"p-{os.getpid()}"


def local_mojo_worker_state_path(config: RuntimeConfig) -> Path:
    return local_mojo_worker_root(config) / "workers.json"


def is_relative_to_path(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(
            parent.expanduser().resolve(strict=False)
        )
    except ValueError:
        return False
    return True


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_worker_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def pids_for_socket_path(socket_path: Path) -> list[int]:
    try:
        completed = subprocess.run(
            ["lsof", "-nP", "-t", str(socket_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode not in {0, 1}:
        return []
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def _legacy_worker_socket_root(raw_path: str) -> str | None:
    marker_index = raw_path.find(LEGACY_GLOBAL_MOJO_WORKER_MARKER)
    if marker_index < 0:
        return None
    return raw_path[: marker_index + len("/algenta-mojo")]


def _run_lsof_unix_socket_scan() -> tuple[str | None, tuple[str, ...]]:
    try:
        completed = subprocess.run(
            ["lsof", "-nP", "-U"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}", ()
    if completed.returncode not in {0, 1}:
        return completed.stderr.strip() or f"lsof exited {completed.returncode}", ()
    return None, tuple(completed.stdout.splitlines())


def legacy_global_mojo_worker_diagnostics(
    scan: tuple[str | None, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Report old default Mojo workers without taking ownership of them."""
    error, lines = scan if scan is not None else _run_lsof_unix_socket_scan()
    if error is not None:
        return {
            "status": "unavailable",
            "owned_by_current_runtime": False,
            "reason": error,
        }
    pids: set[int] = set()
    socket_paths: set[str] = set()
    socket_roots: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 2 or fields[0] != "simulate":
            continue
        raw_path = fields[-1]
        socket_root = _legacy_worker_socket_root(raw_path)
        if socket_root is None:
            continue
        if fields[1].isdigit():
            pids.add(int(fields[1]))
        socket_paths.add(raw_path)
        socket_roots.add(socket_root)
    if not pids and not socket_paths:
        return {
            "status": "clean",
            "owned_by_current_runtime": False,
            "process_count": 0,
            "socket_count": 0,
        }
    sample_pids = sorted(pids)[:10]
    return {
        "status": "legacy_workers_detected",
        "owned_by_current_runtime": False,
        "process_count": len(pids),
        "socket_count": len(socket_paths),
        "socket_roots": sorted(socket_roots),
        "sample_pids": sample_pids,
        "cleanup_policy": "manual_only_unowned_legacy",
        "reason": (
            "These are old default /algenta-mojo workers, not the current "
            "Codna-owned /codna-mojo workers. Codna reports them but does not "
            "kill them automatically."
        ),
        "manual_cleanup_command": LEGACY_GLOBAL_MOJO_WORKER_CLEANUP_COMMAND,
    }


def _lsof_socket_paths_under(
    root: Path,
    scan: tuple[str | None, tuple[str, ...]] | None = None,
) -> tuple[str | None, set[int], set[str]]:
    error, lines = scan if scan is not None else _run_lsof_unix_socket_scan()
    if error is not None:
        return error, set(), set()
    pids: set[int] = set()
    socket_paths: set[str] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        raw_path = fields[-1]
        socket_path = Path(raw_path)
        if not is_relative_to_path(socket_path, root):
            continue
        if fields[1].isdigit():
            pids.add(int(fields[1]))
        socket_paths.add(raw_path)
    return None, pids, socket_paths


def owned_local_mojo_worker_diagnostics(
    config: RuntimeConfig,
    scan: tuple[str | None, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Report workers owned by the current Codna runtime config without cleanup."""
    state_path = local_mojo_worker_state_path(config)
    socket_base = local_mojo_socket_base_dir(config)
    state = read_worker_state(state_path)
    error, live_pids, live_socket_paths = _lsof_socket_paths_under(socket_base, scan)
    if error is not None:
        return {
            "status": "unavailable",
            "owned_by_current_runtime": True,
            "reason": error,
            "state_path": str(state_path),
            "socket_base": str(socket_base),
        }
    workers = state.get("workers") if isinstance(state, dict) else None
    recorded_workers = workers if isinstance(workers, list) else []
    parent_pid = state.get("parent_pid") if isinstance(state, dict) else None
    parent_alive = pid_is_alive(parent_pid) if isinstance(parent_pid, int) else False
    payload = {
        "owned_by_current_runtime": True,
        "state_path": str(state_path),
        "state_present": bool(state),
        "socket_base": str(socket_base),
        "recorded_worker_count": len(recorded_workers),
        "live_process_count": len(live_pids),
        "live_socket_count": len(live_socket_paths),
        "sample_pids": sorted(live_pids)[:10],
        "parent_pid": parent_pid,
        "parent_alive": parent_alive,
    }
    if not state and not live_socket_paths:
        return {"status": "clean", **payload}
    if parent_alive and live_socket_paths:
        return {"status": "active_owned_workers", **payload}
    if state or live_socket_paths:
        return {
            "status": "stale_owned_workers",
            **payload,
            "cleanup_policy": "next_pool_start_or_process_shutdown",
            "reason": (
                "Codna-owned Mojo worker state or sockets exist without a live "
                "owning parent. The next local Mojo pool start will clean only "
                "these owned sockets."
            ),
        }
    return {"status": "clean", **payload}


def mojo_worker_diagnostics(config: RuntimeConfig) -> dict[str, Any]:
    scan = _run_lsof_unix_socket_scan()
    return {
        "owned_local": owned_local_mojo_worker_diagnostics(config, scan),
        "legacy_global": legacy_global_mojo_worker_diagnostics(scan),
    }


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_owned_pids(pids: list[int]) -> None:
    if not pids:
        return
    failures: list[dict[str, Any]] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            failures.append({"pid": pid, "error": str(exc)})
    if failures:
        raise LocalMojoPoolError(
            "local_mojo_pool_cleanup_failed",
            "Codna could not terminate a previously owned local Mojo worker.",
            {"failures": failures},
        )
    time.sleep(0.25)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            failures.append({"pid": pid, "error": str(exc)})
    if failures:
        raise LocalMojoPoolError(
            "local_mojo_pool_cleanup_failed",
            "Codna could not force-terminate a previously owned local Mojo worker.",
            {"failures": failures},
        )


def cleanup_stale_local_mojo_workers(config: RuntimeConfig) -> None:
    if not local_mojo_pool_stale_cleanup_enabled():
        return
    state = read_worker_state(local_mojo_worker_state_path(config))
    workers = state.get("workers")
    if not isinstance(workers, list):
        return
    parent_pid = state.get("parent_pid")
    if isinstance(parent_pid, int) and parent_pid != os.getpid() and pid_is_alive(parent_pid):
        return
    socket_base = local_mojo_socket_base_dir(config)
    raw_socket_root = state.get("socket_dir")
    socket_root = Path(raw_socket_root) if isinstance(raw_socket_root, str) else local_mojo_socket_dir(config)
    if not is_relative_to_path(socket_root, socket_base):
        return
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        raw_socket_path = worker.get("socket_path")
        if not isinstance(raw_socket_path, str) or not raw_socket_path:
            continue
        socket_path = Path(raw_socket_path)
        if not is_relative_to_path(socket_path, socket_root):
            continue
        terminate_owned_pids(pids_for_socket_path(socket_path))
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
    try:
        local_mojo_worker_state_path(config).unlink()
    except FileNotFoundError:
        pass
    try:
        socket_root.rmdir()
    except OSError:
        pass


def current_mojo_pool_module(config: RuntimeConfig) -> Any:
    ensure_decision_engine_imports(config)
    from apps.api_server.compute import mojo_pool

    return mojo_pool


def missing_local_mojo_backend(exc: BaseException) -> bool:
    return isinstance(exc, ModuleNotFoundError)


def local_mojo_backend_available(config: RuntimeConfig) -> bool:
    engine_dir = config.engine_dir.expanduser()
    if not (engine_dir / "apps" / "api_server" / "compute" / "mojo_pool.py").is_file():
        return False
    try:
        current_mojo_pool_module(config)
    except ModuleNotFoundError as exc:
        if missing_local_mojo_backend(exc):
            return False
        raise
    return True


def configure_local_mojo_socket_dir(config: RuntimeConfig) -> None:
    socket_dir = local_mojo_socket_dir(config)
    socket_dir.mkdir(parents=True, exist_ok=True)
    mojo_pool = current_mojo_pool_module(config)
    mojo_pool._SOCKET_DIR = socket_dir


def collect_pool_worker_records(pool: Any) -> list[dict[str, Any]]:
    workers: list[dict[str, Any]] = []
    for worker in getattr(pool, "_workers", []):
        proc = getattr(worker, "proc", None)
        pid = getattr(proc, "pid", None)
        socket_path = getattr(worker, "socket_path", None)
        if pid is None or socket_path is None:
            continue
        workers.append(
            {
                "id": getattr(worker, "id", None),
                "pid": int(pid),
                "socket_path": str(Path(socket_path)),
            }
        )
    return workers


def write_local_mojo_worker_state(config: RuntimeConfig, pool: Any) -> None:
    payload = {
        "schema_version": LOCAL_MOJO_WORKER_STATE_SCHEMA_VERSION,
        "owner": "codna-local-mojo-pool",
        "parent_pid": os.getpid(),
        "socket_dir": str(local_mojo_socket_dir(config)),
        "started_at": time.time(),
        "workers": collect_pool_worker_records(pool),
    }
    write_atomic_json(local_mojo_worker_state_path(config), payload)


def clear_local_mojo_worker_state(config: RuntimeConfig) -> None:
    try:
        local_mojo_worker_state_path(config).unlink()
    except FileNotFoundError:
        pass
    try:
        local_mojo_socket_dir(config).rmdir()
    except OSError:
        pass


def local_mojo_pool_unavailable_error(
    config: RuntimeConfig,
    message: str,
    *,
    reason: str | None = None,
) -> LocalMojoPoolError:
    details: dict[str, Any] = {"engine_dir": str(config.engine_dir)}
    if reason:
        details["reason"] = reason
    return LocalMojoPoolError("local_mojo_pool_unavailable", message, details)


def warm_local_mojo_pool(pool: Any, loop: asyncio.AbstractEventLoop) -> None:
    if not local_mojo_pool_compute_warmup_enabled():
        return
    timeout_s = local_mojo_pool_warmup_timeout_seconds()
    try:
        future = asyncio.run_coroutine_threadsafe(
            pool.invoke(dict(LOCAL_MOJO_WARMUP_PAYLOAD), timeout=timeout_s),
            loop,
        )
        result = future.result(timeout=timeout_s + 1.0)
    except Exception as exc:
        raise LocalMojoPoolError(
            "local_mojo_pool_warmup_failed",
            "Codna local Mojo pool started but failed its compute warmup.",
            {"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    try:
        mean = float((result.get("summary") or {}).get("mean") or 0.0)
    except (TypeError, ValueError) as exc:
        raise LocalMojoPoolError(
            "local_mojo_pool_warmup_failed",
            "Codna local Mojo pool warmup returned an invalid numeric summary.",
            {"result_type": type(result).__name__},
        ) from exc
    if mean == 0.0:
        raise LocalMojoPoolError(
            "local_mojo_pool_warmup_failed",
            "Codna local Mojo pool warmup returned zero mean for a fixed non-zero payload.",
            {"mean": mean},
        )


def current_mojo_pool_setter(config: RuntimeConfig) -> Any:
    set_pool = current_mojo_pool_module(config).set_pool

    return set_pool


def current_simulation_service_module(config: RuntimeConfig) -> Any:
    ensure_decision_engine_imports(config)
    from apps.api_server.services import simulation_service

    return simulation_service


async def local_mojo_invoke_engine_sync_bridge(
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    from apps.api_server.compute.mojo_adapter import invoke_engine_sync

    return invoke_engine_sync(payload, timeout=timeout)


def install_local_mojo_sync_bridge(config: RuntimeConfig) -> None:
    simulation_service = current_simulation_service_module(config)
    if getattr(simulation_service, "invoke_engine", None) is local_mojo_invoke_engine_sync_bridge:
        return
    simulation_service.invoke_engine = local_mojo_invoke_engine_sync_bridge


def publish_local_mojo_pool_state(config: RuntimeConfig) -> None:
    with LOCAL_MOJO_POOL_LOCK:
        state = LOCAL_MOJO_POOL_STATE
    if state is None or not getattr(state.pool, "running", False):
        return
    current_mojo_pool_setter(config)(state.pool)
    install_local_mojo_sync_bridge(config)


def stop_started_mojo_pool(pool: Any, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    try:
        if getattr(pool, "running", False) and not loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(pool.stop(), loop)
            future.result(timeout=10.0)
    except Exception:
        pass
    stop_loop(loop, thread)


def start_local_mojo_pool_state(config: RuntimeConfig) -> LocalMojoPoolState | None:
    cleanup_stale_local_mojo_workers(config)
    configure_local_mojo_socket_dir(config)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=run_local_mojo_loop,
        args=(loop,),
        name="codna-local-mojo-pool",
        daemon=True,
    )
    thread.start()
    pool: Any | None = None
    try:
        mojo_pool = current_mojo_pool_module(config)
        MojoWorkerPool = mojo_pool.MojoWorkerPool
        set_pool = mojo_pool.set_pool

        pool = MojoWorkerPool(size=local_mojo_pool_size())
        future = asyncio.run_coroutine_threadsafe(pool.start(), loop)
        future.result(timeout=local_mojo_pool_start_timeout_seconds())
        if not pool.running:
            stop_loop(loop, thread)
            return None
        warm_local_mojo_pool(pool, loop)
        set_pool(pool)
        write_local_mojo_worker_state(config, pool)
        return LocalMojoPoolState(
            pool=pool,
            loop=loop,
            thread=thread,
            set_pool=set_pool,
            config=config,
        )
    except Exception:
        if pool is not None:
            stop_started_mojo_pool(pool, loop, thread)
        else:
            stop_loop(loop, thread)
        raise


def wait_for_local_mojo_pool_start(
    config: RuntimeConfig,
    event: threading.Event,
    *,
    require_pool: bool,
) -> bool:
    timeout_s = local_mojo_pool_start_timeout_seconds()
    if event.wait(timeout=timeout_s):
        return True
    if require_pool:
        raise local_mojo_pool_unavailable_error(
            config,
            "Timed out while waiting for Codna local Mojo pool startup.",
            reason=f"timeout_seconds={timeout_s}",
        )
    return False


def raise_if_required_pool_unavailable(config: RuntimeConfig, require_pool: bool) -> None:
    if not require_pool:
        return
    reason = None
    if LOCAL_MOJO_POOL_START_ERROR is not None:
        reason = f"{type(LOCAL_MOJO_POOL_START_ERROR).__name__}: {LOCAL_MOJO_POOL_START_ERROR}"
    raise local_mojo_pool_unavailable_error(
        config,
        "Codna local Mojo pool was already attempted and is unavailable.",
        reason=reason,
    )


def ensure_local_mojo_pool(config: RuntimeConfig) -> None:
    global LOCAL_MOJO_POOL_ATTEMPTED, LOCAL_MOJO_POOL_STATE, LOCAL_MOJO_POOL_START_EVENT
    global LOCAL_MOJO_POOL_START_ERROR
    if not env_bool("CODNA_LOCAL_MOJO_POOL_ENABLED", default=True):
        return
    require_pool = env_bool("CODNA_REQUIRE_LOCAL_MOJO_POOL", default=False)
    while True:
        with LOCAL_MOJO_POOL_LOCK:
            if LOCAL_MOJO_POOL_STATE and getattr(LOCAL_MOJO_POOL_STATE.pool, "running", False):
                return
            if LOCAL_MOJO_POOL_START_EVENT is not None:
                event = LOCAL_MOJO_POOL_START_EVENT
            elif LOCAL_MOJO_POOL_ATTEMPTED:
                raise_if_required_pool_unavailable(config, require_pool)
                return
            else:
                event = threading.Event()
                LOCAL_MOJO_POOL_START_EVENT = event
                LOCAL_MOJO_POOL_ATTEMPTED = True
                LOCAL_MOJO_POOL_START_ERROR = None
                break
        if not wait_for_local_mojo_pool_start(config, event, require_pool=require_pool):
            return

    state: LocalMojoPoolState | None = None
    error: BaseException | None = None
    try:
        state = start_local_mojo_pool_state(config)
    except BaseException as exc:
        error = exc
    with LOCAL_MOJO_POOL_LOCK:
        if state is not None:
            LOCAL_MOJO_POOL_STATE = state
        LOCAL_MOJO_POOL_START_ERROR = error
        LOCAL_MOJO_POOL_START_EVENT = None
        event.set()
    if state is not None:
        return
    if error is not None:
        if isinstance(error, LocalMojoPoolError):
            raise error
        if require_pool:
            raise local_mojo_pool_unavailable_error(
                config,
                "Codna could not start the local Mojo worker pool.",
                reason=f"{type(error).__name__}: {error}",
            ) from error
        return
    if require_pool:
        raise local_mojo_pool_unavailable_error(
            config,
            "Compiled Mojo pool did not enter a running state.",
        )


def prewarm_local_mojo_pool(config: RuntimeConfig) -> None:
    if not env_bool("CODNA_LOCAL_MOJO_POOL_ENABLED", default=True):
        return
    if not local_mojo_pool_prewarm_enabled():
        return
    if env_bool("CODNA_REQUIRE_LOCAL_MOJO_POOL", default=False):
        ensure_local_mojo_pool(config)
        return
    with LOCAL_MOJO_POOL_LOCK:
        if (
            (LOCAL_MOJO_POOL_STATE and getattr(LOCAL_MOJO_POOL_STATE.pool, "running", False))
            or LOCAL_MOJO_POOL_START_EVENT is not None
            or LOCAL_MOJO_POOL_ATTEMPTED
        ):
            return
    thread = threading.Thread(
        target=ensure_local_mojo_pool,
        args=(config,),
        name="codna-local-mojo-prewarm",
        daemon=True,
    )
    thread.start()


def shutdown_local_mojo_pool() -> None:
    global LOCAL_MOJO_POOL_STATE, LOCAL_MOJO_POOL_START_EVENT
    with LOCAL_MOJO_POOL_LOCK:
        state = LOCAL_MOJO_POOL_STATE
        LOCAL_MOJO_POOL_STATE = None
        start_event = LOCAL_MOJO_POOL_START_EVENT
    if state is None:
        if start_event is not None:
            start_event.wait(timeout=2.0)
        return
    try:
        state.set_pool(None)
    except Exception:
        pass
    try:
        if getattr(state.pool, "running", False) and not state.loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(state.pool.stop(), state.loop)
            future.result(timeout=10.0)
    except Exception:
        pass
    stop_loop(state.loop, state.thread)
    clear_local_mojo_worker_state(state.config)


atexit.register(shutdown_local_mojo_pool)
