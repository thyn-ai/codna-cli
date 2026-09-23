"""Owned local runtime startup for Codna CLI surfaces.

This module is intentionally narrow: it starts or reuses the existing hidden
supervisor on fixed loopback ports. It does not inspect secrets, change engine
semantics, or manage benchmark behavior.
"""
from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__
from . import supervisor
from .runtime_config import (
    ENGINE_URL_KEYS,
    RuntimeConfigError,
    local_runtime_urls,
    resolve_local_urls_from_env,
    resolve_port_base,
)

STARTUP_LOCK_TIMEOUT_S = 30.0
STARTUP_READY_TIMEOUT_S = 90.0
HEALTH_REQUEST_TIMEOUT_S = 2.0
GRACEFUL_STOP_TIMEOUT_S = 10.0
FORCED_STOP_TIMEOUT_S = 5.0
LOCK_POLL_S = 0.05
STATE_POLL_S = 0.25
MAX_LOG_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 2
SIDECAR_SERVICE = "codna-sidecar"


@dataclass(frozen=True)
class RuntimeEndpoint:
    engine_url: str
    sidecar_url: str | None
    local: bool
    port_base: int | None
    runtime_id: str | None = None


@dataclass(frozen=True)
class RuntimePaths:
    runtime_dir: Path
    lock_path: Path
    state_path: Path
    legacy_state_path: Path
    engine_log_path: Path
    sidecar_log_path: Path
    supervisor_log_path: Path


class RuntimeStartError(RuntimeError):
    """Local runtime could not be started or safely reused."""


def runtime_paths(
    *,
    port_base: int,
    home: Path | None = None,
    state_path: Path | None = None,
    legacy_state_path: Path | None = None,
) -> RuntimePaths:
    root = Path.home() if home is None else home
    runtime_dir = root / ".codna" / "runtime"
    log_dir = root / ".codna" / "logs"
    default_state_path = supervisor.STATE_PATH if home is None else runtime_dir / "local-stack.json"
    default_legacy_state_path = supervisor.LEGACY_STATE_PATH if home is None else root / ".codna" / "launcher.state"
    return RuntimePaths(
        runtime_dir=runtime_dir,
        lock_path=runtime_dir / "local-stack.lock",
        state_path=default_state_path if state_path is None else state_path,
        legacy_state_path=default_legacy_state_path if legacy_state_path is None else legacy_state_path,
        engine_log_path=log_dir / f"local-engine-{port_base}.log",
        sidecar_log_path=log_dir / f"local-sidecar-{port_base + 1}.log",
        supervisor_log_path=log_dir / f"local-supervisor-{port_base}.log",
    )


def ensure_running(
    *,
    env: Mapping[str, str] | None = None,
    state_reader: Callable[[], dict[str, Any] | None] = supervisor.read_state,
    process_starter: Callable[[str, str, str, RuntimePaths, Mapping[str, str]], subprocess.Popen[str] | None] | None = None,
    home: Path | None = None,
    state_path: Path | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    port_available: Callable[[int], bool] | None = None,
    health_reader: Callable[[str], Mapping[str, Any] | None] | None = None,
    runtime_stopper: Callable[[Mapping[str, Any], RuntimePaths], None] | None = None,
) -> RuntimeEndpoint:
    source_env = os.environ if env is None else env
    remote = _remote_override(source_env)
    if remote is not None:
        return remote

    try:
        port_base = resolve_port_base(source_env.get("CODNA_PORT_BASE"))
        engine_url, sidecar_url = resolve_local_urls_from_env(source_env)
    except RuntimeConfigError:
        raise
    except ValueError as exc:
        raise RuntimeConfigError(str(exc)) from exc

    paths = runtime_paths(port_base=port_base, home=home, state_path=state_path)
    _mkdir_runtime_dir(paths.runtime_dir, paths)
    sidecar_health_reader = _read_json_health if health_reader is None else health_reader
    with _startup_lock(paths.lock_path, monotonic=monotonic, sleep=sleep):
        state = state_reader()
        endpoint = _endpoint_from_state(
            state,
            port_base=port_base,
            sidecar_health_reader=sidecar_health_reader,
        )
        if endpoint is not None:
            return endpoint

        if _state_proves_owned_pair(state, port_base=port_base):
            stopper = _stop_owned_runtime_children if runtime_stopper is None else runtime_stopper
            stopper(state, paths)

        port_probe = _loopback_port_available if port_available is None else port_available
        _raise_if_fixed_ports_unavailable(port_base, paths, port_available=port_probe)

        runtime_id = uuid.uuid4().hex
        starter = _start_supervisor if process_starter is None else process_starter
        proc = starter(runtime_id, engine_url, sidecar_url, paths, source_env)
        return _wait_for_state(
            state_reader=state_reader,
            process=proc,
            paths=paths,
            port_base=port_base,
            runtime_id=runtime_id,
            sidecar_health_reader=sidecar_health_reader,
            monotonic=monotonic,
            sleep=sleep,
        )


def format_start_stack_output(endpoint: RuntimeEndpoint, *, json_output: bool) -> str:
    import json

    if not endpoint.local:
        payload = {
            "schema_version": 1,
            "runtime": {
                "local": False,
                "remote_override": True,
                "started": False,
            },
        }
        if json_output:
            return json.dumps(payload, indent=2, sort_keys=True)
        return "Codna runtime: remote override configured; local stack not started."

    paths = runtime_paths(port_base=endpoint.port_base or 0)
    payload = {
        "schema_version": 1,
        "runtime": {
            "local": True,
            "engine_url": endpoint.engine_url,
            "sidecar_url": endpoint.sidecar_url,
            "port_base": endpoint.port_base,
            "runtime_id": endpoint.runtime_id,
            "state_path": str(paths.state_path),
            "legacy_state_path": str(paths.legacy_state_path),
            "engine_log_path": str(paths.engine_log_path),
            "sidecar_log_path": str(paths.sidecar_log_path),
        },
    }
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "\n".join(
        [
            "Codna runtime started/reused:",
            f"  engine_url       : {endpoint.engine_url}",
            f"  sidecar_url      : {endpoint.sidecar_url}",
            f"  port_base        : {endpoint.port_base}",
            f"  runtime_id       : {endpoint.runtime_id or 'n/a'}",
            f"  state_path       : {paths.state_path}",
            f"  legacy_state_path: {paths.legacy_state_path}",
            f"  engine_log       : {paths.engine_log_path}",
            f"  sidecar_log      : {paths.sidecar_log_path}",
        ]
    )


def _remote_override(env: Mapping[str, str]) -> RuntimeEndpoint | None:
    for key in ENGINE_URL_KEYS:
        value = env.get(key)
        if not value:
            continue
        engine_url = value.rstrip("/")
        if _is_loopback_http_url(engine_url):
            if env.get("CODNA_ALLOW_LOOPBACK_ENGINE_URL") == "1":
                return RuntimeEndpoint(engine_url=engine_url, sidecar_url=None, local=False, port_base=None)
            raise RuntimeStartError(
                f"{key} points at a loopback URL. Unset it so Codna can own the fixed local runtime, "
                "or set CODNA_ALLOW_LOOPBACK_ENGINE_URL=1 for an explicit development override."
            )
        return RuntimeEndpoint(engine_url=engine_url, sidecar_url=None, local=False, port_base=None)
    return None


def _endpoint_from_state(
    state: dict[str, Any] | None,
    *,
    port_base: int,
    sidecar_health_reader: Callable[[str], Mapping[str, Any] | None],
    allow_unverified_pid_create_time: bool = False,
) -> RuntimeEndpoint | None:
    if not state or not _state_has_owned_identity(state, port_base=port_base):
        return None
    engine_url = str(state["engine_url"]).rstrip("/")
    sidecar_url = str(state["sidecar_url"]).rstrip("/") if state.get("sidecar_url") else None
    expected_engine_url, expected_sidecar_url = local_runtime_urls(port_base)
    if engine_url != expected_engine_url or sidecar_url != expected_sidecar_url:
        return None
    if state.get("runtime_config_hash") != _expected_runtime_config_hash(
        port_base=port_base,
        engine_url=engine_url,
        sidecar_url=sidecar_url,
    ):
        return None
    if not _state_process_matches(
        state,
        key="engine",
        port=port_base,
        health_url=f"{engine_url}/v1/health",
        allow_unverified_create_time=allow_unverified_pid_create_time,
    ):
        return None
    if not _state_process_matches(
        state,
        key="sidecar",
        port=port_base + 1,
        health_url=f"{sidecar_url}/health",
        allow_unverified_create_time=allow_unverified_pid_create_time,
    ):
        return None
    runtime_id = str(state["runtime_id"])
    if not _sidecar_health_matches(
        f"{sidecar_url}/health",
        runtime_id=runtime_id,
        port=port_base + 1,
        health_reader=sidecar_health_reader,
    ):
        return None
    return RuntimeEndpoint(
        engine_url=engine_url,
        sidecar_url=sidecar_url,
        local=True,
        port_base=port_base,
        runtime_id=runtime_id,
    )


def _sidecar_health_matches(
    health_url: str,
    *,
    runtime_id: str,
    port: int,
    health_reader: Callable[[str], Mapping[str, Any] | None],
) -> bool:
    payload = health_reader(health_url)
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("service") == SIDECAR_SERVICE
        and payload.get("status") == "ok"
        and payload.get("runtime_id") == runtime_id
        and payload.get("port") == port
    )


def _read_json_health(url: str) -> Mapping[str, Any] | None:
    request = Request(url, headers={"accept": "application/json"})
    try:
        with urlopen(request, timeout=HEALTH_REQUEST_TIMEOUT_S) as response:
            status = getattr(response, "status", None)
            if status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _state_has_owned_identity(state: Mapping[str, Any], *, port_base: int) -> bool:
    runtime_id = state.get("runtime_id")
    supervisor_pid = state.get("pid")
    return (
        state.get("schema_version") == 1
        and state.get("owner") == "codna-local-runtime"
        and state.get("port_base") == port_base
        and isinstance(runtime_id, str)
        and bool(runtime_id)
        and isinstance(supervisor_pid, int)
        and _pid_alive(supervisor_pid)
    )


def _state_proves_owned_pair(state: Mapping[str, Any] | None, *, port_base: int) -> bool:
    if not state or not _state_has_owned_identity(state, port_base=port_base):
        return False
    engine_url = str(state.get("engine_url", "")).rstrip("/")
    sidecar_url = str(state.get("sidecar_url", "")).rstrip("/")
    expected_engine_url, expected_sidecar_url = local_runtime_urls(port_base)
    if engine_url != expected_engine_url or sidecar_url != expected_sidecar_url:
        return False
    return _state_process_matches(
        state,
        key="engine",
        port=port_base,
        health_url=f"{engine_url}/v1/health",
        allow_unverified_create_time=False,
    ) and _state_process_matches(
        state,
        key="sidecar",
        port=port_base + 1,
        health_url=f"{sidecar_url}/health",
        allow_unverified_create_time=False,
    )


def _state_process_matches(
    state: Mapping[str, Any],
    *,
    key: str,
    port: int,
    health_url: str,
    allow_unverified_create_time: bool,
) -> bool:
    process = state.get(key)
    if not isinstance(process, Mapping):
        return False
    pid = process.get("pid")
    stored_create_time = process.get("pid_create_time")
    return (
        isinstance(pid, int)
        and isinstance(stored_create_time, int | float)
        and process.get("port") == port
        and process.get("health_url") == health_url
        and _pid_create_time_matches(
            pid,
            float(stored_create_time),
            allow_unverified_create_time=allow_unverified_create_time,
        )
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _pid_create_time_matches(pid: int, expected: float, *, allow_unverified_create_time: bool) -> bool:
    actual = _pid_create_time(pid)
    if actual is None:
        return allow_unverified_create_time and _pid_alive(pid)
    return abs(actual - expected) <= 1.0


def _pid_create_time(pid: int) -> float | None:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = proc.stdout.strip()
    if proc.returncode != 0 or not raw:
        return None
    try:
        parsed = datetime.strptime(raw, "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.timestamp()


def _stop_owned_runtime_children(state: Mapping[str, Any], paths: RuntimePaths) -> None:
    pids = _owned_child_pids(state)
    if not pids:
        return
    for pid in pids:
        _signal_pid(pid, signal.SIGTERM, paths)
    _wait_for_pids_to_exit(pids, timeout_s=GRACEFUL_STOP_TIMEOUT_S)
    for pid in pids:
        if _pid_alive(pid):
            _signal_pid(pid, signal.SIGKILL, paths)
    _wait_for_pids_to_exit(pids, timeout_s=FORCED_STOP_TIMEOUT_S)
    survivors = [pid for pid in pids if _pid_alive(pid)]
    if survivors:
        raise RuntimeStartError(
            _failure_message(
                f"owned Codna local runtime process did not stop after SIGKILL: {', '.join(map(str, survivors))}",
                paths,
            )
        )
    paths.state_path.unlink(missing_ok=True)
    paths.legacy_state_path.unlink(missing_ok=True)


def _owned_child_pids(state: Mapping[str, Any]) -> list[int]:
    pids: list[int] = []
    for key in ("engine", "sidecar"):
        process = state.get(key)
        if isinstance(process, Mapping) and isinstance(process.get("pid"), int):
            pids.append(process["pid"])
    return sorted(set(pids))


def _signal_pid(pid: int, sig: signal.Signals, paths: RuntimePaths) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise RuntimeStartError(
            _failure_message(f"permission denied stopping owned Codna local runtime process {pid}", paths)
        ) from exc
    except OSError as exc:
        raise RuntimeStartError(
            _failure_message(f"failed stopping owned Codna local runtime process {pid}: {exc}", paths)
        ) from exc


def _wait_for_pids_to_exit(pids: list[int], *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(not _pid_alive(pid) for pid in pids):
            return
        time.sleep(STATE_POLL_S)


def _expected_runtime_config_hash(*, port_base: int, engine_url: str, sidecar_url: str) -> str:
    payload = {
        "codna_cli_version": __version__,
        "engine_url": engine_url,
        "port_base": port_base,
        "python_executable": sys.executable,
        "sidecar_url": sidecar_url,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _raise_if_fixed_ports_unavailable(
    port_base: int,
    paths: RuntimePaths,
    *,
    port_available: Callable[[int], bool],
) -> None:
    occupied = [port for port in (port_base, port_base + 1) if not port_available(port)]
    if occupied:
        raise RuntimeStartError(_port_collision_message(occupied=occupied, paths=paths))


def _loopback_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def _port_collision_message(*, occupied: list[int], paths: RuntimePaths) -> str:
    ports = ", ".join(str(port) for port in occupied)
    return _failure_message(
        f"fixed local runtime port already in use: {ports}. "
        "Codna will not kill an unknown or foreign process; stop the listener or set CODNA_PORT_BASE",
        paths,
    )


def _is_loopback_http_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


@contextmanager
def _startup_lock(
    path: Path,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeStartError(f"cannot create Codna local runtime lock directory: {path.parent}") from exc
    deadline = monotonic() + STARTUP_LOCK_TIMEOUT_S
    with path.open("a+") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise RuntimeStartError(f"timed out acquiring Codna local runtime startup lock: {path}") from None
                sleep(LOCK_POLL_S)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _mkdir_runtime_dir(path: Path, paths: RuntimePaths) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeStartError(_failure_message(f"cannot create runtime directory {path}", paths)) from exc


def _start_supervisor(
    runtime_id: str,
    engine_url: str,
    sidecar_url: str,
    paths: RuntimePaths,
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    for log_path in (paths.engine_log_path, paths.sidecar_log_path, paths.supervisor_log_path):
        _rotate_log(log_path)
    child_env = _supervisor_env(runtime_id, engine_url, sidecar_url, paths, env)
    paths.supervisor_log_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor_log = paths.supervisor_log_path.open("a")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "codna.supervisor"],
            env=child_env,
            stdout=supervisor_log,
            stderr=supervisor_log,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        supervisor_log.close()


def _supervisor_env(
    runtime_id: str,
    engine_url: str,
    sidecar_url: str,
    paths: RuntimePaths,
    env: Mapping[str, str],
) -> dict[str, str]:
    return {
        **env,
        "CODNA_LOCAL_RUNTIME": "1",
        "CODNA_RUNTIME_ID": runtime_id,
        "CODNA_PORT_BASE": str(resolve_port_base(env.get("CODNA_PORT_BASE"))),
        "CODNA_ENGINE_PORT": engine_url.rsplit(":", 1)[-1],
        "CODNA_SIDECAR_PORT": sidecar_url.rsplit(":", 1)[-1],
        "CODNA_LOCAL_STACK_STATE": str(paths.state_path),
        "CODNA_LAUNCHER_STATE": str(paths.legacy_state_path),
        "CODNA_ENGINE_LOG_PATH": str(paths.engine_log_path),
        "CODNA_SIDECAR_LOG_PATH": str(paths.sidecar_log_path),
    }


def _wait_for_state(
    *,
    state_reader: Callable[[], dict[str, Any] | None],
    process: subprocess.Popen[str] | None,
    paths: RuntimePaths,
    port_base: int,
    runtime_id: str,
    sidecar_health_reader: Callable[[str], Mapping[str, Any] | None],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> RuntimeEndpoint:
    deadline = monotonic() + STARTUP_READY_TIMEOUT_S
    while monotonic() < deadline:
        state = state_reader()
        endpoint = _endpoint_from_state(
            state,
            port_base=port_base,
            sidecar_health_reader=sidecar_health_reader,
            allow_unverified_pid_create_time=bool(state and state.get("runtime_id") == runtime_id),
        )
        if endpoint is not None:
            return endpoint
        if process is not None and process.poll() is not None:
            raise RuntimeStartError(_failure_message("supervisor exited before readiness", paths))
        sleep(STATE_POLL_S)
    raise RuntimeStartError(_failure_message(f"timed out waiting for local runtime {runtime_id}", paths))


def _failure_message(phase: str, paths: RuntimePaths) -> str:
    return (
        f"Codna local runtime failed: {phase}. "
        f"State: {paths.state_path}. "
        f"Engine log: {paths.engine_log_path}. "
        f"Sidecar log: {paths.sidecar_log_path}. "
        f"Supervisor log: {paths.supervisor_log_path}."
    )


def _rotate_log(path: Path, *, max_bytes: int = MAX_LOG_BYTES, backups: int = LOG_BACKUPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size < max_bytes:
        return
    for index in range(backups, 0, -1):
        current = Path(f"{path}.{index}")
        if index == backups:
            current.unlink(missing_ok=True)
            continue
        previous = Path(f"{path}.{index + 1}")
        if current.exists():
            current.replace(previous)
    Path(f"{path}.1").unlink(missing_ok=True)
    path.replace(Path(f"{path}.1"))
