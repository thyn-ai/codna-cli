from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import local_mojo_pool as local_mojo_pool_module
from .local_mojo_pool import (
    cleanup_stale_local_mojo_workers,
    clear_local_mojo_worker_state,
    ensure_local_mojo_pool,
    env_bool,
    local_mojo_backend_available,
    local_mojo_pool_prewarm_enabled,
    local_mojo_pool_start_timeout_seconds,
    local_mojo_socket_base_dir,
    local_mojo_worker_root,
    mojo_worker_diagnostics,
    shutdown_local_mojo_pool,
)
from .runtime.config import (
    LOG_ROTATE_BYTES,
    LOG_ROTATE_FILES,
    STOP_FORCE_TIMEOUT_S,
    STOP_GRACE_TIMEOUT_S,
    RuntimeConfig,
    resolve_runtime_config,
)
from .runtime.processes import (
    pid_create_time,
    pid_is_alive,
    rotate_log,
    spawn_detached_process,
    terminate_pid,
    write_json_atomic,
    read_json,
)

LOCAL_MOJO_DAEMON_OWNER = "codna-local-mojo-daemon"
LOCAL_MOJO_DAEMON_SCHEMA_VERSION = 1
_FRAME_HEADER_BYTES = 4
_REQUEST_TIMEOUT_S = 30.0


class LocalMojoDaemonError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


_IDLE_EXIT_ENV = "CODNA_LOCAL_MOJO_DAEMON_IDLE_S"
_DEFAULT_IDLE_EXIT_S = 1800.0


def local_mojo_daemon_idle_exit_seconds() -> float:
    """Seconds without a request after which the daemon exits on its own; ``0`` disables.

    The daemon is started on demand (:func:`ensure_local_mojo_daemon`) and, until 2026-09-18, never
    stopped on its own: every test run and every CLI run with a fresh ``CODNA_RUNTIME_ROOT`` left a
    daemon plus its ``simulate`` worker behind for good -- 70 such pairs (634 MB) were found on one
    laptop after a day. The next request simply starts a fresh daemon, so idling out costs one pool
    start. Configurable via ``CODNA_LOCAL_MOJO_DAEMON_IDLE_S``; a request that is still running
    always holds the daemon open (see :class:`_Activity`).
    """
    raw = os.environ.get(_IDLE_EXIT_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_IDLE_EXIT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_IDLE_EXIT_S


class _Activity:
    """Request bookkeeping for the idle timer: time of the last request and requests in flight."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._last = clock()
        self._in_flight = 0

    def begin(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._last = self._clock()

    def end(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._last = self._clock()

    def idle_for(self) -> float:
        """Seconds since the last request ended; ``0.0`` while any request is still running."""
        with self._lock:
            if self._in_flight > 0:
                return 0.0
            return self._clock() - self._last


def local_mojo_daemon_enabled() -> bool:
    if os.environ.get("CODNA_LOCAL_MOJO_DAEMON") == "1":
        return False
    return env_bool("CODNA_LOCAL_MOJO_DAEMON_ENABLED", default=True)


def local_mojo_backend_required() -> bool:
    return env_bool("CODNA_REQUIRE_LOCAL_MOJO_POOL", default=False)


def should_use_local_mojo_backend(config: RuntimeConfig) -> bool:
    if not local_mojo_daemon_enabled():
        return False
    if local_mojo_backend_available(config):
        return True
    if local_mojo_backend_required():
        raise LocalMojoDaemonError(
            "local_mojo_backend_unavailable",
            "Codna local Mojo backend is required but the packaged install does not include the import-backed apps API.",
            {
                "engine_dir": str(config.engine_dir),
                "required_env": "CODNA_REQUIRE_LOCAL_MOJO_POOL",
                "expected_import": "apps.api_server.compute.mojo_pool",
            },
        )
    return False


def local_mojo_daemon_state_path(config: RuntimeConfig) -> Path:
    return local_mojo_worker_root(config) / "daemon.json"


def local_mojo_daemon_socket_path(config: RuntimeConfig) -> Path:
    return local_mojo_socket_base_dir(config) / "daemon.sock"


def local_mojo_daemon_log_path(config: RuntimeConfig) -> Path:
    return config.paths.logs_dir / f"local-mojo-daemon-{config.port_base}.log"


def _write_frame(handle: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handle.sendall(struct.pack(">I", len(data)) + data)


def _read_exact(handle: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = handle.recv(remaining)
        if not chunk:
            raise LocalMojoDaemonError(
                "local_mojo_daemon_protocol_error",
                "Codna local Mojo daemon closed the socket before sending a complete frame.",
                {"expected_bytes": size, "remaining_bytes": remaining},
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(handle: socket.socket) -> dict[str, Any]:
    header = _read_exact(handle, _FRAME_HEADER_BYTES)
    length = struct.unpack(">I", header)[0]
    raw = _read_exact(handle, length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalMojoDaemonError(
            "local_mojo_daemon_protocol_error",
            "Codna local Mojo daemon returned invalid JSON.",
            {"reason": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise LocalMojoDaemonError(
            "local_mojo_daemon_protocol_error",
            "Codna local Mojo daemon returned a non-object response.",
            {"response_type": type(payload).__name__},
        )
    return payload


def _request_daemon(
    config: RuntimeConfig,
    request: dict[str, Any],
    *,
    timeout_s: float = _REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    socket_path = local_mojo_daemon_socket_path(config)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as handle:
            handle.settimeout(timeout_s)
            handle.connect(str(socket_path))
            _write_frame(handle, request)
            response = _read_frame(handle)
    except LocalMojoDaemonError:
        raise
    except OSError as exc:
        raise LocalMojoDaemonError(
            "local_mojo_daemon_unavailable",
            "Codna local Mojo daemon is not reachable.",
            {"socket_path": str(socket_path), "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    if response.get("ok") is True:
        return response
    error = response.get("error")
    details = error if isinstance(error, dict) else {"error": error}
    raise LocalMojoDaemonError(
        str(details.get("code") or "local_mojo_daemon_request_failed"),
        str(details.get("message") or "Codna local Mojo daemon request failed."),
        details if isinstance(details, dict) else {},
    )


def _daemon_state_matches(config: RuntimeConfig, state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    pid = state.get("pid")
    return (
        state.get("schema_version") == LOCAL_MOJO_DAEMON_SCHEMA_VERSION
        and state.get("owner") == LOCAL_MOJO_DAEMON_OWNER
        and state.get("runtime_config_hash") == config.runtime_config_hash
        and state.get("socket_path") == str(local_mojo_daemon_socket_path(config))
        and isinstance(pid, int)
        and pid_is_alive(pid)
    )


def _daemon_ping(config: RuntimeConfig) -> dict[str, Any] | None:
    try:
        return _request_daemon(config, {"type": "ping"}, timeout_s=2.0)
    except LocalMojoDaemonError:
        return None


def _daemon_is_ready(config: RuntimeConfig) -> bool:
    state = read_json(local_mojo_daemon_state_path(config))
    if not _daemon_state_matches(config, state):
        return False
    response = _daemon_ping(config)
    if not response:
        return False
    return response.get("pid") == state.get("pid") and response.get("status") == "ready"


def _cleanup_stale_daemon_state(config: RuntimeConfig) -> None:
    state_path = local_mojo_daemon_state_path(config)
    state = read_json(state_path)
    if _daemon_state_matches(config, state) and _daemon_ping(config):
        return
    pid = state.get("pid") if isinstance(state, dict) else None
    if isinstance(pid, int) and pid_is_alive(pid):
        terminate_pid(
            pid,
            grace_timeout_s=STOP_GRACE_TIMEOUT_S,
            force_timeout_s=STOP_FORCE_TIMEOUT_S,
        )
    socket_path = local_mojo_daemon_socket_path(config)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    if state:
        state_path.unlink(missing_ok=True)
    cleanup_stale_local_mojo_workers(config)


def _daemon_env(config: RuntimeConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["CODNA_RUNTIME_ROOT"] = str(config.paths.root)
    env["CODNA_PORT_BASE"] = str(config.port_base)
    env["CODNA_LOCAL_MOJO_DAEMON"] = "1"
    env["CODNA_LOCAL_MOJO_POOL_PREWARM"] = "0"
    env.setdefault("CODNA_REQUIRE_LOCAL_MOJO_POOL", "1")
    env.setdefault("MOJO_ALLOW_SUBPROCESS_FALLBACK", "false")
    return env


def ensure_local_mojo_daemon(config: RuntimeConfig) -> None:
    if not should_use_local_mojo_backend(config):
        return
    if _daemon_is_ready(config):
        return
    _cleanup_stale_daemon_state(config)
    if _daemon_is_ready(config):
        return
    log_path = local_mojo_daemon_log_path(config)
    rotate_log(log_path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_FILES)
    process = spawn_detached_process(
        [sys.executable, "-m", "codna.local_mojo_daemon", "serve"],
        cwd=Path.cwd(),
        env=_daemon_env(config),
        log_path=log_path,
    )
    deadline = time.monotonic() + local_mojo_pool_start_timeout_seconds()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if _daemon_is_ready(config):
            return
        time.sleep(0.1)
    raise LocalMojoDaemonError(
        "local_mojo_daemon_start_failed",
        "Codna local Mojo daemon did not become ready.",
        {
            "pid": process.pid,
            "socket_path": str(local_mojo_daemon_socket_path(config)),
            "state_path": str(local_mojo_daemon_state_path(config)),
            "log_path": str(log_path),
            "exit_code": process.poll(),
        },
    )


def invoke_local_mojo_daemon(
    config: RuntimeConfig,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    ensure_local_mojo_daemon(config)
    response = _request_daemon(
        config,
        {"type": "invoke", "payload": payload, "timeout": timeout},
        timeout_s=(timeout or _REQUEST_TIMEOUT_S) + 2.0,
    )
    result = response.get("result")
    if not isinstance(result, dict):
        raise LocalMojoDaemonError(
            "local_mojo_daemon_invalid_response",
            "Codna local Mojo daemon returned an invalid invoke response.",
            {"response": response},
        )
    return result


def install_local_mojo_daemon_bridge(config: RuntimeConfig) -> None:
    from apps.api_server.compute import mojo_adapter
    from apps.api_server.services import simulation_service

    async def invoke_engine(payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(invoke_local_mojo_daemon, config, payload, timeout=timeout)

    def invoke_engine_sync(payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return invoke_local_mojo_daemon(config, payload, timeout=timeout)

    simulation_service.invoke_engine = invoke_engine
    mojo_adapter.invoke_engine_sync = invoke_engine_sync
    _patch_imported_sync_bridge("apps.api_server.services.auto_planner", invoke_engine_sync)
    _patch_imported_sync_bridge(
        "apps.api_server.services.repository_intelligence_mojo_repo_classifier",
        invoke_engine_sync,
    )


def _patch_imported_sync_bridge(module_name: str, invoke_engine_sync: Any) -> None:
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, "invoke_engine_sync"):
        module.invoke_engine_sync = invoke_engine_sync


def ensure_local_mojo_runtime(config: RuntimeConfig) -> None:
    if local_mojo_daemon_enabled():
        if not should_use_local_mojo_backend(config):
            return
        ensure_local_mojo_daemon(config)
        install_local_mojo_daemon_bridge(config)
        return
    ensure_local_mojo_pool(config)


def publish_local_mojo_runtime_state(config: RuntimeConfig) -> None:
    if local_mojo_daemon_enabled():
        if not should_use_local_mojo_backend(config):
            return
        ensure_local_mojo_daemon(config)
        install_local_mojo_daemon_bridge(config)
        return
    from .local_mojo_pool import publish_local_mojo_pool_state

    publish_local_mojo_pool_state(config)


def _record_prewarm_failure(config: RuntimeConfig, exc: BaseException) -> None:
    log_path = local_mojo_daemon_log_path(config)
    event = {
        "event": "local_mojo_daemon_prewarm_failed",
        "code": str(getattr(exc, "code", type(exc).__name__)),
        "message": str(exc),
        "details": getattr(exc, "details", {}),
        "ts": time.time(),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate_log(log_path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_FILES)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        return


def _prewarm_local_mojo_daemon(config: RuntimeConfig) -> None:
    try:
        ensure_local_mojo_daemon(config)
    except Exception as exc:  # noqa: BLE001 - optional background warmup must not leak thread exceptions.
        _record_prewarm_failure(config, exc)


def prewarm_local_mojo_runtime(config: RuntimeConfig) -> None:
    if not local_mojo_daemon_enabled():
        from .local_mojo_pool import prewarm_local_mojo_pool

        prewarm_local_mojo_pool(config)
        return
    if not should_use_local_mojo_backend(config):
        return
    if not local_mojo_pool_prewarm_enabled():
        return
    if env_bool("CODNA_REQUIRE_LOCAL_MOJO_POOL", default=False):
        ensure_local_mojo_daemon(config)
        return
    thread = threading.Thread(
        target=_prewarm_local_mojo_daemon,
        args=(config,),
        name="codna-local-mojo-daemon-prewarm",
        daemon=True,
    )
    thread.start()


def stop_local_mojo_daemon(config: RuntimeConfig) -> dict[str, Any]:
    state_path = local_mojo_daemon_state_path(config)
    state = read_json(state_path)
    stopped_by_request = False
    if _daemon_state_matches(config, state):
        try:
            _request_daemon(config, {"type": "shutdown"}, timeout_s=5.0)
            stopped_by_request = True
        except LocalMojoDaemonError:
            stopped_by_request = False
        pid = state.get("pid") if isinstance(state, dict) else None
        if isinstance(pid, int) and pid_is_alive(pid):
            try:
                terminate_pid(
                    pid,
                    grace_timeout_s=STOP_GRACE_TIMEOUT_S,
                    force_timeout_s=STOP_FORCE_TIMEOUT_S,
                )
            except PermissionError as exc:
                raise _stop_permission_error(config, pid=pid, reason=str(exc)) from exc
    try:
        _cleanup_stale_daemon_state(config)
    except PermissionError as exc:
        pid = state.get("pid") if isinstance(state, dict) else None
        raise _stop_permission_error(
            config,
            pid=pid if isinstance(pid, int) else None,
            reason=str(exc),
        ) from exc
    clear_local_mojo_worker_state(config)
    return {
        "status": "stopped" if stopped_by_request or state else "not_running",
        "state_path": str(state_path),
        "socket_path": str(local_mojo_daemon_socket_path(config)),
        "stopped_by_request": stopped_by_request,
    }


def _stop_permission_error(
    config: RuntimeConfig,
    *,
    pid: int | None,
    reason: str,
) -> LocalMojoDaemonError:
    details: dict[str, Any] = {
        "state_path": str(local_mojo_daemon_state_path(config)),
        "socket_path": str(local_mojo_daemon_socket_path(config)),
        "log_path": str(local_mojo_daemon_log_path(config)),
        "reason": reason,
    }
    if pid is not None:
        details["pid"] = pid
        details["manual_stop_command"] = f"kill {pid}"
    return LocalMojoDaemonError(
        "local_mojo_daemon_stop_permission_denied",
        "Codna could not stop its owned local Mojo daemon.",
        details,
    )


def _write_daemon_state(config: RuntimeConfig) -> None:
    pid = os.getpid()
    payload = {
        "schema_version": LOCAL_MOJO_DAEMON_SCHEMA_VERSION,
        "owner": LOCAL_MOJO_DAEMON_OWNER,
        "pid": pid,
        "pid_create_time": pid_create_time(pid),
        "runtime_config_hash": config.runtime_config_hash,
        "socket_path": str(local_mojo_daemon_socket_path(config)),
        "worker_state_path": str(local_mojo_worker_root(config) / "workers.json"),
        "started_at": time.time(),
    }
    write_json_atomic(local_mojo_daemon_state_path(config), payload)


def _handle_request(
    config: RuntimeConfig,
    request: dict[str, Any],
    stop_event: threading.Event,
) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type == "ping":
        return {
            "ok": True,
            "status": "ready",
            "pid": os.getpid(),
        }
    if request_type == "diagnostics":
        return {
            "ok": True,
            "status": "ready",
            "pid": os.getpid(),
            "mojo_workers": mojo_worker_diagnostics(config),
        }
    if request_type == "shutdown":
        stop_event.set()
        return {"ok": True, "status": "stopping", "pid": os.getpid()}
    if request_type == "invoke":
        payload = request.get("payload")
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "error": {
                    "code": "invalid_local_mojo_daemon_request",
                    "message": "invoke payload must be an object.",
                    "details": {"payload_type": type(payload).__name__},
                },
            }
        timeout = request.get("timeout")
        result = _invoke_owned_pool_direct(payload, timeout=float(timeout) if timeout is not None else None)
        return {"ok": True, "result": result}
    return {
        "ok": False,
        "error": {
            "code": "invalid_local_mojo_daemon_request",
            "message": "request type is unsupported.",
            "details": {"type": request_type},
        },
    }


def _invoke_owned_pool_direct(payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    with local_mojo_pool_module.LOCAL_MOJO_POOL_LOCK:
        state = local_mojo_pool_module.LOCAL_MOJO_POOL_STATE
    if state is None or not getattr(state.pool, "running", False):
        raise LocalMojoDaemonError(
            "local_mojo_daemon_pool_not_running",
            "Codna local Mojo daemon has no running owned pool.",
        )
    wait_timeout = (timeout if timeout is not None else 60.0) + 1.0
    future = asyncio.run_coroutine_threadsafe(
        state.pool.invoke(payload, timeout=timeout),
        state.loop,
    )
    result = future.result(timeout=wait_timeout)
    if not isinstance(result, dict):
        raise LocalMojoDaemonError(
            "local_mojo_daemon_invalid_pool_response",
            "Codna local Mojo pool returned a non-object response.",
            {"response_type": type(result).__name__},
        )
    return result


def _serve_client(
    handle: socket.socket,
    config: RuntimeConfig,
    stop_event: threading.Event,
    activity: _Activity | None = None,
) -> None:
    began = False
    try:
        if activity is not None:
            activity.begin()
            began = True  # end() pairs with a begin() that happened; a failed begin() must not reset the idle timer
        with handle:
            try:
                request = _read_frame(handle)
                response = _handle_request(config, request, stop_event)
            except Exception as exc:  # noqa: BLE001
                response = {
                    "ok": False,
                    "error": {
                        "code": getattr(exc, "code", "local_mojo_daemon_request_failed"),
                        "message": str(exc),
                        "details": getattr(exc, "details", {}),
                    },
                }
            try:
                _write_frame(handle, response)
            except OSError:
                return
    finally:
        if began and activity is not None:
            activity.end()


# Per-connection accept() errors that leave the listening socket healthy: retry, never exit.
_RETRYABLE_ACCEPT_ERRNOS = frozenset(
    e for e in (
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "EINTR", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EWOULDBLOCK", None),
        getattr(errno, "EPROTO", None),
    ) if e is not None
)


def _accept_loop(
    server: Any,
    config: RuntimeConfig,
    stop_event: threading.Event,
    *,
    idle_exit_s: float,
    activity: _Activity,
) -> str:
    """Accept clients until asked to stop, or until idle for ``idle_exit_s`` (``0`` = never).

    Returns ``"stopped"``, ``"idle"`` or ``"error"`` -- the reason the loop ended."""
    while not stop_event.is_set():
        try:
            client, _addr = server.accept()
        except TimeoutError:
            if idle_exit_s > 0 and activity.idle_for() >= idle_exit_s:
                print(
                    json.dumps({"event": "local_mojo_daemon_idle_exit", "idle_s": idle_exit_s, "pid": os.getpid()}),
                    file=sys.stderr,
                )
                stop_event.set()
                return "idle"
            continue
        except OSError as exc:
            if exc.errno in _RETRYABLE_ACCEPT_ERRNOS:
                continue  # a client vanished between connect() and accept(): the listening socket is fine
            # EMFILE / ENFILE / EBADF / EINVAL: the listening socket itself is broken, so end cleanly
            # (the caller's finally shuts the pool) instead of dying mid-loop with the stop event never
            # set. The next ensure_local_mojo_daemon starts a fresh daemon.
            print(
                json.dumps({"event": "local_mojo_daemon_accept_error", "error": f"{type(exc).__name__}: {exc}", "pid": os.getpid()}),
                file=sys.stderr,
            )
            stop_event.set()
            return "error"
        thread = threading.Thread(
            target=_serve_client,
            args=(client, config, stop_event, activity),
            daemon=True,
        )
        thread.start()
    return "stopped"


def serve_local_mojo_daemon() -> int:
    config = resolve_runtime_config()
    socket_path = local_mojo_daemon_socket_path(config)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    stop_event = threading.Event()
    try:
        ensure_local_mojo_pool(config)
        _write_daemon_state(config)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen()
            server.settimeout(0.5)
            _accept_loop(
                server,
                config,
                stop_event,
                idle_exit_s=local_mojo_daemon_idle_exit_seconds(),
                activity=_Activity(),
            )
        return 0
    finally:
        shutdown_local_mojo_pool()
        socket_path.unlink(missing_ok=True)
        local_mojo_daemon_state_path(config).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["serve"]:
        return serve_local_mojo_daemon()
    print("usage: python -m codna.local_mojo_daemon serve", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
