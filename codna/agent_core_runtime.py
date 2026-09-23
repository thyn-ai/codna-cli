from __future__ import annotations

import uuid
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .runtime.config import (
    ConfigValue,
    HEALTH_TIMEOUT_S,
    LOCK_TIMEOUT_S,
    LOG_ROTATE_BYTES,
    LOG_ROTATE_FILES,
    READY_TIMEOUT_S,
    RUNTIME_OWNER,
    STOP_FORCE_TIMEOUT_S,
    STOP_GRACE_TIMEOUT_S,
    RuntimeConfig,
    RuntimeConfigError,
    resolve_runtime_config,
)
from .runtime.health import (
    ProbeResult,
    probe_sidecar_health,
    probe_sidecar_ready,
    request_sidecar_shutdown,
    wait_until_ready,
)
from .runtime.local_stack import LocalRuntimeError, _log_paths, _sidecar_spawn
from .runtime.ports import ListenerInfo, listeners_on_port
from .runtime.processes import (
    ensure_directories,
    pid_create_time,
    rotate_log,
    runtime_lock,
    spawn_detached_process,
    terminate_pid,
    write_json_atomic,
    read_json,
)


PROVIDER_RUNTIME_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "AGENT_CORE_FALLBACK_TOOL_PROFILE",
    "AGENT_CORE_TELYS_LOCALIZE_TIMEOUT_MS",
    "AGENT_CORE_TELYS_TOP_K",
    "AGENT_CORE_TELYS_MAX_FILE_BYTES",
    "AGENT_CORE_TELYS_MAX_FILES_WITHOUT_HINTS",
    "AGENT_CORE_TELYS_SKIP_LINE_TARGETS",
    "AGENT_CORE_TELYS_WINDOW_LINES",
    "AGENT_CORE_TELYS_WINDOW_OVERLAP_LINES",
    "CODNA_TELYS_INSTALL_ROOT",
    "CODNA_TELYS_MEMORY_ROOT",
    "CODNA_TELYS_PYTHON",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TELYS_KERNEL",
)
LEGACY_REQUEST_MODE_ENV_KEYS = (
    "AGENT_CORE_TELYS_LOCALIZE_ENABLED",
)
AGENT_CORE_READY_TIMEOUT_ENV = "CODNA_AGENT_CORE_READY_TIMEOUT_SECONDS"
AGENT_CORE_READY_TIMEOUT_DEFAULT_S = 120.0
AGENT_CORE_READY_TIMEOUT_MIN_S = 5.0
AGENT_CORE_READY_TIMEOUT_MAX_S = 300.0
AGENT_CORE_LOG_TAIL_BYTES = 16 * 1024
AGENT_CORE_LOG_TAIL_LINES = 120
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(bearer|basic)\s+[a-z0-9._~+/=-]+|"
    r"pypi-[a-z0-9._~+/=-]+|"
    r"gh[pousr]_[a-z0-9_]+|"
    r"sk-[a-z0-9_-]+"
)
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)"
    r"([\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"
)


@dataclass(frozen=True)
class AgentCoreEndpoint:
    url: str
    port: int
    runtime_id: str
    pid: int | None


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _state_path(config: RuntimeConfig) -> Path:
    return config.paths.runtime_dir / "local-agent-core.json"


def _unlink_state_if_possible(config: RuntimeConfig) -> str | None:
    try:
        _state_path(config).unlink(missing_ok=True)
        return None
    except PermissionError as exc:
        return str(exc)


def _single_listener(port: int) -> ListenerInfo | None:
    listeners = listeners_on_port(port)
    if not listeners:
        return None
    if len(listeners) == 1:
        return listeners[0]
    loopback = [listener for listener in listeners if listener.loopback]
    return loopback[0] if loopback else listeners[0]


def _probe_summary(probe: ProbeResult) -> dict[str, Any]:
    return {
        "ok": probe.ok,
        "url": probe.url,
        "status_code": probe.status_code,
        "payload": probe.payload,
        "error": probe.error,
    }


def _agent_core_ready_timeout_seconds() -> float:
    raw = os.environ.get(AGENT_CORE_READY_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return AGENT_CORE_READY_TIMEOUT_DEFAULT_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise LocalRuntimeError(
            "agent_core_ready_timeout_invalid",
            "Codna agent-core readiness timeout must be numeric seconds.",
            {
                "env": AGENT_CORE_READY_TIMEOUT_ENV,
                "value": raw,
                "min_seconds": AGENT_CORE_READY_TIMEOUT_MIN_S,
                "max_seconds": AGENT_CORE_READY_TIMEOUT_MAX_S,
            },
        ) from exc
    if not AGENT_CORE_READY_TIMEOUT_MIN_S <= value <= AGENT_CORE_READY_TIMEOUT_MAX_S:
        raise LocalRuntimeError(
            "agent_core_ready_timeout_invalid",
            "Codna agent-core readiness timeout is outside the allowed range.",
            {
                "env": AGENT_CORE_READY_TIMEOUT_ENV,
                "value": raw,
                "min_seconds": AGENT_CORE_READY_TIMEOUT_MIN_S,
                "max_seconds": AGENT_CORE_READY_TIMEOUT_MAX_S,
            },
        )
    return value


def _redact_log_text(text: str) -> str:
    redacted = SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)} [redacted]" if match.group(1) else "[redacted]", text)
    return SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2[redacted]", redacted)


def _read_log_tail(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"available": False, "path": str(path), "reason": "missing"}
    except OSError as exc:
        return {"available": False, "path": str(path), "reason": str(exc)}
    try:
        with path.open("rb") as handle:
            if size > AGENT_CORE_LOG_TAIL_BYTES:
                handle.seek(size - AGENT_CORE_LOG_TAIL_BYTES)
            data = handle.read(AGENT_CORE_LOG_TAIL_BYTES)
    except OSError as exc:
        return {"available": False, "path": str(path), "reason": str(exc)}
    text = data.decode("utf-8", errors="replace")
    if size > AGENT_CORE_LOG_TAIL_BYTES and "\n" in text:
        text = text.split("\n", 1)[1]
    lines = _redact_log_text(text).splitlines()[-AGENT_CORE_LOG_TAIL_LINES:]
    return {
        "available": True,
        "path": str(path),
        "truncated": size > AGENT_CORE_LOG_TAIL_BYTES,
        "max_bytes": AGENT_CORE_LOG_TAIL_BYTES,
        "recent_lines": lines,
    }


def _load_state(config: RuntimeConfig) -> dict[str, Any] | None:
    payload = read_json(_state_path(config))
    return payload if isinstance(payload, dict) else None


def _load_local_stack_state(config: RuntimeConfig) -> dict[str, Any] | None:
    payload = read_json(config.paths.state_path)
    return payload if isinstance(payload, dict) else None


def _provider_env_hash_for_keys(
    keys: Mapping[str, ConfigValue] | None,
    env_keys: tuple[str, ...],
    *,
    overrides: Mapping[str, str | None] | None = None,
) -> str:
    payload: dict[str, str | None] = {}
    overrides = overrides or {}
    for key in env_keys:
        value = None
        if key in overrides:
            value = overrides[key]
        elif key in os.environ and os.environ[key]:
            value = os.environ[key]
        elif keys and key in keys and keys[key].value:
            value = keys[key].value
        payload[key] = hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provider_env_hash(keys: Mapping[str, ConfigValue] | None) -> str:
    return _provider_env_hash_for_keys(keys, PROVIDER_RUNTIME_ENV_KEYS)


def _legacy_provider_hash_matches_request_mode_only(
    state: dict[str, Any] | None,
    keys: Mapping[str, ConfigValue] | None,
) -> bool:
    if not state:
        return False
    observed = state.get("provider_env_hash")
    if not isinstance(observed, str) or not observed:
        return False
    legacy_keys = (*PROVIDER_RUNTIME_ENV_KEYS, *LEGACY_REQUEST_MODE_ENV_KEYS)
    for mode in (None, "0", "1"):
        overrides = {"AGENT_CORE_TELYS_LOCALIZE_ENABLED": mode}
        if observed == _provider_env_hash_for_keys(keys, legacy_keys, overrides=overrides):
            return True
    return False


def _state_provider_env_matches(
    state: dict[str, Any] | None,
    keys: Mapping[str, ConfigValue] | None,
) -> bool:
    return bool(state and state.get("provider_env_hash") == _provider_env_hash(keys))


def _state_fingerprint_matches(state: dict[str, Any] | None, config: RuntimeConfig) -> bool:
    if not state:
        return False
    return (
        state.get("codna_cli_version") == config.codna_cli_version
        and state.get("sidecar_build_id") == config.sidecar_build_id
        and state.get("runtime_config_hash") == config.runtime_config_hash
    )


def _fingerprint_details(state: dict[str, Any] | None, config: RuntimeConfig) -> dict[str, bool]:
    state = state or {}
    return {
        "codna_cli_matches_state": state.get("codna_cli_version") == config.codna_cli_version,
        "sidecar_build_id_matches_state": state.get("sidecar_build_id") == config.sidecar_build_id,
        "runtime_config_hash_matches_state": state.get("runtime_config_hash")
        == config.runtime_config_hash,
    }


def _state_health_identity_matches(state: dict[str, Any] | None, payload: dict[str, Any]) -> bool:
    if not state:
        return False
    return (
        payload.get("service") == "codna-sidecar"
        and payload.get("runtime_id") == state.get("runtime_id")
    )


def _sidecar_health_payload(probes: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(probes, dict):
        return {}
    health = probes.get("health")
    if not isinstance(health, dict):
        return {}
    payload = health.get("payload")
    return payload if isinstance(payload, dict) else {}


def _sidecar_health_build_matches(config: RuntimeConfig, probes: dict[str, Any] | None) -> bool:
    payload = _sidecar_health_payload(probes)
    build_id = payload.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        return False
    return build_id == config.sidecar_build_id


def _service_is_codna_sidecar(config: RuntimeConfig) -> tuple[bool, dict[str, Any]]:
    health = probe_sidecar_health(config.sidecar_url, timeout_s=HEALTH_TIMEOUT_S)
    ready = probe_sidecar_ready(config.sidecar_url, timeout_s=READY_TIMEOUT_S)
    payload = health.payload or {}
    ok = (
        health.ok
        and ready.ok
        and payload.get("service") == "codna-sidecar"
        and isinstance(payload.get("runtime_id"), str)
    )
    return ok, {"health": _probe_summary(health), "ready": _probe_summary(ready)}


def _state_owns_listener(state: dict[str, Any] | None, listener: ListenerInfo | None) -> bool:
    if not state or listener is None or listener.pid is None or not listener.loopback:
        return False
    sidecar = state.get("sidecar")
    if not isinstance(sidecar, dict):
        return False
    pid = sidecar.get("listener_pid", sidecar.get("pid"))
    created = sidecar.get("listener_pid_create_time", sidecar.get("pid_create_time"))
    if pid != listener.pid:
        return False
    live_created = pid_create_time(listener.pid)
    if created is None or live_created is None:
        return created is None and live_created is None
    return abs(float(created) - live_created) <= 0.001


def _write_state(
    config: RuntimeConfig,
    *,
    runtime_id: str,
    pid: int | None,
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, Any]:
    listener = _single_listener(config.sidecar_port)
    canonical_pid = listener.pid if listener and listener.pid is not None else pid
    create_time = pid_create_time(canonical_pid) if canonical_pid is not None else None
    payload = {
        "schema_version": 1,
        "owner": RUNTIME_OWNER,
        "service": "codna-sidecar",
        "runtime_id": runtime_id,
        "port": config.sidecar_port,
        "url": config.sidecar_url,
        "started_at": _now(),
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
        "provider_env_hash": _provider_env_hash(keys),
        "sidecar": {
            "pid": canonical_pid,
            "pid_create_time": create_time,
            "listener_pid": canonical_pid,
            "listener_pid_create_time": create_time,
            "port": config.sidecar_port,
            "health_url": f"{config.sidecar_url}/health",
            "ready_url": f"{config.sidecar_url}/ready",
        },
    }
    write_json_atomic(_state_path(config), payload)
    return payload


def _collision_error(config: RuntimeConfig, listener: ListenerInfo, probes: dict[str, Any] | None = None) -> LocalRuntimeError:
    return LocalRuntimeError(
        "agent_core_port_collision",
        f"Codna cannot start agent-core because port {config.sidecar_port} is already in use.",
        {
            "port": config.sidecar_port,
            "listener": {
                "pid": listener.pid,
                "parent_pid": listener.parent_pid,
                "command": listener.command,
                "host": listener.host,
                "raw_name": listener.raw_name,
            },
            "state_path": str(_state_path(config)),
            "provider_env_hash": _provider_env_hash(None),
            "probe": probes,
        },
    )


def _wait_for_listener_exit(config: RuntimeConfig, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _single_listener(config.sidecar_port) is None:
            return True
        time.sleep(0.1)
    return _single_listener(config.sidecar_port) is None


def _request_owned_sidecar_shutdown(config: RuntimeConfig) -> bool:
    healthy, probes = _service_is_codna_sidecar(config)
    if not healthy:
        return False
    payload = _sidecar_health_payload(probes)
    runtime_id = payload.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        return False
    shutdown = request_sidecar_shutdown(
        config.sidecar_url,
        runtime_id=runtime_id,
        timeout_s=HEALTH_TIMEOUT_S,
    )
    if not shutdown.ok:
        return False
    return _wait_for_listener_exit(config, timeout_s=min(STOP_GRACE_TIMEOUT_S, 5.0))


def _stop_owned_sidecar(
    config: RuntimeConfig,
    listener: ListenerInfo,
    *,
    reason: str,
) -> None:
    if listener.pid is None:
        return
    if _request_owned_sidecar_shutdown(config):
        return
    # ONLY the verified listener PID -- never listener.parent_pid. `spawn_detached_process` execs
    # the sidecar command directly (no wrapper process), so while the ORIGINAL caller that started
    # it is still alive, lsof's reported "parent" IS that caller, not an orphaned launcher to clean
    # up. On a single machine running several `codna fix` invocations at once (concurrent webhook
    # jobs, or just two terminals), that caller is frequently a DIFFERENT process doing DIFFERENT,
    # unrelated, still-useful work — and this used to SIGTERM-then-SIGKILL it, not just the
    # sidecar. Confirmed live: two concurrent fixes with different provider keys triggered
    # state_fingerprint_or_provider_mismatch, and the "cleanup" killed the peer's own `codna fix`
    # process before its sidecar. `listener.pid` is independently verified as the thing actually
    # LISTENING on the port (that's what put it in `listener` at all); nothing here is verified
    # about parent_pid beyond "lsof reported some number as PPID at one instant." A leftover
    # parent that genuinely has nothing left to do exits on its own once its child (the sidecar)
    # is gone; a manual `codna doctor` hint can still suggest killing it explicitly (see
    # _manual_stop_command) — that's a human decision, never an automated one.
    attempted_pids = [listener.pid]
    try:
        for pid in attempted_pids:
            terminate_pid(
                pid,
                grace_timeout_s=STOP_GRACE_TIMEOUT_S,
                force_timeout_s=STOP_FORCE_TIMEOUT_S,
            )
    except PermissionError as exc:
        raise LocalRuntimeError(
            "agent_core_stop_permission_denied",
            "Codna agent-core is stale or unhealthy but Codna could not stop its owned process.",
            {
                "reason": reason,
                "port": config.sidecar_port,
                "pid": listener.pid,
                "parent_pid": listener.parent_pid,
                "attempted_pids": attempted_pids,
                "command": listener.command,
                "manual_stop_command": _manual_stop_command(listener),
                "manual_stop_note": "Stop the parent and child together; killing only the listener can respawn it.",
                "state_path": str(_state_path(config)),
                "lock_path": str(config.paths.lock_path),
            },
        ) from exc


def _listener_payload(listener: ListenerInfo | None) -> dict[str, Any] | None:
    if listener is None:
        return None
    return {
        "pid": listener.pid,
        "parent_pid": listener.parent_pid,
        "command": listener.command,
        "host": listener.host,
        "port": listener.port,
        "loopback": listener.loopback,
        "raw_name": listener.raw_name,
    }


def _manual_stop_command(listener: ListenerInfo | None) -> str | None:
    if listener is None or listener.pid is None:
        return None
    if listener.parent_pid is not None and listener.parent_pid != listener.pid:
        return f"kill {listener.parent_pid} {listener.pid}"
    return f"kill {listener.pid}"


def _runtime_recovery(
    *,
    action: str,
    reason: str,
    listener: ListenerInfo | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "reason": reason,
    }
    if action in {"start", "restart"}:
        payload["codna_command"] = "codna doctor --start-stack"
    manual_stop = _manual_stop_command(listener)
    if manual_stop:
        payload["manual_stop_command"] = manual_stop
    return payload


def _ensure_mojo_daemon(config: RuntimeConfig) -> None:
    from .local_mojo_daemon import (
        LocalMojoDaemonError,
        ensure_local_mojo_daemon,
        local_mojo_daemon_enabled,
        should_use_local_mojo_backend,
    )

    if not local_mojo_daemon_enabled():
        return
    try:
        if not should_use_local_mojo_backend(config):
            return
        ensure_local_mojo_daemon(config)
    except LocalMojoDaemonError as exc:
        raise LocalRuntimeError(exc.code, str(exc), exc.details) from exc


def _with_mojo_daemon(config: RuntimeConfig, endpoint: AgentCoreEndpoint) -> AgentCoreEndpoint:
    _ensure_mojo_daemon(config)
    return endpoint


def _inspection_status(
    *,
    config: RuntimeConfig,
    listener: ListenerInfo | None,
    state: dict[str, Any] | None,
    probes: dict[str, Any] | None,
    healthy: bool,
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, Any]:
    if listener is None:
        return {
            "status": "not_running",
            "recovery": _runtime_recovery(
                action="start",
                reason="no sidecar listener is bound to the configured port",
                listener=None,
            ),
        }
    if not listener.loopback:
        return {
            "status": "port_collision",
            "recovery": _runtime_recovery(
                action="manual_stop",
                reason="listener is not bound to loopback",
                listener=listener,
            ),
        }
    if not healthy:
        if _state_owns_listener(state, listener):
            return {
                "status": "unhealthy_owned",
                "recovery": _runtime_recovery(
                    action="restart",
                    reason="owned sidecar failed health checks",
                    listener=listener,
                ),
            }
        return {
            "status": "port_collision",
            "recovery": _runtime_recovery(
                action="manual_stop",
                reason="listener did not report Codna sidecar health identity",
                listener=listener,
            ),
        }
    payload = _sidecar_health_payload(probes)
    if (
        _state_owns_listener(state, listener)
        and _state_health_identity_matches(state, payload)
        and _state_provider_env_matches(state, keys)
        and _state_fingerprint_matches(state, config)
        and _sidecar_health_build_matches(config, probes)
    ):
        return {
            "status": "healthy_owned",
            "recovery": {"action": "none", "reason": "sidecar is healthy and owned"},
        }
    reasons: list[str] = []
    if state is None:
        reasons.append("state_missing")
    elif not _state_owns_listener(state, listener):
        reasons.append("state_does_not_own_listener")
    if not _sidecar_health_build_matches(config, probes):
        reasons.append("health_build_mismatch")
    if state is not None and not _state_health_identity_matches(state, payload):
        reasons.append("runtime_id_mismatch")
    if state is not None and not _state_provider_env_matches(state, keys):
        reasons.append("provider_env_mismatch")
    if state is not None and not _state_fingerprint_matches(state, config):
        reasons.append("fingerprint_mismatch")
    return {
        "status": "stale_codna_sidecar",
        "stale_reasons": reasons or ["unknown_stale_state"],
        "recovery": _runtime_recovery(
            action="restart",
            reason="healthy Codna sidecar is not reusable by the current checkout",
            listener=listener,
        ),
    }


def _resolve_config(keys: Mapping[str, ConfigValue] | None) -> RuntimeConfig:
    try:
        return resolve_runtime_config(keys=keys)
    except RuntimeConfigError as exc:
        raise LocalRuntimeError(exc.code, str(exc), exc.details) from exc


def inspect_agent_core_runtime(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> dict[str, Any]:
    config = _resolve_config(keys)
    _engine_log, sidecar_log = _log_paths(config)
    listener = _single_listener(config.sidecar_port)
    state = _load_state(config)
    healthy = False
    probes = None
    if listener and listener.loopback:
        healthy, probes = _service_is_codna_sidecar(config)
    fingerprint_details = _fingerprint_details(state, config)
    status = _inspection_status(
        config=config,
        listener=listener,
        state=state,
        probes=probes,
        healthy=healthy,
        keys=keys,
    )
    return {
        "mode": "local_agent_core",
        "repository_intelligence": "in_process_algenta_sdk",
        "database": "not_required",
        **status,
        "state_path": str(_state_path(config)),
        "lock_path": str(config.paths.lock_path),
        "log_path": str(sidecar_log),
        "runtime_root": str(config.paths.root),
        "step_artifacts_path": str(config.paths.root / "repository-intelligence" / "steps"),
        "url": config.sidecar_url,
        "port": config.sidecar_port,
        "listener": _listener_payload(listener),
        "state": state,
        "healthy": healthy,
        "state_owns_listener": _state_owns_listener(state, listener),
        "provider_env_matches": _state_provider_env_matches(state, keys),
        "health_build_matches": _sidecar_health_build_matches(config, probes),
        "fingerprint": {
            "codna_cli_version": config.codna_cli_version,
            "sidecar_build_id": config.sidecar_build_id,
            "runtime_config_hash": config.runtime_config_hash,
            "matches_state": all(fingerprint_details.values()),
            **fingerprint_details,
        },
        "probes": probes,
    }


def _start_sidecar(
    config: RuntimeConfig,
    *,
    keys: Mapping[str, ConfigValue] | None,
) -> AgentCoreEndpoint:
    ensure_directories(config.paths.runtime_dir, config.paths.logs_dir)
    _engine_log, sidecar_log = _log_paths(config)
    rotate_log(sidecar_log, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_FILES)
    runtime_id = str(uuid.uuid4())
    command, cwd, env = _sidecar_spawn(config, runtime_id, keys=keys)
    ready_timeout_s = _agent_core_ready_timeout_seconds()
    process = spawn_detached_process(command, cwd=cwd, env=env, log_path=sidecar_log)
    probe = wait_until_ready(
        probe_sidecar_ready,
        config.sidecar_url,
        timeout_s=ready_timeout_s,
        request_timeout_s=READY_TIMEOUT_S,
    )
    if not probe.ok:
        if process.pid:
            terminate_pid(process.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        raise LocalRuntimeError(
            "agent_core_start_timeout",
            "Codna agent-core failed while waiting for sidecar readiness.",
            {
                "target_url": f"{config.sidecar_url}/ready",
                "state_path": str(_state_path(config)),
                "log_path": str(sidecar_log),
                "timeout_seconds": ready_timeout_s,
                "probe": _probe_summary(probe),
                "log_tail": _read_log_tail(sidecar_log),
            },
        )
    healthy, probes = _service_is_codna_sidecar(config)
    if not healthy:
        if process.pid:
            terminate_pid(process.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        raise LocalRuntimeError(
            "agent_core_health_mismatch",
            "Codna agent-core sidecar did not report the expected service identity.",
            {
                "target_url": f"{config.sidecar_url}/health",
                "state_path": str(_state_path(config)),
                "log_path": str(sidecar_log),
                "timeout_seconds": ready_timeout_s,
                "probe": probes,
                "log_tail": _read_log_tail(sidecar_log),
            },
        )
    state = _write_state(config, runtime_id=runtime_id, pid=process.pid, keys=keys)
    endpoint = AgentCoreEndpoint(
        url=config.sidecar_url,
        port=config.sidecar_port,
        runtime_id=runtime_id,
        pid=state["sidecar"]["pid"],
    )
    try:
        return _with_mojo_daemon(config, endpoint)
    except LocalRuntimeError:
        listener = _single_listener(config.sidecar_port)
        if listener is not None and listener.loopback:
            _stop_owned_sidecar(config, listener, reason="mojo_daemon_start_failed")
        _unlink_state_if_possible(config)
        raise


def ensure_agent_core_running(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> AgentCoreEndpoint:
    config = _resolve_config(keys)
    if config.remote_engine_url:
        return AgentCoreEndpoint(url=config.sidecar_url, port=config.sidecar_port, runtime_id="remote-engine", pid=None)
    try:
        with runtime_lock(config.paths.lock_path, timeout_s=LOCK_TIMEOUT_S):
            listener = _single_listener(config.sidecar_port)
            state = _load_state(config)
            stack_state = _load_local_stack_state(config)
            if listener is None:
                return _start_sidecar(config, keys=keys)
            if not listener.loopback:
                raise _collision_error(config, listener)
            healthy, probes = _service_is_codna_sidecar(config)
            if healthy:
                payload = _sidecar_health_payload(probes)
                health_build_matches = _sidecar_health_build_matches(config, probes)
                if _state_owns_listener(state, listener):
                    state_identity_matches = _state_health_identity_matches(state, payload)
                    fingerprint_matches = _state_fingerprint_matches(state, config)
                    if (
                        state_identity_matches
                        and _state_provider_env_matches(state, keys)
                        and fingerprint_matches
                        and health_build_matches
                    ):
                        return _with_mojo_daemon(
                            config,
                            AgentCoreEndpoint(
                                url=config.sidecar_url,
                                port=config.sidecar_port,
                                runtime_id=str(payload.get("runtime_id")),
                                pid=listener.pid,
                            ),
                        )
                    if (
                        state_identity_matches
                        and fingerprint_matches
                        and health_build_matches
                        and _legacy_provider_hash_matches_request_mode_only(state, keys)
                    ):
                        adopted_state = _write_state(
                            config,
                            runtime_id=str(payload.get("runtime_id")),
                            pid=listener.pid,
                            keys=keys,
                        )
                        return _with_mojo_daemon(
                            config,
                            AgentCoreEndpoint(
                                url=config.sidecar_url,
                                port=config.sidecar_port,
                                runtime_id=str(payload.get("runtime_id")),
                                pid=adopted_state["sidecar"]["pid"],
                            ),
                        )
                    _stop_owned_sidecar(config, listener, reason="state_fingerprint_or_provider_mismatch")
                    return _start_sidecar(config, keys=keys)
                if (
                    _state_owns_listener(stack_state, listener)
                    and _state_health_identity_matches(stack_state, payload)
                    and _state_fingerprint_matches(stack_state, config)
                    and health_build_matches
                ):
                    runtime_id = str(payload.get("runtime_id"))
                    adopted_state = _write_state(config, runtime_id=runtime_id, pid=listener.pid, keys=keys)
                    return _with_mojo_daemon(
                        config,
                        AgentCoreEndpoint(
                            url=config.sidecar_url,
                            port=config.sidecar_port,
                            runtime_id=runtime_id,
                            pid=adopted_state["sidecar"]["pid"],
                        ),
                    )
                _stop_owned_sidecar(config, listener, reason="codna_sidecar_missing_or_stale_state")
                return _start_sidecar(config, keys=keys)
            if _state_owns_listener(state, listener) and listener.pid is not None:
                _stop_owned_sidecar(config, listener, reason="health_probe_failed")
                return _start_sidecar(config, keys=keys)
            raise _collision_error(config, listener, probes)
    except TimeoutError as exc:
        raise LocalRuntimeError(
            "agent_core_lock_timeout",
            "Timed out while waiting for the Codna agent-core runtime lock.",
            {"lock_path": str(config.paths.lock_path), "timeout_seconds": LOCK_TIMEOUT_S},
        ) from exc
    except PermissionError as exc:
        raise LocalRuntimeError(
            "agent_core_permission_denied",
            "Codna could not access its local agent-core runtime files.",
            {
                "root": str(config.paths.root),
                "state_path": str(_state_path(config)),
                "lock_path": str(config.paths.lock_path),
                "reason": str(exc),
            },
        ) from exc


def stop_agent_core_runtime(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> dict[str, Any]:
    config = _resolve_config(keys)
    if config.remote_engine_url:
        return {
            "status": "remote_override",
            "url": config.sidecar_url,
            "port": config.sidecar_port,
            "state_path": str(_state_path(config)),
        }

    def _stop_mojo_daemon() -> dict[str, Any]:
        from .local_mojo_daemon import LocalMojoDaemonError, stop_local_mojo_daemon

        try:
            return stop_local_mojo_daemon(config)
        except LocalMojoDaemonError as exc:
            return {
                "status": "error",
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "details": exc.details,
                },
            }

    def _stop_unlocked(*, lock_acquired: bool) -> dict[str, Any]:
        listener = _single_listener(config.sidecar_port)
        if listener is None:
            mojo_daemon = _stop_mojo_daemon()
            state_unlink_error = _unlink_state_if_possible(config)
            payload: dict[str, Any] = {
                "status": "not_running",
                "url": config.sidecar_url,
                "port": config.sidecar_port,
                "state_path": str(_state_path(config)),
                "lock_acquired": lock_acquired,
                "mojo_daemon": mojo_daemon,
            }
            if state_unlink_error:
                payload["state_unlink_error"] = state_unlink_error
            return payload
        if not listener.loopback:
            raise _collision_error(config, listener)
        state = _load_state(config) if lock_acquired else None
        healthy, probes = _service_is_codna_sidecar(config)
        if healthy or _state_owns_listener(state, listener):
            listener_payload = _listener_payload(listener)
            _stop_owned_sidecar(config, listener, reason="stop_requested")
            mojo_daemon = _stop_mojo_daemon()
            state_unlink_error = _unlink_state_if_possible(config)
            payload = {
                "status": "stopped",
                "url": config.sidecar_url,
                "port": config.sidecar_port,
                "listener": listener_payload,
                "state_path": str(_state_path(config)),
                "lock_acquired": lock_acquired,
                "mojo_daemon": mojo_daemon,
            }
            if state_unlink_error:
                payload["state_unlink_error"] = state_unlink_error
            return payload
        raise _collision_error(config, listener, probes)

    try:
        with runtime_lock(config.paths.lock_path, timeout_s=LOCK_TIMEOUT_S):
            return _stop_unlocked(lock_acquired=True)
    except TimeoutError as exc:
        raise LocalRuntimeError(
            "agent_core_lock_timeout",
            "Timed out while waiting for the Codna agent-core runtime lock.",
            {"lock_path": str(config.paths.lock_path), "timeout_seconds": LOCK_TIMEOUT_S},
        ) from exc
    except PermissionError as exc:
        try:
            return _stop_unlocked(lock_acquired=False)
        except LocalRuntimeError as stop_exc:
            stop_exc.details.setdefault("lock_path", str(config.paths.lock_path))
            stop_exc.details.setdefault("lock_error", str(exc))
            raise
