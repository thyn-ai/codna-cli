from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .config import (
    ENGINE_READY_TIMEOUT_S,
    ENGINE_URL_KEYS,
    HEALTH_TIMEOUT_S,
    LOCK_TIMEOUT_S,
    LOG_ROTATE_BYTES,
    LOG_ROTATE_FILES,
    LOCAL_DATABASE_ENV_KEYS,
    READY_TIMEOUT_S,
    RUNTIME_OWNER,
    SCHEMA_VERSION,
    SIDECAR_READY_TIMEOUT_S,
    STOP_FORCE_TIMEOUT_S,
    STOP_GRACE_TIMEOUT_S,
    ConfigValue,
    RuntimeConfig,
    RuntimeConfigError,
    local_engine_defaults,
    load_env_file,
    resolve_runtime_config,
)
from .health import (
    ProbeResult,
    probe_engine_health,
    probe_engine_ready,
    probe_engine_version,
    probe_sidecar_health,
    probe_sidecar_ready,
    wait_until_ready,
)
from .ports import ListenerInfo, listeners_on_port
from .processes import (
    ensure_directories,
    pid_create_time,
    read_json,
    rotate_log,
    runtime_lock,
    spawn_detached_process,
    terminate_pid,
    write_json_atomic,
)


@dataclass(frozen=True)
class RuntimeEndpoint:
    engine_url: str
    sidecar_url: str | None
    local: bool
    port_base: int | None
    runtime_id: str | None = None


class LocalRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
            }
        }


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _log_paths(config: RuntimeConfig) -> tuple[Path, Path]:
    engine_log = config.paths.logs_dir / f"local-engine-{config.engine_port}.log"
    sidecar_log = config.paths.logs_dir / f"local-sidecar-{config.sidecar_port}.log"
    return engine_log, sidecar_log


def _runtime_env(config: RuntimeConfig, runtime_id: str) -> dict[str, str]:
    return {
        "CODNA_LOCAL_RUNTIME": "1",
        "CODNA_RUNTIME_ID": runtime_id,
        "CODNA_PORT_BASE": str(config.port_base),
        "CODNA_ENGINE_PORT": str(config.engine_port),
        "CODNA_SIDECAR_PORT": str(config.sidecar_port),
        "CODNA_CLI_VERSION": config.codna_cli_version,
        "CODNA_ENGINE_BUILD_ID": config.engine_build_id,
        "CODNA_SIDECAR_BUILD_ID": config.sidecar_build_id,
        "RUNTIME_FALLBACK_MODE": "deny",
    }


CHILD_ENV_KEY_ALLOWLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "CODNA_API_KEY",
        "CURSOR_API_KEY",
        "DE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TELYS_KERNEL",
        "XAI_API_KEY",
    }
)
PROVIDER_CHILD_ENV_KEYS = frozenset(
    key for key in CHILD_ENV_KEY_ALLOWLIST if key.endswith("_API_KEY")
)


def _child_env_from_keys(
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, str]:
    if not keys:
        return {}
    blocked = set(ENGINE_URL_KEYS)
    output: dict[str, str] = {}
    for key, config_value in keys.items():
        if key in blocked or key not in CHILD_ENV_KEY_ALLOWLIST:
            continue
        value = config_value.value.strip()
        if value:
            output[key] = value
    return output


def _merged_child_env_from_keys(
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in _child_env_from_keys(keys).items():
        if key in PROVIDER_CHILD_ENV_KEYS and os.environ.get(key):
            continue
        env[key] = value
    return env


def _remove_local_database_env(env: dict[str, str]) -> dict[str, str]:
    for key in LOCAL_DATABASE_ENV_KEYS:
        env.pop(key, None)
    return env


def _algenta_local_paths(config: RuntimeConfig) -> dict[str, Path]:
    root = config.paths.root
    return {
        "runtime_dir": root / "algenta-runtime",
        "shared_runtime_dir": root / "algenta-runtime-shared",
        "sources_dir": root / "algenta-sources",
        "source_artifact_cache_dir": root / "algenta-source-artifact-cache",
    }


def _owned_sidecar_runtime_paths(config: RuntimeConfig) -> dict[str, Path]:
    root = config.paths.root / "sidecar-runtime"
    return {
        "tmp_dir": root / "tmp",
        "bun_cache_dir": root / "bun-cache",
    }


def _apply_owned_sidecar_runtime_env(env: dict[str, str], config: RuntimeConfig) -> None:
    paths = _owned_sidecar_runtime_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(paths["tmp_dir"])
    env["TMPDIR"] = tmp_dir
    env["TMP"] = tmp_dir
    env["TEMP"] = tmp_dir
    env["BUN_INSTALL_CACHE_DIR"] = str(paths["bun_cache_dir"])


def _engine_spawn(
    config: RuntimeConfig,
    runtime_id: str,
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    engine_entrypoint = config.engine_dir / "apps" / "api_server" / "main.py"
    if not engine_entrypoint.is_file():
        raise LocalRuntimeError(
            "local_engine_runtime_not_installed",
            "Codna local engine runtime is not installed.",
            {
                "engine_dir": str(config.engine_dir),
                "expected_entrypoint": str(engine_entrypoint),
                "hint": "Set ALGENTA_ENGINE_DIR to a checkout containing apps/api_server/main.py, "
                "or use the in-process local SDK path for repository intelligence.",
            },
        )
    algenta_paths = _algenta_local_paths(config)
    env = {
        **load_env_file(config.engine_env_file),
        **_merged_child_env_from_keys(keys),
        **local_engine_defaults(),
        **_runtime_env(config, runtime_id),
    }
    _remove_local_database_env(env)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["ALGENTA_AGENT_CORE_URL"] = config.sidecar_url
    env["ALGENTA_ENGINE_URL"] = config.engine_url
    env["ALGENTA_RUNTIME_DIR"] = str(algenta_paths["runtime_dir"])
    env["ALGENTA_REPOSITORY_INTELLIGENCE_SHARED_RUNTIME_DIR"] = str(
        algenta_paths["shared_runtime_dir"]
    )
    env["ALGENTA_SOURCES_DIR"] = str(algenta_paths["sources_dir"])
    env["ALGENTA_SOURCE_ARTIFACT_CACHE_DIR"] = str(
        algenta_paths["source_artifact_cache_dir"]
    )
    env["RUNTIME_FALLBACK_MODE"] = "deny"
    command = [
        config.engine_python,
        "-m",
        "uvicorn",
        "apps.api_server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.engine_port),
    ]
    return command, config.engine_dir, env


def _sidecar_spawn(
    config: RuntimeConfig,
    runtime_id: str,
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    config.cline_data_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **_merged_child_env_from_keys(keys),
        **_runtime_env(config, runtime_id),
        "AGENT_CORE_PORT": str(config.sidecar_port),
        "AGENT_CORE_HOST": "127.0.0.1",
        "CLINE_DATA_DIR": str(config.cline_data_dir),
    }
    _apply_owned_sidecar_runtime_env(env, config)
    _remove_local_database_env(env)

    # Proof writes land in <AGENT_CORE_ROOT_DIR>/build/proofs/... — point them at a dir codna owns,
    # for EVERY spawn shape. Unset, run-server falls back to the resolved repo/install root, and
    # finalizeSessionArtifacts' mkdir is unguarded and awaited AFTER the model turn: an unwritable
    # root fails the fix *after* paying for it and discards the patch. Compiled binary hits it via
    # read-only `/$bunfs` (EROFS); a packaged deploy hits it via a root-owned install dir (codna-
    # webhook: EACCES on /app/build). Setting this on only one branch WAS the bug. See
    # test_sidecar_spawn_source_checkout_also_redirects_proofs_to_writable_root.
    env["AGENT_CORE_ROOT_DIR"] = str(config.cline_data_dir)

    # Preferred: the self-contained compiled sidecar shipped in the wheel (bun build --compile). It
    # embeds the Bun runtime + the agent-core server + SDK in one per-platform binary, so it needs NO
    # node, NO bun, and NO node_modules — a bare `pip install codna` runs `codna fix` with zero
    # JS-runtime setup.
    sidecar_binary = getattr(config, "sidecar_binary", None)
    if sidecar_binary is not None and Path(sidecar_binary).is_file():
        return [str(sidecar_binary)], Path(sidecar_binary).parent, env

    # Dev / source-checkout fallback: the Node supervisor (run-server.mjs) spawns the Bun server.
    run_server = config.sidecar_dir / "run-server.mjs"
    if not config.sidecar_dir.is_dir() or not run_server.is_file():
        raise LocalRuntimeError(
            "agent_core_runtime_not_installed",
            "Codna agent-core sidecar runtime is not installed.",
            {
                "sidecar_dir": str(config.sidecar_dir),
                "expected_entrypoint": str(run_server),
                "hint": "Install a Codna wheel that bundles the self-contained sidecar, or set "
                "CODNA_SIDECAR_DIR to a directory containing run-server.mjs.",
            },
        )
    node = shutil.which("node")
    if not node:
        raise LocalRuntimeError(
            "node_runtime_not_found",
            "Codna agent-core sidecar requires node on PATH (or the packaged self-contained sidecar).",
            {
                "sidecar_dir": str(config.sidecar_dir),
                "expected_entrypoint": str(run_server),
            },
        )
    bun = shutil.which("bun")
    if not bun and not _sidecar_uses_stub_runtime(env):
        raise LocalRuntimeError(
            "bun_runtime_not_found",
            "Codna agent-core sidecar requires bun on PATH (or the packaged self-contained sidecar).",
            {
                "sidecar_dir": str(config.sidecar_dir),
                "expected_entrypoint": str(run_server),
                "hint": "Install Bun 1.3.x or use a Codna wheel that bundles the self-contained sidecar.",
            },
        )
    command = [node, "run-server.mjs"]
    return command, config.sidecar_dir, env


def _sidecar_uses_stub_runtime(env: Mapping[str, str]) -> bool:
    return (
        env.get("ALGENTA_ALLOW_STUB_RUNTIME") == "1"
        and env.get("ALGENTA_RUN_CLASS") == "offline_contract_only"
        and (
            env.get("NODE_ENV") == "test"
            or env.get("CI_OFFLINE_CONTRACT_TEST") == "1"
        )
    )


def _load_state(config: RuntimeConfig) -> dict[str, Any] | None:
    payload = read_json(config.paths.state_path)
    if not isinstance(payload, dict):
        return None
    return payload


def _listener_summary(listener: ListenerInfo | None) -> dict[str, Any] | None:
    if listener is None:
        return None
    return {
        "pid": listener.pid,
        "parent_pid": listener.parent_pid,
        "command": listener.command,
        "host": listener.host,
        "port": listener.port,
        "raw_name": listener.raw_name,
    }


def _single_listener(port: int) -> ListenerInfo | None:
    listeners = listeners_on_port(port)
    if not listeners:
        return None
    if len(listeners) == 1:
        return listeners[0]
    # Same PID can bind multiple families; prefer the exact loopback listener if present.
    loopback = [listener for listener in listeners if listener.loopback]
    return loopback[0] if loopback else listeners[0]


def _matches_process(expected: dict[str, Any] | None, listener: ListenerInfo | None) -> bool:
    if not expected or listener is None:
        return False
    pid = expected.get("listener_pid", expected.get("pid"))
    created = expected.get("listener_pid_create_time", expected.get("pid_create_time"))
    if not isinstance(pid, int) or listener.pid != pid or not listener.loopback:
        return False
    live_create_time = pid_create_time(pid)
    if created is None or live_create_time is None:
        return created is None and live_create_time is None
    if abs(float(created) - live_create_time) > 0.001:
        return False
    return True


def _entry_pids(entry: dict[str, Any] | None) -> list[int]:
    if not isinstance(entry, dict):
        return []
    pids: list[int] = []
    for key in ("listener_pid", "pid"):
        value = entry.get(key)
        if isinstance(value, int) and value not in pids:
            pids.append(value)
    return pids


def _canonical_pid(default_pid: int, listener: ListenerInfo | None) -> int:
    if listener and isinstance(listener.pid, int):
        return listener.pid
    return default_pid


def _state_fingerprint_matches(state: dict[str, Any] | None, config: RuntimeConfig) -> bool:
    if not state:
        return False
    return (
        state.get("codna_cli_version") == config.codna_cli_version
        and state.get("engine_build_id") == config.engine_build_id
        and state.get("sidecar_build_id") == config.sidecar_build_id
        and state.get("python_executable") == config.engine_python
        and state.get("runtime_config_hash") == config.runtime_config_hash
    )


def _fingerprint_details(state: dict[str, Any] | None, config: RuntimeConfig) -> dict[str, bool]:
    state = state or {}
    return {
        "codna_cli_matches_state": state.get("codna_cli_version") == config.codna_cli_version,
        "engine_build_id_matches_state": state.get("engine_build_id") == config.engine_build_id,
        "sidecar_build_id_matches_state": state.get("sidecar_build_id") == config.sidecar_build_id,
        "python_executable_matches_state": state.get("python_executable") == config.engine_python,
        "runtime_config_hash_matches_state": state.get("runtime_config_hash") == config.runtime_config_hash,
    }


def _probe_payload(probes: dict[str, ProbeResult], name: str) -> dict[str, Any]:
    payload = probes.get(name).payload if name in probes else None
    return payload if isinstance(payload, dict) else {}


def _health_proves_legacy_codna_pair(
    engine_listener: ListenerInfo | None,
    sidecar_listener: ListenerInfo | None,
    probes: dict[str, ProbeResult],
) -> bool:
    if engine_listener is None or sidecar_listener is None:
        return False
    if not engine_listener.loopback or not sidecar_listener.loopback:
        return False
    required_probes = (
        probes.get("engine_health"),
        probes.get("engine_ready"),
        probes.get("sidecar_health"),
        probes.get("sidecar_ready"),
    )
    if not all(probe and probe.ok for probe in required_probes):
        return False
    sidecar_health = _probe_payload(probes, "sidecar_health")
    runtime_id = sidecar_health.get("runtime_id")
    return sidecar_health.get("service") == "codna-sidecar" and isinstance(runtime_id, str) and bool(runtime_id)


def _probe_bundle(config: RuntimeConfig) -> dict[str, ProbeResult]:
    return {
        "engine_health": probe_engine_health(config.engine_url, timeout_s=HEALTH_TIMEOUT_S),
        "engine_ready": probe_engine_ready(config.engine_url, timeout_s=READY_TIMEOUT_S),
        "engine_version": probe_engine_version(config.engine_url, timeout_s=HEALTH_TIMEOUT_S),
        "sidecar_health": probe_sidecar_health(config.sidecar_url, timeout_s=HEALTH_TIMEOUT_S),
        "sidecar_ready": probe_sidecar_ready(config.sidecar_url, timeout_s=READY_TIMEOUT_S),
    }


def _probe_summary(probe: ProbeResult) -> dict[str, Any]:
    return {
        "ok": probe.ok,
        "url": probe.url,
        "status_code": probe.status_code,
        "payload": probe.payload,
        "error": probe.error,
    }


def inspect_runtime(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> dict[str, Any]:
    try:
        config = resolve_runtime_config(keys=keys)
    except RuntimeConfigError as exc:
        raise LocalRuntimeError(exc.code, str(exc), exc.details) from exc
    engine_log, sidecar_log = _log_paths(config)
    if config.remote_engine_url:
        return {
            "status": "remote_override",
            "engine_url": config.remote_engine_url,
            "sidecar_url": None,
            "local": False,
            "port_base": None,
            "runtime_id": None,
            "state_path": str(config.paths.state_path),
            "lock_path": str(config.paths.lock_path),
            "logs": {"engine": str(engine_log), "sidecar": str(sidecar_log)},
            "engine_url_override": asdict(config.engine_url_override) if config.engine_url_override else None,
        }
    state = _load_state(config)
    engine_listener = _single_listener(config.engine_port)
    sidecar_listener = _single_listener(config.sidecar_port)
    probes = _probe_bundle(config) if engine_listener and sidecar_listener else {}
    runtime_id = state.get("runtime_id") if isinstance(state, dict) else None
    engine_owned = _matches_process((state or {}).get("engine"), engine_listener)
    sidecar_owned = _matches_process((state or {}).get("sidecar"), sidecar_listener)
    sidecar_runtime_match = False
    sidecar_service_match = False
    sidecar_build_match = False
    if sidecar_owned and "sidecar_health" in probes:
        payload = probes["sidecar_health"].payload or {}
        sidecar_runtime_match = payload.get("runtime_id") == runtime_id
        sidecar_service_match = payload.get("service") == "codna-sidecar"
        sidecar_build_match = payload.get("build_id") == config.sidecar_build_id
    engine_healthy = bool(probes) and probes["engine_health"].ok and probes["engine_ready"].ok
    sidecar_healthy = bool(probes) and probes["sidecar_health"].ok and probes["sidecar_ready"].ok
    fingerprint_details = _fingerprint_details(state, config)
    fingerprint_ok = all(fingerprint_details.values())
    status = "not_running"
    if engine_listener or sidecar_listener:
        if (
            engine_owned
            and sidecar_owned
            and sidecar_runtime_match
            and sidecar_service_match
            and sidecar_build_match
            and engine_healthy
            and sidecar_healthy
            and fingerprint_ok
        ):
            status = "healthy_owned"
        elif engine_owned or sidecar_owned:
            status = "owned_partial_or_stale"
        elif _health_proves_legacy_codna_pair(engine_listener, sidecar_listener, probes):
            status = "legacy_codna_fixed_pair"
        else:
            status = "collision"
    return {
        "status": status,
        "engine_url": config.engine_url,
        "sidecar_url": config.sidecar_url,
        "local": True,
        "port_base": config.port_base,
        "runtime_id": runtime_id,
        "state_path": str(config.paths.state_path),
        "lock_path": str(config.paths.lock_path),
        "logs": {"engine": str(engine_log), "sidecar": str(sidecar_log)},
        "ignored_engine_url": asdict(config.ignored_engine_url) if config.ignored_engine_url else None,
        "fingerprint": {
            "codna_cli_version": config.codna_cli_version,
            "engine_build_id": config.engine_build_id,
            "sidecar_build_id": config.sidecar_build_id,
            "python_executable": config.engine_python,
            "runtime_config_hash": config.runtime_config_hash,
            "matches_state": fingerprint_ok,
            **fingerprint_details,
        },
        "engine_listener": _listener_summary(engine_listener),
        "sidecar_listener": _listener_summary(sidecar_listener),
        "engine_health": _probe_summary(probes["engine_health"]) if "engine_health" in probes else None,
        "engine_ready": _probe_summary(probes["engine_ready"]) if "engine_ready" in probes else None,
        "engine_version": _probe_summary(probes["engine_version"]) if "engine_version" in probes else None,
        "sidecar_health": _probe_summary(probes["sidecar_health"]) if "sidecar_health" in probes else None,
        "sidecar_ready": _probe_summary(probes["sidecar_ready"]) if "sidecar_ready" in probes else None,
        "state": state,
    }


def _legacy_state_candidates(config: RuntimeConfig) -> list[Path]:
    candidates = [
        Path.home() / ".codna" / "launcher.state",
        Path("/private/tmp/codna-launcher-state.json"),
        Path("/private/tmp/codna-launcher-state-foreground.json"),
    ]
    benchmark_root = config.codna_home / "build" / "benchmarks"
    if benchmark_root.is_dir():
        candidates.extend(benchmark_root.glob("*/launcher/launcher.state"))
    env_state = os.environ.get("CODNA_LAUNCHER_STATE")
    if env_state:
        candidates.append(Path(env_state).expanduser())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _command_looks_codna(command: str | None) -> bool:
    if not command:
        return False
    return (
        "uvicorn apps.api_server.main:app" in command
        or "run-server.mjs" in command
        or "algenta/run-server.ts" in command
        or "codna.supervisor" in command
    )


def _terminate_owned_pid(pid: int, *, service: str) -> None:
    try:
        terminate_pid(
            pid,
            grace_timeout_s=STOP_GRACE_TIMEOUT_S,
            force_timeout_s=STOP_FORCE_TIMEOUT_S,
        )
    except PermissionError as exc:
        raise LocalRuntimeError(
            "runtime_stop_permission_denied",
            f"Codna could not stop the owned {service} process.",
            {
                "service": service,
                "pid": pid,
            },
        ) from exc


def _stop_proven_owned_runtime(config: RuntimeConfig, inspection: dict[str, Any]) -> None:
    sidecar = inspection.get("state", {}).get("sidecar") if isinstance(inspection.get("state"), dict) else None
    engine = inspection.get("state", {}).get("engine") if isinstance(inspection.get("state"), dict) else None
    for pid in _entry_pids(engine):
        _terminate_owned_pid(pid, service="engine")
    for pid in _entry_pids(sidecar):
        _terminate_owned_pid(pid, service="sidecar")
    config.paths.state_path.unlink(missing_ok=True)


def _stop_health_proven_legacy_runtime(config: RuntimeConfig, inspection: dict[str, Any]) -> None:
    engine = inspection.get("engine_listener") if isinstance(inspection.get("engine_listener"), dict) else None
    sidecar = inspection.get("sidecar_listener") if isinstance(inspection.get("sidecar_listener"), dict) else None
    engine_pid = engine.get("pid") if isinstance(engine, dict) else None
    sidecar_pid = sidecar.get("pid") if isinstance(sidecar, dict) else None
    if isinstance(engine_pid, int):
        _terminate_owned_pid(engine_pid, service="engine")
    if isinstance(sidecar_pid, int):
        _terminate_owned_pid(sidecar_pid, service="sidecar")
    config.paths.state_path.unlink(missing_ok=True)


def _healthy_runtime_identity_still_matches(inspection: dict[str, Any]) -> bool:
    fingerprint = inspection.get("fingerprint") if isinstance(inspection.get("fingerprint"), dict) else {}
    if fingerprint.get("matches_state") is not True:
        return False
    state = inspection.get("state") if isinstance(inspection.get("state"), dict) else {}
    runtime_id = state.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        return False
    engine_health = inspection.get("engine_health") if isinstance(inspection.get("engine_health"), dict) else {}
    engine_ready = inspection.get("engine_ready") if isinstance(inspection.get("engine_ready"), dict) else {}
    sidecar_health = inspection.get("sidecar_health") if isinstance(inspection.get("sidecar_health"), dict) else {}
    sidecar_ready = inspection.get("sidecar_ready") if isinstance(inspection.get("sidecar_ready"), dict) else {}
    if not (engine_health.get("ok") is True and engine_ready.get("ok") is True):
        return False
    if not (sidecar_health.get("ok") is True and sidecar_ready.get("ok") is True):
        return False
    payload = sidecar_health.get("payload") if isinstance(sidecar_health.get("payload"), dict) else {}
    return (
        payload.get("service") == "codna-sidecar"
        and payload.get("runtime_id") == runtime_id
        and payload.get("build_id") == state.get("sidecar_build_id")
    )


def _listener_pid(inspection: dict[str, Any], service: str) -> int | None:
    listener = inspection.get(f"{service}_listener")
    if not isinstance(listener, dict):
        return None
    pid = listener.get("pid")
    return pid if isinstance(pid, int) else None


def _can_refresh_healthy_runtime_state(inspection: dict[str, Any]) -> bool:
    if not _healthy_runtime_identity_still_matches(inspection):
        return False
    return _listener_pid(inspection, "engine") is not None and _listener_pid(inspection, "sidecar") is not None


def _refresh_healthy_runtime_state(config: RuntimeConfig, inspection: dict[str, Any]) -> RuntimeEndpoint:
    state = inspection.get("state") if isinstance(inspection.get("state"), dict) else {}
    runtime_id = str(state["runtime_id"])
    engine_pid = _listener_pid(inspection, "engine")
    sidecar_pid = _listener_pid(inspection, "sidecar")
    if engine_pid is None or sidecar_pid is None:
        raise LocalRuntimeError(
            "runtime_state_refresh_failed",
            "Codna local runtime could not refresh state because listener PIDs were missing.",
            {
                "runtime_id": runtime_id,
                "state_path": str(config.paths.state_path),
            },
        )
    _write_state(config, runtime_id, engine_pid, sidecar_pid)
    refreshed = inspect_runtime()
    if refreshed.get("status") != "healthy_owned":
        raise LocalRuntimeError(
            "runtime_state_refresh_failed",
            "Codna local runtime refreshed state, but the owned pair did not become healthy.",
            {
                "runtime_id": runtime_id,
                "inspection": refreshed,
                "state_path": str(config.paths.state_path),
            },
        )
    return RuntimeEndpoint(
        engine_url=config.engine_url,
        sidecar_url=config.sidecar_url,
        local=True,
        port_base=config.port_base,
        runtime_id=runtime_id,
    )


def _migrate_legacy_dynamic_state(config: RuntimeConfig) -> None:
    if config.paths.migration_marker_path.exists():
        return
    ensure_directories(config.paths.runtime_dir)
    for state_path in _legacy_state_candidates(config):
        payload = read_json(state_path)
        if not isinstance(payload, dict):
            continue
        engine_url = payload.get("engine_url")
        sidecar_url = payload.get("sidecar_url")
        if not isinstance(engine_url, str) or not isinstance(sidecar_url, str):
            continue
        engine_port: int | None = None
        sidecar_port: int | None = None
        with contextlib.suppress(Exception):
            engine_port = int(engine_url.rsplit(":", 1)[1])
            sidecar_port = int(sidecar_url.rsplit(":", 1)[1])
        engine_listener = _single_listener(engine_port) if engine_port is not None else None
        sidecar_listener = _single_listener(sidecar_port) if sidecar_port is not None else None
        if (
            engine_listener
            and sidecar_listener
            and engine_listener.loopback
            and sidecar_listener.loopback
            and _command_looks_codna(engine_listener.command)
            and _command_looks_codna(sidecar_listener.command)
        ):
            if sidecar_listener.pid:
                terminate_pid(sidecar_listener.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
            if engine_listener.pid:
                terminate_pid(engine_listener.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        migrated = state_path.with_suffix(state_path.suffix + ".migrated")
        with contextlib.suppress(FileNotFoundError):
            state_path.replace(migrated)
    write_json_atomic(config.paths.migration_marker_path, {"migrated_at": _now()})


def _collision_error(config: RuntimeConfig, inspection: dict[str, Any]) -> LocalRuntimeError:
    listener = inspection.get("engine_listener") or inspection.get("sidecar_listener")
    port = listener.get("port") if isinstance(listener, dict) else config.engine_port
    return LocalRuntimeError(
        "port_collision",
        f"Codna cannot start because port {port} is already in use.",
        {
            "port": port,
            "listener": listener,
            "state_path": str(config.paths.state_path),
            "engine_port": config.engine_port,
            "sidecar_port": config.sidecar_port,
        },
    )


def _write_state(config: RuntimeConfig, runtime_id: str, engine_pid: int, sidecar_pid: int) -> dict[str, Any]:
    engine_log, sidecar_log = _log_paths(config)
    engine_listener = _single_listener(config.engine_port)
    sidecar_listener = _single_listener(config.sidecar_port)
    canonical_engine_pid = _canonical_pid(engine_pid, engine_listener)
    canonical_sidecar_pid = _canonical_pid(sidecar_pid, sidecar_listener)
    engine_listener_pid = engine_listener.pid if engine_listener and engine_listener.pid is not None else canonical_engine_pid
    sidecar_listener_pid = sidecar_listener.pid if sidecar_listener and sidecar_listener.pid is not None else canonical_sidecar_pid
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": runtime_id,
        "owner": RUNTIME_OWNER,
        "port_base": config.port_base,
        "started_at": _now(),
        "engine": {
            "port": config.engine_port,
            "pid": canonical_engine_pid,
            "pid_create_time": pid_create_time(canonical_engine_pid),
            "listener_pid": engine_listener_pid,
            "listener_pid_create_time": pid_create_time(engine_listener_pid),
            "health_url": f"{config.engine_url}/v1/health",
            "ready_url": f"{config.engine_url}/v1/health",
            "log_path": str(engine_log),
        },
        "sidecar": {
            "port": config.sidecar_port,
            "pid": canonical_sidecar_pid,
            "pid_create_time": pid_create_time(canonical_sidecar_pid),
            "listener_pid": sidecar_listener_pid,
            "listener_pid_create_time": pid_create_time(sidecar_listener_pid),
            "health_url": f"{config.sidecar_url}/health",
            "ready_url": f"{config.sidecar_url}/ready",
            "log_path": str(sidecar_log),
        },
        "codna_cli_version": config.codna_cli_version,
        "engine_build_id": config.engine_build_id,
        "sidecar_build_id": config.sidecar_build_id,
        "python_executable": config.engine_python,
        "runtime_config_hash": config.runtime_config_hash,
    }
    write_json_atomic(config.paths.state_path, payload)
    return payload


def _start_runtime(
    config: RuntimeConfig,
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> RuntimeEndpoint:
    algenta_paths = _algenta_local_paths(config)
    ensure_directories(config.paths.runtime_dir, config.paths.logs_dir, config.cline_data_dir)
    ensure_directories(*algenta_paths.values())
    _migrate_legacy_dynamic_state(config)
    runtime_id = str(uuid.uuid4())
    engine_log, sidecar_log = _log_paths(config)
    rotate_log(engine_log, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_FILES)
    rotate_log(sidecar_log, max_bytes=LOG_ROTATE_BYTES, backups=LOG_ROTATE_FILES)
    sidecar_cmd, sidecar_cwd, sidecar_env = _sidecar_spawn(config, runtime_id, keys=keys)
    sidecar = spawn_detached_process(sidecar_cmd, cwd=sidecar_cwd, env=sidecar_env, log_path=sidecar_log)
    sidecar_probe = wait_until_ready(
        probe_sidecar_ready,
        config.sidecar_url,
        timeout_s=SIDECAR_READY_TIMEOUT_S,
        request_timeout_s=READY_TIMEOUT_S,
    )
    if not sidecar_probe.ok:
        if sidecar.pid:
            terminate_pid(sidecar.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        raise LocalRuntimeError(
            "sidecar_start_timeout",
            "Codna local runtime failed while waiting for sidecar readiness.",
            {
                "target_url": f"{config.sidecar_url}/ready",
                "state_path": str(config.paths.state_path),
                "log_path": str(sidecar_log),
                "probe": _probe_summary(sidecar_probe),
            },
        )
    engine_cmd, engine_cwd, engine_env = _engine_spawn(config, runtime_id, keys=keys)
    engine = spawn_detached_process(engine_cmd, cwd=engine_cwd, env=engine_env, log_path=engine_log)
    engine_probe = wait_until_ready(
        probe_engine_ready,
        config.engine_url,
        timeout_s=ENGINE_READY_TIMEOUT_S,
        request_timeout_s=READY_TIMEOUT_S,
    )
    if not engine_probe.ok:
        if engine.pid:
            terminate_pid(engine.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        if sidecar.pid:
            terminate_pid(sidecar.pid, grace_timeout_s=STOP_GRACE_TIMEOUT_S, force_timeout_s=STOP_FORCE_TIMEOUT_S)
        raise LocalRuntimeError(
            "engine_start_timeout",
            "Codna local runtime failed while waiting for engine readiness.",
            {
                "target_url": f"{config.engine_url}/v1/health",
                "state_path": str(config.paths.state_path),
                "engine_log": str(engine_log),
                "sidecar_log": str(sidecar_log),
                "probe": _probe_summary(engine_probe),
            },
        )
    _write_state(config, runtime_id, engine.pid, sidecar.pid)
    return RuntimeEndpoint(
        engine_url=config.engine_url,
        sidecar_url=config.sidecar_url,
        local=True,
        port_base=config.port_base,
        runtime_id=runtime_id,
    )


def _permission_error(config: RuntimeConfig, exc: PermissionError) -> LocalRuntimeError:
    return LocalRuntimeError(
        "runtime_permission_denied",
        "Codna could not access its local runtime files.",
        {
            "root": str(config.paths.root),
            "state_path": str(config.paths.state_path),
            "lock_path": str(config.paths.lock_path),
            "reason": str(exc),
        },
    )


def ensure_running(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> RuntimeEndpoint:
    try:
        config = resolve_runtime_config(keys=keys)
    except RuntimeConfigError as exc:
        raise LocalRuntimeError(exc.code, str(exc), exc.details) from exc
    if config.remote_engine_url:
        return RuntimeEndpoint(
            engine_url=config.remote_engine_url,
            sidecar_url=None,
            local=False,
            port_base=None,
            runtime_id=None,
        )
    try:
        with runtime_lock(config.paths.lock_path, timeout_s=LOCK_TIMEOUT_S):
            inspection = inspect_runtime(keys=keys)
            if inspection["status"] == "healthy_owned":
                return RuntimeEndpoint(
                    engine_url=config.engine_url,
                    sidecar_url=config.sidecar_url,
                    local=True,
                    port_base=config.port_base,
                    runtime_id=inspection.get("runtime_id"),
                )
            if inspection["status"] == "owned_partial_or_stale":
                if _can_refresh_healthy_runtime_state(inspection):
                    return _refresh_healthy_runtime_state(config, inspection)
                _stop_proven_owned_runtime(config, inspection)
            elif inspection["status"] == "legacy_codna_fixed_pair":
                _stop_health_proven_legacy_runtime(config, inspection)
            elif inspection["status"] == "collision":
                raise _collision_error(config, inspection)
            return _start_runtime(config, keys=keys)
    except TimeoutError as exc:
        raise LocalRuntimeError(
            "runtime_lock_timeout",
            "Timed out while waiting for the Codna local runtime lock.",
            {
                "lock_path": str(config.paths.lock_path),
                "timeout_seconds": LOCK_TIMEOUT_S,
            },
        ) from exc
    except PermissionError as exc:
        raise _permission_error(config, exc) from exc


def stop_runtime(
    *,
    keys: Mapping[str, ConfigValue] | None = None,
) -> dict[str, Any]:
    try:
        config = resolve_runtime_config(keys=keys)
    except RuntimeConfigError as exc:
        raise LocalRuntimeError(exc.code, str(exc), exc.details) from exc
    if config.remote_engine_url:
        return {
            "status": "remote_override",
            "engine_url": config.remote_engine_url,
            "state_path": str(config.paths.state_path),
        }
    try:
        with runtime_lock(config.paths.lock_path, timeout_s=LOCK_TIMEOUT_S):
            inspection = inspect_runtime(keys=keys)
            if inspection["status"] == "not_running":
                return {
                    "status": "not_running",
                    "state_path": str(config.paths.state_path),
                }
            if inspection["status"] == "collision":
                raise _collision_error(config, inspection)
            if inspection["status"] == "legacy_codna_fixed_pair":
                _stop_health_proven_legacy_runtime(config, inspection)
                return {
                    "status": "stopped",
                    "state_path": str(config.paths.state_path),
                    "runtime_id": inspection.get("runtime_id"),
                }
            _stop_proven_owned_runtime(config, inspection)
            return {
                "status": "stopped",
                "state_path": str(config.paths.state_path),
                "runtime_id": inspection.get("runtime_id"),
            }
    except TimeoutError as exc:
        raise LocalRuntimeError(
            "runtime_lock_timeout",
            "Timed out while waiting for the Codna local runtime lock.",
            {
                "lock_path": str(config.paths.lock_path),
                "timeout_seconds": LOCK_TIMEOUT_S,
            },
        ) from exc
    except PermissionError as exc:
        raise _permission_error(config, exc) from exc


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m codna.runtime.local_stack")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    sub.add_parser("ensure")
    sub.add_parser("stop")
    parser.set_defaults(command="status")
    args = parser.parse_args(argv)
    try:
        if args.command == "ensure":
            endpoint = ensure_running()
            _print_json({"status": "running", **asdict(endpoint)})
            return 0
        if args.command == "stop":
            _print_json(stop_runtime())
            return 0
        _print_json(inspect_runtime())
        return 0
    except LocalRuntimeError as error:
        _print_json(error.to_dict())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
