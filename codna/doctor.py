"""Read-only Codna runtime diagnostics.

`codna doctor` must not start, stop, heal, or ping local services. It reports the
configuration and local-stack state facts needed to debug the runtime safely.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from . import supervisor
from .runtime_config import LOCAL_STACK_STATE_SOURCE

ENGINE_URL_KEYS = ("CODNA_ENGINE_URL", "ALGENTA_ENGINE_URL", "ALGENTA_BASE_URL")
API_KEY_KEYS = ("CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY")
PUBLIC_CONFIGURED_VALUE = "<configured>"
PUBLIC_STATE_PATH = Path.home() / ".codna" / "runtime" / "local-stack.json"
_URL_CREDENTIAL_RE = re.compile(r"(https?://)([^/@\s\"']+@)")
_SECRET_FIELD_RE = re.compile(r"(?i)\b(password|token|secret|api[_-]?key)(\s*[:=]\s*)[^\s,}\]]+")


def _first_config_source(env: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in env:
            return key
    return None


def _public_port_base(env: Mapping[str, str]) -> str | None:
    if "CODNA_PORT_BASE" not in env:
        return None
    try:
        return str(int(env["CODNA_PORT_BASE"]))
    except ValueError:
        return "invalid"


def public_env_snapshot(env: Mapping[str, str]) -> dict[str, str]:
    """Return only non-secret configuration facts safe for terminal output."""
    public_env: dict[str, str] = {}
    for key in ENGINE_URL_KEYS:
        if key in env:
            public_env[key] = PUBLIC_CONFIGURED_VALUE
            break
    for key in API_KEY_KEYS:
        if key in env:
            public_env[key] = PUBLIC_CONFIGURED_VALUE
            break
    port_base = _public_port_base(env)
    if port_base is not None:
        public_env["CODNA_PORT_BASE"] = port_base
    return public_env


def read_public_state(state_path: Path = PUBLIC_STATE_PATH) -> dict[str, Any] | None:
    try:
        state = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    pid = state.get("pid")
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return None
    return state


def build_report(
    *,
    env: Mapping[str, str] | None = None,
    state_reader: Callable[[], dict[str, Any] | None] = supervisor.read_state,
    state_path: Path = supervisor.STATE_PATH,
) -> dict[str, Any]:
    source_env = os.environ if env is None else env
    explicit_url_source = _first_config_source(source_env, ENGINE_URL_KEYS)
    api_key_source = _first_config_source(source_env, API_KEY_KEYS)
    state = state_reader()
    port_error = None
    port_base = None
    default_engine_url = None
    default_sidecar_url = None
    try:
        port_base_value = source_env["CODNA_PORT_BASE"] if "CODNA_PORT_BASE" in source_env else None
        port_base = supervisor.resolve_port_base(port_base_value)
        default_engine_url, default_sidecar_url = supervisor.local_runtime_urls(port_base)
    except ValueError as exc:
        port_error = str(exc)

    if explicit_url_source:
        mode = "explicit_env"
        engine_url = f"<set by {explicit_url_source}>"
        sidecar_url = None
    elif state and state.get("engine_url"):
        mode = LOCAL_STACK_STATE_SOURCE
        engine_url = str(state["engine_url"]).rstrip("/")
        sidecar_url = state.get("sidecar_url")
    else:
        mode = "local_default"
        engine_url = default_engine_url
        sidecar_url = default_sidecar_url

    return {
        "schema_version": 1,
        "runtime": {
            "mode": mode,
            "engine_url": engine_url,
            "sidecar_url": sidecar_url,
            "default_engine_url": default_engine_url,
            "default_sidecar_url": default_sidecar_url,
            "port_base": port_base,
            "state_path": str(state_path),
            "state_file_present": state_path.exists(),
            "state_valid": state is not None,
            "health_checked": False,
        },
        "config": {
            "engine_url_source": explicit_url_source or (LOCAL_STACK_STATE_SOURCE if state else "default"),
            "api_key_source": api_key_source,
            "api_key_present": api_key_source is not None,
            "port_base_source": "CODNA_PORT_BASE" if "CODNA_PORT_BASE" in source_env else "default",
            "port_base_error": port_error,
        },
    }


def build_public_report(
    *,
    env: Mapping[str, str] | None = None,
    state_reader: Callable[[], dict[str, Any] | None] = read_public_state,
    state_path: Path = PUBLIC_STATE_PATH,
) -> dict[str, Any]:
    source_env = os.environ if env is None else env
    return build_report(
        env=public_env_snapshot(source_env),
        state_reader=state_reader,
        state_path=state_path,
    )


def build_cli_public_report(state_path: Path = PUBLIC_STATE_PATH) -> dict[str, Any]:
    state_file_present = state_path.exists()
    state_valid = False

    port_base_error = None
    mode = "local_default"
    engine_url = "http://127.0.0.1:18600"
    sidecar_url = "http://127.0.0.1:18601"
    default_engine_url = engine_url
    default_sidecar_url = sidecar_url
    port_base = 18600

    return {
        "schema_version": 1,
        "runtime": {
            "mode": mode,
            "engine_url": engine_url,
            "sidecar_url": sidecar_url,
            "default_engine_url": default_engine_url,
            "default_sidecar_url": default_sidecar_url,
            "port_base": port_base,
            "state_path": str(state_path),
            "state_file_present": state_file_present,
            "state_valid": state_valid,
            "health_checked": False,
        },
        "config": {
            "engine_url_source": LOCAL_STACK_STATE_SOURCE if state_valid else "default",
            "api_key_source": None,
            "api_key_present": False,
            "port_base_source": "default",
            "port_base_error": port_base_error,
        },
    }


def render_report(report: Mapping[str, Any]) -> str:
    runtime = report["runtime"]
    config = report["config"]
    lines = [
        "Codna runtime:",
        f"  mode             : {runtime['mode']}",
        f"  engine_url       : {runtime['engine_url'] or 'invalid'}",
        f"  sidecar_url      : {runtime['sidecar_url'] or 'n/a'}",
        f"  port_base        : {runtime['port_base'] or 'invalid'}",
        f"  state_path       : {runtime['state_path']}",
        f"  state_present    : {runtime['state_file_present']}",
        f"  state_valid      : {runtime['state_valid']}",
        f"  health_checked   : {runtime['health_checked']}",
        "Codna config:",
        f"  engine_url_source: {config['engine_url_source']}",
        f"  api_key_present  : {config['api_key_present']}",
        f"  api_key_source   : {config['api_key_source'] or 'n/a'}",
        f"  port_base_source : {config['port_base_source']}",
    ]
    if config.get("port_base_error"):
        lines.append(f"  port_base_error  : {config['port_base_error']}")
    return "\n".join(lines)


def redact_display_text(text: str) -> str:
    without_url_credentials = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", text)
    return _SECRET_FIELD_RE.sub(r"\1\2<redacted>", without_url_credentials)


def format_public_output(report: Mapping[str, Any], *, json_output: bool) -> str:
    if json_output:
        return redact_display_text(json.dumps(report, indent=2, sort_keys=True))
    return redact_display_text(render_report(report))
