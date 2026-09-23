"""Runtime endpoint selection for Codna CLI surfaces.

This module owns the deterministic local-port contract and engine URL/key
precedence. It does not start, stop, ping, or mutate runtime processes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

ENGINE_URL_KEYS = ("CODNA_ENGINE_URL", "ALGENTA_ENGINE_URL", "ALGENTA_BASE_URL")
API_KEY_KEYS = ("CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY")
DEFAULT_PORT_BASE = 18600
MIN_PORT = 1024
MAX_BASE_PORT = 65534
LOCAL_STACK_STATE_SOURCE = "local_stack_state"


class RuntimeConfigError(ValueError):
    """Invalid runtime configuration supplied by the user or environment."""


@dataclass(frozen=True)
class EngineSelection:
    engine_url: str
    api_key: str | None
    engine_url_source: str
    api_key_source: str | None


def _first_env(env: Mapping[str, str], keys: tuple[str, ...]) -> tuple[str | None, str | None]:
    for key in keys:
        value = env.get(key)
        if value:
            return value, key
    return None, None


def resolve_api_key(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    return _first_env(env, API_KEY_KEYS)


def resolve_port_base(value: str | None = None) -> int:
    if value is None or value == "":
        return DEFAULT_PORT_BASE
    try:
        base = int(value)
    except ValueError as exc:
        raise RuntimeConfigError("CODNA_PORT_BASE must be an integer.") from exc
    if not MIN_PORT <= base <= MAX_BASE_PORT:
        raise RuntimeConfigError(f"CODNA_PORT_BASE must be between {MIN_PORT} and {MAX_BASE_PORT}.")
    return base


def local_runtime_urls(port_base: int | None = None) -> tuple[str, str]:
    base = DEFAULT_PORT_BASE if port_base is None else resolve_port_base(str(port_base))
    return f"http://127.0.0.1:{base}", f"http://127.0.0.1:{base + 1}"


def resolve_local_urls_from_env(env: Mapping[str, str]) -> tuple[str, str]:
    return local_runtime_urls(resolve_port_base(env.get("CODNA_PORT_BASE")))


def resolve_engine_selection(
    *,
    env: Mapping[str, str],
    state_reader: Callable[[], dict[str, Any] | None],
) -> EngineSelection:
    explicit_url, explicit_url_source = _first_env(env, ENGINE_URL_KEYS)
    api_key, api_key_source = _first_env(env, API_KEY_KEYS)
    if explicit_url:
        return EngineSelection(
            engine_url=explicit_url.rstrip("/"),
            api_key=api_key,
            engine_url_source=explicit_url_source or "env",
            api_key_source=api_key_source,
        )

    state = state_reader()
    if state and state.get("engine_url"):
        return EngineSelection(
            engine_url=str(state["engine_url"]).rstrip("/"),
            api_key=api_key,
            engine_url_source=LOCAL_STACK_STATE_SOURCE,
            api_key_source=api_key_source,
        )

    engine_url, _sidecar_url = resolve_local_urls_from_env(env)
    return EngineSelection(
        engine_url=engine_url,
        api_key=api_key,
        engine_url_source="local_default",
        api_key_source=api_key_source,
    )
