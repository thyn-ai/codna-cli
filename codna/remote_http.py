"""Remote HTTP client configuration helpers."""
from __future__ import annotations

from collections.abc import Mapping

import httpx

DEFAULT_REMOTE_HTTP_TIMEOUT_S = 30.0
DEFAULT_REMOTE_HTTP_CONNECT_TIMEOUT_S = 5.0
MAX_REMOTE_HTTP_TIMEOUT_S = 3600.0


class RemoteHttpTimeoutError(ValueError):
    """Invalid remote HTTP timeout configuration."""


def remote_http_timeout(environ: Mapping[str, str]) -> httpx.Timeout:
    """Build a bounded HTTP timeout for explicit remote engine calls."""
    total = _remote_http_timeout_seconds(environ)
    connect = _remote_http_connect_timeout_seconds(environ, total)
    return httpx.Timeout(timeout=total, connect=connect, write=connect, pool=connect)


def _remote_http_timeout_seconds(environ: Mapping[str, str]) -> float:
    raw = environ.get("CODNA_HTTP_TIMEOUT_S") or environ.get("CODNA_TIMEOUT_S")
    if raw is None:
        return DEFAULT_REMOTE_HTTP_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise RemoteHttpTimeoutError("CODNA_HTTP_TIMEOUT_S/CODNA_TIMEOUT_S must be a number of seconds.") from exc
    if value <= 0 or value > MAX_REMOTE_HTTP_TIMEOUT_S:
        raise RemoteHttpTimeoutError(
            f"CODNA_HTTP_TIMEOUT_S/CODNA_TIMEOUT_S must be > 0 and <= {MAX_REMOTE_HTTP_TIMEOUT_S:g}."
        )
    return value


def _remote_http_connect_timeout_seconds(environ: Mapping[str, str], total_timeout: float) -> float:
    raw = environ.get("CODNA_HTTP_CONNECT_TIMEOUT_S")
    if raw is None:
        return min(DEFAULT_REMOTE_HTTP_CONNECT_TIMEOUT_S, total_timeout)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RemoteHttpTimeoutError("CODNA_HTTP_CONNECT_TIMEOUT_S must be a number of seconds.") from exc
    if value <= 0 or value > total_timeout:
        raise RemoteHttpTimeoutError("CODNA_HTTP_CONNECT_TIMEOUT_S must be > 0 and <= the request timeout.")
    return value
