"""Helpers for rendering engine/SDK failures in CLI-friendly form."""
from __future__ import annotations

import json
from typing import Any


def _pick(mapping: dict[str, Any] | None, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _details_text(details: Any) -> str | None:
    if details in (None, "", {}, []):
        return None
    if isinstance(details, str):
        return details
    try:
        return json.dumps(details, sort_keys=True)
    except TypeError:
        return repr(details)


def format_cli_error(exc: BaseException) -> str:
    """Render a structured exception without depending on a specific SDK version."""
    error_obj = getattr(exc, "error", None)
    head = _pick(error_obj, "message") or str(exc).strip() or exc.__class__.__name__
    code = getattr(exc, "error_code", None) or getattr(exc, "code", None) or _pick(error_obj, "code")
    request_id = getattr(exc, "request_id", None) or _pick(error_obj, "request_id")
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    details = getattr(exc, "details", None)
    if details in (None, "", {}, []):
        details = _pick(error_obj, "details")

    parts = [head]
    if code:
        parts.append(f"code={code}")
    if request_id:
        parts.append(f"request_id={request_id}")
    if status not in (None, ""):
        parts.append(f"status={status}")
    details_text = _details_text(details)
    if details_text:
        parts.append(f"details={details_text}")
    return " | ".join(parts)
