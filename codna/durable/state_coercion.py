"""Coercion helpers for `state.py` — every one names the offending field.

Split out for the repository's 1000-line ceiling (`tests/test_modularity.py`), not for taste: `state.py`
crossed it when `AskRef` landed for D4, and a ceiling enforced by a test is only useful if the response
to it is a real boundary rather than shaving comments.

The boundary is genuine. These helpers are PRIVATE to the state module — nothing outside imports them
(checked: every `from .state import` in the package names public classes only) — and they share exactly
one dependency, `TaskStateError`. So this module has no cycle with `state.py` as long as `state.py`
imports it and it does not import `state.py` back; `TaskStateError` lives in `state.py`, so the error
type is INJECTED as a factory rather than imported.

WHY NOT JUST MOVE `TaskStateError` TOO: it is part of the public surface (`activity_steps.py` imports it
from `.state`), and moving it would either break that import or require a re-export that makes this
module's boundary a fiction. A factory keeps one definition and no cycle.

Every helper raises the SAME structured shape (`code="invalid_task_state"` + `details.field`) so a
corrupted state names which field failed rather than "invalid state".
"""
from __future__ import annotations

from typing import Any, Callable, Mapping


class Coercion:
    """The coercion surface for one state module, bound to that module's error type.

    Built rather than imported, so the error class stays defined in exactly one place (`state.py`) and
    this module never imports back into it.
    """

    __slots__ = ("_error",)

    def __init__(self, error: Callable[..., Exception]) -> None:
        self._error = error

    def _fail(self, label: str, expected: str, value: Any) -> Exception:
        return self._error(
            f"{label} must be {expected}",
            code="invalid_task_state",
            details={"field": label, "type": type(value).__name__},
        )

    def as_mapping(self, raw: Any, label: str) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise self._fail(label, "an object", raw)
        return {str(key): value for key, value in raw.items()}

    def require_str(self, data: Mapping[str, Any], key: str, label: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise self._fail(label, "a non-empty string", value)
        return value

    def optional_str(self, value: Any, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise self._fail(label, "a non-empty string or null", value)
        return value

    def require_int(self, data: Mapping[str, Any], key: str, label: str) -> int:
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._fail(label, "an integer", value)
        return value

    def optional_int(self, value: Any, label: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._fail(label, "an integer or null", value)
        return value

    def optional_number(self, value: Any, label: str) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise self._fail(label, "a number or null", value)
        return float(value)
