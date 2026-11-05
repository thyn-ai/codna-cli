"""The shared VALIDATION primitives every durable contract uses — a LEAF module.

`contracts.py`, `activity_responses.py` and `run_contracts.py` were one file until it crossed the
repository's 1000-line ceiling (`test_modularity.py`). Splitting it any other way creates a cycle:
every one of the three validates a reference or a cap, so the validator would have to live in one of
them and be imported by the other two. It lives here instead — stdlib and `dataclasses` only, and
importable by all three without ordering games.

`ContractError` lives here for the same reason, and it is the error type the whole boundary speaks:
a malformed request, a malformed response and a malformed run spec are the same class of failure
(something crossed a boundary that should not have), and a caller should be able to catch one type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

__all__ = [
    "ContractError",
    "GraphEngine",
    "MAX_STEER_NOTE_CHARS",
    "MAX_STEER_NOTES",
    "optional_cap",
    "require_engine",
    "require_int",
    "require_ref",
]

GraphEngine = Literal["langgraph", "plain"]

MAX_STEER_NOTES: Final = 20
MAX_STEER_NOTE_CHARS: Final = 2000


@dataclass(frozen=True)
class ContractError(ValueError):
    """A request or response that may not cross the workflow boundary.

    A ``ValueError`` (so generic callers and `pytest.raises(ValueError)` see it) that also carries a
    stable ``code`` and a machine-readable ``details`` map — the same structured shape the rest of the
    CLI's errors use, so a failure on this boundary is diagnosable without a traceback.
    """

    message: str
    code: str = "contract_invalid"
    details: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        ValueError.__init__(self, self.message)
        object.__setattr__(self, "details", dict(self.details or {}))

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def require_ref(value: Any, label: str) -> str:
    """A short identifier: non-empty, bounded, single-line.

    Bounded because these values reach workflow history, and a 512-character ceiling is generous for
    an id while still refusing a transcript that was passed where a reference belonged.
    """
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string", details={"field": label})
    text = value.strip()
    if len(text) > 512:
        raise ContractError(
            f"{label} is {len(text)} characters, over the 512-character limit",
            details={"field": label, "length": len(text)},
        )
    return text


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(
            f"{label} must be an integer >= {minimum}",
            details={"field": label, "value": repr(value)},
        )
    return value


def optional_cap(value: Any, label: str) -> float | None:
    """A non-negative number or None — a budget cap, never a negative one."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ContractError(
            f"{label} must be a non-negative number or null",
            details={"field": label, "value": repr(value)},
        )
    return float(value)


def require_engine(value: Any) -> GraphEngine:
    """One of the two engines, refusing anything else.

    No default on a typo: the engines produce different history, and a run started under the wrong one
    cannot be replayed by the worker that picks it up.
    """
    if value not in ("langgraph", "plain"):
        raise ContractError(
            "graph engine must be one of langgraph, plain",
            details={"field": "engine", "value": repr(value)},
        )
    return value  # type: ignore[return-value]  # narrowed by the membership check
