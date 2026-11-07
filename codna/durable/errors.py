"""Structured errors shared by the durable command surfaces."""
from __future__ import annotations

from typing import Any


class DurableCliError(RuntimeError):
    """A `codna durable` command cannot proceed. Carries the same structured body as the rest."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
        if self.hint:
            body["hint"] = self.hint
        return {"error": body}


def cli_error(
    message: str, *, code: str, details: dict[str, object] | None = None, hint: str = ""
) -> Any:
    """The CLI's error type, constructed lazily.

    `DurableCliError` is re-exported by `cli.py`, which imports the command modules for registration
    — so reaching it through the module object avoids a cycle AND keeps one definition: a test that
    swaps `cli.DurableCliError` sees the change here too.

    Lives in this leaf module because THREE modules now raise it (`run_cli` for the commands,
    `run_wait` for the wait loop, and `cli.py` itself). A private copy in each would be three chances
    for the hint or the code style to drift.
    """
    from . import cli as _cli

    return _cli.DurableCliError(message, code=code, details=details, hint=hint)
