"""Structured errors shared by the durable command surfaces."""
from __future__ import annotations


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
