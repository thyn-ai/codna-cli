"""The run-level BOOKKEEPING contracts: episode, notify, worktree cleanup (D5).

Split out of `contracts.py` at the repository's 1000-line ceiling (`test_modularity.py`), on a real
concern boundary rather than by size: every type here describes a step that records or reports
something ABOUT a run, as opposed to the task-transforming requests (`PlanRequest`, `TurnRequest`,
`CheckpointRequest`, …) that live in `contracts.py`.

The distinction matters for review: a change to a bookkeeping request cannot alter what a turn DOES,
and a reviewer reading this file knows the blast radius is the audit trail rather than the code. They
are re-exported from `contracts`, so every caller keeps importing one module.

`episode` carries the task's BRIEF now, because the episode is what a LATER task's plan reads (D5 goal
mode): without it a recap says "task-2 was approved" with no indication of what task-2 was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract_primitives import ContractError, require_ref as _require_ref
from .task_context import TaskContext
from .state import find_secret_like_strings, validated_state

__all__ = [
    "EpisodeRequest",
    "NotifyRequest",
    "WorktreeCleanupOutcome",
    "WorktreeCleanupRequest",
]


@dataclass(frozen=True)
class EpisodeRequest:
    """Record what this task learned, as a reference-only episode file plus a ledger note.

    ``brief`` is the task's one-line statement of work. It is carried because the episode is what a
    LATER task's plan reads (D5 goal mode): without it a recap says "task-2 was approved" with no
    indication of what task-2 was, and the next plan cannot tell whether the work is related. Bounded
    like every other value here, and omitted (``""``) by a caller that has no brief.
    """

    state: dict[str, Any]
    # ``TaskContext`` is imported from its own LEAF module, not from `contracts`: Temporal decodes a
    # request from its annotations, so a forward reference (or ``Any``) is decoded as a plain dict
    # and `validate()` then fails with `'dict' object has no attribute 'validate'` — measured.
    context: TaskContext
    outcome: str
    brief: str = ""

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise ContractError(
                "episode outcome must be non-empty", code="contract_invalid", details={"field": "outcome"}
            )
        if not isinstance(self.brief, str) or len(self.brief) > 32_000:
            raise ContractError(
                "episode brief must be a string of at most 32,000 characters",
                code="contract_invalid",
                details={"field": "brief"},
            )
        findings = find_secret_like_strings({"brief": self.brief})
        if findings:
            raise ContractError(
                "an episode brief contains secret-like data",
                code="contract_contains_secret",
                details={"findings": findings},
            )
        return validated_state(self.state)


@dataclass(frozen=True)
class WorktreeCleanupRequest:
    """Reclaim a finished run's per-task worktrees (D5). Run-level, idempotent, bounded.

    Carries the run's coordinates rather than a `TaskContext` because it is scheduled by `RunWorkflow`,
    not by a task: the sweep is about the run's whole directory, and a run whose tasks all failed still
    has worktrees to reclaim.

    ``reason`` is recorded in the ledger, so "why are these checkouts gone" has an answer that does not
    depend on remembering what a worker did.
    """

    run_id: str
    repo_dir: str
    reason: str
    # ``True`` keeps the checkouts and only records why. A run kept for debugging must be able to say
    # so explicitly rather than by simply not cleaning up, which is indistinguishable from a leak.
    keep: bool = False

    def validate(self) -> None:
        _require_ref(self.run_id, "run_id")
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "cleanup repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 512:
            raise ContractError(
                "cleanup reason must be a non-empty string of at most 512 characters",
                code="contract_invalid",
                details={"field": "reason"},
            )
        if not isinstance(self.keep, bool):
            raise ContractError(
                "cleanup keep must be a boolean",
                code="contract_invalid",
                details={"field": "keep"},
            )


@dataclass(frozen=True)
class WorktreeCleanupOutcome:
    """What the sweep did: the paths reclaimed, and whether the checkouts were kept on purpose."""

    removed: tuple[str, ...]
    kept: bool
    reason: str


@dataclass(frozen=True)
class NotifyRequest:
    """A structured notification for the run's event stream (the Run API renders it).

    ``evidence`` is an optional, bounded mapping merged into the ledger record. It exists for the
    few notifications that are a RECORD rather than a message — D5's goal verdict, whose checks an
    auditor must be able to read back — without inventing a second activity for one caller. Bounded
    and secret-screened like every other value that reaches history.
    """

    run_id: str
    task_id: str | None
    kind: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    # WHO this notification is about, when it is about a person's act. Set by the workflow that
    # executed an Update, from the Update's own payload — never from anything the workflow observed
    # outside it, so a replay derives the same value. Empty for notifications nobody caused.
    actor: str = ""

    def validate(self) -> None:
        _require_ref(self.run_id, "run_id")
        if self.task_id is not None:
            _require_ref(self.task_id, "task_id")
        if not isinstance(self.kind, str) or not self.kind.strip() or len(self.kind) > 64:
            raise ContractError(
                "notify kind must be a non-empty string of at most 64 characters",
                code="contract_invalid",
                details={"field": "kind"},
            )
        if not isinstance(self.message, str) or not self.message.strip() or len(self.message) > 4000:
            raise ContractError(
                "notify message must be a non-empty string of at most 4000 characters",
                code="contract_invalid",
                details={"field": "message"},
            )
        if not isinstance(self.evidence, dict):
            raise ContractError(
                "notify evidence must be a mapping",
                code="contract_invalid",
                details={"field": "evidence", "type": type(self.evidence).__name__},
            )
        if not isinstance(self.actor, str) or len(self.actor) > 320:
            # Bounded but OPTIONAL: an empty actor is "nobody" (a scheduled notification), which is a
            # fact, not a gap. What is refused is an unbounded string reaching a ledger row.
            raise ContractError(
                "notify actor must be a string of at most 320 characters",
                code="contract_invalid",
                details={"field": "actor", "length": len(self.actor) if isinstance(self.actor, str) else -1},
            )
        if len(self.evidence) > 16:
            raise ContractError(
                "notify evidence carries more than 16 keys",
                code="contract_invalid",
                details={"keys": len(self.evidence)},
            )
        findings = find_secret_like_strings({"message": self.message, "evidence": self.evidence})
        if findings:
            raise ContractError(
                "a notification may not carry secret-like data",
                code="contract_contains_secret",
                details={"findings": findings},
            )


