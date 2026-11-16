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
    "DelegationPolicyOutcome",
    "DelegationPolicyRequest",
    "EpisodeRequest",
    "McpAttachmentOutcome",
    "McpAttachmentRequest",
    "NotifyRequest",
    "WorktreeCleanupOutcome",
    "WorktreeCleanupRequest",
]


@dataclass(frozen=True)
class McpAttachmentRequest:
    """Ask which MCP servers a durable turn is EFFECTIVELY handed, given this worker's environment.

    The DECLARED servers travel in the payload (the run's own tasks' union, disabled entries removed) so
    the activity needs nothing else from the spec. What it ADDS is the half a workflow cannot see: the
    worker-wide ``CODNA_MCP_SERVERS`` default and Codna's built-in browser server, which
    `mcp.session_mcp_servers` appends when ``CODNA_BROWSER_TOOL=1``.

    An activity rather than workflow code because the answer reads ``os.environ``, which the workflow
    sandbox refuses — the same reason the delegation-policy read is one. The answer also lands in
    history, so a replay derives the note instead of re-reading an environment that may have changed.
    """

    run_id: str
    servers: tuple[dict[str, Any], ...] = ()

    def validate(self) -> None:
        _require_ref(self.run_id, "run_id")
        if not isinstance(self.servers, tuple):
            raise ContractError(
                "mcp attachment servers must be a tuple",
                code="contract_invalid",
                details={"field": "servers"},
            )
        for server in self.servers:
            if not isinstance(server, dict) or not server.get("name"):
                raise ContractError(
                    "mcp attachment servers must each be a named mapping",
                    code="contract_invalid",
                    details={"field": "servers", "server": repr(server)[:120]},
                )


@dataclass(frozen=True)
class McpAttachmentOutcome:
    """What was effectively attached: ON with the server names, or OFF with the reason and a code.

    ``error_code`` is non-empty ONLY when the configuration itself was at fault (an unset ``from_env``
    variable, an invalid browser switch). It is what lets a surface distinguish "we attach nothing" from
    "we were asked to attach something and the setting is broken" without matching on prose.
    """

    attached: bool
    servers: tuple[str, ...] = ()
    reason: str = ""
    error_code: str = ""


@dataclass(frozen=True)
class DelegationPolicyRequest:
    """Ask what this run's repository POLICY removes from the delegation toolset.

    ``enable_agent_teams`` is a GATE, not a GRANT, and the grant is narrowed twice: by the approval
    profile's ``allowedTools`` and again by ``.codna/policy.yaml`` at session create. A policy written
    for a single-agent run leaves the profile intact and still strips every ``team_*`` name, which is
    how a run reported ``delegation is attached`` and then failed every ``team_run_task`` call with
    ``unavailable tool``.

    IT IS AN ACTIVITY, NOT WORKFLOW CODE, and that is not a style choice: the answer requires reading
    and parsing a file, and Temporal's workflow sandbox REFUSES filesystem access from workflow code
    (``RestrictedWorkflowAccessError: Cannot access os.path from inside a workflow`` — measured by
    ``test_swarm_e2e.py`` when this logic was first written inline). The read belongs in an activity
    for the same reason every other read does.
    """

    run_id: str
    repo_dir: str
    enable_agent_teams: bool

    def validate(self) -> None:
        _require_ref(self.run_id, "run_id")
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "delegation policy repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )
        if not isinstance(self.enable_agent_teams, bool):
            raise ContractError(
                "delegation policy enable_agent_teams must be a boolean",
                code="contract_invalid",
                details={"field": "enable_agent_teams"},
            )


@dataclass(frozen=True)
class DelegationPolicyOutcome:
    """Which prompt-named team tools the policy strips, and the file that decided it.

    ``stripped`` is empty for every run that delegates normally, so the caller reports nothing in the
    ordinary case. ``policy_path`` is carried so the note names the file an operator has to edit —
    the whole point of the report is to say WHERE the fix is, not merely that delegation is off.
    """

    stripped: tuple[str, ...]
    policy_path: str


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
    # WHICH CHECKOUT this run works in, carried by the run's opening notification so ``GET /runs`` can
    # say which repository a run belongs to. Without it every row is labelled with whatever folder the
    # window happens to have open, so a run from repo A is listed under repo B the moment the IDE is
    # reopened elsewhere.
    #
    # It rides on an EXISTING activity call rather than a new one on purpose: a run's history replays
    # its COMMAND sequence, and a second bookkeeping activity would make every recorded history
    # diverge — the one kind of change replay cannot absorb. The value is the run spec's own
    # ``repo_dir``, which `RunSpec.validate` has already required to be absolute; a replay therefore
    # derives the same string from the same input.
    #
    # ``""`` means "this notification is not about a checkout", which is every notification except the
    # opening one — so the field costs nothing on the rows that cannot use it.
    repo_dir: str = ""

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
        if not isinstance(self.repo_dir, str):
            raise ContractError(
                "notify repo_dir must be a string",
                code="contract_invalid",
                details={"field": "repo_dir", "type": type(self.repo_dir).__name__},
            )
        if self.repo_dir and not self.repo_dir.startswith("/"):
            # The same rule `RunSpec.repo_dir` is held to, restated here because this value is what
            # `GET /runs` reports: a relative path would be resolved against whatever directory the
            # reader happened to be in, which is a different run than the one recorded.
            raise ContractError(
                "notify repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )


