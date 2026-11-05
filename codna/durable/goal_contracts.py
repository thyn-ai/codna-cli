"""The CI-gate contracts (D5): what a `watch_ci` call asks for and what it may answer.

Separate from `contracts.py` because that module is at the repository's 1000-line ceiling and this is
a genuinely different boundary: a CI gate reads GITHUB, not the repository. Keeping it here also
keeps `goal.py` (the pure evaluator) free of anything that could reach the network.

**The answer is a closed set.** ``passed`` / ``failed`` / ``pending`` / ``infrastructure`` /
``unknown`` — and only ``passed`` is green. That matters more than it looks: ``unknown`` is what a
missing token or an unreadable rollup produces, and treating it as "not failed" would let a run open
a pull request whose required checks were never read and call the goal met.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from .contract_primitives import ContractError, require_int, require_ref

__all__ = [
    "NAME_RECALL",
    "NAME_WATCH_CI",
    "CIState",
    "RecallOutcome",
    "RecallRequest",
    "RecalledEpisode",
    "WatchCiOutcome",
    "WatchCiRequest",
]

# Registered activity name. Same disjoint `codna.step.` namespace as the other steps — a CI gate is a
# step of the run, not a graph node, so it must not collide with `codna.task.*`.
NAME_WATCH_CI: Final = "codna.step.watch_ci"

# The recall step: reads episode FILES by reference and answers with recaps.
NAME_RECALL: Final = "codna.step.recall"

# An episode file is a small structured record, not a transcript. Bounded on READ as well as on
# write, because a file can be replaced between the run that wrote it and the run that reads it.
MAX_EPISODE_BYTES: Final = 64 * 1024

CIState = Literal["passed", "failed", "pending", "infrastructure", "unknown"]

CI_STATES: Final[tuple[CIState, ...]] = (
    "passed",
    "failed",
    "pending",
    "infrastructure",
    "unknown",
)


@dataclass(frozen=True)
class WatchCiRequest:
    """Read the required checks on one pull request's head, and classify what is red.

    ``repo_dir`` is the SHARED checkout: the origin URL lives there, and a task's worktree inherits
    it. It is the repository, not the tree a turn edits — reading CI touches no tree at all, which
    is why this request carries no `TaskContext`.

    ``head_sha`` is the head the caller BELIEVES it is asking about (the checkpoint it just pushed).
    The activity reports the head it actually read, so "the PR moved under us" is visible in the
    answer rather than silently gating a commit the run did not produce.
    """

    run_id: str
    repo_dir: str
    pr_number: int
    head_sha: str | None = None
    base_branch: str = "main"

    def validate(self) -> None:
        require_ref(self.run_id, "run_id")
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "watch_ci repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )
        require_int(self.pr_number, "pr_number", minimum=1)
        if self.head_sha is not None and not isinstance(self.head_sha, str):
            raise ContractError(
                "watch_ci head_sha must be a string or null",
                code="contract_invalid",
                details={"field": "head_sha"},
            )
        require_ref(self.base_branch, "base_branch")


@dataclass(frozen=True)
class WatchCiOutcome:
    """What CI says, as a reference-sized answer a workflow can carry in history.

    ``checks`` is ``(name, state)`` pairs — no logs, no URLs. The full check payloads stay in the
    ledger (the activity writes them there), because a workflow input/output that grows with a
    repository's check count is a history that eventually breaks.
    """

    state: str
    checks: tuple[tuple[str, str], ...] = ()
    head_sha: str | None = None
    # Why the state is what it is, when the answer is not a plain green/red: a missing token, an
    # unreadable rollup, no pull request at all. Empty for a settled verdict.
    reason: str = ""

    def validate(self) -> None:
        if self.state not in CI_STATES:
            raise ContractError(
                f"watch_ci state must be one of {', '.join(CI_STATES)}",
                code="contract_invalid",
                details={"field": "state", "value": repr(self.state)},
            )
        if not isinstance(self.checks, tuple):
            raise ContractError(
                "watch_ci checks must be a tuple of (name, state) pairs",
                code="contract_invalid",
                details={"field": "checks"},
            )
        for entry in self.checks:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not all(isinstance(part, str) and part for part in entry)
            ):
                raise ContractError(
                    "each watch_ci check must be a (name, state) pair of non-empty strings",
                    code="contract_invalid",
                    details={"entry": repr(entry)},
                )
        if self.head_sha is not None and not isinstance(self.head_sha, str):
            raise ContractError(
                "watch_ci head_sha must be a string or null",
                code="contract_invalid",
                details={"field": "head_sha"},
            )
        if not isinstance(self.reason, str) or len(self.reason) > 512:
            raise ContractError(
                "watch_ci reason must be a string of at most 512 characters",
                code="contract_invalid",
                details={"field": "reason"},
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "checks": [list(entry) for entry in self.checks],
            "head_sha": self.head_sha,
            "reason": self.reason,
        }


# -- recall (D5 goal mode) ---------------------------------------------------------------------
#
# These live in this LEAF module, not in `goal_steps.py` which implements them, for the reason
# `contracts.py` exists at all: a workflow imports the SHAPES, and `goal_steps.py` imports
# `activity_support` (git, subprocess, the sidecar) — importing it from workflow code would break the
# Temporal sandbox. `test_import_hygiene.py` is the guard.


@dataclass(frozen=True)
class RecallRequest:
    """The episodes a plan turn should recall, as references.

    ``refs`` is oldest-first and already bounded by the CALLER's own policy; the activity caps it
    again at ``MAX_EPISODES`` (keeping the NEWEST, which are the ones a plan needs).

    A plain tuple of payload dicts rather than typed objects, because this is what a run carries
    across continue-as-new: the payload shape is explicit in history, and decoding it is a validation
    step rather than a coercion.
    """

    run_id: str
    refs: tuple[dict[str, str], ...] = ()

    def validate(self) -> None:
        require_ref(self.run_id, "run_id")
        if not isinstance(self.refs, tuple):
            raise ContractError(
                "recall refs must be a tuple of episode payloads",
                code="contract_invalid",
                details={"field": "refs", "type": type(self.refs).__name__},
            )
        for ref in self.refs:
            if not isinstance(ref, dict):
                raise ContractError(
                    "every recall ref must be an episode payload",
                    code="contract_invalid",
                    details={"field": "refs", "type": type(ref).__name__},
                )


@dataclass(frozen=True)
class RecalledEpisode:
    """One episode's recap, as the plan-side shape. Small, structured, no transcript."""

    task_id: str
    outcome: str
    brief: str
    verify_passed: bool | None
    review_verdict: str | None
    pr_url: str | None
    plan_ref: str | None
    digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": self.outcome,
            "brief": self.brief,
            "verify_passed": self.verify_passed,
            "review_verdict": self.review_verdict,
            "pr_url": self.pr_url,
            "plan_ref": self.plan_ref,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class RecallOutcome:
    """What the plan turn gets: the recaps it may use, and what could not be read.

    ``missing`` is NOT an error. An episode a sweeping process removed, or one written by an older
    build without the fields this build reads, must not fail a run's next task — the run has already
    paid for that work. It is reported so the ledger can say "task 3 planned without task 1's
    episode" rather than letting a silent gap look like a short history.
    """

    episodes: tuple[RecalledEpisode, ...] = ()
    missing: tuple[str, ...] = ()

    def lines(self) -> str:
        """The recap block, formatted by the SAME pure function the tests use."""
        from .goal import EpisodeRecap, recall_lines

        return recall_lines(
            [
                EpisodeRecap(
                    task_id=episode.task_id,
                    outcome=episode.outcome,
                    brief=episode.brief,
                    verify_passed=episode.verify_passed,
                    review_verdict=episode.review_verdict,
                    pr_url=episode.pr_url,
                    plan_ref=episode.plan_ref,
                )
                for episode in self.episodes
            ]
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "episodes": [episode.to_payload() for episode in self.episodes],
            "missing": list(self.missing),
        }
