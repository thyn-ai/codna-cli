"""Goal mode: a run that stops on EVIDENCE, and recalls EPISODES instead of transcripts (D5).

A task-mode run is "walk this list". A goal-mode run is "do this objective until these things are
true", and those are two different questions at every task boundary:

* **Is the goal met?** ``StopCondition`` says which checks must hold — ``verify ∧ review`` plus an
  OPTIONAL CI gate. ``evaluate_stop`` answers it as a pure function of the evidence recorded at the
  boundaries. No clock, no I/O, no model: a replay must reach the same verdict, and an operator must
  be able to read why the run stopped without trusting a log.
* **What does the next plan know?** The next task's plan turn is handed the OBJECTIVE plus a
  recap of what the previous tasks actually achieved — an EPISODE per task, never a transcript.
  That is the whole token story: a run spanning days must not re-send yesterday's conversation.

**Episodes travel as references.** ``EpisodeRef`` is a path plus a digest (both validated), which is
what lets the run carry one per completed task across `continue_as_new` without growing its input.
The TEXT is re-read by the activity that needs it (``goal_steps.recall``), from the same run
directory the episode was written to — so a retried activity recalls the same bytes, and a plan
cannot be fed a summary the ledger never recorded.

This module is a LEAF: stdlib and `dataclasses` only. `contracts.py` imports it, and the workflow
side imports `contracts`, so anything heavier here would drag I/O into the Temporal sandbox.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, Mapping, Sequence

from .contract_primitives import ContractError, require_int, require_ref
from .state import find_secret_like_strings

__all__ = [
    "CI_STATES",
    "MAX_EPISODES",
    "MAX_OBJECTIVE_CHARS",
    "EpisodeRecap",
    "EpisodeRef",
    "GoalEvidence",
    "GoalSpec",
    "StopCondition",
    "StopDecision",
    "build_ci_fix_brief",
    "ci_verdict",
    "evaluate_stop",
    "format_objective_section",
    "recap_from_payload",
    "recall_lines",
    "required_checks",
]

MAX_OBJECTIVE_CHARS: Final = 4000
# How many episodes a plan turn may be handed. Bounded because the recap is injected into a PROMPT,
# and because a run with a hundred tasks must not build a hundred-episode preamble. The newest win:
# the plan for task N is about what tasks N-1 and N-2 learned, not about task 3.
MAX_EPISODES: Final = 4
# A recap line's free-text field, bounded so one verbose episode cannot dominate the prompt.
MAX_RECAP_CHARS: Final = 400

CI_STATES: Final[tuple[str, ...]] = ("passed", "failed", "pending", "infrastructure", "unknown")

# A red check whose conclusion is one of these did not fail its STEPS: the runner never got that
# far. Naming them is what keeps "fix CI" from being asked to fix a machine.
_INFRA_CONCLUSIONS: Final[frozenset[str]] = frozenset(
    {"TIMED_OUT", "STARTUP_FAILURE", "CANCELLED", "STALE"}
)
# Check names that are fetch/setup work rather than the code under test. The same shape
# `webhook_ci_triage` uses for step names, applied to a CHECK name (which is all the durable path has
# without ``actions: read``). Deliberately narrower than the triage regex: a check called
# "build" or "coverage" is the repo's own gate, not plumbing.
_INFRA_CHECK_NAME = re.compile(
    r"(?i)\b(install|setup|set-up|cache|checkout|download|upload|login|log-in|docker|"
    r"dependencies|deps|provision|toolchain)\b"
)


@dataclass(frozen=True)
class StopCondition:
    """Which checks must hold for the goal to be met. ``verify ∧ review ∧ (ci if required)``.

    Explicit booleans, not a mode string: "verify and review but no CI" and "verify only" are
    different goals, and a mode string would make a new combination a code change.
    """

    verify: bool = True
    review: bool = True
    ci: bool = False

    def validate(self) -> None:
        for name in ("verify", "review", "ci"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(
                    f"stop_condition.{name} must be a boolean",
                    code="contract_invalid",
                    details={"field": f"stop_condition.{name}", "value": repr(getattr(self, name))},
                )
        if not required_checks(self):
            raise ContractError(
                "a stop condition must require at least one check",
                code="contract_invalid",
                details={"field": "stop_condition", "value": self.to_payload()},
            )

    def to_payload(self) -> dict[str, bool]:
        return {"verify": self.verify, "review": self.review, "ci": self.ci}

    @classmethod
    def from_payload(cls, raw: Any) -> "StopCondition":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ContractError(
                "stop_condition must be an object",
                code="contract_invalid",
                details={"field": "stop_condition", "type": type(raw).__name__},
            )
        value = cls(
            verify=bool(raw.get("verify", True)),
            review=bool(raw.get("review", True)),
            ci=bool(raw.get("ci", False)),
        )
        value.validate()
        return value


def required_checks(condition: StopCondition) -> tuple[str, ...]:
    """The check names this condition requires, in a fixed order. Empty means "nothing to prove"."""
    return tuple(
        name
        for name, on in (
            ("verify", condition.verify),
            ("review", condition.review),
            ("ci", condition.ci),
        )
        if on
    )


@dataclass(frozen=True)
class GoalSpec:
    """The objective, the bar for "done", and the two hard stops that bound the search.

    ``max_tasks`` and ``max_ci_attempts`` are BOUNDS, not budgets: the cost cap belongs to the run
    (`RunSpec.max_cost_usd`) and to each task. A goal run that could only be bounded by money would
    sell its whole cap on one task and then stop with nothing to show; these say how much WORK the
    goal is allowed to buy before a human looks at it.
    """

    objective: str
    stop_condition: StopCondition = StopCondition()
    max_tasks: int = 12
    # How many "fix CI" tasks one PR may spend before the run gives up and reports why. Separate from
    # `max_tasks` because a CI loop is a different failure mode: the same PR, the same head, the same
    # red check. Without its own bound a green-looking `max_tasks: 12` would silently authorise
    # twelve fix attempts on one PR.
    max_ci_attempts: int = 3
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError(
                "unsupported goal schema version",
                code="unsupported_contract_schema",
                details={"schema_version": self.schema_version},
            )
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ContractError(
                "a goal needs a non-empty objective",
                code="contract_invalid",
                details={"field": "objective"},
            )
        if len(self.objective) > MAX_OBJECTIVE_CHARS:
            raise ContractError(
                f"objective is over the {MAX_OBJECTIVE_CHARS}-character budget",
                code="contract_invalid",
                details={"length": len(self.objective)},
            )
        findings = find_secret_like_strings({"objective": self.objective})
        if findings:
            raise ContractError(
                "the objective contains secret-like data and may not enter workflow history",
                code="contract_contains_secret",
                details={"findings": findings},
            )
        self.stop_condition.validate()
        require_int(self.max_tasks, "max_tasks", minimum=1)
        require_int(self.max_ci_attempts, "max_ci_attempts", minimum=1)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "stop_condition": self.stop_condition.to_payload(),
            "max_tasks": self.max_tasks,
            "max_ci_attempts": self.max_ci_attempts,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "GoalSpec | None":
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ContractError(
                "goal must be an object",
                code="contract_invalid",
                details={"field": "goal", "type": type(raw).__name__},
            )
        value = cls(
            schema_version=int(raw.get("schema_version", 1)),
            objective=raw.get("objective"),  # type: ignore[arg-type]
            stop_condition=StopCondition.from_payload(raw.get("stop_condition")),
            max_tasks=int(raw.get("max_tasks", 12)),
            max_ci_attempts=int(raw.get("max_ci_attempts", 3)),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class EpisodeRef:
    """One completed task's episode, as a REFERENCE: where it is and what it hashed to.

    A run carries one of these per completed task across `continue_as_new`, which is why it is two
    short strings and not a summary. The digest is what makes a later recall auditable: the plan for
    task 3 names the episodes it read, by hash.
    """

    path: str
    digest: str
    task_id: str

    def validate(self) -> None:
        require_ref(self.path, "episode.path")
        require_ref(self.task_id, "episode.task_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.digest or ""):
            raise ContractError(
                "episode.digest must be a sha256 hex digest",
                code="contract_invalid",
                details={"digest": self.digest},
            )

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {"path": self.path, "digest": self.digest, "task_id": self.task_id}

    @classmethod
    def from_payload(cls, raw: Any) -> "EpisodeRef":
        if not isinstance(raw, Mapping):
            raise ContractError(
                "an episode reference must be an object",
                code="contract_invalid",
                details={"type": type(raw).__name__},
            )
        value = cls(
            path=str(raw.get("path") or ""),
            digest=str(raw.get("digest") or ""),
            task_id=str(raw.get("task_id") or ""),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class EpisodeRecap:
    """What a completed task actually achieved — the plan-side view of an episode.

    Deliberately STRUCTURED rather than a paragraph: every field is a fact the ledger also holds, so
    a plan that says "task 1 was rejected in review" can be checked against the record. The brief is
    an excerpt for orientation; everything else is a verdict or a reference.
    """

    task_id: str
    outcome: str
    brief: str
    verify_passed: bool | None
    review_verdict: str | None
    pr_url: str | None
    plan_ref: str | None

    def line(self) -> str:
        parts = [f"- {self.task_id} ({self.outcome}): {self.brief or '(no brief)'}"]
        if self.verify_passed is not None:
            parts.append(f"verify={'pass' if self.verify_passed else 'FAIL'}")
        if self.review_verdict:
            parts.append(f"review={self.review_verdict}")
        if self.plan_ref:
            parts.append(f"plan={self.plan_ref}")
        if self.pr_url:
            parts.append(f"pr={self.pr_url}")
        return "; ".join(parts)


def recap_from_payload(raw: Any) -> EpisodeRecap:
    """Decode one episode FILE's body into a recap. Pure: the same bytes give the same recap."""
    if not isinstance(raw, Mapping):
        raise ContractError(
            "an episode file must decode to an object",
            code="contract_invalid",
            details={"type": type(raw).__name__},
        )
    verify_ref = raw.get("verify_ref")
    verify_passed = None
    if isinstance(verify_ref, Mapping) and isinstance(verify_ref.get("passed"), bool):
        verify_passed = bool(verify_ref["passed"])
    review_verdict = raw.get("review_verdict")
    pr = raw.get("pr")
    brief = str(raw.get("brief") or "").strip().splitlines()
    return EpisodeRecap(
        task_id=str(raw.get("task_id") or ""),
        outcome=str(raw.get("outcome") or ""),
        brief=(brief[0] if brief else "")[:MAX_RECAP_CHARS],
        verify_passed=verify_passed,
        review_verdict=str(review_verdict) if isinstance(review_verdict, str) else None,
        pr_url=str(pr.get("url")) if isinstance(pr, Mapping) and pr.get("url") else None,
        plan_ref=str(raw["plan_ref"]) if isinstance(raw.get("plan_ref"), str) else None,
    )


def recall_lines(recaps: Sequence[EpisodeRecap]) -> str:
    """The recap block a plan prompt carries. Empty string when there is nothing to recall.

    Oldest first, so a plan reads the run's history in the order it happened; bounded to
    ``MAX_EPISODES`` by the CALLER (the newest are the relevant ones), which keeps this pure.
    """
    if not recaps:
        return ""
    return "\n".join(recap.line() for recap in recaps)


def format_objective_section(objective: str, recaps: Sequence[EpisodeRecap] = ()) -> str:
    """The goal context appended to a task's plan prompt: the objective, plus what is already done.

    Deterministic in its inputs and appended rather than woven in, for the same reason
    `contracts.plan_prompt` appends steering: a recorded prompt must replay byte for byte.
    """
    lines = ["", "Run objective (the goal this task serves):", objective.strip()]
    recalled = recall_lines(recaps)
    if recalled:
        lines.extend(
            [
                "",
                "Already completed in this run (episodes, not transcripts — do not repeat this work):",
                recalled,
            ]
        )
    return "\n".join(lines)


# -- the stop condition ------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalEvidence:
    """What one completed task proved, as far as the stop condition is concerned.

    Every field is optional because "the check did not run" and "the check failed" are different
    answers, and collapsing them is how a goal run would stop on work nobody verified.
    """

    task_id: str
    outcome: str
    verify_passed: bool | None = None
    review_verdict: str | None = None


@dataclass(frozen=True)
class StopDecision:
    """The verdict plus a reason an operator can act on. ``checks`` is ordered by check name."""

    satisfied: bool
    reason: str
    checks: tuple[tuple[str, str], ...] = ()

    def summary(self) -> str:
        return "; ".join(f"{name}={state}" for name, state in self.checks)


def evaluate_stop(
    condition: StopCondition,
    evidence: Sequence[GoalEvidence],
    *,
    ci_state: str | None = None,
) -> StopDecision:
    """``condition`` evaluated over every boundary. PURE and total.

    The rule is ``all``, not ``last``: a run whose task 1 failed verification has not met a goal that
    requires verification, even if task 2 passed. "The goal" is the run's whole body of work, and a
    stop condition that only looked at the latest task would let a failing task be forgotten by
    moving on.

    ``ci_state`` is the CI gate's verdict for the run's CURRENT head — one value, not one per task,
    because a pull request has one head at a time. When the condition requires CI, the gate is
    satisfied only by ``"passed"``: ``pending`` is not green, and neither ``unknown`` ("we could not
    read the rollup") nor ``infrastructure`` ("a runner died") is a pass.
    """
    condition.validate()
    if not evidence:
        return StopDecision(
            satisfied=False,
            reason="no_evidence",
            checks=tuple((name, "not_run") for name in required_checks(condition)),
        )
    verified = all(item.verify_passed is True for item in evidence)
    reviewed = all(item.review_verdict == "approved" for item in evidence)
    checks: list[tuple[str, str]] = []
    if condition.verify:
        checks.append(("verify", "passed" if verified else "failed"))
    if condition.review:
        checks.append(("review", "approved" if reviewed else "failed"))
    if condition.ci:
        checks.append(("ci", "passed" if ci_state == "passed" else "failed"))
    satisfied = all(state in ("passed", "approved") for _name, state in checks)
    return StopDecision(
        satisfied=satisfied,
        reason="goal_satisfied" if satisfied else "goal_unmet",
        checks=tuple(checks),
    )


# -- the CI gate -------------------------------------------------------------------------------


def ci_verdict(red: Sequence[Mapping[str, Any]]) -> str:
    """Classify a red check list: ``passed`` | ``failed`` | ``infrastructure``.

    ``failing_required_checks`` returns the REQUIRED checks that are red on the head, or None when it
    cannot know. This maps that answer onto the three states the run loop acts on, and it is
    deliberately unforgiving: a runner-level conclusion or a plumbing-shaped check name is
    ``infrastructure`` (no fix task — asking an agent to fix a machine burns a task and teaches it to
    patch tests), everything else is ``failed`` (code).

    An EMPTY list is ``passed``: no required check is failing. ``None`` never reaches here — the
    caller maps "unknown" itself, because "we could not read CI" must not be spelled the same way as
    "CI is green".
    """
    if not red:
        return "passed"
    for check in red:
        state = str(check.get("state") or "").upper()
        name = str(check.get("name") or "")
        if state in _INFRA_CONCLUSIONS or _INFRA_CHECK_NAME.search(name):
            return "infrastructure"
    return "failed"


def build_ci_fix_brief(
    *,
    pr_number: int,
    head_sha: str | None,
    checks: Sequence[Any],
    attempt: int,
) -> str:
    """The "fix CI" task's brief: WHICH required checks are red, on which head, and what was tried.

    Built from the checks themselves rather than from a model's summary of them, so the brief names
    what CI actually reported. A check may arrive as a mapping (the GitHub rollup shape) or as a
    ``(name, state)`` pair (the `WatchCiOutcome` shape the workflow actually carries); both are
    accepted because refusing one would make the same fact unrepresentable on one side of the
    activity boundary.

    ``attempt`` is included because a second attempt that repeats the first one's brief is a second
    attempt at the same fix — the loop would look bounded and behave as if it were not.
    """
    names = ", ".join(
        f"`{_check_name(check)}` ({_check_state(check)})" for check in checks
    )
    where = f"head {head_sha[:8]}" if head_sha else "the current head"
    brief = [
        f"Fix CI on pull request #{pr_number}: {where} has failing required check(s): {names}.",
        "",
        "Reproduce the failure locally, fix the cause (not the check), and keep the existing tests "
        "green. Do not disable, skip or loosen a check to make it pass.",
    ]
    if attempt > 1:
        brief.extend(
            [
                "",
                f"This is attempt {attempt}. A previous fix did not turn CI green, so verify the "
                "actual failure output before changing anything.",
            ]
        )
    return "\n".join(brief)


def _check_name(check: Any) -> str:
    if isinstance(check, Mapping):
        return str(check.get("name") or "?")
    if isinstance(check, (tuple, list)) and len(check) == 2:
        return str(check[0] or "?")
    return "?"


def _check_state(check: Any) -> str:
    if isinstance(check, Mapping):
        return str(check.get("state") or "red").lower()
    if isinstance(check, (tuple, list)) and len(check) == 2:
        return str(check[1] or "red").lower()
    return "red"


def _json_loads(text: str) -> Any:
    """Strict JSON decode used by `goal_steps.recall`. Here so the parse rule lives with the shape."""
    return json.loads(text)


_ = (require_int, _json_loads, Literal)
