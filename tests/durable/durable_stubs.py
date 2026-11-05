"""Stub activities and their scripted state, in a WORKFLOW-SAFE module.

Deliberately not a test module. Temporal re-imports a workflow's defining module inside the sandbox,
and importing `pytest` reaches `urllib.request` — so anything the workflow sandbox touches must avoid
it. This file is imported by `test_task_workflow_time_skipping.py` (which does import pytest) but is
never itself re-imported by the sandbox: the workflow classes live in `codna.durable.workflows`, and
this module is only reached from the ACTIVITY side.

The stubs are intentionally thin, but they obey the real contracts:

* the states they return are `TaskState` payloads, produced through `TaskState` so an invalid one
  fails here rather than in the workflow;
* `run_cline_turn` advances ``turn_seq`` and reports a cost delta, exactly like the real one;
* `checkpoint_turn` reports the ``turn_seq`` it was called for, so the test can assert "one per turn"
  without inferring it.

The scripted failures (`verify_failures`, `review_rejections`) are counters rather than closures: the
plugin derives an activity's identity from the function, and a stub that needs to remember something
between calls has to keep it in module state anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

from temporalio import activity

from codna.durable.contracts import (
    NAME_CHECKPOINT,
    NAME_EPISODE,
    NAME_NOTIFY,
    NAME_OPEN_PR,
    NAME_PLAN,
    NAME_REVIEW,
    NAME_TURN,
    NAME_VERIFY,
    ActivityOutcome,
    CheckpointOutcome,
    CheckpointRequest,
    EpisodeOutcome,
    EpisodeRequest,
    NotifyOutcome,
    NotifyRequest,
    PlanRequest,
    PullRequestOutcome,
    PullRequestRequest,
    ReviewRequest,
    TurnRequest,
    VerifyRequest,
)
from codna.durable.state import CheckpointRef, PullRequestRef, TaskState, VerifyRef

STUB_TASK_QUEUE: Final = "codna-durable-stubs"
_VERIFY_DIGEST: Final = "a" * 64


@dataclass
class _Script:
    """What the stubs should do next. Reset per test; module state because activities are functions."""

    verify_failures: int = 0
    review_rejections: int = 0
    turn_cost_usd: float = 0.01
    turn_tokens: int = 10
    duplicate_checkpoint_once: bool = False


_STATE = _Script()
_CALLS: list[str] = []
_CHECKPOINT_SEQS: list[int] = []
_SEEN_TURN_SEQS: set[int] = set()


def reset_stub_state(
    *,
    verify_failures: int = 0,
    review_rejections: int = 0,
    turn_cost_usd: float = 0.01,
    turn_tokens: int = 10,
    duplicate_checkpoint_once: bool = False,
) -> None:
    """Clear the call log and set what the next run's stubs should do."""
    global _STATE
    _STATE = _Script(
        verify_failures=verify_failures,
        review_rejections=review_rejections,
        turn_cost_usd=turn_cost_usd,
        turn_tokens=turn_tokens,
        duplicate_checkpoint_once=duplicate_checkpoint_once,
    )
    _CALLS.clear()
    _CHECKPOINT_SEQS.clear()
    _SEEN_TURN_SEQS.clear()


class StubLog:
    """Read access to what the stubs were called with, in order."""

    @staticmethod
    def calls() -> list[str]:
        return list(_CALLS)

    @staticmethod
    def checkpoint_turn_seqs() -> list[int]:
        return list(_CHECKPOINT_SEQS)

    @staticmethod
    def reset() -> None:
        _CALLS.clear()
        _CHECKPOINT_SEQS.clear()
        _SEEN_TURN_SEQS.clear()


def _record(name: str) -> None:
    _CALLS.append(name)


def _state(payload: dict[str, Any]) -> TaskState:
    return TaskState.from_payload(payload)


def _checkpoint_ref(task_id: str, turn_seq: int) -> CheckpointRef:
    return CheckpointRef(ref=f"refs/checkpoints/{task_id}-{turn_seq}", sha=f"{turn_seq:040x}"[-40:])


@activity.defn(name=NAME_PLAN)
async def stub_plan(request: PlanRequest) -> ActivityOutcome:
    """Seconds instead of a paid turn: a plan reference, derived from the task id."""
    _record("plan_task")
    state = _state(request.state)
    updated = replace(
        state,
        plan_ref=f"refs/plans/{state.task_id}",
        attempts=state.attempts.incremented("plan"),
    )
    return ActivityOutcome(state=updated.to_payload(), details={"plan_ref": updated.plan_ref})


@activity.defn(name=NAME_TURN)
async def stub_turn(request: TurnRequest) -> ActivityOutcome:
    """Advance ``turn_seq`` and report this turn's spend, exactly like the real activity."""
    _record("run_cline_turn")
    state = _state(request.state)
    updated = state.with_events(state.last_event_index + 1).started_turn(
        session_id=state.session_id or f"session-{request.context.task_id}",
        turn_id=f"turn-{request.context.task_id}-{state.turn_seq + 1}",
    )
    updated = replace(updated, attempts=updated.attempts.incremented("implement"))
    updated = updated.with_spend(cost_usd=_STATE.turn_cost_usd, tokens=_STATE.turn_tokens)
    return ActivityOutcome(
        state=updated.to_payload(),
        cost_usd=_STATE.turn_cost_usd,
        tokens=_STATE.turn_tokens,
        details={"turn_seq": updated.turn_seq},
    )


@activity.defn(name=NAME_CHECKPOINT)
async def stub_checkpoint(request: CheckpointRequest) -> CheckpointOutcome:
    """Report the turn it was called for, so "one per turn_seq" is directly assertable."""
    _record("checkpoint_turn")
    state = _state(request.state)
    _CHECKPOINT_SEQS.append(state.turn_seq)
    ref = _checkpoint_ref(state.task_id, state.turn_seq)
    if state.turn_seq in _SEEN_TURN_SEQS:
        # The retry path: the same turn checkpointed again. A real checkpoint would find its commit;
        # the stub reports the same ref, which is what makes the assertion about the LEDGER hold.
        return CheckpointOutcome(
            state=state.to_payload(),
            ref=state.checkpoint_ref.ref if state.checkpoint_ref else ref.ref,
            sha=state.checkpoint_ref.sha if state.checkpoint_ref else ref.sha,
            created=False,
        )
    _SEEN_TURN_SEQS.add(state.turn_seq)
    updated = replace(state, checkpoint_ref=ref)
    return CheckpointOutcome(state=updated.to_payload(), ref=ref.ref, sha=ref.sha, created=True)


@activity.defn(name=NAME_VERIFY)
async def stub_verify(request: VerifyRequest) -> ActivityOutcome:
    """Fail the first N verifications, then pass. A failure is a RESULT, not an exception."""
    _record("verify_task")
    state = _state(request.state)
    failing = _STATE.verify_failures > 0
    if failing:
        _STATE.verify_failures -= 1
    ref = VerifyRef(
        ref=f"verify/{state.task_id}/{state.attempts.verify + 1}",
        passed=not failing,
        command_digest=_VERIFY_DIGEST,
        log_path=None,
    )
    updated = replace(state, verify_ref=ref, attempts=state.attempts.incremented("verify"))
    return ActivityOutcome(state=updated.to_payload(), details={"passed": not failing})


@activity.defn(name=NAME_REVIEW)
async def stub_review(request: ReviewRequest) -> ActivityOutcome:
    """Reject the first N reviews, then approve."""
    _record("review_task")
    state = _state(request.state)
    rejecting = _STATE.review_rejections > 0
    if rejecting:
        _STATE.review_rejections -= 1
    updated = replace(
        state,
        review_verdict="rejected" if rejecting else "approved",
        attempts=state.attempts.incremented("review"),
    )
    return ActivityOutcome(state=updated.to_payload(), details={"verdict": updated.review_verdict})


@activity.defn(name=NAME_OPEN_PR)
async def stub_open_pr(request: PullRequestRequest) -> PullRequestOutcome:
    _record("open_pr")
    state = _state(request.state)
    pr = PullRequestRef(
        number=1,
        url=f"https://github.com/thyn-ai/codna/pull/{1}",
        branch=request.context.branch,
        head_sha=state.checkpoint_ref.sha if state.checkpoint_ref else None,
    )
    updated = replace(state, pr=pr, attempts=state.attempts.incremented("open_pr"))
    return PullRequestOutcome(state=updated.to_payload(), details={"number": pr.number})


@activity.defn(name=NAME_EPISODE)
async def stub_remember_episode(request: EpisodeRequest) -> EpisodeOutcome:
    _record("remember_episode")
    return EpisodeOutcome(path="episodes/task-1.json", digest="b" * 64, ledger_seq=1)


@activity.defn(name=NAME_NOTIFY)
async def stub_notify(request: NotifyRequest) -> NotifyOutcome:
    _record("notify")
    return NotifyOutcome(seq=1, kind=request.kind)


STUB_ACTIVITIES: Final[tuple[Any, ...]] = (
    stub_plan,
    stub_turn,
    stub_checkpoint,
    stub_verify,
    stub_review,
    stub_open_pr,
    stub_remember_episode,
    stub_notify,
)
