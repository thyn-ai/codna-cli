"""Pure conditional edges for the task graph.

These three functions are the *whole* control flow of a task: which node runs next, given only
`TaskState`. They are pure — no clock, no I/O, no mutation, no environment — for three reasons
that all matter to a durable run:

* **Replay.** Temporal replays a workflow by re-running the code. Any impurity here becomes a
  non-determinism error on replay, which surfaces as a stuck workflow in production, not in CI.
* **Parity.** The plain (no-LangGraph) workflow and the LangGraph graph must take the *same*
  route for the same state; sharing one decision function is what makes "the same brief completes
  under both engines" a tested fact rather than an aspiration. `graph.py` wires these into
  conditional edges and `workflows.py` calls them directly.
* **Testability.** The failure modes worth naming — a verify that never ran, a review verdict we
  do not recognize, an exhausted budget, a blown attempt ceiling — are all decidable here, with
  no Temporal server, and are exhaustive below.

Route vocabulary (the nodes are `plan`, `implement`, `verify`, `review`, `open_pr`,
`wait_for_human`, plus the terminal `END`):

* `after_plan`   → `implement` once a plan exists, else re-`plan` while attempts remain, else park.
* `after_verify` → `review` on a pass, `implement` on a failure, park when retries are spent.
* `after_review` → `open_pr` on approval, `implement` on rejection, park when retries are spent.

Every function can also return `wait_for_human`: the run is out of budget or attempts, and a human
must raise the cap or take over. Parking is an outcome, never an exception — a durable run that
throws is a run that lost its history.
"""
from __future__ import annotations

from typing import Final, Literal

from .state import TaskState

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "Route",
    "after_plan",
    "after_review",
    "after_verify",
    "graph_route",
]

Route = Literal[
    "plan",
    "implement",
    "verify",
    "review",
    "open_pr",
    "wait_for_human",
]

END: Final = "end"

DEFAULT_MAX_ATTEMPTS: Final = 3


def _retries_left(state: TaskState, stage: Literal["plan", "implement", "verify", "review", "open_pr"], cap: int) -> bool:
    return getattr(state.attempts, stage) < cap


def after_plan(state: TaskState, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Route:
    """Plan produced a reference → implement. No plan yet → re-plan, bounded. Out of tries → park.

    A plan that "succeeded" without a `plan_ref` is not a plan: the state must point at something a
    later activity can read, so the missing reference is treated as a failed attempt rather than
    as success with no evidence.
    """
    # D4: a turn that stopped on an ASK produced no plan because it was waiting, not because it failed.
    # Re-planning would spend another paid turn only to be blocked on the same call, so the task parks
    # instead — and parking is what puts a human in front of the question.
    if state.ask is not None:
        return "wait_for_human"
    if state.budget.exhausted:
        return "wait_for_human"
    if state.plan_ref is not None:
        return "implement"
    if _retries_left(state, "plan", max_attempts):
        return "plan"
    return "wait_for_human"


def after_verify(state: TaskState, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Route:
    """Verify passed → review. Failed → implement again, bounded. Exhausted → park.

    A missing `verify_ref` is *not* a pass. Verification is the gate that stops an unproven change
    from being reviewed and opened, so its absence is treated as a failure to verify rather than
    optimistically proceeding.
    """
    if state.ask is not None:
        # The implement turn parked on a question: nothing was written, so there is nothing to verify.
        return "wait_for_human"
    if state.budget.exhausted:
        return "wait_for_human"
    if state.verify_ref is None or not state.verify_ref.passed:
        if _retries_left(state, "implement", max_attempts):
            return "implement"
        return "wait_for_human"
    return "review"


def after_review(state: TaskState, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Route:
    """Review approved → open the PR. Rejected → implement again, bounded. Exhausted → park.

    An unrecognized (or absent) verdict is a rejection, never an approval: `review_verdict` is
    written by a model, and the one wrong default that matters is "open a pull request because the
    reviewer returned something we did not understand".
    """
    if state.ask is not None:
        # A review turn that asked a question has no verdict. Treating that as "rejected" would send
        # it back to implement, which is the re-plan trap `after_plan` already refuses.
        return "wait_for_human"
    if state.budget.exhausted:
        return "wait_for_human"
    if state.review_verdict == "approved":
        return "open_pr"
    if _retries_left(state, "implement", max_attempts):
        return "implement"
    return "wait_for_human"


_ROUTERS = {
    "after_plan": after_plan,
    "after_verify": after_verify,
    "after_review": after_review,
}


def graph_route(edge: str, state: TaskState, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Route:
    """Dispatch a named edge, for the code paths that cannot hold a direct function reference.

    The plain workflow keeps its own explicit `if` chain (it must be readable as the reference
    semantics), so this exists for the generic/dynamic caller — the parity test and any future
    declarative wiring — and refuses an unknown edge name rather than defaulting to one.
    """
    try:
        router = _ROUTERS[edge]
    except KeyError as exc:
        raise KeyError(f"unknown routing edge {edge!r}; known edges: {sorted(_ROUTERS)}") from exc
    return router(state, max_attempts=max_attempts)
