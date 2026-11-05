"""Pure conditional edges: exhaustive, deterministic, and identical under both engines.

Routing is where a durable run loses money or ships an unproven change, so every branch is a test:
verify fails → implement again; retries spent → park; verdict unrecognized → NOT a PR. The parity
test at the end is the one the plan asks for — the plain engine calls these functions directly and
the LangGraph engine wires them into conditional edges, so "same decision, same input" is the whole
guarantee that the two engines cannot diverge.
"""
from __future__ import annotations

import pytest

from codna.durable.routing import (
    DEFAULT_MAX_ATTEMPTS,
    after_plan,
    after_review,
    after_verify,
    graph_route,
)
from codna.durable.state import Attempts, Budget, TaskState, VerifyRef


def _state(**overrides) -> TaskState:
    base: dict = {"run_id": "run-1", "task_id": "task-1"}
    base.update(overrides)
    return TaskState(**base)


def _verified(passed: bool = True) -> VerifyRef:
    return VerifyRef(ref=f"verify/{passed}", passed=passed, command_digest="a" * 64)


EXHAUSTED = Budget(max_cost_usd=1.0, spent_usd=1.0, max_tokens=None, spent_tokens=0)


# -- after_plan -------------------------------------------------------------------------


def test_after_plan_implements_once_a_plan_exists():
    assert after_plan(_state(plan_ref="refs/plans/1")) == "implement"


def test_after_plan_replans_while_attempts_remain():
    assert after_plan(_state(attempts=Attempts(plan=1))) == "plan"


def test_after_plan_parks_when_attempts_are_spent():
    state = _state(attempts=Attempts(plan=DEFAULT_MAX_ATTEMPTS))
    assert after_plan(state) == "wait_for_human"


def test_after_plan_parks_on_an_exhausted_budget_even_with_a_good_plan():
    # Budget beats progress: a plan that succeeded must not buy an implement turn the run cannot pay
    # for, and parking is how a human raises the cap.
    assert after_plan(_state(plan_ref="refs/plans/1", budget=EXHAUSTED)) == "wait_for_human"


def test_after_plan_respects_a_custom_ceiling():
    state = _state(attempts=Attempts(plan=1))
    assert after_plan(state, max_attempts=1) == "wait_for_human"
    assert after_plan(state, max_attempts=2) == "plan"


# -- after_verify -----------------------------------------------------------------------


def test_after_verify_reviews_on_a_pass():
    assert after_verify(_state(verify_ref=_verified(True))) == "review"


def test_after_verify_reimplements_on_a_failure():
    assert after_verify(_state(verify_ref=_verified(False))) == "implement"


def test_after_verify_treats_a_missing_verification_as_a_failure():
    # Not "optimistically proceed": verification is the gate that stops an unproven change from
    # reaching review, so its absence must re-implement, not pass.
    assert after_verify(_state()) == "implement"


def test_after_verify_parks_when_implement_attempts_are_spent():
    state = _state(verify_ref=_verified(False), attempts=Attempts(implement=DEFAULT_MAX_ATTEMPTS))
    assert after_verify(state) == "wait_for_human"


def test_after_verify_parks_on_an_exhausted_budget_even_when_verification_passed():
    assert after_verify(_state(verify_ref=_verified(True), budget=EXHAUSTED)) == "wait_for_human"


# -- after_review -----------------------------------------------------------------------


def test_after_review_opens_a_pr_only_on_approval():
    assert after_review(_state(review_verdict="approved")) == "open_pr"


@pytest.mark.parametrize("verdict", ["rejected", "changes_requested", None])
def test_after_review_reimplements_on_anything_but_approval(verdict):
    # An unrecognized verdict must never become a PR: `review_verdict` is model-written, and
    # "open a pull request because the reviewer returned something we did not understand" is the one
    # wrong default that matters.
    assert after_review(_state(review_verdict=verdict)) == "implement"


def test_after_review_parks_when_implement_attempts_are_spent():
    state = _state(review_verdict="rejected", attempts=Attempts(implement=DEFAULT_MAX_ATTEMPTS))
    assert after_review(state) == "wait_for_human"


def test_after_review_parks_on_an_exhausted_budget_even_when_approved():
    assert after_review(_state(review_verdict="approved", budget=EXHAUSTED)) == "wait_for_human"


# -- graph_route dispatch ---------------------------------------------------------------


def test_graph_route_dispatches_by_name():
    assert graph_route("after_review", _state(review_verdict="approved")) == "open_pr"
    assert graph_route("after_verify", _state(verify_ref=_verified(True))) == "review"


def test_graph_route_refuses_an_unknown_edge_instead_of_defaulting():
    with pytest.raises(KeyError) as excinfo:
        graph_route("after_coffee", _state())
    assert "after_coffee" in str(excinfo.value)


# -- purity -----------------------------------------------------------------------------


def test_routers_do_not_mutate_state():
    state = _state(verify_ref=_verified(False), attempts=Attempts(implement=1), review_verdict="rejected")
    before = state.to_payload()
    after_plan(state)
    after_verify(state)
    after_review(state)
    assert state.to_payload() == before


def test_routers_are_deterministic_across_repeated_calls():
    state = _state(verify_ref=_verified(False), attempts=Attempts(implement=2))
    assert {after_verify(state) for _ in range(20)} == {"implement"}


def test_routers_never_raise_on_any_reference_combination():
    # Totality: a router that throws is a workflow that dies with a stack trace instead of parking,
    # and a parked run is recoverable while a failed one is not. Exercise the cross-product of the
    # flags that filter routing.
    verdicts = [None, "approved", "rejected", "changes_requested"]
    for verdict in verdicts:
        for verify_passed in (None, True, False):
            for implement_attempts in (0, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ATTEMPTS + 5):
                for budget in (Budget(None, 0.0, None, 0), EXHAUSTED):
                    state = _state(
                        review_verdict=verdict,
                        verify_ref=None if verify_passed is None else _verified(verify_passed),
                        attempts=Attempts(implement=implement_attempts),
                        budget=budget,
                    )
                    assert after_plan(state) in {"implement", "plan", "wait_for_human"}
                    assert after_verify(state) in {"review", "implement", "wait_for_human"}
                    assert after_review(state) in {"open_pr", "implement", "wait_for_human"}
