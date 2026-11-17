"""An approval records WHY the task parked, on both engines.

── THE MEASURED DEFECT ────────────────────────────────────────────────────────────────────────────

`run-20261113T225225Z` has three ledger `verify` records with ``passed: false`` and
``returncode: 127``, and then an approval. The graph engine recorded that approval with

    "reason": "open_pr_ready"

because `_await_decision` hardcoded exactly that string for every non-budget park. The human gate
therefore reported the work as ready for a pull request on a task whose verification had never once
passed. On the plain engine the same ledger record says `verify_failed`, because it reads
`park_reason` — so the two engines disagreed about the most important fact in the audit trail.

These tests pin the shared vocabulary and the graph engine's use of it. They do not test the approval
GATE (which is unchanged and still refuses an anonymous decision, an ill-timed one, and a `False`).
"""
from __future__ import annotations

import pytest

from codna.durable.parking import PARK_REASONS, park_kind, park_reason
from codna.durable.state import TaskState

PLAN = "refs/plans/1"
CHECKPOINT = {"ref": "refs/codna/runs/run-1/turn-1", "sha": "b" * 40}


def _state(**overrides):
    payload: dict = {
        "schema_version": 1,
        "run_id": "run-1",
        "task_id": "task-1",
        "turn_seq": 1,
        "session_id": "session-1",
        "last_turn_key": "run-1:task-1:1",
        "last_turn_id": "turn-1",
        "last_event_index": 10,
        "budget": {"max_cost_usd": None, "spent_usd": 0.0, "max_tokens": None, "spent_tokens": 0},
    }
    payload.update(overrides)
    return TaskState.from_payload(payload)


def _failed_verify() -> dict:
    return {"ref": "verify/task-1/abc", "passed": False, "command_digest": "c" * 64}


def _passed_verify() -> dict:
    return {"ref": "verify/task-1/abc", "passed": True, "command_digest": "c" * 64}


def test_a_task_parked_after_red_verification_says_verify_failed():
    """The measured case: the attempt budget is spent and the last verify was red."""
    state = _state(plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_failed_verify())
    assert park_reason(state) == "verify_failed"
    assert park_kind(state) == "approve"


def test_a_task_with_no_verify_at_all_is_also_not_ready_for_a_pull_request():
    """"We never checked" must not read as "there is nothing to check"."""
    state = _state(plan_ref=PLAN, checkpoint_ref=CHECKPOINT)
    assert park_reason(state) == "verify_failed"


def test_the_reason_is_a_closed_vocabulary():
    """An audit trail with free-form reasons cannot be counted, and a UI switches on these names."""
    assert PARK_REASONS == (
        "answer_required",
        "budget_exhausted",
        "plan_not_produced",
        "verify_failed",
        "review_not_approved",
        "ready_for_approval",
    )
    produced = {
        park_reason(_state()),
        park_reason(_state(plan_ref=PLAN)),
        park_reason(_state(plan_ref=PLAN, verify_ref=_failed_verify())),
        # A verdict we do not recognize is a rejection, never an approval — so this is a reason.
        park_reason(_state(plan_ref=PLAN, verify_ref=_passed_verify(), review_verdict="changes_requested")),
        park_reason(_state(budget={"max_cost_usd": 1.0, "spent_usd": 2.0, "max_tokens": None, "spent_tokens": 0})),
    }
    assert produced <= set(PARK_REASONS)


# ── DEFECT 2: the gate must not park on a false reason ────────────────────────────────────────
#
# `run-cancelrepro-before`, verbatim from the ledger:
#
#   seq 404  kind=review  verdict="approved"  sha="1686ad67..."
#   seq 406  note  approval.requested  "task task-1 is waiting on a human (review_not_approved)"
#
# Same sha, same run, contradictory. The cause: `park_reason` ENDED in a bare
# ``return "review_not_approved"``, so that string was the FALLBACK rather than a reading of
# `review_verdict`. Any park that reached it — an approved task included — was recorded as "review not
# approved". The gate read a different value than the reviewer wrote.


def test_an_approved_verdict_parking_reports_ready_not_review_failure():
    """THE REGRESSION FOR DEFECT 2. FAILS before the fix: it returned `review_not_approved`.

    The park is CORRECT and must survive — releasing reviewed work is a human's decision, and
    `test_plain_engine_walks_the_full_route_the_plan_names` asserts the task still parks. What the
    gate must not do is say the review FAILED when `review_verdict` says it passed.
    """
    state = _state(
        plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_passed_verify(), review_verdict="approved"
    )
    reason = park_reason(state)
    assert reason == "ready_for_approval", (
        "an approved task's gate must report that the work is ready, not a verdict it does not have"
    )
    assert reason != "review_not_approved"
    assert reason in PARK_REASONS, "a reason outside the vocabulary cannot be counted or switched on"


@pytest.mark.parametrize("verdict", ["rejected", "changes_requested", None])
def test_a_verdict_that_is_not_an_approval_IS_reported_as_one(verdict):
    """The other half: `review_not_approved` still means what it says for a verdict that failed.

    Without this, "always return ready_for_approval" would pass the test above and delete the
    distinction — the same class of bug, inverted.
    """
    state = _state(
        plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_passed_verify(), review_verdict=verdict
    )
    assert park_reason(state) == "review_not_approved"


def test_the_gate_reads_the_verdict_the_reviewer_wrote():
    """The property, stated exactly: the reason is a FUNCTION of the recorded verdict, not a default.

    `park_reason` used to end in a bare ``return "review_not_approved"``, so the value was a constant
    for every fully-passed state regardless of what the review step had just written. Holding plan,
    verify and budget fixed and changing ONLY `review_verdict` must change the answer — if it does
    not, the gate is reading something else again.
    """
    base = dict(plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_passed_verify())
    assert park_reason(_state(review_verdict="approved", **base)) == "ready_for_approval"
    assert park_reason(_state(review_verdict="rejected", **base)) == "review_not_approved"


def test_the_gate_message_matches_the_reason_it_reports():
    """A name cannot mislead if the sentence beside it agrees.

    The ledger keeps the name; a human reads the message. The measured defect was a human being told
    "review_not_approved" about a review that had approved, so the two must be checked together.
    """
    from codna.durable.parking import park_message

    approved = _state(
        plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_passed_verify(), review_verdict="approved"
    )
    message = park_message(approved, park_reason(approved))
    assert "verif" in message and "review" in message
    assert "not" not in message.lower(), f"the gate complains about work that passed: {message!r}"
    rejected = _state(plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_passed_verify())
    assert "did not approve" in park_message(rejected, park_reason(rejected))


def test_a_spent_budget_outranks_the_step_reason():
    """The budget is why the step stopped mattering; the step reason would describe the last try."""
    state = _state(
        plan_ref=PLAN,
        verify_ref=_failed_verify(),
        budget={"max_cost_usd": 1.0, "spent_usd": 2.0, "max_tokens": None, "spent_tokens": 0},
    )
    assert park_reason(state) == "budget_exhausted"
    assert park_kind(state) == "budget"


def test_a_question_outranks_everything_because_the_work_has_not_happened():
    ask = {
        "ask_id": "ask_01234567-89ab-cdef-0123-456789abcdef",
        "kind": "command",
        "tool": "run_commands",
        "prompt": "npm run build -- --prod",
    }
    state = _state(ask=ask)
    assert park_reason(state) == "answer_required"
    assert park_kind(state) == "ask"


def test_the_graph_interrupt_carries_the_park_reason():
    """The prompt is the ONLY channel that survives into `_await_decision`.

    Asserted against the node itself rather than through a whole recorded run: the defect was
    precisely that the value was never put in the prompt, so the workflow had nothing to record but
    its hardcoded default.
    """
    from codna.durable.graph import wait_for_human

    captured: dict = {}

    def fake_interrupt(prompt):
        captured.update(prompt)
        return {"identity": "alice@example.com", "approved": True}

    import sys
    import types

    fake_types = types.ModuleType("langgraph.types")
    fake_types.interrupt = fake_interrupt  # type: ignore[attr-defined]
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph.types = fake_types  # type: ignore[attr-defined]
    fake_config = types.ModuleType("langgraph.config")
    fake_config.get_config = lambda: {"metadata": {"codna_task": {
        "run_id": "run-1", "task_id": "task-1", "task_number": 1,
        "brief": "Build a local app", "repo_dir": "/tmp/repo",
    }}}  # type: ignore[attr-defined]
    fake_langgraph.config = fake_config  # type: ignore[attr-defined]
    monkeypatched = {
        "langgraph": fake_langgraph,
        "langgraph.types": fake_types,
        "langgraph.config": fake_config,
    }
    saved = {name: sys.modules.get(name) for name in monkeypatched}
    sys.modules.update(monkeypatched)
    try:
        import asyncio

        state = _state(plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_failed_verify())
        asyncio.run(wait_for_human(state.to_payload()))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert captured["park_reason"] == "verify_failed", captured
    # The rest of the prompt is unchanged, so a UI reading it keeps working.
    assert captured["kind"] == "approve"
    assert captured["verify_ref"] == "verify/task-1/abc"

def test_a_replayed_park_without_the_reason_falls_back_to_the_state_not_to_empty():
    """An OLD interrupt payload must not produce `reason: ""`.

    Temporal replays a park recorded before this change by re-running `wait_for_human`, and the
    replayed prompt has no `park_reason` key — the field did not exist when the history was written.
    A naive `parked.get("park_reason") or ""` then records an approval whose reason is the empty
    string, which is not in `PARK_REASONS`: an audit record that cannot be counted, and strictly
    worse than the `open_pr_ready` it replaced.

    The fallback derives the reason from the REPLAYED STATE through the same shared function. That is
    not a second copy of the rule — it IS the rule — and it is deterministic on replay because
    `park_reason` is a pure function of `TaskState`. `open_pr_ready` would be the wrong repair: it
    restores exactly the value this change exists to stop recording.

    Driven against the real `TaskWorkflow._await_decision` with the workflow's suspension and notify
    boundaries stubbed, so no Temporal server and no clock is involved.
    """
    import asyncio
    from types import SimpleNamespace

    # `task_workflow_graph` imports `temporalio` at module scope, and the `cli secure subsystem` job
    # runs this suite WITHOUT the durable extra. Guarded like `test_verify_environment.py` guards its
    # own module: the six tests above need only `parking`/`graph`, so they still run everywhere, and
    # this one runs in the job that installs the extra.
    pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

    from codna.durable import task_workflow_graph

    # `TaskWorkflow.__init__` reads `workflow.info()`, which needs a running workflow — so the object
    # is shaped rather than constructed, and only the attributes `_await_decision` touches are set.
    task = SimpleNamespace(
        _stopped=False,
        # An OLD-FORMAT payload: no "park_reason", exactly as a pre-change history replays.
        _interrupts=[{"kind": "approve", "verify_ref": "verify/task-1/abc"}],
        _decision=None,
        _approvals=[],
        _state=_state(
            plan_ref=PLAN, checkpoint_ref=CHECKPOINT, verify_ref=_failed_verify()
        ).to_payload(),
        _task_id="task-1",
        _outcome=None,
        _notify=None,
    )

    async def _noop_notify(*args, **kwargs):
        return None

    async def _already_decided(*args, **kwargs):
        # The condition is met immediately: a decision is present, so nothing waits.
        return None

    # `_decision` is read AFTER the wait, so the predicate must not be what supplies it; setting the
    # decision here mirrors "the human already approved while the park was being armed" — the state
    # `_await_decision` must handle anyway.
    task._decision = {"identity": "auditor@codna.ai", "approved": True}
    task._notify = _noop_notify

    original_wait = task_workflow_graph.workflow.wait_condition
    task_workflow_graph.workflow.wait_condition = _already_decided
    try:
        asyncio.run(task_workflow_graph.TaskWorkflow._await_decision(task))
    finally:
        task_workflow_graph.workflow.wait_condition = original_wait

    assert len(task._approvals) == 1, task._approvals
    recorded = task._approvals[0]["reason"]
    # Not the empty string, and not the wrong legacy constant: the truth the replayed state carries.
    assert recorded == "verify_failed", task._approvals
    assert recorded in PARK_REASONS
