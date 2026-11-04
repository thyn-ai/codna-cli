"""D4: a turn that stops on an ASK parks the durable workflow at `wait_for_human`.

The chain this file proves, end to end and against a real Temporal test server:

    the engine's terminal event says `finish_reason: "ask"` + an ask payload
      → the turn activity records it on the task state (a reference, not the payload)
      → the pure router parks  (instead of sending work that never happened to `verify`)
      → the workflow suspends at `wait_for_human` with `parked_on == "answer_required"`
      → `status` shows WHAT the agent is waiting to be allowed to do

Why the LAST step matters as much as the first: a run that parks is only useful if a human can see the
question. Parking on a nameless "needs input" would be indistinguishable from a broken task, which is
how a parked run gets abandoned rather than answered.

The activity- and state-level assertions live here too (they are the cheap half): the transition from
"turn ended on a question" to "state says so" is exactly where a null would hide.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

import asyncio  # noqa: E402
from datetime import timedelta  # noqa: E402

from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable.activities import (  # noqa: E402
    ACTIVITIES,
    activity_executor,
    configure_default_runtime,
)
from codna.durable.contracts import PlanRequest, TaskSpec, TurnRequest  # noqa: E402
from codna.durable.routing import after_plan, after_review, after_verify  # noqa: E402
from codna.durable.state import AskRef, TaskState  # noqa: E402
from codna.durable.workflow_support import park_kind, park_reason  # noqa: E402
from codna.durable.workflows import TaskWorkflowPlain, registered_workflows  # noqa: E402
from tests.durable import fakes  # noqa: E402

TASK_QUEUE = "codna-durable-ask-tests"
RUN_TIMEOUT = timedelta(hours=24)

THE_ASK = {
    "ask_id": "ask_01234567-89ab-cdef-0123-456789abcdef",
    "kind": "command",
    "tool": "run_commands",
    "prompt": "npm run build -- --prod",
}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _state(**overrides: Any) -> TaskState:
    """A valid post-first-turn state; overrides REPLACE fields rather than merging."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "run-1",
        "task_id": "task-1",
        "turn_seq": 1,
        "session_id": "session-durable-1",
        "last_turn_key": "run-1:task-1:1",
        "last_turn_id": "turn-session-durable-1-1",
        "last_event_index": 10,
        "budget": {"max_cost_usd": None, "spent_usd": 0.0, "max_tokens": None, "spent_tokens": 0},
    }
    payload.update(overrides)
    # `session_id=None` alone would leave `last_turn_id` dangling, which `TaskState` refuses. A test that
    # wants "no turn has run yet" says so once instead of restating three coupled fields.
    if payload.get("session_id") is None:
        payload["last_turn_key"] = None
        payload["last_turn_id"] = None
        payload["turn_seq"] = 0
        payload["last_event_index"] = 0
    return TaskState.from_payload(payload)


# -- the state reference ------------------------------------------------------------------


def test_ask_round_trips_through_history_and_is_validated():
    ask = AskRef.from_payload(dict(THE_ASK))
    assert ask is not None
    assert ask.to_payload() == THE_ASK

    state = _state(ask=THE_ASK)
    assert TaskState.from_payload(state.to_payload()).ask == ask
    # The ask survives exactly one place: the state payload a workflow hands between activities.
    assert state.to_payload()["ask"]["ask_id"] == THE_ASK["ask_id"]


def test_an_ask_reference_refuses_an_unknown_kind_rather_than_coercing_it():
    # "followup" answers a question and "tool" releases a blocked call — a UI that guessed wrong would
    # render the wrong control, so the closed set is enforced at the boundary.
    from codna.durable.state import TaskStateError

    for bad in ("whatever", "", "ASK"):
        with pytest.raises(TaskStateError) as excinfo:
            AskRef.from_payload({**THE_ASK, "kind": bad})
        assert excinfo.value.code == "invalid_ask_ref"


def test_an_ask_reference_refuses_an_id_that_could_traverse_a_path():
    from codna.durable.state import TaskStateError

    with pytest.raises(TaskStateError):
        AskRef.from_payload({**THE_ASK, "ask_id": "../../etc/passwd"})


# -- the routing decision -----------------------------------------------------------------


def test_every_router_parks_on_an_ask_instead_of_retrying_or_proceeding():
    # The failure this prevents, precisely: an implement turn that stopped on a question wrote NOTHING,
    # so `verify` would run against an unmodified tree and a passing verify would send the run on to
    # open a pull request for a change nobody made.
    parked = _state(ask=THE_ASK)
    assert after_plan(parked) == "wait_for_human"
    assert after_verify(parked) == "wait_for_human"
    assert after_review(parked) == "wait_for_human"

    # …and the state WITHOUT an ask keeps its existing meaning, so this did not become a catch-all.
    plain = _state()
    assert after_plan(plain) == "plan"  # no plan_ref yet → re-plan, bounded
    assert after_verify(plain) == "implement"
    assert after_review(plain) == "implement"


def test_the_park_reason_names_the_question_rather_than_a_broken_plan():
    parked = _state(ask=THE_ASK)
    assert park_reason(parked) == "answer_required"
    assert park_kind(parked) == "ask"
    # A park for any other reason keeps its own vocabulary.
    assert park_reason(_state()) == "plan_not_produced"
    assert park_kind(_state()) == "approve"


# -- the activity transition --------------------------------------------------------------


def _plan_request(repo: Path, state: TaskState) -> PlanRequest:
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="Fix the flaky retry",
        repo_dir=str(repo),
        verify_commands=("true",),
        engine="plain",
    )
    from codna.durable.contracts import TaskContext

    return TaskContext.from_spec(spec).plan_request(spec.brief, state.to_payload())


def test_a_plan_turn_that_asks_claims_NO_plan_ref(tmp_path: Path):
    """The subtle one: `after_plan` only checks `plan_ref`, so a turn that asked must not produce one.

    Before this, the plan reference was derived from the turn id unconditionally — so a plan turn that
    never planned would still hand the router a real-looking reference and the task would march on to
    implement a plan that does not exist.
    """
    from codna.durable import activity_steps

    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.sidecar.ask_on_terminal = dict(THE_ASK)

    outcome = activity_steps.plan_task(_plan_request(repo, _state(turn_seq=0, session_id=None)))
    state = TaskState.from_payload(outcome.state)

    assert state.plan_ref is None, "a turn that asked must not claim to have produced a plan"
    assert state.ask is not None
    assert state.ask.ask_id == THE_ASK["ask_id"]
    assert state.ask.kind == "command"
    # The turn was still RECORDED (and still cost money), which is what keeps the run's accounting and
    # its retry idempotency honest.
    assert state.turn_seq == 1
    assert state.budget.spent_usd == pytest.approx(0.01)
    assert outcome.details["finish_reason"] == "ask"


def test_an_implement_turn_that_asks_records_the_ask_and_keeps_the_spend(tmp_path: Path):
    from codna.durable import activity_steps

    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.sidecar.ask_on_terminal = dict(THE_ASK)

    state = _state(turn_seq=1, plan_ref="refs/plans/task-1-deadbeef")
    request = TurnRequest(
        state=state.to_payload(),
        context=_plan_request(repo, state).context,
        prompt="implement it",
        steer=(),
    )
    outcome = activity_steps.run_cline_turn(request)
    updated = TaskState.from_payload(outcome.state)

    assert updated.ask is not None
    assert updated.ask.prompt == THE_ASK["prompt"]
    # The turn is still a paid turn: it is in the ledger, it advanced turn_seq, and a retry keys off it.
    assert updated.turn_seq == 2
    assert harness.sidecar.turns, "the turn must actually have run"


def test_a_terminal_that_says_ask_without_an_ask_payload_is_A_LOUD_FAILURE(tmp_path: Path):
    """The honesty case: a protocol violation must not be filed as a completed turn.

    `finish_reason=ask` with nothing to ask means this build and the engine disagree about the wire. The
    one response that is never acceptable is treating it as "done" — the run would verify and open a PR
    on work that never happened.
    """
    from codna.durable import activity_steps
    from codna.durable.activity_support import ActivityError

    class Bare:
        """A turn outcome that claims an ask it does not carry."""

        session_id = "session-durable-1"
        turn_id = "turn-x"
        event_index = 3
        cost_usd = 0.0
        tokens = 0
        priced = True
        resumed = False
        finish_reason = "ask"
        ask = None

    with pytest.raises(ActivityError) as excinfo:
        activity_steps._ask_ref(Bare())
    assert excinfo.value.code == "activity_ask_malformed"


def test_a_ledger_only_retry_still_sees_the_ask(tmp_path: Path):
    """A crash between the ledger append and the workflow's adoption must not lose the park.

    The recorded `turn.finished` carries `finish_reason` + the ask, so an attempt that never touches the
    engine still reports the turn as parked. Without this, a retried ask would be indistinguishable from
    a completed turn — the run would verify nothing and call it progress.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.sidecar.ask_on_terminal = dict(THE_ASK)

    state = _state(turn_seq=0, session_id=None)
    request = _plan_request(repo, state)
    from codna.durable import activity_steps

    first = activity_steps.plan_task(request)
    assert TaskState.from_payload(first.state).ask is not None

    # Re-running the SAME activity (a Temporal retry): the ledger already has the turn, so `drive_turn`
    # takes the resume path and never touches the engine.
    second = activity_steps.plan_task(request)
    resumed = TaskState.from_payload(second.state)
    assert resumed.ask is not None and resumed.ask.ask_id == THE_ASK["ask_id"]
    assert len(harness.sidecar.turns) == 1, "a retry must not buy a second turn"


# -- the workflow: it PARKS ---------------------------------------------------------------


async def _wait_parked(handle: Any, attempts: int = 600) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for _ in range(attempts):
        last = await handle.query(TaskWorkflowPlain.status)
        if last.get("awaiting_approval"):
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"the task never parked; last status={last!r}")


def test_a_durable_run_that_hits_an_ask_parks_at_wait_for_human(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    # The plan turn is the one that asks: the earliest possible point, so the route proves the task
    # parked BEFORE doing any work rather than after a partial implement.
    harness.sidecar.ask_on_terminal = dict(THE_ASK)

    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="Fix the flaky retry in the client",
        repo_dir=str(repo),
        verify_commands=("true",),
        engine="plain",
    )

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="ask-park",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        status = await _wait_parked(handle)
        # Answer it: a park the engine cannot be released from is a stop, not a park.
        await handle.execute_update(TaskWorkflowPlain.approve, {"identity": "alice@example.com"})
        completion = await handle.result()
        return {"status": status, "completion": completion}

    env = _run(_start_worker_and_run(scenario))

    status = env["status"]
    # 1) Parked, with the reason naming the QUESTION (not a broken plan).
    assert status["awaiting_approval"] is True
    assert status["parked_on"] == "answer_required"
    assert status["pending_approval"]["kind"] == "ask"
    # 2) The park shows WHAT is being asked, so a human can answer it.
    assert status["ask"]["ask_id"] == THE_ASK["ask_id"]
    assert status["ask"]["tool"] == "run_commands"
    assert status["ask"]["prompt"] == THE_ASK["prompt"]

    completion = env["completion"]
    assert completion.outcome == "approved"
    # 3) The route stopped at the ask: nothing was verified, and no PR was opened for work that never
    # ran. This is the assertion the plan's "the run continues" line is really about — the run did not
    # die, and it did not lie.
    assert harness.route() == ["plan"], harness.route()
    assert harness.verify_calls == 0
    assert harness.push_calls == 0


async def _start_worker_and_run(scenario: Any) -> Any:
    env = await WorkflowEnvironment.start_time_skipping()
    try:
        with activity_executor() as executor:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=list(registered_workflows()),
                activities=list(ACTIVITIES),
                activity_executor=executor,
            ):
                with env.auto_time_skipping_disabled():
                    return await scenario(env.client)
    finally:
        await env.shutdown()


# -- the reuse seam ------------------------------------------------------------------------
# `configure_default_runtime` is process-wide, so this module must say so rather than assume another
# test file left it pointed somewhere harmless.
def test_the_activity_runtime_is_the_harness_that_was_installed(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    from codna.durable import activities

    assert activities._use() is harness.runtime
    configure_default_runtime(harness.runtime)  # idempotent
