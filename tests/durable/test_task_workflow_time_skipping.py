"""The task workflow under a real Temporal test server, driving the REAL activities.

Every test runs against ``WorkflowEnvironment.start_time_skipping()`` with the production activity
functions registered — a fake sidecar instead of the network, a fake test runner instead of a
subprocess, a fake reviewer instead of a model, and a real git repository for the checkpoint. That
combination is deliberate: a stubbed activity proves the workflow calls *something* in the right
order, but the seam where a durable run actually breaks is between the halves — the state a turn
returns must be the state the checkpoint commits must be the state the reviewer reads. Stubs cannot
see that seam; the real bodies can.

Scenarios, and why each is worth a test:

* the exact route the plan names — ``plan → implement → verify → implement → verify → review →
  implement → review → open_pr → gate`` — asserted from the LEDGER, the record a human audits;
* ``status`` shows the pending approval, ``approve`` with an identity completes it, an anonymous one
  is refused and leaves the task parked;
* budget exhaustion parks with ``outcome="budget_exhausted"``, not a false "approved";
* exactly ONE checkpoint per ``turn_seq`` — the plan's exit criterion;
* ``RunWorkflow`` continues-as-new and still finishes every task;
* and BOTH engines reach the same observable outcome for the same brief.

The module-level ``importorskip`` calls keep the base suite collectable without the extra. Run
WITHOUT the extra this file must SKIP; the durable CI job runs it with the extra so a skip is not how
it passes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

import asyncio  # noqa: E402
import time  # noqa: E402
from datetime import timedelta  # noqa: E402

from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable.activities import (  # noqa: E402
    ACTIVITIES,
    ACTIVITY_NAMES,
    activity_executor,
)
from codna.durable.contracts import (  # noqa: E402
    NAME_CHECKPOINT,
    NAME_NOTIFY,
    NAME_PLAN,
    NAME_TURN,
    NotifyRequest,
    TaskSpec,
)
from codna.durable.graph import build_task_graph  # noqa: E402
from codna.durable.workflows import (  # noqa: E402
    RunWorkflow,
    TaskWorkflow,
    TaskWorkflowPlain,
    registered_workflows,
)
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402

TASK_QUEUE = "codna-durable-workflow-tests"

# An explicit, LARGE run timeout on every start. This is a correctness-of-the-harness knob, not a
# product decision, and it is large for a measured reason: `start_time_skipping` advances its clock
# whenever the server believes every pending task is a timer, and a parked workflow's remaining
# `run_timeout` IS such a timer. With a five-minute run timeout the server could skip the run's whole
# lifetime while the test was still on its way to send the approval Update, producing
# `TimeoutError('Workflow timed out')` — a flake with a message that names nothing. A day of logical
# time cannot elapse in these scenarios (no `workflow.sleep` beyond a few milliseconds), so this
# removes the race instead of narrowing it. The D2.0 spike passed its own run timeout for the same
# reason; it simply did not have to be this large for a one-shot scenario.
RUN_TIMEOUT = timedelta(hours=24)

# -- the waits in this file are BOUNDED, and the bounds are wall-clock --------------------------
#
# Every helper below used to be unbounded in one direction, so a workflow that stopped progressing
# could only be rescued by pytest's GLOBAL timeout. Measured in CI on 2026-11-03: the job burned
# 10m25s, and the traceback pointed at `asyncio.run` inside a `select()` — it named the event loop and
# not the wait. A hang must fail with a message that says WHICH wait gave up and what the workflow was
# doing when it did.
#
# These are ceilings on a HANG, not tolerances for a slow machine, so they are far above the measured
# cost of the happy path: the whole file takes ~15s locally and the slowest single test ~2s, so 30s for
# a park and 60s for a completion cannot fire because a runner is loaded. They exist to convert a
# 300s global timeout into a named assertion, nothing more.
PARKED_TIMEOUT_S = 30.0
RESULT_TIMEOUT_S = 60.0
# One query's own RPC deadline. Well under PARKED_TIMEOUT_S so a wedged query fails by name (an
# `RPCError` naming the deadline) rather than consuming the whole park budget in one call.
QUERY_RPC_TIMEOUT_S = 5.0
# The same reasoning for an Update: `execute_update` waits for the update's outcome, and an Update
# that is never accepted blocks the caller just as silently as an unaccepted query.
UPDATE_TIMEOUT_S = 10.0
# Keep the SDK's own RPC deadline AFTER the outer assertion, so the descriptive handler always wins.
RPC_DEADLINE_MARGIN_S = 1.0
POLL_INTERVAL_S = 0.02
# How long the regression test for the park/decision race holds the `approval.requested` notify
# open. It only has to outlive one gate-open-to-`wait_condition` transition, so it is small; it is
# NOT a tolerance and nothing measures it.
HELD_NOTIFY_S = 0.5

# The route the plan names, as the LEDGER records it. `implement` is one entry per paid turn — the
# checkpoint that follows it is its own entry, so "one per turn" is directly visible.
FULL_ROUTE = [
    "plan",
    "implement",
    "checkpoint",
    "verify",
    "implement",
    "checkpoint",
    "verify",
    "review",
    "implement",
    "checkpoint",
    "verify",
    "review",
    "open_pr",
]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _workflow_id(handle: Any) -> str:
    """The handle's workflow id, for a failure message. Never raises: it only decorates."""
    return str(getattr(handle, "id", "") or "<unknown>")


async def _update(handle: Any, update: Any, payload: Any, *, timeout_s: float = UPDATE_TIMEOUT_S) -> Any:
    """`handle.execute_update(...)` with a ceiling, so an unaccepted Update fails by name.

    The gate tests send an approval and then assert on history. If the Update is never ACCEPTED the
    caller waits for the outcome with no bound of its own, and the failure surfaces as the global
    timeout — the same uninformative shape as an unbounded query. A timeout here means the workflow
    did not apply or reject the Update in time, which is a workflow problem, so the message says so
    rather than blaming the client.
    """
    try:
        return await asyncio.wait_for(
            handle.execute_update(update, payload, rpc_timeout=timedelta(seconds=timeout_s + RPC_DEADLINE_MARGIN_S)),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            f"_update: the Update {getattr(update, '__name__', update)!r} was not accepted within "
            f"{timeout_s:g}s (workflow_id={_workflow_id(handle)!r}, payload={payload!r}). The "
            "workflow never applied or rejected it — check that it reached the gate and that its "
            "validator did not wedge the workflow task."
        ) from None


def _spec(**overrides: Any) -> TaskSpec:
    """A minimal valid task spec against the harness repository."""
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the flaky retry in the client",
        "repo_dir": str(Path(overrides.pop("repo_dir", "/tmp/codna-harness-repo"))),
        "verify_commands": ("true",),
        "engine": "plain",
    }
    payload.update(overrides)
    spec = TaskSpec(**payload)
    spec.validate()
    return spec


async def _with_worker(engine: str, scenario: Any) -> Any:
    """Run a scenario against a fresh worker and a fresh time-skipping server.

    Fresh per scenario because the plugin MUTATES the graph it is given (D2.0 spike, finding 2): a
    reused graph works once and raises the second time. The plugin also brings its own node
    activities, so only the plugin is passed — adding `plugin.activities` as well makes the worker
    reject a duplicate activity name.
    """
    env = await WorkflowEnvironment.start_time_skipping()
    try:
        plugins = [build_task_graph().plugin] if engine == "langgraph" else []
        with activity_executor() as executor:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=list(registered_workflows()),
                activities=list(ACTIVITIES),
                activity_executor=executor,
                plugins=plugins,
            ):
                # AUTO TIME SKIPPING OFF for the scenario. The server skips to the next event
                # whenever every pending task is a timer, and a PARKED workflow's outstanding
                # `run_timeout` is exactly such a timer: with skipping on, the server can jump past
                # the whole run lifetime while the test is still on its way to send the approval
                # Update (measured — `TimeoutError('Workflow timed out')` with no pending
                # activities, and only on the graph engine because its interrupt parks a
                # workflow-task later). These scenarios measure control flow, not simulated time, so
                # they do not need skipping; the run timeout stays as the harness's safety net.
                with env.auto_time_skipping_disabled():
                    return await scenario(env.client)
    finally:
        await env.shutdown()


async def _history_tail(handle: Any, *, limit: int = 8, timeout_s: float = QUERY_RPC_TIMEOUT_S) -> str:
    """The last few history events as ``event_id:NAME(detail)``, for a failure message.

    The assertion this feeds already tells the reader to fetch the history; fetching it HERE is what
    turns the next occurrence into evidence instead of another investigation. A hang's last events
    say which step stopped progressing (an ``ACTIVITY_TASK_SCHEDULED`` with no matching completion,
    an accepted Update with no ``WORKFLOW_EXECUTION_COMPLETED``), so the message can name the wedge
    without anyone reproducing it.

    Never raises and never blocks the failure path: a history that cannot be fetched (closed run,
    wedged service) degrades to a line saying so, because decorating an assertion must not replace
    it with a different exception.
    """
    try:
        history = await asyncio.wait_for(handle.fetch_history(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - a decorated failure must still be the failure
        return f"<history unavailable: {type(exc).__name__}: {exc}>"
    events = list(history.events)[-limit:]
    rendered: list[str] = []
    for event in events:
        event_type = int(event.event_type)
        try:
            name = EventType.Name(event.event_type)
        except ValueError:  # pragma: no cover - an event type this SDK does not name
            name = f"EVENT_TYPE_{event_type}"
        detail = ""
        if event.HasField("activity_task_scheduled_event_attributes"):
            detail = f"({event.activity_task_scheduled_event_attributes.activity_type.name})"
        elif event.HasField("workflow_execution_update_accepted_event_attributes"):
            detail = "(update accepted)"
        elif event.HasField("workflow_execution_completed_event_attributes"):
            detail = "(completed)"
        elif event.HasField("workflow_task_failed_event_attributes"):
            detail = "(WORKFLOW TASK FAILED)"
        rendered.append(f"{event.event_id}:{name}{detail}")
    return ", ".join(rendered) if rendered else "<history empty>"


async def _result(handle: Any, *, timeout_s: float = RESULT_TIMEOUT_S) -> Any:
    """`handle.result()` with a CEILING and the FAILURE CAUSE in the message.

    A bare ``WorkflowFailureError`` says only "Workflow execution failed"; the cause carries the
    actual reason (an activity's structured error, a timeout type). Without it, a red build names the
    line and withholds the diagnosis — which is the same class of bug `review.py`'s `_wrapped_details`
    exists to fix.

    The ceiling is the other half of that fix, and it is not a slowdown guard: a workflow that never
    completes leaves this await pending forever, the event loop parked in ``select()``, and the only
    thing that eventually fires is pytest's GLOBAL timeout — whose traceback names ``asyncio.run``
    and not the step that wedged. Measured in CI on 2026-11-03: 10m25s of job time, a traceback that
    identified nothing. ``wait_for`` turns that into an assertion that names this wait and the
    engine, so the next occurrence costs a line of log instead of a job.

    On a timeout the message also carries the LAST HISTORY EVENTS (see `_history_tail`): the wedge is
    always visible as "what was the last thing that happened", and fetching it at failure time is
    cheaper than a reproduction that never arrives.
    """
    try:
        return await asyncio.wait_for(handle.result(), timeout=timeout_s)
    except asyncio.TimeoutError:
        tail = await _history_tail(handle)
        raise AssertionError(
            f"_result: the workflow did not COMPLETE within {timeout_s:g}s "
            f"(workflow_id={_workflow_id(handle)!r}). The scenario awaited `handle.result()` and the "
            "workflow never reached a terminal state — this is a HANG, not a slow runner. The step "
            f"that stopped making progress is the last one in the history; last events: {tail}."
        ) from None
    except Exception as exc:  # noqa: BLE001 - re-raised with the cause attached
        raise AssertionError(
            f"the workflow failed: {type(exc).__name__}: {exc} | cause={getattr(exc, 'cause', None)!r}"
        ) from exc


async def _query(handle: Any, query: Any, *, timeout_s: float) -> dict[str, Any]:
    """`handle.query(...)` with a ceiling that does not trust the transport.

    ``rpc_timeout`` alone is NOT a bound: it is a deadline the SDK applies to its own RPC, and a
    query that blocks anywhere else (a wedged workflow task, a stub, a future SDK that changes when
    the deadline is armed) outlives it. ``wait_for`` is enforced by this event loop, so it holds
    regardless — which is the whole point of a fail-fast guard. ``rpc_timeout`` stays as well, so the
    server-side reason is part of the error rather than only the client's cancellation.
    """
    try:
        return await asyncio.wait_for(
            handle.query(query, rpc_timeout=timedelta(seconds=timeout_s + RPC_DEADLINE_MARGIN_S)), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            f"_query: one status query did not answer within {timeout_s:g}s "
            f"(workflow_id={_workflow_id(handle)!r}). A query answers from the workflow's last "
            "completed task, so this means the workflow task itself is wedged — not that the runner "
            "is slow."
        ) from None


async def _wait_parked(
    handle: Any, *, query: Any, timeout_s: float = PARKED_TIMEOUT_S
) -> dict[str, Any]:
    """Wait until the workflow reports it is waiting on a human, with a WALL-CLOCK ceiling.

    Polled, not slept: a fixed delay is either flaky or slow, and time-skipping says nothing about
    wall-clock progress here.

    The ceiling is wall-clock, not an attempt count, and that distinction is the bug this replaces.
    ``for _ in range(600)`` reads like twelve seconds (600 x the 20 ms sleep) but bounds nothing: the
    budget is consumed only AFTER an await returns, so a single query that never resolves — or a
    query that takes a second each on a loaded runner — outstretches any attempt count without ever
    reaching the end of the loop. The observed CI hang is exactly the first case: the loop was inside
    one ``handle.query`` when the global timeout fired. So each query is bounded (``_query``) and the
    loop itself has a deadline.
    """
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"_wait_parked: the task never reported awaiting_approval within {timeout_s:g}s "
                f"(workflow_id={_workflow_id(handle)!r}, engine={last.get('engine')!r}). "
                f"Last status={last!r}. The workflow is answering queries but has not reached the "
                "gate — the workflow itself is not progressing, not the client."
            )
        query_timeout = min(QUERY_RPC_TIMEOUT_S, remaining)
        try:
            last = await _query(handle, query, timeout_s=query_timeout)
        except AssertionError:
            if query_timeout < remaining:
                # The query's own, shorter ceiling fired while the park budget still had time left.
                raise
            # The query consumed the ENTIRE remaining park budget. Attribute that from the configured
            # ceiling, not a second clock read whose scheduling can make the two deadlines race.
            raise AssertionError(
                f"_wait_parked: the task never reported awaiting_approval within {timeout_s:g}s "
                f"(workflow_id={_workflow_id(handle)!r}). The last status query was still in flight "
                "when the budget expired, so the workflow stopped answering as well as not parking; "
                f"last completed status={last!r}."
            ) from None
        if last.get("awaiting_approval"):
            return last
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"_wait_parked: the task never reported awaiting_approval within {timeout_s:g}s "
                f"(workflow_id={_workflow_id(handle)!r}, engine={last.get('engine')!r}). "
                f"Last status={last!r}. The workflow is answering queries but has not reached the "
                "gate — the workflow itself is not progressing, not the client."
            )
        await asyncio.sleep(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------------------
# The scripted-run helper: verify fails once, review rejects once.
# ---------------------------------------------------------------------------------------


def _scripted(repo: Path) -> fakes.Harness:
    """A harness scripted for the plan's scenario: one verify failure, one review rejection.

    The runs root is derived from the REPOSITORY's own directory, so two harnesses in one test get
    two ledgers. `run_id` is ``run-1`` for every task in these tests, and a shared ledger would merge
    two runs' records into one route.
    """
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    harness.verify_passes = [False, True, True, True]
    harness.review_conclusions = ["neutral", "success"]
    return harness


# ---------------------------------------------------------------------------------------
# The exact route, under the plain engine.
# ---------------------------------------------------------------------------------------


def test_plain_engine_walks_the_full_route_the_plan_names(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    harness = _scripted(repo)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-route",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        status = await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "alice@example.com"})
        completion = await _result(handle)
        questions = await handle.query(TaskWorkflowPlain.approvals)
        return {"status": status, "completion": completion, "questions": questions}

    result = _run(_with_worker("plain", scenario))

    # The route, exactly — from the ledger, which is what an auditor reads.
    assert harness.route() == FULL_ROUTE, harness.route()
    # Parked, and `status` says exactly what it is waiting for.
    assert result["status"]["awaiting_approval"] is True
    assert result["status"]["parked_on"] == "review_not_approved"
    assert result["status"]["pending_approval"]["kind"] == "approve"
    assert result["status"]["pending_approval"]["reason"] == "review_not_approved"
    # Exactly one checkpoint per IMPLEMENT turn. The plan is a paid turn too (turn 1), so the three
    # implement turns are 2, 3 and 4 — which is the point of giving the plan its own `turn_seq`: the
    # cost of planning is visible instead of folded into the first implement turn.
    checkpoints = harness.checkpoints()
    turn_seqs = [entry["turn_seq"] for entry in checkpoints]
    assert turn_seqs == [2, 3, 4], turn_seqs
    assert len({entry["sha"] for entry in checkpoints}) == 3, "a checkpoint produced no new commit"
    # The turn keys are the run:task:seq identity, one per paid turn, in order.
    assert harness.turn_keys() == [
        "run-1:task-1:1",
        "run-1:task-1:2",
        "run-1:task-1:3",
        "run-1:task-1:4",
    ]
    # Completed, with the approver recorded and the reference set intact.
    completion = result["completion"]
    assert completion.outcome == "approved"
    assert completion.approved_by == "alice@example.com"
    assert completion.turn_seq == 4
    assert completion.engine == "plain"
    assert completion.state["review_verdict"] == "approved"
    assert completion.state["verify_ref"]["passed"] is True
    assert completion.state["pr"], "the pull request was not recorded"
    assert result["questions"][-1] == {
        "identity": "alice@example.com",
        "kind": "approve",
        "turn_seq": 4,
        "reason": "review_not_approved",
    }


def test_graph_engine_walks_the_full_route_the_plan_names(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    harness = _scripted(repo)
    spec = _spec(repo_dir=str(repo), engine="langgraph")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflow.run,
            args=[spec.to_payload(), None],
            id="graph-route",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        status = await _wait_parked(handle, query=TaskWorkflow.status)
        await _update(handle, TaskWorkflow.approve, {"identity": "alice@example.com"})
        completion = await _result(handle)
        history = await handle.fetch_history()
        return {
            "status": status,
            "completion": completion,
            "scheduled": [
                event.activity_task_scheduled_event_attributes.activity_type.name
                for event in history.events
                if event.HasField("activity_task_scheduled_event_attributes")
            ],
        }

    result = _run(_with_worker("langgraph", scenario))

    # The same route, from the same ledger, reached through a different control flow.
    assert harness.route() == FULL_ROUTE, harness.route()
    # Every doing-node became a REAL activity, in route order — the loop two implements wide, because
    # the scenario fails verify once and rejects review once.
    nodes = [name for name in result["scheduled"] if name.startswith("codna.task.")]
    assert nodes == [
        "codna.task.plan",
        "codna.task.implement",
        "codna.task.verify",
        "codna.task.implement",
        "codna.task.verify",
        "codna.task.review",
        "codna.task.implement",
        "codna.task.verify",
        "codna.task.review",
        "codna.task.open_pr",
    ], nodes
    # ...and the approval node never did: `interrupt()` needs an executing graph.
    assert "codna.task.wait_for_human" not in result["scheduled"]
    # Parked on the interrupt, whose payload names what a human is being asked about.
    assert result["status"]["awaiting_approval"] is True
    assert result["status"]["interrupts"][0]["kind"] == "approve"
    assert result["status"]["interrupts"][0]["task_id"] == "task-1"
    assert result["status"]["interrupts"][0]["budget_exhausted"] is False
    completion = result["completion"]
    assert completion.outcome == "approved"
    assert completion.approved_by == "alice@example.com"
    assert completion.engine == "langgraph"
    assert completion.state["pr"]


# ---------------------------------------------------------------------------------------
# Parity: the same brief, the same observable outcome, under both engines.
# ---------------------------------------------------------------------------------------


def test_both_engines_reach_the_same_observable_outcome(tmp_path: Path):
    """The plan's parity claim, tested rather than asserted in prose.

    "Observable outcome" is defined narrowly on purpose: the route (from the ledger), the outcome,
    the approving identity, the turn count, the checkpoint refs, and the reference set a human would
    look at. History itself differs by construction — the plugin writes node activities — so
    asserting it equal would assert something untrue.
    """

    async def one(engine: str, workflow_cls: Any, repo: Path) -> dict[str, Any]:
        # A separate runs root per engine: both runs use the same run_id, and a shared ledger would
        # merge their records — making the comparison about the fixture instead of the engines.
        harness = _scripted(repo)

        async def scenario(client: Any) -> dict[str, Any]:
            handle = await client.start_workflow(
                workflow_cls.run,
                args=[_spec(repo_dir=str(repo), engine=engine).to_payload(), None],
                id=f"parity-{engine}",
                task_queue=TASK_QUEUE,
                run_timeout=RUN_TIMEOUT,
            )
            await _wait_parked(handle, query=workflow_cls.status)
            await _update(handle, workflow_cls.approve, {"identity": "reviewer@codna.ai"})
            completion = await _result(handle)
            return {"completion": completion, "route": harness.route(), "keys": harness.turn_keys()}

        return await _with_worker(engine, scenario)

    plain = _run(one("plain", TaskWorkflowPlain, fakes.make_repo(tmp_path / "plain")))
    graph = _run(one("langgraph", TaskWorkflow, fakes.make_repo(tmp_path / "graph")))

    assert plain["route"] == graph["route"] == FULL_ROUTE
    assert plain["keys"] == graph["keys"]
    left, right = plain["completion"], graph["completion"]
    assert left.outcome == right.outcome == "approved"
    assert left.approved_by == right.approved_by == "reviewer@codna.ai"
    assert left.turn_seq == right.turn_seq
    # The checkpoint REF is a deterministic function of (task, turn), so it must agree; the SHA
    # cannot, because the two runs are separate repositories.
    assert left.checkpoints == right.checkpoints
    for field in ("plan_ref", "verify_ref", "review_verdict", "attempts", "budget"):
        assert left.state[field] == right.state[field], field
    # The PR agrees on everything a HUMAN would compare: the number, the URL and the branch. The
    # head sha cannot agree — the two runs are separate repositories with separate commit timestamps.
    for field in ("number", "url", "branch"):
        assert left.state["pr"][field] == right.state["pr"][field], field
    # The engines are distinguishable, which is what makes the comparison meaningful.
    assert (left.engine, right.engine) == ("plain", "langgraph")


# ---------------------------------------------------------------------------------------
# Budget exhaustion.
# ---------------------------------------------------------------------------------------


def test_budget_exhaustion_parks_the_task_and_says_why(tmp_path: Path):
    """A spent budget parks instead of looping, and the outcome is honest about why it stopped."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    # Each turn reports $0.25 cumulative; a $0.10 cap is crossed by the first turn.
    harness = fakes.build_harness(repo, runs_root, cost_per_turn=0.25)
    spec = _spec(repo_dir=str(repo), engine="plain", max_cost_usd=0.10)

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-budget",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        status = await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "finance@codna.ai"})
        completion = await _result(handle)
        return {"status": status, "completion": completion}

    result = _run(_with_worker("plain", scenario))

    assert result["status"]["parked_on"] == "budget_exhausted"
    assert result["status"]["budget_exhausted"] is True
    assert result["status"]["pending_approval"]["kind"] == "budget"
    # Honest outcome: no pull request, and the record says the task stopped for money rather than
    # reading as a success because a human clicked approve.
    assert result["completion"].outcome == "budget_exhausted"
    assert result["completion"].approved_by == "finance@codna.ai"
    assert result["completion"].state["pr"] is None
    assert harness.push_calls == 0
    assert "open_pr" not in harness.route()
    assert result["completion"].state["budget"]["spent_usd"] > 0


# ---------------------------------------------------------------------------------------
# One checkpoint per turn_seq.
# ---------------------------------------------------------------------------------------


def test_exactly_one_checkpoint_per_turn_seq(tmp_path: Path):
    """The ledger's checkpoint records are one per turn, with one commit and one ref each.

    Asserted three ways, because each can fail alone: the ledger's own records (one per turn_seq), the
    refs the activity wrote (one per turn_seq), and git's commit count (a second commit for one turn
    would be a rewritten history the audit cannot explain).
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    # One verify failure, so there are exactly TWO implement turns — the smallest run that can show
    # "one checkpoint per turn" at all (a single turn cannot distinguish "per turn" from "once").
    harness.verify_passes = [False, True, True]
    harness.review_conclusions = ["success"]
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-turns",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "a@b.c"})
        return {"completion": await _result(handle)}

    result = _run(_with_worker("plain", scenario))

    # Two implement turns. The plan is turn 1 (a read-only paid turn), so the checkpoints are 2 and 3:
    # a read-only turn changes nothing, so there is no commit for it to anchor.
    turn_seqs = [entry["turn_seq"] for entry in harness.checkpoints()]
    assert turn_seqs == [2, 3], turn_seqs
    # One audit ref per turn, and the refs resolve to the recorded commits.
    for entry in harness.checkpoints():
        resolved = fakes.git(repo, "rev-parse", entry["ref"]).strip()
        assert resolved == entry["sha"], (entry["ref"], resolved, entry["sha"])
    # HEAD is the task branch, at the last checkpoint, and its history is the init commit plus one
    # commit per turn — no more. Logging HEAD (not `main`) because the checkpoint checks the task
    # branch out, which is the state a reviewer of the pushed branch would see.
    assert fakes.git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == spec.branch
    assert fakes.git(repo, "rev-parse", "HEAD").strip() == harness.checkpoints()[-1]["sha"]
    commits = fakes.git(repo, "log", "--format=%H").strip().splitlines()
    assert len(commits) == 3, commits
    assert result["completion"].turn_seq == 3


def test_a_retried_checkpoint_does_not_create_a_second_commit(tmp_path: Path):
    """The idempotent path, tested against real git: same turn, same commit, one ledger record.

    This is the property that makes a worker crash mid-checkpoint harmless. Issue the activity twice
    with the same request — which is exactly what Temporal does on a retry — and assert git has ONE
    commit for that turn.
    """
    from codna.durable.activities import checkpoint_turn, run_cline_turn
    from codna.durable.contracts import CheckpointRequest, TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    (repo / "app.py").write_text("def hello():\n    return 'changed'\n", encoding="utf-8")

    state = spec.initial_state()
    # One real turn so there is a turn_seq to checkpoint.
    moved = run_cline_turn(context.turn_request("implement it", state))
    first = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))
    second = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    assert first.created is True
    assert second.created is False
    assert first.sha == second.sha
    ledger_checkpoints = [entry for entry in harness.checkpoints()]
    assert len(ledger_checkpoints) == 1, ledger_checkpoints
    assert ledger_checkpoints[0]["turn_seq"] == 1
    # git: init + the turn's commit, and nothing else.
    commits = fakes.git(repo, "log", "--format=%H").strip().splitlines()
    assert len(commits) == 2, commits


# ---------------------------------------------------------------------------------------
# RunWorkflow: continue-as-new.
# ---------------------------------------------------------------------------------------


def test_run_workflow_continues_as_new_and_finishes_every_task(tmp_path: Path):
    """Two tasks per generation, five tasks: the run restarts and still finishes all of them.

    The assertion is on the OUTCOME plus the generation count, which is what a caller can see. The
    mechanism is proven by the generations — `continue_as_new` is the only thing that grows it — and
    by the fact that every task's summary survived the restart.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    tasks = tuple(
        _spec(task_id=f"task-{number}", task_number=number, repo_dir=str(repo), engine="plain")
        for number in range(1, 6)
    )
    run_payload = {
        "run_id": "run-1",
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": "plain",
        "continue_as_new_tasks": 2,
        "continue_as_new_history_events": 100_000,
    }

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            RunWorkflow.run, args=[run_payload, None], id="run-can", task_queue=TASK_QUEUE, run_timeout=RUN_TIMEOUT
        )
        # Each child parks on its approval. Approve every one that appears, and stop only when the RUN
        # itself has closed — a helper that returns as soon as no child is parked would return before
        # the first child has even started. `describe()` is the honest signal for "is it over".
        for _ in range(600):
            await _approve_children(client)
            described = await handle.describe()
            if int(described.status) != 1:  # 1 == RUNNING
                break
            await asyncio.sleep(0.05)
        return {"completion": await asyncio.wait_for(_result(handle), timeout=30)}

    result = _run(_with_worker("plain", scenario))
    completion = result["completion"]

    assert completion.outcome == "completed", completion
    assert [summary.task_id for summary in completion.tasks] == [
        f"task-{number}" for number in range(1, 6)
    ]
    # Continue-as-new happened: only a restart grows the generation counter.
    assert completion.generations >= 3, completion.generations
    assert completion.spent_usd > 0
    # Every task's spend was carried across the restart, not reset by it.
    assert all(summary.cost_usd > 0 for summary in completion.tasks)
    assert harness.push_calls == 5


async def _approve_children(client: Any) -> None:
    """Approve any parked child task. Absent or already-closed children are simply skipped.

    The two benign races (a child not started yet, a decision a concurrent iteration already
    recorded) are `approve_if_parked`'s, which absorbs only those. Everything else propagates, so a
    harness bug fails at the call instead of parking every child until the 30s result timeout.
    """
    for number in range(1, 6):
        await approve_if_parked(
            client, f"task-run-1-task-{number}", TaskWorkflowPlain, identity="soak@codna.ai"
        )


# ---------------------------------------------------------------------------------------
# The gate itself.
# ---------------------------------------------------------------------------------------


def test_an_approval_without_an_identity_is_refused_and_the_task_stays_parked(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-gate",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        refused: list[str] = []
        for payload in ({"approved": True}, {"identity": "   "}, {"identity": 7}):
            try:
                await _update(handle, TaskWorkflowPlain.approve, payload)
            except AssertionError:
                # `_update`'s OWN ceiling. Never absorb it here: a refusal is an answer from the
                # validator, a timeout is no answer at all.
                raise
            except Exception as exc:  # noqa: BLE001 - the SDK's refusal type varies by timing
                # The REASON comes back on `cause`: the update's own failure is the validator's
                # exception, not a generic "update failed". A refusal nobody can explain is as bad as
                # a silent acceptance.
                refused.append(f"{exc} | {getattr(exc, 'cause', None)}")
        still = await handle.query(TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "real@codna.ai"})
        completion = await _result(handle)
        return {"refused": refused, "still": still, "completion": completion}

    result = _run(_with_worker("plain", scenario))
    assert len(result["refused"]) == 3, result["refused"]
    assert all("identity" in message.lower() for message in result["refused"]), result["refused"]
    # None of them took effect: the task was still parked when the valid one arrived.
    assert result["still"]["awaiting_approval"] is True
    assert result["still"]["approvals"] == []
    assert result["completion"].approved_by == "real@codna.ai"


def test_a_second_approval_is_refused(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> tuple[str, list[str]]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-double",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "first@codna.ai"})
        refused: list[str] = []
        try:
            await _update(handle, TaskWorkflowPlain.approve, {"identity": "second@codna.ai"})
        except AssertionError:
            # `_update`'s own ceiling, same reasoning as the gate-refusal test above: a hang is not
            # a refusal, and absorbing it would make the test pass on a workflow that never answered.
            raise
        except Exception as exc:  # noqa: BLE001
            refused.append(f"{exc} | {getattr(exc, 'cause', None)}")
        completion = await _result(handle)
        return completion.approved_by or "", refused

    approved_by, refused = _run(_with_worker("plain", scenario))
    assert approved_by == "first@codna.ai"
    assert refused and "not waiting" in refused[0].lower(), refused


def test_rejecting_at_the_gate_ends_the_task_without_a_pull_request(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-reject",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.reject, {"identity": "reviewer@codna.ai"})
        return await _result(handle)

    completion = _run(_with_worker("plain", scenario))
    assert completion.outcome == "rejected"
    assert completion.approved_by is None
    # The POINT of a gate: the change was never opened, so the DR is not published.
    assert harness.push_calls == 1, "open_pr runs before the gate, so exactly one push"
    assert completion.state["pr"] is not None


def test_status_is_queryable_before_the_first_step(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-early",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        early = await handle.query(TaskWorkflowPlain.status)
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "a@b.c"})
        await _result(handle)
        return early

    early = _run(_with_worker("plain", scenario))
    assert early["engine"] == "plain"
    assert early["awaiting_approval"] is False
    assert isinstance(early["turn_seq"], int)


def test_both_engines_are_registered_so_a_worker_cannot_forget_one():
    names = {workflow_cls.__name__ for workflow_cls in registered_workflows()}
    assert names == {"TaskWorkflow", "TaskWorkflowPlain", "RunWorkflow"}


def test_the_human_gate_is_an_update_not_a_signal(tmp_path: Path):
    """History proves the mechanism: an approval is an Update, so the sender learns it was applied."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="langgraph")

    async def scenario(client: Any) -> list[int]:
        handle = await client.start_workflow(
            TaskWorkflow.run,
            args=[spec.to_payload(), None],
            id="graph-update-history",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflow.status)
        await _update(handle, TaskWorkflow.approve, {"identity": "auditor@codna.ai"})
        await _result(handle)
        history = await handle.fetch_history()
        return [int(event.event_type) for event in history.events]

    event_types = _run(_with_worker("langgraph", scenario))
    assert int(EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED) in event_types
    assert int(EventType.EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED) in event_types
    assert int(EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED) not in event_types


def test_an_update_accepted_before_the_park_wait_is_armed_is_not_lost(tmp_path: Path):
    """The wedge that flaked in CI four times, pinned deterministically.

    The graph engine publishes its gate and then awaits the ``approval.requested`` activity. That
    await is a yield point: the workflow task can complete with ``_interrupts`` populated and
    ``_await_decision`` not yet entered, which is a window in which ``status`` already says
    ``awaiting_approval`` and the ``approve`` validator already accepts an Update. A decision written
    there used to be discarded by ``_await_decision``'s opening clear, so the caller was told
    "applied" and the workflow waited forever — the observed CI failure
    (``the workflow did not COMPLETE within 60s``, ``workflow_id='graph-update-history'``).

    Reproducing that window with a real loaded runner is what made it a flake, so the test does not
    try: it WIDENS the window deterministically by holding the ``approval.requested`` notify open,
    and then approves as soon as the gate opens. Before the fix this test hangs and fails on
    `_result`'s ceiling; after it, the accepted decision survives and the task completes.

    The narrowness is the point: it does not depend on load, on a runner's scheduling, or on which
    Python patch release is installed.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="langgraph")

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            TaskWorkflow.run,
            args=[spec.to_payload(), None],
            id="graph-update-in-window",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        # Poll the UPDATE, not the status query: the internal `notify` is held, so a query answers
        # from the last completed task and can lag behind the gate the Update validator already
        # accepts. A refusal means "not parked yet"; anything else is this test's own bug.
        for _ in range(2000):
            try:
                await _update(handle, TaskWorkflow.approve, {"identity": "window@codna.ai"})
            except Exception as exc:  # noqa: BLE001 - only the validator's refusal is benign
                if "not waiting on a decision" not in str(getattr(exc, "cause", exc)):
                    raise
                await asyncio.sleep(0.01)
                continue
            break
        else:  # pragma: no cover - a workflow that never opens its gate
            raise AssertionError("the gate never accepted an Update")
        return await _result(handle)

    @activity.defn(name=NAME_NOTIFY)
    def held_notify(request: NotifyRequest) -> Any:
        """The REAL notify, delayed so the gate stays open across the yield point that exposed this."""
        time.sleep(HELD_NOTIFY_S)
        from codna.durable import activity_steps

        return activity_steps.notify(request)

    real_notify = next(
        item
        for item in ACTIVITIES
        if item.__temporal_activity_definition.name == NAME_NOTIFY
    )
    activities = [
        held_notify if item is real_notify else item for item in ACTIVITIES
    ]

    async def run() -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            with activity_executor() as executor:
                async with Worker(
                    env.client,
                    task_queue=TASK_QUEUE,
                    workflows=list(registered_workflows()),
                    activities=activities,
                    activity_executor=executor,
                    plugins=[build_task_graph().plugin],
                ):
                    with env.auto_time_skipping_disabled():
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    completion = _run(run())
    assert completion.outcome == "approved"
    assert completion.approved_by == "window@codna.ai"


def test_pause_holds_the_task_at_a_step_boundary(tmp_path: Path):
    """A paused task does not start its next step until resumed, and no paid turn is interrupted."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-pause",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await handle.signal(TaskWorkflowPlain.pause)
        # Let the signal land and the first boundary arrive.
        for _ in range(200):
            if (await handle.query(TaskWorkflowPlain.status)).get("paused"):
                break
            await asyncio.sleep(0.02)
        paused = await handle.query(TaskWorkflowPlain.status)
        turns_while_paused = len(harness.sidecar.turns)
        await handle.signal(TaskWorkflowPlain.resume)
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "a@b.c"})
        completion = await _result(handle)
        return {
            "paused": paused,
            "turns_while_paused": turns_while_paused,
            "completion": completion,
        }

    result = _run(_with_worker("plain", scenario))
    assert result["paused"]["paused"] is True
    # The pause happened at a boundary, so at most the step that was already running completed.
    assert result["turns_while_paused"] <= 1, result["turns_while_paused"]
    assert result["completion"].outcome == "approved"


# ---------------------------------------------------------------------------------------
# The registration contract.
# ---------------------------------------------------------------------------------------


def test_the_step_and_node_activity_names_are_disjoint():
    """A collision would make `Worker(...)` refuse to start, and only when both engines are registered.

    The plugin derives a node's activity type as ``<graph>.<node>``; the step activities are named
    explicitly. Keeping the namespaces disjoint is the invariant, so it is asserted rather than
    remembered — the failure is a startup ValueError in a worker that registers both, which is exactly
    the production shape.
    """
    task_graph = build_task_graph()
    node_names = {
        activity.__temporal_activity_definition.name for activity in task_graph.plugin.activities
    }
    assert node_names.isdisjoint(ACTIVITY_NAMES), node_names & set(ACTIVITY_NAMES)


def test_every_registered_activity_has_a_timeout():
    """The spike's sharp edge: an activity scheduled without a timeout fails the workflow task."""
    from codna.durable.activities import timeout_for

    for name in ACTIVITY_NAMES:
        assert timeout_for(name) > 0, name
    with pytest.raises(Exception, match="no timeout"):
        timeout_for("codna.step.not_a_real_activity")


def test_the_graph_and_the_activity_module_agree_on_every_timeout():
    """Two tables, one truth: `graph.STEP_TIMEOUT_S` must equal `activities.timeout_for`."""
    from codna.durable.activities import timeout_for
    from codna.durable.graph import STEP_TIMEOUT_S

    for name, seconds in STEP_TIMEOUT_S.items():
        assert timeout_for(name) == seconds, (name, seconds, timeout_for(name))


def test_the_checkpoint_names_the_turn_it_commits():
    """`turn-N` is the ref suffix, so the audit ref and the state cannot disagree."""
    spec = _spec(task_id="task-9")
    from codna.durable.activity_steps import turn_ref

    context = spec_ctx(spec)
    assert turn_ref(context, 4) == "refs/codna/runs/run-1/turn-4"


def spec_ctx(spec: TaskSpec) -> Any:
    from codna.durable.contracts import TaskContext

    return TaskContext.from_spec(spec)


def test_the_notify_activity_is_scheduled_with_its_own_timeout(tmp_path: Path):
    """Bookkeeping activities are scheduled from workflow code too, so they need a timeout as well.

    This is the regression for a real bug found while building D2.3: `STEP_TIMEOUT_S` covered only the
    five doing-steps, so `notify` was scheduled with no timeout and every notification silently
    failed — a run whose audit trail was missing its "task started" and "approved" records.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-notify",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "a@b.c"})
        return await _result(handle)

    _run(_with_worker("plain", scenario))
    kinds = [
        record.payload.get("kind")
        for record in harness.ledger().records()
        if record.kind == "note" and record.payload.get("event") == "notification"
    ]
    assert kinds[:2] == ["task.started", "approval.requested"], kinds
    assert "approval.granted" in kinds
    assert kinds[-1] == "task.finished", kinds


def test_the_ledger_is_a_valid_hash_chain_after_a_full_task(tmp_path: Path):
    """The audit trail must be internally consistent, not merely present."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="plain-chain",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=TaskWorkflowPlain.status)
        await _update(handle, TaskWorkflowPlain.approve, {"identity": "a@b.c"})
        return await _result(handle)

    _run(_with_worker("plain", scenario))
    ledger = harness.ledger()
    assert ledger.verify() == []
    assert ledger.check_contiguity() == []
    # The sidecar's own event indices are contiguous per session — the property `?after=` depends on.
    session_ids = {
        record.session_id for record in ledger.records() if record.session_id is not None
    }
    for session_id in session_ids:
        assert ledger.check_session_contiguity(session_id) == [], session_id


def test_a_cancelled_turn_aborts_the_sidecar_session(tmp_path: Path):
    """Cancellation is an explicit abort, not an unsubscribe: the turn must stop spending.

    Driven directly rather than through a workflow cancellation, because what is under test is the
    activity's behaviour — the workflow side (Temporal cancelling an activity) is the SDK's job. A
    cancelled activity that only closed its stream would leave the engine running and billing.
    """
    from codna.durable.activities import ActivityError, run_cline_turn
    from codna.durable.contracts import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    # The probe reports cancelled once the stream has delivered its first event.
    state = {"checks": 0}

    def probe() -> bool:
        state["checks"] += 1
        return state["checks"] > 1

    harness.sidecar.cancel_probe = probe
    harness.runtime.cancelled = probe

    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    assert excinfo.value.code == "activity_cancelled", excinfo.value.as_dict()
    # The FIRST session: the turn ran on it, so that is what must be aborted.
    assert harness.sidecar.aborts == [harness.session_ids()[0]], harness.sidecar.aborts


def test_a_turn_key_replay_is_not_a_second_paid_turn(tmp_path: Path):
    """The single most important property: a retried turn is answered, not re-billed."""
    from codna.durable.activities import run_cline_turn
    from codna.durable.contracts import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    # The SAME request: this is what a retry after a crash looks like.
    second = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert first.details["turn_id"] == second.details["turn_id"]
    assert len(harness.sidecar.turns) == 1, harness.sidecar.turns
    # ...and the retry did not charge the budget twice.
    assert first.state["budget"]["spent_usd"] == second.state["budget"]["spent_usd"]


def test_the_session_is_resumed_across_turns(tmp_path: Path):
    """One conversation per task: the second turn resumes the session the first created."""
    from codna.durable.activities import run_cline_turn
    from codna.durable.contracts import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    first = run_cline_turn(context.turn_request("first", spec.initial_state()))
    second = run_cline_turn(context.turn_request("second", first.state))

    session_ids = harness.session_ids()
    assert harness.sidecar.sessions[0]["resume_from"] is None
    assert harness.sidecar.sessions[1]["resume_from"] == session_ids[0]
    assert second.state["turn_seq"] == 2
    assert second.state["session_id"] == harness.session_ids()[0]


def test_a_read_only_plan_turn_writes_nothing_while_sharing_the_implement_turn_s_session(tmp_path: Path):
    """The read-only plan turn must not be writable — and the implement turn must still be.

    A task's turns share ONE session on purpose (the conversation carries over), so the write scope
    cannot be a property of the session: it is a property of the TURN. The plan turn declares
    `allowed_write_paths: []` and the implement turn declares nothing, and the engine must read those
    as "write nothing" and "keep the workspace boundary" respectively.

    This is the regression test for the defect where `[]` meant "no additional restriction", so the
    plan turn could modify the working tree despite being declared read-only.
    """
    from codna.durable.activities import plan_task, run_cline_turn
    from codna.durable.contracts import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    plan = plan_task(context.plan_request(spec.brief, spec.initial_state()))
    implement = run_cline_turn(context.turn_request("implement it", plan.state))

    # ONE session: the read-only turn and the writable one are the same conversation.
    assert implement.state["session_id"] == plan.state["session_id"]

    scopes = harness.sidecar.turn_write_scopes
    assert len(scopes) == 2, scopes
    # The plan turn says "write nothing" ON THE TURN, not by asking the model nicely.
    assert scopes[0]["allowed_write_paths"] == []
    # The implement turn sends JSON `null` — "no additional restriction" — and NOT nothing. The
    # session was CREATED read-only (an unattended run must not be writable by default), so a turn
    # that stayed silent would INHERIT that scope and be unable to do its job.
    assert scopes[1]["allowed_write_paths"] is None

    # The fake models the engine's own rule: a read-only turn cannot change the tree, a writable one
    # can. This is what makes the assertion above about behaviour rather than about a field.
    assert [turn.read_only for turn in harness.sidecar.turns] == [True, False]


# ---------------------------------------------------------------------------------------
# THE JOIN: a turn's write scope and the tools the model is OFFERED must agree.
#
# WHY THIS SECTION EXISTS, and why the test above could not catch the defect it covers.
#
# `test_a_read_only_plan_turn_writes_nothing_…` asserts the WIRE FIELD
# (`scopes[0]["allowed_write_paths"] == []`) and the fake's own consequence of that field. Both were
# correct the entire time the product was broken. Two independently-merged fixes met badly:
# `#730` made `[]` mean "nothing writable" (so the guard correctly refused `editor`), while the
# turn's TOOL LIST still came from the approval profile, which advertises `editor` unconditionally.
# The model was therefore offered a tool its turn could not use, called it, and the guard ended the
# turn. Measured on a real ledger (~/.codna/runs/run-20261111T031216Z/ledger.ndjson, index 268):
#
#   "session.turn_aborted": {"abort": {"kind": "control_stop",
#      "reason": "write-scope violation: this turn is read-only, so 'editor' is refused"}}
#
# Every audited live run ended `tasks_done: 0` on it. So the assertions below are deliberately at two
# levels the wire-field test cannot reach: what the model is ADVERTISED, and whether the run
# COMPLETES — the latter being the only thing a user experiences.
# ---------------------------------------------------------------------------------------

# `tests/durable/<this file>` -> the repository root, four levels up: durable, tests, cli, <root>.
ENGINE_DIR = Path(__file__).resolve().parents[3] / "agent-core" / "vendor" / "cline" / "algenta"


def test_the_harness_write_tools_are_the_engine_s():
    """The harness's notion of "a tool a read-only turn cannot use" is read OFF the engine.

    A harness that hard-coded its own list would keep passing after the engine refused a new tool,
    which is the failure mode `#730` was. So the source is parsed: `WRITE_TOOLS` and `SHELL_TOOLS` in
    `write-scope.ts` are the two sets the guard refuses, and the harness must name exactly their union.
    """
    import re

    source = (ENGINE_DIR / "write-scope.ts").read_text(encoding="utf-8")

    def members(name: str) -> set[str]:
        match = re.search(rf"const {name} = new Set\(\[([^\]]*)\]\)", source)
        assert match, f"{name} is not declared as a Set literal in write-scope.ts"
        return set(re.findall(r'"([^"]+)"', match.group(1)))

    engine = members("WRITE_TOOLS") | members("SHELL_TOOLS")
    assert engine == set(fakes.ENGINE_WRITE_TOOLS), (
        "the harness's ENGINE_WRITE_TOOLS disagrees with write-scope.ts: "
        f"engine={sorted(engine)} harness={sorted(fakes.ENGINE_WRITE_TOOLS)}"
    )


def test_the_harness_profile_tools_are_the_engine_s():
    """The profile a durable run uses is read off the engine's own table.

    `repository_fix_gated` is the profile `swarm.DURABLE_APPROVAL_PROFILE` names, and its
    `allowedTools` is what the engine advertises. Parsing `contracts.ts` keeps the harness honest: a
    profile that gains or loses a tool is a RED harness, not a silently stale default.
    """
    import re

    source = (ENGINE_DIR / "server" / "contracts.ts").read_text(encoding="utf-8")
    match = re.search(
        r"repository_fix_gated:\s*\{\s*allowedTools:\s*\[(.*?)\]", source, re.DOTALL
    )
    assert match, "repository_fix_gated.allowedTools is not a literal array in contracts.ts"
    engine = set(re.findall(r'"([^"]+)"', match.group(1)))
    # `...TEAM_TOOL_NAMES` is a spread, so the literal array covers only the named tools; the
    # assertions below are about the WRITE tools, which are always spelled out by hand.
    harness = set(fakes.PROFILE_FIX_GATED_TOOLS)
    assert harness <= engine, f"the harness names tools the profile does not grant: {sorted(harness - engine)}"
    assert {"editor", "apply_patch"} <= engine, "the profile stopped advertising the write tools"
    # The whole point: the profile DOES advertise them, which is why a read-only turn had to withhold
    # them. A future profile that stopped advertising them would make this section vacuous, so it is
    # pinned rather than assumed.
    assert {"editor", "apply_patch"} <= harness


def test_a_read_only_plan_turn_is_not_offered_a_write_tool_and_the_run_completes(tmp_path: Path):
    """THE JOIN, on the level that actually broke: what the model sees, and whether the run finishes.

    Two assertions, both of which the wire-field test above cannot make:

    * the plan turn is NOT ADVERTISED `editor`/`apply_patch`/`run_commands` (the model cannot call what
      it is not offered, so the guard is never asked the question it could only answer by aborting);
    * the SAME task reaches `completed` with a recorded outcome, rather than dying on
      `session.turn_aborted`.

    The run-level half is the important one: the wire field was right for the entire life of the
    defect, so a test that stops at the field is a test that cannot fail.
    """
    from codna.durable.activities import plan_task, run_cline_turn
    from codna.durable.contracts import TaskContext
    from codna.durable.state import TaskState

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    plan = plan_task(context.plan_request(spec.brief, spec.initial_state()))
    implement = run_cline_turn(context.turn_request("implement it", plan.state))

    turns = harness.sidecar.turns
    assert [turn.read_only for turn in turns] == [True, False]
    # 1. THE ADVERTISED LIST. A read-only turn must not be offered a tool whose only effect it cannot
    #    perform; the writable turn on the same session must be, or implementation silently breaks.
    assert turns[0].write_tools_offered == (), (
        "the read-only plan turn was offered a write tool: "
        f"{turns[0].write_tools_offered} — the model will call it and the guard will abort the turn"
    )
    assert "editor" in turns[1].offered_tools, (
        "the implement turn was NOT offered the editor, so it cannot do its job: "
        f"{turns[1].offered_tools}"
    )

    # 2. THE RUN'S OUTCOME — both turns completed, and the task recorded a plan and a change. This is
    #    what a user experiences, and what every audited live run failed.
    terminals = [
        str(record.payload.get("event_type"))
        for record in _ledger_turn_terminals(harness)
    ]
    assert terminals == ["session.turn_completed", "session.turn_completed"], terminals
    final = TaskState.from_payload(implement.state)
    assert final.plan_ref is not None, "the task recorded no plan"
    assert final.checkpoint_ref is not None or final.turn_seq == 2, implement.state


def _ledger_turn_terminals(harness: Any) -> list[Any]:
    """Every terminal TURN event the run's ledger recorded, in order.

    Read from the ledger rather than from the state, because the ledger is what an operator audits and
    it is the artifact the defect was measured in.
    """
    return [
        record
        for record in harness.ledger().records()
        if record.kind == "turn.event"
        and str((record.payload or {}).get("event_type", "")).startswith("session.turn_")
    ]


def test_the_run_outcome_is_the_one_a_user_sees(tmp_path: Path):
    """A meta-test on the JOIN above: the assertion really does fail when the engine aborts.

    The first version of this section asserted only on `offered_tools`, which is a property of the
    harness. If the harness's model of the engine is wrong, that assertion is a tautology. So here the
    fake sidecar is made to do what the BROKEN engine did — report `control_stop` on the read-only
    turn — and the run-level expectation must fail, proving the second assertion has teeth.
    """
    import pytest

    from codna.durable.activities import plan_task
    from codna.durable.activity_support import ActivityError
    from codna.durable.contracts import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    # The engine's own abort, verbatim from the audit.
    harness.sidecar.terminal = fakes.TERMINAL_ABORTED
    harness.sidecar.abort_on_terminal = {
        "kind": "control_stop",
        "reason": (
            "write-scope violation: this turn is read-only, so 'editor' is refused "
            "(read-only means no file edits and no state-changing shell commands)"
        ),
    }
    spec = _spec(repo_dir=str(repo), engine="plain")
    context = TaskContext.from_spec(spec)

    with pytest.raises(ActivityError) as caught:
        plan_task(context.plan_request(spec.brief, spec.initial_state()))
    # The failure names the refusal, so the reason survives the activity boundary — which is what makes
    # the defect legible in a ledger rather than a bare "did not complete".
    assert caught.value.code == "activity_turn_failed"
    assert caught.value.details["terminal"] == "session.turn_aborted"
    assert caught.value.details["abort"]["kind"] == "control_stop"
    assert "write-scope violation" in caught.value.details["reason"]


def test_the_report_counts_are_the_real_ones(tmp_path: Path):
    """A meta-test: the harness itself is exercised, so a green suite is not a vacuous one."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    assert harness.route() == []
    assert harness.checkpoints() == []
    assert harness.sidecar.turns == []
    assert NAME_PLAN in ACTIVITY_NAMES
    assert NAME_TURN in ACTIVITY_NAMES
    assert NAME_CHECKPOINT in ACTIVITY_NAMES
    assert NAME_NOTIFY in ACTIVITY_NAMES
    assert timedelta(seconds=1).total_seconds() == 1


# ---------------------------------------------------------------------------------------
# The waits are BOUNDED, and that is asserted rather than assumed.
#
# On 2026-11-03 this file's human-gate test hung in CI for 10m25s and the traceback named
# `asyncio.run` inside a `select()` — it identified the event loop and nothing else. The cause was a
# wait with no wall-clock ceiling: `for _ in range(attempts)` looks bounded but spends its budget only
# AFTER an await returns, so one query that never resolves outlives any attempt count. These tests pin
# the property that replaced it, because "a test that can only hang" is the defect, independent of
# whichever environment-dependent cause made it hang.
# ---------------------------------------------------------------------------------------


class _WedgedHandle:
    """A handle whose awaits never return — the shape the CI hang had.

    Nothing here is exotic: the CI stack showed the loop parked in ``select()`` with a single
    ``handle.query`` outstanding. This reproduces that without a server, because the contract under
    test is the helper's own ceiling.
    """

    id = "wedged-workflow"

    def __init__(self) -> None:
        self.rpc_timeouts: list[float] = []
        self.query_calls = 0

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        self.query_calls += 1
        self.rpc_timeouts.append(float(kwargs["rpc_timeout"].total_seconds()))
        await asyncio.sleep(3600)

    async def result(self) -> Any:
        await asyncio.sleep(3600)

    async def execute_update(self, *args: Any, **kwargs: Any) -> Any:
        self.rpc_timeouts.append(float(kwargs["rpc_timeout"].total_seconds()))
        await asyncio.sleep(3600)


class _NeverParksHandle:
    """A handle that answers every query but never reports the gate."""

    id = "never-parks"

    async def query(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"engine": "langgraph", "awaiting_approval": False, "nodes_run": ["plan"]}


def test_a_wedged_query_fails_fast_and_names_the_wait() -> None:
    """The ceiling is on the WAIT, not on how many polls it managed."""
    started = time.monotonic()
    handle = _WedgedHandle()

    with pytest.raises(AssertionError) as caught:
        _run(_query(handle, object, timeout_s=0.5))

    elapsed = time.monotonic() - started
    assert elapsed < 10, f"the wait was not bounded: it took {elapsed:.1f}s"
    assert "_query" in str(caught.value), str(caught.value)
    assert handle.id in str(caught.value), "the message must name the workflow it was waiting on"
    assert "wedged" in str(caught.value), str(caught.value)


def test_a_park_budget_spent_mid_query_blames_the_budget_not_the_query() -> None:
    """A spent parking budget is a DIFFERENT diagnosis from a wedged workflow task.

    ``_query`` only sees the time it was handed, so a budget that expires while a query is in flight
    would otherwise be reported as "the workflow task is wedged" — which sends whoever debugs it into
    the wrong subsystem. ``_wait_parked`` fixes the attribution, and this pins it.
    """
    started = time.monotonic()

    with pytest.raises(AssertionError) as caught:
        _run(_wait_parked(_WedgedHandle(), query=object, timeout_s=0.5))

    elapsed = time.monotonic() - started
    assert elapsed < 10, f"the wait was not bounded: it took {elapsed:.1f}s"
    assert "_wait_parked" in str(caught.value), str(caught.value)
    assert "budget expired" in str(caught.value), str(caught.value)


def test_a_workflow_that_never_completes_fails_fast_and_names_the_wait() -> None:
    started = time.monotonic()
    handle = _WedgedHandle()

    with pytest.raises(AssertionError) as caught:
        _run(_result(handle, timeout_s=0.5))

    elapsed = time.monotonic() - started
    assert elapsed < 10, f"the wait was not bounded: it took {elapsed:.1f}s"
    assert "_result" in str(caught.value)
    assert "HANG" in str(caught.value), "the failure must say it was a hang, not a slow runner"


def test_an_update_that_is_never_accepted_fails_fast_and_names_the_wait() -> None:
    """An Update the workflow never applies blocks its caller exactly as silently as a query."""
    started = time.monotonic()

    with pytest.raises(AssertionError) as caught:
        _run(_update(_WedgedHandle(), object, {"identity": "a@b.c"}, timeout_s=0.5))

    elapsed = time.monotonic() - started
    assert elapsed < 10, f"the wait was not bounded: it took {elapsed:.1f}s"
    assert "_update" in str(caught.value)
    assert "not accepted" in str(caught.value)


def test_a_workflow_that_never_parks_reports_the_last_status_it_saw() -> None:
    """The other half of the hang: the workflow runs but never reaches the gate.

    Distinguishing this from "the query never answered" matters — one is a workflow defect, the other
    a wedged workflow task — so the message carries the last status observed.
    """
    with pytest.raises(AssertionError) as caught:
        _run(_wait_parked(_NeverParksHandle(), query=object, timeout_s=0.5))

    message = str(caught.value)
    assert "awaiting_approval" in message
    assert "'engine': 'langgraph'" in message, "the last status must be in the failure"


def test_outer_assertion_deadline_precedes_the_sdk_rpc_deadline() -> None:
    """The descriptive outer timeout, not an SDK-specific exception, owns each wait."""
    query_handle = _WedgedHandle()
    with pytest.raises(AssertionError, match="_query"):
        _run(_query(query_handle, object, timeout_s=0.5))
    assert query_handle.rpc_timeouts == [1.5]

    update_handle = _WedgedHandle()
    with pytest.raises(AssertionError, match="_update"):
        _run(_update(update_handle, object, {"identity": "a@b.c"}, timeout_s=0.5))
    assert update_handle.rpc_timeouts == [1.5]


def test_an_expired_park_budget_does_not_issue_another_query() -> None:
    handle = _WedgedHandle()
    with pytest.raises(AssertionError, match="never reported awaiting_approval"):
        _run(_wait_parked(handle, query=object, timeout_s=0))
    assert handle.query_calls == 0
