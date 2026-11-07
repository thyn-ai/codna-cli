"""D2.0 spike — the four facts, executed against the real libraries rather than asserted in prose.

Every test here needs the `codna[durable]` extra, so the module is gated with
``pytest.importorskip`` and the base suite stays green without it. Run WITH the extra these pass
rather than skip.

The four facts, and where each is proven:

1. ``test_activity_nodes_become_real_activities`` — a `StateGraph` node marked
   ``execute_in="activity"`` shows up in workflow history as an ``ActivityTaskScheduled`` under the
   plugin's own type name, while an ``execute_in="workflow"`` node does not.
2. ``test_replayer_replays_a_recorded_history`` / ``test_replayer_without_the_plugin_cannot_validate``
   — the same history replays clean *with* the plugin and cannot even be validated *without* it.
3. ``test_command_resume_without_a_checkpointer_raises`` / ``test_inmemory_saver_is_sufficient`` —
   `Command(resume=...)` needs a checkpointer, and `InMemorySaver` is enough inside a workflow.
4. ``test_interrupt_and_update_resume_carries_the_approver_identity`` — `interrupt()` in a
   workflow-bound node suspends the run until an Update resumes it, carrying the approver identity.

Three sharp edges the spike found get their own tests, because each is a production bug if it
regresses: plugin construction MUTATES the graph (Worker and Replayer need separate graphs), an
activity node without ``start_to_close_timeout`` fails the workflow task, and a workflow module may
not import anything that touches `urllib.request` (which `pytest` does).

``WorkflowEnvironment.start_time_skipping()`` downloads/caches a test server on first use, so this
suite needs network access that one time. If that is unavailable the tests FAIL with the SDK's own
error rather than skipping: a silently skipped durability suite is the exact failure this file
exists to prevent.
"""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip(
    "codna.durable.graph", reason="the `durable` extra cannot build the task graph"
)

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.contrib.langgraph import LangGraphPlugin  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402

from codna.durable.graph import (  # noqa: E402
    GRAPH_NAME,
    NODE_EXECUTION,
    _plugin_activity_options,
    build_plugin,
    build_task_graph,
    task_state_from_graph_state,
)
from codna.durable.routing import after_plan, after_review, after_verify  # noqa: E402
from codna.durable.state import TaskState, VerifyRef  # noqa: E402
from tests.durable.spike_workflows import (  # noqa: E402
    RUN_TIMEOUT,
    STUB_NODES,
    TASK_QUEUE,
    SpikeTaskWorkflow,
    initial_state,
    interrupt_payloads,
)

pytestmark = pytest.mark.usefixtures()


async def _with_environment(scenario: Callable[[Any, Any], Awaitable[Any]]) -> Any:
    env = await WorkflowEnvironment.start_time_skipping()
    try:
        return await scenario(env, env.client)
    finally:
        await env.shutdown()


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------------------
# A module-level stub for the "no timeout" graph: the plugin refuses closures.
# ---------------------------------------------------------------------------------------


async def _no_timeout_plan(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "plan_ref": "refs/plans/x"}


# ---------------------------------------------------------------------------------------
# Fact 1: node → activity mapping.
# ---------------------------------------------------------------------------------------


def test_activity_nodes_become_real_activities_and_workflow_nodes_do_not():
    async def scenario(env: Any, client: Any) -> dict[str, Any]:
        task_graph = build_task_graph(nodes=STUB_NODES)
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[SpikeTaskWorkflow], plugins=[task_graph.plugin]
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run,
                initial_state(),
                id="spike-mapping",
                task_queue=TASK_QUEUE,
                run_timeout=RUN_TIMEOUT,
            )
            await _wait_for_approval(handle)
            await handle.execute_update(
                SpikeTaskWorkflow.approve, {"identity": "alice", "approved": True}
            )
            final = await handle.result()
            history = await handle.fetch_history()
        return {"final": final, "history": history, "activity_names": task_graph.activity_names}

    result = _run(_with_environment(scenario))

    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in result["history"].events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    for node in ("plan", "implement", "verify", "review", "open_pr"):
        assert f"{GRAPH_NAME}.{node}" in scheduled, f"{node} did not become an activity: {scheduled}"
    # The approval node never crosses the activity boundary: `interrupt()` needs a running graph,
    # and a suspension is not work to retry.
    assert f"{GRAPH_NAME}.wait_for_human" not in scheduled
    # The module's declared mapping agrees with what actually ran.
    assert set(result["activity_names"]) == {f"{GRAPH_NAME}.{name}" for name, where in NODE_EXECUTION.items() if where == "activity"}
    # The workflow completed and returned a clean TaskState payload.
    assert result["final"]["review_verdict"] == "approved"
    assert result["final"]["plan_ref"] == "refs/plans/task-1"
    assert result["final"]["pr"]["number"] == 1


# ---------------------------------------------------------------------------------------
# Fact 4: interrupt → Update resume.
# ---------------------------------------------------------------------------------------


def test_interrupt_and_update_resume_carries_the_approver_identity():
    async def scenario(env: Any, client: Any) -> dict[str, Any]:
        task_graph = build_task_graph(nodes=STUB_NODES)
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[SpikeTaskWorkflow], plugins=[task_graph.plugin]
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run,
                initial_state(),
                id="spike-approval",
                task_queue=TASK_QUEUE,
                run_timeout=RUN_TIMEOUT,
            )
            status = await _wait_for_approval(handle)
            decision = await handle.execute_update(
                SpikeTaskWorkflow.approve, {"identity": "alice@example.com", "approved": True}
            )
            final = await handle.result()
            approvals = await handle.query(SpikeTaskWorkflow.approvals)
        return {"status": status, "decision": decision, "final": final, "approvals": approvals}

    result = _run(_with_environment(scenario))

    # Parked, and `status` says exactly which decision it waits for — the IDE's "Needs input" card
    # renders from this.
    assert result["status"]["awaiting_approval"] is True
    assert result["status"]["interrupts"], "the interrupt payload must be queryable while parked"
    assert result["status"]["interrupts"][0]["kind"] == "approve"
    assert result["status"]["interrupts"][0]["task_id"] == "task-1"

    # The Update's value is the resume value, and the identity is recorded in WORKFLOW state.
    assert result["decision"] == {"identity": "alice@example.com", "approved": True}
    assert result["approvals"] == ["alice@example.com"]
    # ...while the graph state stayed a clean TaskState payload (no invented keys).
    assert result["final"]["pending_approval"] is None
    assert "approval" not in result["final"]


def test_a_second_approval_update_is_refused():
    """An Update that arrives twice is applied exactly once: the approver list has one entry."""

    async def scenario(env: Any, client: Any) -> tuple[int, str]:
        task_graph = build_task_graph(nodes=STUB_NODES)
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[SpikeTaskWorkflow], plugins=[task_graph.plugin]
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run, initial_state(), id="spike-double-approve",
                task_queue=TASK_QUEUE, run_timeout=RUN_TIMEOUT,
            )
            await _wait_for_approval(handle)
            await handle.execute_update(SpikeTaskWorkflow.approve, {"identity": "a", "approved": True})
            message = ""
            try:
                await handle.execute_update(SpikeTaskWorkflow.approve, {"identity": "b", "approved": True})
            except Exception as exc:  # noqa: BLE001 - the SDK's own refusal type varies by timing
                message = str(exc)
            await handle.result()
            approvals = await handle.query(SpikeTaskWorkflow.approvals)
        return len(approvals), message

    applied, message = _run(_with_environment(scenario))
    # The invariant that matters: the second Update did not take effect. It is refused either by the
    # workflow's own handler ("already been approved") or by the server once the run is closing
    # ("update … not found"); both mean "not applied", and the approver list proves it.
    assert applied == 1
    assert message, "a duplicate approval must be reported, not silently ignored"


# ---------------------------------------------------------------------------------------
# Fact 2: the plugin is required on the Worker AND on the Replayer.
# ---------------------------------------------------------------------------------------


def _record_history(workflow_id: str) -> Any:
    async def scenario(env: Any, client: Any) -> WorkflowHistory:
        task_graph = build_task_graph(nodes=STUB_NODES)
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[SpikeTaskWorkflow], plugins=[task_graph.plugin]
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run, initial_state(), id=workflow_id,
                task_queue=TASK_QUEUE, run_timeout=RUN_TIMEOUT,
            )
            await _wait_for_approval(handle)
            await handle.execute_update(
                SpikeTaskWorkflow.approve, {"identity": "replayer", "approved": True}
            )
            await handle.result()
            return await handle.fetch_history()

    return _run(_with_environment(scenario))


def test_replayer_replays_a_recorded_history():
    history = _record_history("spike-replay")
    restored = WorkflowHistory.from_json("spike-replay", history.to_json())
    assert len(restored.events) >= 10
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in restored.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    # The first replay evidence: the five activity nodes, in route order.
    assert scheduled == [
        f"{GRAPH_NAME}.plan",
        f"{GRAPH_NAME}.implement",
        f"{GRAPH_NAME}.verify",
        f"{GRAPH_NAME}.review",
        f"{GRAPH_NAME}.open_pr",
    ]

    async def replay() -> Any:
        replayer = Replayer(
            workflows=[SpikeTaskWorkflow],
            plugins=[build_plugin(nodes=STUB_NODES)],  # fresh graph: the plugin mutates what it gets
        )
        return await replayer.replay_workflow(restored)

    result = _run(replay())
    assert result.replay_failure is None


def test_replayer_replays_the_committed_cassette():
    """The committed history is the regression net: it must replay under THIS graph version.

    Recorded by `python -m tests.durable.record_history` right after the graph was written, and
    re-recorded ONLY on a `GRAPH_VERSION` bump. If this fails while the in-run replay above passes,
    the graph's commands changed without a version bump — which in production means runs already in
    flight on the old worker version can no longer be replayed by this build.
    """
    from codna.durable.version import GRAPH_VERSION

    cassette = Path(__file__).resolve().parent / "histories" / f"task-graph-g{GRAPH_VERSION}.json"
    assert cassette.is_file(), (
        f"no recorded history for graph version {GRAPH_VERSION}: run "
        "`python -m tests.durable.record_history` to create it"
    )
    payload = json.loads(cassette.read_text(encoding="utf-8"))
    history = WorkflowHistory.from_json("recorded-task-graph", json.dumps(payload))

    # The cassette is what the spike observed: 43 events, five activity types in route order.
    assert len(history.events) == 43
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    assert scheduled == [
        f"{GRAPH_NAME}.plan",
        f"{GRAPH_NAME}.implement",
        f"{GRAPH_NAME}.verify",
        f"{GRAPH_NAME}.review",
        f"{GRAPH_NAME}.open_pr",
    ]
    # The approval arrived as an Update, which is the mechanism the human gate uses.
    event_types = [event.event_type for event in history.events]
    assert EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED in event_types
    assert EventType.EVENT_TYPE_WORKFLOW_EXECUTION_COMPLETED in event_types

    async def replay() -> Any:
        return await Replayer(
            workflows=[SpikeTaskWorkflow], plugins=[build_plugin(nodes=STUB_NODES)]
        ).replay_workflow(history)

    result = _run(replay())
    assert result.replay_failure is None


def test_replayer_without_the_plugin_cannot_even_validate_the_workflow():
    """The safety net, measured — and sharper than the plan assumed.

    Without the plugin the Replayer cannot *validate* the workflow at all, let alone reproduce it.
    The plugin does two jobs: it registers the graph for the run, and it patches the workflow
    sandbox's passthrough modules (``langchain``/``langgraph``/``langchain_core``). Without it, the
    sandbox refuses the workflow's own imports and the replay dies with
    ``RuntimeError: Failed validating workflow <T>`` before a single event is compared.

    So a plugin-less replay suite cannot "accidentally pass": it fails at the start. That is the
    property that makes the committed-history suite meaningful, and it is why `replay_history`
    always builds a plugin rather than accepting none.
    """
    history = _record_history("spike-replay-noplugin")
    restored = WorkflowHistory.from_json("spike-replay-noplugin", history.to_json())

    async def replay_without_plugin() -> Any:
        return await Replayer(workflows=[SpikeTaskWorkflow]).replay_workflow(restored)

    with pytest.raises(RuntimeError) as excinfo:
        _run(replay_without_plugin())
    assert "Failed validating workflow" in str(excinfo.value)

    # With the plugin, the same history is clean (asserted in full by the test above).
    async def replay_with_plugin() -> Any:
        return await Replayer(
            workflows=[SpikeTaskWorkflow], plugins=[build_plugin(nodes=STUB_NODES)]
        ).replay_workflow(restored)

    assert _run(replay_with_plugin()).replay_failure is None


def test_a_history_recorded_for_a_different_graph_version_is_detectable():
    """The pinning hook: a history carries the build id it was recorded under.

    `GRAPH_VERSION` ⇒ `build_id` ⇒ worker version, so a history whose build id differs from this
    build is one this code is not supposed to replay. That is what makes the committed histories in
    `cli/tests/durable/histories/` a safety net rather than a snapshot: bumping the graph version is
    supposed to require re-recording them.
    """
    from codna.durable.config import GRAPH_VERSION, build_id_for

    assert build_id_for("0.2.94", GRAPH_VERSION) == f"codna-0.2.94-g{GRAPH_VERSION}"
    assert build_id_for("0.2.94", GRAPH_VERSION + 1) != build_id_for("0.2.94", GRAPH_VERSION)


def test_the_plugin_mutates_its_graph_so_worker_and_replayer_need_separate_ones():
    shared = build_task_graph(nodes=STUB_NODES)
    assert shared.plugin.activities  # the first plugin registered activities

    # The node metadata was consumed: a second plugin over the SAME graph objects refuses to start.
    with pytest.raises(ValueError, match="execute_in"):
        LangGraphPlugin(graphs={"task": shared.graph})

    # Two independent builds are both fine — that is what `build_task_graph` returning a pair buys.
    assert build_task_graph(nodes=STUB_NODES).plugin.activities
    assert build_task_graph(nodes=STUB_NODES).plugin.activities


def test_an_activity_node_without_a_timeout_fails_the_workflow_task():
    async def scenario(env: Any, client: Any) -> str:
        bare = StateGraph(dict)
        bare.add_node("plan", _no_timeout_plan, metadata={"execute_in": "activity"})
        bare.add_edge(START, "plan")
        bare.add_edge("plan", END)
        plugin = LangGraphPlugin(graphs={GRAPH_NAME: bare})  # no default_activity_options
        # A SHORT run timeout: the failure mode is "the workflow task fails, the run retries the task
        # forever", so the test must bound it rather than hang for the default hour.
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[SpikeTaskWorkflow], plugins=[plugin]
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run, initial_state(), id="spike-no-timeout",
                task_queue=TASK_QUEUE, run_timeout=timedelta(seconds=20),
            )
            try:
                await handle.result()
            except Exception as exc:  # noqa: BLE001 - the SDK wraps the ValueError
                return str(getattr(exc, "cause", exc)) + " | " + str(exc)
        return "completed"

    message = _run(_with_environment(scenario))
    # Either the workflow task failed with the SDK's own message (which the worker logs), or the run
    # timed out because the task keeps retrying — both are "this graph cannot run". The production
    # graph, by contrast, always supplies the timeout.
    assert "start_to_close_timeout" in message or "timed out" in message.lower()
    assert "start_to_close_timeout" in _plugin_activity_options()


def test_plugin_activity_options_carry_a_bounded_retry_policy() -> None:
    """The graph engine must bound retries, exactly as the plain engine does.

    MEASURED DEFECT. The plugin activity options carried only the two timeouts, so the plugin path got
    Temporal's DEFAULT retry policy — no attempt cap. A `codna.task.plan` activity that could never
    succeed (the sidecar refused `create_session` with 403 `policy_verdict_unavailable`) retried 27
    times over 30 minutes: the run's ledger stayed at four records, the thread said "Nothing has
    happened in this run yet", and `codna durable control --run ... stop` reported `stopped: true`
    without ending it. A deterministic failure has to stop on its bound or the run's "alive or dead"
    signal is meaningless.
    """
    from codna.durable.graph import ACTIVITY_RETRIES, _plugin_activity_options
    from codna.durable.workflow_support import _retry_policy

    policy = _plugin_activity_options().get("retry_policy")
    assert policy is not None, "the graph engine must not fall back to Temporal's default retry policy"
    assert policy.maximum_attempts == ACTIVITY_RETRIES
    # The two engines must agree: a graph run and a plain run failing the same way end the same way.
    assert policy.maximum_attempts == _retry_policy().maximum_attempts


# ---------------------------------------------------------------------------------------
# Fact 3: InMemorySaver is mandatory for resume.
# ---------------------------------------------------------------------------------------


async def _approval_only_node(state: dict[str, Any]) -> dict[str, Any]:
    interrupt({"kind": "approve", "task_id": state["task_id"]})
    return {**state, "pending_approval": None}


def _single_node_graph(checkpointer: Any) -> Any:
    node_graph = StateGraph(dict)
    node_graph.add_node("wait_for_human", _approval_only_node, metadata={"execute_in": "workflow"})
    node_graph.add_edge(START, "wait_for_human")
    node_graph.add_edge("wait_for_human", END)
    return node_graph.compile(checkpointer=checkpointer)


def test_command_resume_without_a_checkpointer_raises():
    compiled = _single_node_graph(None)
    config = {"configurable": {"thread_id": "no-saver"}}
    first = asyncio.run(compiled.ainvoke(initial_state(), config))
    assert interrupt_payloads(first), "the node parks even without a checkpointer"
    with pytest.raises(RuntimeError, match="checkpointer"):
        asyncio.run(compiled.ainvoke(Command(resume={"identity": "x", "approved": True}), config))


def test_inmemory_saver_is_sufficient_to_resume():
    compiled = _single_node_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "with-saver"}}
    first = asyncio.run(compiled.ainvoke(initial_state(), config))
    assert interrupt_payloads(first)
    final = asyncio.run(compiled.ainvoke(Command(resume={"identity": "bob", "approved": True}), config))
    assert final["pending_approval"] is None


def test_a_resume_value_without_an_identity_is_refused():
    """The gate in the production node: an approval must name who approved."""
    from codna.durable import graph as graph_module

    production = StateGraph(dict)
    production.add_node("wait_for_human", graph_module.wait_for_human, metadata={"execute_in": "workflow"})
    production.add_edge(START, "wait_for_human")
    production.add_edge("wait_for_human", END)
    app = production.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "production-no-identity"}}
    asyncio.run(app.ainvoke(initial_state(), cfg))
    with pytest.raises(Exception, match="identity"):
        asyncio.run(app.ainvoke(Command(resume={"approved": True}), cfg))


def test_the_production_approval_node_accepts_a_well_formed_identity():
    from codna.durable import graph as graph_module

    production = StateGraph(dict)
    production.add_node("wait_for_human", graph_module.wait_for_human, metadata={"execute_in": "workflow"})
    production.add_edge(START, "wait_for_human")
    production.add_edge("wait_for_human", END)
    app = production.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "production-ok"}}
    asyncio.run(app.ainvoke(initial_state(), cfg))
    final = asyncio.run(
        app.ainvoke(Command(resume={"identity": "alice@example.com", "approved": True}), cfg)
    )
    assert final["pending_approval"] is None
    # Only TaskState fields: the identity lives in workflow state + the ledger, not in graph state.
    assert "approval" not in final
    assert task_state_from_graph_state(final).run_id == "run-1"


def test_the_production_approval_node_refuses_a_rejection_without_looping():
    from codna.durable import graph as graph_module

    production = StateGraph(dict)
    production.add_node("wait_for_human", graph_module.wait_for_human, metadata={"execute_in": "workflow"})
    production.add_edge(START, "wait_for_human")
    production.add_edge("wait_for_human", END)
    app = production.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "production-reject"}}
    asyncio.run(app.ainvoke(initial_state(), cfg))
    with pytest.raises(Exception, match="rejected"):
        asyncio.run(app.ainvoke(Command(resume={"identity": "alice", "approved": False}), cfg))


# ---------------------------------------------------------------------------------------
# Parity: both engines route identically.
# ---------------------------------------------------------------------------------------


def test_activity_and_plain_engines_agree_on_every_routing_decision():
    """The graph's edges and the plain workflow's direct calls are the SAME functions.

    The verify-fails-once loop the plan names, walked through the production routers:
    plan → implement → verify(fail) → implement → verify(pass) → review → open_pr.
    """
    state = TaskState.from_payload(initial_state())
    assert after_plan(state) == "plan"  # no plan yet → re-plan
    planned = TaskState.from_payload({**state.to_payload(), "plan_ref": "refs/plans/x"})
    assert after_plan(planned) == "implement"

    failed = TaskState.from_payload(
        {**planned.to_payload(), "verify_ref": {"ref": "v1", "passed": False, "command_digest": "c" * 64}}
    )
    assert after_verify(failed) == "implement"

    passed = TaskState.from_payload(
        {
            **planned.to_payload(),
            "verify_ref": {"ref": "v2", "passed": True, "command_digest": "d" * 64},
            "attempts": {"implement": 2},
        }
    )
    assert after_verify(passed) == "review"
    assert after_review(TaskState.from_payload({**passed.to_payload(), "review_verdict": "rejected"})) == "implement"
    assert after_review(TaskState.from_payload({**passed.to_payload(), "review_verdict": "approved"})) == "open_pr"

    exhausted = TaskState.from_payload(
        {
            **passed.to_payload(),
            "budget": {"max_cost_usd": 1.0, "spent_usd": 2.0, "max_tokens": None,
                       "spent_tokens": 0, "unpriced_tokens": 0},
        }
    )
    assert after_review(exhausted) == "wait_for_human"


def test_the_stub_activity_output_matches_the_verify_reference_shape():
    """D2.2's activity cannot drift from the state type: the spike's stub already round-trips."""
    assert VerifyRef.from_payload(
        {"ref": "verify/1", "passed": True, "command_digest": "a" * 64}
    ) == VerifyRef(ref="verify/1", passed=True, command_digest="a" * 64)


async def _wait_for_approval(handle: Any, *, attempts: int = 300) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for _ in range(attempts):
        last = await handle.query(SpikeTaskWorkflow.status)
        if last.get("interrupts"):
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"workflow never parked on an approval; last status={last!r}")
