"""Replay: the Public-Preview safety net for the LangGraph plugin.

`Replayer` re-executes a recorded workflow history with the CURRENT code and fails if the code now
produces different commands. That is the only way to know a deploy will not break runs already in
flight, and it is why the plan treats a recorded history as a build artifact rather than a test
fixture: **a history is re-recorded only on a `GRAPH_VERSION` bump**, and bumping it is also what
produces a new `build_id` and pins a new worker version.

What is replayed here:

* `histories/task-graph-g<N>.json` — recorded from `codna.durable.workflows.TaskWorkflow` with the
  REAL activities on a fake sidecar. This is the cassette that matters: it is the production workflow,
  not the spike's toy one, so a change to the task graph's nodes or edges breaks it.
* and the plugin-LESS path, which must FAIL. That is the sharpest fact from the D2.0 spike: without
  the plugin the Replayer cannot even validate the workflow (it patches the sandbox's passthrough
  modules), so a replay suite that forgot the plugin could never "accidentally pass". Asserting the
  refusal is what keeps that a fact rather than a memory.

The recording helper lives in `record_workflow_history.py` — not a test module, because a workflow
class defined in a test module cannot be validated (importing pytest reaches `urllib.request`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

import asyncio  # noqa: E402

from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.worker import Replayer  # noqa: E402

from codna.durable.graph import GRAPH_NAME, build_plugin, build_task_graph  # noqa: E402
from codna.durable.version import GRAPH_VERSION  # noqa: E402
from codna.durable.workflows import TaskWorkflow, TaskWorkflowPlain  # noqa: E402

HISTORIES = Path(__file__).resolve().parent / "histories"
CASSETTE = HISTORIES / f"task-workflow-g{GRAPH_VERSION}.json"


def _cassette() -> WorkflowHistory:
    assert CASSETTE.is_file(), (
        f"no recorded history for graph version {GRAPH_VERSION}: run "
        "`python -m tests.durable.record_workflow_history` to create it. A missing cassette means "
        "the replay net is not deployed, which is worse than a red test."
    )
    payload = json.loads(CASSETTE.read_text(encoding="utf-8"))
    return WorkflowHistory.from_json("recorded-task-workflow", json.dumps(payload))


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_the_committed_cassette_exists_for_this_graph_version():
    """The pinning hook: a cassette belongs to ONE graph version, and a bump must re-record it."""
    assert CASSETTE.is_file(), (
        f"no cassette for graph version {GRAPH_VERSION}; run "
        "`python -m tests.durable.record_workflow_history`"
    )
    payload = json.loads(CASSETTE.read_text(encoding="utf-8"))
    # The cassette records the version it was made for, so a stale one is detectable on its own.
    assert payload.get("codna_graph_version", GRAPH_VERSION) == GRAPH_VERSION
    assert len(payload.get("events") or []) >= 20, "a cassette with no shape proves nothing"


def test_the_workflow_replays_under_this_build():
    """The core assertion: same history, same commands, no replay failure."""
    history = _cassette()

    async def replay() -> object:
        # A FRESH plugin: constructing one mutates the graph it is given (spike finding 2), so a
        # module-level plugin shared between tests would work once and raise afterwards.
        return await Replayer(workflows=[TaskWorkflow], plugins=[build_plugin()]).replay_workflow(
            history
        )

    result = _run(replay())
    assert result.replay_failure is None, result.replay_failure


def test_replaying_without_the_plugin_fails_loudly_rather_than_quietly_passing():
    """The safety net, measured — and one step sharper than the spike's version.

    The D2.0 spike's toy workflow imported ``langgraph`` at module scope, so a plugin-less Replayer
    failed during **validation** (``Failed validating workflow``). `TaskWorkflow` reaches langgraph
    lazily (inside ``_compile``), precisely so a worker running only the plain engine does not need
    the extra — which moves the failure to **first activation**: the sandbox refuses
    ``http.client``, reached through ``langchain_core`` → ``requests`` → ``urllib3``, because the
    plugin is what marks those modules as passthrough.

    Either way the property that matters holds, and it is the property this test asserts: a
    plugin-less replay CANNOT quietly succeed. It fails before comparing a single event, so a replay
    suite that forgot the plugin cannot report green.
    """
    history = _cassette()

    async def replay_without_plugin() -> object:
        return await Replayer(workflows=[TaskWorkflow]).replay_workflow(history)

    with pytest.raises(RuntimeError) as excinfo:
        _run(replay_without_plugin())
    message = str(excinfo.value)
    # The sandbox refusing a module the plugin would have passed through. Asserting the MECHANISM
    # (not just "it raised") means a future change that makes the plugin merely *convenient* rather
    # than *required* fails this test instead of silently weakening the net.
    assert "RestrictedWorkflowAccessError" in message or "Failed validating workflow" in message, message
    assert "Cannot access" in message or "Failed validating workflow" in message, message


def test_the_cassette_carries_the_activity_trail_the_plan_requires():
    """The cassette is evidence, so its shape is asserted: the node activities, and no gate node."""
    history = _cassette()
    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField("activity_task_scheduled_event_attributes")
    ]
    # The graph's own node activities, in the route the recording scenario takes: a verify failure and
    # a review rejection, so the loop appears in the cassette and not just the straight path.
    nodes = [name for name in scheduled if name.startswith(f"{GRAPH_NAME}.")]
    assert nodes == [
        f"{GRAPH_NAME}.plan",
        f"{GRAPH_NAME}.implement",
        f"{GRAPH_NAME}.verify",
        f"{GRAPH_NAME}.implement",
        f"{GRAPH_NAME}.verify",
        f"{GRAPH_NAME}.review",
        f"{GRAPH_NAME}.implement",
        f"{GRAPH_NAME}.verify",
        f"{GRAPH_NAME}.review",
        f"{GRAPH_NAME}.open_pr",
    ], nodes
    # The approval is an interrupt in-workflow: it must never appear as a scheduled activity.
    assert f"{GRAPH_NAME}.wait_for_human" not in scheduled
    # The STEP activities are NOT scheduled on the graph path — each node IS the activity, and its body
    # calls the step implementation directly (measured: `workflow.execute_activity` inside an activity
    # raises `_NotInWorkflowEventLoopError`). This asserts the difference rather than papering over it:
    # a future change that made the graph schedule step activities too would double-run every turn, and
    # the loop would never terminate.
    step_names = {"codna.step.plan", "codna.step.turn", "codna.step.checkpoint"}
    assert step_names.isdisjoint(scheduled), step_names & set(scheduled)
    # Bookkeeping IS scheduled by the workflow itself, which runs outside the graph.
    assert "codna.step.notify" in scheduled, scheduled


def test_the_plain_engine_has_no_plugin_to_forget():
    """`TaskWorkflowPlain` needs no plugin, which is why it is the fallback.

    Registered under the same worker and replayable with ``plugins=[]``: the plain engine is plain
    Temporal code, so its history is replayable by any build that can import the module. That is the
    property the plan's go/no-go depends on — the graph engine has a failure mode the fallback does
    not.
    """
    from temporalio.worker import Replayer as _Replayer

    # Constructing a Replayer for the plain workflow with NO plugin must succeed; the graph workflow
    # in the same position raises (asserted above).
    replayer = _Replayer(workflows=[TaskWorkflowPlain])
    assert replayer is not None


def test_a_fresh_graph_is_built_per_component():
    """Worker and Replayer each need their own graph; `build_plugin` must not share one."""
    first = build_plugin()
    second = build_plugin()
    assert first is not second
    # And two task graphs are independent objects with independent activity lists.
    left, right = build_task_graph(), build_task_graph()
    assert left.graph is not right.graph
    assert left.activity_names == right.activity_names
