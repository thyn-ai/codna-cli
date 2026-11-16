"""THE replay evidence for the stop path: a cancelled run replays with no nondeterminism error.

── WHY THIS FILE EXISTS ────────────────────────────────────────────────────────────────────────

Before this file there was NO replay test covering `RunWorkflow`'s stop path at all. The cancel
machinery sends `SignalExternalWorkflow` to a child, and a divergent command sequence there is not a
failed test — it is a run wedged forever, because Temporal retries the failing workflow task against
the same history. `stop_delivery.py` argues that the delivery is replay-safe because the send is
driven from the MAIN coroutine at a boundary a replayed execution also visits, rather than spawned by
`asyncio.create_task` from a synchronous signal handler. This file is that argument's evidence.

── WHAT IS ACTUALLY DONE, AND WHY IT IS NOT A SHIM ────────────────────────────────────────────

    30|The test runs the PRODUCTION `RunWorkflow` (real `TaskWorkflowPlain` child, real activities, a fake
sidecar and test runner) against a real Temporal test server until it is PARKED on the approval, sends
the exact Signal `POST /runs/:id/cancel` sends, waits for the run to finish `stopped`, and then takes
the HISTORY the server RECORDED for that execution and replays it through the SDK's `Replayer` with
the current code. A replay failure is the SDK's own `NondeterminismError` path, not an assertion this
test invents.

The history is also replayed through the SAME code twice: once as recorded (`raise_on_replay_failure`
defaults to True) and once while asserting the replay RESULT carries no failure. Both are the SDK's
own check; the second also proves the replayer got far enough to compare commands rather than failing
to validate the workflow.

── MODULE-LEVEL IMPORTS ───────────────────────────────────────────────────────────────────────

The offline CI job runs `pytest` WITHOUT the durable extra, and a module-level `temporalio` import
aborts COLLECTION for the whole job. Gated exactly as every sibling file in this directory is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.worker import Replayer  # noqa: E402

from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.run_api_http import SIGNALS  # noqa: E402
from codna.durable.workflows import RunWorkflow, registered_workflows  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.test_stop_delivery import (  # noqa: E402
    _status_query,
    _wait_parked_child,
)
from tests.durable.test_task_workflow_time_skipping import (  # noqa: E402
    RUN_TIMEOUT,
    TASK_QUEUE,
    _result,
    _run,
    _spec,
    _with_worker,
)

#: The child the run starts. Named here because the replay must see the SAME workflow registered.
CHILD_WORKFLOW_ID = "task-run-1-task-1"


def _payload(repo: Path, spec: TaskSpec) -> dict[str, Any]:
    """The minimal valid `RunWorkflow` input: one task, the engine the spec names."""
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [spec.to_payload()],
        "engine": spec.engine,
        "max_parallel_tasks": 1,
    }


def _replay(history: WorkflowHistory) -> Any:
    """Replay ``history`` through the SDK with the CURRENT code, returning the SDK's own result."""

    async def replay() -> Any:
        replayer = Replayer(workflows=list(registered_workflows()))
        return await replayer.replay_workflow(history, raise_on_replay_failure=True)

    return _run(replay())


def test_a_cancelled_run_replays_without_nondeterminism(tmp_path: Path):
    """Record a real cancelled run, then replay its history: the command sequence must agree.

    This is the acceptance test for the stop path. It fails if the delivery ever emits a command in an
    order a replay does not reproduce — which is the failure the previous `asyncio.create_task` design
    could only argue about and never measured.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> Any:
        run = await client.start_workflow(
            RunWorkflow.run,
            args=[_payload(repo, spec)],
            id="run:run-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        # It must genuinely PARK, or the history is an ordinary completion and proves nothing about the
        # stop path. `_wait_parked` is the child's own status query.
        await _wait_parked_child(client, CHILD_WORKFLOW_ID, _status_query("plain"))
        # The exact Signal `POST /runs/:id/cancel` sends, and the ONLY act here.
        await run.signal(SIGNALS["cancel"])
        completion = await _result(run)
        history = await run.fetch_history()
        return completion, history

    completion, history = _run(_with_worker("plain", scenario))

    # The premise: this history really is a CANCELLED run, not a completed one.
    assert completion.outcome == "stopped", completion

    # The evidence: same code, same history, no divergent command. `raise_on_replay_failure=False` so
    # the result carries the SDK's diagnosis instead of an exception whose message this test would
    # have to parse; both are asserted.
    async def replay_result() -> Any:
        replayer = Replayer(workflows=list(registered_workflows()))
        return await replayer.replay_workflow(history, raise_on_replay_failure=False)

    result = _run(replay_result())
    assert result.replay_failure is None, result.replay_failure
    # And with the SDK's strict mode on, replay does not raise at all.
    _replay(history)


def test_the_recording_contains_the_stop_signal_and_a_child_signal_command(tmp_path: Path):
    """The history this test replays is the RIGHT history: it carries the stop and the delivery.

    A replay of a history with no stop would pass without exercising anything, so the recording's own
    shape is asserted: the run was SIGNALLED, and a `SignalExternalWorkflowExecutionInitiated` command
    names the child. That is also the direct measurement of the bot's finding — the delivery is a
    command IN THE HISTORY, at a position the replay reproduces.
    """
    from temporalio.api.enums.v1 import EventType

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = _spec(repo_dir=str(repo), engine="plain")

    async def scenario(client: Any) -> Any:
        run = await client.start_workflow(
            RunWorkflow.run,
            args=[_payload(repo, spec)],
            id="run:run-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked_child(client, CHILD_WORKFLOW_ID, _status_query("plain"))
        await run.signal(SIGNALS["cancel"])
        await _result(run)
        return await run.fetch_history()

    history = _run(_with_worker("plain", scenario))
    types = [int(event.event_type) for event in history.events]

    assert int(EventType.EVENT_TYPE_WORKFLOW_EXECUTION_SIGNALED) in types, (
        "the recorded run was never signalled, so its replay proves nothing about the stop path"
    )
    signalled_children = [
        event.signal_external_workflow_execution_initiated_event_attributes.workflow_execution.workflow_id
        for event in history.events
        if event.HasField("signal_external_workflow_execution_initiated_event_attributes")
    ]
    assert CHILD_WORKFLOW_ID in signalled_children, signalled_children
