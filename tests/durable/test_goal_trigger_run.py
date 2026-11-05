"""The GitHub trigger's idempotency, proven ON a workflow (D5).

`test_schedule_and_trigger.py` proves the dispatch contract (the id, the signal, the conflict policy)
against a fake client. This file proves the property that contract exists for, on the real
time-skipping server and the real `RunWorkflow`:

* a delivery signalled AGAIN reaches ONE execution (the workflow id is the delivery id), and
* the workflow records the repeat as a DUPLICATE rather than treating it as fresh work.

A test that only inspected the client call would pass even if the workflow raised two tasks for one
event — which is the double-bill this replaces the SQLite queue's dedupe to prevent.

The schedule CYCLE (create → unchanged → update) needs a server that implements ``CreateSchedule``,
which the in-memory test server does not. It is gated on ``CODNA_TEST_TEMPORAL=1`` and runs against
the same `DevServer` the CLI starts.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: E402
from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.workflows import RunWorkflow, registered_workflows  # noqa: E402
from tests.durable import fakes  # noqa: E402

TASK_QUEUE = "codna-durable-trigger-tests"
RUN_ID = "gh-fix-delivery-1"
RUN_TIMEOUT = timedelta(hours=24)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _payload(repo: Path) -> dict[str, Any]:
    task = TaskSpec(
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        brief="Fix the flaky retry",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine="plain",
    )
    return {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload()],
        "engine": "plain",
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
    }


def _with_worker(scenario: Any) -> Any:
    async def wrapper() -> Any:
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

    return _run(wrapper())


def test_a_redelivery_reaches_one_execution_and_is_recorded_as_a_duplicate(tmp_path: Path):
    """THE property: GitHub retries a delivery; the second attempt must not become a second run.

    The first delivery starts the run with the signal (Signal-with-Start). The second is sent with
    ``USE_EXISTING`` to the SAME id — the shape a redelivery takes — and must reach the execution
    already parked on its approval, which then reports the duplicate.
    """
    from temporalio.common import WorkflowIDConflictPolicy

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    fakes.build_harness(repo, runs_root)
    payload = _payload(repo)
    signal = {
        "schema_version": 1,
        "delivery_id": "delivery-1",
        "kind": "fix",
        "repo_full_name": "thyn-ai/codna",
        "number": 3,
    }

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id="delivery-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
            start_signal="webhook",
            start_signal_args=[signal],
        )
        # Wait until the child has parked: the run is then definitively alive with its signal
        # recorded, so a second start cannot be racing the first one's own setup.
        child = client.get_workflow_handle(f"task-{RUN_ID}-task-1")
        for _ in range(600):
            try:
                status = await child.query("status")
            except Exception:  # noqa: BLE001 - not started yet
                await asyncio.sleep(0.02)
                continue
            if status.get("awaiting_approval"):
                break
            await asyncio.sleep(0.02)

        # The REDELIVERY: same id, USE_EXISTING. Returns the ALREADY-RUNNING execution.
        again = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id="delivery-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
            start_signal="webhook",
            start_signal_args=[signal],
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        run_status = await handle.query(RunWorkflow.status)
        from codna.durable.workflows import TaskWorkflowPlain

        await child.execute_update(TaskWorkflowPlain.approve, {"identity": "trigger@codna.ai"})
        completion = await asyncio.wait_for(handle.result(), timeout=60)
        return {
            "status_after_redelivery": run_status,
            "completion": completion,
            "second_handle_run_id": again.first_execution_run_id,
            "first_handle_run_id": handle.first_execution_run_id,
        }

    result = _with_worker(scenario)

    # ONE execution: the redelivery resolved to the run that already existed, by the same run id.
    assert result["second_handle_run_id"] == result["first_handle_run_id"]
    # The signal arrived WITH the start (before the first task), so the delivery is already recorded
    # when the workflow queries it — that is what Signal-with-Start buys.
    status = result["status_after_redelivery"]
    assert status["deliveries"] == ["delivery-1"]
    # ...and the repeat is recorded as a DUPLICATE, not as fresh work.
    assert status["duplicate_deliveries"] == ["delivery-1"]
    # The run did its task exactly once, and finished.
    assert result["completion"].outcome == "completed"
    assert [summary.task_id for summary in result["completion"].tasks] == ["task-1"]


def test_a_run_started_without_a_signal_has_no_deliveries(tmp_path: Path):
    """A manually started run must not claim a GitHub delivery it never received."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    fakes.build_harness(repo, runs_root)
    payload = _payload(repo)

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id="manual-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        return await handle.query(RunWorkflow.status)

    status = _with_worker(scenario)
    assert status["deliveries"] == []
    assert status["duplicate_deliveries"] == []


# -- the real schedule cycle (needs a server with CreateSchedule) ----------------------------------

HAS_EXTRA = (
    importlib.util.find_spec("temporalio") is not None
    and importlib.util.find_spec("langgraph") is not None
)
GATED = os.environ.get("CODNA_TEST_TEMPORAL") == "1"


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.skipif(
    not GATED, reason="set CODNA_TEST_TEMPORAL=1 to run against a real dev server"
)
def test_a_schedule_is_created_then_left_alone_then_updated(tmp_path: Path):
    """Register, re-register unchanged, then change the interval — the three outcomes, in order.

    The middle step is the one that matters: an unchanged definition must NOT update, or every
    invocation of a declarative command would report a change it did not make.

    Why this is GATED: the in-memory time-skipping server does not implement ``CreateSchedule``
    (measured: ``Method .../WorkflowService/CreateSchedule is unimplemented``), so the create/compare/
    update cycle can only be proven against the dev server a user actually gets. The SHAPE of the
    schedule it would create is asserted without a server in `test_schedule_and_trigger.py`.
    """
    import contextlib
    import threading

    from codna.durable.config import load_durable_config
    from codna.durable.dev_server import DevServer, DevServerSettings, default_lock_path
    from codna.durable.schedule import ScheduleDefinition, ensure_schedule, schedule_id_for

    port = _free_port()
    root = tmp_path / "temporal"
    dev = DevServer(
        DevServerSettings(
            host="127.0.0.1",
            port=port,
            namespace="codna",
            db_path=root / "dev.db",
            bin_dir=root / "bin",
            log_path=tmp_path / "logs" / "temporal-dev.log",
            pid_path=root / "dev-server.json",
            lock_path=default_lock_path(),
        )
    )
    env = dict(os.environ)
    env["CODNA_RUNTIME_ROOT"] = str(tmp_path / "runtime")
    env["CODNA_DURABLE_RUNS_ROOT"] = str(tmp_path / "runs")
    env["CODNA_TEMPORAL_ADDRESS"] = f"127.0.0.1:{port}"
    config = load_durable_config(env=env, hostname="schedule-host")

    assert dev.ensure_started(timeout_s=180.0, headless=True).running
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    client: Any = None
    try:
        from temporalio.client import Client

        client = asyncio.run_coroutine_threadsafe(
            Client.connect(config.address, namespace=config.namespace), loop
        ).result(timeout=60)

        def call(coro: Any) -> Any:
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=120)

        def definition(every: int) -> Any:
            return ScheduleDefinition(
                schedule_id=schedule_id_for("d5-cycle"),
                every_seconds=every,
                cron=None,
                run_payload={"run_id": "run-d5-cycle-scheduled", "repo_dir": str(tmp_path)},
            )

        first = call(ensure_schedule(client, config, definition(3600)))
        assert first["created"] is True and first["updated"] is False, first

        second = call(ensure_schedule(client, config, definition(3600)))
        assert second["created"] is False and second["updated"] is False, second
        assert second["fingerprint"] == first["fingerprint"]

        third = call(ensure_schedule(client, config, definition(7200)))
        assert third["created"] is False and third["updated"] is True, third
        assert third["previous_fingerprint"] == first["fingerprint"]

        described = call(client.get_schedule_handle(schedule_id_for("d5-cycle")).describe())
        assert int(described.schedule.spec.intervals[0].every.total_seconds()) == 7200
        # The workflow id template survives the update: each fire is still its own execution.
        assert "{{.ScheduledTime}}" in described.schedule.action.id
    finally:
        if client is not None:
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=20)
        loop.close()
        with contextlib.suppress(BaseException):
            dev.stop(timeout_s=40.0)
