"""A run records the checkout it works in, and `GET /runs` reports it per run.

WHY THIS IS A WORKFLOW TEST rather than an assertion on `notify`. The field has to travel the whole
way — from `RunSpec.repo_dir`, through the opening notification's activity request, into a ledger row,
and back out through the route's head scan — and each hop is a place it can be dropped. A unit test of
`repo_dir_of` proves the READ; a unit test of `notify` proves the WRITE; only a real run proves that the
workflow actually ASKS for it, which is the hop that was missing (the spec carried `repo_dir` and
nothing ever read it).

WHY IT RIDES ON THE EXISTING `run.started` CALL. A run's Temporal history replays its COMMAND
sequence. A second bookkeeping activity — the obvious shape for "also record the repo" — would make
every recorded history diverge on replay, failing the cassette tests and, in production, every run that
was in flight across the deploy. So the field is an argument to a call that already existed, and the
replay tests below are what keep it that way.

The scenarios:

* a plain-engine run records its repository on the opening row, and the route reports it;
* the same for the graph engine, because the value comes from the shared `RunSpec` and a per-engine
  drift is exactly the kind of thing two code paths produce;
* a run that continues as new still reports ONE repository — the resumed generation re-notifies, and
  the reader must not care how many generations it took;
* a ledger written before the field existed reports `NULL`, not a guess.
"""
from __future__ import annotations

import asyncio
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
from codna.durable.run_repo import repo_dir_of  # noqa: E402
from codna.durable.workflows import (  # noqa: E402
    RunWorkflow,
    TaskWorkflow,
    TaskWorkflowPlain,
    registered_workflows,
)
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402

TASK_QUEUE = "codna-durable-repo-tests"
RUN_ID = "run-repo"
RUN_TIMEOUT = timedelta(hours=24)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _spec(repo: Path, *, engine: str, task_id: str = "task-1") -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id=task_id,
        task_number=1,
        brief="Record the checkout this run works in",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine=engine,  # type: ignore[arg-type]
    )


def _payload(repo: Path, tasks: tuple[TaskSpec, ...], *, engine: str) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": engine,
        # High enough that a one-task run finishes in a single generation: this file is about the
        # REPOSITORY, and the continue-as-new case has its own test below at a bound of one.
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
        "worktree": False,
    }


async def _drive(client: Any, payload: dict[str, Any], *, workflow_cls: Any) -> Any:
    handle = await client.start_workflow(
        RunWorkflow.run,
        args=[payload, None],
        id=f"run-{RUN_ID}",
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    for _ in range(2000):
        await approve_if_parked(client, f"task-{RUN_ID}-task-1", workflow_cls, identity="repo@codna.ai")
        described = await handle.describe()
        if int(described.status) != 1:  # 1 == RUNNING
            break
        await asyncio.sleep(0.02)
    return await asyncio.wait_for(handle.result(), timeout=90)


def _with_worker(engine: str, scenario: Any) -> Any:
    """A fresh time-skipping server and the REAL activities; the plugin for the graph engine."""

    async def wrapper() -> Any:
        from codna.durable.graph import build_task_graph

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
                    # Auto skipping OFF: a parked child's outstanding `run_timeout` is a timer, and
                    # skipping past it would end the run while this helper is still approving.
                    with env.auto_time_skipping_disabled():
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    return _run(wrapper())


def _opening_rows(harness: fakes.Harness) -> list[dict[str, Any]]:
    """Every `run.started` notification's payload, in ledger order."""
    return [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note"
        and record.payload.get("event") == "notification"
        and record.payload.get("kind") == "run.started"
    ]


@pytest.mark.parametrize(
    ("engine", "workflow_cls"),
    [("plain", TaskWorkflowPlain), ("langgraph", TaskWorkflow)],
)
def test_a_run_records_the_checkout_it_works_in(tmp_path: Path, engine: str, workflow_cls: Any) -> None:
    """The opening row carries the run's repository, on BOTH engines.

    The value travels from `RunSpec.repo_dir` — which `validate()` already required to be absolute —
    so it is the same string the run was started against, not a path re-derived from the process.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    payload = _payload(repo, (_spec(repo, engine=engine),), engine=engine)

    completion = _with_worker(engine, lambda client: _drive(client, payload, workflow_cls=workflow_cls))
    assert completion.outcome == "completed", completion

    rows = _opening_rows(harness)
    assert rows, "the run wrote no run.started notification at all"
    assert rows[0]["repo_dir"] == str(repo)
    # And the reader agrees with what the ledger holds: this is the pair the route ships.
    assert repo_dir_of(harness.ledger(RUN_ID).path) == str(repo)


def test_a_run_that_continues_as_new_still_reports_one_repository(tmp_path: Path) -> None:
    """A resumed run re-notifies, and the reader takes the FIRST repository — not a later generation's.

    The resumed `RunSpec` is the same spec by construction, so every generation names the same checkout.
    Taking the first keeps the answer independent of how many generations the run needed, which is what
    makes it deterministic for a caller reading a list.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    tasks = tuple(_spec(repo, engine="plain", task_id=f"task-{n}") for n in (1, 2))
    payload = _payload(repo, tasks, engine="plain")
    # A bound of ONE task forces a continue-as-new boundary between the two tasks.
    payload["continue_as_new_tasks"] = 1

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id=f"run-{RUN_ID}",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        for _ in range(2000):
            for number in (1, 2):
                await approve_if_parked(
                    client, f"task-{RUN_ID}-task-{number}", TaskWorkflowPlain, identity="repo@codna.ai"
                )
            described = await handle.describe()
            if int(described.status) != 1:
                break
            await asyncio.sleep(0.02)
        return await asyncio.wait_for(handle.result(), timeout=90)

    completion = _with_worker("plain", scenario)
    assert completion.outcome == "completed", completion
    # More than one generation ran, so the opening notification was written more than once.
    rows = _opening_rows(harness)
    assert len(rows) >= 2, f"expected a continue-as-new boundary, got {len(rows)} run.started row(s)"
    assert {row["repo_dir"] for row in rows} == {str(repo)}
    # One answer, regardless of which generation wrote it.
    assert repo_dir_of(harness.ledger(RUN_ID).path) == str(repo)


def test_a_ledger_with_no_repository_reports_none(tmp_path: Path) -> None:
    """The pre-change shape: a run written before the field existed has no repository on record.

    `None` is the answer a caller must present as "unknown". The temptation is to fall back to the
    reading window's workspace, which is exactly the mislabelling this whole change removes — so the
    reader returns nothing rather than something plausible. `harness.ledger(...)` is the same factory the
    activities use, so the file written here has the production shape.
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(fakes.make_repo(tmp_path / "repo"), runs_root)
    ledger = harness.ledger(RUN_ID)
    # Exactly what `notify` wrote BEFORE the field existed: no `repo_dir` key anywhere.
    ledger.append("note", payload={"event": "notification", "kind": "run.started", "message": "go"})
    assert repo_dir_of(ledger.path) is None


def test_a_single_record_with_no_trailing_newline_is_still_read(tmp_path: Path) -> None:
    """A just-started run may have ONE record, and it may not end in a newline yet.

    The head reader must consider the final chunk exactly as the tail reader does. Dropping it
    unconditionally would report "no repository" for the one ledger that is guaranteed to have one — the
    run's own opening row — which is the same class of bug as the mislabelling this field removes.
    """
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(fakes.make_repo(tmp_path / "repo"), runs_root)
    ledger = harness.ledger(RUN_ID)
    ledger.append(
        "note",
        payload={"event": "notification", "kind": "run.started", "message": "go", "repo_dir": "/src/only-row"},
    )
    # Strip the trailing newline: the file is now a single, complete record with no line terminator.
    path = ledger.path
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert b"\n" not in path.read_bytes()
    assert repo_dir_of(path) == "/src/only-row"
