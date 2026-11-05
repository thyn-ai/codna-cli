"""Two tasks of ONE run, each in its own worktree — under a real Temporal test server (D5).

`test_durable_worktree.py` proves the lifecycle and `test_worktree_activities.py` proves each step works in the
right tree. Neither proves the thing the plan's exit criterion actually names: *a run of N tasks creates
N worktrees, and they are the tasks' own*. That is a property of the WORKFLOW — it decides how many
tasks there are, in what order, and when the run is over — so it can only be shown by executing the
workflow.

Time-skipping, the real activity bodies, a fake sidecar and REAL git: the same harness the rest of the
durable suite uses. What is asserted is what a human could check on disk:

* the exact two paths and the exact two branches, from git;
* that each task's session was pointed at its OWN tree (the fake sidecar records the ``working_dir`` it
  was given, so "which tree did the work happen in" is observed rather than inferred);
* that the two branches hold DIFFERENT commits, and that the shared checkout holds neither;
* that the run's finalisation swept both worktrees away, and said so in the ledger.

Auto time skipping stays OFF for the same reason the other scenarios in this suite turn it off: a
parked task's outstanding ``run_timeout`` is a timer, and skipping past it would end the run while the
test was still on its way to approve a child.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from datetime import timedelta  # noqa: E402

from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: E402
from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.workflows import RunWorkflow, TaskWorkflowPlain, registered_workflows  # noqa: E402
from codna.durable.worktree import branch_for_task, worktree_path_for  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402

TASK_QUEUE = "codna-durable-worktree-tests"
RUN_ID = "run-wt"
RUN_TIMEOUT = timedelta(hours=24)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _spec(repo: Path, number: int, *, worktree: bool) -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id=f"task-{number}",
        task_number=number,
        brief=f"Fix area {number}",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine="plain",
        worktree=worktree,
    )


def _payload(repo: Path, tasks: tuple[TaskSpec, ...], *, worktree: bool) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": "plain",
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
        "worktree": worktree,
    }


async def _approve_children(client: Any, count: int) -> None:
    """Approve any parked child. Absent or already-closed children are simply skipped.

    Both benign races — not started yet, and a concurrent iteration already approved it — are
    `approve_if_parked`'s, which absorbs only those. Anything else (including a bug in this harness)
    fails at the call rather than leaving the children parked until the 60s result timeout.
    """
    for number in range(1, count + 1):
        await approve_if_parked(
            client, f"task-{RUN_ID}-task-{number}", TaskWorkflowPlain, identity="wt@codna.ai"
        )


async def _drive(client: Any, run_payload: dict[str, Any], count: int) -> Any:
    handle = await client.start_workflow(
        RunWorkflow.run,
        args=[run_payload, None],
        id=f"run-{RUN_ID}",
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    for _ in range(800):
        await _approve_children(client, count)
        described = await handle.describe()
        if int(described.status) != 1:  # 1 == RUNNING
            break
        await asyncio.sleep(0.02)
    return await asyncio.wait_for(handle.result(), timeout=60)


def _with_worker(scenario: Any) -> Any:
    """Run a scenario against a fresh time-skipping server and the plain engine's real activities."""

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


def test_two_tasks_of_one_run_work_in_two_worktrees(tmp_path: Path):
    """The plan's exit criterion, for N=2: two checkouts, two branches, two independent commits."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)

    tasks = tuple(_spec(repo, number, worktree=True) for number in (1, 2))
    completion = _with_worker(lambda client: _drive(client, _payload(repo, tasks, worktree=True), 2))

    assert completion.outcome == "completed", completion
    assert [summary.task_id for summary in completion.tasks] == ["task-1", "task-2"]

    # BOTH paths, exactly where the plan puts them, with the exact names.
    first_path = worktree_path_for(runs_root, RUN_ID, "task-1")
    second_path = worktree_path_for(runs_root, RUN_ID, "task-2")
    assert first_path.name == "wt-task-1" and second_path.name == "wt-task-2"
    assert first_path != second_path
    assert first_path.parent == second_path.parent == runs_root / RUN_ID

    # Each task's SESSION was pointed at its own tree, and the trees git checked out for those sessions
    # were the paths above. This is the seam the fake sidecar exposes: its edits land in whatever
    # `working_dir` it was given, so the tree the work happened in is OBSERVED rather than inferred.
    #
    # They are asserted from the sessions rather than from disk because the run SWEEPS them when it
    # finishes (the next test), which is the correct end state — a test that demanded the directories
    # still exist here would be asserting the leak.
    # A set, not a list: each task creates TWO sessions (its plan turn and its implement turns), so the
    # property is "only these two trees were ever worked in", not "exactly two sessions".
    session_trees = {
        entry["working_dir"] for entry in harness.sidecar.sessions if entry.get("working_dir")
    }
    assert session_trees == {str(first_path), str(second_path)}

    # Both BRANCHES survive the sweep, each on its own task's line of work.
    assert branch_for_task(RUN_ID, 1) == "codna/run-run-wt/task-1"
    assert branch_for_task(RUN_ID, 2) == "codna/run-run-wt/task-2"
    for number in (1, 2):
        # The branch exists and the checkpoint ref it recorded is reachable from it.
        fakes.git(repo, "rev-parse", "--verify", f"refs/heads/{branch_for_task(RUN_ID, number)}")

    # Two DIFFERENT commits, on two different branches.
    shas = {
        number: fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, number)).strip()
        for number in (1, 2)
    }
    assert shas[1] != shas[2]
    # ...and the shared checkout's HEAD moved for neither of them.
    assert fakes.git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert fakes.git(repo, "rev-parse", "HEAD").strip() not in shas.values()

    # ONE checkpoint per implement turn, each recording its own task's branch.
    branches = {record["branch"] for record in harness.checkpoints(RUN_ID)}
    assert branches == {branch_for_task(RUN_ID, 1), branch_for_task(RUN_ID, 2)}


def test_the_run_sweeps_its_worktrees_when_it_finishes(tmp_path: Path):
    """The cleanup policy, at the run level: a completed run leaves no checkout behind."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)

    tasks = tuple(_spec(repo, number, worktree=True) for number in (1, 2))
    completion = _with_worker(lambda client: _drive(client, _payload(repo, tasks, worktree=True), 2))

    assert completion.outcome == "completed", completion
    # Gone from disk AND from git's own list: a directory alone would be litter, not a released checkout.
    assert not worktree_path_for(runs_root, RUN_ID, "task-1").exists()
    assert not worktree_path_for(runs_root, RUN_ID, "task-2").exists()
    listed = fakes.git(repo, "worktree", "list", "--porcelain")
    assert "wt-task-1" not in listed and "wt-task-2" not in listed

    # The sweep's decision is in the ledger, with a reason a reader can act on.
    notes = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert len(notes) == 1, notes
    assert notes[0]["kept"] is False
    assert notes[0]["count"] == 2
    assert notes[0]["reason"] == "run completed"


def test_a_run_that_fails_still_sweeps_its_worktrees(tmp_path: Path):
    """A FAILED run leaks nothing either: finalisation is on the failure path, not only the happy one.

    The failure is injected where a real one happens — the child task's workflow — so what is exercised
    is the run's own `except` and its sweep, not a bespoke code path a test invented. Without the wrap,
    this run would leave two full checkouts on disk with nothing to reclaim them.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    # A worktree already exists for each task (as if each had started), so the sweep has work to do.
    from codna.durable.worktree import create_task_worktree

    for number in (1, 2):
        create_task_worktree(
            repo_dir=repo,
            runs_root=runs_root,
            run_id=RUN_ID,
            task_id=f"task-{number}",
            task_number=number,
            base_branch="main",
        )
    assert worktree_path_for(runs_root, RUN_ID, "task-1").is_dir()

    # A child that FAILS for a real reason: its turns are pointed at a checkout that does not exist, so
    # the implement turn's write fails, the activity exhausts its retries and the child workflow fails.
    # A failure injected by patching code would not exercise the child's real failure propagation.
    tasks = (_spec(repo, 1, worktree=True),)
    payload = _payload(repo, tasks, worktree=True)
    payload["tasks"][0]["repo_dir"] = str(tmp_path / "does-not-exist")

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id=f"run-{RUN_ID}",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        try:
            await asyncio.wait_for(handle.result(), timeout=60)
        except Exception:  # noqa: BLE001 - the run is EXPECTED to fail; the sweep is what is under test
            return "failed"
        return "completed"

    outcome = _with_worker(scenario)

    assert outcome == "failed"
    # Both checkouts are gone, including the one belonging to a task the run never reached.
    assert not worktree_path_for(runs_root, RUN_ID, "task-1").exists()
    assert not worktree_path_for(runs_root, RUN_ID, "task-2").exists()
    reasons = [
        record.payload.get("reason")
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert reasons == ["run failed"], reasons


def test_a_run_without_the_flag_shares_one_checkout_and_creates_no_worktree(tmp_path: Path):
    """`--no-worktree` (the default) is unchanged behaviour: one tree, no sweeps, no new disk."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)

    tasks = tuple(_spec(repo, number, worktree=False) for number in (1, 2))
    completion = _with_worker(lambda client: _drive(client, _payload(repo, tasks, worktree=False), 2))

    assert completion.outcome == "completed", completion
    # No WORKTREE was created for either task. Asserted on the paths the plan names rather than on the
    # run directory: the run directory itself always exists (the ledger, the episodes and the verify
    # logs live there), so "no `wt-*` inside it" is the honest statement of "no checkout was made".
    run_dir = runs_root / RUN_ID
    assert list(run_dir.glob("wt-*")) == []
    # ...every session used the shared checkout...
    assert {entry["working_dir"] for entry in harness.sidecar.sessions if entry.get("working_dir")} == {
        str(repo.resolve())
    }
    # ...and no cleanup activity was scheduled, which is the replay-safety property for a run that
    # predates this feature.
    assert not [
        record
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
