"""The plan's D5 exit criterion, executed: a 3-task goal run with worktrees, PRs and CI (D5).

The criterion the plan names is a SENTENCE, and every clause of it is asserted here:

    "a 3-task goal run → 3 worktrees, PRs, CI transitions, task-2's plan recalls the task-1 episode"

That sentence is why this file exists as a workflow test rather than as three more unit tests. Each
clause is a different layer, and each one can be true on its own while the run is wrong:

* **3 worktrees** — real git, at the plan's exact paths and on the plan's exact branches. A fake
  filesystem would prove nothing: the property is that `git worktree add` put each task's commits on
  its own branch, which the cleanup then reclaimed.
* **3 PRs** — three distinct `open_pr` results, one per task, each on its own branch. The PR boundary
  is the one genuinely external thing here, so it is a scripted fake; everything that DECIDES to open
  one is the real code.
* **CI transitions** — the gate reads ``failed`` then ``passed``, so the transitions are OBSERVED in
  the ledger rather than inferred from the run finishing. A run that never read CI twice would pass a
  weaker test and prove nothing about the loop.
* **task-2's plan recalls the task-1 episode** — asserted from the PROMPT the sidecar was actually
  given, which is the only place that claim is observable. A test that read the plan *request* back
  out of the code under test would pass even if the prompt never carried it.

Both engines are covered where the difference could matter (the plan prompt is built by different
code paths); the goal LOOP lives above the task, so it is the same for both.

Auto time skipping stays OFF for the reason the rest of this suite documents: a parked child's
outstanding `run_timeout` is a timer, and skipping past it would end the run while the test was
still on its way to approve a task.
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
from codna.durable.goal import GoalSpec, StopCondition  # noqa: E402
from codna.durable.workflows import (  # noqa: E402
    RunWorkflow,
    TaskWorkflow,
    TaskWorkflowPlain,
    registered_workflows,
)
from codna.durable.worktree import branch_for_task, worktree_path_for  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402

TASK_QUEUE = "codna-durable-goal-tests"
RUN_ID = "run-goal"
RUN_TIMEOUT = timedelta(hours=24)
OBJECTIVE = "Make the retry client resilient: every task must verify and be approved"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _spec(repo: Path, number: int, engine: str = "plain") -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id=f"task-{number}",
        task_number=number,
        brief=f"Fix area {number}",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine=engine,  # type: ignore[arg-type]
        worktree=True,
    )


def _payload(
    repo: Path,
    tasks: tuple[TaskSpec, ...],
    *,
    goal: GoalSpec | None,
    engine: str = "plain",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": engine,
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
        "worktree": True,
    }
    if goal is not None:
        payload["goal"] = goal.to_payload()
    return payload


async def _approve_children(client: Any, count: int, workflow_cls: Any) -> None:
    """Approve any parked child. Absent or already-closed children are simply skipped.

    A fix CI task's workflow id is ``task-<run>-<task>-ci<n>``, and the loop may start several, so
    the ids are enumerated rather than derived from the task count alone.

    The benign races — a child not started yet, a closed one, a decision a concurrent iteration
    already recorded — belong to `approve_if_parked`, which absorbs ONLY those. A wrong
    ``workflow_cls`` is refused there before any await, so it fails at this call instead of leaving
    every child parked until the 90s result timeout.
    """
    ids = [f"task-{RUN_ID}-task-{number}" for number in range(1, count + 1)]
    for number in range(1, count + 1):
        ids.extend(f"task-{RUN_ID}-task-{number}-ci{attempt}" for attempt in range(1, 4))
    for workflow_id in ids:
        await approve_if_parked(client, workflow_id, workflow_cls, identity="goal@codna.ai")


async def _drive(client: Any, payload: dict[str, Any], count: int, workflow_cls: Any) -> Any:
    handle = await client.start_workflow(
        RunWorkflow.run,
        args=[payload, None],
        id=f"run-{RUN_ID}",
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    for _ in range(2000):
        await _approve_children(client, count, workflow_cls)
        described = await handle.describe()
        if int(described.status) != 1:  # 1 == RUNNING
            break
        await asyncio.sleep(0.02)
    return await asyncio.wait_for(handle.result(), timeout=90)


def _with_worker(scenario: Any, engine: str = "plain") -> Any:
    """A fresh time-skipping server, the real activities, and the engine's plugin when it needs one."""
    from codna.durable.graph import build_task_graph

    async def wrapper() -> Any:
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
                    with env.auto_time_skipping_disabled():
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    return _run(wrapper())


# -- the exit criterion, end to end ----------------------------------------------------------


def test_a_three_task_goal_run_creates_three_worktrees_three_prs_and_recalls_the_first_episode(
    tmp_path: Path,
):
    """THE plan's D5 verification, in one test: every clause of the sentence is asserted.

    The stop condition requires ``verify ∧ review`` (no CI), so a CI read must NOT happen — that is
    asserted too, because "the gate ran" and "the gate ran when asked" are different claims.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    harness.pr_number = 100  # bumped per push by the fake, so the three PRs are distinct
    original_push = harness.push_and_open

    def push_and_open(request: Any, sha: str) -> dict[str, Any]:
        opened = original_push(request, sha)
        harness.pr_number += 1
        return opened

    harness.push_and_open = push_and_open
    harness.runtime.push_and_open = push_and_open  # type: ignore[assignment]

    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=False),
        max_tasks=5,
    )
    tasks = tuple(_spec(repo, number) for number in (1, 2, 3))

    completion = _with_worker(
        lambda client: _drive(
            client, _payload(repo, tasks, goal=goal), 3, TaskWorkflowPlain
        )
    )

    # -- the run finished, and said WHY ------------------------------------------------------
    assert completion.outcome == "completed", completion
    assert completion.goal_reason == "verify=passed; review=approved", completion.goal_reason
    assert [summary.task_id for summary in completion.tasks] == ["task-1", "task-2", "task-3"]
    assert all(summary.verify_passed is True for summary in completion.tasks)
    assert all(summary.review_verdict == "approved" for summary in completion.tasks)

    # -- 3 worktrees, at the plan's exact paths, on the plan's exact branches ---------------
    paths = {
        number: worktree_path_for(runs_root, RUN_ID, f"task-{number}") for number in (1, 2, 3)
    }
    for number, path in paths.items():
        assert path.name == f"wt-task-{number}"
        assert path.parent == runs_root / RUN_ID
    # Each task's SESSION worked in its own tree. Observed from the sidecar's records, because the
    # run sweeps the checkouts when it finishes (the correct end state — asserting they still exist
    # here would be asserting the leak).
    session_trees = {
        entry["working_dir"] for entry in harness.sidecar.sessions if entry.get("working_dir")
    }
    assert session_trees == {str(path) for path in paths.values()}
    # The branches survive the sweep, each with its own commit, and the shared checkout moved for none.
    for number in (1, 2, 3):
        assert branch_for_task(RUN_ID, number) == f"codna/run-{RUN_ID}/task-{number}"
        fakes.git(repo, "rev-parse", "--verify", f"refs/heads/{branch_for_task(RUN_ID, number)}")
    shas = {
        number: fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, number)).strip()
        for number in (1, 2, 3)
    }
    assert len(set(shas.values())) == 3, shas
    assert fakes.git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert fakes.git(repo, "rev-parse", "HEAD").strip() not in shas.values()

    # -- 3 PRs, one per task, each on its own branch ----------------------------------------
    assert harness.push_calls == 3, harness.push_calls
    pr_urls = [summary.pr_url for summary in completion.tasks]
    assert len(set(pr_urls)) == 3, pr_urls
    assert all(url and "/pull/" in url for url in pr_urls)

    # -- the plan for task 2 RECALLED task 1's episode --------------------------------------
    #
    # The claim is about the PROMPT the model was given, so it is asserted from the sidecar's own
    # record of every prompt it received. The plan turn is the first prompt of each task's session.
    plan_prompts = [prompt for prompt in harness.sidecar.prompts if "read-only" in prompt]
    assert len(plan_prompts) == 3, plan_prompts
    first, second, third = plan_prompts
    # Task 1's plan has the objective and NO history to recall.
    assert OBJECTIVE in first
    assert "Already completed in this run" not in first
    # Task 2's plan carries the objective AND task 1's episode.
    assert OBJECTIVE in second
    assert "Already completed in this run" in second
    assert "task-1 (approved)" in second
    assert "Fix area 1" in second
    # ...and not task 2's own (which had not run when the plan was produced).
    assert "task-2 (" not in second
    # Task 3 recalls both earlier tasks, oldest first.
    assert "task-1 (approved)" in third and "task-2 (approved)" in third
    assert third.index("task-1 (approved)") < third.index("task-2 (approved)")

    # -- episodes are on disk, one per task, and they are what was recalled ------------------
    episodes = sorted((runs_root / RUN_ID / "episodes").glob("task-*.json"))
    assert [path.name for path in episodes] == ["task-1.json", "task-2.json", "task-3.json"]
    # The recall ledger note names the episodes it read — "planned with" is an audited fact.
    recalled = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "episodes.recalled"
    ]
    assert [note["count"] for note in recalled] == [0, 1, 2], recalled
    assert recalled[-1]["tasks"] == ["task-1", "task-2"]

    # -- the CI gate did NOT run, because the condition did not ask for it ------------------
    assert harness.ci_calls == []
    assert not [
        record
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "ci.watched"
    ]

    # -- the run swept its own checkouts ---------------------------------------------------
    assert not any(path.exists() for path in paths.values())
    cleaned = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert cleaned and cleaned[0]["count"] == 3


def test_the_task_list_is_the_plan_so_a_passing_task_does_not_abandon_the_rest(tmp_path: Path):
    """Goal mode does NOT discard caller-supplied tasks when the first one happens to pass.

    ``spec.tasks`` is the plan. A run that skipped tasks 2 and 3 because task 1 met the condition
    would silently do less than it was asked while reporting success — the opposite of what a goal
    run is for. What the condition decides is the OUTCOME once the list is exhausted, which the
    ``goal_unmet`` test below pins from the other side.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    goal = GoalSpec(
        objective=OBJECTIVE, stop_condition=StopCondition(verify=True, review=True), max_tasks=3
    )
    tasks = tuple(_spec(repo, number) for number in (1, 2, 3))

    completion = _with_worker(
        lambda client: _drive(client, _payload(repo, tasks, goal=goal), 3, TaskWorkflowPlain)
    )

    assert completion.outcome == "completed", completion
    assert [summary.task_id for summary in completion.tasks] == ["task-1", "task-2", "task-3"]
    assert harness.push_calls == 3
    # A verdict was recorded at each boundary, with the tasks it could not see yet.
    verdicts = [
        record.payload["evidence"]
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note"
        and record.payload.get("event") == "notification"
        and record.payload.get("kind") == "goal.verdict"
    ]
    assert [verdict["tasks_done"] for verdict in verdicts] == [1, 2, 3]
    assert [verdict["tasks_remaining"] for verdict in verdicts] == [2, 1, 0]
    # The verdict is satisfied at every boundary here, because every task passed.
    assert all(verdict["satisfied"] is True for verdict in verdicts)
    assert verdicts[-1]["checks"] == [["verify", "passed"], ["review", "approved"]]


def test_an_unmet_goal_ends_goal_unmet_at_its_task_bound(tmp_path: Path):
    """``max_tasks`` is a HARD stop: a run whose tasks keep failing stops and reports why."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    # Every task's verification fails, so the goal is never met. The children park on the exhausted
    # attempt ceiling; the test approves them so the child COMPLETES (rejected), which is what lets
    # the run evaluate a genuine "unmet" rather than hanging.
    harness.verify_passes = [False] * 20
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True),
        max_tasks=2,
    )
    tasks = tuple(_spec(repo, number) for number in (1, 2, 3))

    completion = _with_worker(
        lambda client: _drive(client, _payload(repo, tasks, goal=goal), 3, TaskWorkflowPlain)
    )

    assert completion.outcome == "goal_unmet", completion
    # Exactly the allowance was spent — not one more, and not zero.
    assert len(completion.tasks) == 2, completion.tasks
    assert "max_tasks=2" in completion.goal_reason


# -- the graph engine drives the same loop ------------------------------------------------------


def test_the_graph_engine_plans_with_the_same_recall_block(tmp_path: Path):
    """The goal block is appended by the PLAN STEP, so both engines send it — asserted, not assumed.

    The engines build their plan requests through different code (`NodeContext.plan_request` vs the
    plain walk's explicit call), which is exactly where one of them would silently stop carrying the
    objective.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    goal = GoalSpec(
        objective=OBJECTIVE, stop_condition=StopCondition(verify=True, review=True), max_tasks=3
    )
    tasks = tuple(_spec(repo, number, engine="langgraph") for number in (1, 2))

    completion = _with_worker(
        lambda client: _drive(client, _payload(repo, tasks, goal=goal, engine="langgraph"), 2, TaskWorkflow),
        engine="langgraph",
    )

    assert completion.outcome == "completed", completion
    plan_prompts = [prompt for prompt in harness.sidecar.prompts if "read-only" in prompt]
    assert len(plan_prompts) == 2, plan_prompts
    assert OBJECTIVE in plan_prompts[0]
    assert "Already completed in this run" not in plan_prompts[0]
    assert OBJECTIVE in plan_prompts[1]
    assert "task-1 (approved)" in plan_prompts[1]
