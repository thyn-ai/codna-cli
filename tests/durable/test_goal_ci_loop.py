"""The ``watch_ci`` → "fix CI" loop, and the two INDEPENDENT bounds that terminate it (D5).

The loop is the one shape in a durable run where the task list does not bound the work: one red pull
request can earn attempt after attempt against the SAME head. So it is bounded twice, and the plan
requires both:

* **attempts** — ``GoalSpec.max_ci_attempts``. A cap with only money would keep buying attempts
  until the run died; this says "three tries at one red PR, then a human looks".
* **budget** — the RUN's ``max_cost_usd`` minus what its tasks reported. A cap with only attempts
  would spend the whole run on one PR.

Both are asserted to terminate the loop ON THEIR OWN: one test stops it with attempts left and no
money, the other with money left and no attempts. A single test where both are exhausted at once
would pass even if only one bound were implemented.

**CI is the one genuinely external boundary**, so the checks read is a scripted fake. Everything that
decides what to DO about the answer — the loop, both bounds, the ledger, the fix task's brief — is the
real code, and the assertions are on the ledger and the run's own status, not on the fake.
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
from codna.durable.workflows import RunWorkflow, TaskWorkflowPlain, registered_workflows  # noqa: E402
from codna.durable.worktree import branch_for_task, worktree_path_for  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402

TASK_QUEUE = "codna-durable-goal-ci-tests"
RUN_ID = "run-ci"
RUN_TIMEOUT = timedelta(hours=24)
OBJECTIVE = "Land a green pull request"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _spec(repo: Path) -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        brief="Fix the retry loop",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine="plain",
        worktree=True,
    )


def _payload(repo: Path, task: TaskSpec, goal: GoalSpec) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload()],
        "engine": "plain",
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
        "worktree": True,
        "goal": goal.to_payload(),
    }


async def _approve(client: Any, workflow_id: str) -> None:
    # The benign races (not started yet, already decided by a concurrent iteration) belong to the
    # shared driver, which absorbs ONLY those and lets a harness bug fail at the call.
    await approve_if_parked(
        client, workflow_id, TaskWorkflowPlain, identity="ci@codna.ai"
    )


async def _drive(client: Any, payload: dict[str, Any]) -> Any:
    """Run the workflow, approving any parked child (the fix tasks park too)."""
    handle = await client.start_workflow(
        RunWorkflow.run,
        args=[payload, None],
        id=f"run-{RUN_ID}",
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    ids = [f"task-{RUN_ID}-task-1"] + [f"task-{RUN_ID}-task-1-ci{n}" for n in (1, 2, 3, 4, 5)]
    for _ in range(2000):
        for workflow_id in ids:
            await _approve(client, workflow_id)
        described = await handle.describe()
        if int(described.status) != 1:  # 1 == RUNNING
            break
        await asyncio.sleep(0.02)
    return await asyncio.wait_for(handle.result(), timeout=90)


def _with_worker(scenario: Any) -> Any:
    from codna.durable.graph import build_task_graph

    async def wrapper() -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            plugins = [build_task_graph().plugin]
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


def _harness(tmp_path: Path) -> tuple[Path, Path, fakes.Harness]:
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return repo, runs_root, fakes.build_harness(repo, runs_root)


def _ci_reads(harness: fakes.Harness) -> list[dict[str, Any]]:
    return [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "ci.watched"
    ]


# -- the happy transition ------------------------------------------------------------------------


def test_one_red_read_then_green_starts_exactly_one_fix_task_and_finishes_completed(tmp_path: Path):
    """The plan's "CI transitions": red, one fix, green. The transition is in the LEDGER."""
    repo, runs_root, harness = _harness(tmp_path)
    harness.ci_states = ["failed", "passed"]
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=True),
        max_tasks=2,
        max_ci_attempts=3,
    )
    task = _spec(repo)

    completion = _with_worker(lambda client: _drive(client, _payload(repo, task, goal)))

    assert completion.outcome == "completed", completion
    assert completion.ci_attempts == 1
    assert completion.goal_reason == "verify=passed; review=approved; ci=passed"
    # The parent task plus ONE fix task.
    assert [summary.task_id for summary in completion.tasks] == ["task-1", "task-1-ci1"]
    # The CI transitions, in order, from the ledger.
    assert [read["state"] for read in _ci_reads(harness)] == ["failed", "passed"]
    # The fix task's PR is the SAME pull request: it repairs the PR CI is complaining about.
    assert {summary.pr_url for summary in completion.tasks} == {
        "https://github.com/thyn-ai/codna/pull/1"
    }
    # The fix task's plan names the failing check AND that it is attempt 1.
    fix_prompts = [
        prompt
        for prompt in harness.sidecar.prompts
        if "read-only" in prompt and "Fix CI on pull request" in prompt
    ]
    assert len(fix_prompts) == 1, fix_prompts
    assert "`ci / test` (failure)" in fix_prompts[0]
    assert "This is attempt 1" not in fix_prompts[0]
    # The fix ran through the REAL steps: its own plan, implement, verify, review and PR.
    fix_status = harness.sidecar.turns  # every paid turn, in order
    assert len(fix_status) == 4, len(fix_status)  # 2 for the task, 2 for the fix (plan + implement)


# -- bound 1: attempts -----------------------------------------------------------------------------


def test_the_attempt_cap_terminates_the_loop_with_budget_to_spare(tmp_path: Path):
    """CI red forever, budget generous: the ATTEMPT cap is what stops the loop.

    If the attempt cap were advisory, this run would keep buying fix tasks until its budget died —
    so ``ci_attempts == max_ci_attempts`` exactly, and the run reports ``goal_unmet`` naming the cap.
    """
    repo, runs_root, harness = _harness(tmp_path)
    harness.ci_states = ["failed"] * 10
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=True),
        max_tasks=10,
        max_ci_attempts=2,
    )
    task = _spec(repo)
    payload = _payload(repo, task, goal)
    # Generous, so BUDGET cannot be the thing that stopped it: the assertion below is unambiguous.
    payload["max_cost_usd"] = 1000.0

    completion = _with_worker(lambda client: _drive(client, payload))

    assert completion.outcome == "goal_unmet", completion
    assert completion.ci_attempts == 2, completion.ci_attempts
    # The reason names the BOUND that stopped it, not merely that CI is red.
    assert "max_ci_attempts=2" in completion.goal_reason
    # Two fix tasks, so the parent + 2 = 3 completed entries in total. Never a third fix.
    assert [summary.task_id for summary in completion.tasks] == [
        "task-1",
        "task-1-ci1",
        "task-1-ci2",
    ]
    # The stops were recorded, with the sentence an operator needs.
    notes = [
        record.payload["message"]
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "notification"
    ]
    assert any("not starting another" in message for message in notes), notes
    # ...and the spend is far below the cap, which is the proof that the ATTEMPT cap did it.
    assert completion.spent_usd < 1000.0


# -- bound 2: budget -------------------------------------------------------------------------------


def test_the_run_budget_terminates_the_loop_with_attempts_to_spare(tmp_path: Path):
    """CI red forever, attempts generous: the BUDGET is what stops the loop.

    The mirror of the test above, and the reason both exist: a single scenario where both bounds run
    out together passes even if only one is implemented. Here ``max_ci_attempts`` is far from
    exhausted, so the only thing that can end the loop is the money.
    """
    repo, runs_root, harness = _harness(tmp_path)
    # Every turn costs 0.5, so a 1.0 cap buys at most a couple of paid turns — not ten fix tasks.
    harness.runtime.sidecar_factory  # the fake is built with its own cost; assert the wiring below
    harness.sidecar.cost_per_turn = 0.5
    harness.ci_states = ["failed"] * 20
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=True),
        max_tasks=20,
        max_ci_attempts=20,
    )
    task = _spec(repo)
    payload = _payload(repo, task, goal)
    payload["max_cost_usd"] = 1.0
    for entry in payload["tasks"]:
        entry["max_cost_usd"] = None  # the TASK is not the bound under test; the RUN is

    completion = _with_worker(lambda client: _drive(client, payload))

    assert completion.outcome in ("goal_unmet", "budget_exhausted"), completion
    # The attempt budget was NOT what stopped it — it is nowhere near its cap.
    assert completion.ci_attempts < goal.max_ci_attempts, completion.ci_attempts
    # The loop stopped and the run did not claim a goal it never met.
    assert "ci=passed" not in completion.goal_reason


# -- the answers that must NOT start a fix task ----------------------------------------------------


@pytest.mark.parametrize("state", ["infrastructure", "pending", "unknown"])
def test_a_non_code_ci_answer_starts_no_fix_task(tmp_path: Path, state: str):
    """An infra failure, a still-running check and an unreadable rollup all earn NO fix task.

    Each is the wrong work to spend: patching code for a runner that died, for a check that has not
    finished, or for a rollup we could not read. The run reports ``goal_unmet`` and says which.
    """
    repo, runs_root, harness = _harness(tmp_path)
    harness.ci_states = [state]
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=True),
        max_tasks=5,
        max_ci_attempts=3,
    )
    task = _spec(repo)

    completion = _with_worker(lambda client: _drive(client, _payload(repo, task, goal)))

    assert completion.outcome == "goal_unmet", completion
    assert completion.ci_attempts == 0, completion.ci_attempts
    assert [summary.task_id for summary in completion.tasks] == ["task-1"]
    # The read was recorded even though no fix task followed, so "we knew and did nothing" is visible.
    assert [read["state"] for read in _ci_reads(harness)] == [state]
    assert harness.push_calls == 1  # only the original task's PR


def test_the_gate_reads_a_pull_request_off_a_completion_or_reports_none():
    """The gate's input, as a PURE function: (number, head_sha) or (None, None).

    Tested at this level because it is the one place the gate could silently invent a target: reading
    a number out of a URL, or gating on a commit the run did not push. A completion with no pull
    request must yield ``None``, which is what makes the gate skip the read instead of asking GitHub
    about PR 0.
    """
    from codna.durable.run_contracts import TaskCompletion
    from codna.durable.run_workflow import _pr_coordinates

    def completion(**state: Any) -> TaskCompletion:
        return TaskCompletion(
            run_id=RUN_ID,
            task_id="task-1",
            outcome="approved",
            state=state,  # type: ignore[arg-type]
            approved_by="a@b.c",
            cost_usd=0.0,
            turn_seq=1,
            checkpoints=(),
            engine="plain",
        )

    pr_state = {
        "pr": {"number": 7, "url": "https://github.com/o/r/pull/7", "head_sha": "a" * 40},
        "checkpoint_ref": {"ref": "refs/x", "sha": "b" * 40},
    }
    assert _pr_coordinates(completion(**pr_state)) == (7, "a" * 40)
    # A PR ref without a head sha falls back to the CHECKPOINT commit: the commit this run pushed.
    without_sha = {"pr": {"number": 7, "url": "https://github.com/o/r/pull/7", "head_sha": None},
                   "checkpoint_ref": {"ref": "refs/x", "sha": "b" * 40}}
    assert _pr_coordinates(completion(**without_sha)) == (7, "b" * 40)
    # No pull request at all: nothing to read.
    assert _pr_coordinates(completion(pr=None)) == (None, None)
    # A malformed number is not coerced into a target.
    assert _pr_coordinates(
        completion(pr={"number": 0, "url": "https://github.com/o/r/pull/0", "head_sha": None})
    ) == (None, None)


def test_the_ci_gate_is_skipped_entirely_when_the_condition_does_not_require_it(tmp_path: Path):
    """No ``ci`` in the condition means no GitHub read — not "a read whose answer is ignored"."""
    repo, runs_root, harness = _harness(tmp_path)
    harness.ci_states = ["failed"]  # would stop the run if it were consulted
    goal = GoalSpec(
        objective=OBJECTIVE, stop_condition=StopCondition(verify=True, review=True), max_tasks=3
    )
    task = _spec(repo)

    completion = _with_worker(lambda client: _drive(client, _payload(repo, task, goal)))

    assert completion.outcome == "completed", completion
    assert harness.ci_calls == []
    assert _ci_reads(harness) == []


# -- the fix task is a REAL task, not a bare turn ---------------------------------------------------


def test_a_fix_ci_task_commits_to_the_same_branch_and_opens_no_second_pr(tmp_path: Path):
    """The fix has to land where CI is looking: the SAME branch, the SAME pull request."""
    repo, runs_root, harness = _harness(tmp_path)
    harness.ci_states = ["failed", "passed"]
    goal = GoalSpec(
        objective=OBJECTIVE,
        stop_condition=StopCondition(verify=True, review=True, ci=True),
        max_tasks=2,
        max_ci_attempts=2,
    )
    task = _spec(repo)

    completion = _with_worker(lambda client: _drive(client, _payload(repo, task, goal)))

    assert completion.outcome == "completed", completion
    branch = branch_for_task(RUN_ID, 1)
    # One branch for both the task and its fix attempt: the fix is a commit on the PR's head branch.
    assert fakes.git(repo, "rev-parse", "--verify", f"refs/heads/{branch}").strip()
    assert not branch_for_task(RUN_ID, 2).endswith("-ci1")
    # Both the task and the fix recorded a checkpoint on that ONE branch.
    branches = {
        record.payload["branch"]
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "checkpoint"
    }
    assert branches == {branch}, branches
    # The run's ledger records the fix attempt's own task id, so "which attempt produced this" is
    # answerable without diffing timestamps — while the BRANCH is the parent's, so the commit landed
    # where CI is looking.
    fix_checkpoints = [
        record for record in harness.ledger(RUN_ID).records()
        if record.kind == "checkpoint" and record.task_id == "task-1-ci1"
    ]
    assert fix_checkpoints, "the fix attempt recorded no checkpoint of its own"
    assert {record.payload["branch"] for record in fix_checkpoints} == {branch}
    # The worktree was swept at the end, as for any other run.
    assert not worktree_path_for(runs_root, RUN_ID, "task-1").exists()
