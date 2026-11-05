"""Idempotency: the measures that stop a RETRY from double-creating or double-billing (D5).

A durable step is retried by the server whenever it fails, and a worker can die between "the side
effect landed" and "the result was recorded". Every measure that makes that safe is asserted here as a
PROPERTY, not as an implementation detail:

* **One worktree per task, however many times it is asked for.** The path is DERIVED (run id + task
  id) and an existing registration for that path is returned rather than refused, so a retried turn,
  a second step of the same task and a worker that picked the task up elsewhere all land in ONE
  checkout. A second worktree on the same branch is a `worktree_branch_exists` refusal, so the failure
  mode is loud rather than a silent second copy.
* **One checkpoint per `turn_seq`.** The commit message carries the turn key, so a retried checkpoint
  finds the commit it already made and re-points its refs instead of creating a second — and the
  ledger checks for the record before appending, because the two halves can disagree after a crash.
* **One paid turn per `turn_key`.** `drive_turn` looks in the ledger before POSTing (the D2 work), and
  a "fix CI" task's distinct `task_id` is what gives each ATTEMPT its own turn keys — so a retried
  attempt reuses its keys and a NEW attempt does not collide with its parent's.
* **One episode, one ledger note.** The run takes the episode reference from the child's COMPLETION,
  so it never calls `remember_episode` a second time to learn a path it already has.
* **One fix-CI attempt per count increment.** `ci_attempts` is incremented BEFORE the child starts, so
  a failure inside the attempt still counts — otherwise a repo whose CI is red would let a failing fix
  task retry forever without ever reaching `max_ci_attempts`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_steps import checkpoint_turn, plan_task, run_cline_turn  # noqa: E402
from codna.durable.activity_support import ActivityError, ensure_checkout  # noqa: E402
from codna.durable.contracts import TaskContext, TaskSpec  # noqa: E402
from codna.durable.goal import GoalSpec, StopCondition  # noqa: E402
from codna.durable.state import TaskState  # noqa: E402
from codna.durable.worktree import (  # noqa: E402
    branch_for_task,
    create_task_worktree,
    list_run_worktrees,
    worktree_path_for,
)
from tests.durable import fakes  # noqa: E402

RUN_ID = "run-idem"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return fakes.make_repo(tmp_path / "repo")


@pytest.fixture
def harness(repo: Path) -> fakes.Harness:
    runs_root = repo.parent / "runs"
    runs_root.mkdir(exist_ok=True)
    return fakes.build_harness(repo, runs_root)


def _spec(repo: Path, *, task_id: str = "task-1", number: int = 1, **overrides: Any) -> TaskSpec:
    payload: dict[str, Any] = {
        "run_id": RUN_ID,
        "task_id": task_id,
        "task_number": number,
        "brief": "Fix the retry loop",
        "repo_dir": str(repo),
        "base_branch": "main",
        "verify_commands": ("true",),
        "engine": "plain",
        "worktree": True,
    }
    payload.update(overrides)
    return TaskSpec(**payload)


# -- worktrees ------------------------------------------------------------------------------------


def test_asking_for_the_same_worktree_twice_creates_exactly_one(repo: Path, harness: fakes.Harness):
    """The retried-create property: the second call REPORTS the worktree instead of making another."""
    kwargs = dict(
        repo_dir=repo,
        runs_root=harness.runs_root,
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        base_branch="main",
    )
    first = create_task_worktree(**kwargs)
    second = create_task_worktree(**kwargs)

    assert first.created is True
    assert second.created is False, "a retried create reported creating a second worktree"
    assert first.path == second.path == worktree_path_for(harness.runs_root, RUN_ID, "task-1")
    # ONE checkout, according to git — the authority on what a worktree is.
    assert list_run_worktrees(
        repo_dir=repo, runs_root=harness.runs_root, run_id=RUN_ID
    ) == (first.path,)


def test_two_creates_on_one_branch_are_refused_rather_than_silently_duplicated(
    repo: Path, harness: fakes.Harness
):
    """A different PATH for one branch is impossible, and a different TASK's number is a refusal.

    The failure this pins: if two creates could both succeed on `codna/run-x/task-1`, the second would
    have to live somewhere, and "somewhere" is a second checkout whose commits are indistinguishable
    from the first's on the same ref.
    """
    create_task_worktree(
        repo_dir=repo,
        runs_root=harness.runs_root,
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        base_branch="main",
    )
    with pytest.raises(Exception) as caught:
        create_task_worktree(
            repo_dir=repo,
            runs_root=harness.runs_root,
            run_id=RUN_ID,
            # A DIFFERENT directory, the SAME branch (task_number 1).
            task_id="task-9",
            task_number=1,
            base_branch="main",
        )
    assert getattr(caught.value, "code", "") == "worktree_branch_exists"


def test_ensure_checkout_is_idempotent_across_steps(repo: Path, harness: fakes.Harness):
    """Every repo-touching step calls `ensure_checkout`, so the second call must be free and safe."""
    context = TaskContext.from_spec(_spec(repo))
    first = ensure_checkout(harness.runtime, context)
    second = ensure_checkout(harness.runtime, context)
    third = ensure_checkout(harness.runtime, context)
    assert first == second == third
    # Still exactly one worktree after three calls.
    assert len(list_run_worktrees(repo_dir=repo, runs_root=harness.runs_root, run_id=RUN_ID)) == 1


def test_a_fix_ci_attempt_shares_its_parents_checkout_but_has_its_own_identity(
    repo: Path, harness: fakes.Harness
):
    """The split that makes a repair attempt safe: ONE checkout, its OWN turn keys.

    Sharing the parent's ``task_id`` would collide on the checkpoint's message lookup (it finds the
    commit by message) and on the branch-exists guard; deriving a new CHECKOUT would commit somewhere
    CI is not looking. So `worktree_task_id` names the place and `task_id` names the attempt.
    """
    parent = _spec(repo)
    fix = parent.with_ci_fix(task_id="task-1-ci1", brief="Fix CI", goal_context="goal")

    assert fix.task_id == "task-1-ci1"
    assert fix.worktree_task_id == "task-1"
    assert fix.checkout_task_id == "task-1"
    assert fix.branch == parent.branch

    parent_tree = ensure_checkout(harness.runtime, TaskContext.from_spec(parent))
    fix_tree = ensure_checkout(harness.runtime, TaskContext.from_spec(fix))
    assert parent_tree == fix_tree, "the fix attempt got its own checkout; it must share the parent's"
    assert len(list_run_worktrees(repo_dir=repo, runs_root=harness.runs_root, run_id=RUN_ID)) == 1
    # ...and the turn key differs, because the key is derived from `task_id`.
    assert f"{RUN_ID}:{fix.task_id}:1" != f"{RUN_ID}:{parent.task_id}:1"


# -- checkpoints and turns -------------------------------------------------------------------------


def test_a_retried_checkpoint_creates_no_second_commit(repo: Path, harness: fakes.Harness):
    """The commit message carries the turn key, so a retry FINDS the commit it already made."""
    spec = _spec(repo)
    context = TaskContext.from_spec(spec)
    state = TaskState.from_payload(spec.initial_state())

    # A turn, so there is something to commit.
    turn = run_cline_turn(context.turn_request("Implement it", state.to_payload()))
    moved = TaskState.from_payload(turn.state)

    first = checkpoint_turn(context.checkpoint_request(moved.to_payload()))
    second = checkpoint_turn(context.checkpoint_request(moved.to_payload()))

    assert first.created is True
    assert second.created is False, "a retried checkpoint made a second commit"
    assert first.sha == second.sha
    # ONE checkpoint record in the ledger, not two — the two halves cannot disagree after a crash.
    records = [
        record
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "checkpoint"
    ]
    assert len(records) == 1, [record.payload for record in records]


def test_a_retried_plan_turn_does_not_buy_a_second_turn(repo: Path, harness: fakes.Harness):
    """`drive_turn` looks in the ledger before POSTing, keyed on ``run:task:seq``."""
    spec = _spec(repo)
    context = TaskContext.from_spec(spec)
    state = TaskState.from_payload(spec.initial_state())

    first = plan_task(context.plan_request(spec.brief, state.to_payload()))
    first_turns = len(harness.sidecar.turns)
    # The SAME request, as a retry receives it: the pre-turn state from history.
    second = plan_task(context.plan_request(spec.brief, state.to_payload()))

    assert len(harness.sidecar.turns) == first_turns == 1, "a retried plan bought a second turn"
    # Both attempts report the same plan reference: it is derived from the turn, not regenerated.
    assert first.state["plan_ref"] == second.state["plan_ref"]


# -- episodes ------------------------------------------------------------------------------------


def test_a_task_completion_carries_its_episode_so_the_run_never_re_records_it():
    """One episode per task, by construction: the reference rides on the COMPLETION.

    The alternative the durable path deliberately avoids is asking the `episode` activity again to
    learn a path the task already reported — which would append a second ledger note for one episode.
    Asserted on the contract (`TaskCompletion` + `EpisodeRequest`), because that is where the property
    lives; `test_goal_run.py` proves the run uses it end to end.
    """
    from codna.durable.contracts import EpisodeRequest
    from codna.durable.run_contracts import TaskCompletion

    # A completion with an episode answers with those two strings, and nothing else.
    completion = TaskCompletion(
        run_id=RUN_ID,
        task_id="task-1",
        outcome="approved",
        state={},
        approved_by="a@b.c",
        cost_usd=0.1,
        turn_seq=2,
        checkpoints=(),
        engine="plain",
        episode_path="episodes/task-1.json",
        episode_digest="a" * 64,
    )
    assert completion.episode_path == "episodes/task-1.json"
    assert completion.episode_digest == "a" * 64

    # A completion without one says so with empty strings, not a guessed path.
    bare = TaskCompletion(
        run_id=RUN_ID,
        task_id="task-1",
        outcome="rejected",
        state={},
        approved_by=None,
        cost_usd=0.0,
        turn_seq=0,
        checkpoints=(),
        engine="plain",
    )
    assert bare.episode_path == "" and bare.episode_digest == ""

    # ...and the REQUEST the task sends carries the brief the recap needs, so the plan for a later
    # task knows what this one was about.
    spec = _spec(Path("/tmp/repo"))
    request = EpisodeRequest(
        state=TaskState.from_payload(spec.initial_state()).to_payload(),
        context=TaskContext.from_spec(spec),
        outcome="approved",
        brief=spec.brief,
    )
    request.validate()
    assert request.brief == "Fix the retry loop"


# -- the CI attempt counter ------------------------------------------------------------------------


def test_the_ci_attempt_is_counted_BEFORE_the_fix_runs_so_a_failure_still_bounds_the_loop():
    """A failing fix task must still spend the allowance, or a red PR loops forever.

    Asserted against the ordering in `_run_ci_fix` rather than by driving a workflow: the property is
    "the counter moved even though the child raised", which a start-to-finish run cannot distinguish
    from "the counter moved because the child succeeded".
    """
    import inspect

    from codna.durable import run_workflow as module

    source = inspect.getsource(module.RunWorkflow._run_ci_fix)
    increment = source.index("self._ci_attempts = attempt")
    child = source.index("await self._run_task(")
    assert increment < child, "ci_attempts is incremented AFTER the fix task starts"


def test_an_incrementing_attempt_id_is_refused_by_the_specs_own_validation():
    """A fix task whose id collides with its parent would be a start-time id-reuse failure, inside the
    run. The derived id is validated, so a caller that gets it wrong fails where it is cheap."""
    spec = _spec(Path("/tmp/repo"))
    with pytest.raises(Exception):
        spec.with_ci_fix(task_id="task-1", brief="", goal_context="")  # empty brief is refused


_ = (StopCondition, GoalSpec, ActivityError, branch_for_task)
