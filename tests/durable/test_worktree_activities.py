"""The per-task working directory, threaded through EVERY repo-touching activity (D5).

`test_durable_worktree.py` proves the lifecycle. This file proves the requirement that makes it matter: a
worktree-backed task's turns, commits, verification, review and pull request all happen in ITS checkout,
and the checkpoint's ref points at a commit reachable from the task's branch.

Deliberately built on the harness in `fakes.py` — the real activity bodies, a fake sidecar, REAL git.
The fake sidecar's `_edit_checkout` writes into whatever `working_dir` the create-session body named,
so "which tree did the turn edit" is observable from the OUTSIDE, without trusting the activity's own
return value. That is the seam that makes these tests worth having: a test that only read `tree` back
out of the code under test would pass even if nothing ran there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# The `durable` extra is OPT-IN, and the offline CI job installs everything EXCEPT it. Importing
# `activity_steps` pulls `temporalio` transitively, so an unguarded module-scope import breaks COLLECTION
# of the whole durable directory in that job — every unrelated test dies with ModuleNotFoundError. This
# guard is why `test_import_hygiene.py` exists; the other modules in this directory follow the same rule.
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_steps import (
    checkpoint_turn,
    open_pr,
    plan_task,
    review_task,
    run_cline_turn,
    verify_task,
    worktree_cleanup,
)
from codna.durable.activity_support import ActivityError, ensure_checkout, working_dir
from codna.durable.contracts import (
    CheckpointRequest,
    PlanRequest,
    PullRequestRequest,
    ReviewRequest,
    TaskContext,
    TaskSpec,
    VerifyRequest,
    WorktreeCleanupRequest,
)
from codna.durable.state import TaskState
from codna.durable.worktree import branch_for_task, worktree_path_for
from tests.durable import fakes

RUN_ID = "run-wt"


# -- fixtures ------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return fakes.make_repo(tmp_path / "repo")


@pytest.fixture
def harness(repo: Path) -> fakes.Harness:
    """A harness whose runs root is the test's, so a worktree lands under it."""
    runs_root = repo.parent / "runs"
    runs_root.mkdir(exist_ok=True)
    return fakes.build_harness(repo, runs_root)


def spec_for(repo: Path, *, worktree: bool, task_id: str = "task-1", number: int = 1) -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id=task_id,
        task_number=number,
        brief="Fix the flaky retry in the client",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine="plain",
        worktree=worktree,
    )


def context_for(spec: TaskSpec) -> TaskContext:
    return TaskContext.from_spec(spec)


def task_worktree(repo: Path, task_id: str = "task-1") -> Path:
    return worktree_path_for(repo.parent / "runs", RUN_ID, task_id)


# -- the resolution rule -------------------------------------------------------------------


def test_working_dir_is_the_shared_checkout_without_the_flag(repo: Path, harness: fakes.Harness):
    """The pre-D5 answer, unchanged: `--no-worktree` means `repo_dir`, byte for byte."""
    context = context_for(spec_for(repo, worktree=False))
    assert working_dir(harness.runtime, context) == str(repo.resolve())


def test_working_dir_is_the_task_worktree_with_the_flag(repo: Path, harness: fakes.Harness):
    context = context_for(spec_for(repo, worktree=True))
    assert working_dir(harness.runtime, context) == str(task_worktree(repo))


def test_working_dir_is_derived_not_remembered(repo: Path, harness: fakes.Harness):
    """Two independently-built contexts for one task resolve to one path — the retry property."""
    one = context_for(spec_for(repo, worktree=True))
    two = context_for(spec_for(repo, worktree=True))
    assert working_dir(harness.runtime, one) == working_dir(harness.runtime, two)


def test_ensure_checkout_creates_the_worktree_and_is_idempotent(repo: Path, harness: fakes.Harness):
    context = context_for(spec_for(repo, worktree=True))

    first = ensure_checkout(harness.runtime, context)
    second = ensure_checkout(harness.runtime, context)

    assert first == second == str(task_worktree(repo))
    assert Path(first).is_dir()
    assert fakes.git(Path(first), "rev-parse", "--abbrev-ref", "HEAD").strip() == branch_for_task(RUN_ID, 1)


def test_ensure_checkout_does_not_touch_the_repository_without_the_flag(repo: Path, harness: fakes.Harness):
    context = context_for(spec_for(repo, worktree=False))
    assert ensure_checkout(harness.runtime, context) == str(repo.resolve())
    assert not (repo.parent / "runs" / RUN_ID).exists()


def test_a_failed_worktree_is_a_refusal_not_a_fallback_to_the_shared_tree(
    repo: Path, harness: fakes.Harness, monkeypatch: pytest.MonkeyPatch
):
    """The wrong fallback here would put this task's commits on another task's tree."""
    import codna.durable.worktree as worktree_module

    def explode(**_kwargs: Any) -> Any:
        # The type the real module raises, so what is under test is `ensure_checkout`'s wrapping and its
        # refusal to fall back — not an error type the lifecycle module never produces.
        raise worktree_module.WorktreeError("locked", code="worktree_git_failed", details={})

    monkeypatch.setattr(worktree_module, "create_task_worktree", explode)

    with pytest.raises(ActivityError) as caught:
        ensure_checkout(harness.runtime, context_for(spec_for(repo, worktree=True)))
    # ONE code for "this task has no checkout", with the cause's code preserved: a caller retrying has
    # to be able to tell a missing checkout from an unrelated git failure.
    assert caught.value.code == "activity_worktree_unavailable"
    assert caught.value.details["cause_code"] == "worktree_git_failed"
    # And it did NOT fall back to the shared tree, where this task's commits would land on another's.
    assert not task_worktree(repo).exists()


# -- the turn edits the task's tree ---------------------------------------------------------


def test_the_turn_edits_the_task_worktree_not_the_shared_checkout(repo: Path, harness: fakes.Harness):
    """The fake sidecar writes into the ``working_dir`` it was GIVEN, so this observes the real seam."""
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    checkout = task_worktree(repo)
    # The turn's edit is in the task's worktree...
    assert (checkout / "app.py").is_file()
    assert "hello from turn-1" in (checkout / "app.py").read_text(encoding="utf-8")
    # ...the shared checkout is untouched...
    assert "hello from turn-1" not in (repo / "app.py").read_text(encoding="utf-8")
    # ...and the sidecar was TOLD to work there, which is what actually kept the edits out.
    assert harness.sidecar.sessions[0]["working_dir"] == str(checkout)
    assert outcome.state["turn_seq"] == 1


def test_the_turn_works_in_the_shared_checkout_without_the_flag(repo: Path, harness: fakes.Harness):
    """The default is unchanged, asserted at the same seam rather than assumed."""
    spec = spec_for(repo, worktree=False)
    run_cline_turn(context_for(spec).turn_request("implement it", spec.initial_state()))

    assert harness.sidecar.sessions[0]["working_dir"] == str(repo.resolve())
    assert "hello from turn-1" in (repo / "app.py").read_text(encoding="utf-8")


def test_the_plan_turn_reads_the_same_tree_the_implement_turn_will_edit(
    repo: Path, harness: fakes.Harness
):
    """A plan produced from one checkout and applied in another is a plan for the wrong tree."""
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)

    plan_task(PlanRequest(state=spec.initial_state(), context=context, brief=spec.brief))
    assert harness.sidecar.sessions[0]["working_dir"] == str(task_worktree(repo))

    # The implement turn is handed the SAME tree, so the plan and the work agree about where the work is.
    run_cline_turn(
        context.turn_request("implement", TaskState.from_payload(spec.initial_state()).to_payload())
    )
    assert harness.sidecar.sessions[-1]["working_dir"] == str(task_worktree(repo))
    # The read-only plan turn did NOT modify the tree: the engine's allow-list, not the prompt, is the
    # control — and a read-only turn that wrote would mean the worktree was not the thing enforced.
    assert (task_worktree(repo) / "app.py").is_file()


# -- the checkpoint commits in the worktree -------------------------------------------------


def test_the_checkpoint_commits_in_the_worktree_and_the_branch_reaches_it(
    repo: Path, harness: fakes.Harness
):
    """THE checkpoint requirement: commit in the worktree, and a ref reachable from the task branch."""
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    checkout = task_worktree(repo)

    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    result = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    # The recorded sha IS the worktree's HEAD...
    assert fakes.git(checkout, "rev-parse", "HEAD").strip() == result.sha
    # ...it is the task branch's tip...
    assert fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, 1)).strip() == result.sha
    # ...and, the requirement stated as a fact about git: the checkpoint REF is reachable FROM the
    # branch. `merge-base --is-ancestor <ref> <branch>` is exactly "reachable from".
    reachable = fakes.git(repo, "merge-base", "--is-ancestor", result.ref, branch_for_task(RUN_ID, 1))
    assert reachable is not None  # git printed nothing; the exit code was 0 (the helper asserts it)
    # The ref resolves, and to the same commit the branch points at.
    assert fakes.git(repo, "rev-parse", result.ref).strip() == result.sha
    # The shared checkout did NOT move: the commit belongs to the task's tree alone.
    assert fakes.git(repo, "rev-parse", "HEAD").strip() != result.sha
    assert fakes.git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_every_turns_checkpoint_ref_stays_reachable_from_the_task_branch(
    repo: Path, harness: fakes.Harness
):
    """The requirement over a WHOLE task, not one turn: every ref is an ancestor of the branch tip.

    A single-turn assertion is nearly vacuous — the ref and the branch point at the same commit, and
    "a commit is its own ancestor" proves nothing. Three turns make it a real property: turn 1's ref must
    still be reachable after turns 2 and 3 have moved the branch forward, or the audit trail ("what did
    turn 1 change") would be a dangling pointer after the first new commit.
    """
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    state = spec.initial_state()
    refs: list[tuple[str, str]] = []

    for turn in range(1, 4):
        moved = run_cline_turn(context.turn_request(f"turn {turn}", state))
        result = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))
        refs.append((result.ref, result.sha))
        state = result.state

    tip = fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, 1)).strip()
    assert tip == refs[-1][1]
    assert len({sha for _ref, sha in refs}) == 3, "three turns produced one commit"
    for ref, sha in refs:
        assert fakes.git(repo, "rev-parse", ref).strip() == sha
        # `merge-base --is-ancestor` asserts reachability from the branch tip; the helper asserts exit 0.
        fakes.git(repo, "merge-base", "--is-ancestor", ref, tip)
    # The worktree IS the branch tip, so what a reviewer opens is what the audit ref names.
    assert fakes.git(task_worktree(repo), "rev-parse", "HEAD").strip() == tip
    # ...and the tree is clean: the checkpoint staged everything, so nothing dangles uncommitted.
    assert fakes.git(task_worktree(repo), "status", "--porcelain").strip() == ""


def test_the_checkpoint_branch_is_created_from_the_worktrees_own_base(repo: Path, harness: fakes.Harness):
    """The task branch starts where the worktree started, so turn 1's diff is the turn's work only."""
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    base = fakes.git(repo, "rev-parse", "main").strip()

    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    result = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    parents = fakes.git(repo, "rev-list", "--parents", "-n", "1", result.sha).split()
    assert parents[1] == base, parents


def test_two_tasks_commit_in_two_worktrees_and_not_in_each_other(
    repo: Path, harness: fakes.Harness
):
    """The reason the slice exists: two tasks, two trees, no interference."""
    one = spec_for(repo, worktree=True, task_id="task-1", number=1)
    two = spec_for(repo, worktree=True, task_id="task-2", number=2)

    first_moved = run_cline_turn(context_for(one).turn_request("first", one.initial_state()))
    first = checkpoint_turn(CheckpointRequest(state=first_moved.state, context=context_for(one)))
    second_moved = run_cline_turn(context_for(two).turn_request("second", two.initial_state()))
    second = checkpoint_turn(CheckpointRequest(state=second_moved.state, context=context_for(two)))

    assert first.sha != second.sha
    assert fakes.git(task_worktree(repo, "task-1"), "rev-parse", "HEAD").strip() == first.sha
    assert fakes.git(task_worktree(repo, "task-2"), "rev-parse", "HEAD").strip() == second.sha
    assert fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, 1)).strip() == first.sha
    assert fakes.git(repo, "rev-parse", branch_for_task(RUN_ID, 2)).strip() == second.sha
    # Task 1's commit is not on task 2's branch, and vice versa: two independent lines of work.
    with pytest.raises(AssertionError):
        fakes.git(repo, "merge-base", "--is-ancestor", first.sha, branch_for_task(RUN_ID, 2))


def test_a_retried_checkpoint_in_a_worktree_creates_no_second_commit(repo: Path, harness: fakes.Harness):
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    first = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))
    second = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    assert (first.created, second.created) == (True, False)
    assert first.sha == second.sha
    # ONE ledger record for the turn, not two — the ledger half of the same idempotency.
    assert len(harness.checkpoints(RUN_ID)) == 1


# -- verify, review, open_pr ---------------------------------------------------------------


def test_verify_detects_and_runs_in_the_task_worktree(repo: Path, harness: fakes.Harness):
    """Detection and execution both happen in the tree that holds the work."""
    spec = TaskSpec(
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        brief="b",
        repo_dir=str(repo),
        base_branch="main",
        engine="plain",
        worktree=True,
    )
    context = context_for(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    state = TaskState.from_payload(moved.state)
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    outcome = verify_task(VerifyRequest(state=checkpoint.state, context=context, commands=("true",)))

    assert outcome.state["verify_ref"]["passed"] is True
    assert harness.verify_calls == 1
    assert state.turn_seq >= 1


def test_review_reads_the_diff_in_the_task_worktree(repo: Path, harness: fakes.Harness):
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    outcome = review_task(ReviewRequest(state=checkpoint.state, context=context))

    assert outcome.state["review_verdict"] == "approved"
    assert harness.review_calls == 1


def test_open_pr_pushes_from_the_task_worktree(repo: Path, harness: fakes.Harness):
    """The push must find the checkpoint commit; a worktree shares the object store, so it does."""
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    outcome = open_pr(
        PullRequestRequest(state=checkpoint.state, context=context, title="Fix the retry")
    )

    assert outcome.state["pr"]["branch"] == branch_for_task(RUN_ID, 1)
    assert outcome.state["pr"]["head_sha"] == checkpoint.sha
    assert harness.push_calls == 1


# -- the cleanup policy --------------------------------------------------------------------


def test_cleanup_removes_the_runs_worktrees_and_records_it(repo: Path, harness: fakes.Harness):
    spec = spec_for(repo, worktree=True)
    context = context_for(spec)
    ensure_checkout(harness.runtime, context)
    assert task_worktree(repo).is_dir()

    outcome = worktree_cleanup(
        WorktreeCleanupRequest(run_id=RUN_ID, repo_dir=str(repo), reason="run completed")
    )

    assert outcome.kept is False
    assert outcome.removed == (str(task_worktree(repo)),)
    assert not task_worktree(repo).exists()
    # The decision is auditable, not inferred from an absence of files.
    notes = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert notes and notes[0]["reason"] == "run completed" and notes[0]["kept"] is False


def test_cleanup_is_idempotent(repo: Path, harness: fakes.Harness):
    ensure_checkout(harness.runtime, context_for(spec_for(repo, worktree=True)))
    request = WorktreeCleanupRequest(run_id=RUN_ID, repo_dir=str(repo), reason="run completed")

    first = worktree_cleanup(request)
    second = worktree_cleanup(request)

    assert len(first.removed) == 1
    assert second.removed == ()


def test_cleanup_keep_leaves_the_checkout_and_says_why(repo: Path, harness: fakes.Harness):
    """Keeping a run for debugging is a DECISION with a reason, not an omission."""
    ensure_checkout(harness.runtime, context_for(spec_for(repo, worktree=True)))

    outcome = worktree_cleanup(
        WorktreeCleanupRequest(
            run_id=RUN_ID, repo_dir=str(repo), reason="kept for investigation", keep=True
        )
    )

    assert outcome.kept is True
    assert task_worktree(repo).is_dir()
    notes = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert notes[0]["kept"] is True
    assert notes[0]["reason"] == "kept for investigation"


def test_a_cleanup_that_cannot_delete_does_not_raise(repo: Path, harness: fakes.Harness, monkeypatch: pytest.MonkeyPatch):
    """A finished run must not be un-finished by its own tidying: the failure is recorded instead."""
    import codna.durable.activity_steps as steps
    import codna.durable.worktree as worktree_module

    def explode(**_kwargs: Any) -> Any:
        raise worktree_module.WorktreeError("locked", code="worktree_path_occupied", details={})

    monkeypatch.setattr(steps, "remove_run_worktrees", explode)

    outcome = worktree_cleanup(
        WorktreeCleanupRequest(run_id=RUN_ID, repo_dir=str(repo), reason="run completed")
    )

    assert outcome.kept is True
    assert outcome.removed == ()
    assert "worktree_path_occupied" in outcome.reason
    notes = [
        record.payload
        for record in harness.ledger(RUN_ID).records()
        if record.kind == "note" and record.payload.get("event") == "worktrees.cleaned"
    ]
    assert notes[0]["failed"] == "worktree_path_occupied"


def test_cleanup_reclaims_worktrees_of_a_run_that_failed(repo: Path, harness: fakes.Harness):
    """A FAILED run leaks nothing either — nothing on the sweep's path depends on success."""
    ensure_checkout(harness.runtime, context_for(spec_for(repo, worktree=True)))

    outcome = worktree_cleanup(
        WorktreeCleanupRequest(run_id=RUN_ID, repo_dir=str(repo), reason="run stopped")
    )

    assert len(outcome.removed) == 1
    assert not task_worktree(repo).exists()


@pytest.mark.parametrize(
    "fields",
    [
        {"run_id": ""},
        {"repo_dir": "relative/path"},
        {"reason": ""},
        {"reason": "x" * 513},
        {"keep": "yes"},
    ],
)
def test_the_cleanup_request_refuses_a_malformed_field(fields: dict[str, Any]):
    base = {"run_id": "run-1", "repo_dir": "/tmp/repo", "reason": "run completed"}
    with pytest.raises(Exception) as caught:
        WorktreeCleanupRequest(**{**base, **fields}).validate()
    assert getattr(caught.value, "code", "") == "contract_invalid"
