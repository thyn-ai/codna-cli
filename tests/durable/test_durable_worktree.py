"""Worktree-per-task: the lifecycle, and the properties that make it safe (D5).

Everything here runs REAL git against a real repository in a temp directory. That is the point: the
properties under test are git's own (a worktree has its own HEAD, a branch cannot be checked out twice,
`worktree remove --force` leaves no directory behind), so a mocked git would test the mock and prove
nothing about the shipped path.

What is pinned, in the order the module's docstring promises it:

* the EXACT path and the EXACT branch name, at the plan's literal layout;
* a task's commits land in its worktree and NOT in the shared checkout — the requirement the whole
  slice exists for;
* create is idempotent for the thing it was asked to create, and refuses to clobber anything else;
* remove is idempotent, and refuses to delete a directory that is merely in the way;
* a path outside the runs root is refused for BOTH create and remove;
* a failure mid-create leaves no half state — asserted against git's own worktree list and branch list,
  because "no half state" means exactly "git still reports what it reported before".
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codna.durable.worktree import (
    GIT_ENV_DROP,
    WorktreeError,
    branch_for_task,
    create_task_worktree,
    list_run_worktrees,
    remove_run_worktrees,
    remove_task_worktree,
    worktree_path_for,
)

# -- a real repository ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr}"
    return completed.stdout


def git_ok(repo: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    ).returncode == 0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repository on ``main`` with one commit."""
    path = tmp_path / "repo"
    path.mkdir()
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@codna.ai")
    git(path, "config", "user.name", "test")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


def create(repo: Path, runs_root: Path, *, run_id: str = "run-1", task_id: str = "task-1", n: int = 1):
    return create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id=run_id,
        task_id=task_id,
        task_number=n,
        base_branch="main",
    )


# -- naming --------------------------------------------------------------------------------


def test_the_path_and_the_branch_are_the_plans_literal_layout(tmp_path: Path):
    """``~/.codna/runs/<run>/wt-<task>`` on ``codna/run-<run>/task-<n>`` — asserted, not described."""
    root = tmp_path / "runs"
    assert worktree_path_for(root, "run-1", "task-2") == root / "run-1" / "wt-task-2"
    assert branch_for_task("run-1", 2) == "codna/run-run-1/task-2"


def test_the_branch_matches_the_contracts_own_branch_name():
    """ONE branch name per task: the lifecycle module and the contract must not disagree.

    The contracts module cannot import this one (a workflow imports the contracts, and this module runs
    git), so the two are pinned by a test instead of by an import.
    """
    from codna.durable.contracts import TaskSpec

    spec = TaskSpec(
        run_id="run-7", task_id="task-3", task_number=3, brief="b", repo_dir="/tmp"
    )
    assert spec.branch == branch_for_task("run-7", 3)


@pytest.mark.parametrize("value", ["", "..", "a/b", ".hidden", "a" * 200, "/abs"])
def test_a_run_or_task_id_that_could_escape_is_refused(value: str, tmp_path: Path):
    """The tokens become a directory name AND a git ref component, so the class is narrow."""
    with pytest.raises(WorktreeError) as caught:
        worktree_path_for(tmp_path, value, "task-1")
    assert caught.value.code == "worktree_input_invalid"


@pytest.mark.parametrize("number", [0, -1, True, "1", None])
def test_a_bad_task_number_is_refused(number: object):
    with pytest.raises(WorktreeError) as caught:
        branch_for_task("run-1", number)  # type: ignore[arg-type]
    assert caught.value.code == "worktree_input_invalid"


# -- create --------------------------------------------------------------------------------


def test_create_puts_the_worktree_at_the_exact_path_on_the_exact_branch(repo: Path, runs_root: Path):
    made = create(repo, runs_root)

    assert made.created is True
    assert made.path == (runs_root / "run-1" / "wt-task-1").resolve()
    assert made.path.is_dir()
    # The branch is REAL and checked out HERE, verified by git rather than by our own record.
    assert git(made.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "codna/run-run-1/task-1"
    # ...and git agrees this path is a worktree of the repository, at that branch.
    listed = git(repo, "worktree", "list", "--porcelain")
    assert str(made.path) in listed
    assert "refs/heads/codna/run-run-1/task-1" in listed


def test_the_worktree_starts_at_the_base_branch_commit(repo: Path, runs_root: Path):
    """A task's tree starts where the run said, not wherever the shared checkout's HEAD happened to be."""
    base = git(repo, "rev-parse", "main").strip()
    # Move the shared checkout somewhere else entirely; the worktree must still start at main.
    git(repo, "checkout", "-q", "-b", "unrelated")
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "unrelated work")

    made = create(repo, runs_root)
    assert git(made.path, "rev-parse", "HEAD").strip() == base


def test_a_task_commits_in_its_worktree_and_NOT_in_the_shared_checkout(repo: Path, runs_root: Path):
    """THE requirement: the checkout the task works in is the one its commit lands in."""
    made = create(repo, runs_root)
    before_root = git(repo, "rev-parse", "HEAD").strip()
    before_branch = git(repo, "rev-parse", "codna/run-run-1/task-1").strip()

    (made.path / "app.py").write_text("def hello():\n    return 'worktree'\n", encoding="utf-8")
    git(made.path, "add", "-A")
    git(made.path, "commit", "-q", "-m", "codna: turn 1")
    task_sha = git(made.path, "rev-parse", "HEAD").strip()

    # The task branch moved to the new commit...
    assert git(repo, "rev-parse", "codna/run-run-1/task-1").strip() == task_sha
    assert task_sha != before_branch
    # ...the SHARED checkout's HEAD did not move with it...
    assert git(repo, "rev-parse", "HEAD").strip() == before_root
    # ...and the shared checkout's tree is untouched, which is what "not in the repo root" means.
    assert "worktree" not in (repo / "app.py").read_text(encoding="utf-8")
    # The worktree is on that branch and its tree carries the change.
    assert git(made.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "codna/run-run-1/task-1"
    assert "worktree" in (made.path / "app.py").read_text(encoding="utf-8")


def test_a_second_task_gets_a_different_worktree_and_a_different_branch(repo: Path, runs_root: Path):
    first = create(repo, runs_root, task_id="task-1", n=1)
    second = create(repo, runs_root, task_id="task-2", n=2)

    assert first.path != second.path
    assert first.branch != second.branch
    assert second.path.name == "wt-task-2"
    assert git(second.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "codna/run-run-1/task-2"
    # Two independent HEADs: a commit in one is NOT visible in the other's working tree.
    (first.path / "one.txt").write_text("1\n", encoding="utf-8")
    git(first.path, "add", "-A")
    git(first.path, "commit", "-q", "-m", "one")
    assert not (second.path / "one.txt").exists()


def test_a_second_run_gets_its_own_directory(repo: Path, runs_root: Path):
    """Run ids are part of the path, so two runs never share a checkout."""
    one = create(repo, runs_root, run_id="run-1")
    two = create(repo, runs_root, run_id="run-2")
    assert one.path == (runs_root / "run-1" / "wt-task-1").resolve()
    assert two.path == (runs_root / "run-2" / "wt-task-1").resolve()
    assert two.branch == "codna/run-run-2/task-1"


def test_create_is_idempotent_for_the_worktree_it_was_asked_for(repo: Path, runs_root: Path):
    """A retried activity must find its own worktree and continue, not fail."""
    first = create(repo, runs_root)
    second = create(repo, runs_root)

    assert first.created is True
    assert second.created is False
    assert second.path == first.path
    assert second.branch == first.branch
    # And it did not disturb the work in progress.
    (first.path / "work.txt").write_text("in progress\n", encoding="utf-8")
    third = create(repo, runs_root)
    assert third.created is False
    assert (first.path / "work.txt").is_file()


def test_create_refuses_a_path_that_already_exists(repo: Path, runs_root: Path):
    """A directory in the way is never deleted or written into — and leaves NO orphan branch."""
    occupied = runs_root / "run-1" / "wt-task-1"
    occupied.mkdir(parents=True)
    (occupied / "keep.txt").write_text("someone's work\n", encoding="utf-8")

    with pytest.raises(WorktreeError) as caught:
        create(repo, runs_root)
    assert caught.value.code == "worktree_path_exists"
    # The directory and its contents are untouched.
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "someone's work\n"
    # THE ORDER MATTERS HERE: `git worktree add -b` creates the branch FIRST and would leave it behind.
    # The worktree-then-branch order leaves nothing, which is asserted rather than trusted.
    assert not git_ok(repo, "rev-parse", "--verify", "refs/heads/codna/run-run-1/task-1")
    assert str(occupied) not in git(repo, "worktree", "list")


def test_create_refuses_a_branch_that_already_exists(repo: Path, runs_root: Path):
    """An existing branch is a collision, not something to force over."""
    git(repo, "branch", "codna/run-run-1/task-1")

    with pytest.raises(WorktreeError) as caught:
        create(repo, runs_root)
    assert caught.value.code == "worktree_branch_exists"
    assert not (runs_root / "run-1" / "wt-task-1").exists()


def test_create_refuses_a_path_outside_the_runs_root(repo: Path, tmp_path: Path):
    """The guard runs before any git command, so nothing outside the root is ever created."""
    escape = tmp_path / "elsewhere"
    with pytest.raises(WorktreeError) as caught:
        create_task_worktree(
            repo_dir=repo,
            runs_root=escape,
            run_id="..",
            task_id="task-1",
            task_number=1,
            base_branch="main",
        )
    assert caught.value.code == "worktree_input_invalid"
    assert not escape.exists()


def test_create_refuses_an_unknown_base_branch(repo: Path, runs_root: Path):
    """A per-task worktree must NAME its base; a typo is a refusal, not "whatever HEAD is"."""
    with pytest.raises(WorktreeError) as caught:
        create_task_worktree(
            repo_dir=repo,
            runs_root=runs_root,
            run_id="run-1",
            task_id="task-1",
            task_number=1,
            base_branch="not-a-branch",
        )
    assert caught.value.code == "worktree_base_missing"
    assert not (runs_root / "run-1" / "wt-task-1").exists()


def test_create_refuses_a_directory_that_is_not_a_git_repository(tmp_path: Path, runs_root: Path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    with pytest.raises(WorktreeError) as caught:
        create(plain, runs_root)
    assert caught.value.code == "worktree_repo_not_git"


def test_a_failure_mid_create_leaves_no_half_state(repo: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch):
    """The order's whole point: after a failed create, git reports what it reported before.

    The failure is injected at the SECOND step (the branch checkout), which is the only place the
    worktree-then-branch order can leave anything behind. Before the fix this raises a real orphan
    worktree; the assertion is that both halves of git's state — worktrees and branches — are unchanged.
    """
    worktrees_before = git(repo, "worktree", "list", "--porcelain")
    branches_before = git(repo, "branch", "--format=%(refname)")

    import codna.durable.worktree as module

    def explode(path: Path, branch: str) -> None:
        raise subprocess.TimeoutExpired(cmd="git checkout", timeout=1)

    monkeypatch.setattr(module, "_checkout_branch", explode)

    with pytest.raises(subprocess.TimeoutExpired):
        create(repo, runs_root)

    # Nothing left over: not a worktree, not a branch, not a directory.
    assert git(repo, "worktree", "list", "--porcelain") == worktrees_before
    assert git(repo, "branch", "--format=%(refname)") == branches_before
    assert not (runs_root / "run-1" / "wt-task-1").exists()
    # And a retry (with the real checkout) now succeeds, which is what "recoverable" means.
    monkeypatch.undo()
    made = create(repo, runs_root)
    assert made.created is True


def test_create_completes_a_worktree_left_detached_by_an_interrupted_create(repo: Path, runs_root: Path):
    """The ONE reachable half state is COMPLETED, not refused: the branch does not exist yet.

    A worker killed between ``worktree add --detach`` and ``checkout -b`` leaves a registered worktree
    on a detached HEAD. Refusing it would wedge the task forever, so create finishes the job — and the
    "branch already exists" guard still catches the case where it cannot.
    """
    git(repo, "worktree", "add", "--detach", str(runs_root / "run-1" / "wt-task-1"), "main")

    made = create(repo, runs_root)
    assert made.created is False
    assert git(made.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "codna/run-run-1/task-1"


def test_create_refuses_a_path_that_is_a_worktree_on_another_branch(repo: Path, runs_root: Path):
    """The same path on a different branch is somebody else's checkout, so it is a refusal."""
    git(repo, "worktree", "add", "-b", "someone-elses", str(runs_root / "run-1" / "wt-task-1"), "main")

    with pytest.raises(WorktreeError) as caught:
        create(repo, runs_root)
    assert caught.value.code == "worktree_path_conflict"
    assert caught.value.details["found"] == "someone-elses"


def test_create_prunes_a_registration_whose_directory_is_gone(repo: Path, runs_root: Path):
    """A stale admin entry (a manual `rm -rf`) must not block the task forever."""
    made = create(repo, runs_root)
    import shutil

    shutil.rmtree(made.path)

    again = create(repo, runs_root)
    assert again.created is True
    assert again.path.is_dir()


# -- remove --------------------------------------------------------------------------------


def test_remove_reclaims_the_directory_and_is_idempotent(repo: Path, runs_root: Path):
    made = create(repo, runs_root)

    assert remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-1", task_id="task-1") is True
    assert not made.path.exists()
    assert str(made.path) not in git(repo, "worktree", "list")
    # The second call is an ANSWER, not a failure — this is what a retried cleanup sees.
    assert remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-1", task_id="task-1") is False


def test_remove_keeps_the_commits(repo: Path, runs_root: Path):
    """A worktree is a CHECKOUT; the work is the branch, which survives its removal."""
    made = create(repo, runs_root)
    (made.path / "app.py").write_text("def hello():\n    return 'kept'\n", encoding="utf-8")
    git(made.path, "add", "-A")
    git(made.path, "commit", "-q", "-m", "codna: turn 1")
    sha = git(made.path, "rev-parse", "HEAD").strip()

    remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-1", task_id="task-1")

    assert git(repo, "rev-parse", "codna/run-run-1/task-1").strip() == sha
    assert "kept" in git(repo, "show", f"{sha}:app.py")


def test_remove_forces_away_untracked_files_but_not_a_strangers_directory(
    repo: Path, runs_root: Path
):
    """``--force`` is for a turn's leftovers; a NON-worktree directory is never deleted."""
    made = create(repo, runs_root)
    (made.path / "build.log").write_text("noise\n", encoding="utf-8")
    assert remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-1", task_id="task-1") is True

    stranger = runs_root / "run-1" / "wt-task-1"
    stranger.mkdir(parents=True)
    (stranger / "precious.txt").write_text("keep me\n", encoding="utf-8")
    with pytest.raises(WorktreeError) as caught:
        remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-1", task_id="task-1")
    assert caught.value.code == "worktree_path_occupied"
    assert (stranger / "precious.txt").is_file()


def test_remove_never_touches_a_path_outside_the_runs_root(repo: Path, tmp_path: Path):
    """The guard is what makes deletion safe; a widened token class must not become an `rm -rf`."""
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "file.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(WorktreeError) as caught:
        remove_task_worktree(
            repo_dir=repo, runs_root=tmp_path / "runs", run_id="../precious", task_id="task-1"
        )
    assert caught.value.code == "worktree_input_invalid"
    assert (outside / "file.txt").is_file()


# -- list + the run-level sweep --------------------------------------------------------------


def test_list_reports_only_this_runs_worktrees(repo: Path, runs_root: Path):
    one = create(repo, runs_root, run_id="run-1", task_id="task-1", n=1)
    two = create(repo, runs_root, run_id="run-1", task_id="task-2", n=2)
    create(repo, runs_root, run_id="run-2", task_id="task-1", n=1)

    assert list_run_worktrees(repo_dir=repo, runs_root=runs_root, run_id="run-1") == (
        one.path,
        two.path,
    )
    # A directory git does not know is litter, not a worktree: reporting it would hide a leak.
    (runs_root / "run-1" / "wt-ghost").mkdir(parents=True)
    assert len(list_run_worktrees(repo_dir=repo, runs_root=runs_root, run_id="run-1")) == 2


def test_the_run_sweep_removes_every_worktree_of_that_run_only(repo: Path, runs_root: Path):
    first = create(repo, runs_root, run_id="run-1", task_id="task-1", n=1)
    second = create(repo, runs_root, run_id="run-1", task_id="task-2", n=2)
    other = create(repo, runs_root, run_id="run-2", task_id="task-1", n=1)

    removed = remove_run_worktrees(repo_dir=repo, runs_root=runs_root, run_id="run-1")

    assert removed == (first.path, second.path)
    assert not first.path.exists()
    assert not second.path.exists()
    # Another run's checkout is untouched — a sweep is scoped by run id, not by "all worktrees".
    assert other.path.is_dir()
    assert remove_run_worktrees(repo_dir=repo, runs_root=runs_root, run_id="run-1") == ()


# -- the environment ------------------------------------------------------------------------


def test_git_is_run_without_an_inherited_redirect(
    monkeypatch: pytest.MonkeyPatch, repo: Path, runs_root: Path
):
    """An inherited ``GIT_DIR`` would point every command at an unrelated repository.

    The variable is set to a path that does NOT exist, so a command that inherited it would fail
    outright with ``fatal: not a git repository``. ``create`` succeeding is the assertion; the
    verification below uses its own scrubbed runner, because this test's own environment is the
    poisoned one.
    """
    import os

    assert "GIT_DIR" in GIT_ENV_DROP
    monkeypatch.setenv("GIT_DIR", "/nonexistent/not-a-repo")
    monkeypatch.setenv("GIT_WORK_TREE", "/nonexistent/tree")

    made = create(repo, runs_root)
    assert made.path.is_dir()

    clean_env = {key: value for key, value in os.environ.items() if key not in GIT_ENV_DROP}
    head = subprocess.run(
        ["git", "-C", str(made.path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=clean_env,
    ).stdout.strip()
    assert head == "codna/run-run-1/task-1"
