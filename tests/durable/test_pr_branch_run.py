"""Run FROM an existing branch (D5/X8): the worktree half of the pull-request path.

The PR card's `auto-fix`/`address-comments` start a run whose work must land on the branch the pull
request is already open against. Before this, the durable path could only create
``codna/run-<run>/task-<n>``, so the card refused.

What is asserted here is the SET of properties that make adopting a branch safe — an existing branch is
checked out and never reset, a missing one is a refusal rather than a new branch, and the task's audit
refs still point at the commit the turn produced. A "run from a branch" that reset the branch would
pass a naive test and delete the pull request's work.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.worktree import (
    WorktreeError,
    create_task_worktree,
    list_run_worktrees,
    remove_task_worktree,
)
from tests.durable import fakes


def _branch_commits(repo: Path, branch: str) -> list[str]:
    return fakes.git(repo, "log", "--format=%H", branch).split()


def _existing_branch(repo: Path, name: str, *, files: dict[str, str]) -> str:
    """A branch with real work on it, as a pull request's head would have."""
    fakes.git(repo, "checkout", "-q", "-b", name)
    for path, content in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    fakes.git(repo, "add", "-A")
    fakes.git(repo, "commit", "-q", "-m", f"work for {name}")
    sha = fakes.git(repo, "rev-parse", "HEAD").strip()
    fakes.git(repo, "checkout", "-q", "main")
    return sha


def test_adopting_a_branch_checks_it_out_and_NEVER_resets_it(tmp_path: Path):
    """The branch's commits are the work being continued: a reset would delete the pull request."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    head = _existing_branch(repo, "fix/ci", files={"src/fix.py": "x = 1\n"})

    made = create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id="run-pr",
        task_id="task-1",
        task_number=1,
        base_branch="main",
        adopt_branch="fix/ci",
    )

    assert made.branch == "fix/ci"
    assert made.path.is_dir()
    # NO new commit, NO reset: the branch still points at exactly what it did.
    assert fakes.git(repo, "rev-parse", "fix/ci").strip() == head
    # The worktree is ON that branch, with the branch's files present (a reset would have lost them).
    assert fakes.git(made.path, "rev-parse", "--abbrev-ref", "HEAD").strip() == "fix/ci"
    assert (made.path / "src" / "fix.py").read_text(encoding="utf-8") == "x = 1\n"
    # The path is still the plan's `<runs>/<run>/wt-<task>`, so the sweep finds it.
    assert made.path.parent == runs_root / "run-pr"
    assert made.path.name == "wt-task-1"


def test_a_MISSING_branch_is_refused_rather_than_created(tmp_path: Path):
    """`fix this PR` pointing at a branch that is not there is a caller mistake, not an invention."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(WorktreeError) as raised:
        create_task_worktree(
            repo_dir=repo,
            runs_root=runs_root,
            run_id="run-pr",
            task_id="task-1",
            task_number=1,
            base_branch="main",
            adopt_branch="no/such/branch",
        )
    assert raised.value.code == "worktree_branch_missing"
    # Nothing was created: no branch, no worktree.
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "refs/heads/no/such/branch"],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode != 0
    assert list_run_worktrees(repo_dir=repo, runs_root=runs_root, run_id="run-pr") == ()


def test_adopting_is_IDEMPOTENT_under_a_retry(tmp_path: Path):
    """Temporal retries at least once; the second call must find what the first made, not refuse."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _existing_branch(repo, "fix/ci", files={"a.py": "a = 1\n"})

    first = create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id="run-pr",
        task_id="task-1",
        task_number=1,
        base_branch="main",
        adopt_branch="fix/ci",
    )
    second = create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id="run-pr",
        task_id="task-1",
        task_number=1,
        base_branch="main",
        adopt_branch="fix/ci",
    )
    assert first.created is True
    assert second.created is False
    assert second.path == first.path
    assert second.branch == "fix/ci"


def test_a_branch_name_that_git_would_read_as_a_FLAG_is_refused(tmp_path: Path):
    """`worktree add <path> -b` would consume `-b` as an option: the one shape that must never reach argv."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(WorktreeError) as raised:
        create_task_worktree(
            repo_dir=repo,
            runs_root=runs_root,
            run_id="run-pr",
            task_id="task-1",
            task_number=1,
            base_branch="main",
            adopt_branch="--upload-pack=touch /tmp/pwned",
        )
    assert raised.value.code == "worktree_input_invalid"


def test_an_adopted_branch_worktree_is_reclaimed_like_any_other(tmp_path: Path):
    """The sweep must not leak an adopted checkout: it is a worktree of this run like any other."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _existing_branch(repo, "fix/ci", files={"a.py": "a = 1\n"})
    made = create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id="run-pr",
        task_id="task-1",
        task_number=1,
        base_branch="main",
        adopt_branch="fix/ci",
    )

    assert remove_task_worktree(repo_dir=repo, runs_root=runs_root, run_id="run-pr", task_id="task-1") is True
    assert not made.path.exists()
    # The BRANCH survives the sweep: its commits are the run's output, and the PR still points at it.
    assert fakes.git(repo, "rev-parse", "--verify", "refs/heads/fix/ci").strip()


def test_adopting_a_remote_only_branch_creates_the_local_tracking_ref(tmp_path: Path):
    """A fresh clone has the PR's head only as `origin/<branch>`; the run must still be able to work."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _existing_branch(repo, "fix/remote", files={"r.py": "r = 1\n"})
    # Simulate "the branch exists only on the remote": a local ref for origin/fix/remote, no local one.
    sha = fakes.git(repo, "rev-parse", "fix/remote").strip()
    fakes.git(repo, "update-ref", "refs/remotes/origin/fix/remote", sha)
    fakes.git(repo, "branch", "-D", "fix/remote")

    made = create_task_worktree(
        repo_dir=repo,
        runs_root=runs_root,
        run_id="run-pr",
        task_id="task-1",
        task_number=1,
        base_branch="main",
        adopt_branch="fix/remote",
    )
    assert made.branch == "fix/remote"
    assert fakes.git(made.path, "rev-parse", "HEAD").strip() == sha


# -- the run that uses it ---------------------------------------------------------------------


def _spec(repo: Path, **overrides):
    from codna.durable.contracts import TaskSpec

    payload = {
        "run_id": "run-pr",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the failing check",
        "repo_dir": str(repo),
        "verify_commands": ("true",),
        "engine": "plain",
        "worktree": True,
        "worktree_branch": "fix/ci",
    }
    payload.update(overrides)
    return TaskSpec(**payload)


def test_the_task_branch_and_the_context_are_the_ADOPTED_one(tmp_path: Path):
    """One definition, used by the checkpoint, the push and the PR: they cannot disagree."""
    from codna.durable.task_context import TaskContext
    from codna.durable.worktree import branch_for_task

    repo = fakes.make_repo(tmp_path / "repo")
    spec = _spec(repo)
    assert spec.branch == "fix/ci"
    assert TaskContext.from_spec(spec).branch == "fix/ci"
    # ...and the default is still the derived branch, so nothing else changed.
    assert _spec(repo, worktree_branch="").branch == branch_for_task("run-pr", 1)


def test_a_commit_lands_on_the_ADOPTED_branch_and_the_audit_ref(tmp_path: Path):
    """The checkpoint's guarantees hold verbatim: one commit, one ref, on the PR's branch."""
    from codna.durable.activity_steps import checkpoint_turn, turn_ref
    from codna.durable.activity_support import ensure_checkout
    from dataclasses import replace

    from codna.durable.contracts import CheckpointRequest
    from codna.durable.state import TaskState
    from codna.durable.task_context import TaskContext

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    existing = _existing_branch(repo, "fix/ci", files={"src/fix.py": "x = 1\n"})
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo)
    context = TaskContext.from_spec(spec)
    # A turn has happened: the checkpoint is of turn 1.
    state = replace(TaskState.from_payload(spec.initial_state()), turn_seq=1).to_payload()

    tree = ensure_checkout(harness.runtime, context)
    (Path(tree) / "src" / "fix.py").write_text("x = 2\n", encoding="utf-8")

    outcome = checkpoint_turn(CheckpointRequest(state=state, context=context))

    # The commit is ON the pull request's branch, NOT on a new `codna/run-...` one.
    assert outcome.details["branch"] == "fix/ci"
    assert fakes.git(repo, "rev-parse", "fix/ci").strip() == outcome.sha
    assert outcome.sha != existing  # the turn's change was committed
    # The WORKTREE is on the adopted branch…
    assert fakes.git(Path(tree), "rev-parse", "--abbrev-ref", "HEAD").strip() == "fix/ci"
    # …and the developer's SHARED checkout was NOT moved. This is the property that makes adopting a
    # branch safe: the run works in its own checkout, and the person's HEAD is where they left it.
    assert fakes.git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    # The audit ref exists in the repository and points at the same commit.
    assert fakes.git(repo, "rev-parse", turn_ref(context, 1)).strip() == outcome.sha
    # ...and the branch was CONTINUED, not reset: the PR's earlier commit is still its parent.
    parents = fakes.git(repo, "rev-list", "--parents", "-n", "1", outcome.sha).split()
    assert existing in parents


def test_worktree_branch_WITHOUT_worktree_is_refused_by_the_contract(tmp_path: Path):
    """A shared checkout cannot be on two branches: adopting one there would move the developer's HEAD."""
    repo = fakes.make_repo(tmp_path / "repo")
    with pytest.raises(ValueError) as raised:
        _spec(repo, worktree=False).validate()
    assert "worktree" in str(raised.value)


@pytest.mark.parametrize("name", ["-b", "--force", " "])
def test_an_illegal_adopted_branch_name_is_refused_at_the_contract(tmp_path: Path, name: str):
    repo = fakes.make_repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        _spec(repo, worktree_branch=name).validate()


def test_a_recorded_run_without_the_field_adopts_NOTHING():
    """Replay safety: a run started before this feature must recreate the branch it recorded."""
    from codna.durable.workflow_support import spec_from

    payload = _spec(Path("/tmp/repo")).to_payload()
    del payload["worktree_branch"]
    decoded = spec_from(payload)
    assert decoded.worktree_branch == ""
    assert decoded.branch == "codna/run-run-pr/task-1"
