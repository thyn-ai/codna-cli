"""`gitpush.push_branch` — the durable `open_pr`'s push half.

`GitBranchBuilder` exists to turn a PATCH into a branch. A durable run's branch is already built:
`durable.activities.checkpoint_turn` commits each turn itself, and the branch is what the audit refs
point at. Re-applying a patch there would rewrite the history that was just recorded, so the push has
its own entry point that pushes an EXISTING commit.

Tested against a real local bare remote (no network), because the property — the commit that lands on
the remote is the commit that was pushed — is a property of git.
"""
from __future__ import annotations

import subprocess

import pytest

from codna.gitpush import GitPushError, push_branch


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A repository on ``main`` with one commit and a bare remote."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    path = tmp_path / "repo"
    path.mkdir()
    (path / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "a@b.c"], path)
    _git(["config", "user.name", "t"], path)
    _git(["add", "-A"], path)
    _git(["commit", "-q", "-m", "init"], path)
    _git(["remote", "add", "origin", str(bare)], path)
    return path, bare


def test_push_branch_publishes_an_existing_commit(repo):
    path, bare = repo
    _git(["checkout", "-q", "-b", "codna/run-1/task-1"], path)
    (path / "app.py").write_text("x = 2\n", encoding="utf-8")
    _git(["commit", "-q", "-am", "turn 1"], path)
    sha = _git(["rev-parse", "HEAD"], path).stdout.strip()

    pushed = push_branch(str(path), branch="codna/run-1/task-1", sha=sha, remote=str(bare))

    assert pushed == "codna/run-1/task-1"
    remote_sha = _git(["rev-parse", "refs/heads/codna/run-1/task-1"], bare).stdout.strip()
    assert remote_sha == sha, "the remote must carry the EXACT commit, not a copy of its tree"


def test_pushing_the_same_commit_twice_is_a_no_op(repo):
    """Idempotency: a retried activity re-pushes the same sha and must not error."""
    path, bare = repo
    sha = _git(["rev-parse", "HEAD"], path).stdout.strip()
    push_branch(str(path), branch="b", sha=sha, remote=str(bare))
    # Second push of the same sha: git reports "Everything up-to-date" and exits 0.
    assert push_branch(str(path), branch="b", sha=sha, remote=str(bare)) == "b"


def test_an_unknown_sha_is_refused_before_touching_the_remote(repo):
    """Fails with an error about the SHA, not a confusing one about the remote."""
    path, bare = repo
    with pytest.raises(GitPushError, match="rev-parse"):
        push_branch(str(path), branch="b", sha="0" * 40, remote=str(bare))
    assert _git(["ls-remote", str(bare), "b"], path).stdout.strip() == ""
