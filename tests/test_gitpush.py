"""GitBranchBuilder tests — real git, pushed to a LOCAL BARE remote (no network/GitHub)."""
from __future__ import annotations

import subprocess

import pytest

from codna.gitpush import GitBranchBuilder, GitPushError


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _setup(tmp_path):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)])
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text('def q(u):\n    return "X=" + u\n')
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "a@b.c"], repo)
    _git(["config", "user.name", "t"], repo)
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    _git(["branch", "-M", "main"], repo)
    _git(["remote", "add", "origin", str(bare)], repo)
    _git(["push", "-q", "origin", "main"], repo)
    return str(repo), str(bare)


def _make_diff(repo):
    p = f"{repo}/src/app.py"
    original = open(p).read()
    open(p, "w").write(original.replace('"X=" + u', "safe(u)"))
    diff = _git(["diff"], repo).stdout
    _git(["checkout", "--", "."], repo)
    return diff


def test_build_and_push_applies_diff_to_remote_branch(tmp_path):
    repo, bare = _setup(tmp_path)
    diff = _make_diff(repo)
    GitBranchBuilder(repo).build_and_push(
        branch="codna/secure/fix-1", base="main", patch_diff=diff, commit_message="codna: fix sqli"
    )
    # the branch now exists on the remote...
    ls = _git(["ls-remote", bare, "codna/secure/fix-1"], tmp_path).stdout
    assert "refs/heads/codna/secure/fix-1" in ls
    # ...and a fresh clone of that branch carries the applied change.
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "-b", "codna/secure/fix-1", bare, str(clone)])
    assert "safe(u)" in (clone / "src" / "app.py").read_text()


def test_empty_patch_rejected(tmp_path):
    repo, _ = _setup(tmp_path)
    with pytest.raises(GitPushError):
        GitBranchBuilder(repo).build_and_push(branch="b", base="main", patch_diff="  ", commit_message="x")


def test_unapplicable_patch_rejected(tmp_path):
    repo, _ = _setup(tmp_path)
    bad = "diff --git a/nope.py b/nope.py\n--- a/nope.py\n+++ b/nope.py\n@@ -5 +5 @@\n-gone\n+changed\n"
    with pytest.raises(GitPushError):
        GitBranchBuilder(repo).build_and_push(branch="b", base="main", patch_diff=bad, commit_message="x")
