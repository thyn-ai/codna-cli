"""Ephemeral-worktree tests (A–Z test plan: PI-WT-01/03/05, PRIV-05/06). Uses real git, no network."""
from __future__ import annotations

import os
import subprocess

import pytest

from codna.worktree import EphemeralWorktree, WorktreeError


def _run(argv, cwd):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


def _init_repo(tmp_path):
    d = tmp_path / "repo"
    (d / "src").mkdir(parents=True)
    (d / "src" / "app.py").write_text('def q(u):\n    return "X=" + u\n')
    _run(["git", "init", "-q"], d)
    _run(["git", "config", "user.email", "a@b.c"], d)
    _run(["git", "config", "user.name", "t"], d)
    _run(["git", "add", "-A"], d)
    _run(["git", "commit", "-q", "-m", "init"], d)
    sha = _run(["git", "rev-parse", "HEAD"], d).stdout.strip()
    return str(d), sha


def _worktrees(repo):
    return _run(["git", "worktree", "list"], repo).stdout


def test_worktree_detached_at_exact_commit_and_torn_down(tmp_path):
    repo, sha = _init_repo(tmp_path)
    with EphemeralWorktree(repo, sha) as wt:
        path = wt.path
        assert os.path.isdir(path)
        assert wt.head_commit == sha
        assert path in _worktrees(repo)
    assert not os.path.isdir(path)  # removed on exit
    assert path not in _worktrees(repo)


def test_worktree_torn_down_on_exception(tmp_path):
    repo, sha = _init_repo(tmp_path)
    captured = {}
    with pytest.raises(RuntimeError):
        with EphemeralWorktree(repo, sha) as wt:
            captured["path"] = wt.path
            raise RuntimeError("boom")
    assert not os.path.isdir(captured["path"])


def test_apply_patch_changes_tree(tmp_path):
    repo, sha = _init_repo(tmp_path)
    with EphemeralWorktree(repo, sha) as wt:
        clean_tree = wt.tree_digest()
        # produce a real diff via git, then restore so it applies cleanly
        app = os.path.join(wt.path, "src", "app.py")
        original = open(app).read()
        open(app, "w").write(original.replace('"X=" + u', "safe(u)"))
        diff = _run(["git", "diff"], wt.path).stdout
        _run(["git", "checkout", "--", "."], wt.path)
        assert open(app).read() == original  # restored

        wt.apply_patch(diff)
        assert "safe(u)" in open(app).read()
        assert wt.tree_digest() != clean_tree


def test_empty_patch_rejected(tmp_path):
    repo, sha = _init_repo(tmp_path)
    with EphemeralWorktree(repo, sha) as wt:
        with pytest.raises(WorktreeError):
            wt.apply_patch("   \n")
