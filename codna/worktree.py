"""Ephemeral git worktree pinned at an exact commit.

The worker analyzes/patches code in a throwaway worktree checked out at the EXACT commit the
proof is bound to (a detached checkout, never a moving branch ref), and tears it down on both
success and failure. This isolates patch generation from the user's checkout and guarantees
the analyzed tree matches the snapshot (PI-WT-01/03/05, PRIV-05/06).

Uses git directly (no network). `git apply` and tree-hashing run here so the worker can compute
the resulting-tree digest that the attestation binds.
"""
from __future__ import annotations

import subprocess

from .patch_text import GIT_APPLY_FLAG_SETS, normalize_unified_diff
import tempfile


class WorktreeError(Exception):
    pass


def _git(repo_dir: str, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", repo_dir, *args],
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise WorktreeError(f"git {' '.join(args)} -> {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


class EphemeralWorktree:
    """Context manager: a detached worktree at `commit`, removed (force + prune) on exit —
    on success AND on exception."""

    def __init__(self, repo_dir: str, commit: str, prefix: str = "codna-wt-"):
        self.repo_dir = repo_dir
        self.commit = commit
        self.path = tempfile.mkdtemp(prefix=prefix)

    def create(self) -> "EphemeralWorktree":
        # --force lets us reuse the freshly-created empty temp dir; --detach pins the commit.
        _git(self.repo_dir, "worktree", "add", "--detach", "--force", self.path, self.commit)
        return self

    def __enter__(self) -> "EphemeralWorktree":
        return self.create()

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        _git(self.repo_dir, "worktree", "remove", "--force", self.path, check=False)
        _git(self.repo_dir, "worktree", "prune", check=False)

    @property
    def head_commit(self) -> str:
        return _git(self.path, "rev-parse", "HEAD").strip()

    def tree_digest(self) -> str:
        """sha of the current working tree (after staging) — what the patch produced."""
        _git(self.path, "add", "-A")
        return "sha256:tree:" + _git(self.path, "write-tree").strip()

    def apply_patch(self, diff: str) -> None:
        """Apply a unified diff to the worktree. Raises WorktreeError on a rejected/empty patch."""
        diff = normalize_unified_diff(diff)
        if not diff.strip():
            raise WorktreeError("empty patch")
        first_error = ""
        for flags in GIT_APPLY_FLAG_SETS:
            check = subprocess.run(
                ["git", "-C", self.path, "apply", "--check", "--whitespace=nowarn", *flags, "-"],
                input=diff, capture_output=True, text=True,
            )
            if check.returncode == 0:
                proc = subprocess.run(
                    ["git", "-C", self.path, "apply", "--whitespace=nowarn", *flags, "-"],
                    input=diff, capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise WorktreeError(f"git apply failed: {proc.stderr.strip()}")
                return
            first_error = first_error or check.stderr.strip()
        raise WorktreeError(f"git apply failed: {first_error}")
