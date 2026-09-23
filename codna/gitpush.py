"""Git push backend — turns a verified patch into a pushed branch for the PR writer.

The writer (GitHubWriter) opens the PR via the API; the actual "branch from diff" is this
backend's job: from a clean checkout it creates the branch at the base, applies the unified
diff, commits, and pushes. Kept separate from writer.py so the writer's "executes no
repository code" property stays clear — this runs only git plumbing (never the project's
build/test/scanner code), and is exercised offline against a local bare remote in tests.
"""
from __future__ import annotations

import subprocess


class GitPushError(Exception):
    pass


def _git(repo_dir: str, *args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["git", "-C", repo_dir, *args],
        input=stdin, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise GitPushError(f"git {' '.join(args)} -> {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


class GitBranchBuilder:
    def __init__(self, repo_dir: str, remote: str = "origin"):
        self.repo_dir = repo_dir
        self.remote = remote

    def build_and_push(
        self,
        *,
        branch: str,
        base: str,
        patch_diff: str,
        commit_message: str,
        push_url: str | None = None,
    ) -> str:
        """Create `branch` at `base`, apply `patch_diff`, commit, and push. `push_url` (with an
        embedded token) overrides the named remote. Returns the branch name."""
        if not patch_diff.strip():
            raise GitPushError("empty patch")
        base_sha = _git(self.repo_dir, "rev-parse", base).strip()
        _git(self.repo_dir, "checkout", "-B", branch, base_sha)
        apply = subprocess.run(
            ["git", "-C", self.repo_dir, "apply", "--index", "--whitespace=nowarn", "-"],
            input=patch_diff, capture_output=True, text=True,
        )
        if apply.returncode != 0:
            raise GitPushError(f"git apply failed: {apply.stderr.strip()}")
        _git(self.repo_dir, "-c", "user.email=codna@codna.ai", "-c", "user.name=codna",
             "commit", "-m", commit_message)
        _git(self.repo_dir, "push", push_url or self.remote, f"HEAD:refs/heads/{branch}")
        return branch
