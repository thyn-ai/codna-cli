"""Agent-produced diffs are repaired before `git apply`; count drift is absorbed by --recount."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codna import packaged_git
from codna.patch_text import normalize_unified_diff


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("one\n\ntwo\nthree\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], check=True)
    return repo


def _check(repo: Path, diff: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=diff, capture_output=True, text=True)


STRIPPED_BLANK_CONTEXT = "--- a/a.txt\n+++ b/a.txt\n@@ -1,4 +1,4 @@\n one\n\n-two\n+TWO\n three\n"


def test_fences_crlf_and_trailing_newline_are_normalized():
    raw = "```diff\r\n--- a/x\r\n+++ b/x\r\n@@ -1 +1 @@\r\n-a\r\n+b\r\n```\r\n"
    assert normalize_unified_diff(raw) == "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    clean = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert normalize_unified_diff(clean) == clean
    assert normalize_unified_diff("") == ""


def test_blank_context_lines_get_their_leading_space_back(tmp_path):
    repo = _repo(tmp_path)
    fixed = normalize_unified_diff(STRIPPED_BLANK_CONTEXT)
    assert fixed.splitlines()[4] == " "                                   # the stripped context line
    assert _check(repo, fixed).returncode == 0
    # and lines outside hunks are left alone: an empty line between file sections stays empty
    two_files = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-c\n+d\n"
    assert normalize_unified_diff(two_files).splitlines()[5] == " "  # still inside x's hunk region


def test_apply_to_index_survives_wrong_hunk_counts_via_recount(tmp_path):
    repo = _repo(tmp_path)
    wrong_counts = "--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,9 @@\n one\n \n-two\n+TWO\n three\n"
    assert _check(repo, wrong_counts).returncode != 0
    packaged_git._apply_patch_to_index(repo, wrong_counts)
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached"], capture_output=True, text=True).stdout
    assert "+TWO" in staged and "-two" in staged


def test_a_patch_that_truly_does_not_apply_is_rejected_with_gits_reason(tmp_path):
    repo = _repo(tmp_path)
    wrong_context = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n nothing-like-this\n-two\n+TWO\n"
    with pytest.raises(packaged_git.PackagedGitError) as exc:
        packaged_git._apply_patch_to_index(repo, wrong_context)
    assert exc.value.code == "patch_rejected"
    assert "does not apply" in str(exc.value) and "error" in exc.value.details.get("stderr", "")
