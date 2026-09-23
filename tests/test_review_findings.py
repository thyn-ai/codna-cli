"""Pure/offline tests for the codna review findings pipeline (no agent, no network)."""
from __future__ import annotations

import os

import pytest
import subprocess
from types import SimpleNamespace

from codna import review_findings as rf
from codna import review_github as rg
from codna.cline_agent import extract_findings_json


# ---- config -----------------------------------------------------------------------------------

def test_config_defaults_non_blocking():
    c = rf.ReviewConfig()
    assert c.enabled and c.min_confidence == 0.75 and c.max_findings == 10
    assert c.blocking_enabled is False and c.blocking_severities == ("high",)
    assert all(c.category_enabled(cat) for cat in rf.CATEGORIES)


def test_config_from_dict_overrides():
    c = rf.review_config_from_dict({
        "min_confidence": 0.9, "max_findings": 3,
        "blocking": {"enabled": True, "severities": ["high", "medium"]},
        "categories": {"performance": False},
        "ignore_paths": ["docs/**"],
    })
    assert c.min_confidence == 0.9 and c.max_findings == 3
    assert c.blocking_enabled and set(c.blocking_severities) == {"high", "medium"}
    assert c.category_enabled("correctness") and not c.category_enabled("performance")
    assert c.ignore_paths == ("docs/**",)


def test_load_review_config_from_yaml(tmp_path):
    (tmp_path / "codna.yaml").write_text(
        "review:\n  min_confidence: 0.5\n  blocking:\n    enabled: true\n", encoding="utf-8"
    )
    c = rf.load_review_config(str(tmp_path))
    assert c.min_confidence == 0.5 and c.blocking_enabled is True


def test_load_review_config_missing_is_default(tmp_path):
    assert rf.load_review_config(str(tmp_path)).min_confidence == 0.75


# ---- normalize / fingerprint ------------------------------------------------------------------

def _raw(**kw):
    base = {"path": "src/a.py", "line": 10, "severity": "high", "category": "correctness",
            "title": "Off-by-one", "explanation": "loop overruns", "confidence": 0.9}
    base.update(kw)
    return base


def test_normalize_valid():
    f = rf.normalize_finding(_raw())
    assert f and f.path == "src/a.py" and f.line == 10 and f.severity == "high"
    assert f.fingerprint == rf.fingerprint("src/a.py", "correctness", "Off-by-one")


def test_normalize_rejects_bad_domain_and_missing():
    assert rf.normalize_finding(_raw(severity="critical")) is None
    assert rf.normalize_finding(_raw(category="style")) is None
    assert rf.normalize_finding(_raw(line="notint")) is None
    assert rf.normalize_finding(_raw(path="")) is None
    assert rf.normalize_finding({"nope": 1}) is None


def test_normalize_clamps_confidence_and_strips_leading_slash():
    f = rf.normalize_finding(_raw(confidence=5.0, path="/src/a.py"))
    assert f.confidence == 1.0 and f.path == "src/a.py"


def test_fingerprint_is_line_independent():
    a = rf.fingerprint("src/a.py", "correctness", "Off-by-one error  ")
    b = rf.fingerprint("src/a.py", "correctness", "off-by-one ERROR")
    assert a == b  # normalized (case + whitespace), independent of line number


# ---- diff parsing + anchoring -----------------------------------------------------------------

DIFF = """diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,5 @@
 def f(xs):
-    return xs[0]
+    if not xs:
+        return None
+    return xs[0]
 # tail
diff --git a/docs/x.md b/docs/x.md
--- a/docs/x.md
+++ b/docs/x.md
@@ -1 +1,2 @@
 hello
+world
"""


def test_parse_diff_right_lines():
    files = rf.parse_diff(DIFF)
    assert set(files) == {"src/a.py", "docs/x.md"}
    a = files["src/a.py"]
    # new-side lines shown in the hunk: 1 (context), 2,3,4 (added), 5 (context)
    assert a.added_lines == {2, 3, 4}
    assert {1, 2, 3, 4, 5}.issubset(a.right_lines)


def test_anchor_inline_vs_summary():
    files = rf.parse_diff(DIFF)
    inside = rf.normalize_finding(_raw(path="src/a.py", line=3))
    outside = rf.normalize_finding(_raw(path="src/a.py", line=99, title="elsewhere"))
    other = rf.normalize_finding(_raw(path="not/in/diff.py", line=1, title="ghost"))
    rf.anchor_findings([inside, outside, other], files, "deadbeef")
    assert inside.inline and inside.diff_anchor.commit_id == "deadbeef" and inside.diff_anchor.side == "RIGHT"
    assert not outside.inline and not other.inline


def test_anchor_requires_head_sha():
    files = rf.parse_diff(DIFF)
    f = rf.normalize_finding(_raw(path="src/a.py", line=3))
    rf.anchor_findings([f], files, None)
    assert not f.inline


# An added ``++ weird`` renders as ``+++ weird`` and a removed ``-- old`` as ``--- old``: the shapes a
# prefix-based "file header" test mistakes for ``+++ b/path`` / ``--- a/path``.
LOOKALIKE_DIFF = """diff --git a/q.hs b/q.hs
index 111..222 100644
--- a/q.hs
+++ b/q.hs
@@ -1,3 +1,5 @@
 a = [1]
--- old comment
+++ weird
+  ++ [2]
+-- new comment
 b = 2
diff --git a/docs/x.md b/docs/x.md
--- a/docs/x.md
+++ b/docs/x.md
@@ -1 +1,2 @@
 hello
+world
"""


def test_parse_diff_header_lookalikes_inside_a_hunk_are_body():
    """The prefix test opened a bogus DiffFile "weird" at ``+++ weird`` and filed the rest of the hunk
    under it, so findings on those lines lost their inline anchor. The hunk ranges (``-1,3 +1,5``) say
    every one of those lines is body of q.hs; only between hunks may ``+++ path`` open a file."""
    files = rf.parse_diff(LOOKALIKE_DIFF)
    assert set(files) == {"q.hs", "docs/x.md"}                        # no DiffFile named "weird"
    q = files["q.hs"]
    assert q.added_lines == {2, 3, 4}                                   # "++ weird", "  ++ [2]", "-- new comment"
    assert q.right_lines == {1, 2, 3, 4, 5}                             # + the two context lines
    x = files["docs/x.md"]
    assert x.added_lines == {2} and x.right_lines == {1, 2}             # the real header still switches files
    after = rf.normalize_finding(_raw(path="q.hs", line=4))
    rf.anchor_findings([after], files, "deadbeef")
    assert after.inline and after.diff_anchor.line == 4 and after.diff_anchor.side == "RIGHT"


def test_parse_diff_agrees_with_real_git_on_header_lookalikes(tmp_path):
    """Against real ``git diff`` output (same shape as test_review_budget's numstat check): every
    added line is a line we wrote into the new file, no bogus file appears, and findings on the
    lookalike line and on the context line after it both anchor inline."""
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})

    (tmp_path / "q.sql").write_text("SELECT 1;\n-- old comment\n-- another\nFROM t;\n", encoding="utf-8")
    (tmp_path / "h.hs").write_text("a = [1]\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "base")
    new_sql = ["SELECT 1;", "-- new comment", "++ weird", "FROM t;"]
    new_hs = ["a = [1]", "  ++ [2]"]
    (tmp_path / "q.sql").write_text("\n".join(new_sql) + "\n", encoding="utf-8")
    (tmp_path / "h.hs").write_text("\n".join(new_hs), encoding="utf-8")             # no trailing newline

    diff, changed = rf.compute_diff(str(tmp_path), base="HEAD")
    assert "\n+++ weird\n" in diff and "\n--- old comment\n" in diff               # the lookalikes are there
    files = rf.parse_diff(diff)
    assert set(files) == set(changed) == {"h.hs", "q.sql"}
    assert files["q.sql"].added_lines == {2, 3} and files["q.sql"].right_lines == {1, 2, 3, 4}
    assert [new_sql[n - 1] for n in sorted(files["q.sql"].added_lines)] == ["-- new comment", "++ weird"]
    assert files["h.hs"].added_lines == {2} and files["h.hs"].right_lines == {1, 2}
    assert new_hs[1] == "  ++ [2]"

    weird = rf.normalize_finding(_raw(path="q.sql", line=3, title="weird"))
    after = rf.normalize_finding(_raw(path="q.sql", line=4, title="after"))
    tail = rf.normalize_finding(_raw(path="h.hs", line=2, title="tail"))
    rf.anchor_findings([weird, after, tail], files, "deadbeef")
    assert weird.inline and weird.diff_anchor.line == 3
    assert after.inline and after.diff_anchor.line == 4
    assert tail.inline and tail.diff_anchor.line == 2


# ---- noise controls + dedup -------------------------------------------------------------------

def test_noise_controls_confidence_category_ignore_dedup_cap():
    cfg = rf.ReviewConfig(min_confidence=0.8, max_findings=2, ignore_paths=("docs/**",),
                          categories={"correctness": True, "security": True, "performance": False})
    findings = [
        rf.normalize_finding(_raw(title="keep-1", confidence=0.95)),
        rf.normalize_finding(_raw(title="low", confidence=0.5)),                       # dropped: confidence
        rf.normalize_finding(_raw(title="perf", category="performance", confidence=0.99)),  # dropped: category
        rf.normalize_finding(_raw(title="doc", path="docs/y.md", confidence=0.99)),    # dropped: ignore path
        rf.normalize_finding(_raw(title="keep-1", confidence=0.9)),                    # dropped: duplicate fp
        rf.normalize_finding(_raw(title="keep-2", severity="medium", confidence=0.99)),
        rf.normalize_finding(_raw(title="keep-3", severity="low", confidence=0.99)),   # dropped: over cap
    ]
    kept, dropped = rf.apply_noise_controls(findings, cfg)
    titles = [f.title for f in kept]
    assert titles == ["keep-1", "keep-2"]  # sorted high>medium, capped at 2
    assert dropped["low_confidence"] == 1 and dropped["category_disabled"] == 1
    assert dropped["ignored_path"] == 1 and dropped["duplicate"] == 1 and dropped["over_cap"] == 1


def test_check_conclusion_non_blocking_by_default():
    cfg = rf.ReviewConfig()
    assert rf.check_conclusion([], cfg) == "success"
    high = rf.normalize_finding(_raw(severity="high"))
    assert rf.check_conclusion([high], cfg) == "neutral"  # non-blocking even for high severity


def test_check_conclusion_blocking_when_enabled():
    cfg = rf.ReviewConfig(blocking_enabled=True, blocking_severities=("high",))
    high = rf.normalize_finding(_raw(severity="high"))
    med = rf.normalize_finding(_raw(severity="medium", title="m"))
    assert rf.check_conclusion([high], cfg) == "failure"
    assert rf.check_conclusion([med], cfg) == "neutral"  # medium is not a blocking severity


# ---- full finalize pipeline -------------------------------------------------------------------

def test_finalize_findings_end_to_end():
    files = rf.parse_diff(DIFF)
    raw = [
        _raw(path="src/a.py", line=3, title="real bug", confidence=0.9),
        _raw(path="src/a.py", line=999, title="out of hunk", confidence=0.9),
        _raw(severity="bogus"),  # invalid → dropped
    ]
    inline, summary, dropped = rf.finalize_findings(raw, files, "sha123", rf.ReviewConfig())
    assert [f.title for f in inline] == ["real bug"]
    assert [f.title for f in summary] == ["out of hunk"]


# ---- agent JSON extraction --------------------------------------------------------------------

def test_extract_findings_plain_object():
    out = extract_findings_json('{"findings": [{"path": "a.py", "line": 1}]}')
    assert out == [{"path": "a.py", "line": 1}]


def test_extract_findings_code_fence_and_prose():
    text = 'Here you go:\n```json\n{"findings": [{"path": "a.py", "line": 2}]}\n```\nDone.'
    assert extract_findings_json(text) == [{"path": "a.py", "line": 2}]


def test_extract_findings_bare_array():
    assert extract_findings_json('[{"path": "a.py", "line": 3}]') == [{"path": "a.py", "line": 3}]


def test_extract_findings_empty():
    assert extract_findings_json("") == []
    assert extract_findings_json('{"findings": []}') == []


# ---- GitHub payload builders ------------------------------------------------------------------

def _result_with(inline_titles, summary_titles, *, head="abc1234", conclusion="neutral"):
    def mk(t, anchor):
        f = rf.normalize_finding(_raw(title=t))
        if anchor:
            f.diff_anchor = rf.DiffAnchor(commit_id=head, side="RIGHT", line=f.line)
        return f
    return rf.ReviewResult(
        repository="r", base="main...HEAD", head_sha=head, changed_files=["src/a.py"],
        inline_findings=[mk(t, True) for t in inline_titles],
        summary_findings=[mk(t, False) for t in summary_titles],
        conclusion=conclusion, dropped={},
    )


def test_build_review_payload_inline_and_summary():
    res = _result_with(["bug A"], ["bug B"])
    payload = rg.build_review_payload(res)
    assert payload["event"] == "COMMENT" and payload["commit_id"] == "abc1234"
    assert len(payload["comments"]) == 1
    c = payload["comments"][0]
    assert c["path"] == "src/a.py" and c["side"] == "RIGHT" and "bug A" in c["body"]
    assert "codna:review:fp=" in c["body"]
    assert "bug B" in payload["body"]  # summary finding rendered in the body


def test_build_review_payload_skips_known_fingerprints():
    res = _result_with(["bug A"], [])
    fp = res.inline_findings[0].fingerprint
    payload = rg.build_review_payload(res, skip_fingerprints={fp})
    assert payload["comments"] == []  # already posted → not duplicated


def test_existing_fingerprints_scans_markers():
    comments = [{"body": "text " + rg.fp_marker("abc123")}, {"body": "no marker"}]
    assert rg.existing_fingerprints(comments) == {"abc123"}


def test_summary_body_clean():
    res = rf.ReviewResult(repository="r", base="b", head_sha=None, changed_files=[],
                          inline_findings=[], summary_findings=[], conclusion="success", dropped={})
    assert "no high-confidence issues" in rg.summary_body(res)


def test_parse_pr_arg_forms(tmp_path):
    assert rg.parse_pr_arg("owner/repo#42", str(tmp_path)) == ("owner/repo", 42)
    assert rg.parse_pr_arg("https://github.com/o/r/pull/7", str(tmp_path)) == ("o/r", 7)


# ---- remote (URL) review routing --------------------------------------------------------------

def test_is_git_url():
    from codna import review
    assert review._is_git_url("https://github.com/o/r.git")
    assert review._is_git_url("git@github.com:o/r.git")
    assert review._is_git_url("ssh://git@github.com/o/r")
    assert not review._is_git_url("/local/path")
    assert not review._is_git_url(".")


def test_run_findings_review_url_materializes_pr(monkeypatch, tmp_path):
    """A git URL + --pr routes through fetch_pr + _materialize_pr, then the normal pipeline."""
    from codna import review

    monkeypatch.setattr(rg, "fetch_pr",
                        lambda slug, n, tok: {"base_ref": "main", "head_sha": "h",
                                              "head_ref": "f", "is_fork": False})
    monkeypatch.setattr(rg, "fetch_review_context",
                        lambda *a, **k: {"last_reviewed_head": None, "prior_feedback": None})
    monkeypatch.setattr(review, "_materialize_pr",
                        lambda url, n, base, tok, incremental_base=None: (str(tmp_path), "origin/main...codna-pr-head", "headsha", None))
    captured = {}

    def fake_run_diff_review(local, **kw):
        captured.update(kw)
        captured["local"] = local
        return rf.ReviewResult(repository="u", base=kw["diff_range"], head_sha=kw["head_sha"],
                               changed_files=["a.py"], inline_findings=[], summary_findings=[],
                               conclusion="success", dropped={})

    monkeypatch.setattr(rf, "run_diff_review", fake_run_diff_review)
    args = SimpleNamespace(repo="https://github.com/o/r.git", pr="o/r#5", diff=None, base=None,
                           post=False, model=None, github_token="t", min_confidence=None,
                           max_findings=None, blocking=False)
    out = review.run_findings_review(args)
    assert captured["local"] == str(tmp_path)
    assert captured["diff_range"] == "origin/main...codna-pr-head"
    assert captured["head_sha"] == "headsha"
    assert captured["diff_paths"] is None                    # a whole-PR review is not path-confined
    assert out["conclusion"] == "success"


def _remote_review_args():
    return SimpleNamespace(repo="https://github.com/o/r.git", pr="o/r#5", diff=None, base=None,
                           post=False, model=None, github_token="t", min_confidence=None,
                           max_findings=None, blocking=False)


def _stub_remote_pr(monkeypatch, clone_dir):
    from codna import review

    monkeypatch.setattr(rg, "fetch_pr",
                        lambda slug, n, tok: {"base_ref": "main", "head_sha": "h",
                                              "head_ref": "f", "is_fork": False})
    monkeypatch.setattr(rg, "fetch_review_context",
                        lambda *a, **k: {"last_reviewed_head": None, "prior_feedback": None})

    def fake_materialize(url, n, base, tok, incremental_base=None):
        clone_dir.mkdir()
        (clone_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        return str(clone_dir), "origin/main...codna-pr-head", "headsha", None

    monkeypatch.setattr(review, "_materialize_pr", fake_materialize)
    return review


def test_run_findings_review_url_removes_the_materialized_clone(monkeypatch, tmp_path):
    """REGRESSION: the clone `_materialize_pr` makes lives in the process temp dir; the remote
    review path must remove it once the review is done. It never did -- every webhook review
    left a full checkout behind until the machine's root fs filled (ENOSPC, algenta#1023)."""
    clone = tmp_path / "codna-review-abc"
    review = _stub_remote_pr(monkeypatch, clone)
    monkeypatch.setattr(rf, "run_diff_review",
                        lambda local, **kw: rf.ReviewResult(repository="u", base="b", head_sha="h",
                                                            changed_files=["a.py"], inline_findings=[],
                                                            summary_findings=[], conclusion="success",
                                                            dropped={}))
    out = review.run_findings_review(_remote_review_args())
    assert out["conclusion"] == "success"
    assert not clone.exists()


def test_run_findings_review_url_removes_the_clone_when_the_agent_fails(monkeypatch, tmp_path):
    """The failure path leaks just the same unless the cleanup is in a finally."""
    clone = tmp_path / "codna-review-def"
    review = _stub_remote_pr(monkeypatch, clone)

    def boom(local, **kw):
        assert clone.exists()  # the clone is still there WHILE the agent runs
        raise RuntimeError("agent died")

    monkeypatch.setattr(rf, "run_diff_review", boom)
    with pytest.raises(review.ReviewError, match="review agent failed"):
        review.run_findings_review(_remote_review_args())
    assert not clone.exists()


def test_run_findings_review_local_path_is_never_removed(monkeypatch, tmp_path):
    """Only what THIS call cloned is removed -- a user's local checkout is not ours to delete."""
    from codna import review

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(rf, "run_diff_review",
                        lambda local, **kw: rf.ReviewResult(repository="u", base="b", head_sha="h",
                                                            changed_files=["a.py"], inline_findings=[],
                                                            summary_findings=[], conclusion="success",
                                                            dropped={}))
    args = SimpleNamespace(repo=str(tmp_path), pr=None, diff=None, base=None, post=False, model=None,
                           github_token=None, min_confidence=None, max_findings=None, blocking=False)
    review.run_findings_review(args)
    assert tmp_path.exists() and (tmp_path / "a.py").exists()


def test_run_diff_review_end_to_end_real_git(tmp_path, monkeypatch):
    """Full pipeline against a REAL git working-tree diff (only the agent is stubbed)."""
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})

    (tmp_path / "a.py").write_text("def f(xs):\n    return xs[0]\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "init")
    (tmp_path / "a.py").write_text("def f(xs):\n    x = xs[0]\n    return x\n", encoding="utf-8")

    diff, changed = rf.compute_diff(str(tmp_path), base="HEAD")
    assert changed == ["a.py"]
    line = sorted(rf.parse_diff(diff)["a.py"].added_lines)[0]  # a real changed new-side line

    monkeypatch.setattr(
        rf, "run_review_agent",
        lambda cwd, prompt, **kw: (
            [{"path": "a.py", "line": line, "severity": "high", "category": "correctness",
              "title": "bug", "explanation": "boom", "confidence": 0.95}],
            {"model": "stub"},
        ),
    )
    res = rf.run_diff_review(str(tmp_path), repository="local", base="HEAD", config=rf.ReviewConfig())
    assert res.changed_files == ["a.py"]
    assert len(res.inline_findings) == 1 and res.inline_findings[0].inline
    assert res.inline_findings[0].diff_anchor.side == "RIGHT"
    assert res.conclusion == "neutral"  # non-blocking default, even for a high-severity finding


def test_fetch_pr_parses_metadata(monkeypatch):
    class _Resp:
        status_code = 200
        def json(self):
            return {"base": {"ref": "main"},
                    "head": {"sha": "abc", "ref": "feature", "repo": {"full_name": "o/r", "fork": False}}}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    meta = rg.fetch_pr("o/r", 5, "tok")
    assert meta["base_ref"] == "main" and meta["head_sha"] == "abc" and meta["is_fork"] is False


# ---- smoke-test-surfaced fixes: slug-from-URL + follow-redirects ------------------------------

def test_repo_slug_from_url():
    assert rg.repo_slug_from_url("https://github.com/o/r.git") == "o/r"
    assert rg.repo_slug_from_url("https://github.com/o/r") == "o/r"
    assert rg.repo_slug_from_url("https://github.com/o/r/") == "o/r"
    assert rg.repo_slug_from_url("git@github.com:o/r.git") == "o/r"
    assert rg.repo_slug_from_url("nonsense") is None


def test_github_api_calls_follow_redirects(monkeypatch):
    # GitHub 301/307-redirects renamed repos/owners; a client that doesn't follow loses the request.
    captured = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"base": {"ref": "m"}, "head": {"sha": "s", "ref": "f",
                                                    "repo": {"full_name": "o/r", "fork": False}}}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: captured.update(k) or _Resp())
    rg.fetch_pr("o/r", 1, "t")
    assert captured.get("follow_redirects") is True


def test_url_review_derives_slug_from_url_not_cwd(monkeypatch, tmp_path):
    # Regression: `codna review <url> --pr 1` run from inside ANOTHER repo must use the URL's slug,
    # never the CWD's git remote (which used to misroute the review/post to the wrong repo).
    from codna import review
    seen = {}
    monkeypatch.setattr(rg, "fetch_pr",
                        lambda slug, n, tok: seen.update(slug=slug, n=n) or {"base_ref": "main", "head_sha": "h"})
    monkeypatch.setattr(rg, "fetch_review_context",
                        lambda *a, **k: {"last_reviewed_head": None, "prior_feedback": None})
    monkeypatch.setattr(review, "_materialize_pr",
                        lambda url, n, base, tok, incremental_base=None: (str(tmp_path), "origin/main...codna-pr-head", "h", None))
    monkeypatch.setattr(rf, "run_diff_review",
                        lambda local, **kw: rf.ReviewResult(repository="u", base="b", head_sha="h",
                                                            changed_files=["a.py"], inline_findings=[],
                                                            summary_findings=[], conclusion="success", dropped={}))
    args = SimpleNamespace(repo="https://github.com/o/r.git", pr="1", diff=None, base=None, post=False,
                           model=None, github_token="t", min_confidence=None, max_findings=None, blocking=False)
    review.run_findings_review(args)
    assert seen["slug"] == "o/r" and seen["n"] == 1  # slug from the URL, number from --pr


# ---- finding marker round-trip (the @codna fix wedge's reconstruction source) -----------------

def _finding(**kw):
    base = dict(path="src/a.py", line=10, severity="high", category="security", title="SQL injection",
                explanation="user input reaches the query unescaped", confidence=0.9,
                fingerprint="abcdef012345", end_line=12)
    base.update(kw)
    return rf.CodnaReviewFinding(**base)


def test_comment_body_carries_both_markers_and_the_fix_footer():
    body = rg.comment_body(_finding())
    assert "codna:review:fp=abcdef012345" in body   # dedup marker (unchanged)
    assert "codna:finding" in body                    # machine-readable finding marker (new)
    assert "@codna fix" in body                        # per-finding wedge footer


def test_finding_marker_roundtrips_losslessly():
    f = _finding()
    parsed = rg.parse_codna_finding(rg.comment_body(f))
    assert parsed is not None
    assert parsed["path"] == "src/a.py" and parsed["line"] == 10 and parsed["end_line"] == 12
    assert parsed["severity"] == "high" and parsed["category"] == "security"
    assert parsed["title"] == "SQL injection" and "unescaped" in parsed["explanation"]
    assert parsed["fp"] == "abcdef012345"


def test_parse_codna_finding_rejects_non_marker_and_invalid_domain():
    assert rg.parse_codna_finding("just a human comment, no marker") is None
    assert rg.parse_codna_finding("") is None
    # a forged/garbage marker with an out-of-domain severity is rejected
    assert rg.parse_codna_finding('<!-- codna:finding {"path":"a.py","line":1,"severity":"critical","category":"security","title":"x"} -->') is None
    # missing/invalid line is rejected
    assert rg.parse_codna_finding('<!-- codna:finding {"path":"a.py","line":"nope","severity":"high","category":"security","title":"x"} -->') is None


def test_finding_marker_survives_special_chars_in_explanation():
    # explanation with markdown/backticks/quotes/em-dash must not break the JSON marker
    f = _finding(explanation='has `code`, "quotes", and — an em dash; and a }brace{')
    parsed = rg.parse_codna_finding(rg.comment_body(f))
    assert parsed is not None and "em dash" in parsed["explanation"]


def test_marker_in_explanation_does_not_shadow_the_real_trailing_marker():
    # Reviewing marker code: the explanation itself contains `<!-- codna:finding XXXX -->` text.
    # parse must skip that spurious/undecodable marker and return the REAL appended finding.
    f = _finding(title="marker regex bug",
                 explanation="The regex matches <!-- codna:finding XXXX --> anywhere in the body.")
    parsed = rg.parse_codna_finding(rg.comment_body(f))
    assert parsed is not None and parsed["title"] == "marker regex bug" and parsed["path"] == "src/a.py"


def test_parse_returns_last_valid_marker_when_two_are_present():
    # Two valid markers in one body → the LAST (Codna's appended real one) wins.
    a = rg.finding_marker(_finding(title="stale", line=1))
    b = rg.finding_marker(_finding(title="real", line=42))
    assert rg.parse_codna_finding(f"prose {a}\nmore {b}")["title"] == "real"


# ---- parity wins: effort, incremental, prior-feedback ----------------------------------------

def test_effort_iterations_mapping():
    assert rf.effort_iterations("low") < rf.effort_iterations("medium") < rf.effort_iterations("high")
    assert rf.effort_iterations("bogus") == rf.effort_iterations("medium")  # default fallback


def test_config_parses_effort_and_incremental():
    c = rf.review_config_from_dict({"effort": "high", "incremental": False})
    assert c.effort == "high" and c.incremental is False
    d = rf.review_config_from_dict({"effort": "nonsense"})
    assert d.effort == "medium"          # invalid effort ignored → default
    e = rf.ReviewConfig()
    assert e.effort == "medium" and e.incremental is True


def test_build_review_prompt_includes_prior_feedback():
    p = rf.build_review_prompt("diff", ["a.py"], None, rf.ReviewConfig(),
                               prior_feedback="- @alice (a.py:3): this looks risky")
    assert "already on this PR" in p and "do not repeat" in p.lower() and "@alice" in p
    # absent when no prior feedback
    assert "already on this PR" not in rf.build_review_prompt("diff", ["a.py"], None, rf.ReviewConfig())


def test_summary_body_stamps_reviewed_head_for_incremental():
    res = rf.ReviewResult(repository="r", base="b", head_sha="abc1234def", changed_files=["a.py"],
                          inline_findings=[], summary_findings=[], conclusion="success", dropped={})
    assert "codna:reviewed-head=abc1234def" in rg.summary_body(res)


def test_fetch_review_context_extracts_head_and_human_feedback(monkeypatch):
    import httpx

    class _R:
        status_code = 200
        def __init__(self, data): self._d = data
        def json(self): return self._d

    def fake_get(url, **kw):
        if "/reviews" in url:  # review bodies carry the reviewed-head marker; last (newest) wins
            return _R([{"body": "old <!-- codna:reviewed-head=aaaaaaa -->"},
                       {"body": "new <!-- codna:reviewed-head=bbbbbbb -->"}])
        if "/pulls/" in url and "/comments" in url:  # inline comments
            return _R([{"body": "this looks risky", "user": {"login": "alice"}, "path": "a.py", "line": 3},
                       {"body": "finding <!-- codna:review:fp=deadbeef01 --> <!-- codna:finding X -->",
                        "user": {"login": "codna-ai[bot]"}}])
        if "/issues/" in url and "/comments" in url:  # conversation comments
            return _R([{"body": "please fix the naming", "user": {"login": "bob"}}])
        return _R([])

    monkeypatch.setattr(httpx, "get", fake_get)
    ctx = rg.fetch_review_context("o/r", 1, "tok")
    assert ctx["last_reviewed_head"] == "bbbbbbb"                 # newest reviewed head
    assert "@alice" in ctx["prior_feedback"] and "@bob" in ctx["prior_feedback"]
    assert "codna:finding" not in ctx["prior_feedback"]          # Codna's own comment skipped


def test_post_review_completes_the_callers_check_run_instead_of_opening_a_second(monkeypatch):
    """The webhook opens `codna review` before the CLI runs; with its id the CLI completes THAT run
    (one check on the PR). Without an id -- the standalone CLI -- behavior is unchanged."""
    from codna import webhook_github

    calls = []
    monkeypatch.setattr(webhook_github, "create_check_run", lambda *a, **k: calls.append(("create", k["name"])) or 777)
    monkeypatch.setattr(webhook_github, "update_check_run",
                        lambda repo, token, cid, **k: calls.append(("update", cid, k["name"], k["conclusion"])))
    result = rf.ReviewResult(repository="u", base="b", head_sha="h", changed_files=["a.py"], inline_findings=[],
                             summary_findings=[], conclusion="success", dropped={})
    out = rg.post_review(result, repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9001)
    assert calls == [("update", 9001, "codna review", "success")]        # no create: ONE check
    assert out["check"] == {"id": 9001, "conclusion": "success", "updated_existing": True}

    calls.clear()
    out = rg.post_review(result, repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert [c[0] for c in calls] == ["create", "update"] and calls[0][1] == "codna review"
    assert out["check"]["updated_existing"] is False


def test_env_check_run_id_parses_only_digits(monkeypatch):
    from codna import review

    monkeypatch.delenv("CODNA_CHECK_RUN_ID", raising=False)
    assert review._env_check_run_id() is None
    monkeypatch.setenv("CODNA_CHECK_RUN_ID", "9001")
    assert review._env_check_run_id() == 9001
    monkeypatch.setenv("CODNA_CHECK_RUN_ID", "abc")
    assert review._env_check_run_id() is None


# --- the review is a review: contract in the user turn, one bounded repair, read-only -----------
def test_build_review_prompt_restates_the_output_contract_and_says_not_to_fix():
    p = rf.build_review_prompt("diff", ["a.py"], None, rf.ReviewConfig())
    assert "REVIEWING this diff, not fixing it" in p
    assert 'FINAL message MUST be a single JSON object' in p and '{"findings": []}' in p
    assert p.index("REVIEWING this diff") < p.index("----- BEGIN DIFF -----")   # contract before the diff


class _Result:
    def __init__(self, text, *, files_changed=(), patch_diff="", model="stub-model"):
        self.status, self.terminal_state = "succeeded", "succeeded"
        self.agent_run_id, self.session_id = "run_1", "sess_1"
        self.text, self.files_changed, self.patch_diff = text, list(files_changed), patch_diff
        self.telemetry = {"model": model}
        self.artifacts, self.runtime = {}, {}


def _stub_sidecar(monkeypatch, results):
    """Stub everything run_review_agent touches outside the repo: keys, runtime config, the runner."""
    import codna.cli as cli_mod
    import codna.runtime.config as rc
    import codna.packaged_agent_runner as par

    monkeypatch.setattr(cli_mod, "_runtime_keys", lambda include_keychain=True: {})
    monkeypatch.setattr(rc, "resolve_runtime_config", lambda keys=None: object())
    seen = []

    class _Runner:
        def __init__(self, *, config, keys):
            pass

        def run(self, request):
            seen.append(request)
            return results.pop(0)

    monkeypatch.setattr(par, "SidecarPackagedAgentRunner", _Runner)
    return seen


PROSE = ('The diff introduces a deliberately broken "Install deps" step that always exits with `exit 1`. '
         "This is a probe step that must be removed so the CI pipeline can actually function. I'll remove it now.")


def test_run_review_agent_repairs_a_prose_reply_once(monkeypatch, tmp_path):
    seen = _stub_sidecar(monkeypatch, [_Result(PROSE), _Result('{"findings": [{"path": "a.py", "line": 1}]}')])
    raw, agent = rf.run_review_agent(str(tmp_path), "Review the following pull-request diff ...")
    assert raw == [{"path": "a.py", "line": 1}]
    assert agent["repaired"] is True and agent["first_attempt"]["model"] == "stub-model"
    assert len(seen) == 2 and seen[0].task_kind == seen[1].task_kind == "review"
    # the re-ask states the contract, quotes what the model said, and repeats the whole task
    repair = seen[1].issue_text
    assert repair.startswith("Your previous reply to the review task below was NOT the required JSON.")
    assert "I'll remove it now" in repair and "reviewing, not fixing" in repair
    assert repair.endswith("Review the following pull-request diff ...")
    assert seen[1].snapshot_id != seen[0].snapshot_id                          # a new single-shot run


def test_run_review_agent_fails_closed_after_a_second_prose_reply(monkeypatch, tmp_path):
    from codna.cline_agent import UnparseableFindingsError
    seen = _stub_sidecar(monkeypatch, [_Result(PROSE), _Result("Still prose. No JSON here.")])
    with pytest.raises(UnparseableFindingsError) as exc:
        rf.run_review_agent(str(tmp_path), "Review ...")
    assert len(seen) == 2                                                       # exactly one repair, never a third
    d = exc.value.details
    assert d["attempts"] == 2 and d["model"] == "stub-model" and d["terminal_state"] == "succeeded"
    assert d["raw_head"].startswith("Still prose") and d["first_attempt"]["raw_head"].startswith("The diff introduces")
    assert exc.value.code == "review_unparseable_output"                        # -> cause_code in the check summary


def test_run_review_agent_does_not_repair_a_good_reply(monkeypatch, tmp_path):
    seen = _stub_sidecar(monkeypatch, [_Result('{"findings": []}')])
    raw, agent = rf.run_review_agent(str(tmp_path), "Review ...")
    assert raw == [] and "repaired" not in agent and len(seen) == 1


def test_run_review_agent_refuses_findings_from_a_run_that_touched_the_tree(monkeypatch, tmp_path):
    from codna.cline_agent import ClineAgentError
    _stub_sidecar(monkeypatch, [_Result('{"findings": []}', files_changed=["a.py"], patch_diff="--- a.py\n+++ a.py\n")])
    with pytest.raises(ClineAgentError) as exc:
        rf.run_review_agent(str(tmp_path), "Review ...")
    assert "modified the workspace" in str(exc.value) and exc.value.details["files_changed"] == ["a.py"]


def test_unparseable_findings_error_is_a_cline_agent_error_with_its_own_code():
    from codna.cline_agent import ClineAgentError, UnparseableFindingsError, extract_findings_json
    with pytest.raises(UnparseableFindingsError) as exc:
        extract_findings_json("just prose")
    assert isinstance(exc.value, ClineAgentError) and exc.value.code == "review_unparseable_output"


# --- a clean review APPROVES (Scorecard Code-Review + "require approvals" both need an APPROVED review) ---
def _af(sev="high", fp="f1", line=1):
    """An INLINE finding for the approval tests (anchored, as real inline findings are; distinct name:
    `_finding` above has a different signature)."""
    return rf.CodnaReviewFinding(path="a.py", line=line, severity=sev, category="correctness", title=f"t-{fp}",
                                 explanation="x", confidence=0.9, fingerprint=fp, end_line=None,
                                 diff_anchor=rf.DiffAnchor(commit_id="h" * 40, side="RIGHT", line=line))


def _ar(findings=()):
    return rf.ReviewResult(repository="u", base="b", head_sha="h" * 40, changed_files=["a.py"],
                           inline_findings=list(findings), summary_findings=[], conclusion="neutral" if findings else "success",
                           dropped={})


def test_review_event_approves_only_a_clean_pr_with_no_unresolved_blocking_threads():
    assert rg.review_event(_ar(), unresolved_blocking=0) == "APPROVE"
    assert rg.review_event(_ar([_af("low")]), unresolved_blocking=0) == "APPROVE"     # low never blocks
    assert rg.review_event(_ar([_af("medium")]), unresolved_blocking=0) == "COMMENT"
    assert rg.review_event(_ar([_af("high")]), unresolved_blocking=0) == "COMMENT"
    assert rg.review_event(_ar(), unresolved_blocking=1) == "COMMENT"                     # earlier finding still open
    assert rg.review_event(_ar(), unresolved_blocking=None) == "COMMENT"                  # unknown -> fail closed
    assert rg.review_event(_ar(), unresolved_blocking=0, approve=False) == "COMMENT"      # turned off


def test_approval_note_explains_why_not():
    assert "Approved" in rg.approval_note("APPROVE", _ar(), 0)
    n = rg.approval_note("COMMENT", _ar([_af("high")]), 2)
    assert "1 medium/high finding" in n and "2 earlier codna finding thread(s)" in n
    assert "could not be read" in rg.approval_note("COMMENT", _ar(), None)
    assert "turned off" in rg.approval_note("COMMENT", _ar(), 0, approve=False)


def test_build_review_payload_carries_the_event_and_the_approval_line():
    p = rg.build_review_payload(_ar(), event="APPROVE", approval="✅ Approved: x")
    assert p["event"] == "APPROVE" and "✅ Approved: x" in p["body"] and p["comments"] == []
    assert rg.build_review_payload(_ar())["event"] == "COMMENT"          # default unchanged


def _thread(body, resolved=False, typename="Bot"):
    return {"isResolved": resolved, "comments": {"nodes": [{"body": body, "author": {"login": "codna-ai[bot]", "__typename": typename}}]}}


def test_unresolved_blocking_findings_counts_only_open_bot_threads_at_medium_or_high(monkeypatch):
    import httpx
    high = rg.comment_body(_af("high", "aa"))
    low = rg.comment_body(_af("low", "bb"))
    medium = rg.comment_body(_af("medium", "cc"))
    nodes = [_thread(high), _thread(high, resolved=True), _thread(low), _thread(medium),
             _thread(medium, typename="User"), _thread("a human comment")]
    class _Resp:
        status_code = 200
        def json(self):
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": False}, "nodes": nodes}}}}}
    captured = {}
    monkeypatch.setattr(httpx, "post", lambda url, **k: captured.update(k) or _Resp())
    assert rg.unresolved_blocking_findings("o/r", 7, "tok") == 2                 # one open high + one open medium
    assert captured["json"]["variables"] == {"owner": "o", "name": "r", "number": 7, "first": 100}


def test_unresolved_blocking_findings_is_unknown_on_errors_or_overflow(monkeypatch):
    import httpx
    class _Err:
        status_code = 502
        def json(self): return {}
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Err())
    assert rg.unresolved_blocking_findings("o/r", 7, "tok") is None
    class _More:
        status_code = 200
        def json(self):
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": True}, "nodes": []}}}}}
    monkeypatch.setattr(httpx, "post", lambda url, **k: _More())
    assert rg.unresolved_blocking_findings("o/r", 7, "tok") is None
    assert rg.unresolved_blocking_findings("o/r", 7, None) is None


def _stub_checks(monkeypatch, pr_files=None):
    """Stub the check-run calls and the PR file listing post_review reads before anchoring
    (``pr_files=None`` = unknown, so nothing is demoted and the approval logic is tested alone)."""
    from codna import webhook_github
    monkeypatch.setattr(webhook_github, "create_check_run", lambda *a, **k: 777)
    monkeypatch.setattr(webhook_github, "update_check_run", lambda *a, **k: None)
    monkeypatch.setattr(rg, "fetch_pr_files", lambda *a, **k: pr_files)


def test_post_review_posts_an_approve_for_a_clean_pr(monkeypatch):
    """A clean review used to post NOTHING (only the check), so no PR ever carried an approval."""
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    posts = []
    class _Ok:
        status_code = 200
        text = ""
        def json(self): return {}
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append((url, k["json"])) or _Ok())
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert out["event"] == "APPROVE" and out["posted_review"] is True and out["unresolved_blocking"] == 0
    assert posts[0][0].endswith("/repos/o/r/pulls/5/reviews") and posts[0][1]["event"] == "APPROVE"
    assert "✅ Approved" in posts[0][1]["body"]


def test_post_review_comments_when_a_medium_finding_or_an_open_thread_remains(monkeypatch):
    import httpx
    _stub_checks(monkeypatch)
    posts = []
    class _Ok:
        status_code = 200
        text = ""
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append(k["json"]) or _Ok())
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    out = rg.post_review(_ar([_af("medium")]), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert out["event"] == "COMMENT" and posts[-1]["event"] == "COMMENT" and "Not approved" in posts[-1]["body"]
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 3)
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert out["event"] == "COMMENT" and out["posted_review"] is False       # clean pass, but 3 open threads: nothing to post
    assert out["unresolved_blocking"] == 3


def test_post_review_falls_back_to_a_comment_when_github_refuses_the_approval(monkeypatch):
    """The App cannot approve a PR it authored (its own fix PRs): GitHub answers 422."""
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    posts = []
    class _Resp:
        def __init__(self, code): self.status_code, self.text = code, "Unprocessable"
    def fake_post(url, **k):
        posts.append(k["json"]["event"])
        return _Resp(422 if k["json"]["event"] == "APPROVE" else 200)
    monkeypatch.setattr(httpx, "post", fake_post)
    # clean + refused: nothing else to say -> no second post, no error
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert posts == ["APPROVE"] and out["event"] == "COMMENT" and out["posted_review"] is False
    # low-only findings + refused: the findings are still posted as a COMMENT
    posts.clear()
    out = rg.post_review(_ar([_af("low")]), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert posts == ["APPROVE", "COMMENT"] and out["event"] == "COMMENT" and out["posted_review"] is True


def test_post_review_approve_flag_off_never_calls_the_thread_lookup(monkeypatch):
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: pytest.fail("must not be called"))
    monkeypatch.setattr(httpx, "post", lambda url, **k: pytest.fail("nothing to post for a clean, approval-off review"))
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False, approve=False)
    assert out["event"] == "COMMENT" and out["posted_review"] is False


def test_review_config_parses_the_approve_switch():
    assert rf.ReviewConfig().approve_clean is True
    assert rf.review_config_from_dict({"approve": False}).approve_clean is False
    assert rf.review_config_from_dict({"approve": "no"}).approve_clean is True          # only a real bool counts


# --- an incremental review must not swallow an "Update branch" merge (thyn-ai/codna-action#6) -------
#
# PR #6's head bc1219d3 was `Merge branch 'main' into <branch>`: first parent 4f110616 = the head codna
# had reviewed (so the ancestor test passed and the range narrowed to 4f110616...head), second parent
# = main after PR #7 merged. That range's diff was everything main gained -- 27 files, none in the
# PR -- findings anchored to them, and GitHub refused the review: 422 "Path could not be resolved".

def _git_env():
    return {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _update_branch_fixture(tmp_path):
    """A bare "GitHub" remote whose PR #6 branch was reviewed at ``reviewed_head``; then main gained
    files the PR never touches, and the author pressed "Update branch" (main merged into the branch).
    Everything a test asserts against is read back from the repository, never hardcoded."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"

    def git(*a, cwd=work):
        return subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, text=True,
                              env=_git_env()).stdout.strip()

    remote.mkdir()
    git("init", "-q", "--bare", cwd=remote)
    git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
    work.mkdir()
    git("init", "-q")
    git("checkout", "-q", "-b", "main")
    (work / "app.py").write_text("def f(xs):\n    return xs[0]\n", encoding="utf-8")
    git("add", "-A"), git("commit", "-qm", "base")
    git("remote", "add", "origin", str(remote))
    git("push", "-q", "origin", "main")

    git("checkout", "-q", "-b", "feature")
    (work / "app.py").write_text("def f(xs):\n    if not xs:\n        return None\n    return xs[0]\n", encoding="utf-8")
    git("add", "-A"), git("commit", "-qm", "guard the empty list")
    reviewed_head = git("rev-parse", "HEAD")
    git("push", "-q", "origin", "feature", "feature:refs/pull/6/head")

    git("checkout", "-q", "main")
    (work / ".github" / "workflows").mkdir(parents=True)
    (work / ".github" / "workflows" / "ci.yml").write_text("name: ci\non: [push]\n", encoding="utf-8")
    (work / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (work / "SECURITY.md").write_text("report privately\n", encoding="utf-8")
    git("add", "-A"), git("commit", "-qm", "bring the repository to the shared posture")
    main_files = set(git("show", "--name-only", "--format=", "HEAD").splitlines())
    git("push", "-q", "origin", "main")

    git("checkout", "-q", "feature")
    git("merge", "-q", "--no-ff", "--no-edit", "main")           # the "Update branch" button
    new_head = git("rev-parse", "HEAD")
    git("push", "-q", "-f", "origin", "feature", "feature:refs/pull/6/head")
    pr_files = git("diff", "--name-only", "main...feature").splitlines()
    parents = git("rev-list", "--parents", "-n1", "HEAD").split()[1:]
    return SimpleNamespace(remote=remote, work=work, git=git, reviewed_head=reviewed_head, new_head=new_head,
                           main_sha=git("rev-parse", "main"), main_files=main_files, pr_files=pr_files,
                           parents=parents)


def test_incremental_review_over_an_update_branch_merge_reviews_nothing_from_main(tmp_path, monkeypatch):
    """REGRESSION (codna-action#6): the head is the merge of main into the branch; the previously
    reviewed head is its first parent, so the range narrows -- and used to diff in everything main
    gained. Confined to the PR's own files the range has nothing left: a clean, agent-free result,
    with no anchor on any path outside the PR."""
    import shutil
    from codna import review

    fx = _update_branch_fixture(tmp_path)
    assert fx.parents == [fx.reviewed_head, fx.main_sha]              # the exact shape of bc1219d3
    assert fx.main_files and not (fx.main_files & set(fx.pr_files))   # main's files are not the PR's

    local, diff_range, head_sha, diff_paths = review._materialize_pr(str(fx.remote), 6, "main", None,
                                                                     incremental_base=fx.reviewed_head)
    try:
        assert head_sha == fx.new_head
        assert diff_range == f"{fx.reviewed_head}...codna-pr-head"      # narrowed: the old head IS an ancestor
        assert diff_paths == fx.pr_files                                # ...and confined to the PR's files
        # The bug's shape, for the record: unconfined, the narrowed range is main's history, not the PR's.
        _, swallowed = rf.compute_diff(local, diff_range=diff_range)
        assert set(swallowed) == fx.main_files
        # Confined, there is nothing new to review.
        assert rf.compute_diff(local, diff_range=diff_range, diff_paths=diff_paths) == ("", [])

        monkeypatch.setattr(rf, "run_review_agent", lambda *a, **k: pytest.fail("nothing to review: the agent must not run"))
        res = rf.run_diff_review(local, repository="o/r", diff_range=diff_range, config=rf.ReviewConfig(),
                                 head_sha=head_sha, diff_paths=diff_paths)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    assert res.changed_files == [] and res.findings == [] and res.conclusion == "success"
    assert res.note and "no changed files" in res.note
    assert res.head_sha == fx.new_head                                  # the next review resumes from HERE
    payload = rg.build_review_payload(res)
    assert payload["comments"] == []
    assert rg.reviewed_head_marker(fx.new_head) in payload["body"]


def test_incremental_review_after_an_update_branch_merge_still_reviews_the_new_commit(tmp_path, monkeypatch):
    """The confinement drops main's contribution, not the author's: a commit pushed after the merge is
    reviewed through the files it changes, and only those files can anchor a finding."""
    import shutil
    from codna import review

    fx = _update_branch_fixture(tmp_path)
    (fx.work / "app.py").write_text("def f(xs):\n    if not xs:\n        return None\n    return xs[-1]\n", encoding="utf-8")
    fx.git("add", "-A"), fx.git("commit", "-qm", "take the last one")
    fx.git("push", "-q", "-f", "origin", "feature", "feature:refs/pull/6/head")

    local, diff_range, head_sha, diff_paths = review._materialize_pr(str(fx.remote), 6, "main", None,
                                                                     incremental_base=fx.reviewed_head)
    try:
        assert diff_paths == fx.pr_files == ["app.py"]
        diff, changed = rf.compute_diff(local, diff_range=diff_range, diff_paths=diff_paths)
        assert changed == ["app.py"]                                     # main's files are gone from the range
        files = rf.parse_diff(diff)
        assert set(files) == {"app.py"}
        line = sorted(files["app.py"].added_lines)[0]
        main_only = sorted(fx.main_files)[0]
        monkeypatch.setattr(rf, "run_review_agent", lambda *a, **k: (
            [{"path": "app.py", "line": line, "severity": "medium", "category": "correctness",
              "title": "last element", "explanation": "x", "confidence": 0.9},
             {"path": main_only, "line": 1, "severity": "low", "category": "correctness",
              "title": "not in the pull request", "explanation": "x", "confidence": 0.9}],
            {"model": "stub"}))
        res = rf.run_diff_review(local, repository="o/r", diff_range=diff_range, config=rf.ReviewConfig(),
                                 head_sha=head_sha, diff_paths=diff_paths)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    assert [f.path for f in res.inline_findings] == ["app.py"]           # anchored: in the PR's diff
    assert [f.path for f in res.summary_findings] == [main_only]         # not anchored: main's file
    assert all(c["path"] == "app.py" for c in rg.build_review_payload(res)["comments"])


def test_compute_diff_with_an_empty_path_set_reviews_nothing_and_never_lifts_the_restriction(tmp_path):
    """``git diff <range> --`` with no pathspec would diff EVERYTHING -- the opposite of "the
    intersection is empty". An empty list must short-circuit to no changes."""
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True, env=_git_env())
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("init", "-q"), git("add", "-A"), git("commit", "-qm", "init")
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert rf.compute_diff(str(tmp_path), base="HEAD")[1] == ["a.py"]              # unrestricted: the change
    assert rf.compute_diff(str(tmp_path), base="HEAD", diff_paths=["a.py"])[1] == ["a.py"]
    assert rf.compute_diff(str(tmp_path), base="HEAD", diff_paths=["other.py"]) == ("", [])
    assert rf.compute_diff(str(tmp_path), base="HEAD", diff_paths=[]) == ("", [])


def test_run_findings_review_threads_the_incremental_path_set_to_the_diff(monkeypatch, tmp_path):
    from codna import review

    monkeypatch.setattr(rg, "fetch_pr", lambda slug, n, tok: {"base_ref": "main", "head_sha": "h", "head_ref": "f", "is_fork": False})
    monkeypatch.setattr(rg, "fetch_review_context", lambda *a, **k: {"last_reviewed_head": "0ld", "prior_feedback": None})
    monkeypatch.setattr(review, "_materialize_pr",
                        lambda url, n, base, tok, incremental_base=None: (str(tmp_path), f"{incremental_base}...codna-pr-head", "new", ["x.py"]))
    seen = {}

    def fake_run_diff_review(local, **kw):
        seen.update(kw)
        return rf.ReviewResult(repository="u", base=kw["diff_range"], head_sha=kw["head_sha"], changed_files=[],
                               inline_findings=[], summary_findings=[], conclusion="success", dropped={}, note="nothing new")

    monkeypatch.setattr(rf, "run_diff_review", fake_run_diff_review)
    out = review.run_findings_review(_remote_review_args())
    assert seen["diff_range"] == "0ld...codna-pr-head" and seen["diff_paths"] == ["x.py"]
    assert out["conclusion"] == "success" and out["note"] == "nothing new"


# --- belt and braces: never anchor outside the PR's GitHub diff, never fail the check over an anchor ---

def _inline_on(path, title, sev="medium", line=2):
    return rf.CodnaReviewFinding(path=path, line=line, severity=sev, category="correctness", title=title,
                                 explanation="x", confidence=0.9, fingerprint=rf.fingerprint(path, "correctness", title),
                                 diff_anchor=rf.DiffAnchor(commit_id="h" * 40, side="RIGHT", line=line))


def test_demote_to_summary_keeps_every_finding_and_leaves_the_input_alone():
    inside, outside = _inline_on("src/a.py", "inside"), _inline_on("LICENSE", "outside")
    res = _ar([inside, outside])
    kept = rg.demote_to_summary(res, keep_paths={"src/a.py"})
    assert [f.path for f in kept.inline_findings] == ["src/a.py"]
    assert [f.path for f in kept.summary_findings] == ["LICENSE"]
    demoted = kept.summary_findings[0]
    assert demoted.diff_anchor is None and demoted.fingerprint == outside.fingerprint and demoted.title == "outside"
    assert outside.diff_anchor is not None and res.inline_findings == [inside, outside]      # a copy, not a mutation
    assert len(kept.findings) == len(res.findings)
    everything = rg.demote_to_summary(res)
    assert everything.inline_findings == [] and len(everything.summary_findings) == len(res.findings)
    assert rg.demote_to_summary(res, keep_paths={"src/a.py", "LICENSE"}) is res             # nothing to do
    # already threaded on the PR (dedup's skip set): left out of the body rather than said twice
    threaded = rg.demote_to_summary(res, keep_paths={"src/a.py"}, drop_fingerprints={outside.fingerprint})
    assert threaded.inline_findings == [inside] and threaded.summary_findings == []
    assert outside.fingerprint not in rg.summary_body(threaded)


def test_build_review_payload_anchors_only_to_paths_in_the_prs_github_diff():
    inside, outside = _inline_on("src/a.py", "bug inside"), _inline_on("LICENSE", "bug outside")
    res = _ar([inside, outside])
    known = rg.build_review_payload(res, pr_files={"src/a.py"})
    assert [c["path"] for c in known["comments"]] == ["src/a.py"]
    assert "bug outside" in known["body"] and rg.fp_marker(outside.fingerprint) in known["body"]    # listed, not lost
    unknown = rg.build_review_payload(res, pr_files=None)                                          # unknown: unfiltered
    assert sorted(c["path"] for c in unknown["comments"]) == ["LICENSE", "src/a.py"]
    assert "bug outside" not in unknown["body"]


def test_fetch_pr_files_walks_every_page_and_is_unknown_on_any_failure(monkeypatch):
    import httpx

    page1 = [{"filename": f"src/f{i}.py"} for i in range(100)]
    page2 = [{"filename": "LICENSE"}, {"filename": "README.md"}, {"sha": "no filename here"}]

    class _R:
        def __init__(self, data, code=200): self._d, self.status_code = data, code
        def json(self): return self._d

    calls = []
    def paged(url, **kw):
        calls.append(kw["params"]["page"])
        assert url.endswith("/repos/o/r/pulls/6/files") and kw["follow_redirects"] is True
        return _R(page1 if kw["params"]["page"] == 1 else page2)
    monkeypatch.setattr(httpx, "get", paged)
    files = rg.fetch_pr_files("o/r", 6, "tok")
    assert calls == [1, 2]
    assert files == {e["filename"] for e in page1 + page2 if "filename" in e}

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _R({"message": "Not Found"}, 404))
    assert rg.fetch_pr_files("o/r", 6, "tok") is None
    def boom(url, **kw): raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx, "get", boom)
    assert rg.fetch_pr_files("o/r", 6, "tok") is None
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _R(page1))                      # never a short page
    assert rg.fetch_pr_files("o/r", 6, "tok") is None                                   # incomplete -> unknown


def test_post_review_demotes_findings_outside_the_prs_diff_before_posting(monkeypatch):
    import httpx
    from codna import webhook_github

    inside, outside = _inline_on("a.py", "inside the PR"), _inline_on("LICENSE", "brought in by main")
    _stub_checks(monkeypatch, pr_files={"a.py"})
    summaries = []
    monkeypatch.setattr(webhook_github, "update_check_run", lambda repo, token, cid, **k: summaries.append(k["summary"]))
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    posts = []
    class _Ok:
        status_code, text = 200, ""
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append(k["json"]) or _Ok())
    out = rg.post_review(_ar([inside, outside]), repo_slug="o/r", pr_number=6, token="t", dedup=False, check_run_id=1)
    assert len(posts) == 1                                                        # accepted first time
    assert [c["path"] for c in posts[0]["comments"]] == ["a.py"]
    assert "brought in by main" in posts[0]["body"] and "brought in by main" in summaries[-1]
    assert out["posted_review"] is True and out["inline_posted"] == 1 and out["demoted_to_summary"] == 1
    assert out["inline_posted"] + out["skipped_duplicates"] + out["demoted_to_summary"] == 2


def test_post_review_retries_once_with_every_finding_in_the_body_when_github_refuses_an_anchor(monkeypatch):
    """codna-action#6: the payload's inline comments pointed outside the PR's diff, GitHub answered
    422 "Path could not be resolved", post_review raised and the check went red -- findings lost.
    Now: ONE retry with every inline finding demoted to the body. Nothing lost, nothing red."""
    import httpx
    _stub_checks(monkeypatch)                            # file list unknown: the pre-filter cannot help here
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    findings = [_af("medium", fp="f1", line=3), _af("low", fp="f2", line=9)]
    posts = []
    class _Resp:
        def __init__(self, code, text=""): self.status_code, self.text = code, text
    def refuse_anchors(url, **k):
        posts.append(k["json"])
        if k["json"]["comments"]:
            return _Resp(422, '{"message":"Unprocessable Entity","errors":["Path could not be resolved and Path could not be resolved"]}')
        return _Resp(200)
    monkeypatch.setattr(httpx, "post", refuse_anchors)
    out = rg.post_review(_ar(findings), repo_slug="o/r", pr_number=6, token="t", dedup=False)
    assert [len(p["comments"]) for p in posts] == [len(findings), 0]        # refused once, retried once, demoted
    retry = posts[-1]
    assert retry["event"] == "COMMENT" and retry["comments"] == []
    for f in findings:
        assert f.title in retry["body"] and rg.fp_marker(f.fingerprint) in retry["body"]
    assert out["posted_review"] is True and out["inline_posted"] == 0 and out["demoted_to_summary"] == len(findings)
    assert out["inline_posted"] + out["skipped_duplicates"] + out["demoted_to_summary"] == len(findings)

    # ONCE: a 422 to the demoted retry is a real failure and is raised, not retried again.
    posts.clear()
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append(k["json"]) or _Resp(422, "still no"))
    with pytest.raises(RuntimeError, match="create review failed: 422"):
        rg.post_review(_ar(findings), repo_slug="o/r", pr_number=6, token="t", dedup=False)
    assert len(posts) == 2


def test_post_review_anchor_retry_composes_with_the_refused_approval_fallback(monkeypatch):
    """A low-only review APPROVES with its inline comments. The App's own PR + an unanchorable path:
    APPROVE refused -> COMMENT (existing fallback) -> comments refused -> demoted retry. Each fallback
    fires once, the approval one first, and the review lands."""
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    posts = []
    class _Resp:
        def __init__(self, code): self.status_code, self.text = code, "Unprocessable"
    def github(url, **k):
        posts.append(k["json"])
        return _Resp(422 if (k["json"]["event"] == "APPROVE" or k["json"]["comments"]) else 200)
    monkeypatch.setattr(httpx, "post", github)
    out = rg.post_review(_ar([_af("low")]), repo_slug="o/r", pr_number=6, token="t", dedup=False)
    assert [(p["event"], len(p["comments"])) for p in posts] == [("APPROVE", 1), ("COMMENT", 1), ("COMMENT", 0)]
    assert out["event"] == "COMMENT" and out["posted_review"] is True and out["demoted_to_summary"] == 1
    assert out["inline_posted"] + out["skipped_duplicates"] + out["demoted_to_summary"] == 1
    assert "cannot approve its own pull request" in posts[-1]["body"] and "t-f1" in posts[-1]["body"]


def test_post_review_retry_does_not_repeat_a_finding_already_threaded_on_the_pr(monkeypatch):
    """Dedup keeps an already-posted fingerprint out of the inline comments; the demoted retry must
    keep it out of the body too, or the same finding is said twice on one head. The counters still
    partition what this pass found, and the event is decided from all of it."""
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    threaded, fresh = _inline_on("a.py", "already threaded", sev="medium"), _inline_on("a.py", "new this pass", sev="low", line=7)

    class _R:
        status_code = 200
        def json(self): return [{"body": "earlier review " + rg.fp_marker(threaded.fingerprint)}]
    monkeypatch.setattr(httpx, "get", lambda url, **k: _R())              # the PR already carries `threaded`
    posts = []
    class _Resp:
        def __init__(self, code, text=""): self.status_code, self.text = code, text
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append(k["json"]) or _Resp(422 if k["json"]["comments"] else 200))
    out = rg.post_review(_ar([threaded, fresh]), repo_slug="o/r", pr_number=6, token="t", dedup=True)
    assert [len(p["comments"]) for p in posts] == [1, 0]                   # only `fresh` was ever a comment
    assert "new this pass" in posts[-1]["body"] and rg.fp_marker(fresh.fingerprint) in posts[-1]["body"]
    assert "already threaded" not in posts[-1]["body"] and threaded.fingerprint not in posts[-1]["body"]
    assert out["event"] == "COMMENT"                                        # the medium finding still counts
    assert (out["inline_posted"], out["skipped_duplicates"], out["demoted_to_summary"]) == (0, 1, 1)


# ---- codna#569: the check's anchor, the reviewed head and the PR's head, side by side -------------
_REVIEWED = "h" * 40          # _ar().head_sha
_MOVED_TO = "5a3ba5ff" + "0" * 32


def _capture_check_summaries(monkeypatch):
    from codna import webhook_github
    summaries = []
    monkeypatch.setattr(webhook_github, "create_check_run", lambda *a, **k: 777)
    monkeypatch.setattr(webhook_github, "update_check_run", lambda *a, **k: summaries.append(k["summary"]))
    monkeypatch.setattr(rg, "fetch_pr_files", lambda *a, **k: None)
    return summaries


def test_post_review_reports_the_check_anchor_and_the_pr_head_next_to_the_reviewed_head(monkeypatch):
    """The three commits automation needs to compare (algenta#1085 had check != reviewed head)."""
    import httpx
    _stub_checks(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    class _Ok:
        status_code, text = 200, ""
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Ok())
    # the webhook's run: the anchor is whatever the caller says, the PR head is what it read at post time
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9001,
                         check_head_sha=_REVIEWED, pr_head_sha=_REVIEWED)
    assert out["check_head_sha"] == _REVIEWED and out["pr_head_sha"] == _REVIEWED and out["event"] == "APPROVE"
    # the standalone CLI: post_review opens the run itself, on the reviewed head
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False)
    assert out["check_head_sha"] == _REVIEWED and out["pr_head_sha"] is None
    # a caller that knows neither: unknown is reported as unknown, never guessed
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9001)
    assert out["check_head_sha"] is None and out["pr_head_sha"] is None


def test_post_review_never_approves_a_head_that_moved_while_the_review_ran(monkeypatch):
    """An approval counts for the whole pull request, and nobody reviewed the new head: a clean
    review of the old one is posted as a COMMENT (or, with nothing else to say, not at all) and
    the check says why. Findings still go up -- they are anchored to the commit that was reviewed."""
    import httpx
    summaries = _capture_check_summaries(monkeypatch)
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    posts = []
    class _Ok:
        status_code, text = 200, ""
    monkeypatch.setattr(httpx, "post", lambda url, **k: posts.append(k["json"]) or _Ok())
    moved = dict(repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9001,
                 check_head_sha=_REVIEWED, pr_head_sha=_MOVED_TO)
    # clean: would have been an APPROVE
    out = rg.post_review(_ar(), **moved)
    assert out["event"] == "COMMENT" and out["posted_review"] is False and posts == []
    assert out["pr_head_sha"] == _MOVED_TO and out["check_head_sha"] == _REVIEWED
    assert f"head moved from {_REVIEWED[:8]} to {_MOVED_TO[:8]}" in summaries[-1] and "✅ Approved" not in summaries[-1]
    # low-only: still no approval, the finding is posted as a COMMENT that says why
    out = rg.post_review(_ar([_af("low")]), **moved)
    assert out["event"] == "COMMENT" and out["posted_review"] is True
    assert posts[-1]["event"] == "COMMENT" and f"moved from {_REVIEWED[:8]}" in posts[-1]["body"]
    assert posts[-1]["commit_id"] == _REVIEWED                               # the review is about the commit it read
    # medium: not approved anyway; the move is still on record next to the reason
    out = rg.post_review(_ar([_af("medium")]), **moved)
    assert out["event"] == "COMMENT" and "1 medium/high finding(s)" in posts[-1]["body"]
    assert f"moved from {_REVIEWED[:8]}" in posts[-1]["body"]
    # the same head at post time: nothing changes
    out = rg.post_review(_ar(), **{**moved, "pr_head_sha": _REVIEWED})
    assert out["event"] == "APPROVE" and posts[-1]["event"] == "APPROVE"


def test_env_check_run_head_sha_parses_only_a_sha(monkeypatch):
    from codna import review

    monkeypatch.delenv("CODNA_CHECK_RUN_HEAD_SHA", raising=False)
    assert review._env_check_run_head_sha() is None
    monkeypatch.setenv("CODNA_CHECK_RUN_HEAD_SHA", _MOVED_TO.upper())
    assert review._env_check_run_head_sha() == _MOVED_TO
    monkeypatch.setenv("CODNA_CHECK_RUN_HEAD_SHA", "not-a-sha")
    assert review._env_check_run_head_sha() is None


def test_run_findings_review_hands_post_review_the_head_at_post_time_and_the_checks_anchor(monkeypatch, tmp_path):
    """`codna review <url> --pr N --post --json`: the PR head is read AGAIN right before posting (the
    turn is minutes long), the check's anchor comes from the worker's env, and the JSON result carries
    what post_review reported."""
    from codna import review

    heads = iter([_REVIEWED, _MOVED_TO])          # at turn start, then at post time
    monkeypatch.setattr(rg, "fetch_pr",
                        lambda slug, n, tok: {"base_ref": "main", "head_sha": next(heads), "head_ref": "f", "is_fork": False})
    monkeypatch.setattr(rg, "fetch_review_context", lambda *a, **k: {"last_reviewed_head": None, "prior_feedback": None})
    monkeypatch.setattr(review, "_materialize_pr",
                        lambda url, n, base, tok, incremental_base=None: (str(tmp_path), "origin/main...codna-pr-head", _REVIEWED, None))
    monkeypatch.setattr(rf, "run_diff_review", lambda local, **kw: _ar())
    red = [{"name": "Vercel", "state": "FAILURE", "head_sha": _MOVED_TO}]   # read at post time, like the head
    monkeypatch.setattr(rg, "failing_required_checks", lambda slug, n, tok: red)
    posted = {}

    def fake_post_review(result, **kw):
        posted.update(kw)
        return {"event": "COMMENT", "check": {"id": kw["check_run_id"], "updated_existing": True},
                "check_head_sha": kw["check_head_sha"], "pr_head_sha": kw["pr_head_sha"]}

    monkeypatch.setattr(rg, "post_review", fake_post_review)
    anchor = "7679ecbe" + "0" * 32                # what the worker anchored CODNA_CHECK_RUN_ID to
    monkeypatch.setenv("CODNA_CHECK_RUN_ID", "9001")
    monkeypatch.setenv("CODNA_CHECK_RUN_HEAD_SHA", anchor)
    args = SimpleNamespace(repo="https://github.com/o/r.git", pr="o/r#5", diff=None, base=None, post=True,
                           model=None, github_token="t", min_confidence=None, max_findings=None, blocking=False)
    out = review.run_findings_review(args)
    assert posted["check_run_id"] == 9001 and posted["check_head_sha"] == anchor
    assert posted["pr_head_sha"] == _MOVED_TO                              # the head at POST time, not at start
    assert posted["failing_checks"] == red                                 # the red required checks, read at post time
    assert out["head_sha"] == _REVIEWED
    assert out["posted"]["check_head_sha"] == anchor and out["posted"]["pr_head_sha"] == _MOVED_TO


def test_pr_head_now_is_best_effort(monkeypatch):
    from codna import review

    def boom(*a, **k):
        raise RuntimeError("fetch PR #5 failed: 502")

    monkeypatch.setattr(rg, "fetch_pr", boom)
    assert review._pr_head_now("o/r", 5, "t") is None
    assert review._pr_head_now(None, 5, "t") is None and review._pr_head_now("o/r", None, "t") is None
