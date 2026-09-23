from __future__ import annotations

from types import SimpleNamespace

from codna import cli, fix_run, project_rules, review


# ---- project_rules ----------------------------------------------------------------------------

def test_read_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Test cmd: pytest -q\nDo not touch vendored code.\n", encoding="utf-8")
    g = project_rules.read_project_guidance(str(tmp_path))
    assert g and "pytest -q" in g and g.startswith("# AGENTS.md")


def test_read_none_when_absent(tmp_path):
    assert project_rules.read_project_guidance(str(tmp_path)) is None


def test_concatenates_multiple_sources(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents rules", encoding="utf-8")
    (tmp_path / ".cursorrules").write_text("cursor rules", encoding="utf-8")
    g = project_rules.read_project_guidance(str(tmp_path))
    assert "agents rules" in g and "cursor rules" in g


def test_guidance_is_bounded(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 50_000, encoding="utf-8")
    assert len(project_rules.read_project_guidance(str(tmp_path))) <= project_rules._MAX_BYTES + 32


# ---- fix signal carries guidance --------------------------------------------------------------

def test_fix_plan_signal_includes_project_guidance(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("run: pytest -q", encoding="utf-8")
    captured = {}

    class _C:
        def triage_repository(self, rid, body):
            captured["triage_sig"] = body["signals"]
            return {"workspace_evidence_bundle_ref": "b"}

        def create_repository_decision_plan(self, rid, body):
            captured["plan_sig"] = body["signals"]
            return {"decision_plan": {"repository_analysis": {"root_cause": "x"}}}

    monkeypatch.setattr(cli, "_register", lambda c, repo, ref, github_token=None, focus_paths=None: ("rid", {"snapshot_id": "s"}))
    monkeypatch.setattr(cli, "_dump", lambda v: v)
    monkeypatch.setattr(cli, "_issue_focus_paths", lambda local, issue: [])
    rid, snap, plan = fix_run._plan_once(_C(), repo=str(tmp_path), ref=None, issue="fix", failing=[],
                                         model="m", open_pr=False, gh_token=None)
    assert captured["triage_sig"]["project_guidance"] == "# AGENTS.md\nrun: pytest -q"
    assert captured["plan_sig"]["project_guidance"].startswith("# AGENTS.md")


# ---- codna review -----------------------------------------------------------------------------

def test_review_no_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "changed_files", lambda repo, base="HEAD": [])
    monkeypatch.setattr(cli, "_client", lambda **_kwargs: object())
    args = SimpleNamespace(repo=str(tmp_path), base=None, issue=None, ref=None, as_json=True, triage=True)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_review(args)
    assert rc == 0
    import json
    assert json.loads(buf.getvalue())["changed_files"] == []


def test_review_triages_changed_files(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("no vendored edits", encoding="utf-8")
    monkeypatch.setattr(review, "changed_files", lambda repo, base="HEAD": ["a.py", "b.py"])
    captured = {}

    class _C:
        def triage_repository(self, rid, body):
            captured["sig"] = body["signals"]
            return {"suspect_files": ["a.py"], "suspect_symbols": ["a.f"], "reduction_ratio": 4.0}

    monkeypatch.setattr(cli, "_client", lambda **_kwargs: _C())
    monkeypatch.setattr(cli, "_register", lambda c, repo, ref, focus_paths=None: ("rid", {"snapshot_id": "s"}))
    monkeypatch.setattr(cli, "_dump", lambda v: v)
    args = SimpleNamespace(repo=str(tmp_path), base=None, issue=None, ref=None, as_json=True, triage=True)
    import io
    import contextlib
    import json
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_review(args)
    out = json.loads(buf.getvalue())
    assert rc == 0
    assert out["changed_files"] == ["a.py", "b.py"]
    assert out["suspect_files"] == ["a.py"]
    assert captured["sig"]["changed_files"] == ["a.py", "b.py"]
    assert captured["sig"]["project_guidance"].startswith("# AGENTS.md")  # review honors rules too


def test_review_parser():
    a = cli.build_parser().parse_args(["review", ".", "--base", "main", "--json"])
    assert a.base == "main" and a.as_json is True and a.func is cli.cmd_review


def test_review_parser_findings_flags():
    a = cli.build_parser().parse_args(
        ["review", ".", "--diff", "origin/main...HEAD", "--pr", "o/r#3", "--post",
         "--min-confidence", "0.8", "--max-findings", "4", "--blocking", "--json"]
    )
    assert a.diff == "origin/main...HEAD" and a.pr == "o/r#3" and a.post is True
    assert a.min_confidence == 0.8 and a.max_findings == 4 and a.blocking is True and a.triage is False


def test_cmd_review_findings_default_dispatch(monkeypatch):
    # Default (no --triage) → findings path; renders + exits 0 on a non-failing conclusion.
    monkeypatch.setattr(review, "run_findings_review",
                        lambda args: {"conclusion": "neutral", "base": "main...HEAD",
                                      "head_sha": "abcdef1234", "changed_files": ["a.py"],
                                      "findings": [{"path": "a.py", "line": 5, "severity": "high",
                                                    "category": "correctness", "title": "bug",
                                                    "explanation": "boom", "diff_anchor": {}}]})
    args = SimpleNamespace(triage=False, as_json=False)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_review(args)
    assert rc == 0
    assert "1 finding(s)" in buf.getvalue() and "a.py:5" in buf.getvalue()


def test_cmd_review_findings_blocking_exit_code(monkeypatch):
    monkeypatch.setattr(review, "run_findings_review",
                        lambda args: {"conclusion": "failure", "findings": [], "base": "b",
                                      "changed_files": []})
    args = SimpleNamespace(triage=False, as_json=True)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_review(args)
    assert rc == 1  # blocking failure → non-zero exit


def test_render_findings_clean_and_note():
    assert review.render_findings({"note": "no changed files vs HEAD", "findings": []}) == [
        "codna: no changed files vs HEAD"
    ]
    clean = review.render_findings({"base": "main", "changed_files": ["a.py"], "findings": []})
    assert any("no high-confidence issues" in ln for ln in clean)


def test_reads_cursor_bugbot_rules(tmp_path):
    # Drop-in Bugbot compatibility: a repo configured with .cursor/BUGBOT.md is honored by Codna.
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "BUGBOT.md").write_text("Never allow raw SQL string interpolation.", encoding="utf-8")
    g = project_rules.read_project_guidance(str(tmp_path))
    assert g and "raw SQL string interpolation" in g


def test_cmd_review_json_surfaces_the_check_anchor_and_the_pr_head(monkeypatch):
    """`codna review --json` prints post_review's status verbatim, so automation can compare the
    check's commit, the reviewed head and the pull request's head (codna#569)."""
    import contextlib
    import io
    import json

    monkeypatch.setattr(review, "run_findings_review",
                        lambda args: {"conclusion": "success", "findings": [], "base": "b", "changed_files": [],
                                      "head_sha": "5a3ba5ff" + "0" * 32,
                                      "posted": {"event": "COMMENT", "check": {"id": 9001, "updated_existing": True},
                                                 "check_head_sha": "5a3ba5ff" + "0" * 32,
                                                 "pr_head_sha": "7679ecbe" + "0" * 32}})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.cmd_review(SimpleNamespace(triage=False, as_json=True))
    assert rc == 0
    printed = json.loads(buf.getvalue())
    assert printed["posted"]["check_head_sha"] == "5a3ba5ff" + "0" * 32
    assert printed["posted"]["pr_head_sha"] == "7679ecbe" + "0" * 32
    assert printed["head_sha"] == printed["posted"]["check_head_sha"] != printed["posted"]["pr_head_sha"]
