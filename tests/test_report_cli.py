"""`codna report` (`codna.report_cli`).

Three properties matter more than any individual test:

  * it NEVER RAISES from `cmd_report` — a broken submission must degrade to a printed URL, never
    a stack trace, because filing a report is the whole point and losing it would defeat that;
  * it NEVER SENDS A SECRET — diagnostics go through `doctor.build_report`/`format_public_output`,
    the same redaction `codna doctor` itself uses, never a raw env dump;
  * it NEVER BLOCKS a script or an agent — the interactive prompt fires only when stdin is a real
    TTY with nothing else provided.
"""
from __future__ import annotations

import argparse

import httpx
import pytest

import codna.report_cli as rc


# ── normalize_product ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "given, expected",
    [("codna", "codna"), ("ALGENTA", "algenta"), ("  telys  ", "telys"), ("", "not sure"),
     (None, "not sure"), ("wordpress", "not sure")],
)
def test_normalize_product(given, expected):
    assert rc.normalize_product(given) == expected


# ── build_report_body ────────────────────────────────────────────────────────────────────────────
def test_report_body_matches_the_issue_forms_field_structure():
    """Whichever door a report comes through, a human reading it in thyn-ai/feedback must see the
    same shape as the browser form — otherwise triage has to special-case the CLI path."""
    body = rc.build_report_body(body="it crashed", product="codna", attach_diagnostics=False)
    assert "### Which product?" in body and "\n\ncodna" in body
    assert "### Version" in body and f"codna {rc.__version__}" in body
    assert "### Platform" in body
    assert "### What happened?" in body and "it crashed" in body
    assert "Diagnostics" not in body


def test_empty_body_gets_a_placeholder_not_a_blank_section():
    body = rc.build_report_body(body="", product="codna", attach_diagnostics=False)
    assert "(no description given)" in body


def test_diagnostics_are_attached_only_when_asked(monkeypatch):
    monkeypatch.setattr(rc, "_redacted_diagnostics", lambda: '{"marker": "REDACTED_PAYLOAD"}')
    without = rc.build_report_body(body="x", product="codna", attach_diagnostics=False)
    withit = rc.build_report_body(body="x", product="codna", attach_diagnostics=True)
    assert "REDACTED_PAYLOAD" not in without
    assert "REDACTED_PAYLOAD" in withit


def test_diagnostics_use_the_same_redaction_doctor_uses(monkeypatch):
    """REGRESSION-shaped: this must call doctor.build_report + format_public_output, not read
    os.environ directly — that redaction is what keeps a secret out of a PUBLIC issue."""
    calls = []
    from codna import doctor

    monkeypatch.setattr(doctor, "build_report", lambda: {"fake": "report"})
    monkeypatch.setattr(doctor, "format_public_output",
                        lambda report, json_output: calls.append((report, json_output)) or "SAFE")
    out = rc._redacted_diagnostics()
    assert out == "SAFE"
    assert calls == [({"fake": "report"}, True)]


def test_diagnostics_failure_is_swallowed_not_fatal(monkeypatch):
    from codna import doctor

    def _boom():
        raise RuntimeError("state file corrupt")

    monkeypatch.setattr(doctor, "build_report", _boom)
    out = rc._redacted_diagnostics()
    assert "diagnostics_unavailable" in out
    assert "state file corrupt" in out


# ── prefill_url ───────────────────────────────────────────────────────────────────────────────────
def test_prefill_url_is_a_valid_zero_auth_issue_link():
    url = rc.prefill_url(title="a bug", body="the body")
    assert url.startswith(f"https://github.com/{rc.FEEDBACK_REPO}/issues/new?")
    assert "template=bug_report.yml" in url


def test_prefill_url_encodes_characters_that_would_otherwise_break_the_query_string():
    url = rc.prefill_url(title="50% broken & \"quoted\"", body="line one\nline two")
    assert " " not in url.split("?", 1)[1].replace("+", "")  # no literal raw spaces in the query
    assert "&" not in url.split("what-happened=")[1].split("&")[0].replace("%26", "")


# ── _github_token ─────────────────────────────────────────────────────────────────────────────────
def test_token_prefers_github_token_over_gh_token_when_both_are_set(monkeypatch):
    """REGRESSION-shaped: with only ONE of the two set, reordering the checked tuple wouldn't
    change the result at all -- this only actually exercises precedence when BOTH are present and
    DIFFER, which is the only case where "prefers" means anything."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok-github")
    monkeypatch.setenv("GH_TOKEN", "tok-gh")
    assert rc._github_token() == "tok-github"


def test_token_falls_back_to_github_token_when_it_is_the_only_one_set(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok-from-env")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert rc._github_token() == "tok-from-env"


def test_token_falls_back_to_gh_token_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "tok-from-gh-token")
    assert rc._github_token() == "tok-from-gh-token"


def test_token_falls_back_to_gh_cli_when_present(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None)

    class _Result:
        returncode = 0
        stdout = "tok-from-gh-cli\n"

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Result())
    assert rc._github_token() == "tok-from-gh-cli"


def test_token_is_none_without_env_or_gh_cli(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(rc.shutil, "which", lambda name: None)
    assert rc._github_token() is None


def test_token_is_none_when_gh_cli_is_not_logged_in(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/usr/bin/gh")

    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Result())
    assert rc._github_token() is None


def test_token_is_none_when_gh_binary_hangs_or_errors(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(rc.shutil, "which", lambda name: "/usr/bin/gh")

    def _explode(*a, **k):
        raise OSError("no such process")

    monkeypatch.setattr(rc.subprocess, "run", _explode)
    assert rc._github_token() is None


def test_local_fallback_dir_honours_codna_runtime_root(monkeypatch, tmp_path):
    """REGRESSION. A hardcoded ~/.codna here would be the one piece of codna's local state that
    can't be redirected the way the webhook queue and the local-stack state file already are —
    and in practice, exactly what made every test that forgot to patch this write into the real
    developer's home directory during manual verification."""
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "sandboxed"))
    assert rc._local_fallback_dir() == tmp_path / "sandboxed" / "reports"


def test_local_fallback_dir_defaults_to_home_when_unset(monkeypatch):
    monkeypatch.delenv("CODNA_RUNTIME_ROOT", raising=False)
    from pathlib import Path

    assert rc._local_fallback_dir() == Path.home() / ".codna" / "reports"


# ── submit_report: the never-fails contract ─────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_submit_report_files_a_real_issue_when_a_token_is_available(monkeypatch):
    monkeypatch.setattr(rc, "_github_token", lambda: "tok")
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["url"], captured["json"], captured["headers"] = url, json, headers
        return _Resp(201, {"html_url": "https://github.com/thyn-ai/feedback/issues/42"})

    monkeypatch.setattr(httpx, "post", _fake_post)
    result = rc.submit_report(title="t", product="codna", body="b", attach_diagnostics=False)

    assert result.submitted is True
    assert result.url == "https://github.com/thyn-ai/feedback/issues/42"
    assert captured["url"] == f"https://api.github.com/repos/{rc.FEEDBACK_REPO}/issues"
    assert captured["json"]["title"] == "t"
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_submit_report_never_sends_when_no_token_is_available(monkeypatch, tmp_path):
    """No token -> straight to the fallback. A network call here would mean the module tries to
    file an unauthenticated request, which GitHub would just reject anyway -- and might leak the
    report body to a request that goes nowhere useful."""
    monkeypatch.setattr(rc, "_github_token", lambda: None)
    monkeypatch.setattr(rc, "_local_fallback_dir", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append(1))

    result = rc.submit_report(title="t", product="codna", body="b", attach_diagnostics=False)

    assert calls == [], "submit_report must not call httpx.post without a token"
    assert result.submitted is False
    assert result.url.startswith("https://github.com/")


def test_submit_report_falls_back_on_a_network_error(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_github_token", lambda: "tok")
    monkeypatch.setattr(rc, "_local_fallback_dir", lambda: tmp_path)

    def _boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "post", _boom)
    result = rc.submit_report(title="t", product="codna", body="b", attach_diagnostics=False)

    assert result.ok is True
    assert result.submitted is False
    assert result.url.startswith("https://github.com/")


def test_submit_report_falls_back_on_a_non_201_response(monkeypatch, tmp_path):
    """A token that's expired/wrong-scoped returns 401/403/422, not an exception. Must still
    degrade gracefully rather than report success with a garbage URL."""
    monkeypatch.setattr(rc, "_github_token", lambda: "tok")
    monkeypatch.setattr(rc, "_local_fallback_dir", lambda: tmp_path)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(403, {"message": "Forbidden"}))

    result = rc.submit_report(title="t", product="codna", body="b", attach_diagnostics=False)
    assert result.submitted is False


def test_submit_report_writes_a_local_copy_on_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_github_token", lambda: None)
    monkeypatch.setattr(rc, "_local_fallback_dir", lambda: tmp_path)

    result = rc.submit_report(title="a real bug", product="codna", body="details here",
                              attach_diagnostics=False)

    assert result.local_path is not None
    saved = list(tmp_path.glob("*.md"))
    assert len(saved) == 1
    text = saved[0].read_text()
    assert "a real bug" in text and "details here" in text


def test_submit_report_survives_even_if_the_local_write_fails(monkeypatch):
    """The local-file fallback is a nice-to-have, not a requirement -- if the disk is read-only or
    full, the function must still return a usable URL rather than raise."""
    monkeypatch.setattr(rc, "_github_token", lambda: None)

    def _boom(**_kw):
        raise OSError("disk full")

    monkeypatch.setattr(rc, "_write_local_fallback", _boom)
    result = rc.submit_report(title="t", product="codna", body="b", attach_diagnostics=False)

    assert result.ok is True
    assert result.local_path is None
    assert result.url


# ── cmd_report: the CLI entry point ────────────────────────────────────────────────────────────
def _args(**kw):
    base = {"title": "a title", "product": "codna", "body": "", "attach_diagnostics": False,
            "dry_run": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_dry_run_never_calls_submit_report(monkeypatch, capsys):
    def _boom(**_kw):
        raise AssertionError("submit_report must not run under --dry-run")

    monkeypatch.setattr(rc, "submit_report", _boom)
    exit_code = rc.cmd_report(_args(dry_run=True, body="hello"))

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "hello" in out


def test_successful_submission_prints_the_filed_url(monkeypatch, capsys):
    monkeypatch.setattr(rc, "submit_report", lambda **_kw: rc.ReportResult(
        ok=True, submitted=True, url="https://github.com/thyn-ai/feedback/issues/7"))
    exit_code = rc.cmd_report(_args(body="hello"))

    assert exit_code == 0
    assert "issues/7" in capsys.readouterr().out


def test_fallback_submission_prints_the_url_and_local_path(monkeypatch, capsys):
    monkeypatch.setattr(rc, "submit_report", lambda **_kw: rc.ReportResult(
        ok=True, submitted=False, url="https://github.com/x/new", local_path="/tmp/x.md"))
    exit_code = rc.cmd_report(_args(body="hello"))

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "https://github.com/x/new" in out
    assert "/tmp/x.md" in out


def test_a_script_piping_empty_body_is_never_blocked_on_a_prompt(monkeypatch):
    """REGRESSION-shaped: stdin.isatty() must gate the prompt. A CI job or an agent invoking this
    command with no --body and a non-interactive stdin must not hang forever waiting for input."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def _prompt_should_not_run():
        raise AssertionError("_prompt_for_body must not run when stdin is not a TTY")

    monkeypatch.setattr(rc, "_prompt_for_body", _prompt_should_not_run)
    monkeypatch.setattr(rc, "submit_report", lambda **_kw: rc.ReportResult(
        ok=True, submitted=True, url="https://x"))

    exit_code = rc.cmd_report(_args(body=""))
    assert exit_code == 0


def test_an_interactive_terminal_with_no_body_is_prompted(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(rc, "_prompt_for_body", lambda: "typed at the prompt")
    captured = {}
    monkeypatch.setattr(rc, "submit_report",
                        lambda **kw: captured.update(kw) or rc.ReportResult(ok=True, submitted=True, url="u"))

    rc.cmd_report(_args(body=""))
    assert captured["body"] == "typed at the prompt"


# ── register(): the parser itself ────────────────────────────────────────────────────────────────
def test_register_wires_a_working_report_subcommand():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    rc.register(sub)

    args = parser.parse_args(["report", "something broke", "--product", "telys", "--attach-diagnostics"])
    assert args.func is rc.cmd_report
    assert args.title == "something broke"
    assert args.product == "telys"
    assert args.attach_diagnostics is True
    assert args.dry_run is False


def test_register_defaults_product_to_codna():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    rc.register(sub)
    args = parser.parse_args(["report", "x"])
    assert args.product == "codna"


def test_register_rejects_an_unknown_product():
    """argparse `choices=` should refuse a typo'd product rather than silently mapping it to
    "not sure" three steps later where the mistake is harder to notice."""
    parser = argparse.ArgumentParser(exit_on_error=False)
    sub = parser.add_subparsers(dest="cmd")
    rc.register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["report", "x", "--product", "wordpress"])
