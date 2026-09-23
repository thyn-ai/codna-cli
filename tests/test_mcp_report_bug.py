from __future__ import annotations

import asyncio
import json

import pytest

from codna import mcp_server, report_cli


@pytest.fixture(autouse=True)
def _authorized_device(monkeypatch):
    """The uniform execution gate requires the free `codna login`; run these tests authorized."""
    monkeypatch.setenv("CODNA_API_KEY", "test-device-login")


def _mcp_available() -> bool:
    try:
        import mcp.server.fastmcp  # noqa: F401

        return True
    except ImportError:
        return False


def test_report_bug_json_files_through_submit_report(monkeypatch):
    """The MCP tool must call the SAME submission path the CLI uses, not a parallel
    reimplementation -- that's the whole point of an agent and a human landing in one place."""
    captured = {}

    def _fake_submit(*, title, product, body, attach_diagnostics):
        captured.update(title=title, product=product, body=body,
                        attach_diagnostics=attach_diagnostics)
        return report_cli.ReportResult(ok=True, submitted=True,
                                       url="https://github.com/thyn-ai/feedback/issues/9")

    monkeypatch.setattr(report_cli, "submit_report", _fake_submit)

    payload = json.loads(mcp_server.report_bug_json(
        "a title", body="the body", product="telys", include_diagnostics=True,
    ))

    assert captured == {"title": "a title", "product": "telys", "body": "the body",
                        "attach_diagnostics": True}
    assert payload == {"submitted": True, "url": "https://github.com/thyn-ai/feedback/issues/9",
                       "local_path": None}


def test_report_bug_json_requires_a_title():
    assert "error" in mcp_server.report_bug_json("")


def test_report_bug_json_never_crashes_the_server_on_a_submission_error(monkeypatch):
    def _boom(**_kw):
        raise RuntimeError("network is on fire")

    monkeypatch.setattr(report_cli, "submit_report", _boom)
    result = mcp_server.report_bug_json("a title")

    assert "codna_report_bug error" in result
    assert "network is on fire" in result


def test_report_bug_json_surfaces_the_fallback_local_path(monkeypatch):
    monkeypatch.setattr(report_cli, "submit_report", lambda **_kw: report_cli.ReportResult(
        ok=True, submitted=False, url="https://github.com/x/new", local_path="/tmp/x.md"))
    payload = json.loads(mcp_server.report_bug_json("a title"))
    assert payload["submitted"] is False
    assert payload["local_path"] == "/tmp/x.md"


# ── mcp-gated: exercised only when the optional `mcp` extra is installed ───────────────────────────
@pytest.mark.skipif(not _mcp_available(), reason="mcp extra not installed")
def test_report_bug_tool_is_registered_with_write_intent():
    """codna_report_bug files a real GitHub issue -- it must be flagged as a write, unlike
    codna_triage/codna_secure/codna_recall, or an MCP client might treat it as side-effect-free
    and call it speculatively. Uses FastMCP's own public `list_tools()`, not private internals."""
    server = mcp_server._build_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert "codna_report_bug" in tools
    assert tools["codna_report_bug"].annotations.readOnlyHint is False


@pytest.mark.skipif(not _mcp_available(), reason="mcp extra not installed")
def test_capabilities_resource_mentions_report_bug():
    server = mcp_server._build_server()
    resources = {str(r.uri) for r in asyncio.run(server.list_resources())}
    assert "codna://capabilities" in resources
    content = asyncio.run(server.read_resource("codna://capabilities"))
    text = "".join(c.content for c in content if hasattr(c, "content"))
    assert "codna_report_bug" in text
