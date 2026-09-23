"""Uniform execution gate: every MCP tool call requires the free community login.

Owner rule: all codna MCP tools execute under one licensing model — introspection
(initialize/tools/list) stays credential-free, but no tool executes without the one-time free
`codna login` device authorization. The gate reuses codna's existing key layer (env → OS
keychain via the non-secret name index → dev keys.txt); it never touches the network, matching
the offline-after-first-login model and sqai's `login_required` / algenta-mcp's auth gate.

CI fixture: the scrubbed state clears CODNA_API_KEY/ALGENTA_API_KEY/DE_API_KEY, disables the
keychain (codna's own CODNA_DISABLE_KEYCHAIN kill-switch) and removes the dev keys.txt path;
the authorized state is a plain CODNA_API_KEY env var — the same artifact `codna login` stores.
No network login is ever performed in tests.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from codna import cli, mcp_server

LOGIN_SUFFIX = mcp_server.LOGIN_REQUIRED_MESSAGE


def _mcp_available() -> bool:
    try:
        import mcp.server.fastmcp  # noqa: F401
        return True
    except ImportError:
        return False


mcp_required = pytest.mark.skipif(not _mcp_available(), reason="mcp extra not installed")


@pytest.fixture
def scrubbed_device(monkeypatch):
    """A zero-credential device: no codna/algenta key env, no keychain, no dev keys.txt."""
    for name in ("CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CODNA_DISABLE_KEYCHAIN", "1")
    monkeypatch.setattr(cli, "_keys_file_path", lambda: None)


@pytest.fixture
def authorized_device(scrubbed_device, monkeypatch):
    """The post-`codna login` state: the account key present in the environment."""
    monkeypatch.setenv("CODNA_API_KEY", "ci-device-login-fixture")


# ── tool bodies (no mcp dependency — these run in the minimal-dep CI job too) ─────────────────────
def test_module_level_tool_bodies_all_gate(scrubbed_device):
    """fix/secure/recall/report_bug return the structured error — never raise, never exit."""
    assert mcp_server.fix_json(repo=".", issue="x") == f"codna_fix error: {LOGIN_SUFFIX}"
    assert mcp_server.secure_json(repo=".", sarif_path="x.sarif") == f"codna_secure error: {LOGIN_SUFFIX}"
    assert mcp_server.recall_json(repo=".", query="x") == f"codna_recall error: {LOGIN_SUFFIX}"
    assert mcp_server.report_bug_json(title="x") == f"codna_report_bug error: {LOGIN_SUFFIX}"


def test_gate_never_raises_and_needs_no_network(scrubbed_device):
    """The check is a local key-layer read: no engine, no sidecar, no keychain prompt."""
    assert mcp_server._login_required_error("codna_triage") == (  # noqa: SLF001
        f"codna_triage error: {LOGIN_SUFFIX}"
    )


def test_codna_api_key_opens_the_gate(authorized_device):
    assert mcp_server._login_required_error("codna_triage") is None  # noqa: SLF001


def test_secure_executes_normally_when_authorized(authorized_device, tmp_path):
    sarif = tmp_path / "result.sarif"
    sarif.write_text(json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "js/eval"}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40,
                                          "repositoryUri": "https://example.test/o/r"}],
            "results": [{
                "ruleId": "js/eval",
                "message": {"text": "eval"},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.js"},
                                                    "region": {"startLine": 1}}}],
            }],
        }],
    }), encoding="utf-8")
    payload = json.loads(mcp_server.secure_json(str(tmp_path), str(sarif)))
    assert payload["counts"] == {"production-reachable": 1}


def test_recall_gets_past_the_gate_to_its_own_runtime_check(authorized_device, tmp_path):
    """With the login fixture, recall runs (or fails on its own runtime requirement) — never the gate."""
    out = mcp_server.recall_json(repo=str(tmp_path), query="anything")
    assert out != f"codna_recall error: {LOGIN_SUFFIX}"


def test_report_bug_gets_past_the_gate_when_authorized(authorized_device, monkeypatch):
    from codna import report_cli

    monkeypatch.setattr(report_cli, "submit_report", lambda **_kw: report_cli.ReportResult(
        ok=True, submitted=False, url="https://github.com/x/new", local_path=None))
    payload = json.loads(mcp_server.report_bug_json("a title"))
    assert payload["submitted"] is False


# ── full server path (needs the mcp extra): introspection free, all 5 executions gated ────────────
def _result_text(content) -> str:
    return content[0].text if isinstance(content, (list, tuple)) else content


@mcp_required
def test_introspection_stays_credential_free(scrubbed_device):
    server = mcp_server._build_server()  # noqa: SLF001
    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == [
        "codna_fix", "codna_recall", "codna_report_bug", "codna_secure", "codna_triage",
    ]


@mcp_required
def test_all_five_tools_gate_end_to_end(scrubbed_device):
    server = mcp_server._build_server()  # noqa: SLF001
    calls = {
        "codna_triage": {"repo": ".", "issue": "x"},
        "codna_fix": {"repo": ".", "issue": "x"},
        "codna_secure": {"repo": ".", "sarif_path": "x.sarif"},
        "codna_recall": {"repo": ".", "query": "x"},
        "codna_report_bug": {"title": "x"},
    }
    for name, arguments in calls.items():
        content, _structured = asyncio.run(server.call_tool(name, arguments))
        assert _result_text(content) == f"{name} error: {LOGIN_SUFFIX}", name


@mcp_required
def test_triage_executes_when_authorized(authorized_device, monkeypatch, tmp_path):
    class FakeEngine:
        def triage_repository(self, rid, payload):
            return {
                "suspect_files": ["a.py"],
                "reduction_ratio": 10.0,
                "raw_repo_token_estimate": 100,
                "evidence_bundle_token_count": 10,
            }

    monkeypatch.setattr(mcp_server, "_client", lambda: FakeEngine())
    monkeypatch.setattr(mcp_server, "_register", lambda c, repo, ref: ("rid", {"snapshot_id": "s"}))
    monkeypatch.setattr(mcp_server, "_dump", lambda payload: payload)
    server = mcp_server._build_server()  # noqa: SLF001
    content, _structured = asyncio.run(server.call_tool("codna_triage", {"repo": str(tmp_path)}))
    payload = json.loads(_result_text(content))
    assert payload["suspect_files"] == ["a.py"]


@mcp_required
def test_tool_descriptions_disclose_the_login_requirement(scrubbed_device):
    server = mcp_server._build_server()  # noqa: SLF001
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    for name in ("codna_triage", "codna_secure", "codna_recall", "codna_fix", "codna_report_bug"):
        assert "codna login" in tools[name].description, name
