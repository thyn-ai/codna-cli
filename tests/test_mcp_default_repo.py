from __future__ import annotations

from codna import mcp_server


def test_default_repo_env_fallback(monkeypatch):
    monkeypatch.setenv("CODNA_MCP_DEFAULT_REPO", "/work/x")
    assert mcp_server._default_repo(".") == "/work/x"


def test_default_repo_no_env(monkeypatch):
    monkeypatch.delenv("CODNA_MCP_DEFAULT_REPO", raising=False)
    assert mcp_server._default_repo(".") == "."


def test_explicit_repo_wins_over_env(monkeypatch):
    monkeypatch.setenv("CODNA_MCP_DEFAULT_REPO", "/work/x")
    assert mcp_server._default_repo("/explicit") == "/explicit"
