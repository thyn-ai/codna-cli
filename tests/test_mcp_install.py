from __future__ import annotations

import json

import pytest

from codna import mcp_install
from codna.mcp_install import McpInstallError


def test_config_path_cursor_global_and_project(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    assert mcp_install.config_path("cursor", project=False).name == "mcp.json"
    assert "home" in str(mcp_install.config_path("cursor", project=False))
    assert mcp_install.config_path("cursor", project=True) == tmp_path / ".cursor" / "mcp.json"


def test_unknown_client_and_claude_project_error():
    with pytest.raises(McpInstallError):
        mcp_install.config_path("vim", project=False)
    with pytest.raises(McpInstallError):
        mcp_install.config_path("claude", project=True)


def test_server_entry_with_and_without_repo():
    assert mcp_install.server_entry(None) == {"command": "codna", "args": ["mcp"]}
    assert mcp_install.server_entry("/r") == {"command": "codna", "args": ["mcp", "start", "--repo", "/r"]}


def test_install_merges_preserving_other_servers(tmp_path, monkeypatch):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "unrelated": 1}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = mcp_install.install("cursor", project=True)
    assert summary["replaced"] is False
    data = json.loads(cfg.read_text())
    assert data["mcpServers"]["other"] == {"command": "x"}   # preserved
    assert data["mcpServers"]["codna"]["command"] == "codna"  # added
    assert data["unrelated"] == 1                             # untouched
    # re-install reports replaced
    assert mcp_install.install("cursor", project=True)["replaced"] is True


def test_install_rejects_malformed_existing_config(tmp_path, monkeypatch):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(McpInstallError):
        mcp_install.install("cursor", project=True)
