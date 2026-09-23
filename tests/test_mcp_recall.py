"""Tests for the codna_recall MCP tool body (recall_json) + server registration."""
from __future__ import annotations

import json
import sys

import pytest

from codna.mcp_server import recall_json   # import does NOT pull in `mcp` or `telys`


@pytest.fixture(autouse=True)
def _authorized_device(monkeypatch):
    """The uniform execution gate requires the free `codna login`; run these tests authorized."""
    monkeypatch.setenv("CODNA_API_KEY", "test-device-login")


# ── offline: the guard + friendly failure (no mcp, no telys needed) ───────────────────────────────
def test_recall_requires_query():
    assert recall_json(query="") == "codna_recall error: query is required"


def test_recall_without_telys_is_friendly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "telys", None)
    monkeypatch.setitem(sys.modules, "telys.embedding", None)
    out = recall_json(repo=str(tmp_path), query="anything")
    assert out.startswith("codna_recall error:") and "reinstall or upgrade `codna`" in out


def test_recall_auto_index_uses_all_languages_sentinel(monkeypatch, tmp_path):
    from codna import memory as memory_module

    captured: dict[str, object] = {}

    class FakeMemory:
        def __init__(self, repo: str, db_path=None, *, service=None) -> None:
            self.repo = repo
            self.db_path = db_path
            self.service = service
            captured["db_path"] = db_path

        def is_empty(self) -> bool:
            return True

        def index(self, *, languages):
            captured["languages"] = languages

        def recall(self, query: str, *, service=None, language=None, final_k=8):
            return {"symbols": [], "explain": {"plan": "mock"}, "candidate_count": 0}

    monkeypatch.setattr(memory_module, "CodeMemory", FakeMemory)

    out = recall_json(repo=str(tmp_path), query="anything")

    assert json.loads(out)["symbols"] == []
    assert captured["languages"] is None
    assert captured["db_path"] is not None


def test_recall_uses_external_mcp_memory_root(monkeypatch, tmp_path):
    from codna import memory as memory_module

    repo = tmp_path / "repo"
    repo.mkdir()
    memory_root = tmp_path / "memory-root"
    captured: dict[str, object] = {}

    class FakeMemory:
        def __init__(self, repo: str, db_path=None, *, service=None) -> None:
            captured["repo"] = repo
            captured["db_path"] = db_path

        def is_empty(self) -> bool:
            return False

        def recall(self, query: str, *, service=None, language=None, final_k=8):
            return {"symbols": [], "explain": {"plan": "mock"}, "candidate_count": 0}

    monkeypatch.setenv("CODNA_TELYS_MEMORY_ROOT", str(memory_root))
    monkeypatch.setattr(memory_module, "CodeMemory", FakeMemory)

    out = recall_json(repo=str(repo), query="anything")

    assert json.loads(out)["symbols"] == []
    db_path = captured["db_path"]
    assert isinstance(db_path, str)
    assert db_path.startswith(str(memory_root / "mcp"))
    assert db_path != str(repo / ".codna-memory")


# ── kernel-gated: the happy path returns real symbols + a plan ────────────────────────────────────
def _telys_ready():
    try:
        from codna.memory import _configure_telys_kernel_env

        _configure_telys_kernel_env()
        from telys.embedding import AlgentaBigramEmbedder
        AlgentaBigramEmbedder()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _telys_ready(), reason="Telys not installed or Codna could not resolve a local kernel")
def test_recall_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_TELYS_MEMORY_ROOT", str(tmp_path / "memory-root"))
    (tmp_path / "ci.py").write_text(
        'def open_pr_when_ci_fails(branch):\n'
        '    """Open a pull request when CI fails on a branch."""\n'
        '    return branch\n'
    )
    out = recall_json(repo=str(tmp_path), query="open a pull request when CI fails", final_k=3)
    data = json.loads(out)
    assert data["symbols"], "expected recalled symbols"
    assert "plan" in data["explain"]
    assert any("open_pr_when_ci_fails" in s["id"] for s in data["symbols"])


# ── mcp-gated: the FastMCP server builds with the new tool registered ─────────────────────────────
def _mcp_available():
    try:
        import mcp.server.fastmcp  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _mcp_available(), reason="mcp extra not installed")
def test_build_server_includes_recall():
    from codna.mcp_server import _build_server
    server = _build_server()          # must not raise; the @server.tool() decorator registered codna_recall
    assert server is not None
