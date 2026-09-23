from __future__ import annotations

import asyncio
import json

import pytest

# The "cli secure subsystem" CI job installs a minimal dep set without `mcp`; these tests build the
# FastMCP server, so skip cleanly there (they run in the full-dep jobs).
pytest.importorskip("mcp.server.fastmcp")

from codna.mcp_server import _build_server  # noqa: E402


def test_readonly_annotations():
    s = _build_server()
    tools = {t.name: t for t in asyncio.run(s.list_tools())}
    assert tools["codna_triage"].annotations.readOnlyHint is True
    assert tools["codna_secure"].annotations.readOnlyHint is True
    assert tools["codna_recall"].annotations.readOnlyHint is True
    assert tools["codna_fix"].annotations.readOnlyHint is False  # fix is the action, not read-only


def test_full_hint_annotations():
    """Every tool carries accurate destructive/idempotent/open-world hints (TDQS)."""
    s = _build_server()
    tools = {t.name: t for t in asyncio.run(s.list_tools())}

    # Read-only + deterministic + side-effect-free across repeats.
    for name in ("codna_triage", "codna_secure", "codna_recall"):
        assert tools[name].annotations.idempotentHint is True, name
    # Fully local tools never touch the network (secure: SARIF-only, no sidecar; recall: on-device).
    assert tools["codna_secure"].annotations.openWorldHint is False
    assert tools["codna_recall"].annotations.openWorldHint is False

    # fix: plan-only by default; open_pr=true pushes a branch and opens a real PR — additive.
    assert tools["codna_fix"].annotations.destructiveHint is False
    assert tools["codna_fix"].annotations.openWorldHint is True

    # report_bug files a real GitHub issue — a create, never a destructive update.
    assert tools["codna_report_bug"].annotations.readOnlyHint is False
    assert tools["codna_report_bug"].annotations.destructiveHint is False
    assert tools["codna_report_bug"].annotations.openWorldHint is True


def test_resource_and_prompt_registered():
    s = _build_server()
    assert "codna://capabilities" in [str(r.uri) for r in asyncio.run(s.list_resources())]
    assert "fix_bug" in [p.name for p in asyncio.run(s.list_prompts())]


def test_codna_fix_runs_sync_client_path_outside_mcp_event_loop(monkeypatch):
    """Local fix planning uses sync repository-client code that owns its own asyncio runner."""
    from codna import mcp_server

    async def _inner_plan() -> dict[str, object]:
        return {"root_cause": "patched through worker thread"}

    def fake_fix_json(repo=".", issue="", ref="", open_pr=False, model="repository.verified_agentic_v1"):
        del repo, issue, ref, open_pr
        payload = asyncio.run(_inner_plan())
        payload["model"] = model
        return json.dumps(payload)

    monkeypatch.setattr(mcp_server, "fix_json", fake_fix_json)

    async def call_tool():
        server = mcp_server._build_server()
        content, structured = await server.call_tool("codna_fix", {
            "repo": ".",
            "issue": "x",
            "model": "openai-native/gpt-5.5",
        })
        return json.loads(content[0].text), json.loads(structured["result"])

    assert asyncio.run(call_tool()) == (
        {"root_cause": "patched through worker thread", "model": "openai-native/gpt-5.5"},
        {"root_cause": "patched through worker thread", "model": "openai-native/gpt-5.5"},
    )


def test_fix_json_plan_routes_requested_model(monkeypatch):
    from codna import fix_run, mcp_server

    captured = {}

    monkeypatch.setattr(mcp_server, "_client", lambda include_keychain=True: object())

    def fake_plan_once(client, **kwargs):
        del client
        captured.update(kwargs)
        return "rid", {"snapshot_id": "snap"}, {
            "decision_plan": {
                "repository_analysis": {
                    "root_cause": "rc",
                    "impacted_symbols": ["add"],
                    "blast_radius": "localized",
                    "generated_patch_ref": "patch://1",
                },
                "confidence": 0.78,
            },
            "runtime_model": "gpt-5.5",
            "planner_usage": {"cost_usd": 0.01},
        }

    monkeypatch.setattr(fix_run, "_plan_once", fake_plan_once)

    payload = json.loads(mcp_server.fix_json(
        repo="/tmp/repo",
        issue="x",
        model="openai-native/gpt-5.5",
    ))

    assert captured["model"] == "openai-native/gpt-5.5"
    assert captured["open_pr"] is False
    assert payload["model"] == "gpt-5.5"
    assert payload["patch_ref"] == "patch://1"
