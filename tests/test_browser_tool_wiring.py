"""The browser server reaches every path Codna carries MCP servers on — and only when switched on.

Three carriers exist and each is a place this feature can be "built but never joined":

* ``codna.durable.mcp.session_mcp_servers`` — the durable run / goal-mode path;
* ``codna.mcp_bridge.handle`` — the IDE chat path (``.codna/mcp.json`` and its neighbours);
* ``codna.mcp_server._build_server`` — Codna as an MCP server, so an external client gets the tools.

Each is asserted for the SAME two facts — off means nothing was added, on means exactly one validated
``browser`` server was — because the defect this file exists to prevent is one carrier working and another
quietly not. The tool bodies behind each carrier are one implementation (``browser_bridge``), asserted in
``test_browser_tool.py``; what differs per carrier is only how the entry is delivered.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from codna import browser_tool, mcp_bridge
from codna.browser_tool import BROWSER_TOOL_ENV, DEFAULT_CONNECT_TIMEOUT_S, PLAYWRIGHT_MCP_PACKAGE
from codna.durable.mcp import McpConfigError, session_mcp_servers

ON = {BROWSER_TOOL_ENV: "1", "CODNA_RUNTIME_ROOT": "/tmp/codna-browser-wiring"}
OFF = {"CODNA_RUNTIME_ROOT": "/tmp/codna-browser-wiring"}

requires_mcp = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason="mcp extra not installed (the offline lane); the stub-server tests cover registration",
)


# ── carrier 1: the durable path ────────────────────────────────────────────────────────────────


class _Context:
    def __init__(self, mcp_servers: tuple = ()) -> None:
        self.mcp_servers = mcp_servers


def test_durable_off_adds_nothing(monkeypatch) -> None:
    """The pre-feature request stays byte-identical: no servers in, no servers out."""
    monkeypatch.delenv(BROWSER_TOOL_ENV, raising=False)
    assert session_mcp_servers(_Context(), OFF) == []


def test_durable_on_adds_exactly_one_validated_browser_server(monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    servers = session_mcp_servers(_Context(), ON)
    assert len(servers) == 1
    assert servers[0]["name"] == "browser"
    assert servers[0]["transport"]["command"] == "npx"
    assert PLAYWRIGHT_MCP_PACKAGE in servers[0]["transport"]["args"]
    # Already expanded: no `{"from_env": …}` reference is left in a request body.
    assert "env" not in servers[0]["transport"]
    # The connect timeout SURVIVES expansion into the request. Without it the engine's 1.5 s probe loses a
    # ~1.3 s cold start, drops the server, and the session runs with no browser tool at all.
    assert servers[0]["timeout"] == DEFAULT_CONNECT_TIMEOUT_S


def test_durable_on_keeps_the_runs_own_servers_and_their_resolved_credentials(monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    context = _Context((
        {"name": "linear",
         "transport": {"type": "stdio", "command": "npx", "args": [],
                       "env": {"K": {"from_env": "LINEAR_KEY"}}}},
    ))
    servers = session_mcp_servers(context, {**ON, "LINEAR_KEY": "resolved-value"})
    assert [server["name"] for server in servers] == ["linear", "browser"]
    # The run's own credential still resolved exactly the way the durable contract requires…
    assert servers[0]["transport"]["env"] == {"K": "resolved-value"}
    # …and the browser entry carries none.
    assert "env" not in servers[1]["transport"]


def test_durable_on_never_shadows_a_run_that_configured_its_own_browser(monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    context = _Context((
        {"name": "browser", "transport": {"type": "stdio", "command": "my-browser", "args": []}},
    ))
    servers = session_mcp_servers(context, ON)
    assert [(server["name"], server["transport"]["command"]) for server in servers] == [
        ("browser", "my-browser")
    ]


def test_durable_on_refuses_a_missing_credential_the_way_it_always_did(monkeypatch) -> None:
    """Enabling the browser must not weaken the credential rule: an unset reference still refuses."""
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    context = _Context((
        {"name": "linear",
         "transport": {"type": "stdio", "command": "npx", "args": [],
                       "env": {"K": {"from_env": "NOT_SET_ANYWHERE"}}}},
    ))
    with pytest.raises(McpConfigError) as caught:
        session_mcp_servers(context, ON)
    assert caught.value.code == "mcp_env_missing"
    assert caught.value.details["variable"] == "NOT_SET_ANYWHERE"


def test_durable_refuses_a_bad_switch_value_rather_than_dropping_the_server(monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "maybe")
    with pytest.raises(browser_tool.BrowserToolConfigError) as caught:
        session_mcp_servers(_Context(), {**ON, BROWSER_TOOL_ENV: "maybe"})
    assert caught.value.code == "browser_config_invalid"


# ── carrier 2: the IDE chat bridge ────────────────────────────────────────────────────────────


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_bridge_off_adds_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(BROWSER_TOOL_ENV, raising=False)
    path = _write(tmp_path / ".codna" / "mcp.json",
                  {"mcpServers": {"docs": {"transport": {"type": "stdio", "command": "true"}}}})
    response = mcp_bridge.handle({"sources": [{"path": str(path), "kind": "codna"}]})
    assert response["ok"] is True
    assert [server["name"] for server in response["servers"]] == ["docs"]


def test_bridge_on_adds_the_browser_server_alongside_the_users(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    path = _write(tmp_path / ".codna" / "mcp.json",
                  {"mcpServers": {"docs": {"transport": {"type": "stdio", "command": "true"}}}})
    response = mcp_bridge.handle({"sources": [{"path": str(path), "kind": "codna"}]})
    assert response["ok"] is True
    assert [server["name"] for server in response["servers"]] == ["docs", "browser"]
    assert response["servers"][1]["transport"]["type"] == "stdio"


def test_bridge_on_with_NO_config_file_still_carries_the_browser(tmp_path: Path, monkeypatch) -> None:
    """The chat path most repositories are on: no mcp.json at all. The browser must still arrive."""
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    response = mcp_bridge.handle(
        {"sources": [{"path": str(tmp_path / ".codna" / "mcp.json"), "kind": "codna"}]})
    assert response["ok"] is True
    assert [server["name"] for server in response["servers"]] == ["browser"]
    assert response["sources"][0]["found"] is False


def test_bridge_on_with_a_bad_switch_value_REFUSES_instead_of_dropping_it(
    tmp_path: Path, monkeypatch,
) -> None:
    """A typo the user believes enabled a browser is an error, not a silent "off"."""
    monkeypatch.setenv(BROWSER_TOOL_ENV, "maybe")
    response = mcp_bridge.handle({"sources": [{"path": str(tmp_path / "none.json"), "kind": "codna"}]})
    assert response["ok"] is False
    assert response["error"]["code"] == "browser_config_invalid"
    assert response["error"]["details"]["field"] == BROWSER_TOOL_ENV


def test_bridge_on_refuses_to_exceed_the_cap(tmp_path: Path, monkeypatch) -> None:
    """The one validator still decides: the browser entry is not a way past the 16-server cap."""
    from codna.durable.mcp import MAX_MCP_SERVERS

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    servers = {
        f"s{i}": {"transport": {"type": "stdio", "command": "true"}}
        for i in range(MAX_MCP_SERVERS)
    }
    path = _write(tmp_path / ".codna" / "mcp.json", {"mcpServers": servers})
    response = mcp_bridge.handle({"sources": [{"path": str(path), "kind": "codna"}]})
    assert response["ok"] is False
    assert response["error"]["code"] == "mcp_config_invalid"
    assert str(MAX_MCP_SERVERS) in response["error"]["message"]


def test_bridge_keeps_the_users_own_browser_server(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    path = _write(tmp_path / ".codna" / "mcp.json",
                  {"mcpServers": {"browser": {"command": "my-browser"}}})
    response = mcp_bridge.handle({"sources": [{"path": str(path), "kind": "codna"}]})
    assert response["ok"] is True
    assert [(server["name"], server["transport"]["command"]) for server in response["servers"]] == [
        ("browser", "my-browser")
    ]


# ── carrier 3: Codna as an MCP server ─────────────────────────────────────────────────────────


@requires_mcp
@pytest.mark.parametrize("switch", ["1", "0"])
def test_the_registered_tool_set_follows_the_switch(switch: str, monkeypatch) -> None:
    """`tools/list` must agree with the switch: a listed tool the server would refuse is a broken
    contract, and an unlisted one is invisible to the agent."""
    import asyncio

    from codna import mcp_server

    monkeypatch.setenv(BROWSER_TOOL_ENV, switch)
    names = [tool.name for tool in asyncio.run(mcp_server._build_server().list_tools())]  # noqa: SLF001
    browser_tools = [name for name in names if name.startswith("codna_browser_")]
    if switch == "1":
        assert browser_tools == list(browser_tool.BROWSER_TOOL_NAMES)
    else:
        assert browser_tools == []


@requires_mcp
def test_the_core_tools_are_still_all_there_before_the_browser_ones(monkeypatch) -> None:
    """The core tool set is the server's contract; a new tool is APPENDED, never interleaved."""
    import asyncio

    from codna import mcp_server

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    names = [tool.name for tool in asyncio.run(mcp_server._build_server().list_tools())]  # noqa: SLF001
    core = [
        "codna_triage", "codna_fix", "codna_secure", "codna_recall",
        "codna_report_bug", "codna_request_rule", "codna_load_skill",
    ]
    assert names[:len(core)] == core
    assert names[len(core):] == list(browser_tool.BROWSER_TOOL_NAMES)


@requires_mcp
def test_the_capabilities_resource_reports_the_browser_state(monkeypatch) -> None:
    import asyncio

    from codna import mcp_server

    monkeypatch.setenv(BROWSER_TOOL_ENV, "0")
    resource = asyncio.run(  # noqa: SLF001
        mcp_server._build_server().read_resource("codna://capabilities"))
    payload = json.loads(resource[0].content)
    assert payload["browser"]["enabled"] is False
    assert payload["browser"]["enable_with"] == f"{BROWSER_TOOL_ENV}=1"
    assert "codna_browser_navigate" not in payload["tools"]


@requires_mcp
def test_the_capabilities_resource_lists_the_browser_tools_when_on(monkeypatch) -> None:
    import asyncio

    from codna import mcp_server

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    resource = asyncio.run(  # noqa: SLF001
        mcp_server._build_server().read_resource("codna://capabilities"))
    payload = json.loads(resource[0].content)
    assert payload["browser"]["enabled"] is True
    for name in browser_tool.BROWSER_TOOL_NAMES:
        assert name in payload["tools"]


def test_durable_browser_entry_goes_through_expansion_like_every_declared_server(monkeypatch) -> None:
    """One shape, not a mix: the browser entry must not be the one entry that skipped the expander.

    It carries no `env` today so this is a no-op, but the returned list is what goes into a create-session
    body — and that body expects resolved values, never a `{"from_env": …}` reference.
    """
    import codna.browser_tool as browser_tool_module
    from codna.durable.mcp import expand_mcp_servers

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    # Give the browser entry a credential reference it must resolve, which is what a future transport
    # change would look like. Without expansion it would reach the request body unresolved.
    original = browser_tool_module.browser_mcp_servers

    def with_reference(env=None):
        servers = original(env)
        if not servers:
            return servers
        server = dict(servers[0])
        transport = dict(server["transport"])
        transport["env"] = {"BROWSER_KEY": {"from_env": "A_SET_VARIABLE"}}
        server["transport"] = transport
        return (server,)

    monkeypatch.setattr("codna.browser_tool.browser_mcp_servers", with_reference)
    env = {**ON, "A_SET_VARIABLE": "resolved"}
    servers = session_mcp_servers(_Context(), env)
    assert servers[0]["transport"]["env"] == {"BROWSER_KEY": "resolved"}
    # And the shape is exactly what the expander produces for any other server.
    assert servers == expand_mcp_servers(with_reference(env), env)


def test_a_fault_in_the_BUILTIN_browser_entry_is_a_BrowserToolError(monkeypatch) -> None:
    """Finding 3: the browser entry's own fault must not surface as the RUN's config error.

    A caller that catches ``BrowserToolError`` — every browser body does — must not be handed a
    ``McpConfigError`` because the built-in entry's credential reference was unset. The variable is still
    named, and the code is still ``mcp_env_missing``; only the family a caller switches on is the browser
    tool's, which is the one that produced the fault.
    """
    import codna.browser_tool as browser_tool_module
    from codna.browser_bridge import BrowserToolError

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    original = browser_tool_module.browser_mcp_servers

    def with_unset_reference(env=None):
        servers = original(env)
        if not servers:
            return servers
        server = dict(servers[0])
        transport = dict(server["transport"])
        transport["env"] = {"BROWSER_KEY": {"from_env": "NOT_SET_FOR_BROWSER"}}
        server["transport"] = transport
        return (server,)

    monkeypatch.setattr("codna.browser_tool.browser_mcp_servers", with_unset_reference)
    with pytest.raises(BrowserToolError) as caught:
        session_mcp_servers(_Context(), ON)
    # NOT McpConfigError: the built-in entry's fault stays in the browser tool's family…
    assert not isinstance(caught.value, McpConfigError)
    # …while the diagnosis survives intact, so this is a correction of type, not a loss of message.
    assert caught.value.code == "mcp_env_missing"
    assert caught.value.details["variable"] == "NOT_SET_FOR_BROWSER"
    assert caught.value.details["server"] == "browser"


# ── the capability note reports the EFFECTIVE server, not the declared list ────────────────────
#
# Measured twice on installed builds: the ledger held `session.mcp_attached: {servers: ["browser"],
# tools: 25}` and dozens of real `browser__browser_*` calls, while the run's ONE human-readable note said
# "MCP is not attached — no MCP servers are configured for this run". The note was built from the spec's
# DECLARED `mcp_servers`, and the built-in browser server is appended later, by `session_mcp_servers`.
#
# The note is imported from `capability_notes`, NOT `capability_report`, and that is deliberate: the
# latter schedules activities and imports `temporalio`, so importing it here would make a pure string
# builder require the OPTIONAL `durable` extra. This file runs on the offline lane (no extra), which is
# exactly the branch that regressed when the note lived in `capability_report`. The re-export keeps
# `capability_report.capability_notes` working for callers that already import it.


def _run_spec(tmp_path: Path, mcp_servers: tuple = ()):
    from codna.durable.contracts import RunSpec, TaskSpec

    task = TaskSpec(
        run_id="run-note",
        task_id="task-1",
        task_number=1,
        brief="Check the page",
        repo_dir=str(tmp_path),
        verify_commands=("true",),
        engine="plain",
        mcp_servers=mcp_servers,
    )
    return RunSpec(run_id="run-note", repo_dir=str(tmp_path), base_branch="main", tasks=(task,))


def test_the_capability_note_NAMES_the_builtin_browser_server_when_it_is_on(tmp_path: Path) -> None:
    """The sentence a human reads to answer "did the browser run?" must say yes when it did."""
    from codna.durable.capability_notes import capability_notes

    notes = capability_notes(_run_spec(tmp_path), ON)
    mcp = next(message for kind, message in notes if "MCP is" in message)
    assert "MCP is attached" in mcp
    assert "browser" in mcp
    assert "not attached" not in mcp


def test_the_capability_note_still_reports_a_genuine_absence_when_the_tool_is_off(
    tmp_path: Path,
) -> None:
    """The other half: with the switch off and nothing declared, the note must say not attached."""
    from codna.durable.capability_notes import capability_notes

    notes = capability_notes(_run_spec(tmp_path), OFF)
    mcp = next(message for kind, message in notes if "MCP is" in message)
    assert "MCP is not attached" in mcp
    assert "browser" not in mcp


def test_the_capability_note_lists_declared_AND_builtin_servers_together(tmp_path: Path) -> None:
    """A run that declared its own server AND enabled the browser is reported as carrying both."""
    from codna.durable.capability_notes import capability_notes

    spec = _run_spec(tmp_path, (
        {"name": "docs", "transport": {"type": "stdio", "command": "npx", "args": []}},
    ))
    mcp = next(message for kind, message in capability_notes(spec, ON) if "MCP is" in message)
    assert "MCP is attached" in mcp
    assert "docs" in mcp
    assert "browser" in mcp


def test_the_capability_note_reads_the_environment_by_default(tmp_path: Path, monkeypatch) -> None:
    """The workflow passes no env, so the note must read the WORKER's switch, not a frozen empty map."""
    from codna.durable.capability_notes import capability_notes

    monkeypatch.setenv(BROWSER_TOOL_ENV, "1")
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime"))
    mcp = next(message for kind, message in capability_notes(_run_spec(tmp_path)) if "MCP is" in message)
    assert "MCP is attached" in mcp
    assert "browser" in mcp


def test_the_capability_note_does_not_hide_a_configuration_fault(tmp_path: Path) -> None:
    """A broken switch must still name the fault and its CODE — never a plain "not attached"."""
    from codna.durable.capability_notes import capability_notes

    notes = capability_notes(_run_spec(tmp_path), {**OFF, BROWSER_TOOL_ENV: "maybe"})
    mcp = next(message for kind, message in notes if "MCP is" in message)
    assert "MCP is not attached" in mcp
    # The machine half: the code travels, so a surface can branch without matching on prose.
    assert "browser_config_invalid" in mcp
    assert BROWSER_TOOL_ENV in mcp
