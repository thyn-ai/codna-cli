"""The browser tool's config, registration and tool-body surface — no browser, no `mcp` extra.

The tool is a Playwright MCP server, so three things can be wrong independently and each is pinned here:

* **The switch.** ``CODNA_BROWSER_TOOL`` is default-OFF (a browser loads untrusted code), every truthy
  spelling turns it on, and a value that is neither truthy nor falsy is a REFUSAL — never a guess.
* **The entry.** The pinned package, the flags and the output directory resolve into one canonical,
  history-safe server entry that goes through the durable validator, so what travels into a durable run, a
  chat session and the bridge cannot disagree about what "the browser" is.
* **The tools.** Only the six capabilities the tool promises are exposed, each body answers with text
  instead of raising, and the in-page-JavaScript tools Playwright also ships are unreachable.

Registration is exercised against a stub server rather than ``FastMCP``: the offline CI lane installs no
``mcp`` extra (that is the lane's contract), and the thing worth testing — WHICH tools appear under WHICH
switch, and that each delegates to the one body — is visible without it. The live server is covered by
``test_browser_tool_e2e.py``, which starts the real one.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

import pytest

from codna import browser_tool
from codna.browser_tool import (
    BROWSER_TOOL_ENV,
    BROWSER_TOOL_NAMES,
    CONNECT_TIMEOUT_ENV,
    DEFAULT_CONNECT_TIMEOUT_S,
    ENABLE_HINT,
    MAX_CONNECT_TIMEOUT_S,
    MIN_CONNECT_TIMEOUT_S,
    PLAYWRIGHT_MCP_PACKAGE,
    REQUIRED_PLAYWRIGHT_TOOLS,
    BrowserToolConfigError,
    browser_mcp_server,
    browser_mcp_servers,
    browser_server_args,
    browser_tool_config,
    browser_tool_enabled,
    browser_tool_state,
    playwright_tool_for,
    with_browser_tool,
)
from codna.durable.mcp import normalize_mcp_servers

ON = {BROWSER_TOOL_ENV: "1", "CODNA_RUNTIME_ROOT": "/tmp/codna-browser-test"}
OFF = {"CODNA_RUNTIME_ROOT": "/tmp/codna-browser-test"}


# ── the switch: default off, explicit on, and a typo is a refusal ────────────────────────────────


def test_unset_means_off() -> None:
    """The safe default. A browser is a network client; it is never on by accident."""
    assert browser_tool_enabled({}) is False
    assert browser_tool_enabled({"CODNA_BROWSER_TOOL": ""}) is False
    assert browser_mcp_servers({}) == ()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  1  "])
def test_every_truthy_spelling_enables(value: str) -> None:
    assert browser_tool_enabled({BROWSER_TOOL_ENV: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_every_falsy_spelling_disables(value: str) -> None:
    assert browser_tool_enabled({BROWSER_TOOL_ENV: value}) is False


@pytest.mark.parametrize("value", ["maybe", "yes please", "enabled", "2"])
def test_a_value_that_is_neither_truthy_nor_falsy_is_REFUSED(value: str) -> None:
    with pytest.raises(BrowserToolConfigError) as caught:
        browser_tool_enabled({BROWSER_TOOL_ENV: value})
    assert caught.value.code == "browser_config_invalid"
    assert caught.value.details["field"] == BROWSER_TOOL_ENV
    # The refusal names the setting, so a user knows which export to fix.
    assert BROWSER_TOOL_ENV in str(caught.value)


# ── the resolved configuration ───────────────────────────────────────────────────────────────────


def test_defaults_are_the_documented_ones() -> None:
    config = browser_tool_config(OFF)
    assert config.package == PLAYWRIGHT_MCP_PACKAGE
    assert config.headless is True
    assert config.isolated is True
    assert config.viewport == "1280x720"
    assert config.browser == ""
    assert config.file_access == "restricted"
    assert config.action_timeout_ms == 5_000
    assert config.navigation_timeout_ms == 60_000
    # The MCP-level bound, chosen to clear BOTH the engine's 1.5 s connect probe and a 60 s navigation.
    assert config.connect_timeout_s == 180.0
    # Codna's own state root, never the user's checkout.
    assert config.output_dir == "/tmp/codna-browser-test/browser"


def test_output_dir_defaults_under_the_runtime_root_and_honours_an_override(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", "/tmp/codna-rt-test")
    assert browser_tool_config({}).output_dir == "/tmp/codna-rt-test/browser"
    assert browser_tool_config({"CODNA_BROWSER_OUTPUT_DIR": "/tmp/explicit"}).output_dir == "/tmp/explicit"


def test_headless_and_isolated_can_be_turned_off() -> None:
    config = browser_tool_config(
        {**OFF, "CODNA_BROWSER_HEADLESS": "0", "CODNA_BROWSER_ISOLATED": "false"})
    assert config.headless is False
    assert config.isolated is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CODNA_BROWSER_VIEWPORT", "huge"),
        ("CODNA_BROWSER_VIEWPORT", "1280"),
        ("CODNA_BROWSER_VIEWPORT", "0x720"),
        ("CODNA_BROWSER_NAME", "netscape"),
        ("CODNA_BROWSER_FILE_ACCESS", "everything"),
        ("CODNA_BROWSER_ACTION_TIMEOUT_MS", "0"),
        ("CODNA_BROWSER_ACTION_TIMEOUT_MS", "soon"),
        ("CODNA_BROWSER_NAVIGATION_TIMEOUT_MS", "-5"),
        ("CODNA_BROWSER_OUTPUT_MAX_BYTES", "lots"),
        (CONNECT_TIMEOUT_ENV, "soon"),
        (CONNECT_TIMEOUT_ENV, "0"),
        (CONNECT_TIMEOUT_ENV, "-1"),
        (CONNECT_TIMEOUT_ENV, "nan"),
        (CONNECT_TIMEOUT_ENV, "inf"),
        (CONNECT_TIMEOUT_ENV, "5"),          # below MIN_CONNECT_TIMEOUT_S
        (CONNECT_TIMEOUT_ENV, "3601"),       # above the engine's own ceiling
    ],
)
def test_an_invalid_setting_is_REFUSED_and_names_its_field(key: str, value: str) -> None:
    with pytest.raises(BrowserToolConfigError) as caught:
        browser_tool_config({**OFF, key: value})
    assert caught.value.code == "browser_config_invalid"
    assert caught.value.details["field"] == key


# ── the launch: one pinned package, fixed flags, byte-identical twice ────────────────────────────


def test_args_are_pinned_and_ordered() -> None:
    args = browser_server_args(browser_tool_config(OFF))
    assert args == [
        "-y", PLAYWRIGHT_MCP_PACKAGE,
        "--headless", "--isolated",
        "--ignore-roots",
        "--output-dir", "/tmp/codna-browser-test/browser",
        "--output-max-size", "67108864",
        "--viewport-size", "1280x720",
        "--timeout-action", "5000",
        "--timeout-navigation", "60000",
    ]
    # Same configuration, same bytes: the launch is deterministic.
    assert args == browser_server_args(browser_tool_config(OFF))


def test_the_package_is_version_pinned_not_floating() -> None:
    """A floating tag makes "the browser check passed yesterday" mean nothing."""
    assert PLAYWRIGHT_MCP_PACKAGE.startswith("@playwright/mcp@")
    assert PLAYWRIGHT_MCP_PACKAGE != "@playwright/mcp"


def test_file_access_and_channel_are_opt_in_flags() -> None:
    restricted = browser_server_args(browser_tool_config(OFF))
    assert "--allow-unrestricted-file-access" not in restricted
    assert "--browser" not in restricted

    widened = browser_server_args(browser_tool_config({
        **OFF, "CODNA_BROWSER_FILE_ACCESS": "unrestricted", "CODNA_BROWSER_NAME": "chrome",
    }))
    # Opt-in, and last: the widening is visible at the end of the command line, not buried in it.
    assert widened[-3:] == ["--browser", "chrome", "--allow-unrestricted-file-access"]


def test_the_server_entry_survives_the_ONE_durable_validator() -> None:
    server = browser_mcp_server(browser_tool_config(ON))
    assert server["name"] == "browser"
    assert server["transport"]["type"] == "stdio"
    assert server["transport"]["command"] == "npx"


# ── the connect timeout: the entry must declare it, or the engine drops the server ───────────────
#
# These assert on the EMITTED ENTRY, not on the constant. A test of the constant alone passes while the
# entry stays stripped — and a stripped entry is exactly the defect: the engine's unconfigured 1.5 s probe
# loses a ~1.3 s cold start, logs the server as unavailable, and builds a session with no browser tools.


def test_the_emitted_entry_carries_a_finite_positive_timeout() -> None:
    """The defect, at the level it happened: the entry that reaches the engine has a `timeout`."""
    server = browser_mcp_server(browser_tool_config(ON))
    assert "timeout" in server, "the emitted entry has no timeout, so the engine applies its 1.5s probe"
    timeout = server["timeout"]
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert math.isfinite(timeout) and timeout > 0
    # It must clear the race it exists to win: far above the engine's 1.5 s probe and a ~1.3 s cold start.
    assert timeout > 1.5
    assert MIN_CONNECT_TIMEOUT_S <= timeout <= MAX_CONNECT_TIMEOUT_S


def test_the_emitted_timeout_passes_the_validator_unchanged() -> None:
    """The number the engine sees is the number Codna chose — not one a validator adjusted."""
    server = browser_mcp_server(browser_tool_config(ON))
    (again,) = normalize_mcp_servers([server])
    assert again["timeout"] == server["timeout"] == DEFAULT_CONNECT_TIMEOUT_S
    # And the whole entry is a fixed point: re-validating it changes nothing about the transport.
    assert again["transport"] == server["transport"]


def test_the_entry_is_byte_identical_twice_so_history_is_safe() -> None:
    """The entry is recorded in workflow history, so it may not read a clock or drift between calls."""
    first = browser_mcp_server(browser_tool_config(ON))
    second = browser_mcp_server(browser_tool_config(ON))
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_timeout_is_configurable_and_lands_on_the_entry() -> None:
    server = browser_mcp_server(browser_tool_config({**ON, CONNECT_TIMEOUT_ENV: "240"}))
    assert server["timeout"] == 240
    assert server["timeout"] == 240.0


@pytest.mark.parametrize("value", ["soon", "0", "-1", "nan", "inf", "5", "3601"])
def test_a_bad_connect_timeout_is_REFUSED_and_never_silently_defaulted(value: str) -> None:
    """A bad value FAILS CLOSED: an operator who exported a timeout must not get a silent default."""
    with pytest.raises(BrowserToolConfigError) as caught:
        browser_mcp_servers({**ON, CONNECT_TIMEOUT_ENV: value})
    assert caught.value.code == "browser_config_invalid"
    assert caught.value.details["field"] == CONNECT_TIMEOUT_ENV


def test_a_connect_timeout_below_the_navigation_budget_is_REFUSED() -> None:
    """The engine applies this to tool calls too: too small a value would abort a real navigation."""
    with pytest.raises(BrowserToolConfigError) as caught:
        browser_tool_config({**OFF, CONNECT_TIMEOUT_ENV: "30"})
    assert caught.value.details["field"] == CONNECT_TIMEOUT_ENV
    assert "navigation" in str(caught.value)


def test_raising_the_navigation_timeout_raises_the_default_connect_timeout_with_it() -> None:
    """A configuration made slower must not be refused for the default it did not choose."""
    config = browser_tool_config({**OFF, "CODNA_BROWSER_NAVIGATION_TIMEOUT_MS": "600000"})
    assert config.connect_timeout_s == 600.0
    assert browser_mcp_server(config)["timeout"] == 600.0


def test_a_navigation_budget_past_the_engines_ceiling_is_REFUSED_not_clamped() -> None:
    """Otherwise Codna would emit a number the engine silently shrinks below the navigation it covers."""
    with pytest.raises(BrowserToolConfigError) as caught:
        browser_tool_config({**OFF, "CODNA_BROWSER_NAVIGATION_TIMEOUT_MS": "4000000"})
    assert caught.value.details["field"] == "CODNA_BROWSER_NAVIGATION_TIMEOUT_MS"
    assert caught.value.code == "browser_config_invalid"


def test_the_off_state_still_emits_nothing_at_all() -> None:
    """The switch is unchanged: unset means no entry, so there is no timeout to carry either."""
    assert browser_mcp_servers({}) == ()
    assert browser_mcp_servers(OFF) == ()
    assert browser_mcp_servers({"CODNA_BROWSER_TOOL": "0"}) == ()


# ── merging with a user's own configuration ─────────────────────────────────────────────────────


def test_the_user_configured_server_always_wins() -> None:
    """Replacing a server somebody wrote is worse than a built-in tool that is not the one in use."""
    mine = {"name": "browser", "transport": {"type": "stdio", "command": "my-browser", "args": []}}
    assert with_browser_tool([mine], ON) == (mine,)


def test_enabling_the_tool_never_exceeds_the_cap() -> None:
    """The merge goes back through the validator, so the cap refuses rather than drops a server."""
    from codna.durable.mcp import MAX_MCP_SERVERS, McpConfigError

    full = [
        {"name": f"s{i}", "transport": {"type": "stdio", "command": "true", "args": []}}
        for i in range(MAX_MCP_SERVERS)
    ]
    with pytest.raises(McpConfigError) as caught:
        with_browser_tool(full, ON)
    assert caught.value.code == "mcp_config_invalid"


def test_switched_off_the_list_is_untouched() -> None:
    mine = {"name": "x", "transport": {"type": "stdio", "command": "true", "args": []}}
    assert with_browser_tool([mine], OFF) == (mine,)


# ── the tool allowlist ──────────────────────────────────────────────────────────────────────────


def test_every_codna_tool_maps_to_one_playwright_tool() -> None:
    for name in BROWSER_TOOL_NAMES:
        if name == "codna_browser_close":
            with pytest.raises(BrowserToolConfigError):
                playwright_tool_for(name)
            continue
        assert playwright_tool_for(name) in REQUIRED_PLAYWRIGHT_TOOLS


def test_the_six_required_capabilities_are_all_reachable() -> None:
    """navigate, screenshot, click, type, read accessible text, wait — each through a Codna tool."""
    reachable = {playwright_tool_for(name) for name in BROWSER_TOOL_NAMES
                 if name != "codna_browser_close"}
    assert reachable == set(REQUIRED_PLAYWRIGHT_TOOLS)


@pytest.mark.parametrize("name", ["browser_evaluate", "browser_run_code_unsafe", "codna_browser_eval"])
def test_an_unexposed_tool_is_refused(name: str) -> None:
    with pytest.raises(BrowserToolConfigError) as caught:
        playwright_tool_for(name)
    assert caught.value.code == "browser_tool_unknown"


# ── registration: which tools exist, under which switch ────────────────────────────────────────


class _StubToolAnnotations:
    """The fields `mcp_server` passes; the real class validates nothing a test needs."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _StubServer:
    """Records `@server.tool(...)` registrations. Enough to see WHAT is registered and call it."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}
        self.annotations: dict[str, object] = {}

    def tool(self, *, annotations: object = None):
        def wrap(function):
            self.tools[function.__name__] = function
            self.annotations[function.__name__] = annotations
            return function

        return wrap


@pytest.fixture
def stub_mcp(monkeypatch):
    """`mcp.types` as a stub module, so registration runs without the `mcp` extra installed."""
    module = types.ModuleType("mcp.types")
    module.ToolAnnotations = _StubToolAnnotations  # type: ignore[attr-defined]
    module.TextContent = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    module.ImageContent = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp.types", module)
    monkeypatch.setitem(sys.modules, "mcp", sys.modules.get("mcp") or types.ModuleType("mcp"))
    return module


def test_switched_off_nothing_is_registered(stub_mcp, monkeypatch) -> None:
    from codna import mcp_server

    monkeypatch.setattr(browser_tool, "browser_tool_enabled", lambda env=None: False)
    server = _StubServer()
    mcp_server._register_browser_tools(server)  # noqa: SLF001
    assert server.tools == {}


def test_switched_on_every_browser_tool_is_registered(stub_mcp, monkeypatch) -> None:
    from codna import mcp_server

    monkeypatch.setattr(browser_tool, "browser_tool_enabled", lambda env=None: True)
    server = _StubServer()
    mcp_server._register_browser_tools(server)  # noqa: SLF001
    assert tuple(server.tools) == BROWSER_TOOL_NAMES


def test_a_bad_switch_value_registers_NOTHING_and_never_raises(stub_mcp, monkeypatch) -> None:
    """A typo must not break `codna mcp` itself: the server still lists its core tools."""
    from codna import mcp_server

    monkeypatch.setenv(BROWSER_TOOL_ENV, "maybe")
    server = _StubServer()
    mcp_server._register_browser_tools(server)  # noqa: SLF001
    assert server.tools == {}


# ── the tool bodies: text in, text out, never a raise ──────────────────────────────────────────


def test_a_body_without_its_required_argument_answers_with_an_error() -> None:
    from codna import browser_bridge

    assert browser_bridge.browser_navigate_json("") == "codna_browser_navigate error: url is required"
    assert browser_bridge.browser_click_json("") == "codna_browser_click error: target is required"
    assert browser_bridge.browser_type_json("") == "codna_browser_type error: target is required"


def test_wait_without_anything_to_wait_for_is_a_structured_error() -> None:
    from codna import browser_bridge

    monkeypatch_free = browser_bridge.browser_wait_json()
    assert monkeypatch_free.startswith("codna_browser_wait error: ")
    assert "text" in monkeypatch_free


def test_an_unexposed_body_name_is_refused_before_a_browser_starts() -> None:
    from codna import browser_bridge

    text = browser_bridge.browser_json("codna_browser_eval", lambda client: None)
    assert text.startswith("codna_browser_eval error: ")
    assert "not a browser tool Codna exposes" in text


def test_a_body_reports_a_browser_failure_instead_of_raising(monkeypatch) -> None:
    from codna import browser_bridge

    def explode(_client):
        raise browser_bridge.BrowserToolError("boom", code="browser_tool_call_failed")

    monkeypatch.setattr(browser_bridge, "browser_client", lambda config=None: object())
    assert browser_bridge.browser_json("codna_browser_navigate", explode) == (
        "codna_browser_navigate error: boom")


def test_a_body_reports_an_unexpected_failure_instead_of_raising(monkeypatch) -> None:
    from codna import browser_bridge

    def explode(_client):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(browser_bridge, "browser_client", lambda config=None: object())
    assert browser_bridge.browser_json("codna_browser_snapshot", explode) == (
        "codna_browser_snapshot error: RuntimeError: unexpected")


def test_the_registered_bodies_delegate_to_the_module_level_ones(stub_mcp, monkeypatch) -> None:
    """One implementation, two ways to reach it: the registered tool cannot drift from the body."""
    from codna import browser_bridge, mcp_server

    monkeypatch.setattr(browser_tool, "browser_tool_enabled", lambda env=None: True)
    seen: list[tuple] = []
    monkeypatch.setattr(browser_bridge, "browser_navigate_json",
                        lambda url: seen.append(("nav", url)) or "ok")
    monkeypatch.setattr(browser_bridge, "browser_click_json",
                        lambda target, element: seen.append(("click", target, element)) or "ok")

    server = _StubServer()
    mcp_server._register_browser_tools(server)  # noqa: SLF001
    server.tools["codna_browser_navigate"](url="https://example.test")
    server.tools["codna_browser_click"](target="#go", element="Go")

    assert seen == [("nav", "https://example.test"), ("click", "#go", "Go")]


def test_close_reports_closed(monkeypatch) -> None:
    from codna import browser_bridge

    monkeypatch.setattr(browser_bridge, "close_browser_client", lambda: None)
    assert json.loads(browser_bridge.browser_close_json()) == {"closed": True}


# ── screenshots stay inside the output directory ──────────────────────────────────────────────


def test_screenshot_paths_stay_inside_the_output_directory() -> None:
    from codna.browser_bridge import BrowserToolError, screenshot_paths

    name, absolute = screenshot_paths("/tmp/out", "shots/page.png")
    assert name == "shots/page.png"
    assert absolute == "/tmp/out/shots/page.png"

    for bad in ("/etc/passwd", "~/shot.png", "../escape.png", "shots/../../escape.png"):
        with pytest.raises(BrowserToolError) as caught:
            screenshot_paths("/tmp/out", bad)
        assert caught.value.code == "browser_screenshot_path_invalid"


def test_a_screenshot_defaults_to_a_timestamped_name() -> None:
    from codna.browser_bridge import screenshot_paths

    name, absolute = screenshot_paths("/tmp/out", "")
    assert name.startswith("page-") and name.endswith(".png")
    assert absolute == f"/tmp/out/{name}"


# ── a per-call filename can never write into the signed app bundle ─────────────────────────────
#
# The measured defect: `browser_take_screenshot` called with a RELATIVE filename wrote two PNGs into
# `/Applications/Codna.app/Contents/Resources`, because Playwright resolves an EXPLICIT filename against
# the server's cwd, NOT against `--output-dir` (its own help says so). The seal broke. Two things close
# it, and both are asserted over the temp directory that stands in for the bundle:
#
#   * the launcher hands the server `cwd=<output_dir>` (so nothing inherits the app's cwd), and
#   * every per-call name is resolved and REFUSED if it would land outside the output directory.


def test_the_child_working_directory_is_the_output_directory_not_the_hosts_cwd() -> None:
    """The launcher half: the workspace root a relative filename resolves against is Codna's, not the app's."""
    config = browser_tool_config(ON)
    transport = browser_tool.browser_server_transport(config)
    assert transport["cwd"] == config.output_dir
    # It survives the one validator unchanged, so the durable and chat launchers inherit it too.
    assert browser_mcp_server(config)["transport"]["cwd"] == config.output_dir


def test_a_relative_screenshot_name_resolves_inside_the_output_directory(tmp_path) -> None:
    from codna.browser_tool import screenshot_output_path

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    relative, absolute = screenshot_output_path(str(output_dir), "page.png")
    assert relative == "page.png"
    assert absolute == str(output_dir / "page.png")


def test_a_nested_relative_name_keeps_its_subdirectory(tmp_path) -> None:
    from codna.browser_tool import screenshot_output_path

    output_dir = tmp_path / "out"
    (output_dir / "shots").mkdir(parents=True)
    relative, absolute = screenshot_output_path(str(output_dir), "shots/page.png")
    assert relative == "shots/page.png"
    assert absolute == str(output_dir / "shots" / "page.png")


@pytest.mark.parametrize(
    "filename",
    [
        "/Applications/Codna.app/Contents/Resources/page.png",
        "~/page.png",
        "../page.png",
        "shots/../../page.png",
        "./../Codna.app/Contents/Resources/page.png",
        "../../Resources/page.png",
    ],
)
def test_a_per_call_filename_that_escapes_the_output_directory_is_REFUSED(
    tmp_path, filename: str,
) -> None:
    """The bundle is simulated by `tmp_path`; the offending value must be NAMED in a coded error."""
    from codna.browser_tool import BrowserToolConfigError, screenshot_output_path

    output_dir = tmp_path / "Codna.app" / "Contents" / "Resources" / "browser"
    output_dir.mkdir(parents=True)
    with pytest.raises(BrowserToolConfigError) as caught:
        screenshot_output_path(str(output_dir), filename)
    assert caught.value.code == "browser_screenshot_path_invalid"
    # The refusal names what the user asked for, so it is actionable rather than a silent rewrite.
    assert filename in str(caught.value)
    assert caught.value.details["filename"] == filename
    # Nothing was created anywhere: a refusal is a refusal, not a clamped write.
    assert not (output_dir / "page.png").exists()


def test_the_bridge_refuses_the_escape_before_it_calls_the_server(tmp_path) -> None:
    """The tool-body half: a browsing session refuses the write instead of handing it to Playwright."""
    from codna.browser_bridge import BrowserToolError, screenshot_paths

    output_dir = tmp_path / "browser"
    output_dir.mkdir()
    for bad in ("../escape.png", "/Applications/Codna.app/Contents/Resources/x.png", "~/x.png"):
        with pytest.raises(BrowserToolError) as caught:
            screenshot_paths(str(output_dir), bad)
        assert caught.value.code == "browser_screenshot_path_invalid"
        assert bad in str(caught.value)
    assert not (tmp_path / "escape.png").exists()


def test_a_bare_relative_screenshot_name_does_NOT_land_in_the_inherited_cwd(tmp_path) -> None:
    """THE INCIDENT, reproduced with a simulated bundle.

    Playwright resolves an explicit `filename` against the SERVER'S CWD, not `--output-dir`. The installed
    app's server inherited `/Applications/Codna.app/Contents/Resources`, so a bare `"page.png"` landed
    inside the code-signed bundle and `codesign --verify` stopped passing.

    The "bundle" here is a directory whose contents stand for the sealed payload. Before the fix the
    server was launched with the host's cwd and that is where `page.png` resolved; after the fix the server
    is launched INSIDE the output directory, so the same call lands in Codna's own state — and the bundle's
    listing is byte-identical. No real `codesign` is involved (see the PR report for that caveat).
    """
    import os

    from codna.browser_bridge import BrowserToolError, screenshot_paths

    bundle = tmp_path / "Codna.app" / "Contents" / "Resources"
    bundle.mkdir(parents=True)
    (bundle / "Info.plist").write_bytes(b"sealed")
    sealed = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*"))

    output_dir = tmp_path / "runtime" / "browser"
    output_dir.mkdir(parents=True)
    config = browser_tool_config({
        **ON, "CODNA_BROWSER_OUTPUT_DIR": str(output_dir),
    })

    # The workspace root the child gets. A bare name resolves against THIS, so it must be ours.
    child_cwd = browser_tool.browser_server_transport(config)["cwd"]
    assert child_cwd == str(output_dir)
    assert os.path.commonpath([child_cwd, str(bundle)]) != child_cwd, (
        "a relative screenshot name would resolve into the app bundle"
    )

    # The escape forms are refused outright, and neither is written.
    with pytest.raises(BrowserToolError):
        screenshot_paths(str(output_dir), "../../Codna.app/Contents/Resources/page.png")
    with pytest.raises(BrowserToolError):
        screenshot_paths(str(output_dir), str(bundle / "page.png"))

    _, absolute = screenshot_paths(str(output_dir), "page.png")
    Path(absolute).write_bytes(b"\x89PNG\r\n\x1a\n")

    after = sorted(path.relative_to(bundle).as_posix() for path in bundle.rglob("*"))
    assert after == sealed == ["Info.plist"]
    assert (output_dir / "page.png").is_file()


def test_an_empty_filename_yields_NO_path_rather_than_the_directory(tmp_path) -> None:
    """The resolver's documented pair is ``(relative_name, absolute_path)`` — or NONE of it.

    An empty/whitespace name is Playwright's own auto-naming case: the server picks the name, so this
    function knows neither element. Returning ``("", output_dir)`` would hand a caller the output
    DIRECTORY as if it were a screenshot path, and a caller that passed that to Playwright (or opened
    it) would act on a directory. The pair is therefore ``("", "")`` — empty name, empty path — which
    is unambiguous and keeps the two elements the same kind of thing.
    """
    from codna.browser_tool import screenshot_output_path

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in ("", "   ", "\t\n"):
        relative, absolute = screenshot_output_path(str(output_dir), name)
        assert relative == ""
        assert absolute == ""
        # Explicitly: the output DIRECTORY is never handed back as a file path.
        assert absolute != str(output_dir)


# ── the honest status report ──────────────────────────────────────────────────────────────────


def test_state_off_names_the_switch_that_turns_it_on() -> None:
    state = browser_tool_state(OFF)
    assert state["enabled"] is False
    assert state["launchable"] is False
    assert state["enable_with"] == ENABLE_HINT
    assert BROWSER_TOOL_ENV in state["reason"]
    # The machine half of the answer, so a surface that shows a hint does not have to match on prose.
    assert state["reason_code"] == "browser_tool_disabled"


def test_state_on_is_launchable_when_npx_exists() -> None:
    state = browser_tool_state(ON)
    assert state["enabled"] is True
    assert state["launchable"] is True
    assert state["server"]["name"] == "browser"
    assert state["tools"] == list(BROWSER_TOOL_NAMES)
    # Launchable means "the launcher can start", so a reason is the defect: a state that says both is a
    # surface that cannot decide what to display.
    assert state["reason"] == ""
    assert state["reason_code"] == ""


def test_state_on_without_npx_says_it_cannot_start_and_why(monkeypatch) -> None:
    monkeypatch.setattr(browser_tool.shutil, "which", lambda name: None)
    state = browser_tool_state(ON)
    assert state["enabled"] is True
    assert state["launchable"] is False
    assert "npx" in state["reason"]
    assert state["reason_code"] == "browser_tool_npx_missing"


def test_state_reports_an_invalid_setting_as_the_error_it_is() -> None:
    state = browser_tool_state({BROWSER_TOOL_ENV: "maybe"})
    assert state["enabled"] is False
    assert state["error"]["code"] == "browser_config_invalid"
    assert state["error"]["details"]["field"] == BROWSER_TOOL_ENV
    assert state["reason_code"] == "browser_config_invalid"


# ── the runtime requirement, said honestly ───────────────────────────────────────────────────
#
# The MCP server is `@playwright/mcp` (Node) and IT launches Chromium. Codna ships the Python launcher, not a
# browser, so both are runtime prerequisites. The distinction these tests pin is the one that matters to a user:
# a missing `npx` means the tool cannot start at all; a missing browser cache means the FIRST launch downloads
# Chromium, which works online and fails offline. Reporting the second as "cannot start" would be wrong on every
# fresh machine, and hiding it would turn an offline failure into a mystery.


def test_state_without_a_browser_cache_warns_but_stays_launchable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(browser_tool.shutil, "which", lambda name: "/usr/bin/npx")
    empty = tmp_path / "ms-playwright"
    empty.mkdir()
    state = browser_tool_state({**ON, "PLAYWRIGHT_BROWSERS_PATH": str(empty)})
    assert state["launchable"] is True
    assert state["reason"] == ""
    assert state["chromium"]["cached"] is False
    codes = [warning["code"] for warning in state["warnings"]]
    assert codes == ["browser_chromium_not_cached"]
    # The message names the remedy, not just the problem.
    message = state["warnings"][0]["message"]
    assert "npx playwright install chromium" in message
    assert "needs network" in message


def test_state_with_a_cached_chromium_carries_no_warning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(browser_tool.shutil, "which", lambda name: "/usr/bin/npx")
    root = tmp_path / "ms-playwright"
    (root / "chromium_headless_shell-1248").mkdir(parents=True)
    state = browser_tool_state({**ON, "PLAYWRIGHT_BROWSERS_PATH": str(root)})
    assert state["launchable"] is True
    assert state["chromium"]["cached"] is True
    assert state["warnings"] == []


def test_state_cannot_tell_when_playwright_points_inside_the_package(monkeypatch) -> None:
    """`PLAYWRIGHT_BROWSERS_PATH=0` means "inside the package", so the answer is UNKNOWN, never "missing".

    Guessing here would report a first-run download note on a machine that has the browser vendored — a
    warning about something that is present is the same class of lie as silence about something absent.
    """
    monkeypatch.setattr(browser_tool.shutil, "which", lambda name: "/usr/bin/npx")
    state = browser_tool_state({**ON, "PLAYWRIGHT_BROWSERS_PATH": "0"})
    assert state["launchable"] is True
    assert state["chromium"]["cached"] is None
    assert state["warnings"] == []


def test_state_does_not_check_the_bundled_chromium_when_a_channel_is_named(monkeypatch) -> None:
    """`CODNA_BROWSER_NAME=chrome` uses the INSTALLED Chrome, so the Playwright cache is not a prerequisite."""
    monkeypatch.setattr(browser_tool.shutil, "which", lambda name: "/usr/bin/npx")
    state = browser_tool_state({**ON, "CODNA_BROWSER_NAME": "chrome"})
    assert state["launchable"] is True
    assert "chromium" not in state
    assert state["warnings"] == []


# ── startup failure is a PERSISTENT, structured state, not a one-off message ───────────────────


def test_a_startup_timeout_is_remembered_and_reported_by_every_later_call(monkeypatch) -> None:
    """A timeout must not turn the client into a silent no-op with one lost message.

    The attempt is created before the caller waits, so a timeout that lived only in the raising frame
    left every later ``start()`` with no session and no error — the caller got "the browser session is
    not running" forever while the real reason was never shown again.

    ``_serve`` is replaced with a session that never becomes ready, so this exercises the timeout STATE
    MACHINE without spawning npx: a real child abandoned mid-startup would outlive the test and write to a
    closed pipe.
    """
    import asyncio

    from codna.browser_bridge import BrowserToolError, PlaywrightClient

    async def never_ready(self) -> None:
        stop = asyncio.Event()
        with self._settle_lock:  # noqa: SLF001 - the event `close` signals, as `_serve` publishes it
            self._stop_async = stop  # noqa: SLF001
        await stop.wait()

    monkeypatch.setattr(PlaywrightClient, "_serve", never_ready)

    client = PlaywrightClient(browser_tool_config(ON), startup_timeout_s=0.05)
    try:
        with pytest.raises(BrowserToolError) as first:
            client.start()
        assert first.value.code == "browser_startup_timeout"
        # The SAME reason, still, on the second attempt — not a silent return.
        with pytest.raises(BrowserToolError) as second:
            client.start()
        assert second.value.code == "browser_startup_timeout"
        with pytest.raises(BrowserToolError) as call:
            client.call("browser_snapshot", {})
        assert call.value.code == "browser_startup_timeout"
    finally:
        client.close()


def test_a_timeout_is_reported_as_a_timeout_even_when_the_config_cannot_be_read(
    monkeypatch,
) -> None:
    """The deadline's own error must not be able to throw.

    ``_timeout_error`` used to read ``self.config.package`` while building its message. With an invalid
    ``CODNA_BROWSER_VIEWPORT`` that read raises ``BrowserToolConfigError``, so a genuine startup timeout
    escaped as a config error and the real reason — the browser never answered — was lost. The timeout
    verdict must survive an unreadable config and still name the timeout.
    """
    import asyncio

    from codna.browser_bridge import BrowserToolError, PlaywrightClient

    async def never_ready(self) -> None:
        stop = asyncio.Event()
        with self._settle_lock:  # noqa: SLF001 - the event `close` signals, as `_serve` publishes it
            self._stop_async = stop  # noqa: SLF001
        await stop.wait()

    monkeypatch.setattr(PlaywrightClient, "_serve", never_ready)
    monkeypatch.setenv("CODNA_BROWSER_VIEWPORT", "not-a-viewport")
    client = PlaywrightClient(None, startup_timeout_s=0.05)
    try:
        with pytest.raises(BrowserToolError) as timed_out:
            client.start()
        assert timed_out.value.code == "browser_startup_timeout", (
            "a timeout was reported as "
            f"{timed_out.value.code!r}; the config error masked the real reason"
        )
        message = str(timed_out.value)
        assert "did not answer initialize" in message
        # The diagnostic detail is still there, and it is honest about not knowing the package.
        assert timed_out.value.details["timeout_s"] == 0.05
        assert timed_out.value.details["package"] == ""
        assert timed_out.value.details["package_resolved"] is False
        assert "package unknown" in message
    finally:
        client.close()


def test_a_config_error_raised_before_any_session_is_a_BrowserToolError(
    monkeypatch,
) -> None:
    """Resolving the config can itself fail; that failure must not escape as a foreign error type.

    Every tool body catches ``BrowserToolError``. A bad setting raises ``BrowserToolConfigError`` during
    ``self.config`` inside the session thread, so without normalisation it escaped ``start()`` as a type
    the bodies do not catch.

    ``_serve`` is replaced with a body that only resolves the config, so this asserts ``_run``'s
    normalisation and is independent of whether the ``mcp`` extra is installed — the real ``_serve``
    imports ``mcp`` first, and on the offline lane its absence legitimately wins.
    """
    from codna.browser_bridge import BrowserToolError, PlaywrightClient

    async def resolve_config_only(self) -> None:
        _ = self.config  # resolution is the operation under test

    monkeypatch.setattr(PlaywrightClient, "_serve", resolve_config_only)
    monkeypatch.setenv("CODNA_BROWSER_VIEWPORT", "not-a-viewport")
    client = PlaywrightClient(None)
    try:
        with pytest.raises(BrowserToolError) as caught:
            client.start()
        assert caught.value.code == "browser_config_invalid"
        assert caught.value.details["field"] == "CODNA_BROWSER_VIEWPORT"
    finally:
        client.close()


def test_a_reported_startup_failure_is_not_replaced_by_a_later_less_specific_one() -> None:
    """The first verdict wins, whatever it was: a later writer is handed the settled reason.

    This is the property that makes the old race impossible rather than merely unlikely. The session
    thread's own specific reason must survive a later generic write, and — symmetrically — a recorded
    timeout must not be overwritten by a thread that finishes connecting afterwards.
    """
    from codna.browser_bridge import BrowserToolError, PlaywrightClient

    client = PlaywrightClient(browser_tool_config(ON))
    try:
        specific = BrowserToolError("npx is missing", code="browser_startup_failed")
        assert client._settle(specific) is specific  # noqa: SLF001 - the transition under test
        later = BrowserToolError("unhandled errors in a TaskGroup", code="browser_startup_failed")
        # The late writer loses, and is TOLD it lost, rather than silently believing it wrote.
        assert client._settle(later) is specific  # noqa: SLF001
        assert client.state.failure_code == "browser_startup_failed"
        assert "npx is missing" in client.state.failure_message
    finally:
        client.close()


# ── the env mapping is the WHOLE answer: no hidden read of the process environment ────────────
#
# A LOW-severity review finding, and a real one: `_playwright_browser_roots` fell back to
# `os.environ["PLAYWRIGHT_BROWSERS_PATH"]` when the caller's mapping did not name it. Every public entry
# point already passes `_env_map(env)`, which IS `os.environ` for `env=None`, so the fallback changed
# nothing for the ordinary case — it only made an explicitly supplied mapping leak one host variable,
# which is the one case a caller supplies a mapping to control.


def test_a_caller_supplied_env_does_not_inherit_the_hosts_browser_path(monkeypatch, tmp_path) -> None:
    """MEASURED before the fix: the host's `PLAYWRIGHT_BROWSERS_PATH` overrode an explicit map.

    Asserted on `_playwright_browser_roots` because that is the function with the defect and the one
    place the answer is decided. `os.environ` is pointed at a populated cache, and a map that does not
    name the variable must get the PLATFORM DEFAULT — proving both that the host value is invisible
    and that the fix did not simply stop answering.
    """
    populated = tmp_path / "host-cache"
    populated.mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(populated))

    # THE DEFECT: this returned `[str(populated)]` — the host's cache — for a map that never named it.
    assert browser_tool._playwright_browser_roots({}) != [str(populated)]
    # Still a REAL answer, not an empty one: the platform default under the user's home.
    assert browser_tool._playwright_browser_roots({}) == browser_tool._playwright_browser_roots(
        {"PLAYWRIGHT_BROWSERS_PATH": ""}
    )

    # And the mapping still decides when it DOES name the variable — the read is removed, not the feature.
    assert browser_tool._playwright_browser_roots({"PLAYWRIGHT_BROWSERS_PATH": str(populated)}) == [
        str(populated)
    ]
    # Including the special "0", which means "inside the package" and is therefore unknowable here.
    assert browser_tool._playwright_browser_roots({"PLAYWRIGHT_BROWSERS_PATH": "0"}) == []


def test_the_default_env_still_reads_the_process_environment(monkeypatch, tmp_path) -> None:
    """`env=None` means `os.environ`, so the ordinary CLI path is UNCHANGED by the fix.

    Asserted separately because "stop falling back" could be implemented by never reading the host at
    all, which would break `codna mcp browser` on every machine that sets the variable.
    """
    populated = tmp_path / "host-cache"
    populated.mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(populated))
    assert browser_tool._env_map(None) is os.environ
    assert browser_tool._playwright_browser_roots(browser_tool._env_map(None)) == [str(populated)]
