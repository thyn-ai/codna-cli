"""The browser tool — ONE definition of the Playwright MCP server Codna hands to an agent.

**The gap this closes.** Codna can build a web app and had no way to look at it. "navigate, screenshot,
assert" is the instrument a browser check needs, and without one the check was a human's job. Cursor names
a browser tool among an agent's tools; this is Codna's.

**Why an MCP server rather than in-process Playwright.** Codna is already an MCP participant in both
directions: it *is* a server (``codna mcp``) and it *carries* servers into a session (``mcp_servers`` on a
durable run, the bridge on the chat path). Playwright's own MCP server is the upstream, maintained
implementation of the whole surface — a second in-process browser driver would be a second thing to keep
working, with its own selector engine, its own snapshot format and its own bugs. So this module does not
re-implement browsing; it defines the server ENTRY (pinned package, flags, output directory) and the
switch that turns it on, once, for every path:

  * ``codna mcp``                — registers ``codna_browser_*`` tools (see ``mcp_server``), so an
                                   external client (Cursor, Claude Desktop) gets a browser;
  * a durable run / goal mode    — ``codna.durable.mcp.session_mcp_servers`` adds the entry, so the run's
                                   own agent gets the browser;
  * the IDE chat path            — ``codna.mcp_bridge`` adds the entry, so chat does too.

That split is why this file exists rather than the same flags being written three times: three copies of a
pinned package version are three versions that drift, and the drift is a browser check that passed in chat
and failed in a run.

**Default OFF, and the default is the safe one.** A browser is a network client that loads untrusted code
and writes screenshots to disk. It is therefore opt-in: ``CODNA_BROWSER_TOOL=1`` (documented in
``docs/mcp.md`` and the README). Unset means disabled, and nothing is registered, spawned or written. A
value that is neither truthy nor falsy is REFUSED rather than guessed — a typo in an enable switch must
not silently hand an agent a browser.

**Local pages need an explicit widening.** Playwright's server blocks the ``file:`` protocol and restricts
the filesystem to its workspace roots. ``CODNA_BROWSER_FILE_ACCESS=unrestricted`` passes
``--allow-unrestricted-file-access``; it is the only switch here that widens access, it is off by default,
and it is named in the docs beside the flag. ``data:`` URLs and a loopback HTTP server (what a just-built
app is served from) need no widening at all, which is why the smoke test serves a real page over loopback.

**Determinism.** Every value below is resolved from an explicit setting or a documented default; nothing
reads the clock or the filesystem to decide what to launch, so two launches of one configuration are
byte-identical. The output directory is ``$CODNA_RUNTIME_ROOT/browser`` (else ``~/.codna/browser``), like
every other piece of Codna's local state, so screenshots never land in the user's checkout.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Final, Mapping

from .durable.mcp import normalize_mcp_servers

__all__ = [
    "BROWSER_TOOL_ENV",
    "BROWSER_TOOL_NAMES",
    "DEFAULT_SERVER_NAME",
    "ENABLE_HINT",
    "FILE_ACCESS_MODES",
    "PLAYWRIGHT_MCP_PACKAGE",
    "REQUIRED_PLAYWRIGHT_TOOLS",
    "BrowserToolConfig",
    "BrowserToolConfigError",
    "browser_mcp_server",
    "browser_mcp_servers",
    "browser_server_args",
    "browser_server_transport",
    "browser_tool_config",
    "browser_tool_enabled",
    "browser_tool_state",
    "playwright_tool_for",
    "with_browser_tool",
]

#: The single switch. Truthy enables the tool everywhere; unset/empty/false disables it.
BROWSER_TOOL_ENV: Final = "CODNA_BROWSER_TOOL"

#: What the user types to turn it on. One spelling, everywhere it is reported.
ENABLE_HINT: Final = f"{BROWSER_TOOL_ENV}=1"

#: The server name, which prefixes every Playwright tool the agent sees (``browser_navigate`` …). Kept
#: stable because a user's own ``.codna/mcp.json`` may already name a server ``browser``: the merge rule
#: never overwrites an existing name, so the user's server wins and this one is skipped rather than
#: silently replacing a configuration the user wrote.
DEFAULT_SERVER_NAME: Final = "browser"

#: PINNED. A floating ``@playwright/mcp`` would make the tool change under a user between two runs of the
#: same app, and "the browser check passed yesterday" would then mean nothing. Bumped deliberately, in the
#: same commit as the docs and the smoke test — never resolved at launch.
PLAYWRIGHT_MCP_PACKAGE: Final = "@playwright/mcp@0.0.85"

#: The Playwright MCP tools this integration depends on: the six capabilities the tool promises (navigate,
#: screenshot, click, type, read accessible text, wait) map to exactly these. Named here so the contract
#: is ONE list — the live smoke test asserts this tuple against the running server, so a package bump that
#: renames one of them fails a test rather than a chat.
REQUIRED_PLAYWRIGHT_TOOLS: Final[tuple[str, ...]] = (
    "browser_navigate",
    "browser_take_screenshot",
    "browser_click",
    "browser_type",
    "browser_snapshot",
    "browser_wait_for",
)

#: The Codna tools this server exposes. The prefix says which server a tool belongs to
#: (``codna_browser_*``) the way every other Codna tool does.
BROWSER_TOOL_NAMES: Final[tuple[str, ...]] = (
    "codna_browser_navigate",
    "codna_browser_snapshot",
    "codna_browser_click",
    "codna_browser_type",
    "codna_browser_screenshot",
    "codna_browser_wait",
    "codna_browser_close",
)

#: Codna tool -> the Playwright MCP tool it drives. One table, so a name and the call it makes cannot
#: drift. ``codna_browser_close`` maps to nothing because it tears the session down locally rather than
#: forwarding a call. This table IS the allowlist: a Playwright tool not named here is unreachable,
#: including ``browser_evaluate`` and ``browser_run_code_unsafe`` (arbitrary in-page JavaScript) and
#: ``browser_file_upload`` — capabilities nobody asked this tool for, which an agent should not have by
#: accident.
_TOOL_TO_PLAYWRIGHT: Final[dict[str, str]] = {
    "codna_browser_navigate": "browser_navigate",
    "codna_browser_snapshot": "browser_snapshot",
    "codna_browser_click": "browser_click",
    "codna_browser_type": "browser_type",
    "codna_browser_screenshot": "browser_take_screenshot",
    "codna_browser_wait": "browser_wait_for",
    "codna_browser_close": "",
}

#: ``restricted`` (default) keeps Playwright's own filesystem policy; ``unrestricted`` passes
#: ``--allow-unrestricted-file-access``, which is also what makes ``file://`` navigation work.
FILE_ACCESS_MODES: Final[tuple[str, ...]] = ("restricted", "unrestricted")

#: Browser channels Playwright's server accepts for ``--browser``. Empty (the default) means the bundled
#: Chromium — passing the flag with no channel is a launch error, so it is omitted entirely.
_BROWSER_CHANNELS: Final[tuple[str, ...]] = ("chrome", "firefox", "webkit", "msedge")

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})
_VIEWPORT = re.compile(r"^[1-9]\d{1,4}x[1-9]\d{1,4}$")

DEFAULT_VIEWPORT: Final = "1280x720"
DEFAULT_ACTION_TIMEOUT_MS: Final = 5_000
DEFAULT_NAVIGATION_TIMEOUT_MS: Final = 60_000
# Bounds the output directory: Playwright's own eviction threshold, made explicit so a long-running
# session cannot fill a disk with screenshots.
DEFAULT_OUTPUT_MAX_SIZE_BYTES: Final = 64 * 1024 * 1024


class BrowserToolConfigError(ValueError):
    """A browser-tool setting that must not be silently tolerated. Always carries a ``code``."""

    def __init__(self, message: str, *, code: str = "browser_config_invalid",
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass(frozen=True)
class BrowserToolConfig:
    """Everything the launch depends on, resolved. Immutable, so a launched server cannot be mutated."""

    package: str
    headless: bool
    isolated: bool
    viewport: str
    browser: str
    output_dir: str
    file_access: str
    action_timeout_ms: int
    navigation_timeout_ms: int
    output_max_size_bytes: int

    @property
    def server_name(self) -> str:
        return DEFAULT_SERVER_NAME


# -- helpers -------------------------------------------------------------------------------------


def _env_map(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _invalid(field: str, value: Any, expected: str) -> BrowserToolConfigError:
    return BrowserToolConfigError(
        f"{field} must be {expected} (got {value!r})",
        details={"field": field, "value": repr(value)[:120]},
    )


def _flag(value: str | None, *, field: str) -> bool:
    """A boolean setting: truthy, falsy, or REFUSED. Never guessed."""
    text = (value or "").strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise _invalid(field, value, "one of 1/0, true/false, yes/no, on/off")


def _flag_or(value: str | None, *, field: str, default: bool) -> bool:
    """``_flag`` with ``default`` for an unset or empty value (an empty export means "not set")."""
    if not (value or "").strip():
        return default
    return _flag(value, field=field)


def _positive_int(value: str | None, *, field: str, default: int) -> int:
    text = (value or "").strip()
    if not text:
        return default
    try:
        parsed = int(text)
    except ValueError as exc:
        raise _invalid(field, value, "a positive whole number") from exc
    if parsed <= 0:
        raise _invalid(field, value, "a positive whole number")
    return parsed


def _runtime_root(values: Mapping[str, str]) -> str:
    """The root of Codna's local state: ``CODNA_RUNTIME_ROOT`` else ``~/.codna``.

    The same two answers ``keystore._state_root`` and ``report_cli`` give, so browser output sits beside
    every other piece of runtime state. A caller that passes an ``env`` map gets that map's value, which is
    what keeps a test out of the developer's real ``~/.codna``.
    """
    override = (values.get("CODNA_RUNTIME_ROOT") or "").strip()
    if not override:
        override = (os.environ.get("CODNA_RUNTIME_ROOT") or "").strip()
    root = override or os.path.join(os.path.expanduser("~"), ".codna")
    return os.path.abspath(os.path.expanduser(root))


# -- configuration -------------------------------------------------------------------------------


def browser_tool_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the browser tool is switched on. Raises ``BrowserToolConfigError`` on a bad value.

    UNSET MEANS OFF — the default is the safe one for a network client that loads untrusted code. A value
    that is neither truthy nor falsy is REFUSED rather than treated as off: a user who exported
    ``CODNA_BROWSER_TOOL=yes please`` believes a browser is available, and an agent that quietly has none
    produces a "browser check" that checked nothing.
    """
    return _flag_or(_env_map(env).get(BROWSER_TOOL_ENV), field=BROWSER_TOOL_ENV, default=False)


def browser_tool_config(env: Mapping[str, str] | None = None) -> BrowserToolConfig:
    """Resolve the launch configuration. Pure: reads ``env`` (default ``os.environ``) and nothing else."""
    values = _env_map(env)

    channel = (values.get("CODNA_BROWSER_NAME") or "").strip().lower()
    if channel and channel not in _BROWSER_CHANNELS:
        raise _invalid(
            "CODNA_BROWSER_NAME", values.get("CODNA_BROWSER_NAME"),
            f"one of {', '.join(_BROWSER_CHANNELS)}, or empty for the bundled Chromium",
        )

    viewport = (values.get("CODNA_BROWSER_VIEWPORT") or "").strip() or DEFAULT_VIEWPORT
    if not _VIEWPORT.match(viewport):
        raise _invalid("CODNA_BROWSER_VIEWPORT", values.get("CODNA_BROWSER_VIEWPORT"),
                       "WIDTHxHEIGHT, e.g. 1280x720")

    file_access = (values.get("CODNA_BROWSER_FILE_ACCESS") or "").strip().lower() or "restricted"
    if file_access not in FILE_ACCESS_MODES:
        raise _invalid("CODNA_BROWSER_FILE_ACCESS", values.get("CODNA_BROWSER_FILE_ACCESS"),
                       f"one of {', '.join(FILE_ACCESS_MODES)}")

    package = (values.get("CODNA_BROWSER_PACKAGE") or "").strip() or PLAYWRIGHT_MCP_PACKAGE
    output_dir = (values.get("CODNA_BROWSER_OUTPUT_DIR") or "").strip() or os.path.join(
        _runtime_root(values), "browser")

    return BrowserToolConfig(
        package=package,
        headless=_flag_or(values.get("CODNA_BROWSER_HEADLESS"),
                          field="CODNA_BROWSER_HEADLESS", default=True),
        isolated=_flag_or(values.get("CODNA_BROWSER_ISOLATED"),
                          field="CODNA_BROWSER_ISOLATED", default=True),
        viewport=viewport,
        browser=channel,
        output_dir=os.path.abspath(os.path.expanduser(output_dir)),
        file_access=file_access,
        action_timeout_ms=_positive_int(values.get("CODNA_BROWSER_ACTION_TIMEOUT_MS"),
                                        field="CODNA_BROWSER_ACTION_TIMEOUT_MS",
                                        default=DEFAULT_ACTION_TIMEOUT_MS),
        navigation_timeout_ms=_positive_int(values.get("CODNA_BROWSER_NAVIGATION_TIMEOUT_MS"),
                                            field="CODNA_BROWSER_NAVIGATION_TIMEOUT_MS",
                                            default=DEFAULT_NAVIGATION_TIMEOUT_MS),
        output_max_size_bytes=_positive_int(values.get("CODNA_BROWSER_OUTPUT_MAX_BYTES"),
                                            field="CODNA_BROWSER_OUTPUT_MAX_BYTES",
                                            default=DEFAULT_OUTPUT_MAX_SIZE_BYTES),
    )


def playwright_tool_for(tool_name: str) -> str:
    """The Playwright tool a Codna browser tool drives, or raise ``BrowserToolConfigError``.

    The single place a Codna tool name is mapped, and the reason the bridge can refuse an unexposed
    Playwright tool before it is ever called: an allowlist that only exists in a docstring is not an
    allowlist.
    """
    if tool_name not in _TOOL_TO_PLAYWRIGHT:
        raise BrowserToolConfigError(
            f"{tool_name!r} is not a browser tool Codna exposes "
            f"(exposed: {', '.join(BROWSER_TOOL_NAMES)})",
            code="browser_tool_unknown",
            details={"requested": tool_name, "exposed": list(BROWSER_TOOL_NAMES)},
        )
    target = _TOOL_TO_PLAYWRIGHT[tool_name]
    if not target:
        raise BrowserToolConfigError(
            f"{tool_name!r} does not forward to a Playwright tool",
            code="browser_tool_not_forwardable",
            details={"requested": tool_name},
        )
    return target


# -- the server entry ----------------------------------------------------------------------------


def browser_server_args(config: BrowserToolConfig | None = None) -> list[str]:
    """The ``npx`` argument list, in a fixed order so two launches are byte-identical.

    ``-y`` is Playwright's documented non-interactive form. ``--ignore-roots`` keeps the workspace root at
    the launching process's cwd instead of round-tripping a roots request through the client — a client
    that advertises roots but cannot answer one gets a stalled server, and Codna passes absolute paths
    anyway.
    """
    config = config or browser_tool_config()
    args: list[str] = ["-y", config.package]
    if config.headless:
        args.append("--headless")
    if config.isolated:
        args.append("--isolated")
    args += [
        "--ignore-roots",
        "--output-dir", config.output_dir,
        "--output-max-size", str(config.output_max_size_bytes),
        "--viewport-size", config.viewport,
        "--timeout-action", str(config.action_timeout_ms),
        "--timeout-navigation", str(config.navigation_timeout_ms),
    ]
    if config.browser:
        args += ["--browser", config.browser]
    if config.file_access == "unrestricted":
        args.append("--allow-unrestricted-file-access")
    return args


def browser_server_transport(config: BrowserToolConfig | None = None) -> dict[str, Any]:
    """The canonical stdio transport, in the shape ``codna.durable.mcp`` represents one."""
    config = config or browser_tool_config()
    return {"type": "stdio", "command": "npx", "args": browser_server_args(config)}


def browser_mcp_server(config: BrowserToolConfig | None = None) -> dict[str, Any]:
    """One canonical, history-safe server entry, judged by the ONE validator before it is returned.

    Going through ``normalize_mcp_servers`` is not ceremony: the entry travels into a session request and
    into workflow history, so a name or transport the engine refuses must be refused HERE, at the source,
    rather than by a 400 halfway through a run.
    """
    config = config or browser_tool_config()
    entry = {"name": config.server_name, "transport": browser_server_transport(config)}
    (validated,) = normalize_mcp_servers([entry])
    return dict(validated)


def browser_mcp_servers(env: Mapping[str, str] | None = None) -> tuple[dict[str, Any], ...]:
    """``()`` when the tool is off, else a one-element tuple holding the validated server entry."""
    if not browser_tool_enabled(env):
        return ()
    return (browser_mcp_server(browser_tool_config(env)),)


def with_browser_tool(
    servers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """``servers`` plus Codna's browser server when enabled — merged, then re-validated.

    The user's own server always wins: if the configuration already defines a server named ``browser``,
    that entry is left exactly as written and Codna adds nothing. Overwriting a server a user configured is
    a worse failure than a browser tool that is not the built-in one. The merged list goes back through
    ``normalize_mcp_servers``, so enabling the tool can never smuggle a config past the 16-server cap — an
    over-cap list is the validator's refusal, never a silently dropped server.
    """
    base = tuple(servers)
    extra = browser_mcp_servers(env)
    if extra and any(server.get("name") == DEFAULT_SERVER_NAME for server in base):
        extra = ()
    return normalize_mcp_servers((*base, *extra))


def browser_tool_state(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Honest status for ``codna mcp browser`` and the ``codna://capabilities`` resource.

    Four different facts, reported as four: ON and launchable; ON but ``npx`` is missing so it cannot
    start; OFF with the exact switch that turns it on; and an INVALID setting, reported as the error it is
    rather than as "off" — a user with a typo must see the typo.
    """
    try:
        enabled = browser_tool_enabled(env)
        config = browser_tool_config(env)
    except BrowserToolConfigError as exc:
        return {
            "enabled": False,
            "launchable": False,
            "reason": str(exc),
            "error": exc.as_dict()["error"],
            "enable_with": ENABLE_HINT,
        }
    npx = shutil.which("npx")
    state: dict[str, Any] = {
        "enabled": enabled,
        "launchable": False,
        "server_name": DEFAULT_SERVER_NAME,
        "package": config.package,
        "npx": npx or None,
        "tools": list(BROWSER_TOOL_NAMES),
        "required_playwright_tools": list(REQUIRED_PLAYWRIGHT_TOOLS),
    }
    if not enabled:
        state["reason"] = (
            f"the browser tool is off. Set {ENABLE_HINT} (or put it in the MCP client's `env` block) to "
            "give the agent a browser."
        )
        state["enable_with"] = ENABLE_HINT
        return state
    state["server"] = browser_mcp_server(config)
    state["args"] = browser_server_args(config)
    if npx is None:
        # Enabled but not launchable, said plainly: the alternative is an agent that reports a browser tool
        # it can never start.
        state["reason"] = (
            "the browser tool is on but `npx` is not on PATH. Install Node.js 18+, or point "
            "CODNA_BROWSER_PACKAGE at a local @playwright/mcp install — Codna does not bundle Node."
        )
        return state
    state["launchable"] = True
    state["reason"] = ""
    return state
