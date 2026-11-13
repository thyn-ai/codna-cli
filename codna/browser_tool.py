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

**The entry carries its own ``timeout``, because without one the ENGINE decides.** A server entry with no
``timeout`` gets the engine's 1.5 s connect probe (``MCP_CONNECT_PROBE_TIMEOUT_MS`` in
``@cline/core``'s MCP client). A cold ``npx @playwright/mcp`` start measures ~1.3 s, so an unconfigured
entry loses that race intermittently — and the engine's response to a lost race is to log the server as
unavailable and build the session WITHOUT it. The session then has no browser tool at all, the model's
``browser_navigate`` is rejected as an unavailable tool, and the "browser check" verifies nothing: from
the outside, indistinguishable from a model that chose not to browse. This module therefore always emits
``timeout``. See ``DEFAULT_CONNECT_TIMEOUT_S`` for the value and why it is that value.

**Determinism.** Every value below is resolved from an explicit setting or a documented default; nothing
reads the clock or the filesystem to decide what to launch, so two launches of one configuration are
byte-identical. The output directory is ``$CODNA_RUNTIME_ROOT/browser`` (else ``~/.codna/browser``), like
every other piece of Codna's local state, so screenshots never land in the user's checkout.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Final, Mapping

from .durable.mcp import normalize_mcp_servers

__all__ = [
    "BROWSER_TOOL_ENV",
    "BROWSER_TOOL_NAMES",
    "CONNECT_TIMEOUT_ENV",
    "DEFAULT_CONNECT_TIMEOUT_S",
    "DEFAULT_SERVER_NAME",
    "ENABLE_HINT",
    "FILE_ACCESS_MODES",
    "MAX_CONNECT_TIMEOUT_S",
    "MIN_CONNECT_TIMEOUT_S",
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

#: How long one MCP exchange with the browser server may take, in seconds. The same shape as the other
#: ``CODNA_BROWSER_*`` settings: a documented default, and a value that is not an accepted number is a
#: REFUSAL rather than a guess. Unlike the Playwright flags beside it (``--timeout-action``,
#: ``--timeout-navigation``) this one is enforced by the MCP client that hosts the server, not by
#: Playwright — see ``DEFAULT_CONNECT_TIMEOUT_S``.
CONNECT_TIMEOUT_ENV: Final = "CODNA_BROWSER_CONNECT_TIMEOUT_S"

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

#: The MCP-level ``timeout`` the emitted server entry declares, in SECONDS (the engine's unit).
#:
#: **Why the entry must declare one at all.** The engine's MCP client uses a 1.5 s connect probe when a
#: server entry carries no ``timeout`` (``MCP_CONNECT_PROBE_TIMEOUT_MS``). A cold ``npx @playwright/mcp``
#: start measures ~1.3 s, so the entry loses that race intermittently; the engine then logs the server as
#: unavailable and builds the session with NO browser tools, which is indistinguishable from a model that
#: chose not to browse. An explicit value is what raises the probe.
#:
#: **Two concerns ride on this one field, and the value is chosen for the larger one.** The engine applies
#: the same number to the ``initialize`` handshake AND to every ``tools/call`` request (and to the agent
#: tool wrapper), so it is not only a connect bound. That is a real conflation in the engine, and it
#: cannot be split from here — so the value must clear BOTH:
#:
#:   * the connect bound: 180 s is ~138x the measured 1.3 s cold start and 120x the 1.5 s probe, which
#:     also covers the first launch on a machine where ``npx`` downloads the package. It is the same
#:     number ``browser_bridge.DEFAULT_STARTUP_TIMEOUT_S`` gives the same server to answer ``initialize``,
#:     so the in-process client and the engine-carried entry cannot disagree about how patient to be.
#:   * the call bound: a ``browser_navigate`` may legitimately run for Playwright's own navigation timeout
#:     (60 s by default) before it answers, so a 60 s transport bound would cut a successful navigation
#:     short. 180 s clears 60 s of navigation plus the 30 s margin the bridge allows a call.
#:
#: A value below the configured navigation timeout is therefore REFUSED rather than silently truncating
#: legitimate calls (see ``browser_tool_config``).
DEFAULT_CONNECT_TIMEOUT_S: Final = 180.0
#: The floor an override may not go under: ~7.7x the measured cold start and ~6.7x the engine's probe, so
#: no accepted value can lose the race this field exists to win.
MIN_CONNECT_TIMEOUT_S: Final = 10.0
#: The engine's own ceiling (``MAX_MCP_TIMEOUT_SECONDS``). An accepted value never exceeds it, so what
#: Codna writes and what the engine clamps to are the same number.
MAX_CONNECT_TIMEOUT_S: Final = 3_600.0

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
    connect_timeout_s: float

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


def _bounded_seconds(value: str | None, *, field: str, default: float,
                    minimum: float, maximum: float) -> float:
    """A positive, finite number of seconds inside ``[minimum, maximum]``, or a REFUSAL.

    Same contract as ``_positive_int`` and the same reason: an operator who exported a value believes it
    took effect, so a typo or an out-of-range number must be reported rather than replaced by the default.
    ``nan``/``inf`` are refused explicitly — ``float("nan")`` parses, and a NaN timeout would disable the
    bound it was set to establish.
    """
    text = (value or "").strip()
    if not text:
        return default
    expected = f"a number of seconds between {minimum:g} and {maximum:g}"
    try:
        parsed = float(text)
    except ValueError as exc:
        raise _invalid(field, value, expected) from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise _invalid(field, value, expected)
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

    action_timeout_ms = _positive_int(values.get("CODNA_BROWSER_ACTION_TIMEOUT_MS"),
                                      field="CODNA_BROWSER_ACTION_TIMEOUT_MS",
                                      default=DEFAULT_ACTION_TIMEOUT_MS)
    navigation_timeout_ms = _positive_int(values.get("CODNA_BROWSER_NAVIGATION_TIMEOUT_MS"),
                                          field="CODNA_BROWSER_NAVIGATION_TIMEOUT_MS",
                                          default=DEFAULT_NAVIGATION_TIMEOUT_MS)
    # The engine applies this ONE number to the handshake AND to every tool call, so it must clear both:
    # 180 s by default, and never below Playwright's own navigation budget — a lower bound would abort a
    # navigation that would have succeeded. The default rises with a raised navigation timeout rather than
    # refusing a configuration the user only made slower; an EXPLICIT value that cannot clear it is a
    # refusal, because that operator's setting cannot do what they asked.
    navigation_timeout_s = navigation_timeout_ms / 1000.0
    floor_s = max(MIN_CONNECT_TIMEOUT_S, navigation_timeout_s)
    if floor_s > MAX_CONNECT_TIMEOUT_S:
        # The navigation budget alone is past the engine's ceiling, so NO connect timeout can both clear it
        # and stay inside what the engine honours. Refuse rather than emit a number the engine clamps down
        # to something smaller than the navigation it must cover.
        raise _invalid(
            "CODNA_BROWSER_NAVIGATION_TIMEOUT_MS", navigation_timeout_ms,
            f"at most {MAX_CONNECT_TIMEOUT_S * 1000:g}ms, because the MCP timeout bounds one exchange and "
            f"the engine caps it at {MAX_CONNECT_TIMEOUT_S:g}s",
        )
    connect_timeout_s = _bounded_seconds(values.get(CONNECT_TIMEOUT_ENV), field=CONNECT_TIMEOUT_ENV,
                                         default=max(DEFAULT_CONNECT_TIMEOUT_S, floor_s),
                                         minimum=MIN_CONNECT_TIMEOUT_S, maximum=MAX_CONNECT_TIMEOUT_S)
    if connect_timeout_s < floor_s:
        raise _invalid(
            CONNECT_TIMEOUT_ENV, values.get(CONNECT_TIMEOUT_ENV),
            f"at least {floor_s:g}s — the navigation timeout "
            f"(CODNA_BROWSER_NAVIGATION_TIMEOUT_MS={navigation_timeout_ms}) must fit inside it",
        )

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
        action_timeout_ms=action_timeout_ms,
        navigation_timeout_ms=navigation_timeout_ms,
        output_max_size_bytes=_positive_int(values.get("CODNA_BROWSER_OUTPUT_MAX_BYTES"),
                                            field="CODNA_BROWSER_OUTPUT_MAX_BYTES",
                                            default=DEFAULT_OUTPUT_MAX_SIZE_BYTES),
        connect_timeout_s=connect_timeout_s,
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

    **``timeout`` is part of the entry, and its absence was a defect.** Without it the engine applies its
    1.5 s connect probe, which a cold ``npx @playwright/mcp`` start (~1.3 s) intermittently loses; the
    engine then drops the server and the session runs with no browser tools at all. See
    ``DEFAULT_CONNECT_TIMEOUT_S`` for the value and why it is that value. Emitted from the config, so it is
    a documented default or an explicit setting — never a clock read — and two calls with one configuration
    are byte-identical, which is what keeps the entry safe to record in history.
    """
    config = config or browser_tool_config()
    entry = {
        "name": config.server_name,
        "transport": browser_server_transport(config),
        "timeout": config.connect_timeout_s,
    }
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


#: The browser cache Playwright's own launcher populates, per platform. Only read when the user has not
#: pointed ``PLAYWRIGHT_BROWSERS_PATH`` somewhere else, which is Playwright's own override.
_PLAYWRIGHT_BROWSER_ROOTS: Final[dict[str, tuple[str, ...]]] = {
    "darwin": ("Library/Caches/ms-playwright",),
    "linux": (".cache/ms-playwright",),
    "win32": ("AppData/Local/ms-playwright",),
}

#: A downloaded Chromium, under whatever root Playwright uses. ``chromium_headless_shell`` is what a
#: ``--headless`` launch actually runs, so either family counts.
_CHROMIUM_DIR = re.compile(r"^chromium(?:_headless_shell)?-\d+$")


def _playwright_browser_roots(env: Mapping[str, str]) -> list[str]:
    """Where Playwright may have downloaded its browser, most specific first.

    ``PLAYWRIGHT_BROWSERS_PATH`` is Playwright's OWN override and is read when set — including the special
    ``"0"``, which means "inside the package" and therefore cannot be located without running npx. In that
    case the answer is "unknown", not "missing": inventing a failure here would be the silent-lie defect
    this function exists to avoid, pointed the other way.

    ``env`` IS THE WHOLE ANSWER, and ``os.environ`` is not consulted behind it. Every caller already passes
    ``_env_map(env)``, which IS ``os.environ`` when the argument is ``None``, so the ordinary case is
    unchanged; the extra fallback only made a caller that supplied an explicit mapping get the host's path
    for a variable its mapping did not name — a hidden dependency that a test could not pose, and that made
    ``browser_tool_state(env)`` answer differently on two machines with the same argument.
    """
    explicit = (env.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if explicit:
        return [] if explicit == "0" else [os.path.abspath(os.path.expanduser(explicit))]
    home = os.path.expanduser("~")
    relative = _PLAYWRIGHT_BROWSER_ROOTS.get(sys.platform if sys.platform != "linux2" else "linux", ())
    return [os.path.join(home, part) for part in relative]


def _chromium_cached(env: Mapping[str, str]) -> bool | None:
    """Whether a downloaded Chromium is present. ``None`` when it cannot be determined without launching.

    Deliberately NOT a claim that the browser WORKS: an empty cache is a first-run download, which succeeds
    online and fails offline. So this is reported as a warning with a code, and it never flips
    ``launchable`` — ``launchable`` answers "can the launcher start", which depends on ``npx`` alone.
    """
    roots = _playwright_browser_roots(env)
    if not roots:
        return None
    try:
        for root in roots:
            if not os.path.isdir(root):
                continue
            if any(_CHROMIUM_DIR.match(name) for name in os.listdir(root)):
                return True
    except OSError:
        return None
    return False


def browser_tool_state(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Honest status for ``codna mcp browser`` and the ``codna://capabilities`` resource.

    Four different facts, reported as four: ON and launchable; ON but ``npx`` is missing so it cannot
    start; OFF with the exact switch that turns it on; and an INVALID setting, reported as the error it is
    rather than as "off" — a user with a typo must see the typo.

    ``reason_code`` is the MACHINE half of ``reason``, following this repository's error convention: a
    surface that must branch ("show the install-Node hint", "show the first-run note") reads the code, and a
    human reads the sentence. It is ``""`` exactly when the tool is on and launchable.

    **The runtime requirement is stated, not implied.** The MCP server is ``@playwright/mcp`` (Node) and it
    launches Chromium, so ``npx`` and a browser are both needed and neither is bundled — Codna ships the
    Python LAUNCHER, not a browser. ``npx`` missing is blocking and coded. A missing browser cache is a coded
    WARNING rather than a refusal, because Playwright downloads it on first use: calling that "cannot start"
    would be wrong on every fresh install, and calling it silent would hide a real offline failure.
    """
    try:
        enabled = browser_tool_enabled(env)
        config = browser_tool_config(env)
    except BrowserToolConfigError as exc:
        return {
            "enabled": False,
            "launchable": False,
            "reason": str(exc),
            "reason_code": "browser_config_invalid",
            "warnings": [],
            "error": exc.as_dict()["error"],
            "enable_with": ENABLE_HINT,
        }
    values = _env_map(env)
    npx = shutil.which("npx")
    state: dict[str, Any] = {
        "enabled": enabled,
        "launchable": False,
        "reason_code": "",
        "warnings": [],
        "server_name": DEFAULT_SERVER_NAME,
        "package": config.package,
        "npx": npx or None,
        "tools": list(BROWSER_TOOL_NAMES),
        "required_playwright_tools": list(REQUIRED_PLAYWRIGHT_TOOLS),
        # The MCP-level bound the emitted entry declares. Reported because it is the setting whose absence
        # made a session silently carry no browser at all: an operator diagnosing "the browser check
        # verified nothing" needs to see the number the engine will actually apply.
        "connect_timeout_s": config.connect_timeout_s,
    }
    if not enabled:
        state["reason_code"] = "browser_tool_disabled"
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
        state["reason_code"] = "browser_tool_npx_missing"
        state["reason"] = (
            "the browser tool is on but `npx` is not on PATH. Install Node.js 18+, or point "
            "CODNA_BROWSER_PACKAGE at a local @playwright/mcp install — Codna does not bundle Node."
        )
        return state
    state["launchable"] = True
    state["reason"] = ""
    if not config.browser:
        # Only the BUNDLED Chromium needs the cache; `CODNA_BROWSER_NAME=chrome` uses the installed channel.
        cached = _chromium_cached(values)
        state["chromium"] = {"cached": cached}
        if cached is False:
            state["warnings"].append({
                "code": "browser_chromium_not_cached",
                "message": (
                    "no Playwright browser is in the local cache yet, so the FIRST launch downloads Chromium "
                    "through `npx` (needs network). After that it runs from the cache. If this machine is "
                    "offline, run `npx playwright install chromium` once, or set PLAYWRIGHT_BROWSERS_PATH."
                ),
            })
    return state
