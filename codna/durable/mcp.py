"""MCP servers on the durable engine path (D5) — the config that travels, and what reaches the session.

The gap this closes, stated as it was: the engine's `SessionRecord` had no MCP field, so
`mcp_attachment_state` returned a refusal and a run recorded in its ledger that MCP was not carried.
That was honest but incomplete. The engine now accepts `mcp_servers` on a create request and turns
each server into REAL tools (`algenta/mcp-servers.ts`), so a durable run can be handed one.

**What travels in workflow history, and what does not.** A `TaskSpec` is history (and therefore the
ledger), so it may name an MCP server, its transport, and the NAMES of the environment variables that
carry its secrets — never the secret values. A stdio server's `env` is exactly the kind of value that
must not be serialized, so the spec accepts it only as an `{"from_env": "NAME"}` reference (or a
`"${NAME}"` string, which is normalized to the same thing). The activity expands each reference from
the WORKER's environment at the moment it creates the session, and a reference whose variable is unset
is a REFUSAL with the variable named — never a server that starts without the credential it needed.

**Why this mirrors `provider.py`.** The provider key rule is the same rule: the spec names the SOURCE
of a credential, the activity resolves the value, and a value never enters history. One difference is
deliberate: an unset provider key lets the engine read its own environment (the historical behaviour),
while an unset MCP variable referenced by a server is refused, because there is no engine-side
equivalent to fall back on — the server would simply fail to authenticate at connect time, which is a
run that silently lacks a tool its operator configured.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Final, Mapping

__all__ = [
    "MAX_MCP_SERVERS",
    "McpAttachment",
    "McpConfigError",
    "expand_mcp_servers",
    "mcp_attachment_state",
    "normalize_mcp_servers",
]

# Mirrors the engine's own cap (`algenta/mcp-servers.ts`). Kept identical so a spec this side accepts
# is never refused at the request boundary for size alone.
MAX_MCP_SERVERS: Final = 16

# The same character class the engine enforces for a server name, which becomes a tool-name prefix.
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# An environment variable NAME, as it may appear in an `from_env` reference.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TRANSPORTS: Final = ("stdio", "sse", "streamableHttp")
_TRANSPORT_FIELDS: Final = {
    "stdio": ("type", "command", "args", "cwd", "env"),
    "sse": ("type", "url", "headers"),
    "streamableHttp": ("type", "url", "headers"),
}


class McpConfigError(ValueError):
    """A durable run's MCP config that must not be silently tolerated. Always carries a ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


class McpAttachment:
    """Whether this run's turns carry MCP servers, and if not, exactly why.

    A pair rather than a bool because "we did not attach it" and "we could not" are different answers,
    and only the second one is worth telling an operator about at run time.
    """

    __slots__ = ("attached", "reason", "servers")

    def __init__(self, attached: bool, reason: str = "", servers: tuple[str, ...] = ()) -> None:
        self.attached = attached
        self.reason = reason
        self.servers = servers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, McpAttachment):
            return NotImplemented
        return (self.attached, self.reason, self.servers) == (
            other.attached,
            other.reason,
            other.servers,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"McpAttachment(attached={self.attached!r}, reason={self.reason!r}, "
            f"servers={self.servers!r})"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "attached": self.attached,
            "reason": self.reason,
            "servers": list(self.servers),
        }


def _fail(message: str, *, field: str, value: Any = None) -> "McpConfigError":
    details: dict[str, Any] = {"field": field}
    if value is not None:
        details["value"] = repr(value)[:200]
    return McpConfigError(message, code="mcp_config_invalid", details=details)


def _normalize_env_references(value: Any, field: str) -> dict[str, dict[str, str]]:
    """A server's `env`, as references. Only a `from_env` reference is representable here.

    A literal value is REFUSED rather than accepted-and-serialized: the spec is history, and a
    credential written into history is the one thing the provider rule exists to prevent. An operator
    who wants a literal puts it on the server's own launch configuration, not in a run.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object of environment references", field=field)
    output: dict[str, dict[str, str]] = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not _ENV_NAME.match(name):
            raise _fail(f"{field} has an invalid variable name", field=f"{field}.{name}", value=name)
        if isinstance(entry, str):
            # `NAME: "${OTHER_NAME}"` — the shorthand the IDE's own MCP config uses.
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", entry.strip())
            if not match:
                raise _fail(
                    f"{field}.{name} must be an {{\"from_env\": \"NAME\"}} reference or \"${{NAME}}\"; "
                    "a literal value may not enter workflow history",
                    field=f"{field}.{name}",
                )
            output[name] = {"from_env": match.group(1)}
            continue
        if not isinstance(entry, Mapping):
            raise _fail(f"{field}.{name} must be an object or a ${{NAME}} string", field=f"{field}.{name}")
        extra = set(entry) - {"from_env"}
        if extra:
            raise _fail(
                f"{field}.{name} carries unsupported keys: {', '.join(sorted(extra))}",
                field=f"{field}.{name}",
            )
        source = entry.get("from_env")
        if not isinstance(source, str) or not _ENV_NAME.match(source):
            raise _fail(
                f"{field}.{name}.from_env must be an environment variable name",
                field=f"{field}.{name}.from_env",
            )
        output[name] = {"from_env": source}
    return output


def _normalize_transport(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be an object with a type", field=field)
    kind = value.get("type")
    if kind not in _TRANSPORTS:
        raise _fail(
            f"{field}.type must be one of: {', '.join(_TRANSPORTS)}",
            field=f"{field}.type",
            value=kind,
        )
    allowed = _TRANSPORT_FIELDS[kind]
    extra = set(value) - set(allowed)
    if extra:
        raise _fail(
            f"{field} carries fields a {kind} transport does not have: {', '.join(sorted(extra))}",
            field=field,
        )
    if kind == "stdio":
        command = value.get("command")
        if not isinstance(command, str) or not command.strip():
            raise _fail(f"{field}.command must be a non-empty string", field=f"{field}.command")
        args = value.get("args", [])
        if not isinstance(args, (list, tuple)) or not all(isinstance(a, str) for a in args):
            raise _fail(f"{field}.args must be a list of strings", field=f"{field}.args")
        cwd = value.get("cwd")
        if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
            raise _fail(f"{field}.cwd must be a non-empty string when present", field=f"{field}.cwd")
        transport: dict[str, Any] = {
            "type": "stdio",
            "command": command.strip(),
            "args": list(args),
        }
        if cwd is not None:
            transport["cwd"] = cwd.strip()
        env = _normalize_env_references(value.get("env"), f"{field}.env")
        if env:
            transport["env"] = env
        return transport

    url = value.get("url")
    if not isinstance(url, str) or not url.strip():
        raise _fail(f"{field}.url must be a non-empty string", field=f"{field}.url")
    if not url.strip().lower().startswith(("http://", "https://")):
        raise _fail(f"{field}.url must be an absolute http(s) URL", field=f"{field}.url", value=url)
    headers = value.get("headers")
    if headers is not None:
        if not isinstance(headers, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise _fail(f"{field}.headers must be an object of strings", field=f"{field}.headers")
    transport = {"type": kind, "url": url.strip()}
    if headers:
        transport["headers"] = dict(headers)
    return transport


def normalize_mcp_servers(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate a spec's ``mcp_servers`` into a canonical, history-safe tuple. Raises on any defect.

    Accepted shapes are the same two the engine accepts — an object keyed by server name, or an array
    of named servers — so a caller can use whichever its configuration already has. A DISABLED server
    is kept in the tuple (it is configuration, and dropping it would make the spec disagree with what
    the operator wrote), while `expand_mcp_servers` is what omits it from the request.

    Fail closed throughout: one bad server fails the whole config, because a run that silently used
    3 of 4 servers is a run whose operator believes a tool was available when it was not.
    """
    if value is None:
        return ()
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif isinstance(value, (list, tuple)):
        entries = []
        for index, entry in enumerate(value):
            if not isinstance(entry, Mapping):
                raise _fail(f"mcp_servers[{index}] must be an object", field=f"mcp_servers[{index}]")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise _fail(
                    f"mcp_servers[{index}].name must be a non-empty string",
                    field=f"mcp_servers[{index}].name",
                )
            rest = {k: v for k, v in entry.items() if k != "name"}
            entries.append((name.strip(), rest))
    else:
        raise _fail(
            "mcp_servers must be an object keyed by server name, or an array of named servers",
            field="mcp_servers",
        )

    if len(entries) > MAX_MCP_SERVERS:
        raise _fail(
            f"at most {MAX_MCP_SERVERS} MCP servers may be attached to one run (got {len(entries)})",
            field="mcp_servers",
        )

    seen: set[str] = set()
    servers: list[dict[str, Any]] = []
    for name, body in entries:
        if not isinstance(name, str) or not _SERVER_NAME.match(name):
            raise _fail(
                "an MCP server name must match ^[A-Za-z0-9_-]{1,64}$ (it prefixes its tools)",
                field="mcp_servers",
                value=name,
            )
        if name in seen:
            raise _fail(f"duplicate MCP server name {name!r}", field="mcp_servers")
        seen.add(name)
        if not isinstance(body, Mapping):
            raise _fail(
                f"mcp_servers.{name} must be an object with a transport",
                field=f"mcp_servers.{name}",
            )
        extra = set(body) - {"transport", "disabled", "timeout"}
        if extra:
            raise _fail(
                f"mcp_servers.{name} carries unsupported keys: {', '.join(sorted(extra))}",
                field=f"mcp_servers.{name}",
            )
        disabled = body.get("disabled", False)
        if not isinstance(disabled, bool):
            raise _fail(f"mcp_servers.{name}.disabled must be a boolean", field=f"mcp_servers.{name}.disabled")
        timeout = body.get("timeout")
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not timeout > 0
        ):
            raise _fail(
                f"mcp_servers.{name}.timeout must be a positive number of seconds",
                field=f"mcp_servers.{name}.timeout",
            )
        server: dict[str, Any] = {
            "name": name,
            "transport": _normalize_transport(body.get("transport"), f"mcp_servers.{name}.transport"),
        }
        if disabled:
            server["disabled"] = True
        if timeout is not None:
            server["timeout"] = timeout
        servers.append(server)
    return tuple(servers)


def expand_mcp_servers(
    servers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    env: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Expand each server's ``from_env`` references from ``env`` into the create request's shape.

    Raises `McpConfigError` naming the missing VARIABLE — never the server silently starting without
    it. Disabled servers are omitted: the request carries what should run, and a disabled server is
    configuration the operator kept, not something to launch.
    """
    resolved: list[dict[str, Any]] = []
    for server in servers:
        if server.get("disabled") is True:
            continue
        transport = dict(server["transport"])
        references = transport.pop("env", None)
        if references:
            values: dict[str, str] = {}
            for target, reference in references.items():
                source = reference["from_env"]
                value = env.get(source)
                if not value:
                    raise McpConfigError(
                        f"MCP server {server['name']!r} needs environment variable {source}, "
                        "which is not set on this worker",
                        code="mcp_env_missing",
                        details={"server": server["name"], "variable": source},
                    )
                values[target] = value
            transport["env"] = values
        entry: dict[str, Any] = {"name": server["name"], "transport": transport}
        if "timeout" in server:
            entry["timeout"] = server["timeout"]
        resolved.append(entry)
    return resolved


# -- the honest state, reported into the run's ledger -------------------------------------------


def mcp_attachment_state(context: Any = None) -> McpAttachment:
    """Whether a durable turn can carry MCP servers, and what this run's config amounts to.

    Three distinct answers, and the difference matters to an operator:

    * **attached** — the run configured servers and the engine path carries them.
    * **nothing configured** — the ordinary case. Reported as *not attached* with the reason naming the
      absence, because "no server was configured" and "a server was configured and dropped" are
      different facts and only the second one is a defect.
    * **configured but disabled** — every server in the spec is switched off, so there is nothing to
      attach while the configuration still exists.
    """
    raw = getattr(context, "mcp_servers", ()) if context is not None else ()
    servers = tuple(server for server in (raw or ()) if server.get("disabled") is not True)
    if servers:
        names = tuple(str(server["name"]) for server in servers)
        return McpAttachment(
            attached=True,
            reason="",
            servers=names,
        )
    if raw:
        return McpAttachment(
            attached=False,
            reason="every configured MCP server is disabled, so the run attaches none",
        )
    return McpAttachment(
        attached=False,
        reason=(
            "no MCP servers are configured for this run. The engine path DOES accept `mcp_servers` "
            "on a create-session request and turns each server into callable tools, so a run that "
            "needs one sets `mcp_servers` on its TaskSpec (or CODNA_MCP_SERVERS on the worker)."
        ),
    )


def mcp_servers_from_env(env: Mapping[str, str] | None = None) -> tuple[dict[str, Any], ...]:
    """A worker-wide default MCP config from ``CODNA_MCP_SERVERS`` (JSON), or ``()``.

    The same object/array shapes the spec accepts. A malformed value RAISES rather than being ignored:
    an operator who exported a broken JSON blob must learn at start-up, not from a run that quietly
    lacked its tools. Validated through `normalize_mcp_servers`, so both entry points share one rule.
    """
    import json

    source = (env or os.environ).get("CODNA_MCP_SERVERS", "").strip()
    if not source:
        return ()
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise McpConfigError(
            f"CODNA_MCP_SERVERS is not valid JSON: {exc}",
            code="mcp_config_invalid",
            details={"variable": "CODNA_MCP_SERVERS"},
        ) from exc
    return normalize_mcp_servers(parsed)


def session_mcp_servers(
    context: Any,
    env: Mapping[str, str],
) -> list[dict[str, Any]]:
    """This TURN's MCP servers, references expanded, or ``[]`` when the run configured none.

    The spec's list wins; ``CODNA_MCP_SERVERS`` on the worker is the default for a run that did not name
    its own. Both go through the same normalizer and the same expansion, so the two entry points cannot
    diverge — and an unset variable raises `McpConfigError` naming it rather than producing a session
    whose server starts without the credential it needed.

    Lives here rather than in `activity_support` because both halves are MCP's own rule (what is
    representable, and where a credential comes from); the activity only decides to CALL it.

    **Codna's own browser server is added LAST, when ``CODNA_BROWSER_TOOL=1``.** It is added here rather
    than in a caller because every durable turn reaches its servers through this function, and a browser
    that existed on one durable path and not another is the "built but never joined" defect this
    repository keeps finding. The entry goes through the SAME ``expand_mcp_servers`` as every declared
    server, so the returned list is one shape rather than a mix; it carries no credentials today, so
    that expansion resolves nothing, and a run with the switch off is byte-identical to before.

    **Two error families, one per source.** A fault in the RUN's declared servers is a `McpConfigError`
    (unchanged). A fault in the BUILT-IN browser entry is a `BrowserToolError`, because it is the browser
    tool's own switch/config that produced it and a caller catching that family must not be handed a
    foreign type. Without the distinction, expanding the built-in entry was the one path where an
    operator's browser setting could surface as an MCP-config error.
    """
    from ..browser_tool import browser_mcp_servers

    declared = getattr(context, "mcp_servers", ())
    servers = declared or mcp_servers_from_env(env)
    resolved = expand_mcp_servers(servers, env) if servers else []
    builtin = _expanded_builtin_browser_servers(browser_mcp_servers, env)
    if not builtin:
        return resolved
    named = {str(server.get("name")) for server in resolved}
    return [*resolved, *(server for server in builtin if server["name"] not in named)]


def _expanded_builtin_browser_servers(
    browser_mcp_servers: Callable[[Mapping[str, str] | None], tuple[dict[str, Any], ...]],
    env: Mapping[str, str],
) -> list[dict[str, Any]]:
    """The built-in browser entry, expanded, with any fault kept in the BROWSER error family.

    EXPANDED rather than appended in its normalized shape: the list this feeds is one shape, and an entry
    that skipped the expander would be the one place a future ``from_env`` reference could reach a
    create-session body unsubstituted.

    ``expand_mcp_servers`` raises `McpConfigError`; for THIS entry that type would be wrong, because the
    reference came from Codna's own browser config rather than from the run's declared servers. The code
    and details are carried across unchanged, so the message still names the variable — only the family
    a caller switches on is corrected.
    """
    from ..browser_bridge import BrowserToolError

    entries = browser_mcp_servers(env)
    if not entries:
        return []
    try:
        return expand_mcp_servers(entries, env)
    except McpConfigError as exc:
        raise BrowserToolError(
            f"the built-in browser server could not be resolved: {exc}",
            code=exc.code,
            details=dict(exc.details),
        ) from exc


def mcp_session_spec(
    context: Any,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """The ``mcp_servers`` key for a create-session request, or ``{}`` when there are none.

    Returns a SPEC FRAGMENT rather than a value so the caller cannot accidentally send
    ``{"mcp_servers": []}`` for a run that has no servers: an absent key is the answer for every run
    that never configured MCP, and it is what keeps the pre-feature request byte-identical.

    A `McpConfigError` is RAISED, not returned, so the activity can translate it into its own error
    family with the code preserved — the one thing a caller must not do is proceed without the server.
    """
    servers = session_mcp_servers(context, env)
    return {"mcp_servers": servers} if servers else {}
