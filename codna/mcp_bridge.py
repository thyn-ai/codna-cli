"""Bridge: hand the IDE backend the MCP server list, judged by the ONE durable validator.

**The gap this closes.** MCP worked on durable runs (`codna.durable.mcp`) and did nothing in the IDE
chat panel: `CodnaSessionSpec` had no MCP field, so the chat engine never received a server list even
though it accepts one at the request boundary. The IDE must now read a user's MCP configuration and
attach it to a chat session.

**Why a bridge and not a TypeScript parser.** `.codna/mcp.json`, `.cursor/mcp.json` (migration) and
Theia's own `ai-features.mcp.mcpServers` preference are merged and judged by exactly the rules
`codna.durable.mcp` already owns: three transports, a 16-server cap, `{"from_env": "NAME"}`
indirection, fail-closed on one bad entry. A second parser in TypeScript is the precise defect this
repository documents elsewhere (`algenta/server/policy-document.ts`: "ONE validator for sidecar and
worker from one matrix") — the two drift the first time either changes, and the drift is a server the
durable path accepts and the chat path silently drops, or the reverse. So the IDE asks this bridge,
which imports `normalize_mcp_servers` and does NOT restate any of its rules.

**What this module IS allowed to add.** A config file has a different SHAPE from a durable
`TaskSpec`: Cursor writes `{"mcpServers": {"<name>": {"command": …}}}` with no `transport` key, and
Theia writes `serverUrl`/`serverAuthToken`. Turning one shape into the other is a *format adapter*;
whether the result is a legal server is still decided by `normalize_mcp_servers`, which every path
funnels through before returning. Delete the adapter functions and no validation rule is lost.

**Secrets never become values.** A `TaskSpec` is history and a create-session body is a request, so
neither may carry a credential. Codna's own format therefore refuses a literal `env` value outright —
the durable validator already does, and accepting it here would make the chat path permit what a
durable run denies. Cursor's and Theia's formats *do* put literals in the file, and a migrating
repository must work unmodified; so a literal there is not trusted, it is replaced by a same-name
`{"from_env": NAME}` reference and reported as a warning. The value is read from the environment at
launch instead of from the file.

**A remote server that needs an auth header is REFUSED, and that is deliberate.** The engine accepts
only string header values, and a header value from a config file is a credential — sending it would
put it in the create-session body, and sending a reference instead is a 400 from the engine (verified
against both the shipped bundle and the TypeScript source: `mcp_servers.<name>.transport.env.<key>
must be a string`). There is no third option, so the refusal names the server and says why. A remote
server with NO auth works normally.

**Protocol (one JSON object in, one JSON object out).**

Request on stdin::

    {"sources": [{"path": "/abs/.codna/mcp.json", "kind": "codna"},
                 {"path": "/abs/.cursor/mcp.json", "kind": "cursor"},
                 {"path": "/abs/.theia/settings.json", "kind": "theia"}]}

Sources are ordered HIGHEST precedence first; the first file to define a name wins, so
`.codna/mcp.json` beating `.cursor/mcp.json` is the caller's list order rather than a rule hidden in
here. A source that does not exist is reported, not an error (most repositories have none).

Response on stdout::

    {"ok": true,
     "servers": [{"name": "docs", "transport": {...}}],
     "sources": [{"path": "...", "kind": "codna", "found": false, "servers": [], "warnings": []}],
     "warnings": ["..."]}

On a refusal the response is ``{"ok": false, "error": {"code": ..., "message": ..., "details": {}}}``
and the exit code is 1. FAIL CLOSED throughout: a config with one bad entry refuses the whole list,
because a session that silently ran three of four configured servers is a user who believes a tool
was available.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Final, Mapping

from .durable.mcp import McpConfigError, normalize_mcp_servers

__all__ = ["handle", "main", "SOURCE_KINDS"]

#: Which file formats this bridge adapts. Each means "how do I read `mcpServers` out of this file",
#: never "how do I validate a server".
SOURCE_KINDS: Final = ("codna", "cursor", "theia")

#: The Theia preference that IS its MCP server registry (`@theia/ai-mcp` `MCP_SERVERS_PREF`).
_THEIA_SERVERS_KEY: Final = "ai-features.mcp.mcpServers"


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


def _as_record(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _read_servers(path: str, kind: str) -> tuple[dict[str, Any] | None, str | None]:
    """The file's server map, or (None, reason). An absent file, or a file with none, is (None, None)."""
    if not path:
        return None, "an empty path was supplied"
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    if not raw.strip():
        return None, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"{path} is not valid JSON: {exc}"
    if not isinstance(parsed, Mapping):
        return None, f"{path} must contain a JSON object"

    if kind == "theia":
        # Theia writes every preference into one settings file, so the servers are namespaced under
        # `ai-features.mcp.mcpServers` rather than at the top level. Absent = Theia has no registry
        # entry, which is the ordinary case.
        servers = parsed.get(_THEIA_SERVERS_KEY)
    else:
        servers = parsed.get("mcpServers")
    if servers is None:
        # An empty object, or a file that simply has no servers yet. Not an error: `codna mcp install`
        # writes the key, and a hand-written file may not have reached that point.
        return None, None
    record = _as_record(servers)
    if record is None:
        return None, f"{path}: the MCP server map must be an object keyed by server name"
    return record, None


def _reference_or_literal(name: str, key: str, value: Any, warnings: list[str]) -> Any:
    """One `env` entry as a reference. A literal is converted, never forwarded as a value.

    This is the migration rule that lets a repository move from Cursor with **zero changes**: a file
    that says `"LINEAR_API_KEY": "lin_api_…"` keeps working, because the variable is read from the
    environment at launch instead of from the file. The literal is never returned to the caller, so it
    cannot reach the wire — and the warning says so, because silently ignoring a value the user wrote
    would be its own confusion.
    """
    if isinstance(value, Mapping):
        return dict(value)  # an explicit reference; the validator judges it
    if not isinstance(value, str):
        return value  # let the validator name the defect
    stripped = value.strip()
    if stripped.startswith("${") and stripped.endswith("}"):
        return stripped  # `${NAME}` — the validator normalizes it to a reference
    warnings.append(
        f"{name}.env.{key}: the literal value in this file is not used; the environment variable "
        f"{key} is read instead, so the credential never travels in a session request"
    )
    return {"from_env": key}


def _env_references(name: str, env: Any, warnings: list[str]) -> dict[str, Any]:
    record = _as_record(env)
    if record is None:
        return {}
    return {
        key: _reference_or_literal(name, key, value, warnings)
        for key, value in record.items()
        if isinstance(key, str)
    }


def _transport_from_shorthand(
    name: str,
    entry: Mapping[str, Any],
    *,
    migrate: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """A Cursor/Theia entry as a canonical `transport`, which the validator then judges.

    Cursor and Theia spell a local server as `{"command", "args", "env"}` with NO discriminant, and a
    remote one as `{"url": …}` / `{"serverUrl": …}`. Neither carries `type`, so it is DERIVED here —
    that derivation is the adapter's whole job.
    """
    command = entry.get("command")
    if isinstance(command, str) and command.strip():
        transport: dict[str, Any] = {"type": "stdio", "command": command}
        if isinstance(entry.get("args"), list):
            transport["args"] = list(entry["args"])
        cwd = entry.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            transport["cwd"] = cwd
        if "env" in entry:
            transport["env"] = _env_references(name, entry["env"], warnings) if migrate else entry["env"]
        return transport

    url = entry.get("url", entry.get("serverUrl"))
    declared = entry.get("type")
    kind = declared if declared in ("sse", "streamableHttp") else "streamableHttp"
    transport = {"type": kind, "url": url}
    headers = _as_record(entry.get("headers"))
    if headers:
        transport["headers"] = headers
    return transport


def _refuse_if_remote_auth(name: str, entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Refuse a remote server whose credential would have to travel in the request body.

    A remote transport's only credential channel is a header, and the engine accepts only STRING header
    values; a `{"from_env": …}` reference is refused by the engine at the create boundary (400, verified
    against both the shipped bundle and the TypeScript source). So there is no shape that carries this
    server's header without putting it on the wire, and a header read from a config file is a
    credential. Refusing is the honest answer.

    Every header is treated as a credential because this layer cannot tell one from `X-Client: codna`,
    and guessing wrong leaks the one that mattered. An auth-less remote server is unaffected, and a
    LOCAL (stdio) server carries its credentials through `env`, which does have a reference form.

    Checked on the CANONICAL transport, so a `{"transport": {"headers": …}}` written directly into a
    Codna file is refused by the same rule as Theia's `serverAuthToken` shorthand. `normalize_mcp_servers`
    would accept those header strings, which is exactly why this decision cannot be left to it.
    """
    transport = entry.get("transport")
    body = _as_record(transport) if isinstance(transport, Mapping) else None
    if body is None:
        if entry.get("command"):
            return None
        has_token = isinstance(entry.get("serverAuthToken"), str) and bool(str(entry.get("serverAuthToken")).strip())
        headers = entry.get("headers")
        carries_auth = has_token or (isinstance(headers, Mapping) and len(headers) > 0)
    else:
        if body.get("type") == "stdio":
            return None
        headers = body.get("headers")
        carries_auth = isinstance(headers, Mapping) and len(headers) > 0
    if carries_auth:
        return _error(
            "mcp_remote_auth_unsupported",
            f"MCP server {name!r} is remote and carries an auth header. A header value read from a "
            "config file would have to be serialized into the create-session body, and the engine "
            "accepts string header values only (it refuses an env reference). Configure a LOCAL server "
            "for this integration instead — its credentials travel as environment references — or use "
            "a server that authenticates over OAuth.",
            server=name,
        )
    return None


def _canonical(name: str, entry: Mapping[str, Any], kind: str, warnings: list[str]) -> dict[str, Any] | None:
    """One config entry as a canonical server, or an error dict's replacement (see `handle`)."""
    if "transport" in entry:
        # Already canonical. Passed through untouched so the validator, not this function, decides.
        transport: Any = _as_record(entry["transport"]) or entry["transport"]
    else:
        transport = _transport_from_shorthand(name, entry, migrate=kind in ("cursor", "theia"), warnings=warnings)
    canonical: dict[str, Any] = {"transport": transport}
    if entry.get("disabled") is True:
        canonical["disabled"] = True
    if "timeout" in entry:
        canonical["timeout"] = entry["timeout"]
    return canonical


def handle(request: object) -> dict[str, Any]:
    """Answer one bridge request. Pure apart from reading the files it is told about."""
    if not isinstance(request, Mapping):
        return _error("mcp_bridge_bad_request", "the request must be a JSON object")

    raw_sources = request.get("sources")
    if not isinstance(raw_sources, list):
        return _error("mcp_bridge_bad_request", "`sources` must be a list of {path, kind} objects")

    sources: list[dict[str, Any]] = []
    merged: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for index, raw in enumerate(raw_sources):
        entry = _as_record(raw)
        if entry is None:
            return _error(
                "mcp_bridge_bad_request",
                f"sources[{index}] must be an object with `path` and `kind`",
            )
        path = entry.get("path")
        kind = entry.get("kind")
        if not isinstance(path, str):
            return _error("mcp_bridge_bad_request", f"sources[{index}].path must be a string")
        if kind not in SOURCE_KINDS:
            return _error(
                "mcp_bridge_bad_request",
                f"sources[{index}].kind must be one of {', '.join(SOURCE_KINDS)} (got {kind!r})",
            )
        record, problem = _read_servers(path, kind)
        if problem is not None:
            return _error("mcp_config_unreadable", problem, path=path)
        source_report: dict[str, Any] = {
            "path": path,
            "kind": kind,
            "found": bool(record),
            "servers": [],
            "warnings": [],
        }
        if not record:
            source_report["skipped"] = "no MCP servers are configured in this file"
            sources.append(source_report)
            continue

        local_warnings = source_report["warnings"]
        for name, body in record.items():
            if not isinstance(name, str):
                return _error("mcp_config_invalid", f"{path} has a non-string server name", path=path)
            if name in merged:
                # FIRST WINS. A name defined by a higher-precedence file is not overwritten; say which
                # file won so a user with two configs is not left guessing which one took effect.
                local_warnings.append(
                    f"{name} is also defined by a higher-precedence file and is ignored here"
                )
                continue
            server = _as_record(body)
            if server is None:
                return _error(
                    "mcp_config_invalid",
                    f"{path}: `{name}` must be an object",
                    path=path,
                    server=name,
                )
            refusal = _refuse_if_remote_auth(name, server)
            if refusal is not None:
                return refusal
            canonical = _canonical(name, server, str(kind), local_warnings)
            if canonical is None:  # pragma: no cover - `_canonical` refuses by returning an error above
                return _error("mcp_config_invalid", f"{path}: {name} is not usable", path=path)
            merged[name] = canonical
            source_report["servers"].append(name)
        source_report["warnings"] = local_warnings
        warnings.extend(local_warnings)
        sources.append(source_report)

    # ONE validator, applied to the merged result. Everything above produced a shape; only this call
    # decides whether the shape is legal, and its refusal is returned verbatim.
    try:
        servers = normalize_mcp_servers(merged)
    except McpConfigError as exc:
        return {"ok": False, "error": {"code": exc.code, "message": str(exc), "details": exc.details}}

    return {
        "ok": True,
        "servers": [dict(server) for server in servers],
        "sources": sources,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    """Read one request from stdin, write one response to stdout. Returns the exit code."""
    del argv  # stdin/stdout only: flags would be a second, drifting interface
    try:
        raw = sys.stdin.read()
    except OSError as exc:  # pragma: no cover - stdin closed by the parent
        print(json.dumps(_error("mcp_bridge_stdin_failed", str(exc))), file=sys.stdout)
        return 1
    try:
        request = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps(_error("mcp_bridge_bad_json", f"cannot parse the request: {exc}")), file=sys.stdout)
        return 1
    try:
        response = handle(request)
    except Exception as exc:  # noqa: BLE001 - a bridge must answer, never traceback into the host
        response = _error("mcp_bridge_failed", f"{type(exc).__name__}: {exc}")
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    sys.exit(main())
