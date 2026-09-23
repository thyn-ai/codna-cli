"""`codna mcp install` — register codna's MCP server in a client's config in one command.

The website promises "add Codna to Cursor or Claude in one line"; this delivers it instead of asking
users to hand-edit ``mcpServers`` JSON. Merges a ``codna`` entry into the client's config, preserving
any existing servers. Supported clients: Cursor (``~/.cursor/mcp.json`` or ``./.cursor/mcp.json``) and
Claude Desktop (the OS-specific config path). Prints the resolved path + entry; never overwrites the
whole file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


class McpInstallError(Exception):
    """Surfaced to the user with remediation."""


def _cursor_path(*, project: bool) -> Path:
    if project:
        return Path.cwd() / ".cursor" / "mcp.json"
    return Path.home() / ".cursor" / "mcp.json"


def _claude_path(*, project: bool) -> Path:
    # Claude Desktop uses a single OS-specific config; there is no per-project variant.
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "Claude" / "claude_desktop_config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "Claude" / "claude_desktop_config.json"


def config_path(client: str, *, project: bool) -> Path:
    client = (client or "").strip().lower()
    if client == "cursor":
        return _cursor_path(project=project)
    if client == "claude":
        if project:
            raise McpInstallError("Claude Desktop has no per-project config; omit --project.")
        return _claude_path(project=project)
    raise McpInstallError(f"unknown client {client!r}; use 'cursor' or 'claude'.")


def server_entry(repo: str | None) -> dict:
    """The codna MCP server entry. `repo` (optional) sets the default repo for the tools."""
    args = ["mcp"]
    if repo:
        args += ["start", "--repo", repo]
    entry: dict = {"command": "codna", "args": args}
    return entry


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (json.JSONDecodeError, ValueError) as exc:
        raise McpInstallError(f"existing config at {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise McpInstallError(f"existing config at {path} is not a JSON object")
    return data


def install(client: str, *, project: bool = False, repo: str | None = None) -> dict:
    """Merge the codna server into the client config (preserving other servers). Returns a summary."""
    path = config_path(client, project=project)
    data = _load(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    replaced = "codna" in servers
    servers["codna"] = server_entry(repo)
    data["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".codna-tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic; preserves the rest of the file
    return {"client": client.lower(), "path": str(path), "replaced": replaced, "entry": servers["codna"]}
