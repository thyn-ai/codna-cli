"""CLI entrypoint for Codna's MCP server channel."""
from __future__ import annotations

import json
import os
import sys


def _die(message: str) -> None:
    print(f"codna: {message}", file=sys.stderr)
    raise SystemExit(1)


def cmd_mcp(args, die=None) -> int:
    die = die or _die
    action = getattr(args, "action", "start") or "start"
    if action == "browser":
        # Status only. It reads the switch and the launch configuration and prints them; it never starts
        # the browser, so it is safe to run anywhere — including CI and a machine with no Node.
        from .browser_tool import browser_tool_state

        state = browser_tool_state()
        print(json.dumps(state, indent=2))
        # A configured-but-invalid setting is a failure: the user asked what is wrong and "off" is not the
        # answer. "Off" and "on but missing npx" are both successful answers to the question.
        return 1 if state.get("error") else 0
    if action == "install":
        from .mcp_install import McpInstallError, install

        client = getattr(args, "client", None)
        if not client:
            die("`codna mcp install` needs --client cursor|claude.")
            return 1
        try:
            summary = install(client, project=getattr(args, "project", False), repo=getattr(args, "repo", None))
        except McpInstallError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps({"ok": True, **summary}, indent=2))
        return 0

    # The MCP server needs the optional `mcp` package (extra), which pulls a web stack we keep out of
    # the base install. Check up front so users get a clear hint instead of a raw ModuleNotFoundError.
    # (`codna mcp install` above only writes client config and never reaches here.)
    import importlib.util

    if importlib.util.find_spec("mcp") is None:
        die("MCP support isn't installed. Add it with:  pip install 'codna[mcp]'   then re-run `codna mcp`.")
        return 1

    repo = getattr(args, "repo", None)
    if repo:
        local = os.path.abspath(os.path.expanduser(repo))
        os.environ["CODNA_MCP_DEFAULT_REPO"] = local if os.path.isdir(local) else repo
    from .mcp_server import main as mcp_main

    mcp_main()
    return 0
