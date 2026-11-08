"""``codna agents`` — see exactly which subagents a turn would carry, and why one did not load.

The sibling of ``codna rules`` and ``codna skills``, and it exists for the same reason: the defect it
answers is INVISIBLE. A subagent file that is never discovered, or is discovered and shadowed by a
higher-precedence file, or is REFUSED for bad frontmatter, produces no error anywhere — the user
simply watches an agent do nothing. So every question a user can ask about the roster is answered here
from the same ``codna.rules`` call the create-session path uses, and the command cannot drift from the
behaviour it documents.

  * ``codna agents``                       — every subagent discovered, its scope, model and restrictions
  * ``codna agents --agent verifier``      — the system prompt of one subagent (the delegation pull)
  * ``codna agents --budget 400``          — what fits the catalogue, what was dropped, and by how much
  * ``--json``                             — the same answer, structured, for tooling

Exit codes follow the CLI's convention (0 success, 1 refusal). A REFUSED file is reported on stderr
and is not a failure of the command — but ``--strict`` turns a refusal into exit 1 for CI, because a
roster that silently lost a file is the defect this exists to catch.
"""
from __future__ import annotations

import json
import sys

from .rules import Agent, AgentDiscovery, agent_catalogue, discover_agents, load_agent, readonly_profile

#: The default catalogue ceiling for the STANDALONE command. The prompt path passes its own remainder,
#: because there the roster shares one budget with the rules and the skills catalogue.
DEFAULT_CATALOGUE_BYTES = 2_048


def _agent_json(agent: Agent) -> dict:
    return {
        "name": agent.name,
        "description": agent.description,
        "scope": agent.scope.value,
        "source_path": agent.source_path,
        "model": agent.model,
        "readonly": agent.readonly,
        # The PROFILE, not just the boolean: what a caller actually sends to the engine, so a surface
        # can show "this runs read-only" without re-deriving it from `readonly` and drifting.
        "approval_profile": readonly_profile(agent.readonly),
        "bytes": len(agent.body.encode("utf-8")),
    }


def _print_human(discovery: AgentDiscovery, result, max_bytes: int) -> None:
    print(f"   catalogue:  {result.used_bytes}/{max_bytes} bytes")
    print()

    if not discovery.agents:
        print("no subagents found (looked for *.md under .cursor/agents, .claude/agents, .codex/agents, "
              "and the same three under ~)")
        return

    offered = {agent.name for agent in result.catalogued}
    dropped = {entry.name for entry in result.dropped}

    print("SUBAGENTS")
    for agent in discovery.agents:
        state = "over budget" if agent.name in dropped else (
            "CATALOGUED" if agent.name in offered else "not catalogued"
        )
        profile = readonly_profile(agent.readonly) or "inherits session"
        print(f"  [{state:<14}] {agent.scope.value:<12} {agent.source_path}"
              f"  (name: {agent.name}, model: {agent.model}, profile: {profile})")

    if result.dropped:
        print()
        print("NOT CATALOGUED")
        for entry in result.dropped:
            print(f"  [budget      ] {entry.source_path} (name: {entry.name}, {entry.bytes} bytes)")

    if discovery.refusals:
        print()
        print("REFUSED (these files did NOT become subagents)")
        for refusal in discovery.refusals:
            print(f"  [{refusal.code}] {refusal.source_path}: {refusal.message}")

    print()
    print("CATALOGUE THIS TURN WOULD CARRY (bodies are pulled by name)")
    print(result.text if result.text else "(empty)")


def run_agents(args) -> int:
    """Implementation of ``codna agents``. Returns the process exit code."""
    repo = args.repo
    max_bytes = args.budget if args.budget is not None else DEFAULT_CATALOGUE_BYTES
    if max_bytes < 0:
        print(json.dumps({"error": {"code": "agents_budget_invalid",
                                    "message": "--budget must be >= 0", "details": {}}}),
              file=sys.stderr)
        return 1

    # Pull ONE agent: the mechanism a delegation needs, and what a caller reads to know the prompt.
    if args.agent:
        agent = load_agent(repo, args.agent)
        if agent is None:
            print(json.dumps({
                "error": {
                    "code": "agent_not_found",
                    "message": f"no subagent named {args.agent!r}",
                    "details": {"repo": repo, "requested": args.agent},
                },
            }, indent=2), file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps({**_agent_json(agent), "body": agent.body}, indent=2))
        else:
            profile = readonly_profile(agent.readonly) or "inherits session"
            print(f"# {agent.source_path}  (name: {agent.name}, model: {agent.model}, "
                  f"profile: {profile})")
            print(f"# description: {agent.description or '(none)'}")
            print(agent.body)
        return 0

    discovery = discover_agents(repo)
    result = agent_catalogue(discovery, max_bytes=max_bytes)
    if args.as_json:
        print(json.dumps({
            "budget_bytes": result.budget_bytes,
            "used_bytes": result.used_bytes,
            "discovered": [_agent_json(agent) for agent in discovery.agents],
            "catalogued": [_agent_json(agent) for agent in result.catalogued],
            "dropped": [
                {"name": entry.name, "source_path": entry.source_path,
                 "reason": "budget", "bytes_needed": entry.bytes}
                for entry in result.dropped
            ],
            "refusals": [
                {"code": refusal.code, "source_path": refusal.source_path,
                 "message": refusal.message}
                for refusal in discovery.refusals
            ],
            "warnings": [
                {"code": warning.code, "source_path": warning.source_path,
                 "message": warning.message}
                for warning in discovery.warnings
            ],
            "text": result.text,
        }, indent=2))
    else:
        _print_human(discovery, result, max_bytes)

    # Refusals and read-time warnings are reported on stderr in BOTH formats, because a machine
    # consumer that ignores the JSON account must still leave a trace a human sees.
    for refusal in discovery.refusals:
        print(f"codna: error: [agents] {refusal.code}: {refusal.source_path}: {refusal.message}",
              file=sys.stderr)
    for warning in discovery.warnings:
        print(f"codna: warning: [agents] {warning.code}: {warning.source_path}: {warning.message}",
              file=sys.stderr)
    if result.dropped:
        names = ", ".join(f"{entry.name} ({entry.source_path})" for entry in result.dropped)
        print(f"codna: warning: [agents] {len(result.dropped)} subagent(s) exceeded the "
              f"{max_bytes}-byte catalogue budget and were NOT catalogued: {names}",
              file=sys.stderr)
    if args.strict and (result.dropped or discovery.refusals):
        return 1
    return 0


def register(sub) -> None:
    """Add ``codna agents`` to the root parser."""
    parser = sub.add_parser(
        "agents",
        help="Show which subagents (.cursor/agents, .claude/agents, .codex/agents, …) a turn would carry.",
    )
    parser.add_argument("repo", nargs="?", default=".", help="repository path (default: current dir)")
    parser.add_argument("--agent", help="print one subagent's system prompt by name")
    parser.add_argument("--budget", type=int, default=None,
                        help=f"catalogue budget in bytes (default: {DEFAULT_CATALOGUE_BYTES})")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any subagent was refused or dropped from the catalogue (for CI)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="emit the roster as JSON")
    parser.set_defaults(func=run_agents, uses_config=False)


__all__ = ["DEFAULT_CATALOGUE_BYTES", "register", "run_agents"]
