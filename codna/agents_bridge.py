"""Bridge: hand the IDE the SUBAGENT roster as JSON.

The sibling of :mod:`codna.rules_bridge` and :mod:`codna.skills_bridge`, and deliberately the same
shape: one JSON object in, one JSON object out, the SAME ``codna.rules`` engine answering. It exists
because the IDE needs the roster as STRUCTURE (to send it on a create-session request, and to report
why a file did not load) rather than as prompt text, and re-implementing agent discovery in TypeScript
would be the split-reader defect this package removes.

**Protocol.**

Request on stdin, one of::

    {"repo": "/abs/path"}                       → the roster
    {"repo": "/abs/path", "agent": "verifier"}  → one agent, with its body

Response on stdout::

    {"ok": true, "agents": [...], "refusals": [...], "warnings": [...],
     "used_bytes": 123, "budget_bytes": 2048, "text": "..."}          roster
    {"ok": true, "name": "verifier", "body": "...", ...}             one agent

On a refusal the response is ``{"ok": false, "error": {"code": …, "message": …, "details": {}}}`` and
the exit code is 1. The process NEVER guesses: an absent ``repo`` is a refusal, not "use the cwd".

**Why the roster travels WITH its refusals and warnings.** A host that received only the agents it
could load would render a list that is silently short — the "I wrote an agent and nothing happened"
complaint, which is exactly what this feature must not reproduce. Every file that did not load is
named with a stable ``code``, and the host has what it needs to say so.
"""
from __future__ import annotations

import json
import sys

__all__ = ["DEFAULT_MAX_BYTES", "handle", "main"]

# The roster catalogue ceiling. The roster is a list of one-line entries, not prompts, so this is
# generous for a real repository and still bounded.
DEFAULT_MAX_BYTES = 2_048

#: A hard ceiling on the SUMMED system-prompt bytes of the roster that travels to the engine. Separate
#: from `DEFAULT_MAX_BYTES` on purpose: the catalogue budget bounds what a MENU may cost, and a menu that
#: did not fit must not silently disable the delegation it was describing. This bounds the actual payload
#: instead, per agent, and names any agent it excluded.
MAX_ROSTER_BYTES = 256 * 1024


def _error(code: str, message: str, **details) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


def _agent(agent) -> dict:
    """One agent WITHOUT its body, for a DIAGNOSTIC list."""
    return {
        "name": agent.name,
        "description": agent.description,
        "scope": agent.scope.value,
        "source_path": agent.source_path,
        "model": agent.model,
        "readonly": agent.readonly,
    }


def _agent_with_body(agent) -> dict:
    """One agent WITH the markdown body, which is the subagent's system prompt.

    The body travels for the ROSTER answer (and only there) because the engine builds a delegated-agent
    tool whose system prompt IS this text: a roster without bodies would produce tools with empty
    instructions. It is bounded by ``MAX_ROSTER_BYTES`` rather than by the catalogue budget, because a
    budget exists to bound PROMPT space and must not silently disable a delegation capability.
    """
    return {**_agent(agent), "body": agent.body}


def handle(request: object) -> dict:
    """Answer one bridge request. Pure apart from reading the repository."""
    if not isinstance(request, dict):
        return _error("bridge_bad_request", "the request must be a JSON object")
    repo = request.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        return _error("bridge_bad_request", "`repo` is required and must be a non-empty string")

    # The HOME directory the user layer is read from. Absent means the real one — a product host wants
    # `~/.cursor/agents` exactly as Cursor discovers it — and an explicit value lets a test (or a host
    # scoping the machine layer deliberately) choose.
    home = request.get("home")
    if home is not None and (not isinstance(home, str) or not home.strip()):
        return _error("bridge_bad_request", "`home` must be a non-empty string or null")

    from .rules import agent_catalogue, discover_agents, load_agent

    raw_name = request.get("agent")
    if raw_name is not None:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return _error("bridge_bad_request", "`agent` must be a non-empty string")
        agent = load_agent(repo, raw_name, home=home)
        if agent is None:
            return _error(
                "agent_not_found",
                f"no subagent named {raw_name!r}",
                requested=raw_name,
                repo=repo,
            )
        return {"ok": True, **_agent(agent), "body": agent.body}

    max_bytes = request.get("max_bytes", DEFAULT_MAX_BYTES)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        return _error("bridge_bad_request", "`max_bytes` must be an integer >= 0")

    discovery = discover_agents(repo, home=home)
    result = agent_catalogue(discovery, max_bytes=max_bytes)
    # WHICH agents the catalogue admitted, each WITH its body. The body is the subagent's system prompt
    # and the engine cannot build a delegation tool without it, so a roster answer that omitted bodies
    # would hand the engine tools with empty instructions — the half-wired state this feature exists to
    # avoid. It is capped by MAX_ROSTER_BYTES rather than by the catalogue budget: the budget bounds
    # PROMPT space and must not silently disable a delegation capability.
    catalogued: list[dict] = []
    refused_for_size: list[dict] = []
    total = 0
    for agent in result.catalogued:
        entry = _agent_with_body(agent)
        size = len(agent.body.encode("utf-8"))
        if total + size > MAX_ROSTER_BYTES:
            refused_for_size.append({
                "code": "agent_body_over_roster_budget",
                "source_path": agent.source_path,
                "message": (
                    f"body is {size} bytes and the roster budget is {MAX_ROSTER_BYTES}; this agent "
                    "did not travel (raise nothing — split the prompt, or load it on demand)"
                ),
            })
            continue
        total += size
        catalogued.append(entry)
    return {
        "ok": True,
        "agents": catalogued,
        "refusals": [
            {"code": refusal.code, "source_path": refusal.source_path, "message": refusal.message}
            for refusal in discovery.refusals
        ] + refused_for_size,
        "warnings": [
            {"code": warning.code, "source_path": warning.source_path, "message": warning.message}
            for warning in discovery.warnings
        ],
        "used_bytes": result.used_bytes,
        "budget_bytes": result.budget_bytes,
        "roster_bytes": total,
        "text": result.text,
    }


def main(argv: list[str] | None = None) -> int:
    """Read one request from stdin, write one response to stdout. Returns the exit code."""
    del argv  # the protocol is stdin/stdout only; flags would be a second, drifting interface
    try:
        raw = sys.stdin.read()
    except OSError as exc:  # pragma: no cover - stdin closed by the parent
        print(json.dumps(_error("bridge_stdin_failed", str(exc))), file=sys.stdout)
        return 1
    try:
        request = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps(_error("bridge_bad_json", f"cannot parse the request: {exc}")), file=sys.stdout)
        return 1

    try:
        response = handle(request)
    except Exception as exc:  # noqa: BLE001 — a bridge must answer, never traceback into the host
        response = _error("bridge_failed", f"{type(exc).__name__}: {exc}")
    print(json.dumps(response))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    sys.exit(main())
