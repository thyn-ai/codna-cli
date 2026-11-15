"""The capability sentences a run carries — built WITHOUT the optional workflow runtime.

`capability_report` schedules the activities whose answers these sentences report, so it imports
`workflow_support` → ``temporalio``. The sentences themselves are pure string builders and a pure
read of the worker's environment, which is why they live here: a test (or a CLI report) that asks
"what would the ledger say?" must not need the ``durable`` extra installed to get an answer.

**Why the split is load-bearing, not tidy.** `tests/durable/conftest.py` states the rule: nothing may
import ``temporalio``/``langgraph`` at module scope, because that turns "extra not installed" into a
collection error and hides the branch the base suite exists to exercise. `delegation_policy.py` was
split out of `capability_report.py` for exactly this reason; the MCP note, its declared-server helper
and `capability_notes` follow it here.

**The import chain stays stdlib + paints.** `contracts` (dataclasses), `mcp` (os/re/typing) and
`swarm` (typing) import nothing above this layer, and `tests/durable/test_import_hygiene.py` pins
that with a ``sys.meta_path`` blocker. Re-exported from `capability_report`, so every existing caller
keeps its import site.

The activity SCHEDULING half stays in `capability_report`: this module answers with the WORKER's
environment and is for the paths that run outside a workflow (a CLI report, a test).
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import RunSpec
from .mcp import resolve_mcp_attachment
from .swarm import swarm_attachment_state

__all__ = [
    "capability_notes",
    "declared_mcp_servers",
    "mcp_attachment_note",
]


def declared_mcp_servers(spec: RunSpec) -> tuple[dict[str, Any], ...]:
    """The run's own tasks' MCP servers, disabled entries removed — the DECLARED half of the answer.

    Built from the RUN, not from an empty context: a note that said "no servers configured" for a run
    whose tasks all configured one would be the exact wrong-audit-note this reporting exists to avoid. A
    run's tasks may differ, so the union is what was carried.
    """
    return tuple(
        server
        for task in spec.tasks
        for server in task.mcp_servers
        if server.get("disabled") is not True
    )


def mcp_attachment_note(run_id: str, outcome: Any) -> str:
    """The one sentence a human reads to answer "did the browser run?", from the EFFECTIVE answer.

    Built from `McpAttachmentOutcome` rather than from the spec, because the spec only names what the run
    DECLARED; the built-in browser server and the worker-wide default are added by the session request.
    A genuine absence and a broken configuration read differently — the second names the fault and its
    code — so an operator is never told "not attached" when the truth is "your switch is invalid".
    """
    if getattr(outcome, "attached", False):
        names = ", ".join(getattr(outcome, "servers", ()) or ())
        return f"run {run_id}: MCP is attached — servers: {names}"
    reason = str(getattr(outcome, "reason", "") or "no MCP servers are attached to this run")
    code = str(getattr(outcome, "error_code", "") or "")
    if not code:
        return f"run {run_id}: MCP is not attached — {reason}"
    return f"run {run_id}: MCP is not attached — {reason} [code: {code}]"


def capability_notes(
    spec: RunSpec,
    env: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """The MCP and delegation notes this run should carry, as ``(kind, message)`` pairs.

    The DELEGATION entry says only that the flags were CARRIED; whether the policy lets them act is the
    activity's answer, appended by `report_capabilities`. That sentence is built here too (a function of
    the spec alone) so both paths produce the SAME words rather than two narrations of one fact.

    **The MCP note names the EFFECTIVE servers, and getting that wrong was a shipped defect.** The
    built-in browser server is appended by `mcp.session_mcp_servers` when ``CODNA_BROWSER_TOOL=1``; a note
    built from the spec's DECLARED list alone printed "MCP is not attached" for a run that went on to make
    dozens of real ``browser__browser_*`` calls. The one sentence a human reads to answer "did the browser
    run?" therefore said no.

    This function answers with the WORKER's environment, so it is for the paths that run outside a
    workflow (a CLI report, a test). A workflow SCHEDULES `NAME_MCP_ATTACHMENT` and builds the note from
    its outcome — a workflow may not read ``os.environ``.
    """
    outcome = resolve_mcp_attachment(
        _McpRequest(run_id=spec.run_id, servers=declared_mcp_servers(spec)),
        env,
    )
    mcp = mcp_attachment_note(spec.run_id, outcome)

    swarm = swarm_attachment_state(
        type(
            "_RunSwarmView",
            (),
            {
                "enable_spawn_agent": any(task.enable_spawn_agent for task in spec.tasks),
                "enable_agent_teams": any(task.enable_agent_teams for task in spec.tasks),
            },
        )()
    )
    if swarm.attached:
        carried = [
            name
            for name, on in (
                ("enable_spawn_agent", swarm.spawn),
                ("enable_agent_teams", swarm.teams),
            )
            if on
        ]
        delegation = f"run {spec.run_id}: delegation is attached — {', '.join(carried)}"
    else:
        delegation = f"run {spec.run_id}: delegation is not attached — {swarm.reason}"
    return [("note", mcp), ("note", delegation)]


class _McpRequest:
    """The minimal request shape `resolve_mcp_attachment` reads, so this path needs no contract import."""

    __slots__ = ("run_id", "servers")

    def __init__(self, run_id: str, servers: tuple[dict[str, Any], ...]) -> None:
        self.run_id = run_id
        self.servers = servers
