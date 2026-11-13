"""What a run's OPENING rows say about the integrations it did and did not carry (D5/D7).

Split out of `run_workflow.py` at the repository's 1000-line ceiling (`test_modularity.py`) on a real
concern boundary: nothing here affects what a run DOES. Every function either builds a sentence for the
ledger or schedules the one ACTIVITY whose answer a sentence needs. A reviewer reading this file knows
the blast radius is the audit trail.

**Both integrations are reported in BOTH directions, and that is the point.** An operator who
configured servers learns which ones reached the engine; one who configured none reads that the field
exists rather than assuming their IDE-level server was consulted. A one-sided note (only "not
attached") cannot distinguish "carried" from "silently dropped", which is the confusion this area
exists to remove.

**The delegation note has a second half, and it is a defect fix rather than decoration.**
`enable_agent_teams` is a GATE, not a GRANT: the tools are narrowed by the approval profile's
`allowedTools` AND by `.codna/policy.yaml` at session create. A policy written for a single-agent run
leaves the profile intact and still strips every `team_*` name, so a run can report
`delegation is attached` — with a session snapshot to match — and then fail every `team_run_task` call
with `unavailable tool 'team_run_task'`. Measured on `run-20261112T223954Z`: 898 records, ZERO
`team_progress`, ZERO subagent sessions. The policy read therefore happens HERE, before the first turn,
and its answer is a ledger row rather than a mid-run surprise.

**Why the read is an activity.** It touches the filesystem, and Temporal's workflow sandbox refuses
that from workflow code — measured as
``RestrictedWorkflowAccessError: Cannot access os.path from inside a workflow`` when this logic was
first written inline. Running it in an activity also puts the ANSWER in history, so a replay derives
the report instead of re-reading a file that may have changed since.

**The note and the activity body live in `delegation_policy`.** This module schedules the activity, so
it imports ``temporalio`` through `workflow_support`; the pure string builder and the pure read must
NOT, or asserting either would require the OPTIONAL extra. They are imported here — and re-exported,
so every existing caller keeps importing this module — rather than duplicated.
"""
from __future__ import annotations

from typing import Any, Protocol

from .contracts import (
    NAME_DELEGATION_POLICY,
    DelegationPolicyRequest,
    RunSpec,
)
from .delegation_policy import delegation_policy_note
from .workflow_support import workflow_activity

__all__ = [
    "capability_notes",
    "delegation_policy_note",
    "report_capabilities",
]


class _Notifier(Protocol):
    """The one capability this module needs from the workflow: append a ledger row."""

    async def __call__(self, kind: str, message: str, *, repo_dir: str = "") -> None: ...


def capability_notes(spec: RunSpec) -> list[tuple[str, str]]:
    """The MCP and delegation notes this run should carry, as ``(kind, message)`` pairs.

    PURE, so the sentences are testable without a workflow, a sandbox or a sidecar. The delegation
    entry says only that the flags were CARRIED; whether the policy lets them act is the activity's
    answer, appended by `report_capabilities`.
    """
    from .rules import mcp_attachment_state
    from .swarm import swarm_attachment_state

    # Built from the RUN's own tasks, not from an empty context: a note that said "no servers
    # configured" for a run whose tasks all configured one would be the exact wrong-audit-note this
    # reporting exists to avoid. A run's tasks may differ, so the union is what was carried.
    servers = tuple(
        server
        for task in spec.tasks
        for server in task.mcp_servers
        if server.get("disabled") is not True
    )
    attachment = mcp_attachment_state(type("_RunMcpView", (), {"mcp_servers": servers})())
    if attachment.attached:
        mcp = f"run {spec.run_id}: MCP is attached — servers: {', '.join(attachment.servers)}"
    else:
        mcp = f"run {spec.run_id}: MCP is not attached — {attachment.reason}"

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


async def report_capabilities(
    spec: RunSpec,
    notify: _Notifier,
    *,
    logger: Any = None,
) -> None:
    """Append this run's capability rows, including the policy verdict on its delegation flags.

    Deliberately a notification path, not a failure path: a run must not refuse to start because one
    optional integration is unavailable, and the information is only actionable for an operator who
    expected it. That is also why the activity result is read defensively — a report about a run must
    never be able to fail the run.
    """
    for kind, message in capability_notes(spec):
        await notify(kind, message)

    enable_agent_teams = any(task.enable_agent_teams for task in spec.tasks)
    if not enable_agent_teams:
        # Only a run that asked to form a team can have its team tools stripped. Asking an activity
        # here would spend a round trip to learn nothing.
        return
    try:
        verdict = await workflow_activity(
            NAME_DELEGATION_POLICY,
            DelegationPolicyRequest(
                run_id=spec.run_id,
                repo_dir=spec.repo_dir,
                enable_agent_teams=True,
            ),
        )
    except Exception:  # noqa: BLE001 - a report must never fail a run
        if logger is not None:
            logger.warning("run %s: delegation policy report failed", spec.run_id)
        return
    stripped = tuple(getattr(verdict, "stripped", ()) or ())
    message = delegation_policy_note(
        spec.run_id,
        stripped,
        str(getattr(verdict, "policy_path", "") or ""),
    )
    if message is not None:
        await notify("note", message)
