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

**The MCP note follows the same rule, for the same measured reason.** `capability_notes`, its
`declared_mcp_servers` helper and `mcp_attachment_note` live in `capability_notes`, which imports only
stdlib, typing, ``contracts``, ``mcp`` and ``swarm``. They were defined HERE when first written, and a
test importing them turned "extra not installed" into ``ModuleNotFoundError: No module named
'temporalio'`` on the offline lane — the regression `tests/durable/test_import_hygiene.py` now pins.
Re-exported below, so a caller that already imports this module is unaffected.
"""
from __future__ import annotations

from typing import Any, Protocol

from .capability_notes import (
    capability_notes,
    declared_mcp_servers,
    mcp_attachment_note,
)
from .contracts import (
    NAME_DELEGATION_POLICY,
    NAME_MCP_ATTACHMENT,
    DelegationPolicyRequest,
    McpAttachmentRequest,
    RunSpec,
)
from .delegation_policy import delegation_policy_note
from .workflow_support import workflow_activity

__all__ = [
    "capability_notes",
    "declared_mcp_servers",
    "delegation_policy_note",
    "mcp_attachment_note",
    "report_capabilities",
]


class _Notifier(Protocol):
    """The one capability this module needs from the workflow: append a ledger row."""

    async def __call__(self, kind: str, message: str, *, repo_dir: str = "") -> None: ...


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

    **Both answers come from ACTIVITIES, and that is what makes them true.** The MCP answer depends on
    the WORKER's environment (``CODNA_BROWSER_TOOL``, ``CODNA_MCP_SERVERS``), which a workflow may not
    read; the delegation answer depends on the repository's policy file, for the same reason. Asking in
    an activity also puts each answer in history, so a replay derives these rows instead of re-reading a
    variable or a file that may have changed since.
    """
    enough = _declared_delegation_carried(spec)
    message = await _mcp_note(spec, logger=logger)
    if message is not None:
        await notify("note", message)
    await notify("note", _delegation_carried_note(spec, enough))
    if not enough:
        return
    verdict = await _delegation_policy_verdict(spec, logger=logger)
    if verdict is None:
        return
    policy_message = delegation_policy_note(
        spec.run_id,
        tuple(getattr(verdict, "stripped", ()) or ()),
        str(getattr(verdict, "policy_path", "") or ""),
    )
    if policy_message is not None:
        await notify("note", policy_message)


async def _mcp_note(spec: RunSpec, *, logger: Any = None) -> str | None:
    """The MCP row, from the ``mcp_attachment`` activity; ``None`` when even that could not be asked."""
    request = McpAttachmentRequest(run_id=spec.run_id, servers=declared_mcp_servers(spec))
    try:
        outcome = await workflow_activity(NAME_MCP_ATTACHMENT, request)
    except Exception as exc:  # noqa: BLE001 - a report must never fail a run
        if logger is not None:
            logger.warning(
                "run %s: MCP attachment report failed: %r (cause: %r)",
                spec.run_id, exc, getattr(exc, "cause", None),
            )
        return None
    return mcp_attachment_note(spec.run_id, outcome)


def _declared_delegation_carried(spec: RunSpec) -> bool:
    """Whether the run CARRIED a delegation flag — the gate on asking the policy at all.

    Only a run that asked to spawn agents or form a team can have its delegation tools stripped, so a run
    that carried neither schedules no policy read: it would spend a round trip to learn nothing.
    """
    return any(
        task.enable_spawn_agent or task.enable_agent_teams for task in spec.tasks
    )


def _delegation_carried_note(spec: RunSpec, carried: bool) -> str:
    """The delegation row BEFORE the policy verdict: what the run asked for, and whether it was kept.

    Built here rather than in `capability_notes` because the workflow path must not touch `swarm`'s
    module-level state from workflow code; the sentence is a function of the spec alone.
    """
    if carried:
        names = [
            name
            for name, on in (
                ("enable_spawn_agent", any(t.enable_spawn_agent for t in spec.tasks)),
                ("enable_agent_teams", any(t.enable_agent_teams for t in spec.tasks)),
            )
            if on
        ]
        return f"run {spec.run_id}: delegation is attached — {', '.join(names)}"
    return (
        f"run {spec.run_id}: delegation is not attached — the run carried no delegation flags, so it "
        "gets no subagents and no team tools"
    )


async def _delegation_policy_verdict(spec: RunSpec, *, logger: Any = None) -> Any:
    """The policy read for a run that carried team flags; ``None`` when it could not be answered."""
    try:
        return await workflow_activity(
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
        return None
