"""Subagent swarms on the durable engine path: the two request fields, and the gate that bounds them.

**The gap this closes, stated as it was.** The engine can spawn a subagent and can form a team, but it
gates both behind two create-session fields — ``enable_spawn_agent`` and ``enable_agent_teams`` — and
``algenta/server/policy.ts`` REFUSES either one when the session's approval profile does not allow it.
The durable path sent neither, so a Temporal/LangGraph run could not delegate AT ALL, whatever the
engine supported. `grep -rn "enable_spawn_agent" cli/codna/durable/` returned nothing, which is the
honest measure of that gap.

**What travels, and what does not.** Two booleans. No tool names, no roster, no prompts: which tools an
opted-in session gains is the ENGINE's table (``DEFAULT_APPROVAL_PROFILES`` in
``algenta/server/contracts.ts``), and duplicating it here would be a second truth that drifts the first
time a team tool is added. What this module owns is the DECISION the engine would otherwise make
invisibly: whether the run's approval profile permits the capability at all.

**Fail closed, and fail HERE.** `PROFILE_SWARM_PERMISSIONS` mirrors the engine's own table for exactly
the two flags (it is asserted against the TypeScript in the engine's test suite, not merely believed
here). A spec asking for a capability its profile forbids is refused when the spec is VALIDATED — on
the CLI, before a workflow exists — instead of at session-create time three activities into the run.
The engine still refuses it independently: this is a second gate in front of the first, never a
replacement for it, and a run that somehow got past this one would still be refused by
``resolveSessionPolicy``.

**Why the default is every profile.** Every profile in the engine's table currently allows both
capabilities — they were granted together when delegation was made the ordinary case (teammates
inherit the session's tool policy, so delegating cannot escalate write capability). So the mirror is a
mirror, not a policy change: a spec that names nothing keeps sending nothing, and the table is what
makes a FUTURE profile that forbids delegation fail closed on this side too.
"""
from __future__ import annotations

from typing import Any, Final

__all__ = [
    "PROFILE_SWARM_PERMISSIONS",
    "PROMPT_NAMED_TEAM_TOOLS",
    "SWARM_FIELDS",
    "SwarmAttachment",
    "SwarmConfigError",
    "normalize_swarm",
    "policy_stripped_team_tools",
    "swarm_attachment_state",
    "swarm_session_spec",
]

#: The two create-session fields, in the engine's own spelling. The spec uses snake_case for both so
#: the recorded payload reads like the request it produces — `TaskSpec.mcp_servers` follows the same
#: rule, and for the same reason: a decode/encode round trip must be byte-identical.
SWARM_FIELDS: Final[tuple[str, ...]] = ("enable_spawn_agent", "enable_agent_teams")


class SwarmConfigError(ValueError):
    """A swarm request that must not be silently tolerated. Always carries a stable ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


#: The engine's approval profiles and whether each may set the two flags.
#:
#: Mirrored from ``algenta/server/contracts.ts::DEFAULT_APPROVAL_PROFILES``. Kept as data with the
#: engine's own names so a reader can diff the two by eye, and asserted field-for-field by
#: `tests/durable/test_swarm_contract.py::test_the_profile_table_matches_the_engines`.
#:
#: A profile ABSENT from this table is treated as forbidding both, never as permitting them: a profile
#: added on the engine side that this side has not learned about must fail closed rather than default
#: to "everything is allowed".
PROFILE_SWARM_PERMISSIONS: Final[dict[str, dict[str, bool]]] = {
    "repository_review": {"enable_spawn_agent": True, "enable_agent_teams": True},
    "repository_fix_gated": {"enable_spawn_agent": True, "enable_agent_teams": True},
    "local_validation": {"enable_spawn_agent": True, "enable_agent_teams": True},
    "ci_controlled_repo_automation": {"enable_spawn_agent": True, "enable_agent_teams": True},
}

#: The profile a durable run's session is created under. `repository_fix_gated` is the engine's
#: `DEFAULT_APPROVAL_PROFILE`, and the durable path never sends `approval_profile` — so this constant
#: is what the engine will actually resolve, not an assumption about it.
DURABLE_APPROVAL_PROFILE: Final = "repository_fix_gated"


class SwarmAttachment:
    """Whether this run's turns may delegate, and if not, exactly why.

    A pair rather than a bool for the same reason `McpAttachment` is one: "we did not ask for it" and
    "we asked and it was refused/not carried" are different facts, and only the second is a defect.
    """

    __slots__ = ("attached", "reason", "spawn", "teams")

    def __init__(
        self,
        attached: bool,
        reason: str = "",
        *,
        spawn: bool = False,
        teams: bool = False,
    ) -> None:
        self.attached = attached
        self.reason = reason
        self.spawn = spawn
        self.teams = teams

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SwarmAttachment):
            return NotImplemented
        return (self.attached, self.reason, self.spawn, self.teams) == (
            other.attached,
            other.reason,
            other.spawn,
            other.teams,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SwarmAttachment(attached={self.attached!r}, reason={self.reason!r}, "
            f"spawn={self.spawn!r}, teams={self.teams!r})"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "attached": self.attached,
            "reason": self.reason,
            "enable_spawn_agent": self.spawn,
            "enable_agent_teams": self.teams,
        }


def _fail(message: str, *, field: str, value: Any = None) -> "SwarmConfigError":
    details: dict[str, Any] = {"field": field}
    if value is not None:
        details["value"] = repr(value)[:200]
    return SwarmConfigError(message, code="swarm_config_invalid", details=details)


def normalize_swarm(
    *,
    enable_spawn_agent: Any = False,
    enable_agent_teams: Any = False,
    approval_profile: str = DURABLE_APPROVAL_PROFILE,
) -> tuple[bool, bool]:
    """Validate the two swarm flags against the approval profile, or raise.

    Returns ``(enable_spawn_agent, enable_agent_teams)`` as real booleans. Refuses:

    * a non-boolean (including ``1``/``"true"``): the value enters workflow history and then a request
      body, and a truthy string that the engine reads as ``true`` while this side recorded ``"yes"``
      is a spec that does not describe the request it produces;
    * a capability the profile forbids — the same boundary the engine enforces, refused earlier so the
      operator learns at the CLI rather than three activities into a run.
    """
    allowed = PROFILE_SWARM_PERMISSIONS.get(approval_profile)
    if allowed is None:
        raise _fail(
            f"unknown approval profile {approval_profile!r}; cannot decide whether it may delegate",
            field="approval_profile",
            value=approval_profile,
        )
    values: dict[str, bool] = {}
    for field in SWARM_FIELDS:
        raw = enable_spawn_agent if field == "enable_spawn_agent" else enable_agent_teams
        if not isinstance(raw, bool):
            raise _fail(f"{field} must be a boolean", field=field, value=raw)
        if raw and not allowed[field]:
            # The engine's own message shape, so an operator who has seen one refusal recognises the
            # other. The engine refuses this too; this refusal only arrives sooner.
            raise SwarmConfigError(
                f"{field} is not allowed for approval_profile={approval_profile}.",
                code="swarm_not_permitted",
                details={
                    "field": field,
                    "approval_profile": approval_profile,
                    "allowed_profiles": sorted(
                        name for name, perms in PROFILE_SWARM_PERMISSIONS.items() if perms[field]
                    ),
                },
            )
        values[field] = raw
    return values["enable_spawn_agent"], values["enable_agent_teams"]


def swarm_session_spec(context: Any) -> dict[str, Any]:
    """The swarm keys for a create-session request, or ``{}`` when this run delegates nothing.

    A SPEC FRAGMENT rather than a pair of values, for the same reason `mcp.mcp_session_spec` returns
    one: a run that never asked must produce a request byte-identical to the pre-feature request, and
    ``{"enable_spawn_agent": False}`` is not identical to its absence. The engine's `booleanValue`
    reads both the same way, so this is about the LEDGER and the recorded request agreeing with the
    run that was asked for — not about engine behaviour.

    The values are re-validated HERE as well as at the spec boundary: a context is what a REQUEST
    carries, and a request that crossed a channel is still untrusted. Re-raising keeps one error family
    for the activity to translate.
    """
    spawn = bool(getattr(context, "enable_spawn_agent", False))
    teams = bool(getattr(context, "enable_agent_teams", False))
    if not spawn and not teams:
        return {}
    spawn, teams = normalize_swarm(enable_spawn_agent=spawn, enable_agent_teams=teams)
    return {"enable_spawn_agent": spawn, "enable_agent_teams": teams}


def swarm_attachment_state(context: Any = None) -> SwarmAttachment:
    """Whether this run's turns can delegate, reported into the run's ledger.

    Three distinct answers, and the difference matters to an operator:

    * **attached** — the run asked for delegation and both capabilities are in the request;
    * **nothing requested** — the ordinary case. Reported as *not attached* naming the absence,
      because "the run never asked" and "the run asked and it was dropped" are different facts;
    * **spawn only** — a run that enabled subagents without teams. Reported as attached, with ``teams``
      false, so a reader can tell a swarm from a single delegation.
    """
    raw_spawn = bool(getattr(context, "enable_spawn_agent", False)) if context is not None else False
    raw_teams = bool(getattr(context, "enable_agent_teams", False)) if context is not None else False
    if not raw_spawn and not raw_teams:
        return SwarmAttachment(
            attached=False,
            reason=(
                "this run does not delegate: no TaskSpec set `enable_spawn_agent` or "
                "`enable_agent_teams`. The engine path DOES accept both on a create-session request "
                "and builds the team runtime when either is set, so a run that needs a swarm sets "
                "them on its TaskSpec (`codna run --swarm`)."
            ),
        )
    return SwarmAttachment(
        attached=True,
        reason="",
        spawn=raw_spawn,
        teams=raw_teams,
    )


#: The team tools the delegation PROMPT names (see `algenta/adapter.ts`: `TEAM_MAILBOX_TOOLS` +
#: `TEAM_CRITIQUE_TOOLS`). A policy that does not name them cannot run a swarm, so the run reports it
#: instead of the operator discovering it from an `unavailable tool` failure mid-turn.
PROMPT_NAMED_TEAM_TOOLS: Final[tuple[str, ...]] = (
    "team_send_message",
    "team_broadcast",
    "team_status",
    "team_await_runs",
    "team_mission_log",
    "team_review_outcome_fragment",
    "team_run_task",
)


def policy_stripped_team_tools(repo_dir: Any, *, enable_agent_teams: bool) -> tuple[str, ...]:
    """The prompt-named team tools this repository's POLICY removes from the toolset.

    Returns an empty tuple when delegation is not requested, when the repository has no policy (the
    approval profile then decides alone — and every profile the durable path can resolve grants all
    of them), when the policy names all seven, or when the document yields no permissions at all (a
    policy that cannot be evaluated, or permits no tool, is a REFUSAL at create; reporting it as
    "delegation stripped" would describe a run that never starts).

    ``repo_dir`` is the run's repository, which is the file the operator wrote and the one the spec
    was validated against. A task working in its own worktree is governed by that branch's copy — this
    reads the run's, and the note names the path it read so the two cannot be confused.

    WHY THIS EXISTS. `enable_agent_teams` is a GATE, not a GRANT, and the grant is narrowed TWICE: by
    the approval profile's `allowedTools` and, again, by `.codna/policy.yaml` at session create
    (`run-server.ts` → `applyPolicyToToolPolicies` → `narrowToolPolicies`). A policy written for a
    single-agent run — `tools.allow: [read_files, editor, run_commands]` is the obvious one — leaves
    the profile intact and still sets `enabled: false` on every `team_*` name, which the engine's
    `filterAvailableTools` then drops. Measured on `run-20261112T223954Z`: the ledger's own note said
    `delegation is attached — enable_spawn_agent, enable_agent_teams`, the session snapshot said
    `enableTeams: true`, and the model's `team_run_task` call died with
    `unavailable tool 'team_run_task'` across 898 records and ZERO subagent sessions.

    A POLICY CANNOT GRANT (only the profile + the tool list do), so this reports rather than repairs:
    a run whose policy excludes the tools cannot delegate, and the operator needs to know that the
    fix is in their policy file rather than in the delegation flags they already set correctly.
    """
    if not enable_agent_teams:
        return ()
    # Imported HERE, not at module scope: `policy_document` pulls in the YAML reader and the findings
    # digest, and a spec-validation-only caller of this module should not pay for either.
    from ..policy_document import policy_path_for, policy_verdict

    path = policy_path_for(repo_dir)
    verdict = policy_verdict(path, required=False)
    if verdict.mode == "absent" or not verdict.permissions:
        return ()
    permitted = set(verdict.permissions.effective_tools())
    return tuple(name for name in PROMPT_NAMED_TEAM_TOOLS if name not in permitted)


# -- the CLI surface -----------------------------------------------------------------------
#
# Here rather than in `run_cli.py` because this module owns the capability DECISION, and because that
# file is at the repository's 1000-line ceiling (`test_modularity.py`). The flags are registered by
# `add_delegation_flags`, so both spellings of `run` get them from one place.


class SwarmFlagError(RuntimeError):
    """A contradictory delegation request. Carries a ``code`` so the CLI renders it structured."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = dict(details or {})


def add_delegation_flags(parser: Any) -> None:
    """Register ``--swarm`` / ``--spawn-agent`` / ``--agent-teams`` / ``--no-swarm`` / ``--parallel``.

    ``--swarm`` is the convenience spelling for BOTH delegation fields; the per-capability flags exist
    because the engine gates the two independently (`runtime-builder.ts` enables the generic
    ``spawn_agent`` under one and the team runtime under the other), so a team-only session is a legal
    thing to want.

    There is deliberately NO ``--teammates``/roster flag: which specialist agents a swarm may draw from
    is the ENGINE's own registry (the same personas the IDE lets you @-mention), and a second roster in
    a run spec would be a copy that drifts.
    """
    parser.add_argument(
        "--swarm",
        action="store_true",
        help="let this run's turns spawn subagents AND form a team (both delegation fields)",
    )
    parser.add_argument(
        "--spawn-agent",
        action="store_true",
        help="let this run's turns spawn a single subagent (enable_spawn_agent)",
    )
    parser.add_argument(
        "--agent-teams",
        action="store_true",
        help="let this run's turns form a team of teammates (enable_agent_teams)",
    )
    parser.add_argument(
        "--no-swarm",
        action="store_true",
        help="refuse delegation explicitly (the default), overriding --swarm in a wrapper",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="run up to N tasks concurrently (default: 1, strictly sequential); requires --worktree",
    )


def swarm_choice(args: Any, capability: str) -> bool:
    """Whether this run asked for one delegation capability.

    ``--swarm`` means BOTH capabilities — that is what "spin off a swarm of subagents" asks for, and
    making the caller spell two flags to get it would be a footgun with a silent half-on state.

    ``--no-swarm`` is the explicit OFF, so a caller wrapping this in a template can override an
    environment where a default was changed. ``--swarm`` together with ``--no-swarm`` is a REFUSAL
    rather than "last one wins": which of two contradictory requests took effect is not something a
    run's ledger should have to imply.
    """
    swarm = bool(getattr(args, "swarm", False))
    no_swarm = bool(getattr(args, "no_swarm", False))
    if swarm and no_swarm:
        raise SwarmFlagError(
            "--swarm and --no-swarm are mutually exclusive.",
            code="durable_swarm_conflict",
            details={"flags": ["--swarm", "--no-swarm"]},
        )
    if no_swarm:
        return False
    if swarm:
        return True
    return bool(getattr(args, capability, False))
