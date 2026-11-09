"""User-defined subagents on the durable engine path: the roster a run's session is given.

**The gap this closes.** ``.cursor/agents/*.md`` (and ``.claude``/``.codex``, project and user) is how
Cursor defines a subagent, and Codna's delegation engine could spawn one — but the roster was hardcoded.
So a durable run could only ever delegate to the agents CODNA ships, never to the ones the repository
wrote. ``rg 'cursor/agents'`` over all source returned nothing, which is the honest measure of that gap.

**The roster is read from the TREE THE TURN EDITS, and the user layer is OFF.** Both follow the rule
``rules.project_context_for`` already documents for project rules, and for the same two reasons: a task's
worktree may be on a branch that changed the agents, and a durable run executes on a WORKER — possibly a
shared one — so reading ``~/.cursor/agents`` would put one machine's private roster into a repository's
turn, and would make two workers running the same recorded task build two different sessions.

**What travels in workflow history, and what does not.** A ``TaskSpec`` is history, so it carries the
POLICY (``attach_subagents``) and never the roster: the agents are file contents, exactly the class of
data ``TaskState`` refuses for rules. The activity resolves the roster from the working tree when it
creates the session.

**A refusal is reported, not swallowed.** The reader refuses a file with a named code (bad frontmatter, a
reserved built-in name, ``is_background: true``), and those codes ride on the request beside the roster
so the engine records them on the session. A run whose agent file did not load is a run whose operator
can find out why, instead of watching a delegation never happen.
"""
from __future__ import annotations

from typing import Any, Final

__all__ = [
    "MAX_ROSTER_BYTES",
    "RosterRead",
    "SubagentAttachment",
    "read_subagents",
    "subagent_attachment_state",
    "subagent_session_spec",
]

#: A hard ceiling on the summed system-prompt bytes the roster may carry. A delegated agent's system
#: prompt IS its markdown body, so an unbounded roster is an unbounded create request. The reader reports
#: any agent this excluded, by name, so the bound is never silent.
MAX_ROSTER_BYTES: Final = 256 * 1024


class RosterRead:
    """What the reader found: the roster, the files it refused, and the problems worth reporting.

    Three lists rather than two because the questions are different. A REFUSAL is "this file is not a
    subagent" (bad frontmatter, a reserved name, `is_background: true`); a WARNING is "this agent loaded,
    and something about it is worth knowing" (no description, shadowed by a higher-precedence file).
    Collapsing them would make a shadowed-but-valid file look broken, which is the opposite of the
    report a user needs.
    """

    __slots__ = ("roster", "refusals", "warnings")

    def __init__(
        self,
        roster: list[dict[str, Any]] | None = None,
        refusals: list[dict[str, Any]] | None = None,
        warnings: list[dict[str, Any]] | None = None,
    ) -> None:
        self.roster = list(roster or ())
        self.refusals = list(refusals or ())
        self.warnings = list(warnings or ())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RosterRead):
            return NotImplemented
        return (self.roster, self.refusals, self.warnings) == (
            other.roster,
            other.refusals,
            other.warnings,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RosterRead(agents={len(self.roster)}, refusals={len(self.refusals)}, "
            f"warnings={len(self.warnings)})"
        )


class SubagentAttachment:
    """Whether this run's turns can delegate to user-defined agents, and if not, exactly why.

    A pair rather than a bool for the same reason ``McpAttachment`` is one: "we did not attach it" and
    "we could not" are different answers, and only the second is worth telling an operator about.
    """

    __slots__ = ("attached", "reason", "agents", "refused")

    def __init__(
        self,
        attached: bool,
        reason: str = "",
        *,
        agents: tuple[str, ...] = (),
        refused: tuple[str, ...] = (),
    ) -> None:
        self.attached = attached
        self.reason = reason
        self.agents = agents
        self.refused = refused

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubagentAttachment):
            return NotImplemented
        return (self.attached, self.reason, self.agents, self.refused) == (
            other.attached,
            other.reason,
            other.agents,
            other.refused,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SubagentAttachment(attached={self.attached!r}, reason={self.reason!r}, "
            f"agents={self.agents!r}, refused={self.refused!r})"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "attached": self.attached,
            "reason": self.reason,
            "agents": list(self.agents),
            "refused": list(self.refused),
        }


def read_subagents(working_tree: str) -> RosterRead:
    """The roster ``working_tree`` defines, plus every refusal and warning.

    Reads through the ONE engine (``codna.rules.discover_agents``) and renders each agent with its body,
    because the engine builds a delegated-agent tool whose system prompt is that body. ``include_user`` is
    ``False``: a worker's home directory is not the repository's roster (see the module docstring).

    An agent whose body would push the roster past ``MAX_ROSTER_BYTES`` is moved to the refusals with a
    code — never dropped, because a silently short roster is the complaint this feature answers.
    """
    if not working_tree:
        return RosterRead()
    from ..rules import discover_agents

    discovery = discover_agents(working_tree, include_user=False)
    refusals: list[dict[str, Any]] = [
        {"code": refusal.code, "source_path": refusal.source_path, "message": refusal.message}
        for refusal in discovery.refusals
    ]
    warnings: list[dict[str, Any]] = [
        {"code": warning.code, "source_path": warning.source_path, "message": warning.message}
        for warning in discovery.warnings
    ]
    roster: list[dict[str, Any]] = []
    total = 0
    for agent in discovery.agents:
        size = len(agent.body.encode("utf-8"))
        if total + size > MAX_ROSTER_BYTES:
            refusals.append({
                "code": "agent_body_over_roster_budget",
                "source_path": agent.source_path,
                "message": (
                    f"body is {size} bytes and the roster budget is {MAX_ROSTER_BYTES}; this agent did "
                    "not travel"
                ),
            })
            continue
        total += size
        roster.append(agent.to_payload())
    return RosterRead(roster, refusals, warnings)


def subagent_session_spec(context: Any, working_tree: str = "") -> dict[str, Any]:
    """The ``subagents`` key for a create-session request, or ``{}`` when there are none.

    Returns a SPEC FRAGMENT rather than a value, so a run that did not opt in — or a repository that
    defines no agents — produces a request byte-identical to the pre-feature one. That is the same
    contract ``mcp_session_spec`` and ``swarm_session_spec`` follow, and it is what lets a reviewer see
    the feature's whole footprint in one diff.

    ``attach_subagents`` defaults to TRUE (unlike ``attach_project_rules``, which defaults to True too —
    both are "honour what the repository wrote"). A task that opts out sends neither key.

    The roster is only read when the session can actually delegate. A profile or spec that refuses
    delegation would otherwise pay a filesystem walk for a roster the engine must ignore — and, worse,
    would record an attachment the run did not have.
    """
    if not getattr(context, "attach_subagents", False):
        return {}
    if not getattr(context, "enable_spawn_agent", False):
        # No delegation, so no roster. Stated here rather than left to the engine so the LEDGER is
        # truthful: `subagent_attachment_state` reports the same reason.
        return {}
    if not working_tree:
        # No tree named means the caller did not say where the turn works; falling back to a shared
        # checkout would read ANOTHER task's branch's roster.
        return {}
    read = read_subagents(working_tree)
    if not read.roster:
        return {}
    spec: dict[str, Any] = {"subagents": read.roster}
    if read.refusals:
        spec["subagent_refusals"] = read.refusals
    # Warnings travel too, so an operator learns that a file they wrote lost to a higher-precedence copy
    # (or has no description the parent could choose it by). Without this the shadow report would exist
    # only in `codna agents` and never reach the run that was actually affected.
    if read.warnings:
        spec["subagent_warnings"] = read.warnings
    return spec


def subagent_attachment_state(context: Any = None, working_tree: str = "") -> SubagentAttachment:
    """Whether this run's turns can delegate to user-defined agents, reported into its ledger.

    Four distinct answers, and the difference matters to an operator:

    * **attached** — the run can delegate and the tree defines a roster;
    * **nothing requested** — the ordinary case: the run never opted in;
    * **delegation disabled** — the run opted in, but its spec does not allow spawning, so the roster
      would be ignored by the engine;
    * **no roster** — the run can delegate and the tree defines no agent files.
    """
    if context is None:
        return SubagentAttachment(
            attached=False,
            reason=(
                "no run context: nothing to read a roster for. A durable run attaches subagents from "
                "its TaskSpec (`attach_subagents`) and its working tree."
            ),
        )
    if not getattr(context, "attach_subagents", False):
        return SubagentAttachment(
            attached=False,
            reason=(
                "this run does not attach user-defined subagents: its TaskSpec did not set "
                "`attach_subagents`. A repository's `.cursor/agents/*.md` roster is only read for a "
                "run that asked for it."
            ),
        )
    if not getattr(context, "enable_spawn_agent", False):
        return SubagentAttachment(
            attached=False,
            reason=(
                "this run cannot delegate: `enable_spawn_agent` is not set on its TaskSpec, so a "
                "subagent roster would be ignored by the engine. Set both (`codna run --swarm "
                "--agents`) to delegate to the repository's agents."
            ),
        )
    read = read_subagents(working_tree)
    if not read.roster:
        return SubagentAttachment(
            attached=False,
            reason=(
                "this run may delegate, but the working tree defines no subagents (looked for *.md "
                "under .cursor/agents, .claude/agents and .codex/agents)"
            ),
            refused=tuple(str(refusal["code"]) for refusal in read.refusals),
        )
    return SubagentAttachment(
        attached=True,
        agents=tuple(str(entry["name"]) for entry in read.roster),
        refused=tuple(str(refusal["code"]) for refusal in read.refusals),
    )
