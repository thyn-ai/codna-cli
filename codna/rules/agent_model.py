"""The subagent model: one normalised shape for every agent markdown file Codna can read.

An :class:`Agent` is what a subagent FILE means once Codna has understood it, and — like
:class:`~codna.rules.model.Rule` and :class:`~codna.rules.skill_model.Skill` — it is deliberately
source-agnostic: ``.cursor/agents``, ``.claude/agents``, ``.codex/agents`` and the same three under
``~`` all become the same fields plus their provenance.

**Identity is the FILE, and that is where this differs from a skill.** A skill is the directory that
CONTAINS ``SKILL.md`` (Cursor's rule, and what makes category folders work). A subagent is the
markdown file itself, and its ``name`` defaults to the FILENAME — so ``.cursor/agents/verifier.md`` is
the agent ``verifier``, and ``.cursor/agents/review/security.md`` is ``security`` (the directory is
grouping, not identity). Getting this backwards would make two agents in one folder collide, which is
the opposite of how a user organises a roster.

**Precedence is Cursor's, written down once.** Project beats user, and within either layer ``.cursor``
beats ``.claude`` beats ``.codex``. :data:`SCOPE_RANK` is the single statement of that order; the
discovery sort and the shadow report both read it, so a reader cannot invent a second ordering.

**The one restriction knob is ``readonly``.** Cursor's subagent format has no tool allow-list —
per-tool limits are hooks — and inventing one here would be a second vocabulary for the same intent.
``readonly: true`` maps onto the engine's EXISTING read-only profile (``repository_review``), so the
flag is a name for a profile that already exists rather than a parallel mechanism beside it.

**``is_background`` has no engine behind it, so it is refused rather than faked.** Every delegated
agent the engine can build is awaited (``createConfiguredAgentTools`` awaits ``subAgent.run``), so a
background subagent would either block silently — a file claiming something that is not happening — or
need an output directory we do not have. A file that asks for it is REFUSED with a code that says so,
because a user's honest "this did not take effect" beats a quiet lie.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "AGENT_NAME_PATTERN",
    "Agent",
    "AgentRefusal",
    "AgentScope",
    "AgentWarning",
    "BUILTIN_AGENT_IDS",
    "MAX_NAME_LENGTH",
    "SCOPE_RANK",
    "validate_name",
]

#: What a subagent name may be: lowercase words joined by single hyphens. Deliberately narrow, because
#: the name is what a parent agent types to delegate and what a diagnostic prints — a name with a space
#: or a capital is a name two surfaces will disagree about. It is VALIDATED, never repaired: silently
#: lowercasing `MyAgent` would make the file work and hide the typo that says the user expected
#: something else (a display name, an `@`-mention, a case-sensitive id).
AGENT_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: The longest name a file may declare. Matches the tool-name budget the engine builds around a
#: configured agent (`subagent_<name>`, capped at 64 chars in `configured-agent-tool.ts`).
MAX_NAME_LENGTH = 64


class AgentScope(str, Enum):
    """Where a subagent came from. The enum ORDER is the precedence order (lowest rank wins)."""

    CURSOR = "cursor"                  # .cursor/agents
    CLAUDE = "claude"                  # .claude/agents
    CODEX = "codex"                    # .codex/agents
    USER_CURSOR = "user-cursor"        # ~/.cursor/agents
    USER_CLAUDE = "user-claude"        # ~/.claude/agents
    USER_CODEX = "user-codex"          # ~/.codex/agents

    @property
    def is_user(self) -> bool:
        """Whether this layer is the machine-local one (which legitimately lives outside a repo)."""
        return self in (AgentScope.USER_CURSOR, AgentScope.USER_CLAUDE, AgentScope.USER_CODEX)


#: Search order, written down ONCE. A lower rank is consulted first and wins a name conflict, which is
#: why `Agent.sort_key` reads this and no caller invents its own ordering. Project-before-user and
#: `.cursor`-before-`.claude`-before-`.codex` are both encoded here as one flat order, so the two rules
#: cannot be applied inconsistently.
SCOPE_RANK: dict[AgentScope, int] = {
    AgentScope.CURSOR: 0,
    AgentScope.CLAUDE: 1,
    AgentScope.CODEX: 2,
    AgentScope.USER_CURSOR: 3,
    AgentScope.USER_CLAUDE: 4,
    AgentScope.USER_CODEX: 5,
}

#: The ids of the agents this build ALREADY ships (the ``@``-agent roster the IDE resolves personas
#: from). A file-defined agent is an EXTENSION of that roster, so a file that claims one of these ids
#: is a REFUSAL: the name is what the parent and the ``@``-mention both key on, and two different
#: agents answering to one id is precisely the ambiguity this feature must not introduce.
#:
#: Mirrored from ``ide/packages/codna-engine-client/src/policy.ts`` and asserted against it by
#: `tests/test_agents_engine.py::test_the_builtin_roster_matches_the_ides`, which PARSES the
#: TypeScript. A mirror that is merely believed is a drift waiting to happen; a parser makes the drift
#: a failing test that names the id.
BUILTIN_AGENT_IDS: frozenset[str] = frozenset({
    "architect",
    "projectinfo",
    "explore",
    "code-reviewer",
    "pr-reviewer",
    "universal",
    "orchestrator",
    "chat session naming",
    "chat-session-summary-agent",
    "coder",
    "apptester",
    "command",
    "github",
    "createskill",
})


@dataclass(frozen=True)
class AgentWarning:
    """A structured, non-fatal problem found while reading agents.

    Every one of these is a case that would otherwise be silent: an agent with no description (so the
    parent cannot decide when to delegate to it), an agent shadowed by a higher-precedence file of the
    same name, a directory that left the repository through a symlink, a catalogue line that did not
    fit the budget.
    """

    code: str
    message: str
    source_path: str = ""


@dataclass(frozen=True)
class AgentRefusal:
    """A file that did NOT become an agent, and exactly why.

    Distinct from :class:`AgentWarning` on purpose. A warning is "this agent is loaded and something
    about it is worth knowing"; a refusal is "this file is not an agent". Collapsing the two would let
    a user's typo look like "my agent doesn't work" with no way to tell which of the two happened.
    """

    code: str
    message: str
    source_path: str = ""


@dataclass(frozen=True)
class Agent:
    """One subagent, normalised.

    ``body`` is the markdown after the frontmatter: the subagent's SYSTEM PROMPT, which is what the
    engine hands a delegated agent. ``source_path`` is a POSIX, machine-independent label
    (repository-relative, or ``~/…`` for a user agent) — it is what every diagnostic names, so a
    message can be acted on.
    """

    name: str
    description: str
    body: str
    source_path: str
    scope: AgentScope
    #: ``inherit`` (run on the parent's model) or an explicit model id.
    model: str = "inherit"
    #: No file edits and no state-changing shell commands — mapped to the engine's read-only profile.
    readonly: bool = False
    #: Frontmatter keys Codna does not act on. Preserved for forward compatibility and diagnostics.
    extra: dict[str, object] = field(default_factory=dict)

    # -- identity -----------------------------------------------------------------------------

    def name_key(self) -> str:
        """The agent's name, case-folded — what a delegation or a pull-by-name matches."""
        return self.name.casefold()

    def sort_key(self) -> tuple[int, str, str]:
        """Deterministic ordering: precedence, then path, then name.

        The path tail makes the order total, so two runs over the same tree produce byte-identical
        catalogues — and the scope head is what makes a ``.cursor`` agent win over a ``.claude`` agent
        of the same name regardless of where either lives on disk.
        """
        return (SCOPE_RANK[self.scope], self.source_path, self.name)

    @property
    def inherits_model(self) -> bool:
        return self.model.strip().casefold() == "inherit"

    def to_payload(self) -> dict[str, object]:
        """The agent as a create-session field. The ONE shape a host sends and the engine reads.

        Named once here rather than assembled at each call site: the bridge, the CLI's JSON output and
        the IDE's create request must agree about what an agent IS, and a hand-built dict per caller is
        how they would stop agreeing.
        """
        return {
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "source_path": self.source_path,
            "scope": self.scope.value,
            "model": self.model,
            "readonly": self.readonly,
        }

    @staticmethod
    def normalise_rel(path: str) -> str:
        """Normalise a caller-supplied path to a POSIX repository-relative form."""
        text = str(path or "").replace("\\", "/").strip()
        if not text:
            return ""
        text = posixpath.normpath(text)
        while text.startswith("./"):
            text = text[2:]
        if text.startswith("/"):
            text = text.lstrip("/")
        if text == "." or text.startswith("../"):
            return ""
        return text


def validate_name(name: str) -> str | None:
    """``None`` when ``name`` is usable, else a human explanation of what is wrong with it.

    The default a name falls back to (the filename stem) goes through here too, so a file named
    ``My Agent.md`` is refused with the reason rather than quietly becoming something else.
    """
    if not name:
        return "a subagent name must not be empty"
    if len(name) > MAX_NAME_LENGTH:
        return f"a subagent name must be at most {MAX_NAME_LENGTH} characters (got {len(name)})"
    if not AGENT_NAME_PATTERN.fullmatch(name):
        return (
            f"{name!r} is not a valid subagent name: use lowercase letters, digits and single "
            "hyphens (e.g. 'security-auditor')"
        )
    return None
