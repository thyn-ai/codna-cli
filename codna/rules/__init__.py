"""The Codna rule engine: one reader, many consumers, four modes, deterministic rendering.

**Why this is a package and not one file.** ``test_modularity.py`` caps every module at 1000 lines,
and the rule engine genuinely has four separable concerns that each deserve their own tests: parsing
(``frontmatter``), matching (``globs``), finding (``discovery``), and deciding + formatting
(``select``, ``render``). Folding them together would put a regex translator, a filesystem walk and a
budget allocator in one namespace, and every future change to one would be reviewed against the other
two.

**The one decision this package exists to enforce.** Codna had TWO rules readers that disagreed — a
Python reader that concatenated eight fixed root filenames and truncated at 16 KB, and the vendored
SDK's "root AGENTS.md plus every ``.md`` beside it". A repository's conventions must not depend on
which of Codna's surfaces observed them. So there is now exactly one reader (``discover``), exactly
one applicability decision (``select``), and exactly one formatter (``render``); every consumer —
durable runs, the packaged agent runner, ``codna fix``, ``codna review``, IDE chat, and the
``codna_request_rule`` MCP tool — calls these and nothing else.

**What this package replaces, without regressing it.** ``cli/codna/project_rules.py`` is now a thin
compatibility re-export over :func:`guidance_for_repo`. Its callers (``fix_run``, ``review``, the
packaged runner) keep calling ``read_project_guidance`` and keep getting a string — but that string
now comes from the engine, so a rule that is scoped out is scoped out for them too, and a rule that
does not fit is reported instead of silently sliced away.

Public surface, in the order a consumer needs it:

  * :func:`discover` — find rules (``discovery.discover``)
  * :func:`select`   — decide which apply (``select.select``)
  * :func:`render`   — format within a budget (``render.render``)
  * :func:`load`     — discover + select + render + diagnostics in one call
  * :func:`guidance_for_repo` — the string-only compatibility entry point
  * :func:`request_rule` — pull one rule's body by name (the description-only mechanism)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, TextIO

from .discovery import Discovery, discover, discover_user_rules
from .agent_model import (
    BUILTIN_AGENT_IDS,
    Agent,
    AgentRefusal,
    AgentScope,
    AgentWarning,
    validate_name,
)
from .agents import (
    AGENT_CATALOGUE_HEADER,
    AgentCatalogueResult,
    AgentDiscovery,
    catalogue as agent_catalogue,
    discover as discover_agents,
    load_agent,
    readonly_profile,
)
from .model import Rule, RuleMode, RuleWarning, Scope
from .render import (
    DEFAULT_MAX_BYTES,
    CatalogueLine,
    CatalogueRender,
    DroppedRule,
    RenderResult,
    emit_diagnostics,
    render,
    render_catalogue,
    render_guidance,
)
from .select import Selection, select
from .skill_model import Skill, SkillMode, SkillScope, SkillWarning
from .skills import (
    CATALOGUE_HEADER as SKILL_CATALOGUE_HEADER,
    SkillCatalogueResult,
    SkillDiscovery,
    catalogue as skill_catalogue,
    discover as discover_skills,
    load_skill,
)

__all__ = [
    "AGENT_CATALOGUE_HEADER",
    "BUILTIN_AGENT_IDS",
    "DEFAULT_MAX_BYTES",
    "SKILL_CATALOGUE_HEADER",
    "Agent",
    "AgentCatalogueResult",
    "AgentDiscovery",
    "AgentRefusal",
    "AgentScope",
    "AgentWarning",
    "CatalogueLine",
    "CatalogueRender",
    "Discovery",
    "DroppedRule",
    "RenderResult",
    "Rule",
    "RuleMode",
    "RuleWarning",
    "Scope",
    "Selection",
    "Skill",
    "SkillCatalogueResult",
    "SkillDiscovery",
    "SkillMode",
    "SkillScope",
    "SkillWarning",
    "LoadedContext",
    "LoadedRules",
    "agent_catalogue",
    "discover",
    "discover_agents",
    "discover_skills",
    "discover_user_rules",
    "emit_diagnostics",
    "guidance_for_repo",
    "load",
    "load_agent",
    "load_context",
    "load_skill",
    "readonly_profile",
    "render",
    "render_catalogue",
    "render_guidance",
    "request_rule",
    "select",
    "skill_catalogue",
    "validate_name",
]


@dataclass(frozen=True)
class LoadedRules:
    """One turn's rules, with every drop and every read-time warning still attached."""

    discovery: Discovery
    selection: Selection
    result: RenderResult


def load(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_rules_dir: str | None = None,
    include_user: bool = True,
    include_catalog: bool = True,
) -> LoadedRules:
    """Discover, select and render in one call. Pure apart from reading the repository."""
    discovery = discover(repo_dir, user_rules_dir=user_rules_dir, include_user=include_user)
    selection = select(discovery.rules, files_in_context, explicit_mentions)
    result = render(selection, max_bytes=max_bytes, include_catalog=include_catalog)
    return LoadedRules(discovery=discovery, selection=selection, result=result)


def guidance_for_repo(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_rules_dir: str | None = None,
    include_user: bool = True,
    stream: TextIO | None = None,
    report: bool = True,
) -> str | None:
    """The repository's rules as prompt text, or ``None`` when there are none.

    ``None`` (not ``""``) is the historical contract of ``read_project_guidance`` and is preserved:
    several callers test ``if guidance:`` and a repository with no rules is the normal case, not an
    error. What is NEW is that a rule which does not fit is printed to ``stream`` (stderr by default)
    as a structured warning instead of being sliced off the end of the string.
    """
    loaded = load(
        repo_dir,
        files_in_context=files_in_context,
        explicit_mentions=explicit_mentions,
        max_bytes=max_bytes,
        user_rules_dir=user_rules_dir,
        include_user=include_user,
    )
    if report:
        emit_diagnostics(
            loaded.result,
            warnings=loaded.discovery.warnings,
            stream=stream if stream is not None else sys.stderr,
        )
    return loaded.result.text or None


@dataclass(frozen=True)
class LoadedContext:
    """One turn's rules AND skills, with every drop and every read-time warning still attached.

    ``text`` is what goes on the wire: the rules block, then the skills catalogue. They are ONE text
    under ONE budget because they are one prompt — a second budget would let the two halves each stay
    inside their ceiling while the pair overflowed the session's.
    """

    rules: LoadedRules
    skills: SkillDiscovery
    skill_catalogue: SkillCatalogueResult
    text: str
    budget_bytes: int
    used_bytes: int

    @property
    def warnings(self) -> tuple[RuleWarning | SkillWarning, ...]:
        """Every read-time warning from both readers, so a surface reports one list, not two."""
        return (*self.rules.discovery.warnings, *self.skills.warnings)


def load_context(
    repo_dir: str | None,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    skill_mentions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_rules_dir: str | None = None,
    home: str | None = None,
    include_user: bool = True,
    include_catalog: bool = True,
    include_skills: bool = True,
) -> LoadedContext:
    """Rules and skills for one turn, rendered together within ``max_bytes``.

    Ordering, and why: RULES FIRST. A rule is instruction the agent must honour on every turn it
    applies to; a skill catalogue is an INDEX of things it may ask for. When the budget cannot hold
    both, the instructions must survive and the index must degrade — the reverse would spend the
    prompt on a menu and lose the requirements.

    The skills catalogue is therefore charged against what the rules left, never against a fresh
    ceiling. That is the same relationship the rule catalogue already has to the rule bodies, so a
    caller sees one budget and one account of what did not fit.
    """
    loaded_rules = load(
        repo_dir,
        files_in_context=files_in_context,
        explicit_mentions=explicit_mentions,
        max_bytes=max_bytes,
        user_rules_dir=user_rules_dir,
        include_user=include_user,
        include_catalog=include_catalog,
    )
    rules_text = loaded_rules.result.text
    rules_used = loaded_rules.result.used_bytes

    empty = SkillCatalogueResult(budget_bytes=0)
    if not include_skills:
        return LoadedContext(
            rules=loaded_rules,
            skills=SkillDiscovery(),
            skill_catalogue=empty,
            text=rules_text,
            budget_bytes=max_bytes,
            used_bytes=rules_used,
        )

    discovery = discover_skills(repo_dir, home=home, include_user=include_user)
    # `explicit_mentions` is shared on purpose: a caller that names a skill by name says it ONCE, and
    # the rule side ignores a name it does not own. Two mention channels would make `/skill-name` and
    # `@rule` behave differently for no reason a user could explain.
    mentions = tuple(explicit_mentions or ()) + tuple(skill_mentions or ())
    remaining = max(0, max_bytes - rules_used - (2 if rules_text else 0))
    catalogue = skill_catalogue(
        discovery,
        files_in_context=files_in_context,
        explicit_mentions=mentions,
        max_bytes=remaining,
        include_catalog=include_catalog,
    )
    text = rules_text
    if catalogue.text:
        text = f"{rules_text}\n\n{catalogue.text}" if rules_text else catalogue.text
    return LoadedContext(
        rules=loaded_rules,
        skills=discovery,
        skill_catalogue=catalogue,
        text=text,
        budget_bytes=max_bytes,
        used_bytes=len(text.encode("utf-8")),
    )


def request_rule(
    repo_dir: str | None,
    name: str,
    *,
    user_rules_dir: str | None = None,
    include_user: bool = True,
) -> Rule | None:
    """Pull one rule by name — the mechanism behind the ``codna_request_rule`` MCP tool.

    This is how a ``description``-only (catalogued) rule's body is obtained. Matching is by the same
    case-folded key that precedence uses, so a name the catalog printed is a name this resolves. The
    first match in precedence order wins, identically to rendering, so pulling a shadowed rule cannot
    return the body that rendering would have refused.

    Returns ``None`` when no rule matches. That is a real answer — the caller names it in the error.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    from .discovery import discover as _discover
    from .select import normalise_mentions

    keys = normalise_mentions((wanted,))
    for rule in _discover(repo_dir, user_rules_dir=user_rules_dir, include_user=include_user).rules:
        if rule.name_key() in keys:
            return rule
    return None
