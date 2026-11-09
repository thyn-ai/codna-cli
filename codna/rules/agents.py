"""Subagents: discovery of ``.cursor/agents/**/*.md`` and its five compat locations.

**The gap this closes.** Codna's delegation engine is real — the engine spawns subagents and forms
teams, and the IDE reaches both from the composer — but the ROSTER is hardcoded. A user cannot define
``verifier``, cannot mark one read-only, cannot point one at another model. Cursor defines a subagent
as a markdown file with frontmatter, so a repository migrating from Cursor has a roster Codna silently
ignores: ``rg 'cursor/agents'`` over the whole source returned nothing.

**This reuses the rule engine rather than growing a fourth reader.** The same table as
:mod:`codna.rules.skills`, and for the same reason — the split-reader defect is the one this package
exists to remove::

============================  ==========================================================
concern                       module reused
============================  ==========================================================
frontmatter parsing           :mod:`codna.rules.frontmatter`
traversal bounds, containment :mod:`codna.rules.fsutil` (``MAX_DEPTH``, ``contained``, ``iter_named_dirs``)
the ``readonly`` profile      :data:`~codna.rules.agent_model.BUILTIN_AGENT_IDS` + the engine's
                              existing ``repository_review`` table (see ``readonly_profile``)
catalogue budget + reporting  :mod:`codna.rules.render` (``render_catalogue``)
============================  ==========================================================

**Identity is the FILE, not the folder.** ``.cursor/agents/verifier.md`` is the agent ``verifier``
(``name`` defaults to the filename). ``.cursor/agents/review/security.md`` is ``security`` — the
directory groups, it does not name. A skill is the opposite (the folder containing ``SKILL.md``), and
the two must not be conflated: a folder-identity rule here would make every agent in one directory
answer to that directory's name.

**Precedence is Cursor's, in ONE place.** :data:`~codna.rules.agent_model.SCOPE_RANK` says project
beats user and ``.cursor`` beats ``.claude`` beats ``.codex``. Discovery sorts by it and the shadow
report reads it, so no caller can apply the two rules inconsistently.

**Nothing is silent.** A file whose frontmatter cannot be parsed is a REFUSAL with a code and the
path (a user's typo must not look like "my agent doesn't work"); a file shadowed by a
higher-precedence agent of the same name is a WARNING naming both paths; an agent with no
``description`` is loaded and WARNED, because the parent cannot decide when to delegate to it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from . import frontmatter as _frontmatter
from . import fsutil
from .agent_model import (
    BUILTIN_AGENT_IDS,
    Agent,
    AgentRefusal,
    AgentScope,
    AgentWarning,
    validate_name,
)
from .render import CatalogueLine, render_catalogue

__all__ = [
    "AGENT_CATALOGUE_HEADER",
    "MAX_AGENTS",
    "AgentCatalogueResult",
    "AgentDiscovery",
    "AgentSearchDir",
    "catalogue",
    "discover",
    "load_agent",
    "readonly_profile",
    "search_dirs",
]

#: Bound on how many agents are read at all. A tree with more than this is not an agent folder; the
#: overflow is reported rather than silently truncated (`agent_truncated_by_count`).
MAX_AGENTS = 200

#: The frontmatter keys parsed into first-class fields; everything else is preserved in ``Agent.extra``.
_KNOWN_KEYS = frozenset({"name", "description", "model", "readonly", "is_background", "isbackground"})

#: The header a catalogue carries above the agent lines. Names the roster, because a catalogue the
#: model cannot act on is a list that costs tokens and buys nothing.
AGENT_CATALOGUE_HEADER = (
    "# SUBAGENTS AVAILABLE FOR DELEGATION\n"
    "Each entry below is a subagent this repository defines. Delegate to one by name when its "
    "description matches the task. A subagent marked [readonly] cannot edit files or run "
    "state-changing shell commands."
)

#: The engine profile a ``readonly: true`` agent runs under. NOT a new flag: this is the same
#: ``repository_review`` profile the IDE already uses for @Architect and the reviewers, and the
#: engine's ``DEFAULT_APPROVAL_PROFILES`` table is what turns it into a tool list. Exposed as data so
#: a caller never invents a second spelling.
READONLY_PROFILE = "repository_review"


def readonly_profile(readonly: bool) -> str | None:
    """The approval profile a subagent runs under, or ``None`` to inherit the parent's.

    Returning ``None`` for a writable agent is the whole point: an agent that says nothing about
    permissions must not quietly narrow OR widen the session it is delegated from.
    """
    return READONLY_PROFILE if readonly else None


@dataclass(frozen=True)
class AgentSearchDir:
    """One directory searched for agents, and the layer it belongs to."""

    #: Absolute path.
    path: str
    #: Repository-relative label (``.cursor/agents``) or ``~/…`` for a user directory.
    label: str
    scope: AgentScope
    #: Whether the directory is inside the repository (containment is only required there).
    in_repo: bool
    #: ``None`` when the directory is usable, else WHY it cannot be read.
    #:
    #: A SYMLINKED directory is not followed — the containment guarantee — but it must not vanish
    #: silently either: "I symlinked my shared agent folder in and nothing appeared" is the complaint
    #: this code exists to prevent. Reported, never traversed.
    blocked: str = ""


@dataclass(frozen=True)
class AgentDiscovery:
    """Every agent found, every refusal, and every warning. Never a silent partial read."""

    agents: tuple[Agent, ...] = ()
    refusals: tuple[AgentRefusal, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(agent.name for agent in self.agents)


@dataclass(frozen=True)
class AgentCatalogueResult:
    """The catalogue text, the entries that fit, and the complete account of what did not."""

    text: str = ""
    catalogued: tuple[Agent, ...] = ()
    dropped: tuple[CatalogueLine, ...] = ()
    budget_bytes: int = 0
    used_bytes: int = 0


def _home(explicit: str | None) -> str | None:
    """The home directory to search, or ``None``. An explicit value wins so a test is hermetic."""
    if explicit is not None:
        expanded = os.path.abspath(os.path.expanduser(explicit)) if explicit else ""
        return expanded or None
    try:
        home = os.path.expanduser("~")
    except (KeyError, RuntimeError):  # no HOME: no user layer, not an error
        return None
    if not home or home == "~" or not os.path.isdir(home):
        return None
    return home


#: The three directory names, in Cursor's precedence order. One tuple, so the project and user layers
#: cannot disagree about the order of ``.cursor`` / ``.claude`` / ``.codex``.
_AGENT_DIRS: tuple[tuple[str, AgentScope, AgentScope], ...] = (
    (".cursor/agents", AgentScope.CURSOR, AgentScope.USER_CURSOR),
    (".claude/agents", AgentScope.CLAUDE, AgentScope.USER_CLAUDE),
    (".codex/agents", AgentScope.CODEX, AgentScope.USER_CODEX),
)


def search_dirs(
    repo_dir: str | None,
    home: str | None = None,
    *,
    include_user: bool = True,
) -> list[AgentSearchDir]:
    """The directories searched for agents, in SEARCH ORDER.

    The order IS the precedence order (see :data:`~codna.rules.agent_model.SCOPE_RANK`), and it is
    data rather than a condition inside the walk, so reading this function is enough to know which of
    two same-named agents wins.

    A PROJECT directory is returned for every depth at which it occurs (``pkg/api/.cursor/agents`` is a
    real, subtree-local roster), which is why this delegates the walk to ``fsutil.iter_named_dirs``
    rather than listing one path.
    """
    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    dirs: list[AgentSearchDir] = []
    for rel_name, project_scope, _user_scope in _AGENT_DIRS:
        for absolute, rel_dir in fsutil.iter_named_dirs(root, rel_name):
            dirs.append(AgentSearchDir(absolute, fsutil.posix(rel_dir), project_scope, True))
        # `iter_named_dirs` deliberately does not follow a symlinked directory, so a repository that
        # linked one in would silently contribute nothing. Report every candidate this walk refused for
        # being a symlink, at every depth, so that cannot happen quietly.
        for absolute, rel_dir in _symlinked_named_dirs(root, rel_name):
            dirs.append(AgentSearchDir(
                absolute, fsutil.posix(rel_dir), project_scope, True,
                blocked="is a symlink; not followed, because it could resolve outside the repository",
            ))
    if include_user:
        base = _home(home)
        if base:
            for rel_name, _project_scope, user_scope in _AGENT_DIRS:
                candidate = os.path.join(base, rel_name)
                if os.path.isdir(candidate):
                    dirs.append(AgentSearchDir(candidate, f"~/{rel_name}", user_scope, False))
    return dirs


def _symlinked_named_dirs(repo_dir: str, rel_name: str) -> Iterable[tuple[str, str]]:
    """``(absolute_dir, rel_dir)`` for every ``rel_name`` that exists but is a SYMLINK.

    The companion to ``fsutil.iter_named_dirs``: that walk skips symlinks on purpose (it must not be
    talked into reading an unnamed tree), so this one reports them instead of leaving the skip silent.
    Same depth bound and same pruning, so the two walks cannot disagree about what they saw.
    """
    for current, dirnames, _files in os.walk(repo_dir, followlinks=False):
        rel_current = fsutil.posix(os.path.relpath(current, repo_dir))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > fsutil.MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in fsutil.SKIP_DIRS and not os.path.islink(os.path.join(current, name))
        )
        parent = "" if rel_current == "." else rel_current
        candidate = os.path.join(current, rel_name)
        if os.path.islink(candidate):
            yield candidate, fsutil.posix(os.path.join(parent, rel_name)) if parent else rel_name


# ---------------------------------------------------------------------------------------------
# reading one agent file
# ---------------------------------------------------------------------------------------------

def _as_bool(value: object) -> tuple[bool | None, str | None]:
    """A boolean frontmatter value, refusing a non-boolean rather than coercing it.

    ``readonly: "yes"`` coerced to ``True`` would be a coincidence and coerced to ``False`` would hand
    a user's read-only reviewer write access. Refusing is the only answer that cannot be wrong, and
    the reason is returned so the caller can name it.
    """
    if value is None:
        return False, None
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in ("true", "yes", "on"):
            return True, None
        if lowered in ("false", "no", "off"):
            return False, None
    return None, "must be true or false"


def _fold_background(metadata: dict[str, object]) -> tuple[bool | None, str | None]:
    """Read ``is_background`` in the spellings that actually arrive, refusing a non-boolean.

    Both the kebab-case key Cursor documents and its camelCase twin are accepted; DISAGREEMENT between
    them is reported rather than resolved by a coin flip.
    """
    values: list[bool] = []
    for spelling in ("is_background", "isBackground"):
        if spelling not in metadata:
            continue
        parsed, problem = _as_bool(metadata[spelling])
        if problem is not None:
            return None, f"{spelling} {problem}"
        if parsed is not None:
            values.append(parsed)
    if not values:
        return False, None
    if len(set(values)) > 1:
        return None, "is_background / isBackground disagree"
    return values[0], None


def _agent_from_file(
    path: str,
    *,
    display: str,
    scope: AgentScope,
    real_root: str,
) -> tuple[Agent | None, AgentRefusal | None, list[AgentWarning]]:
    """Read one candidate file into an :class:`Agent`, or explain why it is not one."""
    warnings: list[AgentWarning] = []

    if not fsutil.contained(real_root, path):
        return None, AgentRefusal(
            code="agent_file_outside_repository",
            message="resolves outside the repository; skipped",
            source_path=display,
        ), warnings

    text = fsutil.read_text(path)
    if text is None:
        too_big = (fsutil.size_of(path) or 0) > fsutil.MAX_FILE_BYTES
        return None, AgentRefusal(
            code="agent_file_too_large" if too_big else "agent_file_unreadable",
            message=(f"larger than {fsutil.MAX_FILE_BYTES} bytes; skipped" if too_big
                     else "could not be read; skipped"),
            source_path=display,
        ), warnings

    try:
        metadata, body, had_frontmatter = _frontmatter.parse_markdown(text, source_path=display)
    except _frontmatter.FrontmatterError as exc:
        return None, AgentRefusal(
            code="agent_frontmatter_invalid", message=str(exc), source_path=display,
        ), warnings

    if not had_frontmatter:
        # Unlike a rule or a skill, an agent file with no frontmatter has no `description` and no way
        # to say `readonly`. It is still an agent (the filename names it), so it is loaded and warned
        # — refusing would make "I wrote an agent and it vanished" the answer.
        metadata = {}
        warnings.append(AgentWarning(
            code="agent_without_frontmatter",
            message="no frontmatter; loaded under its filename with no description or restrictions",
            source_path=display,
        ))

    stem = os.path.basename(path)
    if stem.casefold().endswith(".md"):
        stem = stem[: -len(".md")]

    raw_name = metadata.get("name")
    if raw_name is not None and not isinstance(raw_name, str):
        return None, AgentRefusal(
            code="agent_frontmatter_invalid",
            message=f"name must be a string (got {type(raw_name).__name__})",
            source_path=display,
        ), warnings
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else stem
    problem = validate_name(name)
    if problem is not None:
        return None, AgentRefusal(
            code="agent_name_invalid", message=problem, source_path=display,
        ), warnings
    if name.casefold() in BUILTIN_AGENT_IDS:
        # The roster is EXTENDED, not replaced, so a file may not claim a built-in id: the name is what
        # the parent and the @-mention both key on, and two agents answering to one id is the ambiguity
        # this feature must not introduce. An ERROR, never a silent override of a shipped agent.
        return None, AgentRefusal(
            code="agent_name_reserved",
            message=(
                f"{name!r} is a built-in Codna agent; a file-defined agent may extend the roster but "
                "not replace one of its ids. Rename the file or its `name:` field"
            ),
            source_path=display,
        ), warnings

    raw_description = metadata.get("description")
    if raw_description is not None and not isinstance(raw_description, str):
        return None, AgentRefusal(
            code="agent_frontmatter_invalid",
            message=f"description must be a string (got {type(raw_description).__name__})",
            source_path=display,
        ), warnings
    description = raw_description.strip() if isinstance(raw_description, str) else ""

    raw_model = metadata.get("model")
    if raw_model is not None and not isinstance(raw_model, str):
        return None, AgentRefusal(
            code="agent_frontmatter_invalid",
            message=f"model must be a string (got {type(raw_model).__name__})",
            source_path=display,
        ), warnings
    model = raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else "inherit"

    readonly, problem = _as_bool(metadata.get("readonly"))
    if problem is not None:
        return None, AgentRefusal(
            code="agent_frontmatter_invalid", message=f"readonly {problem}", source_path=display,
        ), warnings
    readonly = bool(readonly)

    background, problem = _fold_background(metadata)
    if problem is not None:
        return None, AgentRefusal(
            code="agent_frontmatter_invalid", message=problem, source_path=display,
        ), warnings
    if background:
        # The engine has no non-blocking spawn path: `createConfiguredAgentTools` AWAITS
        # `subAgent.run`, so a "background" agent would block the parent exactly like a foreground one
        # while claiming otherwise. Cursor writes background output to `~/.cursor/subagents/`; Codna has
        # no equivalent, and inventing one would be a fake. So this is a NAMED REFUSAL, not a silent
        # downgrade — the user is told the capability is missing instead of watching it do nothing.
        return None, AgentRefusal(
            code="agent_background_unsupported",
            message=(
                "`is_background: true` is not supported: Codna's delegation engine awaits every "
                "subagent (there is no non-blocking spawn path and no background-output directory), "
                "so the parent would block anyway. Remove the key, or set it to false, to load this "
                "agent as a foreground subagent"
            ),
            source_path=display,
        ), warnings

    if not body.strip():
        return None, AgentRefusal(
            code="agent_body_empty",
            message="no system prompt after the frontmatter; a subagent with no instructions cannot run",
            source_path=display,
        ), warnings

    if not description:
        # Loadable, and WARNED. The description is what the parent model reads to decide when to
        # delegate, so an agent without one is an agent the parent cannot choose deliberately.
        warnings.append(AgentWarning(
            code="agent_without_description",
            message=(
                "no `description`; the agent loads, but the parent cannot decide when to delegate to "
                "it. Add a one-line description of when to use it"
            ),
            source_path=display,
        ))

    extra = {
        str(key): value for key, value in metadata.items()
        if str(key).casefold() not in _KNOWN_KEYS
    }
    return Agent(
        name=name,
        description=description,
        body=body.strip(),
        source_path=display,
        scope=scope,
        model=model,
        readonly=readonly,
        extra=extra,
    ), None, warnings


def _markdown_files(directory: str) -> Iterable[tuple[str, str]]:
    """Yield ``(absolute, rel_dir)`` for every ``*.md`` under ``directory``, bounded and pruned.

    Recursive because a roster is grouped in folders (``agents/review/security.md``), and the folder
    is grouping rather than identity. The bounds and the pruning are ``fsutil``'s, so a traversal fix
    happens in one place for rules, skills and agents alike.
    """
    root = os.path.abspath(directory)
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        rel_current = fsutil.posix(os.path.relpath(current, root))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > fsutil.MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in fsutil.SKIP_DIRS and not os.path.islink(os.path.join(current, name))
        )
        for filename in sorted(filenames):
            if not filename.casefold().endswith(".md"):
                continue
            yield os.path.join(current, filename), ("" if rel_current == "." else rel_current)


# ---------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------

def discover(
    repo_dir: str | None,
    *,
    home: str | None = None,
    include_user: bool = True,
) -> AgentDiscovery:
    """Every subagent visible to a turn in ``repo_dir``. Deterministic, bounded, and non-silent."""
    agents: list[Agent] = []
    refusals: list[AgentRefusal] = []
    warnings: list[AgentWarning] = []

    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    if not os.path.isdir(root):
        return AgentDiscovery(warnings=(AgentWarning(
            code="agents_root_missing",
            message="not a directory; no subagents were read",
            source_path=fsutil.posix(root),
        ),))
    real_root = os.path.realpath(root)

    for entry in search_dirs(root, home, include_user=include_user):
        if entry.blocked:
            warnings.append(AgentWarning(
                code="agents_dir_outside_repository",
                message=entry.blocked,
                source_path=entry.label,
            ))
            continue
        if not os.path.isdir(entry.path):
            continue  # the ordinary case: a repository has no agents in two of three directories
        real_dir = os.path.realpath(entry.path)
        in_repo = real_dir == real_root or real_dir.startswith(real_root + os.sep)
        if entry.in_repo and not in_repo:
            warnings.append(AgentWarning(
                code="agents_dir_outside_repository",
                message="symlinked outside the repository; skipped",
                source_path=entry.label,
            ))
            continue
        # A USER directory legitimately lives outside the repository, so containment is required
        # against the directory's OWN root there. Every FILE is still checked below.
        file_root = real_root if entry.in_repo else real_dir
        for path, rel_dir in _markdown_files(entry.path):
            display = fsutil.posix(
                f"{entry.label}/{rel_dir}/{os.path.basename(path)}" if rel_dir
                else f"{entry.label}/{os.path.basename(path)}"
            )
            agent, refusal, extra = _agent_from_file(
                path, display=display, scope=entry.scope, real_root=file_root,
            )
            warnings.extend(extra)
            if refusal is not None:
                refusals.append(refusal)
            if agent is not None:
                agents.append(agent)

    ordered, shadow_warnings = _dedupe(agents)
    warnings.extend(shadow_warnings)
    if len(ordered) > MAX_AGENTS:
        overflow = ordered[MAX_AGENTS:]
        ordered = ordered[:MAX_AGENTS]
        warnings.append(AgentWarning(
            code="agents_truncated_by_count",
            message=(
                f"{len(overflow)} agent(s) beyond the {MAX_AGENTS} limit were not read: "
                + ", ".join(agent.source_path for agent in overflow[:10])
            ),
        ))
    return AgentDiscovery(agents=tuple(ordered), refusals=tuple(refusals), warnings=tuple(warnings))


def _dedupe(agents: list[Agent]) -> tuple[list[Agent], list[AgentWarning]]:
    """One agent per (scope, source_path), and one per NAME — search order decides, reported.

    Two identities, and they are different questions:

    * ``(scope, source_path)`` — the same file reached twice must not be listed twice.
    * ``name`` — a name is how a delegation is addressed, so two agents answering to one name would
      make the parent's choice ambiguous. The first in search order wins, and every shadowed file is
      reported by its full path so a user who added one and watched it do nothing has an answer.
    """
    warnings: list[AgentWarning] = []
    seen_paths: set[tuple[AgentScope, str]] = set()
    seen_names: dict[str, Agent] = {}
    kept: list[Agent] = []
    for agent in sorted(agents, key=lambda item: item.sort_key()):
        path_key = (agent.scope, agent.source_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        winner = seen_names.get(agent.name_key())
        if winner is not None:
            warnings.append(AgentWarning(
                code="agent_name_shadowed",
                message=(
                    f"a subagent named {agent.name!r} is already defined at {winner.source_path}; "
                    f"this one ({agent.source_path}) is shadowed — the first in search order wins "
                    "(project before user, .cursor before .claude before .codex)"
                ),
                source_path=agent.source_path,
            ))
            continue
        seen_names[agent.name_key()] = agent
        kept.append(agent)
    return kept, warnings


# ---------------------------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------------------------

def _catalogue_line(agent: Agent) -> str:
    """One agent as it appears in a catalogue: name, description, model, restrictions."""
    detail = agent.description or "no description"
    marks: list[str] = []
    if agent.readonly:
        marks.append("readonly")
    if not agent.inherits_model:
        marks.append(f"model: {agent.model}")
    suffix = f" [{' · '.join(marks)}]" if marks else ""
    return f"- {agent.name}{suffix}: {detail}"


def catalogue(
    discovery: AgentDiscovery,
    *,
    max_bytes: int = 2048,
    include_catalog: bool = True,
) -> AgentCatalogueResult:
    """The catalogue text for a roster, within ``max_bytes``, reporting everything left out."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")

    lines = [
        CatalogueLine(name=agent.name, source_path=agent.source_path, line=_catalogue_line(agent))
        for agent in discovery.agents
    ]
    if not include_catalog or not lines:
        return AgentCatalogueResult(budget_bytes=max_bytes, used_bytes=0)

    rendered = render_catalogue(lines, header=AGENT_CATALOGUE_HEADER, max_bytes=max_bytes)
    kept_names = {line.name for line in rendered.kept}
    return AgentCatalogueResult(
        text=rendered.text,
        catalogued=tuple(agent for agent in discovery.agents if agent.name in kept_names),
        dropped=rendered.dropped,
        budget_bytes=max_bytes,
        used_bytes=len(rendered.text.encode("utf-8")),
    )


def load_agent(
    repo_dir: str | None,
    name: str,
    *,
    home: str | None = None,
    include_user: bool = True,
) -> Agent | None:
    """Pull one agent by name — the mechanism behind ``codna agents --agent <name>``.

    The same shape as :func:`codna.rules.load_skill` and :func:`codna.rules.request_rule`: matching is
    by the case-folded key the catalogue printed, and the first match in search order wins, so pulling
    a shadowed agent cannot return a body the roster would have refused to offer.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    key = wanted.casefold()
    for agent in discover(repo_dir, home=home, include_user=include_user).agents:
        if agent.name_key() == key:
            return agent
    return None


def agents_payload(discovery: AgentDiscovery) -> list[dict[str, object]]:
    """The roster as a create-session ``agents`` field — the ONE shape a host sends.

    Empty when there is no roster, so a repository that defines none produces a request byte-identical
    to the pre-feature one (the same contract ``mcp_session_spec`` and ``swarm_session_spec`` follow).
    """
    return [agent.to_payload() for agent in discovery.agents]
