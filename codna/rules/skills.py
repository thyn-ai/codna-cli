"""Skills: discovery, the catalogue, and the pull — the ONE skill reader.

**The gap this closes.** Codna discovered skills and could not reach them. The vendored SDK has a
skills executor that reads ``.clinerules/skills``, ``.cline/skills`` and ``.agents/skills``; Theia has
its own ``SkillService`` scanning ``.prompts/skills`` and ``.agents/skills``. Neither reads
``.cursor/skills``, and the two disagree about everything else — the same split-reader defect rules
had before ``cli/codna/rules/`` became the single engine. So skills get the same treatment, and they
reuse the same machinery rather than growing a third reader:

============================  ==========================================================
concern                       module reused
============================  ==========================================================
frontmatter parsing           :mod:`codna.rules.frontmatter`
``paths`` glob semantics      :mod:`codna.rules.globs` (identical to a rule's ``globs:``)
traversal bounds, containment :mod:`codna.rules.fsutil`
catalogue budget + reporting  :mod:`codna.rules.render` (``render_catalogue``)
pull-by-name                  the ``request_rule`` shape (``load_skill``)
============================  ==========================================================

**Progressive disclosure is the whole design.** A skill's body is never in the prompt. The prompt
carries one line per skill — name, description, ``paths`` — and the body arrives only when the model
calls ``codna_load_skill`` or the user types ``/skill-name``. A repository with 60 skills therefore
costs 60 lines of catalogue instead of 60 full prompts, and a skill nobody asks for costs nothing but
its line.

**The catalogue is bounded by the SAME budget as rules.** Not a second budget: the skills catalogue is
appended to the same injected-context text, so it must count against the same ceiling and report the
same way when it does not fit. :func:`codna.rules.render.render_catalogue` is that one report, shared
with the rule catalog, so a user sees one consistent account of what was left out and why.

**Manual-only is a permission, not a second path.** ``disable-model-invocation: true`` keeps the skill
in the catalogue (so ``/skill-name`` and a human reading ``codna skills`` can see it) and marks the
entry ``manual``; :func:`load_skill` still resolves it. Invoked and auto-attached skills differ only in
WHO ASKS — never in how the body is obtained.

**Two other readers exist in this repository, and they are why this module is the union.** The vendored
SDK reads ``.clinerules/skills`` / ``.cline/skills`` / ``.agents/skills`` for its ``skills`` TOOL, and
Theia's ``SkillService`` reads ``.prompts/skills`` / ``.agents/skills`` for its own catalogue. Neither
reads ``.cursor/skills``, and neither is a place a repository's conventions should have to be copied to.
The overlap with ``.agents/skills`` is deliberate and harmless: the catalogued line names
``codna_load_skill``, whose handler resolves the same file the SDK's tool would, so a skill in that
directory is reachable either way — and the IDE's ``/`` commands are de-duplicated against Theia's own
registrations (``codna-skill-command-model.ownedElsewhere``), so one skill never becomes two commands.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from . import frontmatter as _frontmatter
from . import globs as _globs
from . import fsutil
from .render import CatalogueLine, render_catalogue
from .skill_model import (
    BUNDLED_DIRECTORIES,
    MAX_BUNDLED_FILES,
    Skill,
    SkillMode,
    SkillScope,
    SkillWarning,
)

__all__ = [
    "CATALOGUE_HEADER",
    "MAX_SKILLS",
    "SkillCatalogueResult",
    "SkillDiscovery",
    "SkillSearchDir",
    "catalogue",
    "discover",
    "load_skill",
    "search_dirs",
]

#: The header the prompt carries above the catalogue. Names the tool, because a catalogue line the
#: model cannot act on is a line that costs tokens and buys nothing — and explains the one marking the
#: lines carry, because a marker the model does not understand is an invitation to ignore the rule it
#: encodes.
CATALOGUE_HEADER = (
    "# SKILLS AVAILABLE ON REQUEST\n"
    "Each skill below is an INDEX entry: its instructions are NOT included. Pull one by name with "
    "the codna_load_skill tool when its description matches the task. A skill marked "
    "[manual-only: /name] may be invoked by the USER with /name and must not be pulled on your own "
    "initiative."
)

#: Bound on how many skills are read at all. A tree with more than this is not a skills folder; the
#: overflow is reported rather than silently truncated (`SkillWarning` ``skills_truncated_by_count``).
MAX_SKILLS = 500

#: The frontmatter keys parsed into first-class fields; everything else is preserved in ``Skill.extra``.
_KNOWN_KEYS = frozenset({
    "name", "description", "paths", "disable-model-invocation", "disable_model_invocation",
    "disablemodelinvocation", "license", "compatibility", "metadata", "allowed-tools",
    "allowed_tools", "allowedtools",
})

_SKILL_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class SkillSearchDir:
    """One directory searched for skills, and the layer it belongs to."""

    #: Absolute path.
    path: str
    #: Repository-relative label (``.cursor/skills``) or ``~/…`` for a user directory.
    label: str
    scope: SkillScope


@dataclass(frozen=True)
class SkillDiscovery:
    """Every skill found, plus every problem found while finding it. Never a silent partial read."""

    skills: tuple[Skill, ...] = ()
    warnings: tuple[SkillWarning, ...] = ()

    def names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.skills)


@dataclass(frozen=True)
class SkillCatalogueResult:
    """The catalogue text, the entries that fit, and the complete account of what did not."""

    text: str = ""
    catalogued: tuple[Skill, ...] = ()
    dropped: tuple[CatalogueLine, ...] = ()
    budget_bytes: int = 0
    used_bytes: int = 0


def search_dirs(repo_dir: str | None, home: str | None = None, *, include_user: bool = True) -> list[SkillSearchDir]:
    """The directories searched for skills, in SEARCH ORDER.

    Codna-native first, then the editors Codna imports from, then the user's machine-local layer. The
    order is the precedence order (see ``SkillScope``) and is data rather than a condition inside the
    walk, so reading this function is enough to know which of two same-named skills wins.

    ``.claude/skills`` and ``.codex/skills`` are here for the same reason ``.cursor/rules`` is in the
    rule reader: a repository migrating from either must not have to move its files to be honoured.
    """
    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    dirs: list[SkillSearchDir] = [
        SkillSearchDir(os.path.join(root, ".codna", "skills"), ".codna/skills", SkillScope.CODNA),
        SkillSearchDir(os.path.join(root, ".cursor", "skills"), ".cursor/skills", SkillScope.CURSOR),
        SkillSearchDir(os.path.join(root, ".agents", "skills"), ".agents/skills", SkillScope.AGENTS),
        SkillSearchDir(os.path.join(root, ".claude", "skills"), ".claude/skills", SkillScope.CLAUDE),
        SkillSearchDir(os.path.join(root, ".codex", "skills"), ".codex/skills", SkillScope.CODEX),
    ]
    if include_user:
        base = _home(home)
        if base:
            dirs.extend([
                SkillSearchDir(os.path.join(base, ".codna", "skills"), "~/.codna/skills", SkillScope.USER),
                SkillSearchDir(os.path.join(base, ".cursor", "skills"), "~/.cursor/skills", SkillScope.USER),
                SkillSearchDir(os.path.join(base, ".agents", "skills"), "~/.agents/skills", SkillScope.USER),
                SkillSearchDir(os.path.join(base, ".claude", "skills"), "~/.claude/skills", SkillScope.USER),
                SkillSearchDir(os.path.join(base, ".codex", "skills"), "~/.codex/skills", SkillScope.USER),
            ])
    return dirs


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


# ---------------------------------------------------------------------------------------------
# reading one SKILL.md
# ---------------------------------------------------------------------------------------------

def _as_bool(value: object, *, key: str, source_path: str) -> tuple[bool, list[SkillWarning]]:
    """A boolean frontmatter key, refusing a non-boolean rather than coercing it.

    ``disable-model-invocation: "yes"`` coerced to ``True`` would be a coincidence; coerced to
    ``False`` it would be a manual-only skill silently attached to every turn. Refusing is the only
    answer that cannot be wrong.
    """
    if value is None:
        return False, []
    if isinstance(value, bool):
        return value, []
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in ("true", "yes", "on"):
            return True, []
        if lowered in ("false", "no", "off"):
            return False, []
    return False, [SkillWarning(
        code="skill_frontmatter_invalid",
        message=f"{key} must be true or false; ignored",
        source_path=source_path,
    )]


def _fold_switches(metadata: dict[str, object], source_path: str) -> tuple[bool, list[SkillWarning]]:
    """Read ``disable-model-invocation`` in the spellings that actually arrive.

    Cursor documents the kebab-case key; YAML keys are often written snake_case or camelCase by
    hand. All three are accepted, and DISAGREEMENT is reported rather than resolved by a coin flip.
    """
    spellings = ("disable-model-invocation", "disable_model_invocation", "disableModelInvocation")
    values: list[bool] = []
    warnings: list[SkillWarning] = []
    for spelling in spellings:
        if spelling not in metadata:
            continue
        parsed, extra = _as_bool(metadata[spelling], key=spelling, source_path=source_path)
        values.append(parsed)
        warnings.extend(extra)
    if len(values) > 1 and len(set(values)) > 1:
        warnings.append(SkillWarning(
            code="skill_frontmatter_conflict",
            message="disable-model-invocation spellings disagree; treating the skill as manual-only",
            source_path=source_path,
        ))
        return True, warnings
    return (any(values), warnings)


def _skill_from_dir(
    skill_dir: str,
    *,
    rel_dir: str,
    scope: SkillScope,
    display: str,
    real_root: str,
) -> tuple[Skill | None, list[SkillWarning]]:
    """Read ``<skill_dir>/SKILL.md`` into a :class:`Skill`, or report why it is not one."""
    warnings: list[SkillWarning] = []
    path = os.path.join(skill_dir, _SKILL_FILENAME)
    if not fsutil.contained(real_root, path):
        warnings.append(SkillWarning(
            code="skill_file_outside_repository",
            message="resolves outside the repository; skipped",
            source_path=display,
        ))
        return None, warnings

    text = fsutil.read_text(path)
    if text is None:
        too_big = (fsutil.size_of(path) or 0) > fsutil.MAX_FILE_BYTES
        warnings.append(SkillWarning(
            code="skill_file_too_large" if too_big else "skill_file_unreadable",
            message=(f"larger than {fsutil.MAX_FILE_BYTES} bytes; skipped" if too_big
                     else "could not be read; skipped"),
            source_path=display,
        ))
        return None, warnings

    try:
        metadata, body, had_frontmatter = _frontmatter.parse_markdown(text, source_path=display)
    except _frontmatter.FrontmatterError as exc:
        warnings.append(SkillWarning(
            code="skill_frontmatter_invalid",
            message=str(exc),
            source_path=display,
        ))
        return None, warnings

    # The FOLDER is the identity, so its name is the fallback when the frontmatter omits `name`. Using
    # the frontmatter name when present honours "the name the author declared", and falling back to the
    # folder keeps a skill addressable even when its frontmatter is minimal.
    folder_name = os.path.basename(skill_dir.rstrip(os.sep))
    raw_name = metadata.get("name")
    name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else folder_name
    raw_description = metadata.get("description")
    description = str(raw_description).strip() if isinstance(raw_description, str) else ""

    try:
        paths = _globs.parse_globs(metadata.get("paths"))
    except ValueError as exc:
        warnings.append(SkillWarning(
            code="skill_frontmatter_invalid", message=str(exc), source_path=display,
        ))
        return None, warnings

    manual, extra_warnings = _fold_switches(metadata, display)
    warnings.extend(extra_warnings)

    if not had_frontmatter:
        # A SKILL.md with no frontmatter is legal (the folder names it) but it has no description, so
        # the catalogue line would be uninformative. Kept, and said out loud.
        warnings.append(SkillWarning(
            code="skill_without_frontmatter",
            message="no frontmatter; catalogued under its folder name with no description",
            source_path=display,
        ))

    bundled, truncated = fsutil.iter_bundled_files(
        skill_dir, BUNDLED_DIRECTORIES, limit=MAX_BUNDLED_FILES,
    )
    if truncated:
        warnings.append(SkillWarning(
            code="skill_bundled_files_truncated",
            message=f"more than {MAX_BUNDLED_FILES} files beside SKILL.md; the listing stops there",
            source_path=display,
        ))

    extra = {
        str(key): value for key, value in metadata.items()
        if str(key).casefold() not in _KNOWN_KEYS
    }
    del rel_dir  # the label already carries the repository-relative location
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        source_path=display,
        scope=scope,
        bundled=tuple(bundled),
        paths=paths,
        disable_model_invocation=manual,
        extra=extra,
    ), warnings


# ---------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------

def discover(
    repo_dir: str | None,
    *,
    home: str | None = None,
    include_user: bool = True,
) -> SkillDiscovery:
    """Every skill visible to a turn in ``repo_dir``. Deterministic, bounded, and non-silent."""
    warnings: list[SkillWarning] = []
    skills: list[Skill] = []

    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    if not os.path.isdir(root):
        return SkillDiscovery(warnings=(SkillWarning(
            code="skills_root_missing",
            message="not a directory; no skills were read",
            source_path=fsutil.posix(root),
        ),))
    real_root = os.path.realpath(root)

    for entry in search_dirs(root, home, include_user=include_user):
        if not os.path.isdir(entry.path):
            continue  # the ordinary case: a repository has no skills in four of five directories
        real_dir = os.path.realpath(entry.path)
        # A USER directory legitimately lives outside the repository, so containment is only required
        # of the repository's own layers. Every FILE is still checked against its own root below.
        in_repo = real_dir == real_root or real_dir.startswith(real_root + os.sep)
        if not in_repo and not entry.scope.is_user:
            warnings.append(SkillWarning(
                code="skills_dir_outside_repository",
                message="symlinked outside the repository; skipped",
                source_path=entry.label,
            ))
            continue
        file_root = real_root if in_repo else real_dir
        for skill_dir, rel_dir in fsutil.iter_dirs_containing(entry.path, _SKILL_FILENAME):
            display = fsutil.posix(
                f"{entry.label}/{rel_dir}/{_SKILL_FILENAME}"
            )
            skill, extra = _skill_from_dir(
                skill_dir,
                rel_dir=rel_dir,
                scope=entry.scope,
                display=display,
                real_root=file_root,
            )
            warnings.extend(extra)
            if skill is not None:
                skills.append(skill)

    ordered = _dedupe(skills, warnings)
    if len(ordered) > MAX_SKILLS:
        overflow = ordered[MAX_SKILLS:]
        ordered = ordered[:MAX_SKILLS]
        warnings.append(SkillWarning(
            code="skills_truncated_by_count",
            message=(
                f"{len(overflow)} skill(s) beyond the {MAX_SKILLS} limit were not read: "
                + ", ".join(skill.source_path for skill in overflow[:10])
            ),
        ))
    return SkillDiscovery(skills=tuple(ordered), warnings=tuple(warnings))


def _dedupe(skills: list[Skill], warnings: list[SkillWarning]) -> list[Skill]:
    """One skill per (scope, source_path), and one per NAME — search order decides.

    Two identities are deduplicated, and they are different questions:

    * ``(scope, source_path)`` — the same file reached twice must not catalogue twice.
    * ``name`` — a name is how a skill is INVOKED, so two skills answering to one name would make
      ``/skill-name`` ambiguous. The first in search order wins, and every shadowed skill is reported
      by its full path so a user who added one and watched it do nothing has an answer.
    """
    seen_paths: set[tuple[SkillScope, str]] = set()
    seen_names: dict[str, Skill] = {}
    kept: list[Skill] = []
    for skill in sorted(skills, key=lambda item: item.sort_key()):
        path_key = (skill.scope, skill.source_path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        winner = seen_names.get(skill.name_key())
        if winner is not None:
            warnings.append(SkillWarning(
                code="skill_name_shadowed",
                message=(
                    f"a skill named {skill.name!r} is already defined at {winner.source_path}; "
                    "this one is shadowed (the first in search order wins)"
                ),
                source_path=skill.source_path,
            ))
            continue
        seen_names[skill.name_key()] = skill
        kept.append(skill)
    return kept


# ---------------------------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------------------------

def _catalogue_line(skill: Skill) -> str:
    """One skill as it appears in the prompt: name, invocation, description, ``paths``.

    Enough to decide whether to pull it, and never the body — that is the progressive-disclosure
    contract, and it is what keeps a 60-skill repository from costing 60 prompts.
    """
    suffix = " [manual-only: /" + skill.name + "]" if skill.mode is SkillMode.MANUAL else ""
    detail = skill.description or "no description"
    patterns = ", ".join(skill.paths)
    scope = f" — paths: {patterns}" if patterns else ""
    return f"- {skill.name}{suffix}: {detail}{scope}"


def catalogue(
    discovery: SkillDiscovery,
    *,
    files_in_context: Iterable[str] | None = None,
    explicit_mentions: Iterable[str] | None = None,
    max_bytes: int = 4096,
    include_catalog: bool = True,
) -> SkillCatalogueResult:
    """The catalogue text for this turn, within ``max_bytes``, reporting everything left out.

    ``files_in_context`` scopes ``paths``-restricted skills: a skill declaring ``paths: "src/**/*.ts"``
    is absent from a docs-only turn, exactly as a ``globs``-scoped rule is. ``None`` means the caller
    did not say — a durable run may edit anything, so a ``paths``-scoped skill cannot be ruled out and
    is kept. ``[]`` means the caller said the set is empty and is honoured literally.

    ``explicit_mentions`` are skills the caller asked for BY NAME. A named skill is catalogued even when
    its ``paths`` do not match — the ask is evidence enough — which is what makes ``/skill-name`` work
    in a turn whose file set the host does not know.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")

    files_known = files_in_context is not None
    files = _normalise_paths(files_in_context)
    mentions = _normalise_mentions(explicit_mentions)

    lines: list[CatalogueLine] = []
    eligible: list[Skill] = []
    for skill in discovery.skills:
        requested = skill.name_key() in mentions
        # Manual-only skills are still CATALOGUED (with their `/name`), because a human and the model
        # both need to know it exists to invoke it. They are never auto-attached — and here "attached"
        # means offered as a BODY, which a catalogue never does. The distinction that matters is the
        # one the mode names: a manual-only entry is reachable by `/skill-name` and never chosen by the
        # model on its own, which is precisely what the catalogue line's `[manual-only: /name]` says.
        if not requested and files_known and not skill.matches_any_file(files):
            continue
        eligible.append(skill)
        lines.append(CatalogueLine(
            name=skill.name,
            source_path=skill.source_path,
            line=_catalogue_line(skill),
        ))

    if not include_catalog or not lines:
        return SkillCatalogueResult(
            text="",
            catalogued=(),
            dropped=(),
            budget_bytes=max_bytes,
            used_bytes=0,
        )

    rendered = render_catalogue(lines, header=CATALOGUE_HEADER, max_bytes=max_bytes)
    kept_names = {line.name for line in rendered.kept}
    return SkillCatalogueResult(
        text=rendered.text,
        catalogued=tuple(skill for skill in eligible if skill.name in kept_names),
        dropped=rendered.dropped,
        budget_bytes=max_bytes,
        used_bytes=len(rendered.text.encode("utf-8")),
    )


def _normalise_paths(files: Iterable[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in files or ():
        rel = Skill.normalise_rel(raw)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        ordered.append(rel)
    return tuple(ordered)


def _normalise_mentions(mentions: Iterable[str] | None) -> frozenset[str]:
    """Skill names an explicit reference names, case-folded, with ``/``/``@``/``.md`` stripped."""
    out: set[str] = set()
    for raw in mentions or ():
        text = str(raw or "").strip()
        if text[:1] in ("/", "@"):
            text = text[1:]
        text = text.strip()
        if not text:
            continue
        out.add(text.casefold())
        base = text.rsplit("/", 1)[-1]
        if base.casefold().endswith(".md"):
            out.add(base[: -len(".md")].casefold())
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# the pull
# ---------------------------------------------------------------------------------------------

def load_skill(
    repo_dir: str | None,
    name: str,
    *,
    home: str | None = None,
    include_user: bool = True,
) -> Skill | None:
    """Pull one skill by name — the mechanism behind ``codna_load_skill`` and ``/skill-name``.

    This is deliberately the SAME shape as :func:`codna.rules.request_rule`, because rules and skills
    must not present a user with two mechanisms for "the prompt catalogued this; give me the body".
    Matching is by the same case-folded key the catalogue printed, and the first match in search order
    wins — identically to rendering, so pulling a shadowed skill cannot return a body the catalogue
    would have refused to offer.

    Returns ``None`` when no skill matches. That is a real answer, and the caller names it in its error.
    A manual-only skill IS loadable: ``disable-model-invocation`` restricts who may ASK, not whether
    the answer exists.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    keys = _normalise_mentions((wanted,))
    for skill in discover(repo_dir, home=home, include_user=include_user).skills:
        if skill.name_key() in keys:
            return skill
    return None
