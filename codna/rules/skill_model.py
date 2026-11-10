"""The skill model: one normalised shape for every ``SKILL.md`` Codna can read.

A :class:`Skill` is what a skill FOLDER means once Codna has understood it, and — like
:class:`~codna.rules.model.Rule` — it is deliberately source-agnostic: ``.codna/skills``,
``.cursor/skills``, ``.agents/skills``, ``.claude/skills`` and ``.codex/skills`` all become the same
fields plus their provenance. Everything downstream reads this type and never re-opens a file.

**Identity is the folder, not the filename.** A skill is the directory that CONTAINS ``SKILL.md``
(Cursor's documented rule), so ``.cursor/skills/testing/api/SKILL.md`` is the skill ``api`` inside the
category ``testing`` — the category is not itself a skill because it holds no ``SKILL.md``. That is
what makes category folders work without inventing a second layout convention.

**Skills are catalogued, never inlined.** Unlike a rule, a skill is almost always larger than the turn
that might use it — a skill ships a body plus ``scripts/``, ``references/`` and ``assets/``. So the
prompt carries the CATALOGUE (name, description, ``paths``) and the body arrives only when someone
asks for it, through :func:`codna.rules.skills.load_skill` (the same pulled-by-name mechanism rules
use for a ``description``-only rule). A 60-skill repository therefore costs 60 lines, not 60 prompts.

**Two invocation paths, one mechanism.** The model pulls a skill with ``codna_load_skill``; a user
types ``/skill-name`` and Theia expands it into the same pull. They differ only in WHO asks, so
``disable-model-invocation`` is a *permission* on the catalogue entry, not a second code path.
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from enum import Enum

from . import globs as _globs

__all__ = [
    "BUNDLED_DIRECTORIES",
    "SCOPE_RANK",
    "Skill",
    "SkillMode",
    "SkillScope",
    "SkillWarning",
]

#: The folders a skill may ship beside its ``SKILL.md``. Listed in a loaded skill so a caller knows
#: where its executable and reference material live WITHOUT the body having to describe them.
BUNDLED_DIRECTORIES: tuple[str, ...] = ("scripts", "references", "assets")

#: Bound on the files listed for ONE skill. A skill that ships thousands of assets is a dataset, and
#: a prompt is not the place to enumerate it; the bound is reported rather than silently applied.
MAX_BUNDLED_FILES = 200


@dataclass(frozen=True)
class SkillWarning:
    """A structured, non-fatal problem found while reading skills.

    Every one of these is a case that would otherwise be silent: a folder with no ``SKILL.md``, a
    frontmatter key of the wrong type, a directory that left the repository through a symlink, a
    catalogue line that did not fit the budget. They travel back to the caller so a surface can
    report them instead of quietly losing a skill.
    """

    code: str
    message: str
    source_path: str = ""


class SkillScope(str, Enum):
    """Where a skill came from. The enum ORDER is the search order (lowest rank wins a name conflict)."""

    CODNA = "codna"           # .codna/skills
    CURSOR = "cursor"         # .cursor/skills
    AGENTS = "agents"         # .agents/skills
    CLAUDE = "claude"         # .claude/skills
    CODEX = "codex"           # .codex/skills
    USER = "user"             # ~/.codna/skills, ~/.cursor/skills, ~/.agents/skills, …

    @property
    def is_user(self) -> bool:
        return self is SkillScope.USER


#: Search order, written down ONCE. A lower rank is consulted first and wins a name conflict, which
#: is why `Skill.sort_key` reads this and no caller invents its own ordering.
SCOPE_RANK: dict[SkillScope, int] = {
    SkillScope.CODNA: 0,
    SkillScope.CURSOR: 1,
    SkillScope.AGENTS: 2,
    SkillScope.CLAUDE: 3,
    SkillScope.CODEX: 4,
    SkillScope.USER: 5,
}


class SkillMode(str, Enum):
    """How a skill may be invoked — the one frontmatter switch, named."""

    AUTO = "auto"        # model may pull it from the catalogue
    MANUAL = "manual"    # disable-model-invocation: /skill-name only


@dataclass(frozen=True)
class Skill:
    """One skill, normalised.

    ``body`` is the ``SKILL.md`` text with the frontmatter removed. ``source_path`` is a POSIX,
    machine-independent label (repository-relative, or ``~/…`` for a user skill) — it is what every
    diagnostic names, so a message can be acted on. ``bundled`` are the files the skill ships beside
    its ``SKILL.md``, relative to the skill folder.
    """

    name: str
    description: str
    body: str
    source_path: str
    scope: SkillScope
    # Files, relative to the skill folder, that the body may reference. Empty for a single-file skill.
    bundled: tuple[str, ...] = ()
    # ``paths`` from the frontmatter: when non-empty, the skill is offered only for a matching file set.
    paths: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    # Frontmatter keys Codna does not act on. Preserved for forward compatibility and diagnostics.
    extra: dict[str, object] = field(default_factory=dict)

    # -- invocation ---------------------------------------------------------------------------

    @property
    def mode(self) -> SkillMode:
        """The single source of truth for who may invoke this skill."""
        return SkillMode.MANUAL if self.disable_model_invocation else SkillMode.AUTO

    # -- applicability ------------------------------------------------------------------------

    def matches_any_file(self, files: tuple[str, ...]) -> bool:
        """Whether any of this skill's ``paths`` globs matches ``files``.

        A skill with no ``paths`` is unrestricted and always matches. Matching reuses
        :mod:`codna.rules.globs` verbatim, so ``paths: "src/**/*.ts"`` means exactly what a rule's
        ``globs: "src/**/*.ts"`` means — the same ``**`` semantics, the same brace expansion, the same
        comma-separated list form. Two scoping languages would be the defect this package removes.
        """
        if not self.paths:
            return True
        return any(_globs.matches_any(self.paths, path) for path in files)

    # -- identity -----------------------------------------------------------------------------

    def name_key(self) -> str:
        """The skill's name, case-folded — what a pull-by-name or a ``/slash`` reference matches."""
        return self.name.casefold()

    def sort_key(self) -> tuple[int, str, str]:
        """Deterministic ordering: search order, then path, then name.

        The path tail makes the order total, so two runs over the same tree render byte-identically —
        and the scope head is what makes a ``.codna`` skill win over a ``.cursor`` skill of the same
        name regardless of where either lives on disk.
        """
        return (SCOPE_RANK[self.scope], self.source_path, self.name)

    @staticmethod
    def normalise_rel(path: str) -> str:
        """Normalise a caller-supplied path to a POSIX repository-relative form.

        The same contract as ``Rule.normalise_rel``: Windows separators, a leading ``./``, duplicate
        slashes and a leading ``/`` are all handled, and ``..`` is resolved rather than trusted so a
        path cannot escape the repository and match a glob it has no business matching.
        """
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
