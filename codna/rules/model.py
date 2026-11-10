"""The rule model: one normalised shape for every rule Codna can read.

A ``Rule`` is what a rules FILE means once Codna has understood it, and it is deliberately
source-agnostic: ``AGENTS.md``, ``.cursor/rules/*.mdc``, ``.codna/rules/*.md``, ``.cursorrules`` and
``BUGBOT.md`` all become the same four fields plus their provenance. Everything downstream
(``select``, ``render``) reads this type and never re-opens a file, so the two readers that used to
disagree (Python's concatenation reader and the vendored SDK's root+beside-it reader) cannot disagree
about a rule again — they would have to disagree about this dataclass first.

**Scope is provenance, and precedence is scope order.** Cursor's documented order is
Team → Project → User. Codna has no hosted team dashboard, so the mapping is the repository-committed
layer that governs everyone on this checkout:

  * ``TEAM``    — ``.codna/rules/**`` (Codna-native, committed) and the legacy ``.codna/*`` files.
  * ``PROJECT`` — ``.cursor/rules/**`` (Cursor-imported) plus ``AGENTS.md`` at any depth, and the
                  legacy ``.cursorrules`` / ``BUGBOT.md`` files.
  * ``USER``    — ``~/.codna/rules/**``, the machine-local layer.

The Codna-native directory outranking the imported Cursor one is the intended consequence: a
repository that has migrated a rule into Codna's own format must not be silently overridden by the
``.cursor`` copy it replaced.

**Nesting is a subtree, not a different scope.** A rule discovered under ``pkg/api`` (whether from
``pkg/api/AGENTS.md`` or ``pkg/api/.cursor/rules/x.mdc``) carries ``base_dir="pkg/api"`` and applies
only to paths under it. It keeps its scope for precedence; specificity only breaks ties *within* a
scope.
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from enum import Enum

from . import globs as _globs

__all__ = ["Rule", "RuleMode", "RuleWarning", "SCOPE_RANK", "Scope"]


@dataclass(frozen=True)
class RuleWarning:
    """A structured, non-fatal problem found while reading rules.

    Every one of these is a case that used to be silent: a file too large to read, frontmatter that
    could not be parsed, a rule with no way to attach, a directory that left the repository through a
    symlink. They travel back to the caller so a run can report them instead of quietly losing a rule.
    """

    code: str
    message: str
    source_path: str = ""


class Scope(str, Enum):
    """Where a rule came from. The enum ORDER is the precedence order (lowest rank wins)."""

    TEAM = "team"
    PROJECT = "project"
    USER = "user"


# Precedence: a lower rank is consulted first and wins a name conflict. This is the ONE place the
# order is written down; `select` sorts by it and never invents its own ordering.
SCOPE_RANK: dict[Scope, int] = {Scope.TEAM: 0, Scope.PROJECT: 1, Scope.USER: 2}


class RuleMode(str, Enum):
    """How a rule attaches to a turn — the four Cursor modes, named."""

    ALWAYS = "always"        # alwaysApply: true
    GLOB = "glob"            # globs: … → only when a matching file is in context
    CATALOG = "catalog"      # description only → catalogued, pulled on request
    MANUAL = "manual"        # none of the above → explicit @-mention only


@dataclass(frozen=True)
class Rule:
    """One rule, normalised.

    ``body`` is the rule text with the frontmatter removed. ``source_path`` is a POSIX,
    machine-independent label (repository-relative for repository rules, ``~/…`` for user rules) —
    it is what the rendered prompt and every diagnostic name, so a message can be acted on.
    """

    name: str
    body: str
    source_path: str
    scope: Scope
    # Repository-relative directory the rule is scoped to ("" = whole repository). Enumeration and
    # sorting are stable because this is a plain string.
    base_dir: str = ""
    description: str = ""
    globs: tuple[str, ...] = ()
    always_apply: bool = False
    # True when the SOURCE's default is always-on (AGENTS.md, .codna/rules, .cursorrules,
    # BUGBOT.md). Kept as a field rather than re-derived from ``source_path`` so a caller holding a
    # hand-built Rule cannot accidentally change how a legacy file behaves.
    legacy_default_always: bool = False
    # Frontmatter keys Codna does not act on. Preserved for forward compatibility and diagnostics.
    extra: dict[str, object] = field(default_factory=dict)

    # -- mode ---------------------------------------------------------------------------------

    @property
    def mode(self) -> RuleMode:
        """The single source of truth for which of the four modes this rule is in.

        Order matters and is Cursor's: an explicit ``alwaysApply`` beats ``globs``, and a rule with
        neither but a description is catalogued rather than silently dropped.
        """
        if self.always_apply:
            return RuleMode.ALWAYS
        if self.globs:
            return RuleMode.GLOB
        if self.description:
            return RuleMode.CATALOG
        return RuleMode.MANUAL

    # -- applicability ------------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """How deeply nested the rule's subtree is; 0 for a repository-root rule."""
        return 0 if not self.base_dir else self.base_dir.count("/") + 1

    def scope_relative(self, rel_path: str) -> str | None:
        """``rel_path`` re-expressed relative to this rule's subtree, or None when outside it."""
        if not self.base_dir:
            return rel_path
        if rel_path == self.base_dir:
            return ""
        prefix = self.base_dir + "/"
        if rel_path.startswith(prefix):
            return rel_path[len(prefix):]
        return None

    def in_subtree(self, files: tuple[str, ...]) -> bool:
        """Whether this rule's subtree is in play for the given file set.

        Only meaningful when the caller NAMED a file set — ``select`` decides that, because an unknown
        set is a different question from an empty one. Given a set, a nested rule applies only when some
        file lies inside its subtree.
        """
        if not self.base_dir:
            return True
        return any(self.scope_relative(path) is not None for path in files)

    def matches_file(self, rel_path: str) -> bool:
        """Whether any of this rule's globs matches ``rel_path`` (as seen from its subtree)."""
        inner = self.scope_relative(rel_path)
        if inner is None:
            return False
        return _globs.matches_any(self.globs, inner)

    def matches_any_file(self, files: tuple[str, ...]) -> bool:
        return any(self.matches_file(path) for path in files)

    # -- identity -----------------------------------------------------------------------------

    def name_key(self) -> str:
        """The rule's name alone, case-folded — what an ``@``-mention or a pull-by-name matches."""
        return self.name.casefold()

    def conflict_key(self) -> str:
        """The key two rules conflict on.

        A conflict is "two rules claim the same name IN THE SAME SUBTREE". The subtree is part of the
        key because nested rules are ADDITIVE, not conflicting: ``web/AGENTS.md`` and the root
        ``AGENTS.md`` both apply to a file under ``web/``, and they refine each other rather than
        competing. Only rules that govern the same base directory and share a name are a real conflict
        (that is where Team → Project → User has to pick one).

        Case-folded because a rule name is an identifier a user types, not a case-sensitive symbol.
        """
        return f"{self.base_dir}\x00{self.name.casefold()}"

    def sort_key(self) -> tuple[int, int, str, str]:
        """Deterministic ordering for rendering: precedence scope, then specificity, then path.

        ``-depth`` puts the more specific rule first *within* a scope, so a nested override is
        consulted before the root rule it refines. The path and name tails make the order total, so
        two runs over the same tree always render byte-identically.
        """
        return (SCOPE_RANK[self.scope], -self.depth, self.source_path, self.name)

    @staticmethod
    def normalise_rel(path: str) -> str:
        """Normalise a caller-supplied path to a POSIX repository-relative form.

        Handles the shapes that actually arrive: absolute-ish separators from Windows, a leading
        ``./``, duplicate slashes, and a leading ``/``. ``..`` is resolved rather than trusted, so a
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
