"""Gitignore-semantics ignore matching — the ONE decision every Codna repo reader asks.

Before this module, each reader carried its own hardcoded tuple of directory NAMES
(``codeunits._DEFAULT_IGNORE``, ``impact.DEFAULT_IGNORE``, ``IGNORED_DIR_NAMES``). A name
blocklist has no path semantics, no negation and no user control, so a repository that
gitignores ``data/`` still had ``data/`` embedded into Telys and recalled: the top-k budget
was spent on files the user had explicitly excluded.

This module replaces that with a real matcher. It is deliberately a dependency-free port of
gitignore's documented pattern rules rather than a new dependency: ``pathspec`` is NOT a
declared dependency of ``codna`` (it is only present transitively via build tooling), and the
shipped wheel must not grow one for a ~300-line pure function. ``cli/tests/test_ignore.py``
pins the semantics against git's documented pattern table, so the port is proven rather than
asserted.

Sources, in application order (later wins), all gitignore syntax:

1. ``.gitignore``     — implicit. A repo that ignores a path almost never wants it embedded.
2. ``.cursorignore``  — migration. A Cursor repo works unchanged.
3. ``.codnaignore``   — Codna-native, wins over both.

and a FLOOR (``FLOOR_PATTERNS``) that is evaluated FIRST and can never be negated. That is the
security property: a ``.codnaignore`` containing ``!.env`` must not re-expose a secret, and a
``!node_modules/`` must not re-index a dependency tree. User files may ADD exclusions and may
re-include inside their own entries, but they cannot reach past the floor.

Nested files are supported: a ``.codnaignore`` in ``sub/`` scopes to that subtree, which is
what makes a monorepo workable. Deeper files are applied later, so they override shallower
ones for the paths they cover.
"""
from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# The names of the per-directory ignore files, in application order (later wins).
GITIGNORE = ".gitignore"
CURSORIGNORE = ".cursorignore"
CODNAIGNORE = ".codnaignore"
IGNORE_FILENAMES: tuple[str, ...] = (GITIGNORE, CURSORIGNORE, CODNAIGNORE)

# The FLOOR for the code-unit indexer: never negotiable, never negatable.
#
# * VCS / heavy / tool-cache directories mirror the old hardcoded name blocklist, expressed as
#   path patterns so the rule now has semantics instead of matching a name at any depth.
# * the secret globs are the INDEXER's equivalent of ``DEFAULT_PROTECTED_GLOBS`` in
#   ``ide/packages/codna-engine-client/src/workspaceSnapshot.ts``. They are here so that a
#   ``!.env`` in a user file cannot re-expose a credential to the embedder.
#
# Hidden DIRECTORIES (any segment starting with ``.``) are floored separately, in
# ``_hidden_dir_excluded``: the old walker pruned directories with ``not d.startswith(".")`` but
# never filtered FILES by that prefix, so making it a glob rule would newly drop ``.eslintrc.js``
# and change behaviour for a repo with no ignore files at all. The ancestor-only check keeps the
# old semantics exactly.
FLOOR_PATTERNS: tuple[str, ...] = (
    # VCS + vendored dependency trees + tool caches
    "**/.git/",
    "**/.hg/",
    "**/.svn/",
    "**/.venv/",
    "**/venv/",
    "**/env/",
    "**/node_modules/",
    "**/__pycache__/",
    "**/.mypy_cache/",
    "**/.pytest_cache/",
    "**/.ruff_cache/",
    "**/dist/",
    "**/build/",
    "**/.codna-memory/",
    # Secrets: the index must never embed a credential, whatever a user file says
    "**/.env",
    "**/.env.*",
    "**/*.env",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/*.keystore",
    "**/id_rsa",
    "**/id_dsa",
    "**/id_ecdsa",
    "**/id_ed25519",
    "**/.npmrc",
    "**/.netrc",
    "**/.pgpass",
    "**/.git-credentials",
    "**/credentials.json",
    "**/.ssh/",
    "**/.aws/",
    "**/.gnupg/",
    "**/.docker/config.json",
    "**/secrets/",
)

# Directories never worth descending into while DISCOVERING nested ignore files. Discovery
# must not pay for the trees the floor is about to exclude.
_DISCOVERY_PRUNE = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".codna-memory",
    ".codna",
})


class IgnorePatternError(ValueError):
    """A pattern could not be compiled. Raised instead of silently dropping the rule."""


@dataclass(frozen=True)
class Rule:
    """One compiled ignore pattern, scoped to ``base`` ("" for the repository root)."""

    pattern: str
    negated: bool
    anchored: bool
    dir_only: bool
    base: str
    source: str
    line: int
    regex: re.Pattern[str]

    def matches(self, relpath: str) -> bool:
        return self.regex.match(relpath) is not None


def _translate(pattern: str) -> str:
    """Translate a gitignore glob body into a regex body (no anchors, no base prefix).

    ``*`` never crosses ``/``; ``**`` does; ``?`` is one non-``/`` character; ``[...]`` is a
    character class (``[!...]`` negates, as git documents).
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                after = i + 2
                if after < n and pattern[after] == "/":
                    # "**/" — zero or more leading directories.
                    out.append("(?:.*/)?")
                    i = after + 1
                else:
                    out.append(".*")
                    i = after
                continue
            out.append("[^/]*")
            i += 1
            continue
        if char == "?":
            out.append("[^/]")
            i += 1
            continue
        if char == "[":
            end = i + 1
            if end < n and pattern[end] in "!^":
                end += 1
            if end < n and pattern[end] == "]":
                end += 1
            while end < n and pattern[end] != "]":
                end += 1
            if end >= n:
                # Unterminated class: git treats the "[" literally.
                out.append(r"\[")
                i += 1
                continue
            body = pattern[i + 1:end]
            if body.startswith("!"):
                body = "^" + body[1:]
            elif body.startswith("^"):
                body = "\\" + body
            out.append("[" + body + "]")
            i = end + 1
            continue
        if char == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        out.append(re.escape(char))
        i += 1
    return "".join(out)


def _strip_unescaped_trailing_spaces(text: str) -> str:
    """gitignore ignores trailing spaces unless they are backslash-escaped."""
    end = len(text)
    while end > 0 and text[end - 1] == " ":
        # Count the backslashes immediately before this space; an odd count escapes it.
        backslashes = 0
        k = end - 2
        while k >= 0 and text[k] == "\\":
            backslashes += 1
            k -= 1
        if backslashes % 2 == 1:
            break
        end -= 1
    return text[:end]


def compile_rule(raw: str, *, base: str, source: str, line: int) -> Rule | None:
    """Compile one gitignore line. Returns ``None`` for a comment or a blank line."""
    text = _strip_unescaped_trailing_spaces(raw.rstrip("\n").rstrip("\r"))
    if not text.strip():
        return None
    if text.startswith("#"):
        return None

    negated = False
    if text.startswith("!"):
        negated = True
        text = text[1:]
    elif text.startswith("\\!") or text.startswith("\\#"):
        text = text[1:]
    if not text:
        return None

    dir_only = text.endswith("/")
    if dir_only:
        text = text[:-1]
    if not text:
        return None

    anchored = text.startswith("/")
    if anchored:
        text = text[1:]
    # A separator anywhere in the middle anchors the pattern to this file's directory.
    if "/" in text:
        anchored = True
    if not text:
        return None

    body = _translate(text)
    prefix = re.escape(base) + "/" if base else ""
    if anchored:
        head = "^" + prefix
    else:
        # No slash: git applies the pattern to every path COMPONENT, so an arbitrary directory
        # prefix is allowed (`data` excludes `a/data/x`).
        head = "^" + prefix + "(?:.*/)?"
    # A match on a directory covers everything beneath it (git: excluding a directory excludes its
    # contents). That is why the tail is always present, and why `data/` + `!data/keep.py` cannot
    # re-include — the parent rule is enforced in `IgnoreStack._decide`, via the ancestor check.
    tail = "(?:/.*)?$"
    try:
        regex = re.compile(head + body + tail)
    except re.error as exc:  # pragma: no cover - _translate escapes literals
        raise IgnorePatternError(f"{source}:{line}: bad pattern {raw!r}: {exc}") from exc
    return Rule(
        pattern=raw.rstrip("\n").rstrip("\r"),
        negated=negated,
        anchored=anchored,
        dir_only=dir_only,
        base=base,
        source=source,
        line=line,
        regex=regex,
    )


def parse_patterns(lines: Iterable[str], *, base: str, source: str) -> list[Rule]:
    """Compile a whole file's worth of patterns, preserving their order (last match wins)."""
    rules: list[Rule] = []
    for index, raw in enumerate(lines, start=1):
        rule = compile_rule(raw, base=base, source=source, line=index)
        if rule is not None:
            rules.append(rule)
    return rules


@dataclass(frozen=True)
class IgnoreLayer:
    """The rules from one ignore file, scoped to one directory (``base``, "" for the root)."""

    base: str
    path: str
    rules: tuple[Rule, ...]

    @property
    def depth(self) -> int:
        return 0 if not self.base else self.base.count("/") + 1


@dataclass
class IgnoreStack:
    """The resolved ignore decision for one repository root.

    Evaluation order is (1) the floor, which is absolute; then (2) every layer, shallow first,
    and within a layer line order. The last matching layer/line decides, exactly as git
    documents for a single file — and, across files, a deeper file overrides a shallower one.
    """

    root: str
    layers: tuple[IgnoreLayer, ...] = ()
    floor: tuple[Rule, ...] = ()

    # ── construction ────────────────────────────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        root: str | os.PathLike[str],
        *,
        floor: Sequence[str] = FLOOR_PATTERNS,
        include_gitignore: bool = True,
        filenames: Sequence[str] = IGNORE_FILENAMES,
        extras: Sequence[str] = (),
    ) -> "IgnoreStack":
        """Discover every ignore file under ``root`` and compile the stack.

        ``include_gitignore=False`` drops the implicit ``.gitignore`` layer (a caller that needs
        only the explicit user opt-outs), never the floor. ``extras`` is an ADDITIONAL layer applied
        last — a caller's own extra patterns stay subject to the floor (which is evaluated before
        every layer), so an extra cannot be used to widen the surface either.
        """
        root_str = os.fspath(root)
        names = tuple(filenames) if include_gitignore else tuple(
            n for n in filenames if n != GITIGNORE
        )
        layers = _discover_layers(root_str, names)
        if extras:
            layers = layers + (
                IgnoreLayer(
                    base="",
                    path="<caller>",
                    rules=tuple(parse_patterns(extras, base="", source="<caller>")),
                ),
            )
        floor_rules = tuple(
            rule
            for index, pattern in enumerate(floor)
            for rule in [compile_rule(pattern, base="", source="<floor>", line=index + 1)]
            if rule is not None
        )
        return cls(
            root=root_str,
            layers=layers,
            floor=floor_rules,
        )

    # ── queries ────────────────────────────────────────────────────────────────────────────
    def is_ignored(self, relpath: str) -> bool:
        """Whether a repo-relative path is excluded. Directories pass a trailing-slash-free path."""
        rel = _normalize(relpath)
        if not rel:
            return False
        # An excluded ANCESTOR excludes everything beneath it, and commit history cannot be
        # re-included from below (git: "It is not possible to re-include a file if a parent
        # directory of that file is excluded"). This is a traversal rule, not a match rule, and
        # it is why `data/` + `!data/keep.py` still excludes keep.py — see test_ignore.py.
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            if self._decide("/".join(parts[:depth]), is_dir=True):
                return True
        return self._decide(rel, is_dir=False)

    def is_ignored_dir(self, reldir: str) -> bool:
        """Whether a directory is excluded (used to prune a walk)."""
        rel = _normalize(reldir)
        if not rel:
            return False
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            if self._decide("/".join(parts[:depth]), is_dir=True):
                return True
        return self._decide(rel, is_dir=True)

    def _decide(self, rel: str, *, is_dir: bool) -> bool:
        # (0) hidden directories are excluded as a floor, reproducing the old
        # `not name.startswith(".")` pruning. Ancestor-only: a dot-FILE is still indexable.
        if is_dir and rel.rsplit("/", 1)[-1].startswith("."):
            return True
        # (1) the floor: absolute, cannot be negated by any layer.
        for rule in self.floor:
            # A trailing-slash pattern is DIRECTORY-ONLY. git: `data/` does not exclude a FILE
            # named `data`. Without this guard every `**/dist/` floor rule would also swallow a
            # file called `dist`, which no user asked for.
            if rule.dir_only and not is_dir:
                continue
            if rule.matches(rel):
                return True
        # (2) layers, shallow first, line order within each; the LAST match decides.
        decision: bool | None = None
        for layer in self.layers:
            for rule in layer.rules:
                if rule.dir_only and not is_dir:
                    continue
                if rule.matches(rel):
                    decision = not rule.negated
        return bool(decision)

    def is_floor_ignored_dir(self, reldir: str) -> bool:
        """Whether a directory is excluded by a rule NO user file can negate.

        A walker may prune such a directory UNCONDITIONALLY: the floor is evaluated before every
        layer and lives outside the negatable stack, so no re-include can resurrect a file beneath
        it. This is what keeps ``node_modules/`` pruned in a repo that uses `!` negation — without
        it a single `!keep.log` would make the walk descend every dependency tree, file by file.

        DOT-DIRECTORIES ARE INCLUDED ON PURPOSE. Codna has always pruned every dot-directory: the
        pre-ignore walker did ``dirs[:] = [d for d in dirs if ... and not d.startswith(".")]``.
        This is therefore NOT a new exclusion and not something the floor invented — it preserves
        the existing index surface, and a repo's ``.github/workflows`` remains unindexed exactly as
        it was before this change. Removing it here would silently START indexing dot-trees, which
        is the opposite of an ignore feature. Only directories are affected: a dot-FILE such as
        ``.eslintrc.js`` is still indexed (see ``test_hidden_directories_are_floored_but_dot_files_are_not``).
        """
        rel = _normalize(reldir)
        if not rel:
            return False
        parts = rel.split("/")
        for depth in range(1, len(parts) + 1):
            candidate = "/".join(parts[:depth])
            if candidate.rsplit("/", 1)[-1].startswith("."):
                return True
            for rule in self.floor:
                # No `dir_only` guard is needed: every `candidate` here is a directory by
                # construction (this method is only ever asked about directories), so a dir_only
                # rule and a file-or-dir rule are evaluated identically. Keeping a guard would be
                # a tautology; `_decide` owns the file/directory distinction instead.
                if rule.matches(candidate):
                    return True
        return False

    def has_negation(self) -> bool:
        """True when any user layer negates, so a pruned directory could still re-include a file.

        A walker uses this to decide whether pruning an ignored directory is safe: with no
        negation anywhere, an ignored directory can never contain an included file.
        """
        return any(rule.negated for layer in self.layers for rule in layer.rules)

    def sources(self) -> tuple[str, ...]:
        """The ignore files that contributed rules (repo-relative or absolute), for diagnostics."""
        return tuple(layer.path for layer in self.layers)


def _normalize(relpath: str) -> str:
    rel = relpath.replace(os.sep, "/")
    # Strip leading "./" segments BEFORE normpath. `normpath` collapses most of them, but the
    # repeated form `././x` would otherwise survive as a `.` segment that then looks like a hidden
    # directory. Linear in the prefix: each step consumes two characters, no regex backtracking.
    while rel.startswith("./"):
        rel = rel[2:]
    rel = posixpath.normpath(rel)
    if rel in (".", ""):
        return ""
    return rel.lstrip("/")


def _discover_layers(root: str, filenames: Sequence[str]) -> tuple[IgnoreLayer, ...]:
    """Walk ``root`` once, collecting ignore files. Sorted shallow-first, then by file KIND.

    The sort key MUST NOT be the path: ``".codnaignore" < ".cursorignore" < ".gitignore"``
    alphabetically, which would silently invert the documented precedence and let the implicit
    ``.gitignore`` win over the Codna-native file. Depth first, then the caller's filename order.
    """
    filename_rank = {name: index for index, name in enumerate(filenames)}
    found: list[tuple[int, str, int, IgnoreLayer]] = []
    if not os.path.isdir(root):
        return ()
    for current, dirnames, files in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _DISCOVERY_PRUNE)
        rel_dir = _normalize(os.path.relpath(current, root))
        present = [name for name in filenames if name in files]
        if not present:
            continue
        for name in present:
            path = os.path.join(current, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    lines = handle.read().splitlines()
            except OSError:
                # An unreadable ignore file must not silently widen the indexed surface:
                # failing open here would embed exactly what the user tried to exclude.
                raise
            layer = IgnoreLayer(
                base=rel_dir,
                path=os.path.relpath(path, root).replace(os.sep, "/"),
                rules=tuple(parse_patterns(lines, base=rel_dir, source=name)),
            )
            found.append((layer.depth, layer.base, filename_rank[name], layer))
    # Shallow first: a deeper file is applied later and therefore wins for its subtree.
    found.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return tuple(entry[3] for entry in found)


def build_stack(root: str | os.PathLike[str], **kwargs) -> IgnoreStack:
    """Shorthand for :meth:`IgnoreStack.build`."""
    return IgnoreStack.build(root, **kwargs)
