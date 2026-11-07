"""Discovery: find every rule a repository has, and say which layer each one belongs to.

This is the ONE reader. It replaces both predecessors that disagreed — the Python reader's eight
fixed root filenames concatenated in place, and the vendored SDK's "root AGENTS.md plus every ``.md``
beside it" — by walking the layout once and returning normalised :class:`~codna.rules.model.Rule`
objects. Consumers (``project_rules``, ``durable.rules``, the packaged runner, the MCP pull tool) all
go through here, so they cannot drift apart again.

**What is discovered, and at what scope**

  * ``.codna/rules/**/*.mdc``, ``*.md``                     → TEAM, subtree-scoped by directory
  * ``AGENTS.md`` at any depth                              → PROJECT, legacy always-on, subtree-scoped
  * ``.cursor/rules/**/*.mdc``, ``*.md``                    → PROJECT, subtree-scoped by directory
  * ``<user>/rules/**`` (default ``~/.codna/rules``)        → USER, subtree-scoped by directory
  * the legacy root files                                   → see ``_LEGACY_ROOT_FILES``

**Two deliberate divergences from Cursor, both stated rather than hidden**

1. ``.md`` inside ``.cursor/rules`` is READ. Cursor silently ignores it; a file a user wrote and
   named "rules" must not be a no-op.
2. A file in a rules directory with NO frontmatter is applied always, **with a structured warning**.
   Cursor would treat it as manual, which is the "I added a rule and nothing happened" footgun. Codna
   keeps the file effective and tells the user how to scope it instead of guessing silently.

**Legacy files keep legacy behaviour.** ``AGENTS.md``, ``.codna/rules``, ``.cursorrules`` and
``BUGBOT.md`` with no frontmatter are always-on and are NOT subtree-scoped at the repository root, so
`test_legacy_file_remains_always_on` is a property of this module, not of a caller's care. A legacy
file that DOES carry frontmatter opts in to the new scoping — the frontmatter is the opt-in.

**Safety.** Traversal is bounded (depth, file count), skips build/VCS/vendor trees, and refuses to
read a rule file whose real path escapes the repository — a symlinked ``AGENTS.md`` pointing at
``/etc`` must not become this turn's system prompt. Nothing here changes the secret/ignore posture;
this module only ever opens markdown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from . import frontmatter as _frontmatter
from . import globs as _globs
from .model import Rule, RuleMode, RuleWarning, Scope

__all__ = ["Discovery", "MAX_DEPTH", "MAX_RULES", "discover", "discover_user_rules", "legacy_rule"]

# Bound the walk: a rules directory deep inside a monorepo must not turn every turn's prompt build
# into a filesystem crawl. 12 levels is deeper than any real rules layout and still cheap.
MAX_DEPTH = 12
MAX_RULES = 2_000

# Directories that cannot contain this repository's rules, and are expensive or unsafe to enter.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages", ".turbo", ".next",
    "dist", "build", ".cache", ".gradle", ".idea", ".vscode-test",
})

# Rule-bearing directory names, relative to any directory in the tree.
_CODNA_RULES_DIR = ".codna/rules"
_CURSOR_RULES_DIR = ".cursor/rules"

# The LEGACY root files, in the historical precedence order. Each is read as-is (concatenation
# semantics preserved) unless it carries frontmatter, which opts it into scoping. `AGENTS.md` is in
# this set because it predates the engine and must keep applying everywhere.
_LEGACY_ROOT_FILES: tuple[str, ...] = (
    "AGENTS.md",
    ".codna/rules.md",
    ".codna/rules",          # only when it is a FILE; the directory form is discovered as a package
    ".codna/review.md",
    ".cursorrules",
    ".cursor/rules.md",
    ".cursor/BUGBOT.md",
    "BUGBOT.md",
)

# Frontmatter keys parsed into first-class fields; everything else is preserved in `Rule.extra`.
_KNOWN_KEYS = frozenset({"description", "globs", "alwaysapply", "always_apply", "always"})

_MAX_RULE_BYTES = 256 * 1024


@dataclass(frozen=True)
class Discovery:
    """Everything found, plus every problem found while finding it. Never a silent partial read."""

    rules: tuple[Rule, ...] = ()
    warnings: tuple[RuleWarning, ...] = ()

    def for_scope(self, *scopes: Scope) -> tuple[Rule, ...]:
        wanted = set(scopes)
        return tuple(rule for rule in self.rules if rule.scope in wanted)


def _posix(path: str) -> str:
    return str(path).replace(os.sep, "/")


def _read_text(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size > _MAX_RULE_BYTES:
        return None  # reported by the caller, which knows the label
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _contained(real_root: str, path: str) -> bool:
    """Whether ``path`` (following symlinks) stays inside the repository."""
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    if real == real_root:
        return True
    return real.startswith(real_root + os.sep)


def _label(rel_dir: str, filename: str) -> str:
    return f"{rel_dir}/{filename}" if rel_dir else filename


def _apply_frontmatter(
    rule: Rule, metadata: dict[str, object], *, in_rules_dir: bool,
) -> tuple[Rule, list[RuleWarning]]:
    """Fold parsed frontmatter onto a rule and report the shapes that are legal but surprising."""
    warnings: list[RuleWarning] = []
    values = dict(metadata)

    name = values.pop("name", None)
    description = values.pop("description", None)
    globs_raw = values.pop("globs", None)
    # Cursor writes `alwaysApply`; Codna's own YAML prefers snake_case. Accept both, and refuse to
    # guess when they disagree — two spellings of one setting in conflict is an error, not a coin flip.
    always_camel = values.pop("alwaysApply", None)
    always_snake = values.pop("always_apply", None)
    always_bare = values.pop("always", None)
    always_values = [value for value in (always_camel, always_snake, always_bare) if value is not None]

    if len(always_values) > 1 and len({bool(value) for value in always_values}) > 1:
        warnings.append(RuleWarning(
            code="rule_frontmatter_conflict",
            message="alwaysApply / always_apply disagree; keeping alwaysApply",
            source_path=rule.source_path,
        ))
    always_raw = always_camel if always_camel is not None else (
        always_snake if always_snake is not None else always_bare
    )
    if always_raw is not None and not isinstance(always_raw, bool):
        raise ValueError("alwaysApply must be true or false")

    try:
        globs = _globs.parse_globs(globs_raw)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    final_name = str(name).strip() if isinstance(name, str) and name.strip() else rule.name
    final_description = str(description).strip() if isinstance(description, str) else ""

    # A LEGACY file (AGENTS.md / .codna/rules / .cursorrules / BUGBOT.md) stays always-on unless the
    # author explicitly scoped it. Frontmatter that merely documents the rule (`description:`) must not
    # silently turn an always-on file into a catalog line — that would be exactly the regression the
    # migration is supposed to be impossible to cause. `alwaysApply: false` or any `globs:` is the
    # explicit opt-in to scoping; the frontmatter IS the opt-in, but only when it says so.
    if rule.legacy_default_always:
        if always_raw is False or (always_raw is None and globs):
            always_apply = False
        else:
            always_apply = True
    else:
        always_apply = bool(always_raw) if always_raw is not None else False

    updated = Rule(
        name=final_name,
        body=rule.body,
        source_path=rule.source_path,
        scope=rule.scope,
        base_dir=rule.base_dir,
        description=final_description,
        globs=globs,
        always_apply=always_apply,
        legacy_default_always=rule.legacy_default_always,
        extra={str(k): v for k, v in values.items() if str(k).casefold() not in _KNOWN_KEYS},
    )

    if in_rules_dir and updated.mode is RuleMode.MANUAL:
        warnings.append(RuleWarning(
            code="rule_manual_only",
            message=(
                "no alwaysApply/globs/description, so this rule is manual-only and will not apply "
                "on its own; add `alwaysApply: true`, `globs: …`, or `description: …`"
            ),
            source_path=updated.source_path,
        ))
    return updated, warnings


def _rule_from_file(
    path: str,
    *,
    rel_dir: str,
    scope: Scope,
    in_rules_dir: bool,
    legacy_default_always: bool,
    display: str,
) -> tuple[Rule | None, list[RuleWarning]]:
    """Read one candidate file into a Rule (or report why it is not one)."""
    warnings: list[RuleWarning] = []
    text = _read_text(path)
    if text is None:
        try:
            too_big = os.path.getsize(path) > _MAX_RULE_BYTES
        except OSError:
            too_big = False
        if too_big:
            warnings.append(RuleWarning(
                code="rule_file_too_large",
                message=f"larger than {_MAX_RULE_BYTES} bytes; skipped",
                source_path=display,
            ))
        return None, warnings

    filename = os.path.basename(path)
    # A dotfile like `.cursorrules` has no name before its dot; the filename IS the rule name.
    stem = filename.rsplit(".", 1)[0] or filename
    placeholder = Rule(
        name=stem,
        body=text,
        source_path=display,
        scope=scope,
        base_dir=_posix(rel_dir),
        legacy_default_always=legacy_default_always,
    )
    try:
        metadata, body, had_frontmatter = _frontmatter.parse_markdown(text, source_path=display)
    except _frontmatter.FrontmatterError as exc:
        warnings.append(RuleWarning(
            code="rule_frontmatter_invalid",
            message=str(exc),
            source_path=display,
        ))
        return None, warnings

    rule = Rule(
        name=placeholder.name,
        body=body.strip(),
        source_path=display,
        scope=scope,
        base_dir=placeholder.base_dir,
        legacy_default_always=legacy_default_always,
    )
    if not rule.body:
        warnings.append(RuleWarning(
            code="rule_body_empty", message="no rule text after the frontmatter; skipped",
            source_path=display,
        ))
        return None, warnings

    if had_frontmatter:
        # A declared frontmatter block is authoritative, even when it is empty: writing the fences is
        # the author saying "this file has a mode", and an empty one means manual. Treating an empty
        # fence as "no frontmatter" would apply a deliberately-manual rule always — the inverse of the
        # footgun this engine removes, but just as wrong.
        try:
            rule, extra_warnings = _apply_frontmatter(rule, metadata, in_rules_dir=in_rules_dir)
        except ValueError as exc:
            warnings.append(RuleWarning(
                code="rule_frontmatter_invalid", message=str(exc), source_path=display,
            ))
            return None, warnings
        warnings.extend(extra_warnings)
    elif in_rules_dir:
        # No frontmatter at all in a rules directory: keep the file effective, and say so. Silently
        # treating it as manual is the footgun this engine exists to remove.
        rule = Rule(
            name=rule.name,
            body=rule.body,
            source_path=rule.source_path,
            scope=rule.scope,
            base_dir=rule.base_dir,
            always_apply=True,
            legacy_default_always=rule.legacy_default_always,
        )
        warnings.append(RuleWarning(
            code="rule_without_frontmatter",
            message=(
                "no frontmatter; applied always for this subtree. Add `alwaysApply: false` with "
                "`globs:` or `description:` to scope it"
            ),
            source_path=display,
        ))
    elif legacy_default_always:
        rule = Rule(
            name=rule.name,
            body=rule.body,
            source_path=rule.source_path,
            scope=rule.scope,
            base_dir=rule.base_dir,
            always_apply=True,
            legacy_default_always=True,
        )
    return rule, warnings


def legacy_rule(repo_dir: str, rel: str) -> tuple[Rule | None, list[RuleWarning]]:
    """Read one of the historical root files with its historical (always-on) default."""
    path = os.path.join(repo_dir, rel)
    if not os.path.isfile(path):
        return None, []
    return _rule_from_file(
        path, rel_dir="", scope=_legacy_scope(rel), in_rules_dir=False,
        legacy_default_always=True, display=_posix(rel),
    )


def _legacy_scope(rel: str) -> Scope:
    """Which layer a legacy file belongs to. ``.codna/*`` is Codna's own (TEAM)."""
    return Scope.TEAM if rel.startswith(".codna/") else Scope.PROJECT


def _walk_rule_dirs(repo_dir: str, rel_dir_name: str) -> Iterable[tuple[str, str]]:
    """Yield ``(absolute_dir, rel_dir)`` for every rule directory of one flavour, any depth."""
    for current, dirnames, _files in os.walk(repo_dir, followlinks=False):
        rel_current = _posix(os.path.relpath(current, repo_dir))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _SKIP_DIRS and not os.path.islink(os.path.join(current, name))
        )
        parent = "" if rel_current == "." else rel_current
        candidate = os.path.join(current, rel_dir_name)
        if os.path.isdir(candidate) and not os.path.islink(candidate):
            yield candidate, _posix(os.path.join(parent, rel_dir_name)) if parent else rel_dir_name


def _rule_files(directory: str) -> list[str]:
    """``*.mdc`` and ``*.md`` directly inside a rules directory (not recursive; nesting is per-dir)."""
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    return [
        name for name in entries
        if name.lower().endswith((".mdc", ".md"))
        and os.path.isfile(os.path.join(directory, name))
    ]


def _discover_agents_md(repo_dir: str) -> Iterable[tuple[str, str, str]]:
    """Yield ``(absolute, rel_dir, display)`` for every AGENTS.md at any depth."""
    for current, dirnames, filenames in os.walk(repo_dir, followlinks=False):
        rel_current = _posix(os.path.relpath(current, repo_dir))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _SKIP_DIRS and not os.path.islink(os.path.join(current, name))
        )
        parent = "" if rel_current == "." else rel_current
        for name in sorted(filenames):
            if name.casefold() != "agents.md":
                continue
            yield os.path.join(current, name), parent, _label(parent, name)


def discover(
    repo_dir: str | None,
    *,
    user_rules_dir: str | None = None,
    include_user: bool = True,
) -> Discovery:
    """Every rule visible to a turn in ``repo_dir``. Deterministic, bounded, and non-silent."""
    warnings: list[RuleWarning] = []
    rules: list[Rule] = []

    root = os.path.abspath(os.path.expanduser(repo_dir or "."))
    if not os.path.isdir(root):
        return Discovery(warnings=(RuleWarning(
            code="rules_root_missing",
            message="not a directory; no repository rules were read",
            source_path=_posix(root),
        ),))
    real_root = os.path.realpath(root)

    def add(rule: Rule | None, extra: list[RuleWarning]) -> None:
        warnings.extend(extra)
        if rule is not None:
            rules.append(rule)

    # 1. Rules PACKAGES: `.codna/rules/**` (TEAM) and `.cursor/rules/**` (PROJECT).
    for rel_dir_name, scope in ((_CODNA_RULES_DIR, Scope.TEAM), (_CURSOR_RULES_DIR, Scope.PROJECT)):
        for directory, rel_dir in _walk_rule_dirs(root, rel_dir_name):
            if not _contained(real_root, directory):
                warnings.append(RuleWarning(
                    code="rules_dir_outside_repository",
                    message="symlinked outside the repository; skipped",
                    source_path=_posix(rel_dir),
                ))
                continue
            # The SUBTREE a rules directory governs is its parent (e.g. `pkg/api/.codna/rules` scopes
            # to `pkg/api`), not the `.codna/rules` directory itself.
            base_dir = rel_dir[: -len("/" + rel_dir_name)] if "/" in rel_dir else ""
            for filename in _rule_files(directory):
                path = os.path.join(directory, filename)
                if not _contained(real_root, path):
                    warnings.append(RuleWarning(
                        code="rule_file_outside_repository",
                        message="resolves outside the repository; skipped",
                        source_path=_posix(_label(rel_dir, filename)),
                    ))
                    continue
                add(*_rule_from_file(
                    path, rel_dir=base_dir, scope=scope, in_rules_dir=True,
                    legacy_default_always=False, display=_posix(_label(rel_dir, filename)),
                ))

    # 2. AGENTS.md at any depth (PROJECT, legacy always-on, subtree-scoped by its directory).
    seen_agents: set[str] = set()
    for path, rel_dir, display in _discover_agents_md(root):
        if not _contained(real_root, path):
            warnings.append(RuleWarning(
                code="rule_file_outside_repository",
                message="resolves outside the repository; skipped",
                source_path=display,
            ))
            continue
        seen_agents.add(display)
        add(*_rule_from_file(
            path, rel_dir=rel_dir, scope=Scope.PROJECT, in_rules_dir=False,
            legacy_default_always=True, display=display,
        ))

    # 3. The legacy root files. `AGENTS.md` was already read above by the depth walk.
    for rel in _LEGACY_ROOT_FILES:
        if rel == "AGENTS.md":
            continue
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        if not _contained(real_root, path):
            warnings.append(RuleWarning(
                code="rule_file_outside_repository",
                message="resolves outside the repository; skipped",
                source_path=_posix(rel),
            ))
            continue
        add(*_rule_from_file(
            path, rel_dir="", scope=_legacy_scope(rel), in_rules_dir=False,
            legacy_default_always=True, display=_posix(rel),
        ))

    # 4. USER layer: machine-local rules. Read from an explicit directory when given (tests, or a
    #    host with a non-standard home) and never from the repository.
    if include_user:
        user_dir = user_rules_dir if user_rules_dir is not None else _default_user_rules_dir()
        if user_dir:
            user_discovery = discover_user_rules(user_dir)
            rules.extend(user_discovery.rules)
            warnings.extend(user_discovery.warnings)

    ordered = _dedupe(rules)
    if len(ordered) > MAX_RULES:
        overflow = ordered[MAX_RULES:]
        ordered = ordered[:MAX_RULES]
        warnings.append(RuleWarning(
            code="rules_truncated_by_count",
            message=f"{len(overflow)} rule(s) beyond the {MAX_RULES} limit were not read",
            source_path=", ".join(rule.source_path for rule in overflow[:10]),
        ))
    return Discovery(rules=tuple(ordered), warnings=tuple(warnings))


def _default_user_rules_dir() -> str | None:
    try:
        home = os.path.expanduser("~")
    except (KeyError, RuntimeError):  # no HOME: no user layer, not an error
        return None
    if not home or home == "~":
        return None
    return os.path.join(home, ".codna", "rules")


def discover_user_rules(user_rules_dir: str) -> Discovery:
    """The USER layer: ``<dir>/**/*.mdc`` and ``*.md``, labelled with ``~`` for readability."""
    directory_root = os.path.abspath(os.path.expanduser(user_rules_dir))
    if not os.path.isdir(directory_root):
        return Discovery()
    real_root = os.path.realpath(directory_root)
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except (KeyError, RuntimeError):
        home = ""
    rules: list[Rule] = []
    warnings: list[RuleWarning] = []
    for current, dirnames, filenames in os.walk(directory_root, followlinks=False):
        rel_current = _posix(os.path.relpath(current, directory_root))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _SKIP_DIRS and not os.path.islink(os.path.join(current, name))
        )
        base_dir = "" if rel_current == "." else rel_current
        for filename in sorted(filenames):
            if not filename.lower().endswith((".mdc", ".md")):
                continue
            path = os.path.join(current, filename)
            if not _contained(real_root, path):
                continue
            display_abs = path if not home else path.replace(home, "~", 1)
            rule, extra = _rule_from_file(
                path, rel_dir=base_dir, scope=Scope.USER, in_rules_dir=True,
                legacy_default_always=False, display=_posix(display_abs),
            )
            warnings.extend(extra)
            if rule is not None:
                rules.append(rule)
    return Discovery(rules=tuple(_dedupe(rules)), warnings=tuple(warnings))


def _dedupe(rules: list[Rule]) -> list[Rule]:
    """One rule per (scope, source_path). A path reached twice must not render twice."""
    seen: set[tuple[Scope, str]] = set()
    kept: list[Rule] = []
    # Sorted first so de-duplication itself is deterministic regardless of walk order.
    for rule in sorted(rules, key=lambda r: r.sort_key()):
        key = (rule.scope, rule.source_path)
        if key in seen:
            continue
        seen.add(key)
        kept.append(rule)
    return kept
