"""Filesystem primitives shared by the rule reader and the skill reader.

**Why this module exists.** Rules and skills answer the same question — "which of this repository's
hand-written files govern this turn?" — and they walk the same tree to answer it. Two readers that
each carried their own depth bound, their own skip list and their own containment check would be two
places to fix a traversal bug, and the containment check is security-relevant: a symlinked
``SKILL.md`` pointing at ``/etc`` must not become part of a prompt. So the bounds and the checks live
here once, and both readers call them.

Nothing here understands rules or skills. It opens markdown, refuses to leave the tree it was given,
and bounds the walk; the semantics belong to :mod:`codna.rules.discovery` and
:mod:`codna.rules.skills`.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Iterator

__all__ = [
    "MAX_DEPTH",
    "MAX_FILE_BYTES",
    "SKIP_DIRS",
    "contained",
    "iter_dirs_containing",
    "iter_named_dirs",
    "label",
    "posix",
    "read_text",
    "size_of",
]

# Bound every walk: a rules/skills directory deep inside a monorepo must not turn a prompt build into
# a filesystem crawl. 12 levels is deeper than any real layout and still cheap.
MAX_DEPTH = 12

# One markdown file larger than this is not a rule or a skill; it is an artifact. Reported, never read.
MAX_FILE_BYTES = 256 * 1024

# Directories that cannot contain a repository's rules or skills, and are expensive or unsafe to enter.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages", ".turbo", ".next",
    "dist", "build", ".cache", ".gradle", ".idea", ".vscode-test",
})


def posix(path: str) -> str:
    """The machine-independent form every label in this package uses."""
    return str(path).replace(os.sep, "/")


def size_of(path: str) -> int | None:
    """The file's size in bytes, or ``None`` when it cannot be stat'ed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def read_text(path: str) -> str | None:
    """Read one candidate file, or ``None`` when it is unreadable or over the size bound."""
    size = size_of(path)
    if size is None or size > MAX_FILE_BYTES:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def contained(real_root: str, path: str) -> bool:
    """Whether ``path`` (following symlinks) stays inside ``real_root``."""
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    if real == real_root:
        return True
    return real.startswith(real_root + os.sep)


def label(rel_dir: str, filename: str) -> str:
    """``rel_dir/filename``, without a leading slash when the directory is the repository root."""
    return f"{rel_dir}/{filename}" if rel_dir else filename


def _descend(current: str, dirnames: list[str]) -> None:
    """Prune one level of ``os.walk``: sorted, no build/VCS trees, and never through a symlink."""
    dirnames[:] = sorted(
        name for name in dirnames
        if name not in SKIP_DIRS and not os.path.islink(os.path.join(current, name))
    )


def iter_named_dirs(
    repo_dir: str,
    rel_name: str,
    *,
    max_depth: int = MAX_DEPTH,
    on_error: Callable[[OSError], None] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(absolute_dir, rel_dir)`` for every ``rel_name`` directory at any depth.

    ``rel_name`` is a relative path with a fixed tail (``.codna/rules``, ``.cursor/skills``), so the
    same monorepo package can define its own instance of it. A directory reached through a symlink is
    never yielded: the caller cannot then be talked into reading a tree it did not name.

    ``on_error`` receives every directory ``os.walk`` could not list. Without it the walk is SILENT
    about them — ``os.walk`` swallows the ``OSError`` by default — and a caller cannot tell "this
    directory holds nothing" from "this directory could not be read". That distinction is the whole
    difference between an empty catalogue and a catalogue missing a layer the user can see on disk.
    """
    for current, dirnames, _files in os.walk(repo_dir, followlinks=False, onerror=on_error):
        rel_current = posix(os.path.relpath(current, repo_dir))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        _descend(current, dirnames)
        parent = "" if rel_current == "." else rel_current
        candidate = os.path.join(current, rel_name)
        if os.path.isdir(candidate) and not os.path.islink(candidate):
            yield candidate, posix(os.path.join(parent, rel_name)) if parent else rel_name


def iter_dirs_containing(
    start_dir: str,
    filename: str,
    *,
    max_depth: int = MAX_DEPTH,
    on_error: Callable[[OSError], None] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(absolute_dir, rel_dir)`` for every directory under ``start_dir`` holding ``filename``.

    This is what makes "the identity of a skill is the folder containing ``SKILL.md``" true for nested
    layouts: ``.cursor/skills/testing/api/SKILL.md`` is the skill ``api`` inside the category
    ``testing``, and the category itself is not a skill because it holds no ``SKILL.md``.

    ``start_dir`` itself is not yielded — the caller has already decided that it is the container,
    not a member.

    ``on_error`` receives every directory ``os.walk`` could not list, for the reason given on
    :func:`iter_named_dirs`: an unreadable directory must be reportable, not indistinguishable from
    an empty one.
    """
    root = os.path.abspath(start_dir)
    for current, dirnames, filenames in os.walk(root, followlinks=False, onerror=on_error):
        rel_current = posix(os.path.relpath(current, root))
        depth = 0 if rel_current == "." else rel_current.count("/") + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        _descend(current, dirnames)
        if current == root or filename not in filenames:
            continue
        yield current, rel_current


def iter_bundled_files(skill_dir: str, directories: Iterable[str], *, limit: int) -> tuple[list[str], bool]:
    """The files a skill ships beside its ``SKILL.md`` — ``scripts/``, ``references/``, ``assets/``.

    Returned as POSIX paths RELATIVE TO THE SKILL FOLDER, because that is how a loaded skill has to
    name them (the prompt must not depend on where the repository is checked out). ``limit`` bounds
    the listing so a pathological skill cannot dominate a prompt; the second element says whether the
    bound was hit, so the caller can report it instead of quietly listing a truncated set.
    """
    found: list[str] = []
    for name in directories:
        directory = os.path.join(skill_dir, name)
        if not os.path.isdir(directory) or os.path.islink(directory):
            continue
        for current, dirnames, filenames in os.walk(directory, followlinks=False):
            dirnames[:] = sorted(dirnames)
            for filename in sorted(filenames):
                found.append(posix(os.path.relpath(os.path.join(current, filename), skill_dir)))
                if len(found) >= limit:
                    return found, True
    return found, False
