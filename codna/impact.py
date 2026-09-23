"""Repo-aware test-impact analysis — Codna's repo intelligence applied to CI selection.

Given a repository root and a set of changed files (typically a ``git diff`` name-list), compute
which test files are transitively affected so CI runs only those. Deliberate bias: when in doubt,
run MORE tests, never fewer — a false "affected" costs minutes; a false "unaffected" ships a
regression.

How it works
------------
1. Python changes are traced PRECISELY through the import graph (stdlib ``ast``): every file is a
   module key (plus every package-suffix alias it is importable as), edges are imports, and a
   changed module selects the tests in its reverse-transitive importer closure.
2. Changes confined to non-Python languages Codna knows about (``codeunits``'s extractor registry)
   select only that language's own tests — and nothing in a repo whose tests are all in another
   language. This path needs a ``codeunits`` module passed in (the per-file seam from #445);
   without it every non-Python change conservatively selects the full suite.
3. Conservative fallbacks select the FULL suite: broad/config changes, unclassifiable or
   unreadable changes, shared-core changes (imported by a large share of the tests), or anything
   the graph cannot place.

Nothing in here is repo-specific: source roots, package aliases, and test files are derived from
the tree itself.
"""

from __future__ import annotations

import ast
import fnmatch
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Ecosystem-wide conventions (not any one repo's): dependency manifests, test configuration, and
# CI/infra files can affect anything, so changing one selects the full suite. Overridable.
DEFAULT_BROAD_GLOBS = (
    "pyproject.toml", "setup.py", "setup.cfg", "conftest.py", "**/conftest.py",
    ".github/**", "requirements*.txt", "**/requirements*.txt", "pixi.toml", "**/pixi.toml",
    "pytest.ini", "tox.ini", ".env*",
)

# A changed module reached (transitively) by more than this fraction of all tests is shared core:
# "its tests" ARE the suite, so run everything.
SHARED_CORE_FRACTION = 0.34

# Mirrors codeunits._DEFAULT_IGNORE plus common local env/build trees CI checkouts never carry.
DEFAULT_IGNORE = (
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".codna-memory",
    "vendor",
)


@dataclass(frozen=True)
class Impact:
    mode: str            # "subset" | "all"
    tests: list[str]     # repo-relative test paths when mode == "subset"
    reason: str


def _is_test_file(rel: str) -> bool:
    name = os.path.basename(rel)
    return name.startswith("test_") or name.endswith("_test.py")


def _is_broad(rel: str, broad_globs) -> bool:
    return any(fnmatch.fnmatch(rel, g) for g in broad_globs)


def _module_key(rel_path: str) -> str | None:
    if not rel_path.endswith(".py"):
        return None
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _iter_py_files(repo_root: Path, ignore) -> list[Path]:
    ignore_set = set(ignore)
    out: list[Path] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_set and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".py"):
                out.append(Path(root) / fn)
    return out


class _Graph:
    """The Python import graph: module keys (with package-suffix aliases) and reverse edges."""

    def __init__(self, repo_root: Path, py_files: list[Path]):
        self.root = repo_root
        self.primary: dict[str, str] = {}         # rel path -> primary (repo-root) module key
        self.keys_by_file: dict[str, list[str]] = {}
        self.trees: dict[str, ast.AST | None] = {}
        self.importers: dict[str, set[str]] = defaultdict(set)   # primary key -> importing files
        self._packages: set[str] = set()          # primary keys that are real packages (__init__.py)
        self._name_map: dict[str, str] = {}       # ANY importable name -> primary key
        rels = [str(p.relative_to(repo_root)) for p in py_files]
        for rel in rels:                          # pass 1: package census (aliases depend on it,
            if os.path.basename(rel) == "__init__.py":      # so it must complete before pass 2)
                pkg = _module_key(rel)
                if pkg:
                    self._packages.add(pkg)
        for rel in rels:                          # pass 2: names, trees
            keys = self._candidate_keys(rel)
            self.keys_by_file[rel] = keys
            primary = keys[0] if keys else None
            self.primary[rel] = primary
            for key in keys:
                self._name_map.setdefault(key, primary)
            try:
                self.trees[rel] = ast.parse(
                    (self.root / rel).read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                self.trees[rel] = None            # opaque file -> callers fall back conservatively
        for rel, tree in self.trees.items():      # pass 3: edges, targets resolved to primaries
            if tree is None:
                continue
            src = self.primary.get(rel)
            for target in self._import_targets(tree, rel):
                resolved = self._resolve(target)
                if resolved is not None and resolved != src:
                    self.importers[resolved].add(rel)

    def _resolve(self, target: str) -> str | None:
        """Map an import target to the primary key of the module it loads: an exact name first,
        then the longest known prefix (importing ``pkg.mod.attr`` still loads ``pkg.mod``).
        Unresolvable targets (third-party imports) get no edge."""
        if target in self._name_map:
            return self._name_map[target]
        parts = target.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self._name_map:
                return self._name_map[prefix]
        return None

    def _candidate_keys(self, rel: str) -> list[str]:
        """Every dotted name this file may be imported as: the repo-root-relative key, plus a
        package-suffix alias for every ancestor that is a REAL package (``a/b/pkg/mod.py`` under
        a package ``a.b.pkg`` is also importable as ``pkg.mod`` in src/package layouts). Derived
        from the tree, never hardcoded; extra aliases only add edges, which biases toward running
        more — the safe direction."""
        primary = _module_key(rel)
        if not primary:
            return []
        keys = [primary]
        parts = primary.split(".")
        for i in range(2, len(parts) + 1):
            if ".".join(parts[:i]) in self._packages:
                alias = ".".join(parts[i - 1:])
                if alias != primary and alias not in keys:
                    keys.append(alias)
        return keys

    def _import_targets(self, tree: ast.AST, file_rel: str) -> set[str]:
        targets: set[str] = set()
        file_mod = _module_key(file_rel) or ""
        file_pkg_parts = file_mod.split(".")[:-1] if file_mod else []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    targets.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    base_parts = file_pkg_parts[: len(file_pkg_parts) - (node.level - 1)]
                    base = ".".join(base_parts + ([node.module] if node.module else []))
                    if base:
                        targets.add(base)
                elif node.module:
                    targets.add(node.module)
        return targets


def _language_disjoint(changed: list[str], repo_root: Path, codeunits):
    """Codna-backed fast path: every changed file is source in a known NON-Python language and
    none of them is a test in that language -> only same-language tests can be affected (often
    none exist, e.g. a Python-only suite with a TypeScript diff). Returns (tests, reason) or None
    to fall through. Any uncertainty falls through — run more, never fewer."""
    langs = {c: codeunits.language_for(c) for c in changed}
    if not all(langs.values()) or any(lang == "python" for lang in langs.values()):
        return None
    for c in changed:
        try:
            source = (repo_root / c).read_text(encoding="utf-8", errors="replace")
            units = codeunits.extract_file("impact", c, source)
        except (OSError, SyntaxError, ValueError):
            return None
        if any(u.symbol_type == "test" for u in units):
            return None  # a test changed in another language; this engine selects Python tests
    known = sorted(set(langs.values()))
    return ([], f"changes confined to non-Python languages ({', '.join(known)}); "
                "no Python test can be affected by them")


def compute_impact(repo_root, changed: list[str], *, codeunits=None,
                   broad_globs=DEFAULT_BROAD_GLOBS, shared_core_fraction: float = SHARED_CORE_FRACTION,
                   ignore=DEFAULT_IGNORE) -> Impact:
    """Compute the test selection for ``changed`` (repo-relative paths) in ``repo_root``."""
    root = Path(repo_root)
    if not changed:
        return Impact("all", [], "no changed files reported; running full suite to be safe")
    broad = [c for c in changed if _is_broad(c, broad_globs)]
    if broad:
        return Impact("all", [], f"broad/config change touches: {', '.join(sorted(broad)[:5])}")

    if codeunits is not None:
        disjoint = _language_disjoint(changed, root, codeunits)
        if disjoint is not None:
            tests, reason = disjoint
            return Impact("subset", tests, reason)

    graph = _Graph(root, _iter_py_files(root, ignore))
    test_files = sorted(rel for rel in graph.trees if _is_test_file(rel))
    if not test_files:
        return Impact("all", [], "no test files discovered; running full suite")

    for c in changed:
        if not c.endswith(".py"):
            return Impact("all", [], f"non-Python change: {c}")
        if _module_key(c) is None:
            return Impact("all", [], f"unmappable changed file: {c}")
        if c not in graph.trees:
            return Impact("all", [], f"changed file outside indexed sources: {c}")
        if graph.trees[c] is None:
            return Impact("all", [], f"unparseable changed file: {c}")

    # Seed the importer-closure with non-test source modules only. A changed TEST file affects
    # just itself: nothing meaningfully imports tests, and propagating test modules through the
    # importer graph blows up (tests import each other / conftest heavily).
    affected: set[str] = set()
    seeds: set[str] = set()
    for c in changed:
        if _is_test_file(c):
            affected.add(c)
            continue
        primary = graph.primary.get(c)
        if primary:
            seeds.add(primary)

    visited: set[str] = set()
    frontier = set(seeds)
    while frontier:
        m = frontier.pop()
        if m in visited:
            continue
        visited.add(m)
        for importer_rel in graph.importers.get(m, ()):
            if _is_test_file(importer_rel):
                affected.add(importer_rel)
                continue
            im_primary = graph.primary.get(importer_rel)
            if im_primary and im_primary not in visited:
                frontier.add(im_primary)
        # No unconditional climb to enclosing packages: package-hub nodes collect enormous
        # importer sets and blow straight through the shared-core guard. Real re-export paths are
        # already covered by edges: `from pkg import x` resolves to pkg/__init__.py, and if that
        # __init__ imports the changed module the chain is complete without any synthetic climb.

    if test_files and len(affected) / len(test_files) >= shared_core_fraction:
        return Impact("all", [],
                      f"change reaches {len(affected)}/{len(test_files)} tests "
                      f"(>= {shared_core_fraction:.0%}); shared core — running full suite")
    return Impact("subset", sorted(affected), f"{len(affected)}/{len(test_files)} tests affected")
