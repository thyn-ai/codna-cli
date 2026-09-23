"""What a repository says about its own dependencies -- read for review grounding.

The repo-side half of :mod:`codna.review_grounding`: lockfile and manifest readers (package-lock.json,
pnpm-lock.yaml, yarn.lock, uv.lock, poetry.lock, requirements*.txt, package.json), the runtime a
repository declares (``.nvmrc`` / ``.node-version`` / ``node-version:`` in workflows / ``FROM node:N``
in a Dockerfile / ``engines.node``; ``.python-version`` / ``python-version:`` / ``requires-python``)
and a small node-semver + PEP 440 range evaluator. Pure and offline: nothing here touches the network.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# The Node majors a hosting platform can be running today (Vercel picks the newest one that satisfies
# ``engines.node``). Only consulted when the repo declares a RANGE and pins nothing.
_NODE_MAJORS = (26, 24, 22, 20, 18)


# ── Lockfile / manifest readers ────────────────────────────────────────────────
@dataclass
class LockEntry:
    name: str
    version: str | None = None
    integrity: str | None = None
    license: str | None = None
    hashes: tuple[str, ...] = ()   # PyPI: sha256 hex digests the lockfile records


def _read(path: Path, limit: int = 40_000_000) -> str | None:
    try:
        if path.is_file() and path.stat().st_size <= limit:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return None


def _license_str(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict) and isinstance(value.get("type"), str):
        return value["type"].strip() or None
    return None


def _package_lock_entries(text: str) -> dict[str, LockEntry]:
    out: dict[str, LockEntry] = {}
    try:
        data = json.loads(text)
    except ValueError:
        return out
    if not isinstance(data, dict):
        return out
    packages = data.get("packages")
    if isinstance(packages, dict):                   # lockfileVersion 2/3
        for key, val in packages.items():
            if not isinstance(val, dict) or "node_modules/" not in key:
                continue
            name = key.rsplit("node_modules/", 1)[1]
            if name in out and key.count("node_modules/") > 1:
                continue                             # a hoisted (top-level) entry wins over a nested one
            out[name] = LockEntry(name=name, version=val.get("version"), integrity=val.get("integrity"),
                                  license=_license_str(val.get("license")))
    deps = data.get("dependencies")
    if isinstance(deps, dict):                       # lockfileVersion 1 (or the v2 compatibility tree)
        for name, val in deps.items():
            if isinstance(val, dict) and name not in out:
                out[name] = LockEntry(name=name, version=val.get("version"), integrity=val.get("integrity"))
    return out


_PNPM_KEY_RE = re.compile(r"^  ['\"]?/?(?P<name>@?[^@'\"\s/(]+(?:/[^@'\"\s/(]+)?)@(?P<ver>[^'\"\s(:]+)[^:]*:\s*$")
_PNPM_INTEGRITY_RE = re.compile(r"integrity:\s*(sha\d+-[A-Za-z0-9+/=]+)")


def _pnpm_lock_entries(text: str) -> dict[str, LockEntry]:
    out: dict[str, LockEntry] = {}
    current: LockEntry | None = None
    in_packages = False
    for line in text.splitlines():
        if line and not line[0].isspace():
            in_packages = line.startswith("packages:")
            current = None
            continue
        if not in_packages:
            continue
        m = _PNPM_KEY_RE.match(line)
        if m:
            name, version = m.group("name"), m.group("ver")
            existing = out.get(name)
            if existing is None:
                current = out[name] = LockEntry(name=name, version=version)
            elif existing.version == version:
                current = existing          # the same version under another peer suffix: the same tarball
            else:
                current = None              # a second version of the name: first seen wins, and its
            continue                        # integrity is never overwritten by the other's resolution
        if current is not None and current.integrity is None:
            im = _PNPM_INTEGRITY_RE.search(line)
            if im:
                current.integrity = im.group(1)
    return out


_YARN_HEADER_RE = re.compile(r"^\"?(?P<name>@?[^@\"\s,]+(?:/[^@\"\s,]+)?)@")
_YARN_VERSION_RE = re.compile(r"^\s+version:?\s*\"?([^\"\s]+)\"?")
_YARN_INTEGRITY_RE = re.compile(r"^\s+integrity:?\s*\"?(sha\d+-[A-Za-z0-9+/=]+)")


def _yarn_lock_entries(text: str) -> dict[str, LockEntry]:
    out: dict[str, LockEntry] = {}
    current: LockEntry | None = None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            m = _YARN_HEADER_RE.match(line)
            current = out.setdefault(m.group("name"), LockEntry(name=m.group("name"))) if m else None
            continue
        if current is None:
            continue
        vm = _YARN_VERSION_RE.match(line)
        if vm and current.version is None:
            current.version = vm.group(1)
            continue
        im = _YARN_INTEGRITY_RE.match(line)
        if im and current.integrity is None:
            current.integrity = im.group(1)
    return out


_TOML_PKG_RE = re.compile(r"^\[\[package\]\]\s*$")
_TOML_NAME_RE = re.compile(r"^name\s*=\s*\"([^\"]+)\"")
_TOML_VERSION_RE = re.compile(r"^version\s*=\s*\"([^\"]+)\"")
_SHA256_RE = re.compile(r"sha256[:=]\s*\"?([0-9a-f]{64})")


def _toml_lock_entries(text: str) -> dict[str, LockEntry]:
    """uv.lock / poetry.lock: ``[[package]]`` blocks with ``name``/``version`` and sha256 hashes."""
    out: dict[str, LockEntry] = {}
    current: LockEntry | None = None
    hashes: list[str] = []

    def _flush() -> None:
        if current is not None and current.name:
            current.hashes = tuple(hashes)
            out.setdefault(current.name.lower(), current)

    for line in text.splitlines():
        if _TOML_PKG_RE.match(line):
            _flush()
            current, hashes = LockEntry(name=""), []
            continue
        if current is None:
            continue
        nm, vm = _TOML_NAME_RE.match(line), _TOML_VERSION_RE.match(line)
        if nm:
            current.name = nm.group(1)
        elif vm:
            current.version = vm.group(1)
        hashes.extend(_SHA256_RE.findall(line))
    _flush()
    return out


_REQ_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*(?P<ver>[^\s;\\]+)")


def _requirements_entries(text: str) -> dict[str, LockEntry]:
    out: dict[str, LockEntry] = {}
    for line in text.replace("\\\n", " ").splitlines():
        m = _REQ_RE.match(line)
        if not m:
            continue
        entry = LockEntry(name=m.group("name"), version=m.group("ver"),
                          hashes=tuple(re.findall(r"--hash=sha256:([0-9a-f]{64})", line)))
        out.setdefault(entry.name.lower(), entry)
    return out


def _package_json_deps(text: str) -> dict[str, str]:
    """Every dependency spec in a package.json (all sections), name -> range."""
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
            block = data.get(section)
            if isinstance(block, dict):
                for k, v in block.items():
                    if isinstance(v, str):
                        out.setdefault(k, v)
    return out


_LOCK_READERS = {
    "package-lock.json": _package_lock_entries, "npm-shrinkwrap.json": _package_lock_entries,
    "pnpm-lock.yaml": _pnpm_lock_entries, "yarn.lock": _yarn_lock_entries,
    "uv.lock": _toml_lock_entries, "poetry.lock": _toml_lock_entries,
}
_NPM_LOCKS = ("package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock")
_PY_LOCKS = ("uv.lock", "poetry.lock")
# package-lock.json keys that open an object but are not a package entry (lockfileVersion 1 tree).
_LOCK_STRUCTURAL_KEYS = frozenset({"dependencies", "devDependencies", "optionalDependencies", "peerDependencies",
                                   "peerDependenciesMeta", "requires", "packages", "engines", "bin", "funding",
                                   "dist", "os", "cpu", "scripts", "workspaces", "overrides"})


class RepoManifests:
    """What the repository (at the reviewed head) says about its packages, read lazily per directory:
    lockfile entries and manifest specs nearest the finding's path, then the repo root."""

    def __init__(self, repo_dir: str, changed_files: list[str]):
        self.root = Path(repo_dir)
        self.changed = list(changed_files or [])
        self._locks: dict[tuple[Path, str], dict[str, LockEntry]] = {}
        self._specs: dict[Path, dict[str, str]] = {}

    def _dirs_for(self, path: str) -> list[Path]:
        out: list[Path] = []
        d = Path(path).parent
        while True:
            cand = self.root / d
            if cand not in out:
                out.append(cand)
            if str(d) in ("", "."):
                break
            d = d.parent
        if self.root not in out:
            out.append(self.root)
        return out

    def lock_entries(self, path: str, eco: str) -> dict[str, LockEntry]:
        merged: dict[str, LockEntry] = {}
        for d in self._dirs_for(path):
            for entry in self._lock_dir(d, eco).values():
                merged.setdefault(entry.name.lower(), entry)
        return merged

    def _lock_dir(self, d: Path, eco: str) -> dict[str, LockEntry]:
        key = (d, eco)
        if key in self._locks:
            return self._locks[key]
        found: dict[str, LockEntry] = {}
        for name in (_NPM_LOCKS if eco == "npm" else _PY_LOCKS):
            text = _read(d / name)
            if text:
                for k, v in _LOCK_READERS[name](text).items():
                    found.setdefault(k.lower(), v)
        if eco == "pypi":
            try:
                for req in sorted(d.glob("requirements*.txt")):
                    text = _read(req)
                    if text:
                        for k, v in _requirements_entries(text).items():
                            found.setdefault(k.lower(), v)
            except OSError:
                pass
        self._locks[key] = found
        return found

    def specs(self, path: str) -> dict[str, str]:
        """package.json dependency ranges nearest ``path`` (npm only)."""
        merged: dict[str, str] = {}
        for d in self._dirs_for(path):
            if d not in self._specs:
                text = _read(d / "package.json")
                self._specs[d] = _package_json_deps(text) if text else {}
            for k, v in self._specs[d].items():
                merged.setdefault(k, v)
        return merged

    def candidates(self, path: str, eco: str) -> set[str]:
        """Package names the change can be about: everything in the nearest lockfile(s) + manifest."""
        names = {e.name for e in self.lock_entries(path, eco).values()}
        if eco == "npm":
            names |= set(self.specs(path))
        return names

    def package_at_line(self, path: str, line: int) -> str | None:
        """The package the lockfile/manifest entry containing ``line`` belongs to, or None."""
        text = _read(self.root / path)
        if not text or line < 1:
            return None
        lines = text.splitlines()
        if not lines:
            return None
        base = os.path.basename(path).lower()
        idx = min(line, len(lines)) - 1
        if base in ("package-lock.json", "npm-shrinkwrap.json"):
            for i in range(idx, -1, -1):
                m = re.match(r'^\s*"(?P<key>[^"]+)":\s*\{\s*$', lines[i])
                if not m:
                    continue
                key = m.group("key")
                if "node_modules/" in key:
                    return key.rsplit("node_modules/", 1)[1]
                if key not in _LOCK_STRUCTURAL_KEYS and lines[i].startswith("      "):
                    return key                       # lockfileVersion 1: "dependencies": { "<name>": {
            return None
        if base == "pnpm-lock.yaml":
            for i in range(idx, -1, -1):
                m = _PNPM_KEY_RE.match(lines[i])
                if m:
                    return m.group("name")
                if lines[i] and not lines[i][0].isspace():
                    return None
            return None
        if base == "yarn.lock":
            for i in range(idx, -1, -1):
                if lines[i] and not lines[i][0].isspace():
                    m = _YARN_HEADER_RE.match(lines[i])
                    return m.group("name") if m else None
            return None
        if base == "package.json":
            m = re.match(r'^\s*"(?P<name>[^"]+)"\s*:\s*"[^"]*"', lines[idx])
            return m.group("name") if m else None
        if base in _PY_LOCKS:
            start = next((i for i in range(idx, -1, -1) if _TOML_PKG_RE.match(lines[i])), None)
            if start is None:
                return None
            for j in range(start + 1, min(start + 8, len(lines))):
                nm = _TOML_NAME_RE.match(lines[j])
                if nm:
                    return nm.group(1)
            return None
        if base.startswith("requirements") and base.endswith(".txt"):
            m = _REQ_RE.match(lines[idx])
            return m.group("name") if m else None
        return None


# ── Declared runtimes ──────────────────────────────────────────────────────────
_NODE_VERSION_RE = re.compile(r"node-version:\s*(?:\[([^\]]*)\]|['\"]?(v?\d+(?:\.\d+)*(?:\.x)?)['\"]?)", re.I)
_MATRIX_NODE_RE = re.compile(r"^\s*node(?:[-_]?version)?:\s*\[([^\]]*)\]", re.I | re.M)
_DOCKER_NODE_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(?:[\w.-]+/)*node:(\d+)", re.I | re.M)
_PY_VERSION_RE = re.compile(r"python-version:\s*(?:\[([^\]]*)\]|['\"]?(\d+(?:\.\d+)*)['\"]?)", re.I)
_VERSION_TOKEN_RE = re.compile(r"v?(\d+(?:\.\d+){0,2})(?:\.x)?")


def _versions_in(blob: str) -> list[str]:
    return [m.group(1) for m in _VERSION_TOKEN_RE.finditer(blob or "") if m.group(1)]


def _workflow_files(root: Path) -> list[Path]:
    try:
        return sorted((root / ".github" / "workflows").glob("*.y*ml"))
    except OSError:
        return []


def declared_node_runtimes(repo_dir: str, near: str | None = None) -> list[str]:
    """The Node versions this repository says it runs on. Pins first -- ``.nvmrc`` / ``.node-version``,
    ``node-version:`` (and ``node: [...]`` matrices) in workflows, ``FROM node:N`` in Dockerfiles -- and
    only when nothing is pinned, the newest supported major inside ``package.json``'s ``engines.node``
    (a range is a compatibility promise, not a runtime: a Vercel site declaring ``>=20`` runs on the
    newest Node Vercel offers). Empty when the repo declares nothing."""
    root = Path(repo_dir)
    dirs = [root]
    if near:
        d = root / Path(near).parent
        if d != root:
            dirs.insert(0, d)
    pins: list[str] = []
    for d in dirs:
        for name in (".nvmrc", ".node-version"):
            text = (_read(d / name) or "").strip()
            if text:
                pins.extend(_versions_in(text.splitlines()[0]))
    for wf in _workflow_files(root):
        text = _read(wf) or ""
        for m in _NODE_VERSION_RE.finditer(text):
            pins.extend(_versions_in(m.group(1) or m.group(2) or ""))
        for m in _MATRIX_NODE_RE.finditer(text):
            pins.extend(_versions_in(m.group(1)))
    try:
        dockerfiles = sorted(root.glob("Dockerfile*")) + sorted(root.glob("*/Dockerfile*")) + sorted(root.glob("infra/**/Dockerfile*"))
    except OSError:
        dockerfiles = []
    for df in dockerfiles:
        pins.extend(_DOCKER_NODE_RE.findall(_read(df) or ""))
    pins = [p for p in dict.fromkeys(pins) if _parse_version(p) is not None]
    if pins:
        return pins
    for d in dirs:
        text = _read(d / "package.json")
        if not text:
            continue
        try:
            engines = (json.loads(text).get("engines") or {}).get("node")
        except (ValueError, AttributeError):
            engines = None
        if isinstance(engines, str) and engines.strip():
            for major in _NODE_MAJORS:
                if satisfies(str(major), engines) is True:
                    return [str(major)]
    return []


def declared_python_runtimes(repo_dir: str, near: str | None = None) -> list[str]:
    """The Python versions the repository pins: ``.python-version``, ``python-version:`` in workflows;
    else the newest 3.x inside ``requires-python``."""
    root = Path(repo_dir)
    pins: list[str] = []
    for d in ([root / Path(near).parent] if near else []) + [root]:
        text = (_read(d / ".python-version") or "").strip()
        if text:
            pins.extend(_versions_in(text.splitlines()[0]))
    for wf in _workflow_files(root):
        for m in _PY_VERSION_RE.finditer(_read(wf) or ""):
            pins.extend(_versions_in(m.group(1) or m.group(2) or ""))
    pins = [p for p in dict.fromkeys(pins) if _parse_version(p) is not None and p.startswith("3")]
    if pins:
        return pins
    text = _read(root / "pyproject.toml") or ""
    m = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
    if m:
        for minor in range(15, 7, -1):
            if satisfies(f"3.{minor}", m.group(1)) is True:
                return [f"3.{minor}"]
    return []


# ── Version ranges (node semver ranges + simple PEP 440) ───────────────────────
_V3 = tuple[int, int, int]


def _parse_version(text: str) -> tuple[_V3, int] | None:
    """``(major, minor, patch), precision`` for ``22``, ``22.11``, ``v22.11.0``, ``3.12``; None otherwise."""
    m = re.fullmatch(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+.][0-9A-Za-z.*+-]*)?\s*", text or "")
    if not m:
        return None
    parts = [m.group(1), m.group(2), m.group(3)]
    precision = sum(1 for p in parts if p is not None)
    nums = tuple(int(p) if p is not None else 0 for p in parts)
    return (nums[0], nums[1], nums[2]), precision


def _bounds(version: str) -> list[_V3] | None:
    """The concrete versions a possibly partial version stands for: ``22`` is anything in 22.x.y, so both
    ends are tried and a range is satisfied when either end is (``^22.12.0`` accepts runtime "22")."""
    parsed = _parse_version(version)
    if parsed is None:
        return None
    (ma, mi, pa), precision = parsed
    if precision >= 3:
        return [(ma, mi, pa)]
    if precision == 2:
        return [(ma, mi, 0), (ma, mi, 10**6)]
    return [(ma, 0, 0), (ma, 10**6, 10**6)]


_CMP_RE = re.compile(r"^(?P<op>>=|<=|==|!=|~=|>|<|=|\^|~)?\s*v?(?P<ma>\d+|x|X|\*)(?:\.(?P<mi>\d+|x|X|\*))?(?:\.(?P<pa>\d+|x|X|\*))?(?:[-+][0-9A-Za-z.-]+)?$")


def _is_wild(part: str | None) -> bool:
    return part is None or part in ("x", "X", "*")


def _comparator(token: str) -> Callable[[_V3], bool] | None:
    token = token.strip()
    if token in ("", "*", "x", "X", "latest"):
        return lambda v: True
    m = _CMP_RE.match(token)
    if not m:
        return None
    op = m.group("op") or ""
    ma_s, mi_s, pa_s = m.group("ma"), m.group("mi"), m.group("pa")
    if _is_wild(ma_s):
        return lambda v: True
    ma = int(ma_s)
    mi = None if _is_wild(mi_s) else int(mi_s)
    pa = None if _is_wild(pa_s) else int(pa_s)
    low: _V3 = (ma, mi or 0, pa or 0)
    if op in ("", "=", "==", "!="):
        def equal(v: _V3) -> bool:
            """Everything the bare token stands for: ``3`` is 3.x.y, ``3.9`` (or ``3.9.*``) is 3.9.z."""
            if mi is None:
                return (ma, 0, 0) <= v < (ma + 1, 0, 0)
            if pa is None:
                return (ma, mi, 0) <= v < (ma, mi + 1, 0)
            return v == low

        if op == "!=":
            return lambda v: not equal(v)   # the exclusion covers the same set the bare token would match
        return equal
    if op == ">=":
        return lambda v: v >= low
    if op == ">":
        if mi is None:
            return lambda v: v >= (ma + 1, 0, 0)
        if pa is None:
            return lambda v: v >= (ma, mi + 1, 0)
        return lambda v: v > low
    if op == "<":
        return lambda v: v < low
    if op == "<=":
        if mi is None:
            return lambda v: v < (ma + 1, 0, 0)
        if pa is None:
            return lambda v: v < (ma, mi + 1, 0)
        return lambda v: v <= low
    if op == "^":
        if ma > 0 or mi is None:
            high: _V3 = (ma + 1, 0, 0)
        elif mi > 0 or pa is None:
            high = (0, mi + 1, 0)
        else:
            high = (0, 0, (pa or 0) + 1)
        return lambda v: low <= v < high
    if op in ("~", "~="):
        if mi is None:
            return lambda v: (ma, 0, 0) <= v < (ma + 1, 0, 0)
        if op == "~=" and pa is None:
            return lambda v: low <= v < (ma + 1, 0, 0)     # PEP 440 ~=3.8 -> >=3.8,<4
        return lambda v: low <= v < (ma, mi + 1, 0)
    return None


def satisfies(version: str, range_text: str) -> bool | None:
    """Does ``version`` (possibly partial: ``"22"``) satisfy a node-semver / simple PEP 440 range?
    None when the range cannot be parsed -- the caller treats that as unverifiable, never as a fact."""
    ends = _bounds(version)
    if ends is None or not isinstance(range_text, str):
        return None
    parsed_alts: list[list[Callable[[_V3], bool]]] = []
    for alt in range_text.split("||"):
        alt = alt.strip()
        if " - " in alt:                                            # hyphen range  a - b
            lo, _, hi = alt.partition(" - ")
            cmp_lo, cmp_hi = _comparator(">=" + lo.strip()), _comparator("<=" + hi.strip())
            if cmp_lo is None or cmp_hi is None:
                return None
            parsed_alts.append([cmp_lo, cmp_hi])
            continue
        comps: list[Callable[[_V3], bool]] = []
        for token in re.split(r"[,\s]+", alt):
            if not token:
                continue
            c = _comparator(token)
            if c is None:
                return None
            comps.append(c)
        parsed_alts.append(comps or [lambda v: True])
    return any(all(c(v) for c in comps) for comps in parsed_alts for v in ends)
