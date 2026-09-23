"""Ground a review finding's checkable *registry claims* in the npm / PyPI registries.

The review agent reads a dependency manifest or lockfile diff and states facts about packages --
"``tinyexec@1.3.0`` does not exist on npm", "malformed SHA-512 integrity hash", "the lock file's
license is wrong", "``@eloqnt/*`` have zero downloads", "requires Node >=22 but the project declares
Node 20", "peer dependency violated" -- and rates them HIGH. Nothing in the pipeline checked any of
them: ``review_findings.normalize_finding`` validates the *shape* of a finding, not its claims, and the
model has no registry access (thyn-ai: sqai#53, codna-site#54, cohenta-site#15, accounts#71,
telys#134, ...). Every one of those was false, and every one had to be refuted by hand.

This module runs between normalisation and the noise controls (``review_findings.finalize_findings``)
and, for a finding that makes a claim of one of the kinds below about a package it names -- or, when
the finding sits on a manifest/lockfile, about the package at its anchored line or one the change
touches -- asks the registry:

* **version existence**  ``GET registry.npmjs.org/<name>/<version>`` / ``pypi.org/pypi/<name>/<version>/json``
* **integrity**          the lockfile's (or the finding's) SRI / sha256 against ``dist.integrity`` / ``urls[].digests``
* **license**            the lockfile's recorded license against the registry's
* **downloads**          ``api.npmjs.org/downloads/point/last-month/<name>`` (npm only)
* **engines**            ``engines.node`` / ``requires_python`` against the runtime the repo actually
                         declares (.nvmrc, .node-version, ``node-version:`` in workflows, ``FROM node:N``
                         in a Dockerfile, else the newest supported major inside ``engines.node``)
* **peer**               ``peerDependencies`` + ``peerDependenciesMeta.optional`` against what the
                         lockfile / manifest installs; an optional peer is never a violation

A claim the registry **contradicts** is dropped (and logged); one it **confirms** keeps its severity
(the explanation says so); one that **cannot be checked** -- no network, a timeout, an unparseable
answer -- is downgraded to LOW with "Unverified" in its text, so it reads as a question, never as a
fact. Offline is the safe default: ``CI_OFFLINE_CONTRACT_TEST=1`` and ``CODNA_REQUIRE_EGRESS_DENY=1``
(``privacy.egress: fail-closed``) both mean "ask nothing"; tests inject a recorded ``fetch``.

Egress: the review process already talks to api.github.com on every posted review; the two npm hosts
and pypi.org are the only additions, each request bounded by ``_TIMEOUT_S`` and the whole pass by
``_MAX_LOOKUPS``. Codna's kernel-level egress denial (netjail / sandbox) applies to the *repo's test
run*, never to this process, and is untouched.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from .review_findings import CodnaReviewFinding
from .review_manifests import (
    RepoManifests,
    _license_str,
    declared_node_runtimes,
    declared_python_runtimes,
    satisfies,
)

NPM_REGISTRY = "https://registry.npmjs.org"
NPM_DOWNLOADS = "https://api.npmjs.org/downloads/point/last-month"
PYPI = "https://pypi.org/pypi"

_TIMEOUT_S = 5.0
_MAX_LOOKUPS = 24            # registry requests per review pass; beyond it a claim is unverified
_LOW_DOWNLOADS_PER_MONTH = 1000   # below this a "zero/low downloads" claim is taken as confirmed

CONTRADICTED, CONFIRMED, UNVERIFIED = "contradicted", "confirmed", "unverified"

# ``Fetch(url) -> (status_code, parsed_json_or_None)``; ``(0, None)`` = no answer (offline/timeout).
Fetch = Callable[[str], tuple[int, object]]

_NPM_MANIFESTS = frozenset({"package.json", "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml",
                            "yarn.lock", "bun.lock", "bun.lockb"})
_PY_MANIFESTS = frozenset({"pyproject.toml", "poetry.lock", "uv.lock", "pipfile", "pipfile.lock", "setup.py",
                           "setup.cfg", "pixi.toml", "pixi.lock"})


def ecosystem_of(path: str) -> str | None:
    """``"npm"`` / ``"pypi"`` for a dependency manifest or lockfile path, else None."""
    base = os.path.basename(path or "").lower()
    if base in _NPM_MANIFESTS:
        return "npm"
    if base in _PY_MANIFESTS or (base.startswith("requirements") and base.endswith(".txt")):
        return "pypi"
    return None


# ── Claim detection ────────────────────────────────────────────────────────────
_CLAIM_RES: dict[str, re.Pattern[str]] = {
    "version_exists": re.compile(
        r"(?:does not|doesn'?t|do not|don'?t|did not|didn'?t) exist|\bnon-?existent\b|\bnot (?:been )?published\b"
        r"|\bno such version\b|\bunpublished\b|\bnot (?:found |available |present )?(?:on|in) (?:the )?(?:npm|pypi|registry)\b"
        r"|\bcannot be found\b|\bphantom (?:version|package|release)\b|\bfabricated (?:version|package)\b"
        r"|\bhallucinated\b|\bno (?:published )?(?:version|release) [\w.-]+ (?:exists|on|in)\b", re.I),
    "integrity": re.compile(
        r"\b(?:integrity|sri|sha-?512|sha-?256|sha-?1|checksum|hash)\b[^.]{0,120}?"
        r"\b(?:malformed|mismatch(?:ed|es)?|invalid|incorrect|wrong|tamper|does not match|doesn'?t match|truncated|corrupt(?:ed)?|structurally)\b"
        r"|\b(?:malformed|mismatch(?:ed|es)?|invalid|incorrect|wrong|truncated|corrupt(?:ed)?)\b[^.]{0,60}?"
        r"\b(?:integrity|sri|sha-?512|sha-?256|checksum|hash)\b", re.I),
    "license": re.compile(
        r"\blicen[cs]e\b[^.]{0,120}?\b(?:wrong|incorrect|mismatch(?:ed|es)?|changed|different|does not match|doesn'?t match|invalid|misreport(?:ed|s)?|stale|outdated)\b"
        r"|\b(?:wrong|incorrect|mismatch(?:ed|es)?|invalid|stale|outdated)\b[^.]{0,60}?\blicen[cs]e\b", re.I),
    "downloads": re.compile(
        r"\b(?:zero|no|0|low|few|little|minimal|negligible)(?:[- ](?:recorded|weekly|monthly))?[- ]download", re.I),
    "engines": re.compile(
        r"\bengines?\b|\brequires? node(?:\.js)?\b|\bnode(?:\.js)?\s*(?:>=|≥|>|version)\s*\d|\bminimum node\b"
        r"|\bnode(?:\.js)? \d+(?:\.\d+)* (?:or (?:higher|later|newer)|\+)|\bincompatible with node\b|\brequires?[_ ]python\b", re.I),
    "peer": re.compile(r"\bpeer[- ]?dep|\bpeerdependenc|\bpeer (?:requirement|range|constraint)", re.I),
}


def detect_claims(text: str) -> set[str]:
    """The registry-checkable claim kinds a finding's text makes (possibly several)."""
    return {kind for kind, rx in _CLAIM_RES.items() if rx.search(text or "")}


# ── Package identification ─────────────────────────────────────────────────────
_NAME = r"(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*"
_VER = r"v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?"
# ``name@1.2.3`` (npm spec), ``name 1.2.3`` / ``name==1.2.3`` (prose / PEP 440).
_SPEC_RE = re.compile(rf"(?<![\w@/.-])(?P<name>{_NAME})(?P<sep>@|==|\s+v?|\s+@)(?P<ver>{_VER})(?![\w.-])", re.I)
_SCOPED_RE = re.compile(r"(?<![\w@/.-])(@[a-z0-9][a-z0-9._-]*/(?:[a-z0-9][a-z0-9._-]*|\*))", re.I)
_QUOTED_RE = re.compile(rf"[`'\"](?P<name>{_NAME})[`'\"]", re.I)


@dataclass(frozen=True)
class PackageRef:
    name: str
    version: str | None


def packages_in_text(text: str, candidates: set[str], anchored: str | None = None, *, title: str = "") -> list[PackageRef]:
    """The packages a finding is about, in the order they are named.

    Explicit mentions first (``name@1.2.3`` / ``name 1.2.3`` for a candidate or a hyphenated / scoped
    name, any ``@scope/name``, a quoted candidate), then the package the finding's anchored line belongs
    to, then any *candidate* (a package the change touches) named as a whole word -- but a bare
    single word is only taken as a package when it is in the title or contains ``-``/``.``/``/``:
    "debug" in prose is not the ``debug`` package. ``@scope/*`` expands to every candidate in that
    scope."""
    text = text or ""
    found: dict[str, str | None] = {}
    lower_candidates = {c.lower(): c for c in candidates}
    for m in _SPEC_RE.finditer(text):
        name = m.group("name")
        explicit = m.group("sep").strip() in ("@", "==")       # ``name@1.2.3`` / ``name==1.2.3`` is a spec, whatever the name
        if explicit or name.startswith("@") or name.lower() in lower_candidates or "-" in name or "." in name:
            found.setdefault(name.lower(), m.group("ver").lstrip("v"))
    for m in _SCOPED_RE.finditer(text):
        name = m.group(1)
        if name.endswith("/*"):
            scope = name[:-1].lower()
            for cand in sorted(candidates):
                if cand.lower().startswith(scope):
                    found.setdefault(cand.lower(), None)
        else:
            found.setdefault(name.lower(), None)
    for m in _QUOTED_RE.finditer(text):
        name = m.group("name")
        if name.lower() in lower_candidates:
            found.setdefault(name.lower(), None)
    if anchored:
        found.setdefault(anchored.lower(), None)
    title_l = (title or "").lower()
    for cand in sorted(candidates):
        key = cand.lower()
        if key in found:
            continue
        if not (any(ch in cand for ch in "-./@") or key in title_l):
            continue
        if re.search(rf"(?<![\w@/.-]){re.escape(cand)}(?![\w/-])", text, re.I):
            found[key] = None
    return [PackageRef(name=lower_candidates.get(key, key), version=ver) for key, ver in found.items()]


# ── Registry access ────────────────────────────────────────────────────────────
def offline_by_policy() -> bool:
    """No registry request may leave this process: the offline CI contract, or fail-closed egress."""
    return os.environ.get("CI_OFFLINE_CONTRACT_TEST") == "1" or os.environ.get("CODNA_REQUIRE_EGRESS_DENY") == "1"


def _httpx_fetch(url: str) -> tuple[int, object]:
    try:
        import httpx
    except Exception:  # noqa: BLE001 -- no client, no answer
        return 0, None
    try:
        r = httpx.get(url, headers={"Accept": "application/json"}, timeout=_TIMEOUT_S, follow_redirects=True)
    except Exception:  # noqa: BLE001 -- offline / timeout / DNS: unverified, never a fact
        return 0, None
    try:
        body = r.json() if r.content else None
    except ValueError:
        body = None
    return int(r.status_code), body


class Registry:
    """Cached, bounded registry lookups. ``fetch`` is injectable (tests pass recorded answers);
    ``offline`` answers nothing at all. Every method returns ``(status, body)`` where status 0 means
    "no answer" and the caller must report the claim as unverified."""

    def __init__(self, fetch: Fetch | None = None, *, offline: bool | None = None):
        self.offline = offline_by_policy() if offline is None else offline
        self._fetch = fetch or _httpx_fetch
        self._cache: dict[str, tuple[int, object]] = {}
        self.lookups = 0

    def get(self, url: str) -> tuple[int, object]:
        if url in self._cache:
            return self._cache[url]
        if self.offline or self.lookups >= _MAX_LOOKUPS:
            return 0, None
        self.lookups += 1
        try:
            status, body = self._fetch(url)
        except Exception:  # noqa: BLE001 -- a fetch that raises is "no answer"
            status, body = 0, None
        self._cache[url] = (int(status or 0), body)
        return self._cache[url]

    def npm_version(self, name: str, version: str) -> tuple[int, dict]:
        status, body = self.get(f"{NPM_REGISTRY}/{quote(name, safe='@')}/{quote(version, safe='')}")
        return status, body if isinstance(body, dict) else {}

    def npm_exists(self, name: str) -> tuple[int, dict]:
        status, body = self.get(f"{NPM_REGISTRY}/{quote(name, safe='@')}/latest")
        return status, body if isinstance(body, dict) else {}

    def npm_downloads(self, name: str) -> tuple[int, int | None]:
        status, body = self.get(f"{NPM_DOWNLOADS}/{name}")
        count = body.get("downloads") if isinstance(body, dict) else None
        return status, count if isinstance(count, int) else None

    def pypi_version(self, name: str, version: str) -> tuple[int, dict]:
        status, body = self.get(f"{PYPI}/{quote(name, safe='')}/{quote(version, safe='')}/json")
        return status, body if isinstance(body, dict) else {}


# ── Verdicts ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Verdict:
    kind: str
    package: str
    outcome: str      # CONTRADICTED | CONFIRMED | UNVERIFIED
    note: str


_SRI_RE = re.compile(r"\bsha(?:1|256|384|512)-[A-Za-z0-9+/]+=*")
_HEX256_RE = re.compile(r"\b[0-9a-f]{64}\b")
_LICENSE_QUOTE_RE = re.compile(r"[\"']?licen[cs]e[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", re.I)


def _min_version_of_spec(spec: str) -> str | None:
    """The lowest version a manifest range admits (``^2.108.2`` -> 2.108.2; ``>=1.2`` -> 1.2)."""
    m = re.search(r"\d+(?:\.\d+){0,2}", spec or "")
    return m.group(0) if m else None


class Grounder:
    """One review pass's registry grounding: shared cache + lookup budget across all findings."""

    def __init__(self, repo_dir: str, changed_files: list[str], *, registry: Registry | None = None,
                 log: Callable[[str, dict], None] | None = None):
        self.repo_dir = repo_dir
        self.manifests = RepoManifests(repo_dir, changed_files)
        self.registry = registry or Registry()
        self.changed_ecos = {e for e in (ecosystem_of(p) for p in changed_files or []) if e}
        self._log = log or _log_event
        self._node_runtimes: dict[str, list[str]] = {}
        self._py_runtimes: dict[str, list[str]] = {}

    # -- entry point ------------------------------------------------------------------------------
    def __call__(self, findings: list[CodnaReviewFinding]) -> tuple[list[CodnaReviewFinding], dict[str, int]]:
        counts = {"registry_contradicted": 0, "registry_confirmed": 0, "registry_unverified": 0}
        kept: list[CodnaReviewFinding] = []
        for f in findings:
            verdicts = self.verdicts_for(f)
            if not verdicts:
                kept.append(f)
                continue
            outcomes = {v.outcome for v in verdicts}
            if CONFIRMED in outcomes:
                counts["registry_confirmed"] += 1
                kept.append(_confirmed(f, verdicts))
            elif outcomes == {CONTRADICTED}:
                counts["registry_contradicted"] += 1
                self._log("review_finding_registry_contradicted", {
                    "path": f.path, "line": f.line, "severity": f.severity, "title": f.title,
                    "fingerprint": f.fingerprint, "checks": [v.note for v in verdicts],
                })
            else:
                counts["registry_unverified"] += 1
                kept.append(_unverified(f, verdicts))
        return kept, counts

    # -- per finding ------------------------------------------------------------------------------
    def verdicts_for(self, f: CodnaReviewFinding) -> list[Verdict]:
        text = f"{f.title}\n{f.explanation}"
        kinds = detect_claims(text)
        if not kinds:
            return []
        on_manifest = ecosystem_of(f.path)
        eco = on_manifest or (next(iter(self.changed_ecos)) if len(self.changed_ecos) == 1 else None)
        if eco is None:
            return []
        # Only a finding ON a manifest/lockfile may be about a package it does not name outright (the
        # entry at its line, or a whole-word candidate): a finding on source code that happens to say
        # "hash mismatch" next to a dependency's name must not be grounded away.
        candidates = self.manifests.candidates(f.path, eco) if on_manifest else set()
        anchored = self.manifests.package_at_line(f.path, f.line) if on_manifest else None
        packages = packages_in_text(text, candidates, anchored, title=f.title)
        if not packages:
            return []
        entries = self.manifests.lock_entries(f.path, eco)
        out: list[Verdict] = []
        for pkg in packages:
            entry = entries.get(pkg.name.lower())
            version = pkg.version or (entry.version if entry else None)
            for kind in sorted(kinds):
                check = getattr(self, f"_check_{eco}_{kind}", None)
                if check is None:
                    out.append(Verdict(kind, pkg.name, UNVERIFIED, f"{kind}: not checkable on {eco}"))
                    continue
                out.append(check(f, pkg.name, version, entry, text, packages))
        return out

    # -- npm --------------------------------------------------------------------------------------
    def _check_npm_version_exists(self, f, name, version, entry, text, packages) -> Verdict:
        status, _ = self.registry.npm_version(name, version) if version else self.registry.npm_exists(name)
        label = f"{name}@{version}" if version else name
        if status == 200:
            return Verdict("version_exists", name, CONTRADICTED, f"{label} is published on npm")
        if status == 404:
            return Verdict("version_exists", name, CONFIRMED, f"npm has no {label}")
        return Verdict("version_exists", name, UNVERIFIED, f"npm registry unreachable for {label}")

    def _npm_pinned(self, name: str, version: str | None, kind: str) -> tuple[dict | None, Verdict | None]:
        """The registry manifest of ``name@version``, or the verdict that stands in for it."""
        if not version:
            return None, Verdict(kind, name, UNVERIFIED, f"no version of {name} could be resolved from the finding or the lockfile")
        status, manifest = self.registry.npm_version(name, version)
        if status != 200:
            return None, Verdict(kind, name, UNVERIFIED, f"npm registry unreachable for {name}@{version}")
        return manifest, None

    def _check_npm_integrity(self, f, name, version, entry, text, packages) -> Verdict:
        manifest, pending = self._npm_pinned(name, version, "integrity")
        if pending:
            return pending
        published = str((manifest.get("dist") or {}).get("integrity") or "").strip()
        if not published:
            return Verdict("integrity", name, UNVERIFIED, f"npm publishes no integrity for {name}@{version}")
        recorded = entry.integrity if entry and entry.integrity else None
        compare = [c for c in (recorded, *_SRI_RE.findall(text)) if c]
        if not compare:
            return Verdict("integrity", name, UNVERIFIED, f"no integrity string to compare for {name}@{version}")
        if any(c == published for c in compare):
            return Verdict("integrity", name, CONTRADICTED, f"the integrity of {name}@{version} matches npm's dist.integrity byte for byte")
        if recorded:
            return Verdict("integrity", name, CONFIRMED, f"the lockfile integrity for {name}@{version} differs from npm's")
        return Verdict("integrity", name, UNVERIFIED, f"the quoted hash is not the one npm publishes for {name}@{version} and no lockfile entry was readable")

    def _check_npm_license(self, f, name, version, entry, text, packages) -> Verdict:
        manifest, pending = self._npm_pinned(name, version, "license")
        if pending:
            return pending
        published = _license_str(manifest.get("license"))
        recorded = entry.license if entry and entry.license else None
        if recorded is None:
            m = _LICENSE_QUOTE_RE.search(text)
            recorded = m.group(1).strip() if m else None
        if not published or not recorded:
            return Verdict("license", name, UNVERIFIED, f"no recorded license to compare for {name}@{version}")
        if recorded.lower() == published.lower():
            return Verdict("license", name, CONTRADICTED, f"the recorded license {recorded!r} for {name}@{version} is what npm publishes")
        return Verdict("license", name, CONFIRMED, f"npm publishes license {published!r} for {name}@{version}; the lockfile records {recorded!r}")

    def _check_npm_downloads(self, f, name, version, entry, text, packages) -> Verdict:
        status, count = self.registry.npm_downloads(name)
        if status != 200 or count is None:
            return Verdict("downloads", name, UNVERIFIED, f"npm download counts unreachable for {name}")
        if count >= _LOW_DOWNLOADS_PER_MONTH:
            return Verdict("downloads", name, CONTRADICTED, f"{name} had {count:,} downloads last month")
        return Verdict("downloads", name, CONFIRMED, f"{name} had {count:,} downloads last month")

    def _check_npm_engines(self, f, name, version, entry, text, packages) -> Verdict:
        manifest, pending = self._npm_pinned(name, version, "engines")
        if pending:
            return pending
        engines = manifest.get("engines")
        node_range = engines.get("node") if isinstance(engines, dict) else None
        if not isinstance(node_range, str) or not node_range.strip():
            return Verdict("engines", name, CONTRADICTED, f"{name}@{version} declares no engines.node on npm")
        runtimes = self._node_runtimes_for(f.path)
        if not runtimes:
            return Verdict("engines", name, UNVERIFIED, f"{name}@{version} needs node {node_range}; this repo declares no Node runtime")
        results = {rt: satisfies(rt, node_range) for rt in runtimes}
        if any(r is None for r in results.values()):
            return Verdict("engines", name, UNVERIFIED, f"could not evaluate engines.node {node_range!r} of {name}@{version}")
        failing = [rt for rt, ok in results.items() if ok is False]
        if failing:
            return Verdict("engines", name, CONFIRMED, f"{name}@{version} needs node {node_range}; the repo declares Node {', '.join(failing)}")
        return Verdict("engines", name, CONTRADICTED,
                       f"{name}@{version} needs node {node_range}; the repo's declared runtime (Node {', '.join(runtimes)}) satisfies it")

    def _check_npm_peer(self, f, name, version, entry, text, packages) -> Verdict:
        manifest, pending = self._npm_pinned(name, version, "peer")
        if pending:
            return pending
        peers = manifest.get("peerDependencies") if isinstance(manifest.get("peerDependencies"), dict) else {}
        meta = manifest.get("peerDependenciesMeta") if isinstance(manifest.get("peerDependenciesMeta"), dict) else {}
        named = [p.name for p in packages if p.name.lower() != name.lower()]
        if not peers:
            return Verdict("peer", name, CONTRADICTED, f"{name}@{version} declares no peer dependencies on npm")
        named_peers = [p for p in named if p in peers]
        if named and not named_peers:
            return Verdict("peer", name, CONTRADICTED, f"{name}@{version} declares no peer dependency on {', '.join(named)}")
        entries = self.manifests.lock_entries(f.path, "npm")
        specs = self.manifests.specs(f.path)
        violated: list[str] = []
        unknown: list[str] = []
        for peer in (named_peers or list(peers)):
            opt = meta.get(peer)
            if isinstance(opt, dict) and opt.get("optional"):
                continue                                              # an optional peer is never a violation
            rng = peers.get(peer)
            installed = entries.get(peer.lower())
            installed_v = installed.version if installed else None
            if installed_v is None and peer in specs:
                installed_v = _min_version_of_spec(specs[peer])
            if installed_v is None:
                unknown.append(peer)
                continue
            ok = satisfies(installed_v, rng) if isinstance(rng, str) else None
            if ok is None:
                unknown.append(peer)
            elif not ok:
                violated.append(f"{peer}@{installed_v} vs {rng}")
        if violated:
            return Verdict("peer", name, CONFIRMED, f"{name}@{version} peer not satisfied: {'; '.join(violated)}")
        if unknown:
            return Verdict("peer", name, UNVERIFIED, f"could not resolve the installed version of {', '.join(unknown)}")
        return Verdict("peer", name, CONTRADICTED,
                       f"every required peer of {name}@{version} is satisfied by what the lockfile installs (optional peers excluded)")

    # -- PyPI -------------------------------------------------------------------------------------
    def _pypi_pinned(self, name: str, version: str | None, kind: str) -> tuple[dict | None, Verdict | None]:
        if not version:
            return None, Verdict(kind, name, UNVERIFIED, f"no version of {name} could be resolved from the finding or the lockfile")
        status, body = self.registry.pypi_version(name, version)
        if status != 200:
            return None, Verdict(kind, name, UNVERIFIED, f"PyPI unreachable for {name}=={version}")
        return body, None

    def _check_pypi_version_exists(self, f, name, version, entry, text, packages) -> Verdict:
        if not version:
            return Verdict("version_exists", name, UNVERIFIED, f"no version named for {name}")
        status, _ = self.registry.pypi_version(name, version)
        if status == 200:
            return Verdict("version_exists", name, CONTRADICTED, f"{name}=={version} is published on PyPI")
        if status == 404:
            return Verdict("version_exists", name, CONFIRMED, f"PyPI has no {name}=={version}")
        return Verdict("version_exists", name, UNVERIFIED, f"PyPI unreachable for {name}=={version}")

    def _check_pypi_integrity(self, f, name, version, entry, text, packages) -> Verdict:
        body, pending = self._pypi_pinned(name, version, "integrity")
        if pending:
            return pending
        published = {str((u.get("digests") or {}).get("sha256") or "").lower()
                     for u in (body.get("urls") or []) if isinstance(u, dict)}
        published.discard("")
        recorded = set(entry.hashes) if entry else set()
        compare = recorded | set(_HEX256_RE.findall(text.lower()))
        if not published or not compare:
            return Verdict("integrity", name, UNVERIFIED, f"no sha256 to compare for {name}=={version}")
        if compare & published:
            return Verdict("integrity", name, CONTRADICTED, f"the recorded sha256 for {name}=={version} is one PyPI publishes")
        if recorded:
            return Verdict("integrity", name, CONFIRMED, f"no recorded sha256 for {name}=={version} matches PyPI")
        return Verdict("integrity", name, UNVERIFIED, f"the quoted sha256 is not one PyPI publishes for {name}=={version} and no lockfile entry was readable")

    def _check_pypi_license(self, f, name, version, entry, text, packages) -> Verdict:
        body, pending = self._pypi_pinned(name, version, "license")
        if pending:
            return pending
        info = body.get("info") if isinstance(body.get("info"), dict) else {}
        published = _license_str(info.get("license_expression")) or _license_str(info.get("license"))
        m = _LICENSE_QUOTE_RE.search(text)
        recorded = m.group(1).strip() if m else None
        if not published or not recorded:
            return Verdict("license", name, UNVERIFIED, f"no recorded license to compare for {name}=={version}")
        if recorded.lower() == published.lower():
            return Verdict("license", name, CONTRADICTED, f"the license {recorded!r} for {name}=={version} is what PyPI publishes")
        return Verdict("license", name, CONFIRMED, f"PyPI publishes license {published!r} for {name}=={version}, not {recorded!r}")

    def _check_pypi_engines(self, f, name, version, entry, text, packages) -> Verdict:
        body, pending = self._pypi_pinned(name, version, "engines")
        if pending:
            return pending
        info = body.get("info") if isinstance(body.get("info"), dict) else {}
        rng = info.get("requires_python")
        if not isinstance(rng, str) or not rng.strip():
            return Verdict("engines", name, CONTRADICTED, f"{name}=={version} declares no requires_python")
        runtimes = self._py_runtimes_for(f.path)
        if not runtimes:
            return Verdict("engines", name, UNVERIFIED, f"{name}=={version} needs python {rng}; this repo declares no Python runtime")
        results = {rt: satisfies(rt, rng) for rt in runtimes}
        if any(r is None for r in results.values()):
            return Verdict("engines", name, UNVERIFIED, f"could not evaluate requires_python {rng!r}")
        failing = [rt for rt, ok in results.items() if ok is False]
        if failing:
            return Verdict("engines", name, CONFIRMED, f"{name}=={version} needs python {rng}; the repo declares Python {', '.join(failing)}")
        return Verdict("engines", name, CONTRADICTED, f"{name}=={version} needs python {rng}; the repo's declared Python ({', '.join(runtimes)}) satisfies it")

    # -- runtimes ---------------------------------------------------------------------------------
    def _node_runtimes_for(self, path: str) -> list[str]:
        key = str(Path(path).parent)
        if key not in self._node_runtimes:
            self._node_runtimes[key] = declared_node_runtimes(self.repo_dir, near=path)
        return self._node_runtimes[key]

    def _py_runtimes_for(self, path: str) -> list[str]:
        key = str(Path(path).parent)
        if key not in self._py_runtimes:
            self._py_runtimes[key] = declared_python_runtimes(self.repo_dir, near=path)
        return self._py_runtimes[key]


_UNVERIFIED_PREFIX = "Unverified: "
_TITLE_MAX = 80


def _unverified(f: CodnaReviewFinding, verdicts: list[Verdict]) -> CodnaReviewFinding:
    """LOW, with "Unverified" in the title and the reason first in the explanation. The fingerprint is
    kept, so a re-run of the same finding still dedups against the thread it opened."""
    reasons = "; ".join(dict.fromkeys(v.note for v in verdicts if v.outcome == UNVERIFIED)) or "registry unreachable"
    title = f.title if f.title.startswith(_UNVERIFIED_PREFIX) else (_UNVERIFIED_PREFIX + f.title)[:_TITLE_MAX]
    lead = (f"Unverified: codna could not check this claim against the package registry ({reasons}). "
            "Treat it as a question to confirm, not as a fact. ")
    return replace(f, severity="low", title=title, explanation=lead + f.explanation)


def _confirmed(f: CodnaReviewFinding, verdicts: list[Verdict]) -> CodnaReviewFinding:
    notes = "; ".join(dict.fromkeys(v.note for v in verdicts if v.outcome == CONFIRMED))
    return replace(f, explanation=f"{f.explanation.rstrip()}\n\nRegistry check: confirmed ({notes}).")


def _log_event(event: str, fields: dict) -> None:
    """One JSON line on stderr (the worker's log shape) for a claim the registry refuted."""
    payload = {"service": "codna-review", "level": "info", "event": event, "ts": round(time.time(), 3)}
    payload.update(fields)
    try:
        print(json.dumps(payload, ensure_ascii=False, default=str), file=sys.stderr)
    except Exception:  # noqa: BLE001 -- logging must never fail a review
        pass


def grounder_for(repo_dir: str, changed_files: list[str], *, registry: Registry | None = None) -> Grounder | None:
    """A grounder for this review, or None when the change touches no dependency manifest or lockfile
    (the fast path: no file is read, no request is made, findings pass through untouched)."""
    if not any(ecosystem_of(p) for p in changed_files or []):
        return None
    return Grounder(repo_dir, changed_files, registry=registry)
