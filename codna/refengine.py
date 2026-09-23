"""LocalReferenceEngine — a bounded, self-hostable implementation of the SecurityEngine seam.

This is the in-repo reference engine for offline / air-gapped runs. It is deliberately
CONSERVATIVE and HONEST: it does NOT perform independent interprocedural taint analysis, so it
NEVER returns `exploitable` — the strongest verdict it will assert is `production-reachable`
(the vulnerable operation is callable in production; taint is not independently proven). The
full independent taint engine that earns `exploitable` lives in `decision-engine`; this
reference exists so `codna secure` runs end to end without a remote engine and so the
four-outcome discipline is exercised faithfully:

    validated source->sink (independent)  -> exploitable      (ONLY the real engine; never here)
    production entrypoint reaches op       -> production-reachable
    no path + COMPLETE sound envelope      -> unreachable
    no path + INCOMPLETE/UNSUPPORTED       -> unknown          (never silently 'unreachable')

The support matrix is keyed by (language, framework, build_mode) — "java" is not "java+spring"
(reflection makes the envelope incomplete). Scanner code-flows are recorded as CORROBORATION,
never trusted as proof. Closure is a bounded diff heuristic (recognized sanitizer at the sink).
"""
from __future__ import annotations

import posixpath
import re
from enum import Enum
from typing import Callable

from .evasion import added_lines, changed_paths
from .findings import Classification, ClosureStatus, IngestResult, NormalizedFinding
from .secure import ClosureVerdict, ReachVerdict


class Envelope(Enum):
    COMPLETE = "complete"   # sound enough to assert 'unreachable'
    BOUNDED = "bounded"     # production-callability only; cannot prove absence
    UNSUPPORTED = "unsupported"


_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".go": "go", ".java": "java", ".rb": "ruby", ".php": "php",
}

# (language, framework, build_mode) -> envelope completeness. Lookup falls back from most to
# least specific. Reflection-heavy stacks are intentionally BOUNDED.
_SUPPORT: dict[tuple, Envelope] = {
    ("python", None, None): Envelope.COMPLETE,
    ("javascript", None, None): Envelope.COMPLETE,
    ("typescript", None, None): Envelope.COMPLETE,
    ("go", None, None): Envelope.COMPLETE,
    ("java", None, None): Envelope.COMPLETE,
    ("java", "spring", None): Envelope.BOUNDED,
    ("ruby", None, None): Envelope.BOUNDED,
    ("php", None, None): Envelope.BOUNDED,
}

_TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)(/|$)|(_test\.|\.test\.|_spec\.)", re.IGNORECASE)
_SANITIZER = re.compile(
    r"(parameteri|saniti[sz]e|escape|shlex\.quote|bleach\.clean|html\.escape|quote\(|"
    r"prepared\s*statement|placeholder|bindparam"
    # any `.execute(<sql>, (params))` / `(<sql>, [params])` — a bound-parameter query
    # (greedy across the SQL literal so commas inside it don't break the match)
    r"|\.execute\(.+,\s*[(\[])",
    re.IGNORECASE,
)


def _lang(path: str | None) -> str | None:
    if not path:
        return None
    return _EXT_LANG.get(posixpath.splitext(path)[1].lower())


def _envelope(lang: str | None, framework: str | None, build_mode: str | None) -> Envelope:
    if lang is None:
        return Envelope.UNSUPPORTED
    for key in ((lang, framework, build_mode), (lang, framework, None), (lang, None, None)):
        if key in _SUPPORT:
            return _SUPPORT[key]
    return Envelope.UNSUPPORTED


class LocalReferenceEngine:
    """SecurityEngine implementation (analyze / reprove_closure). `framework_for` and
    `build_mode_for` let a deployment refine the envelope per path; both default to None."""

    def __init__(
        self,
        *,
        framework_for: Callable[[str], str | None] | None = None,
        build_mode_for: Callable[[str], str | None] | None = None,
    ):
        self._framework_for = framework_for or (lambda path: None)
        self._build_mode_for = build_mode_for or (lambda path: None)

    def analyze(self, ingest: IngestResult, finding: NormalizedFinding) -> ReachVerdict:
        path = finding.primary_location.uri if finding.primary_location else None
        lang = _lang(path)
        env = _envelope(lang, self._framework_for(path or ""), self._build_mode_for(path or ""))

        if env is Envelope.UNSUPPORTED:
            return ReachVerdict(Classification.UNKNOWN, envelope_complete=False,
                                proof_type=f"off-matrix ({lang or 'unknown-language'})")

        if path and _TEST_PATH.search(path):
            if env is Envelope.COMPLETE:
                return ReachVerdict(Classification.UNREACHABLE, envelope_complete=True,
                                    proof_type="test-only sink; no production entrypoint (complete envelope)")
            return ReachVerdict(Classification.UNKNOWN, envelope_complete=False,
                                proof_type="test-only but envelope is bounded")

        if env is Envelope.COMPLETE:
            # Honest ceiling: production-reachable. 'exploitable' requires the independent
            # taint engine; scanner code-flows only corroborate.
            corro = "scanner-flow-corroborated" if finding.code_flows else "production-callability"
            return ReachVerdict(Classification.PRODUCTION_REACHABLE, envelope_complete=True, proof_type=corro)

        return ReachVerdict(Classification.UNKNOWN, envelope_complete=False,
                            proof_type="bounded envelope: production-reachability not independently provable")

    def reprove_closure(self, finding: NormalizedFinding, patch) -> ClosureVerdict:
        diff = getattr(patch, "diff", "") or ""
        sink = finding.primary_location
        if not diff.strip() or sink is None:
            return ClosureVerdict(ClosureStatus.UNKNOWN, alternate_path_found=False, new_blocking_findings=[])
        touched = {posixpath.normpath(p) for p in changed_paths(diff)}
        if posixpath.normpath(sink.uri) not in touched:
            # the sink file wasn't modified -> the obligation cannot be closed
            return ClosureVerdict(ClosureStatus.OPEN, alternate_path_found=False, new_blocking_findings=[])
        if any(_SANITIZER.search(line) for line in added_lines(diff)):
            return ClosureVerdict(ClosureStatus.CLOSED, alternate_path_found=False, new_blocking_findings=[])
        # sink touched but no recognized barrier added -> conservatively OPEN (e.g. sink moved)
        return ClosureVerdict(ClosureStatus.OPEN, alternate_path_found=False, new_blocking_findings=[])
