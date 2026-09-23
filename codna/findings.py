"""Data models for codna's security-proof pipeline.

`NormalizedFinding` is the canonical, scanner-agnostic representation produced by
`codna.sarif.ingest_sarif`. Everything downstream (reachability proof, remediation,
the PR-open gate) keys off `canonical_id` — a deterministic content hash over a
finding's SEMANTIC identity (rule + sink + dataflow paths), NEVER over volatile fields
(result/run order, message text, tool fingerprints, column offsets). See the A–Z test
plan (SAR-*, RPRV-*) for the contract these models must satisfy.

Pure stdlib, no engine/http/llm imports — this module is the bottom of the dependency
graph and stays trivially testable offline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def stable_json(obj: Any) -> str:
    """The ONE canonical serialization used for every digest/id: sorted keys, compact,
    UTF-8 preserved. Same input -> byte-identical output, so runs are reproducible."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(obj: Any) -> str:
    """sha256: over the canonical JSON of `obj`."""
    return "sha256:" + sha256_hex(stable_json(obj).encode("utf-8"))


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


_SEV_ORDER = {
    Severity.UNKNOWN: -1,
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def severity_at_least(a: Severity, b: Severity) -> bool:
    """True if band `a` is >= band `b`. UNKNOWN is below everything (never gates on
    severity alone — it must be surfaced, not silently treated as low)."""
    return _SEV_ORDER[a] >= _SEV_ORDER[b]


class FindingKind(str, Enum):
    TAINT = "taint"
    SCA = "sca"
    CONTROL_FLOW = "control_flow"
    CONFIG = "config"
    SECRET = "secret"
    UNSUPPORTED = "unsupported"


class Classification(str, Enum):
    """Pre-patch reachability verdict for the ORIGINAL finding. `EXPLOITABLE` requires a
    validated source→sink dataflow; `UNREACHABLE` requires a *complete* analysis envelope —
    "no path found" in an incomplete envelope is `UNKNOWN`, never `UNREACHABLE`."""

    EXPLOITABLE = "exploitable"
    PRODUCTION_REACHABLE = "production-reachable"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class ClosureStatus(str, Enum):
    """Post-patch verdict — a SEPARATE judgment from reachability. `CLOSED` means the
    original semantic proof obligation is no longer violated (a sanitizer/barrier can close
    it even while the operation stays production-reachable) and no alternate path was found."""

    CLOSED = "closed"
    OPEN = "open"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Location:
    """A physical/logical code location. `start_column` is deliberately EXCLUDED from
    identity (`key()`) — it is too volatile across scanner versions."""

    uri: str
    start_line: int | None = None
    start_column: int | None = None
    logical: str | None = None  # logicalLocation fullyQualifiedName / symbol

    def key(self) -> dict:
        return {"uri": self.uri, "line": self.start_line, "logical": self.logical}

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "logical": self.logical,
        }


def compute_canonical_id(
    rule_id: str,
    kind: "FindingKind | str",
    primary: Location | None,
    code_flows: list[list[Location]],
) -> str:
    """Deterministic semantic identity. Two byte-identical findings collapse to the same
    id (dedupe); two findings sharing a sink but with DISTINCT dataflows get DIFFERENT
    ids (so distinct flows are never merged away — SAR-21/SAR-22). Volatile fields are
    excluded so the id is stable under result/run reordering (SAR-25/SAR-26)."""
    flows = sorted(
        ([loc.key() for loc in flow] for flow in code_flows),
        key=stable_json,
    )
    ident = {
        "rule_id": rule_id,
        "kind": kind.value if isinstance(kind, FindingKind) else kind,
        "primary": primary.key() if primary else None,
        "flows": flows,
    }
    return digest_of(ident)


@dataclass
class NormalizedFinding:
    rule_id: str
    message: str
    scanner: str
    finding_kind: FindingKind
    severity: Severity
    original_severity: str | None
    severity_rationale: str
    primary_location: Location | None
    locations: list[Location] = field(default_factory=list)
    related_locations: list[Location] = field(default_factory=list)
    code_flows: list[list[Location]] = field(default_factory=list)
    fingerprints: dict = field(default_factory=dict)
    partial_fingerprints: dict = field(default_factory=dict)
    suppressions: list = field(default_factory=list)
    baseline_state: str | None = None
    logical_locations: list = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    run_index: int = 0
    occurrence_count: int = 1
    quarantined: bool = False
    quarantine_reason: str | None = None
    canonical_id: str = ""

    @classmethod
    def minimal(cls, *, canonical_id: str, rule_id: str, finding_kind: "FindingKind | str") -> "NormalizedFinding":
        """Reconstruct a lightweight finding from an evidence descriptor (the writer only needs
        identity + rule + kind to name the branch/PR; the full finding lives in the attestation)."""
        kind = finding_kind if isinstance(finding_kind, FindingKind) else FindingKind(str(finding_kind))
        f = cls(
            rule_id=rule_id, message="", scanner="", finding_kind=kind,
            severity=Severity.UNKNOWN, original_severity=None, severity_rationale="",
            primary_location=None,
        )
        f.canonical_id = canonical_id
        return f

    @property
    def is_actively_suppressed(self) -> bool:
        """SARIF suppression is "active" unless explicitly rejected. An unsuppressed
        occurrence must never be shadowed by a suppressed duplicate (SAR-23/24)."""
        if not self.suppressions:
            return False
        for s in self.suppressions:
            status = s.get("status") if isinstance(s, dict) else None
            if status in (None, "accepted"):
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "rule_id": self.rule_id,
            "message": self.message,
            "scanner": self.scanner,
            "finding_kind": self.finding_kind.value,
            "severity": self.severity.value,
            "original_severity": self.original_severity,
            "severity_rationale": self.severity_rationale,
            "primary_location": self.primary_location.to_dict() if self.primary_location else None,
            "locations": [loc.to_dict() for loc in self.locations],
            "related_locations": [loc.to_dict() for loc in self.related_locations],
            "code_flows": [[loc.to_dict() for loc in flow] for flow in self.code_flows],
            "fingerprints": self.fingerprints,
            "partial_fingerprints": self.partial_fingerprints,
            "suppressions": self.suppressions,
            "baseline_state": self.baseline_state,
            "logical_locations": self.logical_locations,
            "properties": self.properties,
            "run_index": self.run_index,
            "occurrence_count": self.occurrence_count,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
        }


@dataclass
class ScannerIdentity:
    name: str  # canonical: codeql|semgrep|snyk|trivy|opengrep|gitleaks|osv-scanner|...
    raw_name: str
    version: str | None
    guid: str | None
    rules_count: int
    rules_digest: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_name": self.raw_name,
            "version": self.version,
            "guid": self.guid,
            "rules_count": self.rules_count,
            "rules_digest": self.rules_digest,
        }


@dataclass
class IngestResult:
    schema_version: str
    sarif_digest: str
    scanner: ScannerIdentity
    findings: list[NormalizedFinding]
    provenance_valid: bool
    provenance_errors: list[str]
    commit: str | None
    tree_digest: str | None
    repo_uri: str | None
    scanner_self_report: dict
    # Scanner-reported success (exitCode/executionSuccessful) is NEVER authoritative —
    # codna reruns the scanner itself under a pinned manifest (G2/G6). Always False.
    scanner_self_report_trusted: bool
    quarantined_artifacts: list[dict]
    raw_result_count: int

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "sarif_digest": self.sarif_digest,
            "scanner": self.scanner.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "provenance_valid": self.provenance_valid,
            "provenance_errors": list(self.provenance_errors),
            "commit": self.commit,
            "tree_digest": self.tree_digest,
            "repo_uri": self.repo_uri,
            "scanner_self_report": self.scanner_self_report,
            "scanner_self_report_trusted": self.scanner_self_report_trusted,
            "quarantined_artifacts": self.quarantined_artifacts,
            "raw_result_count": self.raw_result_count,
        }

    def to_canonical_json(self) -> str:
        """Byte-reproducible serialization — no clock/uuid/random leak (SAR-35,
        INFRA-DETERMINISM)."""
        return stable_json(self.to_dict())
