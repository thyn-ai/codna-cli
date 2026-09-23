"""SARIF ingestion — the front door of codna's security-proof pipeline.

`ingest_sarif` turns any scanner's SARIF (CodeQL, Semgrep, Snyk, Trivy, …) into a list of
`NormalizedFinding`s plus a digest-bound `IngestResult`. It is an INGESTION guarantee, not
a proof guarantee: it parses, validates provenance, PRESERVES the full property set,
normalizes severity (recording the original + rationale), deduplicates WITHOUT discarding
distinct dataflows, quarantines path-escaping artifacts, and binds everything to a content
digest. Reachability/exploitability is decided later by the engine.

Pure stdlib, no engine/http/llm/subprocess imports, no network — so the Tier-1 ingest path
stays fast and trivially testable offline. See the A–Z test plan (SAR-*) for the contract.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import unquote

from .findings import (
    FindingKind,
    IngestResult,
    Location,
    NormalizedFinding,
    ScannerIdentity,
    Severity,
    compute_canonical_id,
    digest_of,
)

# --- Resource caps: adversarial SARIF must fail closed, never hang/OOM (SAR-DOS). ---
MAX_BYTES = 64 * 1024 * 1024  # 64 MiB raw input
MAX_RESULTS = 200_000  # across all runs
MAX_JSON_DEPTH = 200  # guards pathological deep nesting before recursion blows up

_SUPPORTED_SCHEMA = re.compile(r"sarif-schema-2\.1\.0|/2\.1\.0/")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SarifError(Exception):
    """Base for every ingestion failure (all fail closed)."""


class SarifParseError(SarifError):
    pass


class SarifSchemaError(SarifError):
    pass


class ProvenanceIncompleteError(SarifError):
    pass


class CommitBindingError(SarifError):
    pass


class BoundedInputError(SarifError):
    pass


_SCANNER_ALIASES = {
    "codeql": "codeql",
    "github codeql": "codeql",
    "semgrep": "semgrep",
    "semgrep oss": "semgrep",
    "semgrep ci": "semgrep",
    "snyk": "snyk",
    "snyk code": "snyk",
    "snykcode": "snyk",
    "snyk open source": "snyk",
    "trivy": "trivy",
    "aqua trivy": "trivy",
    # thyn-ai/security-toolchain drivers, `tool.driver.name` exactly as each binary writes it
    # (lower-cased by `_canon_scanner`): opengrep `--sarif-output` says "Opengrep OSS" (toolchain
    # tests/test_gate.py:23), `gitleaks -f sarif` says "gitleaks" (report/constants.go:4),
    # `osv-scanner --format sarif` says "osv-scanner" (internal/output/sarif.go).
    # Opengrep is a Semgrep fork but a distinct, separately pinned tool: never fold it into semgrep
    # (the cross-run identity check below relies on distinct canonical names).
    "opengrep": "opengrep",
    "opengrep oss": "opengrep",
    "gitleaks": "gitleaks",
    "osv-scanner": "osv-scanner",
}


def _canon_scanner(raw: str) -> str:
    return _SCANNER_ALIASES.get((raw or "").strip().lower(), (raw or "").strip().lower())


# --------------------------------------------------------------------------- parse

def _json_depth_ok(raw: str) -> None:
    """Cheap depth guard so a multi-megabyte string of nested brackets can't recurse the
    parser into a stack overflow before we even see the structure."""
    depth = 0
    for ch in raw:
        if ch in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise BoundedInputError(f"SARIF nesting exceeds {MAX_JSON_DEPTH} levels")
        elif ch in "]}":
            depth -= 1


def _looks_like_path(s: str) -> bool:
    # A SARIF JSON string always contains '{'; a path never does. This keeps a JSON
    # literal from being mistaken for a filename (and vice versa).
    return "{" not in s and "\n" not in s and len(s) < 4096 and os.path.exists(s)


def _load(data: "bytes | str | os.PathLike") -> tuple[dict, str]:
    """Return (parsed_doc, raw_text). Reads bytes ONCE so the digest is bound to exactly
    what was ingested, immune to a later on-disk mutation (SAR-36)."""
    if isinstance(data, os.PathLike) or (isinstance(data, str) and _looks_like_path(data)):
        path = os.fspath(data)
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            raise SarifParseError(f"cannot stat SARIF {path!r}: {exc}") from exc
        if size > MAX_BYTES:
            raise BoundedInputError(f"SARIF {path!r} is {size} bytes (> {MAX_BYTES} cap)")
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
    elif isinstance(data, bytes):
        raw_bytes = data
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    else:  # pragma: no cover - defensive
        raise SarifParseError(f"unsupported input type {type(data)!r}")

    if len(raw_bytes) > MAX_BYTES:
        raise BoundedInputError(f"SARIF input is {len(raw_bytes)} bytes (> {MAX_BYTES} cap)")
    if not raw_bytes.strip():
        raise SarifParseError("SARIF input is empty")
    text = raw_bytes.decode("utf-8", errors="strict")
    _json_depth_ok(text)
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SarifParseError(f"malformed SARIF JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SarifParseError("SARIF root is not a JSON object")
    return doc, text


def _check_schema(doc: dict) -> str:
    schema = doc.get("$schema", "")
    version = doc.get("version", "")
    if version and version != "2.1.0":
        raise SarifSchemaError(f"unsupported SARIF version {version!r} (only 2.1.0); not coerced")
    if not version and not (schema and _SUPPORTED_SCHEMA.search(schema)):
        raise SarifSchemaError(f"unrecognized SARIF $schema {schema!r}; refusing to coerce")
    if schema and not _SUPPORTED_SCHEMA.search(schema) and version != "2.1.0":
        raise SarifSchemaError(f"unrecognized SARIF $schema {schema!r}")
    return version or "2.1.0"


# ----------------------------------------------------------------------- scanner id

def _collect_rules(driver: dict) -> dict[str, dict]:
    rules: dict[str, dict] = {}
    for rule in driver.get("rules", []) or []:
        rid = rule.get("id")
        if rid:
            rules[rid] = rule
    for ext in driver.get("extensions", []) or []:
        for rule in ext.get("rules", []) or []:
            rid = rule.get("id")
            if rid and rid not in rules:
                rules[rid] = rule
    return rules


def _rules_digest(rules: dict[str, dict]) -> str:
    summary = [
        {
            "id": rid,
            "tags": (rules[rid].get("properties", {}) or {}).get("tags", []),
            "security_severity": (rules[rid].get("properties", {}) or {}).get("security-severity"),
            "level": (rules[rid].get("defaultConfiguration", {}) or {}).get("level"),
        }
        for rid in sorted(rules)
    ]
    return digest_of(summary)


def _scanner_identity(run: dict) -> ScannerIdentity:
    driver = (run.get("tool", {}) or {}).get("driver", {}) or {}
    raw_name = driver.get("name", "")
    version = driver.get("version") or driver.get("semanticVersion")
    rules = _collect_rules(driver)
    return ScannerIdentity(
        name=_canon_scanner(raw_name),
        raw_name=raw_name,
        version=version,
        guid=driver.get("guid"),
        rules_count=len(rules),
        rules_digest=_rules_digest(rules),
    )


# --------------------------------------------------------------------- provenance

def _provenance(run: dict) -> tuple[str | None, str | None, str | None]:
    """(commit, tree_digest, repo_uri) from versionControlProvenance."""
    vcp = run.get("versionControlProvenance") or []
    if not vcp:
        return None, None, None
    first = vcp[0] if isinstance(vcp, list) else vcp
    commit = first.get("revisionId")
    repo_uri = first.get("repositoryUri")
    tree_digest = (first.get("properties", {}) or {}).get("tree_digest")
    return commit, tree_digest, repo_uri


# ----------------------------------------------------------------------- severity

_WORD_TO_BAND = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "error": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "moderate": Severity.MEDIUM,
    "warning": Severity.MEDIUM,
    "low": Severity.LOW,
    "info": Severity.INFO,
    "informational": Severity.INFO,
    "note": Severity.INFO,
}


def _band_from_number(x: float) -> Severity:
    if x >= 9.0:
        return Severity.CRITICAL
    if x >= 7.0:
        return Severity.HIGH
    if x >= 4.0:
        return Severity.MEDIUM
    if x > 0:
        return Severity.LOW
    return Severity.INFO


def _scanner_severity(
    result: dict, rule: dict, scanner: str
) -> tuple[Severity, str | None, str]:
    """Derive a band from an explicit SCANNER SEVERITY field. SARIF `level` ALONE is
    insufficient → UNKNOWN with a rationale that says so (SAR-19)."""
    rprops = rule.get("properties", {}) or {}
    res_props = result.get("properties", {}) or {}

    ss = rprops.get("security-severity")
    if ss is not None:
        try:
            return _band_from_number(float(ss)), str(ss), "codeql rule property 'security-severity'"
        except (TypeError, ValueError):
            pass

    for val, src in (
        (res_props.get("severity"), "result property 'severity'"),
        (rprops.get("severity"), "rule property 'severity'"),
        (rprops.get("problem.severity"), "rule property 'problem.severity'"),
        (res_props.get("cvssv3_severity"), "result property 'cvssv3_severity'"),
    ):
        if isinstance(val, str) and val.strip():
            band = _WORD_TO_BAND.get(val.strip().lower(), Severity.UNKNOWN)
            return band, val, f"{scanner} {src}"

    level = result.get("level")
    return (
        Severity.UNKNOWN,
        level,
        "SARIF 'level' alone is insufficient for security severity; "
        "no scanner severity property present",
    )


def _normalize_severity(result, rule, scanner, policy) -> tuple[Severity, str | None, str]:
    band, original, rationale = _scanner_severity(result, rule, scanner)
    rule_id = result.get("ruleId")
    overrides = getattr(policy, "severity_overrides", {}) if policy else {}
    if rule_id in overrides:
        forced = overrides[rule_id]
        forced_band = forced if isinstance(forced, Severity) else Severity(str(forced).lower())
        digest = getattr(policy, "digest", "?")
        # original severity is PRESERVED — overrides are recorded, never silent (SAR-20).
        return (
            forced_band,
            original,
            f"policy override -> {forced_band.value} (scanner said {band.value}); policy_digest {digest}",
        )
    return band, original, rationale


# ----------------------------------------------------------------------- locations

def _is_escape(path: str) -> bool:
    p = unquote(path or "")
    p = p.replace("\\", "/")
    if not p:
        return False
    if "://" in p:  # file:// / http(s):// / any scheme -> outside the tree
        return True
    if re.match(r"^[a-zA-Z]:", p):  # windows drive letter
        return True
    if p.startswith("/"):  # absolute
        return True
    depth = 0
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            depth -= 1
            if depth < 0:
                return True
        else:
            depth += 1
    return False


def _resolve_uri(uri: str | None, base_id: str | None, base_ids: dict) -> tuple[str, bool]:
    """Resolve an artifactLocation against originalUriBaseIds (preserved but NOT trusted)
    and report whether it escapes the repository root."""
    if uri is None:
        return "", False
    base = ""
    if base_id and isinstance(base_ids, dict):
        base = (base_ids.get(base_id, {}) or {}).get("uri", "") or ""
    combined = f"{base.rstrip('/')}/{uri}" if base else uri
    return combined, _is_escape(combined)


def _loc_from_location(loc: dict, base_ids: dict) -> tuple[Location | None, bool]:
    physical = loc.get("physicalLocation", {}) or {}
    art = physical.get("artifactLocation", {}) or {}
    resolved, escape = _resolve_uri(art.get("uri"), art.get("uriBaseId"), base_ids)
    region = physical.get("region", {}) or {}
    logicals = loc.get("logicalLocations", []) or []
    logical = logicals[0].get("fullyQualifiedName") if logicals else None
    return (
        Location(resolved, region.get("startLine"), region.get("startColumn"), logical),
        escape,
    )


def _code_flows(result: dict, base_ids: dict) -> tuple[list[list[Location]], bool]:
    flows: list[list[Location]] = []
    escaped = False
    for cf in result.get("codeFlows", []) or []:
        seq: list[Location] = []
        for tf in cf.get("threadFlows", []) or []:
            for entry in tf.get("locations", []) or []:
                loc, esc = _loc_from_location(entry.get("location", {}) or {}, base_ids)
                escaped = escaped or esc
                if loc:
                    seq.append(loc)
        if seq:
            flows.append(seq)
    return flows, escaped


# ----------------------------------------------------------------------- kind

def _tags(rule: dict, result: dict) -> set[str]:
    tags = set((rule.get("properties", {}) or {}).get("tags", []) or [])
    tags |= set((result.get("properties", {}) or {}).get("tags", []) or [])
    return {t.lower() for t in tags}


def _has_cve(rule_id: str, rule: dict, result: dict) -> bool:
    blob = json.dumps([rule_id, rule.get("properties", {}), result.get("properties", {})]).lower()
    return bool(re.search(r"\bcve-\d", blob) or "ghsa-" in blob or '"package"' in blob or '"affected' in blob)


def _classify_kind(rule_id: str, rule: dict, result: dict, scanner: str) -> FindingKind:
    tags = _tags(rule, result)
    has_flows = bool(result.get("codeFlows"))
    # gitleaks is a secrets-only scanner and its SARIF rules carry no tags: classify by identity.
    if scanner == "gitleaks" or tags & {"secret", "credential", "credentials", "key", "token"}:
        return FindingKind.SECRET
    if _has_cve(rule_id, rule, result) or tags & {"sca", "supply-chain", "dependency", "vulnerability"}:
        return FindingKind.SCA
    if tags & {"misconfiguration", "configuration", "security-config", "iac"}:
        return FindingKind.CONFIG
    if has_flows or tags & {"taint", "injection", "dataflow", "data-flow", "security"}:
        return FindingKind.TAINT
    if tags & {"control-flow", "control_flow", "reachability"}:
        return FindingKind.CONTROL_FLOW
    return FindingKind.UNSUPPORTED


# ----------------------------------------------------------------------- assembly

def _build_finding(result, rule, scanner, run_index, base_ids, policy) -> NormalizedFinding:
    rule_id = result.get("ruleId") or rule.get("id") or "(unknown-rule)"
    msg = (result.get("message", {}) or {}).get("text", "")
    kind = _classify_kind(rule_id, rule, result, scanner)
    band, original, rationale = _normalize_severity(result, rule, scanner, policy)

    locations: list[Location] = []
    quarantined = False
    for loc in result.get("locations", []) or []:
        nloc, esc = _loc_from_location(loc, base_ids)
        quarantined = quarantined or esc
        if nloc:
            locations.append(nloc)
    related = []
    for loc in result.get("relatedLocations", []) or []:
        nloc, esc = _loc_from_location(loc, base_ids)
        quarantined = quarantined or esc
        if nloc:
            related.append(nloc)
    flows, flow_escape = _code_flows(result, base_ids)
    quarantined = quarantined or flow_escape

    primary = locations[0] if locations else None
    finding = NormalizedFinding(
        rule_id=rule_id,
        message=msg,
        scanner=scanner,
        finding_kind=kind,
        severity=band,
        original_severity=original,
        severity_rationale=rationale,
        primary_location=primary,
        locations=locations,
        related_locations=related,
        code_flows=flows,
        fingerprints=result.get("fingerprints", {}) or {},
        partial_fingerprints=result.get("partialFingerprints", {}) or {},
        suppressions=result.get("suppressions", []) or [],
        baseline_state=result.get("baselineState"),
        logical_locations=[
            lg
            for loc in (result.get("locations", []) or [])
            for lg in (loc.get("logicalLocations", []) or [])
        ],
        properties=result.get("properties", {}) or {},
        run_index=run_index,
        quarantined=quarantined,
        quarantine_reason="artifact path escapes repository root" if quarantined else None,
    )
    finding.canonical_id = compute_canonical_id(rule_id, kind, primary, flows)
    return finding


def _dedupe(findings: list[NormalizedFinding]) -> list[NormalizedFinding]:
    """Collapse byte-identical findings (SAR-21) while keeping distinct dataflows apart
    (SAR-22). An unsuppressed occurrence wins over a suppressed duplicate (SAR-23/24)."""
    by_id: dict[str, NormalizedFinding] = {}
    order: list[str] = []
    for f in findings:
        cid = f.canonical_id
        existing = by_id.get(cid)
        if existing is None:
            by_id[cid] = f
            order.append(cid)
            continue
        existing.occurrence_count += 1
        if not f.is_actively_suppressed and existing.is_actively_suppressed:
            existing.suppressions = []  # unsuppressed wins
    return [by_id[c] for c in order]


def ingest_sarif(
    data: "bytes | str | os.PathLike",
    *,
    policy: Any = None,
    expected_commit: str | None = None,
) -> IngestResult:
    """Parse and normalize a SARIF document into an `IngestResult`. Fails closed on bad
    input. `expected_commit` (if given) is checked against the SARIF's revisionId."""
    doc, text = _load(data)
    schema_version = _check_schema(doc)
    sarif_digest = digest_of(doc)  # canonical -> reorder/whitespace-invariant (SAR-27/28)

    runs = doc.get("runs", []) or []
    findings: list[NormalizedFinding] = []
    quarantined_artifacts: list[dict] = []
    provenance_errors: list[str] = []
    raw_result_count = 0

    scanner_identity: ScannerIdentity | None = None
    commit = tree_digest = repo_uri = None
    self_report: dict = {}

    for run_index, run in enumerate(runs):
        ident = _scanner_identity(run)
        if scanner_identity is None:
            scanner_identity = ident
        elif ident.name != scanner_identity.name or ident.version != scanner_identity.version:
            provenance_errors.append(
                f"inconsistent scanner identity across runs: "
                f"{scanner_identity.name}@{scanner_identity.version} vs {ident.name}@{ident.version}"
            )

        rc, td, ru = _provenance(run)
        commit = commit or rc
        tree_digest = tree_digest or td
        repo_uri = repo_uri or ru

        invocations = run.get("invocations", []) or []
        if invocations and not self_report:
            inv = invocations[0]
            self_report = {
                "exitCode": inv.get("exitCode"),
                "executionSuccessful": inv.get("executionSuccessful"),
            }

        base_ids = run.get("originalUriBaseIds", {}) or {}
        rules = _collect_rules((run.get("tool", {}) or {}).get("driver", {}) or {})

        for result in run.get("results", []) or []:
            raw_result_count += 1
            if raw_result_count > MAX_RESULTS:
                raise BoundedInputError(f"SARIF has > {MAX_RESULTS} results (resource cap)")
            rule_id = result.get("ruleId") or ""
            rule = rules.get(rule_id, {})
            finding = _build_finding(result, rule, ident.name, run_index, base_ids, policy)
            if finding.quarantined:
                quarantined_artifacts.append(
                    {
                        "rule_id": finding.rule_id,
                        "uri": finding.primary_location.uri if finding.primary_location else None,
                        "reason": finding.quarantine_reason,
                    }
                )
            findings.append(finding)

    findings = _dedupe(findings)

    if scanner_identity is None:
        scanner_identity = ScannerIdentity("", "", None, None, 0, digest_of([]))

    if not scanner_identity.raw_name:
        provenance_errors.append("driver name missing")
    if not scanner_identity.version:
        provenance_errors.append("driver version missing")
    if not commit:
        provenance_errors.append("no versionControlProvenance revisionId")
    elif not _HEX40.match(commit):
        provenance_errors.append(f"revisionId {commit!r} is not a full 40-hex commit")
    if expected_commit and commit and commit != expected_commit:
        provenance_errors.append(f"commit mismatch: sarif {commit} != expected {expected_commit}")

    provenance_valid = not provenance_errors

    return IngestResult(
        schema_version=schema_version,
        sarif_digest=sarif_digest,
        scanner=scanner_identity,
        findings=findings,
        provenance_valid=provenance_valid,
        provenance_errors=provenance_errors,
        commit=commit,
        tree_digest=tree_digest,
        repo_uri=repo_uri,
        scanner_self_report=self_report,
        scanner_self_report_trusted=False,  # never authoritative; codna reruns the scanner
        quarantined_artifacts=quarantined_artifacts,
        raw_result_count=raw_result_count,
    )


def validate_commit_binding(ingest: IngestResult, expected: str | None = None) -> str:
    """Strict commit binding for the gate (G1). Raises unless the SARIF carries a full
    40-hex revisionId (rejecting abbreviated/symbolic refs) that matches `expected`."""
    if not ingest.commit:
        raise CommitBindingError("no versionControlProvenance / revisionId in SARIF")
    if not _HEX40.match(ingest.commit):
        raise CommitBindingError(
            f"revisionId {ingest.commit!r} is not a full 40-hex commit (abbreviated/symbolic refs rejected)"
        )
    if expected and ingest.commit != expected:
        raise CommitBindingError(f"commit mismatch: sarif {ingest.commit} != expected {expected}")
    return ingest.commit


def build_security_analysis_request(
    ingest: IngestResult,
    *,
    snapshot_id: str,
    policy_digest: str,
    model_pack_digest: str,
    configuration_digest: str | None = None,
) -> dict:
    """Build the exact batch payload for the engine's `security-analyses` endpoint.
    Refuses to build when provenance is incomplete (G1). Quarantined (escaping) findings
    are NOT sent for analysis."""
    if not ingest.provenance_valid:
        raise ProvenanceIncompleteError(
            "cannot build security-analysis request: " + "; ".join(ingest.provenance_errors)
        )
    return {
        "snapshot_id": snapshot_id,
        "sarif_digest": ingest.sarif_digest,
        "scanner": {
            "name": ingest.scanner.name,
            "version": ingest.scanner.version,
            "rules_digest": ingest.scanner.rules_digest,
            "configuration_digest": configuration_digest,
        },
        "policy_digest": policy_digest,
        "model_pack_digest": model_pack_digest,
        "findings": [
            {
                "canonical_id": f.canonical_id,
                "rule_id": f.rule_id,
                "finding_kind": f.finding_kind.value,
                "severity": f.severity.value,
                "primary_location": f.primary_location.to_dict() if f.primary_location else None,
                "code_flows": [[loc.to_dict() for loc in flow] for flow in f.code_flows],
            }
            for f in ingest.findings
            if not f.quarantined
        ],
    }
