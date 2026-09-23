"""`run_secure` — the orchestration brain of codna's security-proof autofix.

It sequences the proof loop and the PR-open gate by DELEGATING to three injected
privilege-separated collaborators; it contains no proof, sandbox, or GitHub logic itself
(those live in the engine, the worker, and the writer respectively). Keeping orchestration
free of execution means CLI/MCP/Action can all call `run_secure` as thin adapters and the
whole flow is testable offline with stubs.

Privilege domains (see the plan):
  * `SecurityEngine` — independent reachability + closure proofs (the authority).
  * `Worker` — runs UNTRUSTED code (baseline scan, patch-gen, build/tests, scanner rerun)
    in a sandbox with no write token; emits a signed attestation.
  * `Writer` — cannot execute repo code; verifies the attestation and opens the draft PR
    with a narrowly-scoped token.

Per the spec: ineligible findings are never remediated, and there is exactly ONE verified
finding per PR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .evidence import Attestation
from .findings import Classification, ClosureStatus, IngestResult, NormalizedFinding
from .gate import GateDecision, GateInput, evaluate_gate
from .policy import Policy


# --------------------------------------------------------------------- verdicts

@dataclass
class ReachVerdict:
    classification: Classification
    envelope_complete: bool = True
    proof_type: str | None = None


@dataclass
class Patch:
    patch_digest: str
    integrity_ok: bool = True
    integrity_reason: str = "ok"
    diff: str = ""  # the unified diff; the writer re-verifies digest_of(diff) == patch_digest


@dataclass
class VerifyResult:
    build_passed: bool
    tests_passed: bool
    mojo_ok: bool
    scanner_confirms_gone: bool
    scan_not_degraded: bool


@dataclass
class ClosureVerdict:
    closure_status: ClosureStatus
    alternate_path_found: bool = False
    new_blocking_findings: list = field(default_factory=list)


# --------------------------------------------------------------------- seams

class SecurityEngine(Protocol):
    def analyze(self, ingest: IngestResult, finding: NormalizedFinding) -> ReachVerdict: ...
    def reprove_closure(self, finding: NormalizedFinding, patch: Patch) -> ClosureVerdict: ...


class Worker(Protocol):
    def reproduce_baseline(self, finding: NormalizedFinding) -> bool: ...
    def remediate(self, finding: NormalizedFinding) -> Patch: ...
    def verify(self, finding: NormalizedFinding, patch: Patch) -> VerifyResult: ...
    def attest(self, finding: NormalizedFinding, patch: Patch, decision_inputs: dict) -> Attestation: ...


class Writer(Protocol):
    def base_unchanged(self, ingest: IngestResult) -> bool: ...
    def open_draft_pr(self, finding: NormalizedFinding, patch: Patch, attestation: Attestation) -> str | None: ...


# --------------------------------------------------------------------- report

@dataclass
class FindingDecision:
    canonical_id: str
    classification: Classification
    gate: GateDecision | None
    opened: bool
    pr_url: str | None
    skipped_reason: str | None


@dataclass
class EvidenceBundle:
    finding: NormalizedFinding
    patch: Patch
    attestation: Attestation


@dataclass
class SecureReport:
    findings_total: int
    analyzed: int
    eligible: int
    decisions: list[FindingDecision]
    evidence_bundles: list[EvidenceBundle] = field(default_factory=list)

    @property
    def opened_prs(self) -> list[str]:
        return [d.pr_url for d in self.decisions if d.opened and d.pr_url]


def run_secure(
    ingest: IngestResult,
    *,
    engine: SecurityEngine,
    worker: Worker,
    writer: Writer,
    policy: Policy,
    open_pr: bool = False,
) -> SecureReport:
    """Drive the full loop for every (non-quarantined) finding. Returns a `SecureReport`;
    opens at most one draft PR per finding, and only when every gate condition holds."""
    decisions: list[FindingDecision] = []
    evidence_bundles: list[EvidenceBundle] = []
    analyzed = eligible = 0

    for finding in ingest.findings:
        if finding.quarantined:
            decisions.append(_skip(finding, Classification.UNKNOWN, "artifact quarantined (path escape)"))
            continue

        analyzed += 1
        verdict = engine.analyze(ingest, finding)

        is_eligible, elig_reason = policy.is_autofix_eligible(verdict.classification)
        if not is_eligible:
            # Ineligible findings are reported but NEVER remediated (no patch generated).
            decisions.append(_skip(finding, verdict.classification, f"not autofix-eligible: {elig_reason}"))
            continue
        eligible += 1

        baseline_ok = worker.reproduce_baseline(finding)
        patch = worker.remediate(finding)
        verify = worker.verify(finding, patch)
        closure = engine.reprove_closure(finding, patch)

        gate_input = GateInput(
            provenance_valid=ingest.provenance_valid,
            baseline_reproduced=baseline_ok,
            classification=verdict.classification,
            patch_integrity_ok=patch.integrity_ok,
            build_passed=verify.build_passed,
            tests_passed=verify.tests_passed,
            mojo_ok=verify.mojo_ok,
            scanner_confirms_gone=verify.scanner_confirms_gone,
            scan_not_degraded=verify.scan_not_degraded,
            closure_status=closure.closure_status,
            alternate_path_found=closure.alternate_path_found,
            new_blocking_findings=closure.new_blocking_findings,
            attestation_ok=False,  # set below once attested + verified
            base_unchanged=False,
        )

        # Attestation + base revalidation are only meaningful if everything else holds and
        # we actually intend to open a PR; compute them last (steps 20–22).
        attestation = worker.attest(finding, patch, {"classification": verdict.classification.value})
        base_unchanged = writer.base_unchanged(ingest)
        gate_input.attestation_ok = attestation is not None
        gate_input.base_unchanged = base_unchanged

        decision = evaluate_gate(gate_input, policy)
        if decision.should_open_pr:
            evidence_bundles.append(EvidenceBundle(finding=finding, patch=patch, attestation=attestation))
        opened = False
        pr_url = None
        if decision.should_open_pr and open_pr:
            # The writer independently re-verifies (attestation, digest, base unchanged); any
            # failure there means no PR — never propagate a writer error into the report.
            try:
                pr_url = writer.open_draft_pr(finding, patch, attestation)
            except Exception:  # noqa: BLE001 - writer abort -> simply no PR
                pr_url = None
            opened = pr_url is not None

        decisions.append(
            FindingDecision(
                canonical_id=finding.canonical_id,
                classification=verdict.classification,
                gate=decision,
                opened=opened,
                pr_url=pr_url,
                skipped_reason=None,
            )
        )

    return SecureReport(
        findings_total=len(ingest.findings),
        analyzed=analyzed,
        eligible=eligible,
        decisions=decisions,
        evidence_bundles=evidence_bundles,
    )


# --------------------------------------------------------------------- Tier-1 report

@dataclass
class ClassifyRow:
    canonical_id: str
    rule_id: str
    finding_kind: str
    classification: str
    eligible: bool
    reason: str


@dataclass
class Tier1Report:
    """Read-only, ZERO-LLM-token reachability report — the fast wedge. Says which findings
    are exploitable/production-reachable and which would be autofix-eligible, WITHOUT
    remediating anything (no patch-gen, build, or scanner rerun = no Tier-2 work)."""

    rows: list[ClassifyRow]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.classification] = out.get(r.classification, 0) + 1
        return out

    @property
    def eligible(self) -> int:
        return sum(1 for r in self.rows if r.eligible)


def classify_only(ingest: IngestResult, *, engine: "SecurityEngine", policy: Policy) -> Tier1Report:
    """Tier-1: classify every finding's reachability and report eligibility. Uses the
    engine's batch analysis (one request per snapshot) and spends no LLM tokens."""
    rows: list[ClassifyRow] = []
    for f in ingest.findings:
        if f.quarantined:
            rows.append(ClassifyRow(f.canonical_id, f.rule_id, f.finding_kind.value,
                                    "quarantined", False, "artifact path escape"))
            continue
        verdict = engine.analyze(ingest, f)
        eligible, reason = policy.is_autofix_eligible(verdict.classification)
        rows.append(ClassifyRow(f.canonical_id, f.rule_id, f.finding_kind.value,
                                verdict.classification.value, eligible, reason))
    return Tier1Report(rows)


def _skip(finding: NormalizedFinding, classification: Classification, reason: str) -> FindingDecision:
    return FindingDecision(
        canonical_id=finding.canonical_id,
        classification=classification,
        gate=None,
        opened=False,
        pr_url=None,
        skipped_reason=reason,
    )
