"""The PR-open gate — the nine conditions that must ALL hold before codna opens a draft PR.

This is the decision core: pure logic that composes already-computed verdicts (provenance,
baseline reproduction, reachability classification, patch integrity, regression safety,
scanner confirmation, closure, alternate/new findings, attestation) into one auditable
`GateDecision`. A PR opens only on a logical-AND of every condition; a single failure blocks
it and the failing condition is named (BRCP-38/39).

Note the separation the spec insists on: G6 (scanner confirmation, non-degraded) is about
the scan; G7 (closure) requires `closure_status == CLOSED` and is satisfied by a sanitizer
even when the operation stays production-reachable — closure is NOT "all reachability gone".

Pure stdlib, no engine/http/subprocess — the gate is decided from inputs, never by executing
anything. Producing those inputs is the worker's job (a different privilege domain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .findings import Classification, ClosureStatus
from .policy import Policy

GATE_ORDER = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")


@dataclass
class GateInput:
    # G1 immutable provenance (all digests bound, commit pinned)
    provenance_valid: bool
    # G2 baseline reproduction on the exact unpatched snapshot
    baseline_reproduced: bool
    # G3 policy-eligible proof
    classification: Classification
    # G4 patch integrity (no verification-evasion, within change scope)
    patch_integrity_ok: bool
    # G5 regression safety
    build_passed: bool
    tests_passed: bool
    mojo_ok: bool
    # G6 scanner confirmation under an identical, non-degraded config
    scanner_confirms_gone: bool
    scan_not_degraded: bool
    # G7 independent closure proof
    closure_status: ClosureStatus
    # G8 no alternate / newly-introduced vulnerability
    alternate_path_found: bool
    new_blocking_findings: list = field(default_factory=list)
    # G9 attested handoff (signature/digests verified) + base unchanged at open time
    attestation_ok: bool = False
    base_unchanged: bool = False


@dataclass
class ConditionResult:
    gate: str
    passed: bool
    reason: str


@dataclass
class GateDecision:
    conditions: list[ConditionResult]
    should_open_pr: bool

    @property
    def failed(self) -> list[str]:
        return [c.gate for c in self.conditions if not c.passed]

    def reason_for(self, gate: str) -> str:
        for c in self.conditions:
            if c.gate == gate:
                return c.reason
        raise KeyError(gate)


def _cond(gate: str, passed: bool, ok_reason: str, fail_reason: str) -> ConditionResult:
    return ConditionResult(gate, passed, ok_reason if passed else fail_reason)


def evaluate_gate(inp: GateInput, policy: Policy) -> GateDecision:
    """Evaluate all nine conditions. `should_open_pr` is the logical-AND — every condition
    must pass; otherwise the PR is blocked and `.failed` lists the offending gate(s)."""
    eligible, elig_reason = policy.is_autofix_eligible(inp.classification)

    g5 = inp.build_passed and inp.tests_passed and inp.mojo_ok
    g5_fail = "; ".join(
        r
        for r, ok in (
            ("build failed", inp.build_passed),
            ("tests failed", inp.tests_passed),
            ("Mojo verdict below threshold", inp.mojo_ok),
        )
        if not ok
    )

    g6 = inp.scanner_confirms_gone and inp.scan_not_degraded
    g6_fail = (
        "originating scanner still reports the finding"
        if not inp.scanner_confirms_gone
        else "patched scan is degraded vs baseline (fewer rules/files/queries)"
    )

    g8 = (not inp.alternate_path_found) and (not inp.new_blocking_findings)
    g8_fail = (
        "an equivalent alternate violating path remains"
        if inp.alternate_path_found
        else f"patch introduces {len(inp.new_blocking_findings)} new policy-blocking finding(s)"
    )

    g9 = inp.attestation_ok and inp.base_unchanged
    g9_fail = (
        "evidence attestation failed verification"
        if not inp.attestation_ok
        else "base commit changed before PR open (proof bound to a stale tree)"
    )

    conditions = [
        _cond("G1", inp.provenance_valid, "provenance digest-bound", "incomplete/unbound provenance"),
        _cond("G2", inp.baseline_reproduced, "baseline reproduced on unpatched snapshot",
              "could not reproduce a semantically-equivalent finding on the unpatched snapshot"),
        _cond("G3", eligible, elig_reason, elig_reason),
        _cond("G4", inp.patch_integrity_ok, "patch integrity ok",
              "patch weakens verification or escapes the permitted change scope"),
        _cond("G5", g5, "build/tests pass and Mojo meets threshold", g5_fail),
        _cond("G6", g6, "scanner confirms finding gone under identical config", g6_fail),
        _cond("G7", inp.closure_status == ClosureStatus.CLOSED,
              "obligation closed (sanitizer/barrier or path removed)",
              f"closure_status is {inp.closure_status.value} (need 'closed')"),
        _cond("G8", g8, "no alternate path and no new blocking findings", g8_fail),
        _cond("G9", g9, "attested handoff verified and base unchanged", g9_fail),
    ]
    return GateDecision(conditions, should_open_pr=all(c.passed for c in conditions))
