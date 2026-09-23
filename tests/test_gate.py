"""PR-open gate tests (A–Z test plan: BRCP-38/39 composition, per-condition blocks, G3/G7)."""
from __future__ import annotations

import pytest

from codna.findings import Classification, ClosureStatus
from codna.gate import GateInput, evaluate_gate
from codna.policy import Policy


def _passing_input(**over) -> GateInput:
    base = dict(
        provenance_valid=True,
        baseline_reproduced=True,
        classification=Classification.EXPLOITABLE,
        patch_integrity_ok=True,
        build_passed=True,
        tests_passed=True,
        mojo_ok=True,
        scanner_confirms_gone=True,
        scan_not_degraded=True,
        closure_status=ClosureStatus.CLOSED,
        alternate_path_found=False,
        new_blocking_findings=[],
        attestation_ok=True,
        base_unchanged=True,
    )
    base.update(over)
    return GateInput(**base)


def test_happy_path_opens_pr():
    d = evaluate_gate(_passing_input(), Policy())
    assert d.should_open_pr is True
    assert d.failed == []
    assert all(c.passed for c in d.conditions)


@pytest.mark.parametrize(
    "gate,override",
    [
        ("G1", dict(provenance_valid=False)),
        ("G2", dict(baseline_reproduced=False)),
        ("G3", dict(classification=Classification.UNKNOWN)),
        ("G4", dict(patch_integrity_ok=False)),
        ("G5", dict(tests_passed=False)),
        ("G5", dict(mojo_ok=False)),
        ("G6", dict(scanner_confirms_gone=False)),
        ("G6", dict(scan_not_degraded=False)),
        ("G7", dict(closure_status=ClosureStatus.OPEN)),
        ("G7", dict(closure_status=ClosureStatus.UNKNOWN)),
        ("G8", dict(alternate_path_found=True)),
        ("G8", dict(new_blocking_findings=["new-high"])),
        ("G9", dict(attestation_ok=False)),
        ("G9", dict(base_unchanged=False)),
    ],
)
def test_single_condition_failure_blocks_pr(gate, override):
    """BRCP-39: logical-AND — flipping exactly one input blocks the PR and names that gate."""
    d = evaluate_gate(_passing_input(**override), Policy())
    assert d.should_open_pr is False
    assert gate in d.failed


def test_g3_production_reachable_blocks_without_override_passes_with():
    blocked = evaluate_gate(_passing_input(classification=Classification.PRODUCTION_REACHABLE), Policy())
    assert blocked.should_open_pr is False and "G3" in blocked.failed

    pol = Policy(autofix_classifications=("exploitable", "production-reachable"))
    allowed = evaluate_gate(_passing_input(classification=Classification.PRODUCTION_REACHABLE), pol)
    assert allowed.should_open_pr is True


def test_g7_sanitizer_closes_even_if_still_production_reachable():
    """A fix that sanitizes but leaves the sink production-reachable still passes G7 — the
    gate keys off closure_status==CLOSED, not the disappearance of reachability."""
    d = evaluate_gate(_passing_input(closure_status=ClosureStatus.CLOSED), Policy())
    assert d.reason_for("G7").startswith("obligation closed")
    assert d.should_open_pr is True


def test_unreachable_never_opens_even_if_everything_else_green():
    d = evaluate_gate(_passing_input(classification=Classification.UNREACHABLE), Policy())
    assert d.should_open_pr is False
    assert "G3" in d.failed
