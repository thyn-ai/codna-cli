"""End-to-end orchestration tests for run_secure (A–Z test plan: E2E-GATE-00, G3 skip,
gate-fail blocks, one-PR-per-finding, ineligible-never-remediated, open_pr gating).

All offline: stub engine/worker/writer satisfy the privilege-domain protocols. No real
engine, subprocess, or network.
"""
from __future__ import annotations

import json


from codna.evidence import HmacSigner, build_attestation
from codna.findings import Classification, ClosureStatus
from codna.policy import Policy
from codna.sarif import ingest_sarif
from codna.secure import (
    ClosureVerdict,
    Patch,
    ReachVerdict,
    VerifyResult,
    run_secure,
)

SIGNER = HmacSigner(b"worker-secret", key_id="worker-key")
PASS_VERIFY = VerifyResult(
    build_passed=True, tests_passed=True, mojo_ok=True,
    scanner_confirms_gone=True, scan_not_degraded=True,
)


# --------------------------------------------------------------------- fixtures

def _ingest(n_findings=1):
    results = [
        {
            "ruleId": f"js/sqli-{i}",
            "message": {"text": "sqli"},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": f"src/h{i}.py"},
                                                 "region": {"startLine": 10 + i}}}],
            "codeFlows": [{"threadFlows": [{"locations": [
                {"location": {"physicalLocation": {"artifactLocation": {"uri": f"src/in{i}.py"},
                                                    "region": {"startLine": 1}}}}]}]}],
        }
        for i in range(n_findings)
    ]
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": f"js/sqli-{i}",
                                           "properties": {"security-severity": "9.1", "tags": ["security"]}}
                                          for i in range(n_findings)]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": results,
        }],
    }
    return ingest_sarif(json.dumps(doc))


# --------------------------------------------------------------------- stubs

class StubEngine:
    def __init__(self, classification=Classification.EXPLOITABLE, closure=ClosureStatus.CLOSED,
                 alt=False, new=None):
        self.classification = classification
        self.closure = closure
        self.alt = alt
        self.new = new or []
        self.analyze_calls = 0
        self.closure_calls = 0

    def analyze(self, ingest, finding):
        self.analyze_calls += 1
        return ReachVerdict(self.classification, proof_type="interprocedural-source-to-sink")

    def reprove_closure(self, finding, patch):
        self.closure_calls += 1
        return ClosureVerdict(self.closure, self.alt, list(self.new))


class StubWorker:
    def __init__(self, baseline=True, integrity=True, verify=PASS_VERIFY):
        self.baseline = baseline
        self.integrity = integrity
        self.verify_result = verify
        self.reproduce_calls = 0
        self.remediate_calls = 0

    def reproduce_baseline(self, finding):
        self.reproduce_calls += 1
        return self.baseline

    def remediate(self, finding):
        self.remediate_calls += 1
        return Patch(patch_digest="sha256:patch", integrity_ok=self.integrity)

    def verify(self, finding, patch):
        return self.verify_result

    def attest(self, finding, patch, decision_inputs):
        return build_attestation(
            SIGNER, nonce="n-" + finding.canonical_id[-8:],
            original_commit="a" * 40, patch_digest=patch.patch_digest, resulting_tree="sha256:tree",
            scanner_outputs={}, proof_results={}, test_logs="", mojo_verdict={},
            policy_decision=decision_inputs,
        )


class StubWriter:
    def __init__(self, base_unchanged=True):
        self._base = base_unchanged
        self.opened: list[str] = []

    def base_unchanged(self, ingest):
        return self._base

    def open_draft_pr(self, finding, patch, attestation):
        url = f"https://github.com/o/r/pull/{len(self.opened) + 1}"
        self.opened.append(url)
        return url


def _run(ingest=None, engine=None, worker=None, writer=None, policy=None, open_pr=True):
    return run_secure(
        ingest or _ingest(),
        engine=engine or StubEngine(),
        worker=worker or StubWorker(),
        writer=writer or StubWriter(),
        policy=policy or Policy(),
        open_pr=open_pr,
    )


# --------------------------------------------------------------------- tests

def test_e2e_happy_path_opens_draft_pr():
    writer = StubWriter()
    report = _run(writer=writer)
    assert report.analyzed == 1 and report.eligible == 1
    d = report.decisions[0]
    assert d.opened is True and d.pr_url == "https://github.com/o/r/pull/1"
    assert d.gate.should_open_pr is True and d.gate.failed == []
    assert writer.opened == [d.pr_url]


def test_ineligible_unknown_is_skipped_and_never_remediated():
    worker = StubWorker()
    report = _run(engine=StubEngine(classification=Classification.UNKNOWN), worker=worker)
    d = report.decisions[0]
    assert d.opened is False and d.gate is None
    assert "not autofix-eligible" in d.skipped_reason
    assert report.eligible == 0
    assert worker.remediate_calls == 0  # no patch generated for an ineligible finding


def test_closure_open_blocks_pr():
    writer = StubWriter()
    report = _run(engine=StubEngine(closure=ClosureStatus.OPEN), writer=writer)
    d = report.decisions[0]
    assert d.opened is False
    assert "G7" in d.gate.failed
    assert writer.opened == []


def test_new_blocking_finding_blocks_pr():
    report = _run(engine=StubEngine(new=["new-critical"]))
    assert "G8" in report.decisions[0].gate.failed
    assert report.decisions[0].opened is False


def test_base_changed_blocks_pr():
    report = _run(writer=StubWriter(base_unchanged=False))
    assert "G9" in report.decisions[0].gate.failed
    assert report.decisions[0].opened is False


def test_open_pr_false_decides_but_does_not_open():
    writer = StubWriter()
    report = _run(writer=writer, open_pr=False)
    d = report.decisions[0]
    assert d.gate.should_open_pr is True  # would qualify
    assert d.opened is False and writer.opened == []  # but report-only
    assert len(report.evidence_bundles) == 1
    assert report.evidence_bundles[0].finding.canonical_id == d.canonical_id
    assert report.evidence_bundles[0].patch.patch_digest == "sha256:patch"


def test_gate_failed_decision_does_not_emit_writer_evidence():
    report = _run(engine=StubEngine(closure=ClosureStatus.OPEN), open_pr=False)
    assert report.decisions[0].gate.should_open_pr is False
    assert report.evidence_bundles == []


def test_one_pr_per_finding():
    writer = StubWriter()
    report = _run(ingest=_ingest(n_findings=3), writer=writer)
    assert report.analyzed == 3 and report.eligible == 3
    assert sum(1 for d in report.decisions if d.opened) == 3
    assert len(writer.opened) == 3 and len(set(writer.opened)) == 3


def test_production_reachable_requires_policy_override():
    eng = StubEngine(classification=Classification.PRODUCTION_REACHABLE)
    blocked = _run(engine=eng)
    assert blocked.decisions[0].opened is False  # default policy -> ineligible (skipped)

    pol = Policy(autofix_classifications=("exploitable", "production-reachable"))
    allowed = _run(engine=StubEngine(classification=Classification.PRODUCTION_REACHABLE), policy=pol)
    assert allowed.decisions[0].opened is True
