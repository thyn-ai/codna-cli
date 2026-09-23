"""Engine-adapter tests (A–Z test plan: RPRV-01/02 build-once, classification mapping,
RPRV-30 response binding, unreachable-needs-complete-envelope, closure mapping)."""
from __future__ import annotations

import json

import pytest

from codna.engine import EngineAdapter, EngineError
from codna.findings import Classification, ClosureStatus
from codna.policy import Policy
from codna.sarif import ingest_sarif
from codna.secure import classify_only


def _ingest(n=1, quarantine_last=False):
    results = []
    for i in range(n):
        uri = "../../etc/passwd" if (quarantine_last and i == n - 1) else f"src/h{i}.py"
        results.append({
            "ruleId": f"js/sqli-{i}",
            "message": {"text": "sqli"},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": uri},
                                                "region": {"startLine": 10 + i}}}],
            "codeFlows": [{"threadFlows": [{"locations": [
                {"location": {"physicalLocation": {"artifactLocation": {"uri": f"in{i}.py"},
                                                   "region": {"startLine": 1}}}}]}]}],
        })
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": f"js/sqli-{i}",
                                           "properties": {"security-severity": "9.1", "tags": ["security"]}}
                                          for i in range(n)]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": results,
        }],
    }
    return ingest_sarif(json.dumps(doc))


class StubPost:
    def __init__(self, *, classification="exploitable", complete=True, closure="closed",
                 alt=False, new=None, classifier=None, snapshot_id_echo=None, sarif_echo=True,
                 drop_ids=()):
        self.classification = classification
        self.complete = complete
        self.closure = closure
        self.alt = alt
        self.new = new or []
        self.classifier = classifier
        self.snapshot_id_echo = snapshot_id_echo
        self.sarif_echo = sarif_echo
        self.drop_ids = set(drop_ids)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path, body):
        self.calls.append((path, body))
        if path.endswith("/security-analyses"):
            proofs = []
            for f in body["findings"]:
                if f["canonical_id"] in self.drop_ids:
                    continue
                cls = self.classifier(f) if self.classifier else self.classification
                proofs.append({
                    "canonical_finding_id": f["canonical_id"],
                    "classification": cls,
                    "analysis_envelope": {"is_complete_sound": self.complete},
                    "proof_type": "interprocedural-source-to-sink",
                })
            resp = {"analysis_id": "sec_1", "proofs": proofs}
            resp["snapshot"] = {"id": self.snapshot_id_echo or body["snapshot_id"]}
            resp["sarif_digest"] = body["sarif_digest"] if self.sarif_echo else "sha256:WRONG"
            return resp
        if path.endswith("/closure"):
            return {"closure_status": self.closure, "alternate_path_found": self.alt,
                    "new_blocking_findings": self.new}
        return {}

    @property
    def analysis_calls(self):
        return [c for c in self.calls if c[0].endswith("/security-analyses")]


def _adapter(post, ingest):
    return EngineAdapter(post, repository_id="r1", snapshot_id="snap_1",
                         policy_digest="sha256:pol", model_pack_digest="sha256:mp")


def test_rprv02_build_once_query_many():
    ing = _ingest(n=3)
    post = StubPost(classification="exploitable")
    eng = _adapter(post, ing)
    for f in ing.findings:
        eng.analyze(ing, f)
    assert len(post.analysis_calls) == 1  # one batch request for all 3 findings


@pytest.mark.parametrize("cls,expected", [
    ("exploitable", Classification.EXPLOITABLE),
    ("production-reachable", Classification.PRODUCTION_REACHABLE),
    ("unreachable", Classification.UNREACHABLE),
    ("unknown", Classification.UNKNOWN),
    ("garbage-value", Classification.UNKNOWN),
])
def test_classification_mapping(cls, expected):
    ing = _ingest(1)
    eng = _adapter(StubPost(classification=cls, complete=True), ing)
    assert eng.analyze(ing, ing.findings[0]).classification is expected


def test_unreachable_requires_complete_envelope_else_unknown():
    ing = _ingest(1)
    eng = _adapter(StubPost(classification="unreachable", complete=False), ing)
    v = eng.analyze(ing, ing.findings[0])
    assert v.classification is Classification.UNKNOWN  # not silently 'unreachable'
    assert v.envelope_complete is False


def test_missing_proof_is_unknown_not_unreachable():
    ing = _ingest(2)
    drop = ing.findings[1].canonical_id
    eng = _adapter(StubPost(classification="exploitable", drop_ids=[drop]), ing)
    assert eng.analyze(ing, ing.findings[0]).classification is Classification.EXPLOITABLE
    assert eng.analyze(ing, ing.findings[1]).classification is Classification.UNKNOWN


def test_rprv30_snapshot_mismatch_rejected():
    ing = _ingest(1)
    eng = _adapter(StubPost(snapshot_id_echo="someone-elses-snapshot"), ing)
    with pytest.raises(EngineError):
        eng.analyze(ing, ing.findings[0])


def test_rprv30_sarif_digest_mismatch_rejected():
    ing = _ingest(1)
    eng = _adapter(StubPost(sarif_echo=False), ing)
    with pytest.raises(EngineError):
        eng.analyze(ing, ing.findings[0])


@pytest.mark.parametrize("status,expected", [
    ("closed", ClosureStatus.CLOSED),
    ("open", ClosureStatus.OPEN),
    ("unknown", ClosureStatus.UNKNOWN),
    ("weird", ClosureStatus.UNKNOWN),
])
def test_closure_mapping(status, expected):
    ing = _ingest(1)
    eng = _adapter(StubPost(closure=status), ing)

    class _P:
        patch_digest = "sha256:patch"

    assert eng.reprove_closure(ing.findings[0], _P()).closure_status is expected


def test_classify_only_tier1_report():
    ing = _ingest(3)
    # finding 0 exploitable, 1 unknown, 2 unreachable(complete)
    by_index = {0: "exploitable", 1: "unknown", 2: "unreachable"}
    order = [f.canonical_id for f in ing.findings]

    def classifier(f):
        return by_index[order.index(f["canonical_id"])]

    eng = _adapter(StubPost(classifier=classifier, complete=True), ing)
    report = classify_only(ing, engine=eng, policy=Policy())
    assert report.eligible == 1  # only exploitable
    assert report.counts() == {"exploitable": 1, "unknown": 1, "unreachable": 1}


def test_classify_only_quarantined_short_circuits_engine():
    ing = _ingest(2, quarantine_last=True)
    post = StubPost(classification="exploitable")
    eng = _adapter(post, ing)
    report = classify_only(ing, engine=eng, policy=Policy())
    quarantined = [r for r in report.rows if r.classification == "quarantined"]
    assert len(quarantined) == 1 and quarantined[0].eligible is False
