"""LocalReferenceEngine tests — the bounded self-host engine. Verifies it applies the
four-outcome discipline honestly and NEVER overclaims `exploitable`."""
from __future__ import annotations

import json

import pytest

from codna.findings import Classification, ClosureStatus
from codna.policy import Policy
from codna.refengine import LocalReferenceEngine
from codna.sarif import ingest_sarif
from codna.secure import Patch, classify_only


def _ingest(uri="src/app.py", with_flow=True, lang_ext=None):
    path = uri if lang_ext is None else uri
    flow = ([{"threadFlows": [{"locations": [
        {"location": {"physicalLocation": {"artifactLocation": {"uri": "src/in.py"}, "region": {"startLine": 1}}}},
        {"location": {"physicalLocation": {"artifactLocation": {"uri": path}, "region": {"startLine": 9}}}}]}]}]
        if with_flow else [])
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "x/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": [{"ruleId": "x/sqli", "message": {"text": "sqli"},
                         "locations": [{"physicalLocation": {"artifactLocation": {"uri": path}, "region": {"startLine": 9}}}],
                         "codeFlows": flow}],
        }],
    }
    return ingest_sarif(json.dumps(doc))


def test_never_returns_exploitable_even_with_scanner_flow():
    ing = _ingest("src/app.py", with_flow=True)  # supported language + scanner code-flow
    v = LocalReferenceEngine().analyze(ing, ing.findings[0])
    assert v.classification is Classification.PRODUCTION_REACHABLE  # honest ceiling, NOT exploitable
    assert v.envelope_complete is True
    assert v.proof_type == "scanner-flow-corroborated"


def test_off_matrix_language_is_unknown():
    ing = _ingest("legacy/main.cobol", with_flow=True)
    v = LocalReferenceEngine().analyze(ing, ing.findings[0])
    assert v.classification is Classification.UNKNOWN
    assert "off-matrix" in v.proof_type


def test_bounded_framework_is_unknown_not_reachable():
    ing = _ingest("src/Main.java", with_flow=True)
    eng = LocalReferenceEngine(framework_for=lambda p: "spring")  # reflection -> bounded envelope
    v = eng.analyze(ing, ing.findings[0])
    assert v.classification is Classification.UNKNOWN
    assert v.envelope_complete is False


def test_plain_java_is_complete_and_reachable():
    ing = _ingest("src/Main.java", with_flow=False)
    v = LocalReferenceEngine().analyze(ing, ing.findings[0])  # no framework override
    assert v.classification is Classification.PRODUCTION_REACHABLE


def test_test_only_sink_is_unreachable_under_complete_envelope():
    ing = _ingest("tests/test_app.py", with_flow=True)
    v = LocalReferenceEngine().analyze(ing, ing.findings[0])
    assert v.classification is Classification.UNREACHABLE
    assert v.envelope_complete is True


def test_no_path_in_incomplete_envelope_is_unknown_never_unreachable():
    ing = _ingest("spec/thing.rb", with_flow=False)  # ruby = bounded
    v = LocalReferenceEngine().analyze(ing, ing.findings[0])
    assert v.classification is Classification.UNKNOWN  # NOT unreachable


def test_classify_only_with_default_policy_yields_no_eligible():
    # The bounded reference can only assert production-reachable, which the default policy
    # does NOT autofix — so the reference engine alone never green-lights a PR (honest).
    ing = _ingest("src/app.py")
    report = classify_only(ing, engine=LocalReferenceEngine(), policy=Policy())
    assert report.eligible == 0
    # ...unless the operator explicitly opts production-reachable into the policy.
    pol = Policy(autofix_classifications=("exploitable", "production-reachable"))
    assert classify_only(ing, engine=LocalReferenceEngine(), policy=pol).eligible == 1


# -------------------------------------------------------------- closure heuristic

DIFF_SANITIZED = (
    "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
    "@@ -9 +9 @@\n-    cursor.execute(\"...\" + u)\n+    cursor.execute(\"...\", (u,))  # parameterized\n"
)
DIFF_ELSEWHERE = (
    "diff --git a/src/other.py b/src/other.py\n--- a/src/other.py\n+++ b/src/other.py\n"
    "@@ -1 +1 @@\n-x\n+y\n"
)
DIFF_TOUCH_NO_BARRIER = (
    "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
    "@@ -9 +9 @@\n-    cursor.execute(q)\n+    cursor.execute(q2)\n"
)
# Real parameterization with the `cur.` alias + a SQL literal that itself contains commas
# ("id, email") — the old regex missed this; the generalized one recognizes the bound-param call.
DIFF_PARAMETERIZED = (
    "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -9 +9 @@\n"
    "-    cur.execute(\"SELECT id, email FROM t WHERE id = '\" + u + \"'\")\n"
    "+    cur.execute(\"SELECT id, email FROM t WHERE id = ?\", (u,))\n"
)


@pytest.mark.parametrize("diff,expected", [
    (DIFF_SANITIZED, ClosureStatus.CLOSED),
    (DIFF_PARAMETERIZED, ClosureStatus.CLOSED),    # bound-param query (cur.execute(sql, (u,)))
    (DIFF_ELSEWHERE, ClosureStatus.OPEN),          # sink file untouched
    (DIFF_TOUCH_NO_BARRIER, ClosureStatus.OPEN),   # touched but no recognized barrier
    ("", ClosureStatus.UNKNOWN),
])
def test_closure_heuristic(diff, expected):
    ing = _ingest("src/app.py")
    cv = LocalReferenceEngine().reprove_closure(ing.findings[0], Patch("sha256:p", diff=diff))
    assert cv.closure_status is expected
