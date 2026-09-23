"""Cross-repo integration: codna (agent head) <-> decision-engine (the real reachability engine).

Runs BOTH codepaths in one process to prove they agree on the wire contract end to end:
  1. codna ingests SARIF and builds the security-analyses request.
  2. the ENGINE's real pydantic schema validates that request (shape agreement).
  3. the ENGINE's real classifier produces a response (over a fake snapshot manifest).
  4. codna's EngineAdapter parses that exact response — binding echoes pass, verdicts map to
     codna's Classification, and closure maps to codna's ClosureStatus.

Skips cleanly when the decision-engine repo isn't on disk / importable (so codna CI is unaffected).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

ENGINE_ROOT = os.environ.get("DECISION_ENGINE_ROOT", "/Users/angel/Developer/decision-engine")
if not os.path.isdir(ENGINE_ROOT):
    pytest.skip("decision-engine repo not present", allow_module_level=True)
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

# codna side (cli/ is on path via conftest)
from codna.engine import EngineAdapter  # noqa: E402
from codna.findings import Classification, ClosureStatus  # noqa: E402
from codna.patchgen import EnginePatchGenerator  # noqa: E402
from codna.sarif import build_security_analysis_request, ingest_sarif  # noqa: E402
from codna.secure import Patch  # noqa: E402

# engine side — skip the whole module if the engine can't be imported here
try:
    from apps.api_server.schemas.repositories import (  # noqa: E402
        RepositoryDecisionPlanResponse,
        SecurityAnalysisClosureRequest,
        SecurityAnalysisRequest,
    )
    from apps.api_server.services.repository_intelligence_security_analysis_workflow import (  # noqa: E402
        analyze_repository_security_batch,
        analyze_repository_security_closure,
    )
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"decision-engine not importable: {exc}", allow_module_level=True)

SNAPSHOT_ID = "snap_itest"
CONNECTOR = SimpleNamespace(org_id="itest-org", id="itest-repo")


def _sarif():
    return json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "py/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": [{"ruleId": "py/sqli", "message": {"text": "tainted query"},
                         "locations": [{"physicalLocation": {"artifactLocation": {"uri": "db.py"}, "region": {"startLine": 10}}}],
                         "codeFlows": [{"threadFlows": [{"locations": [
                             {"location": {"physicalLocation": {"artifactLocation": {"uri": "app.py"}, "region": {"startLine": 2}}}},
                             {"location": {"physicalLocation": {"artifactLocation": {"uri": "db.py"}, "region": {"startLine": 10}}}}]}]}]}],
        }],
    })


# fake snapshot: db.py (the sink) is reachable from app.py (an entrypoint by path)
_MANIFEST = {
    "files": [
        {"file_path": "app.py", "language": "python", "symbols": [],
         "dependencies": [{"target_file": "db.py", "dependency_type": "imports"}], "symbol_references": []},
        {"file_path": "db.py", "language": "python",
         "symbols": [{"symbol_id": "s1", "symbol_name": "run_query", "symbol_kind": "function",
                      "start_line": 1, "end_line": 30, "parent_symbol_id": None, "signature": ""}],
         "dependencies": [], "symbol_references": []},
    ],
    "resolved_revision": "a" * 40, "content_hash": "sha256:tree",
}


def _ingest_and_request():
    ingest = ingest_sarif(_sarif())
    assert ingest.provenance_valid, ingest.provenance_errors
    request_dict = build_security_analysis_request(
        ingest, snapshot_id=SNAPSHOT_ID, policy_digest="sha256:pol", model_pack_digest="sha256:mp",
    )
    return ingest, request_dict


def _engine_response_dict(request_dict):
    # 2. the ENGINE's pydantic schema validates codna's request (shape agreement)
    engine_req = SecurityAnalysisRequest.model_validate(request_dict)
    # 3. the ENGINE's real classifier runs over the fake snapshot
    resp = analyze_repository_security_batch(
        connector=CONNECTOR, request=engine_req, engine_revision="engine-itest",
        require_repository_connector=lambda c: None,
        load_snapshot_manifest=lambda **kw: _MANIFEST,
    )
    return resp.model_dump(mode="json")


def test_request_shape_validates_against_engine_schema():
    _ingest, request_dict = _ingest_and_request()
    # the engine accepts codna's exact request body without coercion errors
    engine_req = SecurityAnalysisRequest.model_validate(request_dict)
    assert engine_req.snapshot_id == SNAPSHOT_ID
    assert engine_req.findings[0].canonical_id == _ingest.findings[0].canonical_id
    assert engine_req.findings[0].primary_location.uri == "db.py"


def test_engine_response_parses_through_codna_adapter_with_binding_and_verdict():
    ingest, request_dict = _ingest_and_request()
    engine_resp = _engine_response_dict(request_dict)

    # 4. codna's EngineAdapter parses the engine's real response. The stub transport returns the
    # engine output verbatim; the adapter enforces snapshot + sarif_digest binding internally.
    calls = []

    def stub_post(path, body):
        calls.append((path, body))
        assert path.endswith("/security-analyses")
        return engine_resp

    adapter = EngineAdapter(
        stub_post, repository_id="itest-repo", snapshot_id=SNAPSHOT_ID,
        policy_digest="sha256:pol", model_pack_digest="sha256:mp",
    )
    verdict = adapter.analyze(ingest, ingest.findings[0])
    # binding echoes matched (no EngineError raised) and the verdict mapped across repos
    assert verdict.classification is Classification.PRODUCTION_REACHABLE
    assert verdict.envelope_complete is True
    assert len(calls) == 1  # build-once on the codna side too


def test_closure_round_trip_maps_to_codna_closure_status():
    diff = ("diff --git a/db.py b/db.py\n--- a/db.py\n+++ b/db.py\n"
            "@@ -10 +10 @@\n-    cursor.execute('...' + u)\n+    cursor.execute('...', (u,))  # parameterized\n")
    original_proof = {
        "sinks": [{"uri": "db.py"}],
        "propagation_paths": [[{"from_file": "app.py", "to_file": "db.py", "edge_type": "reaches"}]],
    }
    # engine closure verdict for a sanitizing patch on the sink path
    engine_closure = analyze_repository_security_closure(
        connector=CONNECTOR,
        request=SecurityAnalysisClosureRequest(
            snapshot_id=SNAPSHOT_ID, canonical_finding_id="sha256:f", patch_digest="sha256:p"),
        engine_revision="engine-itest",
        require_repository_connector=lambda c: None,
        load_security_proof_by_finding=lambda **kw: original_proof,
        load_patch_diff_by_digest=lambda **kw: diff,
    ).model_dump(mode="json")
    assert engine_closure["closure_status"] == "closed"

    # codna's EngineAdapter maps the engine closure response to its ClosureStatus
    adapter = EngineAdapter(
        lambda path, body: engine_closure, repository_id="itest-repo", snapshot_id=SNAPSHOT_ID,
        policy_digest="sha256:pol", model_pack_digest="sha256:mp",
    )
    cv = adapter.reprove_closure(SimpleNamespace(canonical_id="sha256:f"), Patch("sha256:p", diff=diff))
    assert cv.closure_status is ClosureStatus.CLOSED
    assert cv.alternate_path_found is False


def test_decision_plan_patch_diff_field_satisfies_patch_generator_contract():
    # the engine's RepositoryDecisionPlanResponse now exposes patch_diff, which is exactly what
    # codna's EnginePatchGenerator reads from the decision-plan response body.
    assert "patch_diff" in RepositoryDecisionPlanResponse.model_fields
    diff = "diff --git a/db.py b/db.py\n--- a/db.py\n+++ b/db.py\n@@ -1 +1 @@\n-x\n+safe(x)\n"
    gen = EnginePatchGenerator(lambda path, body: {"patch_diff": diff},
                               repository_id="itest-repo", snapshot_id=SNAPSHOT_ID)

    class _F:
        canonical_id = "sha256:f"
        rule_id = "py/sqli"
        finding_kind = SimpleNamespace(value="taint")
        severity = SimpleNamespace(value="critical")
        message = "x"
        primary_location = None
        code_flows: list = []

    assert gen(_F(), "/tmp/wt") == diff
