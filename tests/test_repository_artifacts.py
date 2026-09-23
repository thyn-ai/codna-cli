from __future__ import annotations

import json

import pyarrow.parquet as pq

from codna.repository_artifacts import (
    STEP_ARTIFACT_SCHEMA_VERSION,
    STEP_ARTIFACT_TYPE,
    RepositoryStepArtifactStore,
    RepositoryStepRecord,
    extract_step_metrics,
    sha256_json,
)


def test_repository_step_artifact_store_writes_parquet_and_manifest(tmp_path) -> None:
    store = RepositoryStepArtifactStore(tmp_path / ".codna")
    record = RepositoryStepRecord(
        schema_version=STEP_ARTIFACT_SCHEMA_VERSION,
        artifact_type=STEP_ARTIFACT_TYPE,
        operation_id="op-1",
        repository_id="repo-1",
        connector_type="local_repo",
        connector_fingerprint="fingerprint-1",
        step_name="triage",
        status="succeeded",
        started_at_utc="2026-06-30T00:00:00+00:00",
        completed_at_utc="2026-06-30T00:00:01+00:00",
        duration_ms=1000.0,
        request_sha256=sha256_json({"issue": "parse error"}),
        response_sha256=sha256_json({"snapshot_id": "snap-1"}),
        error_code=None,
        error_message=None,
        snapshot_id="snap-1",
        evidence_bundle_ref="bundle-1",
        decision_plan_id=None,
        simulation_id=None,
        apply_result_id=None,
        metrics_json='{"bundle_tokens":612}',
    )

    artifact = store.write_step(record)

    assert artifact.parquet_path.exists()
    assert artifact.manifest_path.exists()
    row = pq.read_table(artifact.parquet_path).to_pylist()[0]
    assert row["artifact_type"] == STEP_ARTIFACT_TYPE
    assert row["repository_id"] == "repo-1"
    assert row["step_name"] == "triage"
    assert row["status"] == "succeeded"
    assert row["metrics_json"] == '{"bundle_tokens":612}'
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parquet_path"] == str(artifact.parquet_path)
    assert manifest["metrics"] == {"bundle_tokens": 612}


def test_extract_step_metrics_preserves_simulation_risk_contract() -> None:
    metrics = extract_step_metrics(
        {
            "simulation_id": "simulation-1",
            "confidence": 0.5593,
            "recommended_action": "apply_patch",
            "metrics": {
                "probability_of_loss": 0.44066666666666665,
                "var_95": -17.18149308087658,
            },
            "score_breakdown": {
                "apply_gate": {
                    "passed": True,
                    "var_95": -17.18149308087658,
                },
            },
        }
    )

    assert metrics["confidence"] == 0.5593
    assert metrics["simulation_metrics"] == {
        "probability_of_loss": 0.44066666666666665,
        "var_95": -17.18149308087658,
    }
    assert metrics["simulation_probability_of_loss"] == 0.44066666666666665
    assert metrics["simulation_var_95"] == -17.18149308087658
    assert metrics["simulation_recommended_action"] == "apply_patch"
    assert metrics["simulation_gate_passed"] is True
