from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

STEP_ARTIFACT_SCHEMA_VERSION = 1
STEP_ARTIFACT_TYPE = "codna.repository_intelligence.step.v1"

StepStatus = Literal["succeeded", "failed"]


class RepositoryArtifactError(RuntimeError):
    def __init__(self, message: str, *, code: str = "repository_artifact_error", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class RepositoryStepRecord:
    schema_version: int
    artifact_type: str
    operation_id: str
    repository_id: str
    connector_type: str
    connector_fingerprint: str
    step_name: str
    status: StepStatus
    started_at_utc: str
    completed_at_utc: str
    duration_ms: float
    request_sha256: str
    response_sha256: str | None
    error_code: str | None
    error_message: str | None
    snapshot_id: str | None
    evidence_bundle_ref: str | None
    decision_plan_id: str | None
    simulation_id: str | None
    apply_result_id: str | None
    metrics_json: str


@dataclass(frozen=True)
class RepositoryStepArtifact:
    operation_id: str
    directory: Path
    parquet_path: Path
    manifest_path: Path


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if normalized and len(normalized) <= 96:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"id-{digest}"


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _arrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        raise RepositoryArtifactError(
            "Local repository-intelligence step artifacts require pyarrow.",
            code="pyarrow_unavailable",
            details={"reason": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return pa, pq


def _record_to_table(record: RepositoryStepRecord) -> Any:
    pa, _pq = _arrow_modules()
    schema = pa.schema(
        [
            ("schema_version", pa.int64()),
            ("artifact_type", pa.string()),
            ("operation_id", pa.string()),
            ("repository_id", pa.string()),
            ("connector_type", pa.string()),
            ("connector_fingerprint", pa.string()),
            ("step_name", pa.string()),
            ("status", pa.string()),
            ("started_at_utc", pa.string()),
            ("completed_at_utc", pa.string()),
            ("duration_ms", pa.float64()),
            ("request_sha256", pa.string()),
            ("response_sha256", pa.string()),
            ("error_code", pa.string()),
            ("error_message", pa.string()),
            ("snapshot_id", pa.string()),
            ("evidence_bundle_ref", pa.string()),
            ("decision_plan_id", pa.string()),
            ("simulation_id", pa.string()),
            ("apply_result_id", pa.string()),
            ("metrics_json", pa.string()),
        ]
    )
    values = asdict(record)
    return pa.Table.from_pydict({name: [values[name]] for name in schema.names}, schema=schema)


def _write_parquet_atomic(path: Path, record: RepositoryStepRecord) -> None:
    _pa, pq = _arrow_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(_record_to_table(record), tmp, compression="zstd")
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def extract_step_refs(response: dict[str, Any] | None) -> dict[str, str | None]:
    if not response:
        return {
            "snapshot_id": None,
            "evidence_bundle_ref": None,
            "decision_plan_id": None,
            "simulation_id": None,
            "apply_result_id": None,
        }
    return {
        "snapshot_id": _string_or_none(response.get("snapshot_id")),
        "evidence_bundle_ref": _string_or_none(
            response.get("workspace_evidence_bundle_ref")
            or response.get("evidence_bundle_ref")
            or response.get("bundle_ref")
        ),
        "decision_plan_id": _string_or_none(
            response.get("decision_plan_id") or response.get("plan_id")
        ),
        "simulation_id": _string_or_none(
            response.get("simulation_id") or response.get("simulation_ref")
        ),
        "apply_result_id": _string_or_none(
            response.get("apply_result_id") or response.get("patch_ref") or response.get("result_id")
        ),
    }


def extract_step_metrics(response: dict[str, Any] | None) -> dict[str, Any]:
    if not response:
        return {}
    metrics: dict[str, Any] = {}
    for key in (
        "file_count",
        "symbol_count",
        "bundle_tokens",
        "raw_repo_tokens",
        "reduction_ratio",
        "confidence",
        "risk_score",
    ):
        if key in response:
            metrics[key] = response[key]
    simulation_metrics = response.get("metrics")
    if isinstance(simulation_metrics, dict):
        metrics["simulation_metrics"] = simulation_metrics
        for source_key, target_key in (
            ("probability_of_loss", "simulation_probability_of_loss"),
            ("var_95", "simulation_var_95"),
            ("expected_value", "simulation_expected_value"),
        ):
            if source_key in simulation_metrics:
                metrics[target_key] = simulation_metrics[source_key]
    if "recommended_action" in response:
        metrics["simulation_recommended_action"] = response["recommended_action"]
    score_breakdown = response.get("score_breakdown")
    if isinstance(score_breakdown, dict):
        metrics["score_breakdown"] = score_breakdown
        apply_gate = score_breakdown.get("apply_gate")
        if isinstance(apply_gate, dict):
            if "passed" in apply_gate:
                metrics["simulation_gate_passed"] = apply_gate["passed"]
            if "var_95" in apply_gate and "simulation_var_95" not in metrics:
                metrics["simulation_var_95"] = apply_gate["var_95"]
    for nested_key in ("token_usage", "usage", "timing", "simulation"):
        nested = response.get(nested_key)
        if isinstance(nested, dict):
            metrics[nested_key] = nested
    return metrics


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


class RepositoryStepArtifactStore:
    def __init__(self, runtime_root: Path) -> None:
        self._root = runtime_root / "repository-intelligence" / "steps"

    def write_step(self, record: RepositoryStepRecord) -> RepositoryStepArtifact:
        if record.schema_version != STEP_ARTIFACT_SCHEMA_VERSION:
            raise RepositoryArtifactError(
                "Unsupported repository step artifact schema version.",
                code="unsupported_step_artifact_schema",
                details={"schema_version": record.schema_version},
            )
        step_dir = (
            self._root
            / _safe_segment(record.repository_id)
            / _safe_segment(record.step_name)
            / _safe_segment(record.operation_id)
        )
        parquet_path = step_dir / "step.parquet"
        manifest_path = step_dir / "manifest.json"
        try:
            _write_parquet_atomic(parquet_path, record)
            _write_json_atomic(manifest_path, self._manifest(record, parquet_path))
        except OSError as exc:
            raise RepositoryArtifactError(
                "Could not write repository-intelligence step artifact.",
                code="repository_step_artifact_write_failed",
                details={"path": str(step_dir), "reason": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return RepositoryStepArtifact(
            operation_id=record.operation_id,
            directory=step_dir,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
        )

    def _manifest(self, record: RepositoryStepRecord, parquet_path: Path) -> dict[str, Any]:
        return {
            "schema_version": STEP_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": STEP_ARTIFACT_TYPE,
            "operation_id": record.operation_id,
            "repository_id": record.repository_id,
            "connector_type": record.connector_type,
            "connector_fingerprint": record.connector_fingerprint,
            "step_name": record.step_name,
            "status": record.status,
            "started_at_utc": record.started_at_utc,
            "completed_at_utc": record.completed_at_utc,
            "duration_ms": record.duration_ms,
            "request_sha256": record.request_sha256,
            "response_sha256": record.response_sha256,
            "error_code": record.error_code,
            "snapshot_id": record.snapshot_id,
            "evidence_bundle_ref": record.evidence_bundle_ref,
            "decision_plan_id": record.decision_plan_id,
            "simulation_id": record.simulation_id,
            "apply_result_id": record.apply_result_id,
            "metrics": json.loads(record.metrics_json),
            "parquet_path": str(parquet_path),
        }
