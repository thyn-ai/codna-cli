from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Literal, Mapping

from .agent_core_runtime import ensure_agent_core_running, stop_agent_core_runtime
from .local_mojo_daemon import (
    ensure_local_mojo_runtime as _ensure_local_mojo_runtime,
    prewarm_local_mojo_runtime as _prewarm_local_mojo_runtime,
    publish_local_mojo_runtime_state as _publish_local_mojo_runtime_state,
)
from .packaged_agent_runner import SidecarPackagedAgentRunner
from .python_import_paths import compatible_engine_site_packages as _compatible_engine_site_packages
from .packaged_repository_backend import PackagedRepositoryBackend
from .local_repository_shims import (
    git_apply_compatible_unified_diff_from_tree_states
    as _git_apply_compatible_unified_diff_from_tree_states,
    install_patch_artifact_newline_preservation
    as _install_patch_artifact_newline_preservation,
    install_precise_patch_changed_files as _install_precise_patch_changed_files,
    install_repository_module_suffix_support as _install_repository_module_suffix_support,
    install_verified_agentic_diff_compatibility as _install_verified_agentic_diff_compatibility,
    precise_patch_changed_files as _precise_patch_changed_files,
    read_text_preserving_newlines as _read_text_preserving_newlines,
)
from .repository_artifacts import (
    STEP_ARTIFACT_SCHEMA_VERSION,
    STEP_ARTIFACT_TYPE,
    RepositoryArtifactError,
    RepositoryStepArtifactStore,
    RepositoryStepRecord,
    canonical_json,
    extract_step_metrics,
    extract_step_refs,
    sha256_json,
)
from .runtime.config import (
    ENGINE_URL_KEYS,
    ConfigValue,
    LOCAL_DATABASE_ENV_KEYS,
    RuntimeConfig,
    local_engine_defaults,
    resolve_runtime_config,
)

# These aliases are intentionally surfaced on codna.local_client for compatibility tests that
# verify the local Algenta shim behavior without importing private shim modules directly.
_LOCAL_REPOSITORY_SHIM_TEST_SEAMS = (
    _git_apply_compatible_unified_diff_from_tree_states,
    _precise_patch_changed_files,
    _read_text_preserving_newlines,
)

_LOCAL_ORG_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
_VERIFIED_AGENTIC_MODEL = "repository.verified_agentic_v1"
_LOCAL_AGENT_CORE_DEFAULTS = {
    "ALGENTA_AGENT_TOKEN_BUDGET": "6000",
    "ALGENTA_AGENT_MAX_TURNS": "4",
    "ALGENTA_AGENT_TURN_TIMEOUT_MS": "600000",
    "ALGENTA_REPOSITORY_AGENT_TOOL_PROFILE": "no_exploration_local_validation",
}
_SIMULATION_FAIL_CLOSED_RISK_THRESHOLD = 0.95
_AGENT_CORE_TRANSPORT_MARKERS = (
    "agent-core sidecar unreachable",
    "agent core sidecar unreachable",
    "agent-core transport",
    "agent_core_transport",
)
_AGENT_CORE_CONNECTION_MARKERS = (
    "all connection attempts failed",
    "connection refused",
    "connecterror",
    "readerror",
    "writeerror",
    "connection reset",
    "connection aborted",
)
@dataclass(frozen=True)
class _ConnectorRecord:
    id: str
    name: str
    connector_type: str
    config: dict[str, Any]
    config_fingerprint: str


@dataclass(frozen=True)
class _RepositoryModules:
    core: ModuleType
    schemas: ModuleType
    data_connector: ModuleType


_CONNECTORS: dict[str, _ConnectorRecord] = {}
_CONNECTORS_LOCK = threading.Lock()
_INSERTED_DECISION_ENGINE_IMPORT_PATHS: set[str] = set()


class LocalCodnaClientError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return canonical_json(dict(payload))


def _fingerprint_config(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_code(exc: BaseException) -> str:
    # Same `.code` vs decision_engine's `.error_code` gap as cli.py's _structured_error —
    # without this fallback an SDK ServerError degrades to the uninformative "ServerError"
    # class name instead of its real engine-reported code.
    code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
    return str(code) if code else type(exc).__name__


def _error_message(exc: BaseException) -> str:
    return str(exc)[:1000]


def _is_agent_core_transport_error(exc: BaseException) -> bool:
    code = _error_code(exc).lower()
    message = _error_message(exc).lower()
    if "agent_core" in code and any(
        token in code for token in ("transport", "unreachable", "connection", "sidecar")
    ):
        return True
    if any(marker in message for marker in _AGENT_CORE_TRANSPORT_MARKERS):
        return True
    if ("agent-core" in message or "agent core" in message or "sidecar" in message) and any(
        marker in message for marker in _AGENT_CORE_CONNECTION_MARKERS
    ):
        return True
    return False


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    raise LocalCodnaClientError(
        "invalid_local_repository_response",
        "Local repository-intelligence service returned a non-object response.",
        {"type": type(value).__name__},
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _simulation_id_from_response(response: Mapping[str, Any]) -> str | None:
    validated_inputs = response.get("validated_inputs")
    if isinstance(validated_inputs, dict) and validated_inputs.get("simulation_id"):
        return str(validated_inputs["simulation_id"])
    for key in ("simulation_id", "id"):
        value = response.get(key)
        if value:
            return str(value)
    return None


def _simulation_gate_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _dict_or_empty(response.get("metrics"))
    score_breakdown = _dict_or_empty(response.get("score_breakdown"))
    apply_gate = _dict_or_empty(score_breakdown.get("apply_gate"))
    var_95 = _float_or_none(metrics.get("var_95"))
    if var_95 is None:
        var_95 = _float_or_none(apply_gate.get("var_95"))
    return {
        "probability_of_loss": _float_or_none(metrics.get("probability_of_loss")),
        "var_95": var_95,
        "recommended_action": response.get("recommended_action")
        if isinstance(response.get("recommended_action"), str)
        else None,
        "apply_gate_passed": apply_gate.get("passed")
        if isinstance(apply_gate.get("passed"), bool)
        else None,
    }


def _simulation_requires_fail_closed(response: Mapping[str, Any]) -> dict[str, Any] | None:
    summary = _simulation_gate_summary(response)
    risk = summary["probability_of_loss"]
    var_95 = summary["var_95"]
    if risk is None or var_95 is None:
        return None
    if risk < _SIMULATION_FAIL_CLOSED_RISK_THRESHOLD or var_95 >= 0:
        return None
    if summary["apply_gate_passed"] is True or summary["recommended_action"] == "apply_patch":
        return summary
    return None


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise LocalCodnaClientError(
        "local_repository_async_context_unsupported",
        "Local Codna repository calls are synchronous; call them outside an active event loop.",
    )


def _insert_import_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
        _INSERTED_DECISION_ENGINE_IMPORT_PATHS.add(text)


def _decision_engine_import_paths(config: RuntimeConfig) -> list[Path]:
    engine_dir = config.engine_dir.expanduser().resolve()
    return [
        engine_dir,
        engine_dir / "packages" / "algenta-core",
        engine_dir / "packages" / "python-sdk",
        *_compatible_engine_site_packages(engine_dir),
    ]


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _remove_inserted_decision_engine_import_paths(config: RuntimeConfig) -> None:
    removed_paths: list[Path] = []
    for path in _decision_engine_import_paths(config):
        text = str(path)
        if text not in _INSERTED_DECISION_ENGINE_IMPORT_PATHS:
            continue
        sys.path[:] = [entry for entry in sys.path if entry != text]
        _INSERTED_DECISION_ENGINE_IMPORT_PATHS.discard(text)
        removed_paths.append(path.resolve())
    if not removed_paths:
        return
    for module_name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            origin_path = Path(origin).resolve()
        except OSError:
            continue
        if any(_path_is_relative_to(origin_path, removed) for removed in removed_paths):
            sys.modules.pop(module_name, None)


def _install_local_privacy_stub() -> None:
    module_name = "apps.api_server.services.privacy_service"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return
    module = ModuleType(module_name)

    async def record_service_egress_event(**_kwargs: Any) -> None:
        return None

    async def get_egress_policy_payload() -> dict[str, Any]:
        return {"deployment_mode": "codna_local_offline", "audit": "local_noop"}

    async def list_egress_events(**_kwargs: Any) -> dict[str, Any]:
        return {"entries": [], "total": 0, "page": 1, "limit": 0}

    async def get_privacy_report(**_kwargs: Any) -> dict[str, Any]:
        return {"entries": [], "policy": await get_egress_policy_payload()}

    module.record_service_egress_event = record_service_egress_event
    module.get_egress_policy_payload = get_egress_policy_payload
    module.list_egress_events = list_egress_events
    module.get_privacy_report = get_privacy_report
    module.CODNA_LOCAL_PRIVACY_STUB = True
    sys.modules[module_name] = module


def _ensure_decision_engine_imports(config: RuntimeConfig) -> None:
    if _explicit_algenta_engine_dir_configured():
        expected_core = (
            config.engine_dir.expanduser().resolve()
            / "apps"
            / "api_server"
            / "services"
            / "repository_intelligence_core.py"
        )
        if not expected_core.exists():
            raise LocalCodnaClientError(
                "local_repository_import_failed",
                "Explicit ALGENTA_ENGINE_DIR does not contain the local Algenta repository-intelligence SDK/core modules.",
                {
                    "engine_dir": str(config.engine_dir),
                    "reason": f"missing expected module: {expected_core}",
                    "required_backend": "packaged Algenta local repository-intelligence backend",
                    "development_override": "Unset ALGENTA_ENGINE_DIR to use the installed local SDK/package fallback.",
                    "expected_imports": [
                        "apps.api_server.models.data_connector",
                        "apps.api_server.schemas.repositories",
                        "apps.api_server.services.repository_intelligence_core",
                    ],
                },
            )
    for path in _decision_engine_import_paths(config):
        _insert_import_path(path)
    _install_local_privacy_stub()


def _import_repository_modules(config: RuntimeConfig) -> _RepositoryModules:
    _ensure_decision_engine_imports(config)
    try:
        from apps.api_server.models import data_connector
        from apps.api_server.schemas import repositories as schemas
        from apps.api_server.services import repository_intelligence_core as core
        _install_precise_patch_changed_files()
        _install_repository_module_suffix_support()
        _install_verified_agentic_diff_compatibility()
        _install_patch_artifact_newline_preservation()
    except Exception as exc:  # noqa: BLE001
        raise LocalCodnaClientError(
            "local_repository_import_failed",
            "Codna could not import the local Algenta repository-intelligence SDK/core modules.",
            {
                "engine_dir": str(config.engine_dir),
                "reason": f"{type(exc).__name__}: {exc}",
                "required_backend": "packaged Algenta local repository-intelligence backend",
                "development_override": "Set ALGENTA_ENGINE_DIR only for local development checkouts.",
                "expected_imports": [
                    "apps.api_server.models.data_connector",
                    "apps.api_server.schemas.repositories",
                    "apps.api_server.services.repository_intelligence_core",
                ],
            },
        ) from exc
    return _RepositoryModules(core=core, schemas=schemas, data_connector=data_connector)


def _explicit_algenta_engine_dir_configured() -> bool:
    return bool(os.environ.get("ALGENTA_ENGINE_DIR"))


def _can_use_packaged_backend_after_import_error(error: LocalCodnaClientError) -> bool:
    if _explicit_algenta_engine_dir_configured():
        return False
    return error.code == "local_repository_import_failed"


def _runtime_env(config: RuntimeConfig) -> dict[str, str]:
    root = config.paths.root
    return {
        **local_engine_defaults(),
        "ALGENTA_RUNTIME_DIR": str((root / "algenta-runtime").resolve()),
        "ALGENTA_REPOSITORY_INTELLIGENCE_SHARED_RUNTIME_DIR": str(
            (root / "algenta-runtime-shared").resolve()
        ),
        "ALGENTA_SOURCES_DIR": str((root / "algenta-sources").resolve()),
        "ALGENTA_SOURCE_ARTIFACT_CACHE_DIR": str((root / "algenta-source-artifact-cache").resolve()),
        "RUNTIME_FALLBACK_MODE": "deny",
    }


def _provider_key_env(keys: Mapping[str, ConfigValue] | None) -> dict[str, str]:
    if not keys:
        return {}
    blocked = set(ENGINE_URL_KEYS)
    allowed_suffixes = ("_API_KEY", "_AUTH_TOKEN")
    output: dict[str, str] = {}
    for key, config_value in keys.items():
        if key in blocked or not key.endswith(allowed_suffixes):
            continue
        if key in os.environ and os.environ[key]:
            continue
        value = config_value.value.strip()
        if value:
            output[key] = value
    return output


def _agent_core_default_env() -> dict[str, str]:
    return {
        key: os.environ.get(key) or value
        for key, value in _LOCAL_AGENT_CORE_DEFAULTS.items()
    }


@contextlib.contextmanager
def _scoped_env(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validated_connector_type(modules: _RepositoryModules, connector_type: str) -> Any:
    try:
        return modules.data_connector.ConnectorType(connector_type)
    except Exception as exc:  # noqa: BLE001
        raise LocalCodnaClientError(
            "invalid_connector_type",
            f"Connector type '{connector_type}' is not supported by local repository intelligence.",
            {"connector_type": connector_type},
        ) from exc


def _connector_from_record(modules: _RepositoryModules, record: _ConnectorRecord) -> Any:
    connector_type = _validated_connector_type(modules, record.connector_type)
    return modules.data_connector.DataConnector(
        id=uuid.UUID(record.id),
        org_id=_LOCAL_ORG_ID,
        workspace_id=None,
        name=record.name,
        description=None,
        connector_type=connector_type,
        config_json=_canonical_json(record.config),
        config_fingerprint=record.config_fingerprint,
        status=modules.data_connector.ConnectorStatus.LIVE,
        visibility=modules.data_connector.ConnectorVisibility.PRIVATE.value,
        owner_user_id=None,
        owner_key_id=None,
    )


class LocalCodnaRuntimeClient:
    """In-process Codna repository-intelligence client.

    This is the normal local product path. It calls Algenta repository-intelligence
    SDK/core functions directly with an unsaved connector object, so repository state
    is stored in local Arrow/Parquet/manifests under the configured runtime dirs and
    never routed through the FastAPI/Postgres repository API.
    """

    def __init__(self, *, keys: Mapping[str, ConfigValue] | None = None) -> None:
        self._keys = dict(keys or {})
        self._config = resolve_runtime_config(keys=self._keys)
        self._modules: _RepositoryModules | None = None
        self._packaged_backend: PackagedRepositoryBackend | None = None
        self._repository_import_error: LocalCodnaClientError | None = None
        self._step_store = RepositoryStepArtifactStore(self._config.paths.root)
        self._simulation_results: dict[str, dict[str, Any]] = {}
        _prewarm_local_mojo_runtime(self._config)

    def close(self) -> None:
        return None

    def _repository_modules(self) -> _RepositoryModules:
        if self._modules is None:
            self._modules = _import_repository_modules(self._config)
        return self._modules

    def _packaged_repository_backend(self) -> PackagedRepositoryBackend:
        if self._packaged_backend is None:
            self._packaged_backend = PackagedRepositoryBackend(
                self._config.paths.root,
                agent_runner=SidecarPackagedAgentRunner(config=self._config, keys=self._keys).run,
            )
        return self._packaged_backend

    def _repository_modules_or_packaged(self) -> _RepositoryModules | None:
        if self._packaged_backend is not None:
            return None
        try:
            return self._repository_modules()
        except LocalCodnaClientError as exc:
            if not _can_use_packaged_backend_after_import_error(exc):
                raise
            _remove_inserted_decision_engine_import_paths(self._config)
            self._repository_import_error = exc
            self._packaged_backend = PackagedRepositoryBackend(
                self._config.paths.root,
                agent_runner=SidecarPackagedAgentRunner(config=self._config, keys=self._keys).run,
            )
            return None

    def _record(self, repository_id: str) -> _ConnectorRecord:
        with _CONNECTORS_LOCK:
            record = _CONNECTORS.get(repository_id)
        if record is None:
            raise LocalCodnaClientError(
                "unknown_local_repository",
                f"Local repository connector '{repository_id}' is not registered in this process.",
                {"repository_id": repository_id},
            )
        return record

    def _connector(self, repository_id: str) -> Any:
        modules = self._repository_modules()
        return _connector_from_record(modules, self._record(repository_id))

    def _record_step_artifact(
        self,
        *,
        connector_record: _ConnectorRecord,
        operation_id: str,
        step_name: str,
        status: Literal["succeeded", "failed"],
        started_at_utc: str,
        completed_at_utc: str,
        duration_ms: float,
        request: Mapping[str, Any],
        response: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        refs = extract_step_refs(response)
        metrics = extract_step_metrics(response)
        record = RepositoryStepRecord(
            schema_version=STEP_ARTIFACT_SCHEMA_VERSION,
            artifact_type=STEP_ARTIFACT_TYPE,
            operation_id=operation_id,
            repository_id=connector_record.id,
            connector_type=connector_record.connector_type,
            connector_fingerprint=connector_record.config_fingerprint,
            step_name=step_name,
            status=status,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            duration_ms=duration_ms,
            request_sha256=sha256_json(dict(request)),
            response_sha256=sha256_json(response) if response is not None else None,
            error_code=_error_code(error) if error else None,
            error_message=_error_message(error) if error else None,
            snapshot_id=refs["snapshot_id"],
            evidence_bundle_ref=refs["evidence_bundle_ref"],
            decision_plan_id=refs["decision_plan_id"],
            simulation_id=refs["simulation_id"],
            apply_result_id=refs["apply_result_id"],
            metrics_json=canonical_json(metrics),
        )
        self._step_store.write_step(record)

    def _run_repository_step(
        self,
        repository_id: str,
        *,
        step_name: str,
        request: Mapping[str, Any],
        operation: Any,
    ) -> dict[str, Any]:
        connector_record = self._record(repository_id)
        operation_id = str(uuid.uuid4())
        started_at_utc = _utc_now()
        started = time.perf_counter()
        try:
            response = operation()
        except Exception as exc:
            completed_at_utc = _utc_now()
            duration_ms = (time.perf_counter() - started) * 1000.0
            try:
                self._record_step_artifact(
                    connector_record=connector_record,
                    operation_id=operation_id,
                    step_name=step_name,
                    status="failed",
                    started_at_utc=started_at_utc,
                    completed_at_utc=completed_at_utc,
                    duration_ms=duration_ms,
                    request=request,
                    response=None,
                    error=exc,
                )
            except RepositoryArtifactError as artifact_exc:
                raise LocalCodnaClientError(
                    "repository_step_failed_and_artifact_write_failed",
                    "Local repository-intelligence step failed and Codna could not write its Parquet artifact.",
                    {
                        "repository_id": repository_id,
                        "step_name": step_name,
                        "step_error_code": _error_code(exc),
                        "artifact_error_code": artifact_exc.code,
                        "artifact_error": str(artifact_exc),
                    },
                ) from exc
            raise
        completed_at_utc = _utc_now()
        duration_ms = (time.perf_counter() - started) * 1000.0
        try:
            self._record_step_artifact(
                connector_record=connector_record,
                operation_id=operation_id,
                step_name=step_name,
                status="succeeded",
                started_at_utc=started_at_utc,
                completed_at_utc=completed_at_utc,
                duration_ms=duration_ms,
                request=request,
                response=response,
                error=None,
            )
        except RepositoryArtifactError as exc:
            raise LocalCodnaClientError(
                "repository_step_artifact_write_failed",
                "Local repository-intelligence step completed but Codna could not write its Parquet artifact.",
                {
                    "repository_id": repository_id,
                    "step_name": step_name,
                    "artifact_error_code": exc.code,
                    "artifact_error": str(exc),
                },
            ) from exc
        return response

    @contextlib.contextmanager
    def _local_service_env(self, *, agent_core: bool = False) -> Iterator[None]:
        updates: dict[str, str | None] = {
            **dict.fromkeys(LOCAL_DATABASE_ENV_KEYS, None),
            **_runtime_env(self._config),
            **_provider_key_env(self._keys),
        }
        if agent_core:
            endpoint = ensure_agent_core_running(keys=self._keys)
            updates.update(_agent_core_default_env())
            updates["ALGENTA_AGENT_CORE_URL"] = endpoint.url
            updates["ALGENTA_ENGINE_URL"] = None
            updates["CODNA_ENGINE_URL"] = None
        with _scoped_env(updates):
            yield

    def create_connector(self, *, name: str, connector_type: str, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise LocalCodnaClientError(
                "invalid_connector_config",
                "Connector config must be a JSON object.",
                {"connector_type": connector_type},
            )
        modules = self._repository_modules_or_packaged()
        if modules is None:
            self._packaged_repository_backend().validate_connector_type(connector_type)
        else:
            _validated_connector_type(modules, connector_type)
        connector_id = str(uuid.uuid4())
        record = _ConnectorRecord(
            id=connector_id,
            name=name,
            connector_type=connector_type,
            config=dict(config),
            config_fingerprint=_fingerprint_config(config),
        )
        with _CONNECTORS_LOCK:
            _CONNECTORS[connector_id] = record
        return {
            "id": connector_id,
            "name": name,
            "connector_type": connector_type,
            "status": "live",
            "local": True,
        }

    def get_repository_intelligence_capabilities(self) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._packaged_repository_backend().get_repository_intelligence_capabilities()
        with self._local_service_env():
            return _dump(modules.core.get_repository_intelligence_capabilities())

    def create_repository_snapshot(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="snapshot",
                request=request,
                operation=lambda: self._packaged_repository_backend().create_repository_snapshot(
                    connector_record=self._record(repository_id),
                    request=request,
                ),
            )
        typed_request = modules.schemas.RepositorySnapshotCreateRequest.model_validate(dict(request))
        return self._run_repository_step(
            repository_id,
            step_name="snapshot",
            request=request,
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: modules.core.create_repository_snapshot(
                        connector=self._connector(repository_id),
                        request=typed_request,
                    )
                )
            ),
        )

    def get_repository_snapshot(self, repository_id: str, snapshot_id: str) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="get_snapshot",
                request={"snapshot_id": snapshot_id},
                operation=lambda: self._packaged_repository_backend().get_repository_snapshot(
                    repository_id=repository_id,
                    snapshot_id=snapshot_id,
                ),
            )
        return self._run_repository_step(
            repository_id,
            step_name="get_snapshot",
            request={"snapshot_id": snapshot_id},
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: modules.core.get_repository_snapshot(
                        connector=self._connector(repository_id),
                        snapshot_id=snapshot_id,
                    )
                )
            ),
        )

    def triage_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="triage",
                request=request,
                operation=lambda: self._packaged_repository_backend().triage_repository(
                    repository_id=repository_id,
                    request=request,
                ),
            )
        typed_request = modules.schemas.RepositoryTriageRequest.model_validate(dict(request))
        return self._run_repository_step(
            repository_id,
            step_name="triage",
            request=request,
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: modules.core.triage_repository(
                        connector=self._connector(repository_id),
                        request=typed_request,
                    )
                )
            ),
        )

    def query_repository_graph(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="graph_query",
                request=request,
                operation=lambda: self._packaged_repository_backend().query_repository_graph(
                    repository_id=repository_id,
                    request=request,
                ),
            )
        typed_request = modules.schemas.RepositoryGraphQueryRequest.model_validate(dict(request))
        return self._run_repository_step(
            repository_id,
            step_name="graph_query",
            request=request,
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: modules.core.query_repository_graph(
                        connector=self._connector(repository_id),
                        request=typed_request,
                    )
                )
            ),
        )

    def create_repository_decision_plan(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="decision_plan",
                request=request,
                operation=lambda: self._packaged_repository_backend().create_repository_decision_plan(
                    repository_id=repository_id,
                    request=request,
                ),
            )
        typed_request = modules.schemas.RepositoryDecisionPlanCreateRequest.model_validate(dict(request))
        request_model = str(dict(request).get("model") or "")
        needs_agent_core = request_model == _VERIFIED_AGENTIC_MODEL
        return self._run_repository_step(
            repository_id,
            step_name="decision_plan",
            request=request,
            operation=lambda: _dump(
                (
                    self._call_agent_core_with_transport_recovery
                    if needs_agent_core
                    else self._call_with_local_env
                )(
                    lambda: _run_async(
                        modules.core.create_repository_decision_plan(
                            connector=self._connector(repository_id),
                            request=typed_request,
                        )
                    )
                )
            ),
        )

    def _call_with_local_env(
        self,
        callback: Any,
        *,
        agent_core: bool = False,
        mojo_pool: bool = False,
    ) -> Any:
        with self._local_service_env(agent_core=agent_core):
            if mojo_pool:
                _ensure_local_mojo_runtime(self._config)
                _publish_local_mojo_runtime_state(self._config)
            return callback()

    def _recover_agent_core_transport(self, error: BaseException) -> None:
        try:
            stop_agent_core_runtime(keys=self._keys)
        except Exception as recovery_exc:  # noqa: BLE001
            raise LocalCodnaClientError(
                "agent_core_transport_recovery_failed",
                "Codna lost its local agent-core connection and could not restart the owned sidecar.",
                {
                    "original_error_code": _error_code(error),
                    "original_error": _error_message(error),
                    "recovery_error_code": _error_code(recovery_exc),
                    "recovery_error": _error_message(recovery_exc),
                    "sidecar_url": self._config.sidecar_url,
                },
            ) from recovery_exc

    def _call_agent_core_with_transport_recovery(self, callback: Any) -> Any:
        try:
            return self._call_with_local_env(callback, agent_core=True)
        except Exception as error:  # noqa: BLE001
            if not _is_agent_core_transport_error(error):
                raise
            self._recover_agent_core_transport(error)
            first_error = error
        try:
            return self._call_with_local_env(callback, agent_core=True)
        except Exception as retry_error:  # noqa: BLE001
            if not _is_agent_core_transport_error(retry_error):
                raise
            raise LocalCodnaClientError(
                "agent_core_transport_recovery_exhausted",
                "Codna lost its local agent-core connection again after one owned-runtime restart.",
                {
                    "first_error_code": _error_code(first_error),
                    "first_error": _error_message(first_error),
                    "retry_error_code": _error_code(retry_error),
                    "retry_error": _error_message(retry_error),
                    "sidecar_url": self._config.sidecar_url,
                },
            ) from retry_error

    def simulate_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="simulate",
                request=request,
                operation=lambda: self._packaged_repository_backend().simulate_repository(
                    repository_id=repository_id,
                    request=request,
                ),
            )
        typed_request = modules.schemas.RepositorySimulationRequest.model_validate(dict(request))
        response = self._run_repository_step(
            repository_id,
            step_name="simulate",
            request=request,
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: _run_async(
                        modules.core.simulate_repository_decision(
                            connector=self._connector(repository_id),
                            request=typed_request,
                        )
                    ),
                    mojo_pool=True,
                )
            ),
        )
        simulation_id = _simulation_id_from_response(response)
        if simulation_id:
            self._simulation_results[simulation_id] = dict(response)
        return response

    def _enforce_local_simulation_gate(
        self,
        repository_id: str,
        request: Mapping[str, Any],
    ) -> None:
        if request.get("mode") == "patch_only":
            return
        simulation_id = request.get("simulation_id")
        if not isinstance(simulation_id, str) or not simulation_id:
            return
        simulation = self._simulation_results.get(simulation_id)
        if simulation is None:
            return
        violation = _simulation_requires_fail_closed(simulation)
        if violation is None:
            return
        raise LocalCodnaClientError(
            "repository_apply_gate_failed",
            "Codna blocked apply because local Monte Carlo risk metrics exceed the fail-closed threshold.",
            {
                "repository_id": repository_id,
                "simulation_id": simulation_id,
                "mode": request.get("mode"),
                "risk_threshold": _SIMULATION_FAIL_CLOSED_RISK_THRESHOLD,
                **violation,
            },
        )

    def apply_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        modules = self._repository_modules_or_packaged()
        if modules is None:
            return self._run_repository_step(
                repository_id,
                step_name="apply",
                request=request,
                operation=lambda: self._packaged_repository_backend().apply_repository(
                    repository_id=repository_id,
                    request=request,
                ),
            )
        request_payload = dict(request)
        typed_request = modules.schemas.RepositoryApplyRequest.model_validate(request_payload)
        return self._run_repository_step(
            repository_id,
            step_name="apply",
            request=request,
            operation=lambda: _dump(
                self._call_with_local_env(
                    lambda: (
                        self._enforce_local_simulation_gate(repository_id, request_payload)
                        or _run_async(
                            modules.core.apply_repository_decision(
                                connector=self._connector(repository_id),
                                request=typed_request,
                            )
                        )
                    )
                )
            ),
        )
