from __future__ import annotations

from typing import Any, Mapping

from .local_client import LocalCodnaRuntimeClient

_GET = "GET"
_POST = "POST"


class CodnaRepositoryBackend:
    """Algenta local Runtime backend for Codna repository intelligence.

    This backend is intentionally in-process. It does not call Codna's FastAPI
    router and does not require local Postgres; Algenta repository-intelligence
    services persist their own local manifests and Arrow/Parquet artifacts.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._local_client: LocalCodnaRuntimeClient | None = None

    def _local(self) -> LocalCodnaRuntimeClient:
        if self._local_client is None:
            self._local_client = LocalCodnaRuntimeClient()
        return self._local_client

    def _req(self, method: str, path: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._client is not None:
            if request is None:
                return self._client._request(method, path)
            return self._client._request(method, path, json=dict(request))
        raise RuntimeError("HTTP compatibility client is not configured.")

    def get_repository_intelligence_capabilities(self) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_GET, "/v1/repositories/capabilities")
        return self._local().get_repository_intelligence_capabilities()

    def create_repository_snapshot(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/snapshots", request)
        return self._local().create_repository_snapshot(repository_id, dict(request))

    def get_repository_snapshot(self, repository_id: str, snapshot_id: str) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_GET, f"/v1/repositories/{repository_id}/snapshots/{snapshot_id}")
        return self._local().get_repository_snapshot(repository_id, snapshot_id)

    def triage_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/triage", request)
        return self._local().triage_repository(repository_id, dict(request))

    def create_repository_decision_plan(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/decision-plans", request)
        return self._local().create_repository_decision_plan(repository_id, dict(request))

    def query_repository_graph(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/graph-query", request)
        return self._local().query_repository_graph(repository_id, dict(request))

    def simulate_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/simulate", request)
        return self._local().simulate_repository(repository_id, dict(request))

    def apply_repository(self, repository_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return self._req(_POST, f"/v1/repositories/{repository_id}/apply", request)
        return self._local().apply_repository(repository_id, dict(request))


def build_backend() -> CodnaRepositoryBackend:
    """Zero-arg factory for ``codna.repo_backend:build_backend``."""
    return CodnaRepositoryBackend()
