"""Codna local repository-intelligence backend compatibility tests."""
from __future__ import annotations

import pytest

from codna import cli as cli_module
from codna.cli import CodnaError
from codna.repo_backend import CodnaRepositoryBackend


def test_repo_backend_uses_internal_http_client_json_contract(monkeypatch):
    class FakeHttpClient:
        instances = []

        def __init__(self, **_kwargs):
            self.requests = []
            self.instances.append(self)

        def request(self, method, path, *, json=None, params=None):
            self.requests.append({"method": method, "path": path, "json": json, "params": params})
            return cli_module.httpx.Response(200, json={"suspect_files": ["src/app.py"]})

    monkeypatch.setattr(cli_module.httpx, "Client", FakeHttpClient)
    client = cli_module._HttpCodnaClient(
        api_key="codna_test_key",
        base_url="https://api.codna.ai",
        timeout=12.0,
        max_retries=0,
    )

    result = CodnaRepositoryBackend(client=client).triage_repository(
        "repo_1",
        {"snapshot_id": "snap_1", "signals": {"issue_text": "broken"}},
    )

    assert result == {"suspect_files": ["src/app.py"]}
    assert FakeHttpClient.instances[0].requests == [{
        "method": "POST",
        "path": "/v1/repositories/repo_1/triage",
        "json": {"snapshot_id": "snap_1", "signals": {"issue_text": "broken"}},
        "params": None,
    }]


def test_http_client_rejects_ambiguous_json_keywords(monkeypatch):
    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(cli_module.httpx, "Client", FakeHttpClient)
    client = cli_module._HttpCodnaClient(
        api_key="codna_test_key",
        base_url="https://api.codna.ai",
        timeout=12.0,
        max_retries=0,
    )

    with pytest.raises(CodnaError, match="both json_payload and json"):
        client._request("POST", "/v1/test", json_payload={"a": 1}, json={"b": 2})
