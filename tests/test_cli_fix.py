"""`codna fix` error-surfacing tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from codna import cli as cli_module
from codna.cli import cmd_fix, CodnaError
from codna.cli_errors import format_cli_error


class FakeEngineError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
        details=None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.details = details
        self.status_code = status_code


def test_format_cli_error_renders_structured_fields():
    err = FakeEngineError(
        "Generated patch targets are not present in snapshot.",
        code="generated_patch_targets_missing",
        request_id="req_123",
        details={"missing_targets": ["find_sax_parse.sh"]},
        status_code=422,
    )
    text = format_cli_error(err)
    assert "Generated patch targets are not present in snapshot." in text
    assert "code=generated_patch_targets_missing" in text
    assert "request_id=req_123" in text
    assert "status=422" in text
    assert '"missing_targets": ["find_sax_parse.sh"]' in text


def test_cmd_fix_surfaces_structured_engine_failure(monkeypatch):
    class FailingClient:
        def triage_repository(self, *_args, **_kwargs):
            raise FakeEngineError(
                "Generated patch targets are not present in snapshot.",
                code="generated_patch_targets_missing",
                request_id="req_123",
                details={"missing_targets": ["find_sax_parse.sh"]},
                status_code=422,
            )

    monkeypatch.setattr("codna.cli._client", lambda **_kwargs: FailingClient())
    monkeypatch.setattr("codna.cli._register", lambda *args, **kwargs: ("repo_1", {"snapshot_id": "snap_1"}))
    monkeypatch.setattr("codna.cli._dump", lambda value: value)
    args = SimpleNamespace(
        repo=".",
        issue="broken parser",
        failing_test=[],
        from_junit=None,
        ref=None,
        model="repository.verified_agentic_v1",
        apply=False,
        open_pr=False,
        github_token=None,
        base_branch=None,
        pr_title=None,
        pr_body=None,
        memory="off",
    )

    # main's _die raises CodnaError (main() prints it + exits 1); cmd_fix surfaces the structured error.
    with pytest.raises(CodnaError) as excinfo:
        cmd_fix(args)

    msg = str(excinfo.value)
    assert "fix failed:" in msg
    assert "code=generated_patch_targets_missing" in msg
    assert "request_id=req_123" in msg
    assert '"missing_targets": ["find_sax_parse.sh"]' in msg


def test_cmd_fix_raises_when_apply_fails(monkeypatch):
    class FailingApplyClient:
        def triage_repository(self, *_args, **_kwargs):
            return {"workspace_evidence_bundle_ref": "bundle-1"}

        def create_repository_decision_plan(self, *_args, **_kwargs):
            return {
                "decision_plan_id": "plan-1",
                "planner_usage": {"input_tokens": 1, "output_tokens": 1},
                "decision_plan": {
                    "confidence": 0.7,
                    "repository_analysis": {
                        "root_cause": "bug",
                        "generated_patch_ref": "patch-ref-1",
                    },
                },
            }

        def simulate_repository(self, *_args, **_kwargs):
            return {"validated_inputs": {"simulation_id": "sim-1"}}

        def apply_repository(self, *_args, **_kwargs):
            raise FakeEngineError(
                "blocked by apply gate",
                code="repository_apply_gate_failed",
                details={"simulation_id": "sim-1"},
            )

    monkeypatch.setattr("codna.cli._client", lambda **_kwargs: FailingApplyClient())
    monkeypatch.setattr("codna.cli._register", lambda *args, **kwargs: ("repo_1", {"snapshot_id": "snap_1"}))
    monkeypatch.setattr("codna.cli._dump", lambda value: value)
    args = SimpleNamespace(
        repo=".",
        issue="broken parser",
        failing_test=[],
        from_junit=None,
        ref=None,
        model="repository.verified_agentic_v1",
        apply=True,
        open_pr=False,
        github_token=None,
        base_branch=None,
        pr_title=None,
        pr_body=None,
        memory="off",
    )

    with pytest.raises(CodnaError) as excinfo:
        cmd_fix(args)

    msg = str(excinfo.value)
    assert "apply failed:" in msg
    assert "code=repository_apply_gate_failed" in msg
    assert "patch ref: patch-ref-1" in msg


def test_cmd_fix_open_pr_uses_remote_pr_apply(monkeypatch, capsys):
    captured = {}

    class OpenPrClient:
        def triage_repository(self, *_args, **_kwargs):
            return {"workspace_evidence_bundle_ref": "bundle-1"}

        def create_repository_decision_plan(self, *_args, **_kwargs):
            return {
                "decision_plan_id": "plan-1",
                "planner_usage": {"input_tokens": 3, "output_tokens": 2},
                "decision_plan": {
                    "confidence": 0.8,
                    "repository_analysis": {
                        "root_cause": "bug",
                        "generated_patch_ref": "patch-ref-1",
                    },
                },
            }

        def simulate_repository(self, *_args, **_kwargs):
            return {"validated_inputs": {"simulation_id": "sim-1"}}

        def apply_repository(self, repository_id, request):
            captured["repository_id"] = repository_id
            captured["apply_request"] = dict(request)
            return {"pull_request_url": "https://github.com/owner/repo/pull/123"}

    def fake_register(_client, repo, ref, github_token=None, focus_paths=None):
        captured["register"] = {
            "repo": repo,
            "ref": ref,
            "github_token": github_token,
            "focus_paths": focus_paths,
        }
        return "repo_1", {"snapshot_id": "snap_1"}

    monkeypatch.setattr("codna.cli._client", lambda **_kwargs: OpenPrClient())
    monkeypatch.setattr("codna.cli._register", fake_register)
    monkeypatch.setattr("codna.cli._dump", lambda value: value)
    args = SimpleNamespace(
        repo="https://github.com/owner/repo.git",
        issue="broken parser",
        failing_test=[],
        from_junit=None,
        ref="abc123",
        model="repository.verified_agentic_v1",
        apply=False,
        open_pr=True,
        github_token="ghp_test",
        base_branch="main",
        pr_title="codna: fix parser",
        pr_body="body",
        memory="off",
    )

    cmd_fix(args)

    assert captured["register"] == {
        "repo": "https://github.com/owner/repo.git",
        "ref": "abc123",
        "github_token": "ghp_test",
        "focus_paths": [],
    }
    assert captured["repository_id"] == "repo_1"
    assert captured["apply_request"] == {
        "mode": "remote_pr",
        "write_permission": True,
        "decision_plan_id": "plan-1",
        "simulation_id": "sim-1",
        "base_branch": "main",
        "pull_request_title": "codna: fix parser",
        "pull_request_body": "body",
    }
    assert "opened pull request: https://github.com/owner/repo/pull/123" in capsys.readouterr().out


def test_http_client_create_connector_uses_engine_http_contract(monkeypatch):
    class FakeHttpClient:
        instances = []

        def __init__(self, *, base_url, timeout, headers):
            self.base_url = base_url
            self.timeout = timeout
            self.headers = headers
            self.requests = []
            self.instances.append(self)

        def request(self, method, path, *, json=None, params=None):
            self.requests.append({"method": method, "path": path, "json": json, "params": params})
            return cli_module.httpx.Response(200, json={"id": "conn_1"})

    monkeypatch.setattr(cli_module.httpx, "Client", FakeHttpClient)

    client = cli_module._HttpCodnaClient(
        api_key="codna_test_key",
        base_url="https://api.codna.ai/",
        timeout=12.0,
        max_retries=0,
    )
    result = client.create_connector(
        name="codna-test",
        connector_type="github_repo",
        config={"repository_url": "https://github.com/owner/repo.git"},
    )

    fake = FakeHttpClient.instances[0]
    assert result == {"id": "conn_1"}
    assert fake.base_url == "https://api.codna.ai"
    assert fake.timeout == 12.0
    assert fake.headers["Authorization"] == "Bearer codna_test_key"
    assert fake.requests == [{
        "method": "POST",
        "path": "/v1/connectors",
        "json": {
            "name": "codna-test",
            "connector_type": "github_repo",
            "config": {"repository_url": "https://github.com/owner/repo.git"},
        },
        "params": None,
    }]


def test_http_client_missing_base_url_points_to_local_runtime_diagnostics():
    with pytest.raises(CodnaError) as excinfo:
        cli_module._HttpCodnaClient(
            api_key="codna_test_key",
            base_url="",
            timeout=12.0,
            max_retries=0,
        )

    msg = str(excinfo.value)
    assert "no engine URL resolved" in msg
    assert "codna doctor --start-stack" in msg
    assert "set CODNA_ENGINE_URL" not in msg


def test_http_client_surfaces_structured_engine_error(monkeypatch):
    class FakeHttpClient:
        def __init__(self, **_kwargs):
            pass

        def request(self, method, path, *, json=None, params=None):
            return cli_module.httpx.Response(
                422,
                json={
                    "error": {
                        "code": "bad_request",
                        "message": "snapshot_id is required",
                        "details": {"field": "snapshot_id"},
                    }
                },
                headers={"x-request-id": "req_123"},
            )

    monkeypatch.setattr(cli_module.httpx, "Client", FakeHttpClient)
    client = cli_module._HttpCodnaClient(
        api_key="codna_test_key",
        base_url="https://api.codna.ai",
        timeout=12.0,
        max_retries=0,
    )

    with pytest.raises(CodnaError) as excinfo:
        client.triage_repository("repo_1", {})

    msg = str(excinfo.value)
    assert "status=422" in msg
    assert "snapshot_id is required" in msg
    assert "request_id=req_123" in msg
    assert '"field": "snapshot_id"' in msg
    assert "codna_test_key" not in msg


def test_remote_http_timeout_defaults_to_bounded_connect_timeout(monkeypatch):
    monkeypatch.delenv("CODNA_HTTP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CODNA_TIMEOUT_S", raising=False)
    monkeypatch.delenv("CODNA_HTTP_CONNECT_TIMEOUT_S", raising=False)

    timeout = cli_module._remote_http_timeout()

    assert timeout.connect == 5.0
    assert timeout.read == 30.0
    assert timeout.write == 5.0
    assert timeout.pool == 5.0


def test_remote_http_timeout_honors_legacy_total_and_connect_override(monkeypatch):
    monkeypatch.delenv("CODNA_HTTP_TIMEOUT_S", raising=False)
    monkeypatch.setenv("CODNA_TIMEOUT_S", "120")
    monkeypatch.setenv("CODNA_HTTP_CONNECT_TIMEOUT_S", "7")

    timeout = cli_module._remote_http_timeout()

    assert timeout.connect == 7.0
    assert timeout.read == 120.0


def test_remote_http_timeout_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("CODNA_HTTP_TIMEOUT_S", "nope")

    with pytest.raises(CodnaError, match="must be a number"):
        cli_module._remote_http_timeout()

    monkeypatch.setenv("CODNA_HTTP_TIMEOUT_S", "10")
    monkeypatch.setenv("CODNA_HTTP_CONNECT_TIMEOUT_S", "11")
    with pytest.raises(CodnaError, match="request timeout"):
        cli_module._remote_http_timeout()
