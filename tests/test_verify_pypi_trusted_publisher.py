from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_pypi_trusted_publisher.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_pypi_trusted_publisher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self, *args):
        return json.dumps(self._payload).encode("utf-8")


def _jwt(payload: dict[str, object]) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return ".".join(
        [
            encode({"alg": "RS256", "typ": "JWT"}),
            encode(payload),
            "signature",
        ]
    )


def test_missing_github_oidc_env_fails_structured():
    tool = _load_script()

    with pytest.raises(tool.TrustedPublisherProbeError) as excinfo:
        tool.load_github_oidc_config({})

    assert excinfo.value.code == "missing_github_oidc_env"
    assert excinfo.value.details == {
        "missing": ["ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    }


def test_github_oidc_url_replaces_existing_audience():
    tool = _load_script()

    url = tool._request_url_with_audience(
        "https://token.actions.githubusercontent.com?id=1&audience=old",
        "pypi",
    )

    assert url == "https://token.actions.githubusercontent.com?id=1&audience=pypi"


def test_trusted_publisher_probe_succeeds_without_secret_leakage(monkeypatch):
    tool = _load_script()
    requests = []
    oidc_token = _jwt(
        {
            "aud": "pypi",
            "environment": "pypi",
            "ref": "refs/heads/main",
            "repository": "thyn-ai/codna",
            "repository_owner": "thyn-ai",
            "sha": "abc123",
            "sub": "repo:thyn-ai/codna:environment:pypi",
            "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
        }
    )

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if len(requests) == 1:
            assert request.get_header("Authorization") == "Bearer github-request-secret"
            assert request.full_url.endswith("audience=pypi")
            return FakeResponse({"value": oidc_token})
        assert json.loads(request.data.decode("utf-8")) == {"token": oidc_token}
        return FakeResponse({"token": "pypi" + "-upload-secret"})

    monkeypatch.setattr(tool, "urlopen", fake_urlopen)

    result = tool.verify_trusted_publisher(
        {
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.actions.githubusercontent.com?id=1",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "github-request-secret",
        }
    )

    rendered = json.dumps(result, sort_keys=True)
    assert result == {
        "ok": True,
        "audience": "pypi",
        "mint_url": "https://pypi.org/_/oidc/mint-token",
        "token_received": True,
        "upload_attempted": False,
    }
    assert "github-request-secret" not in rendered
    assert oidc_token not in rendered
    assert "pypi" + "-upload-secret" not in rendered
    assert len(requests) == 2


def test_pypi_mint_failure_reports_status_without_oidc_token(monkeypatch):
    tool = _load_script()
    calls = 0
    oidc_token = _jwt(
        {
            "aud": "pypi",
            "environment": "pypi",
            "ref": "refs/heads/main",
            "repository": "thyn-ai/codna",
            "repository_owner": "thyn-ai",
            "sha": "abc123",
            "sub": "repo:thyn-ai/codna:environment:pypi",
            "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
        }
    )

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse({"value": oidc_token})
        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            hdrs=None,
            fp=io.BytesIO(
                b'{"errors":[{"code":"invalid-publisher","description":"repo mismatch"}],'
                b'"message":"Token request failed"}'
            ),
        )

    monkeypatch.setattr(tool, "urlopen", fake_urlopen)

    with pytest.raises(tool.TrustedPublisherProbeError) as excinfo:
        tool.verify_trusted_publisher(
            {
                "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.actions.githubusercontent.com?id=1",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "github-request-secret",
            }
        )

    assert excinfo.value.code == "pypi_mint_http_error"
    assert excinfo.value.details["status"] == 422
    assert excinfo.value.details["body_omitted"] is True
    assert excinfo.value.details["body_bytes_observed"] > 0
    assert excinfo.value.details["pypi_error_codes"] == ["invalid-publisher"]
    assert excinfo.value.details["github_oidc_claims"] == {
        "aud": "pypi",
        "environment": "pypi",
        "ref": "refs/heads/main",
        "repository": "thyn-ai/codna",
        "repository_owner": "thyn-ai",
        "sha": "abc123",
        "sub": "repo:thyn-ai/codna:environment:pypi",
        "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
    }
    assert excinfo.value.details["remediation"] == {
        "reason": "PyPI did not find a trusted publisher matching this GitHub OIDC identity.",
        "pypi_expected_publisher": {
            "project_name": "codna",
            "owner": "thyn-ai",
            "repository_name": "codna",
            "workflow_filename": "publish-cli.yml",
            "environment_name": "pypi",
        },
        "github_claims_to_compare": {
            "repository": "thyn-ai/codna",
            "repository_owner": "thyn-ai",
            "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
            "environment": "pypi",
            "sub": "repo:thyn-ai/codna:environment:pypi",
            "ref": "refs/heads/main",
            "sha": "abc123",
        },
        "safe_rerun_command": (
            "gh workflow run publish-cli.yml --repo thyn-ai/codna --ref main "
            "-f publish=false -f verify_pypi_oidc=true"
        ),
    }
    assert "github-request-secret" not in repr(excinfo.value)
    assert oidc_token not in repr(excinfo.value)
    assert "repo mismatch" not in repr(excinfo.value)
    assert "Token request failed" not in repr(excinfo.value)


def test_safe_github_oidc_claims_whitelists_non_secret_claims():
    tool = _load_script()
    oidc_token = _jwt(
        {
            "aud": "pypi",
            "environment": "pypi",
            "repository": "thyn-ai/codna",
            "repository_id": 123,
            "secret_like_claim": "must-not-print",
            "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
        }
    )

    claims = tool.safe_github_oidc_claims(oidc_token)

    assert claims == {
        "aud": "pypi",
        "environment": "pypi",
        "repository": "thyn-ai/codna",
        "repository_id": "123",
        "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
    }


def test_pypi_error_code_parser_ignores_untrusted_descriptions():
    tool = _load_script()

    codes = tool._safe_pypi_error_codes(
        b'{"errors":['
        b'{"code":"invalid-pending-publisher","description":"secret detail"},'
        b'{"code":"unexpected","description":"secret detail"}'
        b']}'
    )

    assert codes == ["invalid-pending-publisher"]


def test_trusted_publisher_remediation_only_for_publisher_mismatch():
    tool = _load_script()
    claims = {
        "environment": "pypi",
        "repository": "thyn-ai/codna",
        "repository_owner": "thyn-ai",
        "sub": "repo:thyn-ai/codna:environment:pypi",
        "workflow_ref": "thyn-ai/codna/.github/workflows/publish-cli.yml@refs/heads/main",
    }

    mismatch = tool.trusted_publisher_remediation(
        github_claims=claims,
        pypi_error_codes=["invalid-publisher"],
    )
    non_mismatch = tool.trusted_publisher_remediation(
        github_claims=claims,
        pypi_error_codes=["invalid-token"],
    )

    assert mismatch["pypi_expected_publisher"] == {
        "project_name": "codna",
        "owner": "thyn-ai",
        "repository_name": "codna",
        "workflow_filename": "publish-cli.yml",
        "environment_name": "pypi",
    }
    assert non_mismatch is None


def test_main_never_prints_minted_token(monkeypatch, capsys):
    tool = _load_script()

    monkeypatch.setattr(
        tool,
        "verify_trusted_publisher",
        lambda env: {
            "ok": True,
            "audience": "pypi",
            "mint_url": "https://pypi.org/_/oidc/mint-token",
            "token_received": True,
            "upload_attempted": False,
        },
    )

    assert tool.main() == 0
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert "pypi" + "-upload-secret" not in output
    assert "github-oidc-secret" not in output
