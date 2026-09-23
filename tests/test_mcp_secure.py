from __future__ import annotations

import json

import pytest

from codna import cli, mcp_server


@pytest.fixture(autouse=True)
def _authorized_device(monkeypatch):
    """The uniform execution gate requires the free `codna login`; run these tests authorized."""
    monkeypatch.setenv("CODNA_API_KEY", "test-device-login")


def _write_sarif(path):
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0", "rules": [{"id": "js/eval"}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://example.test/o/r"}],
            "results": [{
                "ruleId": "js/eval",
                "message": {"text": "eval"},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.js"}, "region": {"startLine": 1}}}],
            }],
        }],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_mcp_secure_uses_local_reference_engine_without_runtime(monkeypatch, tmp_path):
    sarif = tmp_path / "result.sarif"
    _write_sarif(sarif)

    monkeypatch.setattr(cli, "_engine_url_key", lambda: (_ for _ in ()).throw(AssertionError("must not start runtime")))
    monkeypatch.setattr(cli, "_client", lambda: (_ for _ in ()).throw(AssertionError("must not register repo")))

    payload = json.loads(mcp_server.secure_json("/repo", str(sarif)))

    assert payload["repository"] == "/repo"
    assert payload["counts"] == {"production-reachable": 1}
    assert payload["autofix_eligible"] == 0
    assert payload["findings"][0]["classification"] == "production-reachable"
