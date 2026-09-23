"""Engine patch-generator tests (the remediation seam): finding_issue rendering + diff fetch."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from codna.patchgen import ClinePatchGenerator, EnginePatchGenerator, PatchGenError, finding_issue
from codna.runtime.config import ConfigValue
from codna.sarif import ingest_sarif

DIFF = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x\n+safe(x)\n"


def _finding():
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "js/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}]}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": [{"ruleId": "js/sqli", "message": {"text": "tainted query"},
                         "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.py"}, "region": {"startLine": 12}}}],
                         "codeFlows": [{"threadFlows": [{"locations": [
                             {"location": {"physicalLocation": {"artifactLocation": {"uri": "src/in.py"}, "region": {"startLine": 3}}}},
                             {"location": {"physicalLocation": {"artifactLocation": {"uri": "src/app.py"}, "region": {"startLine": 12}}}}]}]}]}],
        }],
    }
    return ingest_sarif(json.dumps(doc)).findings[0]


def test_finding_issue_renders_obligation():
    issue = finding_issue(_finding())
    assert "js/sqli" in issue
    assert "src/app.py:12" in issue
    assert "Dataflow: src/in.py:3 -> src/app.py:12" in issue
    assert "Do NOT modify tests" in issue


class StubPost:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def __call__(self, path, body):
        self.calls.append((path, body))
        return self.resp


def test_generator_returns_diff_and_sends_obligation():
    post = StubPost({"patch_diff": DIFF})
    gen = EnginePatchGenerator(post, repository_id="r1", snapshot_id="s1")
    out = gen(_finding(), "/tmp/wt")
    assert out == DIFF
    path, body = post.calls[0]
    assert path == "/v1/repositories/r1/decision-plan"
    assert "js/sqli" in body["signals"]["issue_text"]
    assert body["signals"]["security_finding"]["rule_id"] == "js/sqli"


def test_generator_nested_patch_diff_supported():
    gen = EnginePatchGenerator(StubPost({"decision_plan": {"patch_diff": DIFF}}),
                               repository_id="r1", snapshot_id="s1")
    assert gen(_finding(), "/tmp/wt") == DIFF


def test_generator_raises_when_no_patch():
    gen = EnginePatchGenerator(StubPost({"decision_plan": {}}), repository_id="r1", snapshot_id="s1")
    with pytest.raises(PatchGenError):
        gen(_finding(), "/tmp/wt")


def test_cline_generator_uses_packaged_sidecar_runner(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    source = "\n".join(f"line {idx}" for idx in range(1, 16)) + "\n"
    (repo / "src" / "app.py").write_text(source, encoding="utf-8")
    captured = {}

    class FakeRunner:
        def __init__(self, *, config, keys):
            captured["config"] = config
            captured["keys"] = keys

        def run(self, request):
            captured["request"] = request
            captured["turns"] = __import__("os").environ.get("ALGENTA_AGENT_MAX_TURNS")
            captured["timeout"] = __import__("os").environ.get("ALGENTA_AGENT_TURN_TIMEOUT_MS")
            return SimpleNamespace(patch_diff=DIFF)

    monkeypatch.setattr("codna.packaged_agent_runner.SidecarPackagedAgentRunner", FakeRunner)
    monkeypatch.setattr(
        "codna.runtime.config.resolve_runtime_config",
        lambda *, keys=None: SimpleNamespace(paths=tmp_path, keys_seen=keys),
    )

    keys = {"ANTHROPIC_API_KEY": ConfigValue("ANTHROPIC_API_KEY", "sk-test", "keychain")}
    gen = ClinePatchGenerator(model="claude-sonnet-4-6", max_iterations=7, timeout_s=12, keys=keys)
    assert gen(_finding(), str(repo)) == DIFF

    request = captured["request"]
    assert captured["config"].keys_seen == keys
    assert captured["keys"] == keys
    assert request.model == "anthropic/claude-sonnet-4-6"
    assert request.repo_root == repo.resolve()
    assert request.evidence_bundle["suspect_files"] == ["src/app.py"]
    assert "12: line 12" in request.evidence_bundle["evidence_items"][0]["snippet"]
    assert captured["turns"] == "7"
    assert captured["timeout"] == "12000"
