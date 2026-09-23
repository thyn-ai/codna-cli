from __future__ import annotations

from base64 import b64decode
import json
import subprocess
from pathlib import Path

import pytest

from codna import local_client as local_client_module
from codna.local_client import LocalCodnaClientError, LocalCodnaRuntimeClient
from codna.packaged_git import PackagedGitError, _git_auth_env, _validated_ref, materialize_repository
from codna.packaged_repository_advanced import PackagedAgentRunResult


def test_packaged_backend_clones_github_repo_connector(tmp_path, monkeypatch) -> None:
    remote = _bare_remote_with_app(tmp_path, "return 1\n")
    _force_packaged_backend(tmp_path, monkeypatch)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="remote-test",
        connector_type="github_repo",
        config={"repository_url": str(remote), "repository_slug": "owner/repo"},
    )
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "app return value bug"}},
    )

    assert snapshot["connector_type"] == "github_repo"
    assert snapshot["repository_url"] == str(remote)
    assert snapshot["repository_slug"] == "owner/repo"
    assert "app.py" in triage["suspect_files"]


def test_packaged_backend_remote_pr_pushes_branch_and_opens_pr(tmp_path, monkeypatch) -> None:
    remote = _bare_remote_with_app(tmp_path, "return 1\n")
    _force_packaged_backend(tmp_path, monkeypatch)
    _fake_agent_patch(monkeypatch, "return 1\n", "return 2\n")
    opened: list[dict[str, str]] = []

    def fake_open_pr(**kwargs):
        opened.append(dict(kwargs))
        return "https://github.com/owner/repo/pull/123"

    monkeypatch.setattr("codna.packaged_git.create_github_pull_request", fake_open_pr)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="remote-test",
        connector_type="github_repo",
        config={
            "repository_url": str(remote),
            "repository_slug": "owner/repo",
            "access_token": "test-write-token",
        },
    )
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "app return value bug"}},
    )
    plan = client.create_repository_decision_plan(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
            "signals": {"issue_text": "app return value bug"},
            "model": "repository.verified_agentic_v1",
        },
    )
    simulation = client.simulate_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "decision_plan_id": plan["decision_plan_id"]},
    )
    result = client.apply_repository(
        connector["id"],
        {
            "mode": "remote_pr",
            "write_permission": True,
            "decision_plan_id": plan["decision_plan_id"],
            "simulation_id": simulation["validated_inputs"]["simulation_id"],
            "pull_request_title": "codna: fix app return",
            "pull_request_body": "body",
        },
    )

    assert result["status"] == "opened_pull_request"
    assert result["pull_request_url"] == "https://github.com/owner/repo/pull/123"
    assert result["repository_slug"] == "owner/repo"
    assert result["branch_name"].startswith("codna/")
    assert opened == [
        {
            "repository_slug": "owner/repo",
            "token": "test-write-token",
            "branch": result["branch_name"],
            "base_branch": "main",
            "title": "codna: fix app return",
            "body": "body",
        }
    ]
    assert _git_bare(remote, "show", f"{result['branch_name']}:app.py") == "return 2\n"
    assert "test-write-token" not in json.dumps(result)
    for artifact in (tmp_path / ".codna").glob("repository-intelligence/packaged/**/*.json"):
        assert "test-write-token" not in artifact.read_text(encoding="utf-8")


def test_packaged_backend_remote_pr_requires_write_token(tmp_path, monkeypatch) -> None:
    remote = _bare_remote_with_app(tmp_path, "return 1\n")
    _force_packaged_backend(tmp_path, monkeypatch)
    _fake_agent_patch(monkeypatch, "return 1\n", "return 2\n")

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="remote-test",
        connector_type="github_repo",
        config={"repository_url": str(remote), "repository_slug": "owner/repo"},
    )
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "app return value bug"}},
    )
    plan = client.create_repository_decision_plan(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
            "signals": {"issue_text": "app return value bug"},
            "model": "repository.verified_agentic_v1",
        },
    )
    simulation = client.simulate_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "decision_plan_id": plan["decision_plan_id"]},
    )

    with pytest.raises(PackagedGitError) as excinfo:
        client.apply_repository(
            connector["id"],
            {
                "mode": "remote_pr",
                "write_permission": True,
                "decision_plan_id": plan["decision_plan_id"],
                "simulation_id": simulation["validated_inputs"]["simulation_id"],
            },
        )

    assert excinfo.value.code == "repository_remote_pr_token_required"


def test_packaged_github_repo_rejects_embedded_url_credentials(tmp_path) -> None:
    with pytest.raises(PackagedGitError) as excinfo:
        materialize_repository(
            runtime_root=tmp_path / ".codna",
            connector_id="repo-1",
            connector_type="github_repo",
            config={"repository_url": "https://secret@github.com/owner/repo.git"},
            request={},
        )

    assert excinfo.value.code == "invalid_connector_config"


def test_git_auth_env_uses_github_https_basic_header(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_ENV", "1")

    env = _git_auth_env("https://github.com/owner/private-repo.git", "ghs_test_token")

    assert env is not None
    assert env["EXISTING_ENV"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "http.extraHeader"
    assert env["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
    assert "ghs_test_token" not in env["GIT_CONFIG_VALUE_0"]
    encoded = env["GIT_CONFIG_VALUE_0"].removeprefix("AUTHORIZATION: basic ")
    assert b64decode(encoded).decode("utf-8") == "x-access-token:ghs_test_token"


def test_validated_ref_rejects_option_injection_and_accepts_real_refs() -> None:
    """A ref that git could read as an option is an arg-injection RCE (--upload-pack=<cmd>)."""
    # real refs pass through unchanged
    assert _validated_ref("main") == "main"
    assert _validated_ref("refs/heads/feature") == "refs/heads/feature"
    assert _validated_ref("0123abcd4567ef89") == "0123abcd4567ef89"  # commit sha
    # every option-like / malformed ref is rejected before it can reach `git fetch`
    for bad in ("--upload-pack=touch /tmp/PWNED;", "-x", "--exec=evil", " --upload-pack", "--", "-"):
        with pytest.raises(PackagedGitError) as exc:
            _validated_ref(bad)
        assert exc.value.code == "repository_invalid_ref"


def test_git_auth_env_disables_prompts_without_token() -> None:
    env = _git_auth_env("https://github.com/owner/public-repo.git", None)

    assert env is not None
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_CONFIG_COUNT" not in env
    assert "GIT_CONFIG_VALUE_0" not in env


def _force_packaged_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)


def _raise_missing_apps_backend(_config):
    raise LocalCodnaClientError(
        "local_repository_import_failed",
        "synthetic missing full backend",
        {"required_backend": "packaged Algenta local repository-intelligence backend"},
    )


def _fake_agent_patch(monkeypatch, before: str, after: str) -> None:
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        f"-{before}"
        f"+{after}"
    )

    def fake_run(_runner, request):
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="Fixed app return.",
            files_changed=["app.py"],
            telemetry={
                "tokens_in_uncached": 10,
                "tokens_out": 4,
                "total_cost": 0.01,
                "model": request.model,
            },
            artifacts={},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff=patch,
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)


def _bare_remote_with_app(tmp_path: Path, content: str) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    (source / "app.py").write_text(content, encoding="utf-8")
    _git(source, "add", "app.py")
    _git(source, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    remote = tmp_path / "remote.git"
    _git(source, "clone", "--bare", str(source), str(remote))
    return remote


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _git_bare(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--git-dir", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout
