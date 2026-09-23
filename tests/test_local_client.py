from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest

from codna.local_client import LocalCodnaRuntimeClient
from codna.packaged_repository_advanced import PackagedAgentRunResult, PackagedRepositoryAdvancedError
import codna.local_client as local_client_module
import codna.local_mojo_pool as mojo_pool_module
import codna.packaged_agent_runner as packaged_agent_runner_module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_packaged_agent_patch_capture_excludes_generated_untracked_artifacts(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def price():\n    return 1\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "app.py")
    _git(repo, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    (repo / "app.py").write_text("def price():\n    return 2\n", encoding="utf-8")
    (repo / "new_module.py").write_text("def tax():\n    return 0\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"cache")
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "README.md").write_text("cache\n", encoding="utf-8")

    patch = packaged_agent_runner_module._capture_patch(repo)

    assert "diff --git a/app.py b/app.py" in patch
    assert "diff --git a/new_module.py b/new_module.py" in patch
    assert "__pycache__" not in patch
    assert ".pytest_cache" not in patch
    assert ".pyc" not in patch


class _Dumpable:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, *, mode="python"):
        assert mode == "json"
        return dict(self.payload)


class _Request:
    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def model_validate(cls, payload):
        return cls(payload)


class _ConnectorType:
    def __init__(self, value):
        if value not in {"local_repo", "github_repo", "repo_archive"}:
            raise ValueError(value)
        self.value = value


def _fake_modules(core):
    data_connector = SimpleNamespace(
        ConnectorType=_ConnectorType,
        ConnectorStatus=SimpleNamespace(LIVE="live"),
        ConnectorVisibility=SimpleNamespace(PRIVATE=SimpleNamespace(value="private")),
        DataConnector=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    schemas = SimpleNamespace(
        RepositorySnapshotCreateRequest=_Request,
        RepositoryTriageRequest=_Request,
        RepositoryGraphQueryRequest=_Request,
        RepositoryDecisionPlanCreateRequest=_Request,
        RepositorySimulationRequest=_Request,
        RepositoryApplyRequest=_Request,
    )
    return local_client_module._RepositoryModules(
        core=core,
        schemas=schemas,
        data_connector=data_connector,
    )


def _raise_missing_apps_backend(_config):
    try:
        raise ModuleNotFoundError("No module named 'apps'", name="apps")
    except ModuleNotFoundError as exc:
        raise local_client_module.LocalCodnaClientError(
            "local_repository_import_failed",
            "Codna could not import the local Algenta repository-intelligence SDK/core modules.",
            {"reason": "ModuleNotFoundError: No module named 'apps'"},
        ) from exc


def _raise_broken_default_apps_backend(_config):
    try:
        raise AssertionError("dev checkout import failed")
    except AssertionError as exc:
        raise local_client_module.LocalCodnaClientError(
            "local_repository_import_failed",
            "Codna could not import the local Algenta repository-intelligence SDK/core modules.",
            {"reason": "AssertionError: dev checkout import failed"},
        ) from exc


def _reset_local_mojo_pool_state() -> None:
    with mojo_pool_module.LOCAL_MOJO_POOL_LOCK:
        mojo_pool_module.LOCAL_MOJO_POOL_STATE = None
        mojo_pool_module.LOCAL_MOJO_POOL_ATTEMPTED = False
        mojo_pool_module.LOCAL_MOJO_POOL_START_EVENT = None
        mojo_pool_module.LOCAL_MOJO_POOL_START_ERROR = None


@pytest.fixture(autouse=True)
def _local_mojo_pool_test_isolation(monkeypatch):
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_PREWARM", "0")
    _reset_local_mojo_pool_state()
    yield
    _reset_local_mojo_pool_state()


def _isolate_from_packaged_backend(monkeypatch) -> None:
    """Make a 'backend absent' assertion hermetic against suite order.

    Another test may have imported the real packaged Algenta backend (a dev checkout on
    sys.path) and cached ``apps.*`` in ``sys.modules``, which makes the import SUCCEED and
    masks the absent-backend path these tests assert. Drop the cached modules and any
    sys.path entry that can still resolve them, so the import genuinely fails.
    """
    for name in [m for m in list(sys.modules) if m == "apps" or m.startswith("apps.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def _resolves_apps(entry: str) -> bool:
        try:
            return (Path(entry) / "apps" / "api_server").is_dir()
        except OSError:
            return False

    monkeypatch.setattr(sys, "path", [p for p in sys.path if not _resolves_apps(p)])



def test_import_repository_modules_reports_packaged_backend_guidance(tmp_path, monkeypatch) -> None:
    engine_dir = tmp_path / "missing-algenta-backend"
    engine_dir.mkdir()
    monkeypatch.setenv("ALGENTA_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    _isolate_from_packaged_backend(monkeypatch)

    with pytest.raises(local_client_module.LocalCodnaClientError) as excinfo:
        local_client_module._import_repository_modules(config)

    assert excinfo.value.code == "local_repository_import_failed"
    assert excinfo.value.details["required_backend"] == (
        "packaged Algenta local repository-intelligence backend"
    )
    assert "apps.api_server.services.repository_intelligence_core" in excinfo.value.details["expected_imports"]


def test_local_client_uses_packaged_backend_for_clean_install_triage(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    source = repo / "src" / "payment.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def charge_card(amount):\n"
        "    if amount < 0:\n"
        "        raise ValueError('amount must be positive')\n"
        "    return amount\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("Payment service docs\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.js").write_text("function ignored() {}\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="clean-install",
        connector_type="local_repo",
        config={"path": str(repo)},
    )
    snapshot = client.create_repository_snapshot(connector["id"], {"focus_paths": ["src/payment.py"]})
    triage = client.triage_repository(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "signals": {
                "issue_text": "File \"src/payment.py\", line 2: negative amount handling is broken",
                "changed_files": ["src/payment.py"],
            },
        },
    )

    assert snapshot["backend_mode"] == "packaged_local_triage"
    assert snapshot["file_count"] == 2
    assert snapshot["language_counts"] == {"markdown": 1, "python": 1}
    assert triage["backend_mode"] == "packaged_local_triage"
    assert triage["suspect_files"][0] == "src/payment.py"
    assert triage["workspace_evidence_bundle_ref"].startswith("packaged-local://")
    assert triage["raw_repo_token_estimate"] >= triage["evidence_bundle_token_count"]
    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    assert len(artifacts) == 2


def test_packaged_triage_snippet_prefers_strong_function_match_over_import(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    package = repo / "jmespath"
    package.mkdir()
    (package / "functions.py").write_text(
        "from jmespath import exceptions\n\n"
        "def _func_ends_with(search, suffix):\n"
        "    return search.endswith(suffix)\n\n"
        "def _func_join(separator, array):\n"
        "    return \"\".join(array)\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="jmespath",
        connector_type="local_repo",
        config={"path": str(repo)},
    )
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "signals": {
                "issue_text": (
                    "Regression: jmespath join ignores the separator. "
                    "join(`-`, items) returns alphabetagamma. Fix the join implementation."
                )
            },
        },
    )

    top = triage["evidence_items"][0]
    assert top["file_path"] == "jmespath/functions.py"
    assert top["symbol_name"] == "_func_join"
    assert "_func_join" in top["snippet"]
    assert "return \"\".join(array)" in top["snippet"]
    assert top["snippet"].strip().splitlines()[0] != "from jmespath import exceptions"


def test_local_client_does_not_mask_explicit_algenta_engine_dir_failure(tmp_path, monkeypatch) -> None:
    engine_dir = tmp_path / "broken-engine"
    engine_dir.mkdir()
    monkeypatch.setenv("ALGENTA_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    client = LocalCodnaRuntimeClient()

    with pytest.raises(local_client_module.LocalCodnaClientError) as excinfo:
        client.create_connector(name="broken", connector_type="local_repo", config={"path": str(tmp_path)})

    assert excinfo.value.code == "local_repository_import_failed"


def test_local_client_ignores_broken_default_dev_checkout_for_clean_install(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.go").write_text("package main\nfunc price() int { return 1 }\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_broken_default_apps_backend)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="clean-install", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})

    assert snapshot["backend_mode"] == "packaged_local_triage"
    assert snapshot["language_counts"] == {"go": 1}


def test_local_client_skips_cross_interpreter_engine_site_packages(tmp_path) -> None:
    engine_dir = tmp_path / "decision-engine"
    current_major, current_minor = sys.version_info[:2]
    compatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor}"
        / "site-packages"
    )
    incompatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor + 1}"
        / "site-packages"
    )
    compatible.mkdir(parents=True)
    incompatible.mkdir(parents=True)

    assert local_client_module._compatible_engine_site_packages(engine_dir) == [compatible]


def test_packaged_backend_removes_failed_default_dev_checkout_import_paths(tmp_path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    fake_engine_dir = fake_home / "Developer" / "decision-engine"
    fake_site_packages = fake_engine_dir / ".venv" / "lib" / "python3.12" / "site-packages"
    fake_site_packages.mkdir(parents=True)
    fake_apps = fake_engine_dir / "apps" / "api_server"
    fake_apps.mkdir(parents=True)
    (fake_engine_dir / "apps" / "__init__.py").write_text("", encoding="utf-8")
    (fake_apps / "__init__.py").write_text("", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.rb").write_text("def price\n  1\nend\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    # Ensure the real dev checkout isn't already importable (suite-order pollution), so the
    # import genuinely fails and the packaged backend fallback + path cleanup are exercised.
    _isolate_from_packaged_backend(monkeypatch)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="clean-install", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})

    assert snapshot["backend_mode"] == "packaged_local_triage"
    assert str(fake_engine_dir.resolve()) not in sys.path
    assert str(fake_site_packages.resolve()) not in sys.path


def test_local_client_packaged_backend_runs_packaged_fix_steps(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.ts").write_text("export function price() { return 1 }\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    def fake_run(_runner, request):
        assert request.repo_root == repo.resolve()
        assert request.issue_text == "price bug"
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="Fixed price return value.",
            files_changed=["app.ts"],
            telemetry={
                "tokens_in_uncached": 12,
                "tokens_out": 5,
                "cache_read_tokens": 2,
                "cache_write_tokens": 1,
                "total_cost": 0.01,
                "model": "claude-test",
            },
            artifacts={"messages": "messages.json"},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff=(
                "diff --git a/app.ts b/app.ts\n"
                "--- a/app.ts\n"
                "+++ b/app.ts\n"
                "@@ -1 +1 @@\n"
                "-export function price() { return 1 }\n"
                "+export function price() { return 2 }\n"
            ),
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="clean-install", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "price bug"}},
    )

    plan = client.create_repository_decision_plan(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
            "signals": {"issue_text": "price bug"},
            "model": "repository.verified_agentic_v1",
        },
    )
    simulation = client.simulate_repository(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "decision_plan_id": plan["decision_plan_id"],
        },
    )
    patch_only = client.apply_repository(
        connector["id"],
        {
            "mode": "patch_only",
            "decision_plan_id": plan["decision_plan_id"],
            "simulation_id": simulation["validated_inputs"]["simulation_id"],
        },
    )

    assert plan["backend_mode"] == "packaged_local_fix"
    assert plan["planner_usage"] == {
        "input_tokens": 12,
        "output_tokens": 5,
        "cache_read_tokens": 2,
        "cache_write_tokens": 1,
        "cost_usd": 0.01,
    }
    assert plan["decision_plan"]["repository_analysis"]["generated_patch_ref"].startswith(
        "packaged-local-patch://"
    )
    assert simulation["recommended_action"] == "apply_patch"
    assert patch_only["status"] == "patch_ready"
    assert "-export function price() { return 1 }" in patch_only["patch"]
    assert (repo / "app.ts").read_text(encoding="utf-8") == "export function price() { return 1 }\n"
    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    assert len(artifacts) == 5
    rows = [pq.read_table(path).to_pylist()[0] for path in artifacts]
    assert {row["step_name"] for row in rows} == {"snapshot", "triage", "decision_plan", "simulate", "apply"}
    assert all(row["status"] == "succeeded" for row in rows)


def test_packaged_local_branch_allows_unrelated_untracked_files(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.ts").write_text("export function price() { return 1 }\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"cache")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "app.ts")
    _git(repo, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    def fake_run(_runner, _request):
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="Fixed price return value.",
            files_changed=["app.ts"],
            telemetry={"tokens_in_uncached": 12, "tokens_out": 5, "total_cost": 0.01},
            artifacts={},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff=(
                "diff --git a/app.ts b/app.ts\n"
                "--- a/app.ts\n"
                "+++ b/app.ts\n"
                "@@ -1 +1 @@\n"
                "-export function price() { return 1 }\n"
                "+export function price() { return 2 }\n"
            ),
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="repo", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "price bug"}},
    )
    plan = client.create_repository_decision_plan(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
            "signals": {"issue_text": "price bug"},
        },
    )
    simulation = client.simulate_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "decision_plan_id": plan["decision_plan_id"]},
    )

    result = client.apply_repository(
        connector["id"],
        {
            "mode": "local_branch",
            "write_permission": True,
            "decision_plan_id": plan["decision_plan_id"],
            "simulation_id": simulation["validated_inputs"]["simulation_id"],
        },
    )

    assert result["status"] == "applied"
    assert result["branch_name"].startswith("codna/")
    assert result["commit_sha"] == _git(repo, "rev-parse", "HEAD").strip()
    assert (repo / "app.ts").read_text(encoding="utf-8") == "export function price() { return 2 }\n"
    assert (repo / "__pycache__" / "app.cpython-313.pyc").exists()
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_packaged_local_branch_commits_each_iteration_for_refix(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.ts").write_text("export function price() { return 1 }\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("- Keep fixes minimal.\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "app.ts", "AGENTS.md")
    _git(repo, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)
    patches = iter([
        (
            "diff --git a/AGENTS.md b/AGENTS.md\n"
            "--- a/AGENTS.md\n"
            "+++ b/AGENTS.md\n"
            "@@ -1 +1,2 @@\n"
            " - Keep fixes minimal.\n"
            "+- Mention the price bug.\n"
        ),
        (
            "diff --git a/app.ts b/app.ts\n"
            "--- a/app.ts\n"
            "+++ b/app.ts\n"
            "@@ -1 +1 @@\n"
            "-export function price() { return 1 }\n"
            "+export function price() { return 2 }\n"
        ),
    ])

    def fake_run(_runner, _request):
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="Generated a patch.",
            files_changed=[],
            telemetry={"tokens_in_uncached": 12, "tokens_out": 5, "total_cost": 0.01},
            artifacts={},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff=next(patches),
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)
    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="repo", connector_type="local_repo", config={"path": str(repo)})

    applied = []
    for issue in ("document price bug", "fix price bug"):
        snapshot = client.create_repository_snapshot(connector["id"], {})
        triage = client.triage_repository(
            connector["id"],
            {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": issue}},
        )
        plan = client.create_repository_decision_plan(
            connector["id"],
            {
                "snapshot_id": snapshot["snapshot_id"],
                "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
                "signals": {"issue_text": issue},
            },
        )
        simulation = client.simulate_repository(
            connector["id"],
            {"snapshot_id": snapshot["snapshot_id"], "decision_plan_id": plan["decision_plan_id"]},
        )
        applied.append(client.apply_repository(
            connector["id"],
            {
                "mode": "local_branch",
                "write_permission": True,
                "decision_plan_id": plan["decision_plan_id"],
                "simulation_id": simulation["validated_inputs"]["simulation_id"],
            },
        ))

    assert [applied[1]["commit_sha"], applied[0]["commit_sha"]] == (
        _git(repo, "rev-list", "--max-count=2", "HEAD").splitlines()
    )
    assert (repo / "app.ts").read_text(encoding="utf-8") == "export function price() { return 2 }\n"
    assert "Mention the price bug" in (repo / "AGENTS.md").read_text(encoding="utf-8")
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_packaged_local_branch_rejects_conflicting_untracked_target(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    (repo / "app.ts").write_text("local untracked content\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    def fake_run(_runner, _request):
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="Added app.",
            files_changed=["app.ts"],
            telemetry={"tokens_in_uncached": 12, "tokens_out": 5, "total_cost": 0.01},
            artifacts={},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff=(
                "diff --git a/app.ts b/app.ts\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/app.ts\n"
                "@@ -0,0 +1 @@\n"
                "+export function price() { return 2 }\n"
            ),
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="repo", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "missing app"}},
    )
    plan = client.create_repository_decision_plan(
        connector["id"],
        {
            "snapshot_id": snapshot["snapshot_id"],
            "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
            "signals": {"issue_text": "missing app"},
        },
    )
    simulation = client.simulate_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "decision_plan_id": plan["decision_plan_id"]},
    )

    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        client.apply_repository(
            connector["id"],
            {
                "mode": "local_branch",
                "write_permission": True,
                "decision_plan_id": plan["decision_plan_id"],
                "simulation_id": simulation["validated_inputs"]["simulation_id"],
            },
        )

    assert excinfo.value.code == "repository_apply_dirty_worktree"
    assert excinfo.value.details["conflicting_untracked_files"] == ["app.ts"]


def test_local_client_packaged_backend_fails_closed_when_agent_makes_no_patch(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.ts").write_text("export function price() { return 1 }\n", encoding="utf-8")
    monkeypatch.delenv("ALGENTA_ENGINE_DIR", raising=False)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", _raise_missing_apps_backend)

    def fake_run(_runner, _request):
        return PackagedAgentRunResult(
            status="succeeded",
            terminal_state="succeeded",
            agent_run_id="run-1",
            session_id="session-1",
            text="No edits needed.",
            files_changed=[],
            telemetry={},
            artifacts={},
            runtime={"kind": "cline", "is_stub": False},
            patch_diff="",
        )

    monkeypatch.setattr(local_client_module.SidecarPackagedAgentRunner, "run", fake_run)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="clean-install", connector_type="local_repo", config={"path": str(repo)})
    snapshot = client.create_repository_snapshot(connector["id"], {})
    triage = client.triage_repository(
        connector["id"],
        {"snapshot_id": snapshot["snapshot_id"], "signals": {"issue_text": "price bug"}},
    )

    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        client.create_repository_decision_plan(
            connector["id"],
            {
                "snapshot_id": snapshot["snapshot_id"],
                "workspace_evidence_bundle_ref": triage["workspace_evidence_bundle_ref"],
                "signals": {"issue_text": "price bug"},
                "model": "repository.verified_agentic_v1",
            },
        )

    assert excinfo.value.code == "local_repository_agent_no_patch"
    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    failed_rows = [pq.read_table(path).to_pylist()[0] for path in artifacts if "decision_plan" in str(path)]
    assert failed_rows[0]["status"] == "failed"


def test_verified_agentic_diff_adds_no_newline_marker_for_git_apply(tmp_path) -> None:
    target = "RECOGNITION/README.md"
    injected = (
        "Recognition for Projects\r\n"
        "=========================\r\n"
        "\r\n"
        "Check other screenshots in this repo for more GH explore page rankings following that day.\n"
        "<!-- codna benchmark injected docs-only restore fault -->\n"
    )
    restored = (
        "Recognition for Projects\r\n"
        "=========================\r\n"
        "\r\n"
        "Check other screenshots in this repo for more GH explore page rankings following that day."
    )
    patch_diff = local_client_module._git_apply_compatible_unified_diff_from_tree_states(
        {target: injected},
        {target: restored},
    )
    assert "\\ No newline at end of file\n" in patch_diff

    repo = tmp_path / "repo"
    target_path = repo / target
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(injected.encode("utf-8"))

    patch_path = repo / ".algenta.patch.diff"
    patch_path.write_text(patch_diff, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    check = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    subprocess.run(["git", "-C", str(repo), "apply", str(patch_path)], check=True)
    assert target_path.read_bytes() == restored.encode("utf-8")


def test_verified_agentic_diff_filters_generated_test_cache_artifacts(tmp_path) -> None:
    before = {
        "roles.py": 'def has_admin_role(roles):\n    return "admin" in roles\n',
    }
    after = {
        **before,
        ".pytest_cache/.gitignore": "# Created by pytest automatically.\n*\n",
        ".pytest_cache/v/cache/nodeids": '[\n  "tests/test_roles.py::T::test_case_insensitive_roles"\n]',
        "__pycache__/roles.cpython-313.pyc": "binary-cache-placeholder",
        "roles.py": 'def has_admin_role(roles):\n    return "admin" in [role.lower() for role in roles]\n',
    }

    patch_diff = local_client_module._git_apply_compatible_unified_diff_from_tree_states(before, after)

    assert ".pytest_cache" not in patch_diff
    assert "__pycache__" not in patch_diff
    assert "roles.py" in patch_diff
    assert "role.lower()" in patch_diff

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "roles.py"
    target.write_text(before["roles.py"], encoding="utf-8")
    patch_path = repo / ".algenta.patch.diff"
    patch_path.write_text(patch_diff, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    check = subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    subprocess.run(["git", "-C", str(repo), "apply", str(patch_path)], check=True)
    assert target.read_text(encoding="utf-8") == after["roles.py"]


def test_patch_artifact_read_preserves_crlf_payload_lines(tmp_path) -> None:
    patch_path = tmp_path / "patch.diff"
    patch_bytes = (
        b"--- a/file.txt\n"
        b"+++ b/file.txt\n"
        b"@@ -1 +1 @@\n"
        b"-old line\r\n"
        b"+new line\r\n"
    )
    patch_path.write_bytes(patch_bytes)

    text = local_client_module._read_text_preserving_newlines(patch_path)

    assert text.encode("utf-8") == patch_bytes


def test_verified_agentic_model_env_shim_supplies_provider_without_engine_url(monkeypatch) -> None:
    captured = {}
    verified_agentic = ModuleType("apps.api_server.services.repository_intelligence_verified_agentic")
    planner_generation = ModuleType("apps.api_server.services.repository_intelligence_planner_generation")

    def original_diff(_before, _after):
        return ""

    async def original_run_agent_with_failover(
        _run_agent,
        *,
        working_dir,
        task_spec,
        injected_context,
        limits,
        root_path,
    ):
        captured["working_dir"] = working_dir
        captured["task_spec"] = task_spec
        captured["injected_context"] = injected_context
        captured["limits"] = limits
        captured["root_path"] = root_path
        engine = task_spec.get("engine") if isinstance(task_spec.get("engine"), dict) else None
        if engine is not None:
            provider = os.environ.get("ALGENTA_AGENT_PROVIDER") or None
            model = os.environ.get("ALGENTA_AGENT_MODEL") or None
            if provider is not None:
                engine["provider"] = provider
            if model is not None:
                engine["model"] = model
        return {"status": "succeeded"}

    verified_agentic._unified_diff_from_tree_states = original_diff
    verified_agentic._run_agent_with_failover = original_run_agent_with_failover
    planner_generation._unified_diff_from_tree_states = original_diff
    monkeypatch.setitem(sys.modules, verified_agentic.__name__, verified_agentic)
    monkeypatch.setitem(sys.modules, planner_generation.__name__, planner_generation)
    monkeypatch.setenv("ALGENTA_AGENT_PROVIDER", "openai-native")
    monkeypatch.setenv("ALGENTA_AGENT_MODEL", "gpt-5.5")

    local_client_module._install_verified_agentic_diff_compatibility()
    result = asyncio.run(
        verified_agentic._run_agent_with_failover(
            lambda **_kwargs: {"status": "succeeded"},
            working_dir="/repo",
            task_spec={"issue_text": "broken"},
            injected_context={"evidence": []},
            limits={"max_turns": 4},
            root_path=Path("/repo"),
        )
    )

    assert result == {"status": "succeeded"}
    assert captured["task_spec"]["engine"] == {
        "provider": "openai-native",
        "model": "gpt-5.5",
    }
    assert "url" not in captured["task_spec"]["engine"]
    assert captured["working_dir"] == "/repo"


def test_local_client_snapshot_uses_in_process_connector_and_runtime_env(tmp_path, monkeypatch) -> None:
    captured = {}

    def create_repository_snapshot(*, connector, request):
        captured["connector"] = connector
        captured["request"] = request.payload
        captured["runtime_dir"] = os.environ.get("ALGENTA_RUNTIME_DIR")
        captured["fallback_mode"] = os.environ.get("RUNTIME_FALLBACK_MODE")
        captured["database_url"] = os.environ.get("DATABASE_URL")
        captured["async_database_url"] = os.environ.get("ASYNC_DATABASE_URL")
        return _Dumpable({"repository_id": str(connector.id), "snapshot_id": "snapshot-1"})

    core = SimpleNamespace(create_repository_snapshot=create_repository_snapshot)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("DATABASE_URL", "should-not-leak")
    monkeypatch.setenv("ASYNC_DATABASE_URL", "should-not-leak")
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="local-test",
        connector_type="local_repo",
        config={"path": str(tmp_path)},
    )
    snapshot = client.create_repository_snapshot(connector["id"], {"focus_paths": ["src"]})

    assert snapshot["snapshot_id"] == "snapshot-1"
    assert captured["connector"].config_json == '{"path":"' + str(tmp_path) + '"}'
    assert captured["request"] == {"focus_paths": ["src"]}
    assert captured["runtime_dir"] == str((tmp_path / ".codna" / "algenta-runtime").resolve())
    assert captured["fallback_mode"] == "deny"
    assert captured["database_url"] is None
    assert captured["async_database_url"] is None
    assert os.environ["DATABASE_URL"] == "should-not-leak"
    assert os.environ["ASYNC_DATABASE_URL"] == "should-not-leak"
    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    assert len(artifacts) == 1
    row = pq.read_table(artifacts[0]).to_pylist()[0]
    assert row["step_name"] == "snapshot"
    assert row["status"] == "succeeded"
    assert row["snapshot_id"] == "snapshot-1"
    assert row["request_sha256"]
    assert row["response_sha256"]


def test_precise_patch_changed_files_counts_only_edited_lines() -> None:
    patch_diff = """--- a/src/serena/util/exception.py
+++ b/src/serena/util/exception.py
@@ -37,7 +37,7 @@
             # This is a simplified check - could be improved
             return True

-    return True
+    return False


 def show_fatal_exception_safe(e: Exception) -> None:
"""

    assert local_client_module._precise_patch_changed_files(patch_diff) == {
        "src/serena/util/exception.py": [(40, 40)]
    }


def test_precise_patch_changed_files_compacts_adjacent_edits() -> None:
    patch_diff = """--- a/src/service.py
+++ b/src/service.py
@@ -10,5 +10,6 @@
 def service():
-    old_value = 1
-    return old_value
+    new_value = 2
+    return new_value
+# fixed
"""

    assert local_client_module._precise_patch_changed_files(patch_diff) == {
        "src/service.py": [(11, 13)]
    }


def test_repository_module_suffix_support_indexes_modern_modules_and_config_files(
    monkeypatch,
) -> None:
    shared_suffixes = {".js", ".jsx", ".ts", ".tsx"}
    text_registry = ModuleType("apps.api_server.services.repository_intelligence_text_file_registry")
    file_selection = ModuleType("apps.api_server.services.repository_intelligence_file_selection")
    classification = ModuleType("apps.api_server.services.repository_intelligence_indexing_classification")
    indexing = ModuleType("apps.api_server.services.repository_intelligence_indexing")
    bindings = ModuleType("apps.api_server.services.repository_intelligence_core_bindings")
    core = ModuleType("apps.api_server.services.repository_intelligence_core")
    text_registry.TEXT_FILE_SUFFIXES = shared_suffixes
    file_selection.TEXT_FILE_SUFFIXES = shared_suffixes
    classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP = {
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }

    def original_language_for_path(path, **_kwargs):
        return classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP.get(
            Path(path).suffix.lower(),
            "unknown",
        )

    for module in (classification, indexing, file_selection, bindings, core):
        module._language_for_path = original_language_for_path
    indexing._JS_LIKE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
    for module in (text_registry, file_selection, classification, indexing, bindings, core):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    local_client_module._install_repository_module_suffix_support()
    local_client_module._install_repository_module_suffix_support()

    for suffix in (".mjs", ".cjs", ".mts", ".cts", ".xml", ".properties"):
        assert suffix in text_registry.TEXT_FILE_SUFFIXES
        assert suffix in file_selection.TEXT_FILE_SUFFIXES
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".mjs"] == "javascript"
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".cjs"] == "javascript"
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".mts"] == "typescript"
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".cts"] == "typescript"
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".xml"] == "xml"
    assert classification._DIRECT_TEXT_SUFFIX_LANGUAGE_MAP[".properties"] == "properties"
    assert indexing._JS_LIKE_SUFFIXES.count(".mjs") == 1
    assert indexing._JS_LIKE_SUFFIXES.count(".cjs") == 1
    assert indexing._JS_LIKE_SUFFIXES.count(".mts") == 1
    assert indexing._JS_LIKE_SUFFIXES.count(".cts") == 1
    assert ".xml" not in indexing._JS_LIKE_SUFFIXES
    assert ".properties" not in indexing._JS_LIKE_SUFFIXES
    for module in (classification, indexing, file_selection, bindings, core):
        assert module._language_for_path(Path("src/check-content.mjs")) == "javascript"
        assert module._language_for_path(Path("src/check-content.cjs")) == "javascript"
        assert module._language_for_path(Path("src/check-content.mts")) == "typescript"
        assert module._language_for_path(Path("src/check-content.cts")) == "typescript"
        assert module._language_for_path(Path("app/src/main/AndroidManifest.xml")) == "xml"
        assert module._language_for_path(Path("gradle.properties")) == "properties"
        assert module._language_for_path(Path("src/check-content.py")) == "unknown"


def test_local_client_simulate_initializes_local_mojo_pool(tmp_path, monkeypatch) -> None:
    captured = {}

    async def simulate_repository_decision(*, connector, request):
        captured["fallback_mode"] = os.environ.get("RUNTIME_FALLBACK_MODE")
        captured["request"] = request.payload
        return _Dumpable({"repository_id": str(connector.id), "simulation_id": "sim-1"})

    core = SimpleNamespace(simulate_repository_decision=simulate_repository_decision)
    pool_roots = []
    published_roots = []
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))
    monkeypatch.setattr(
        local_client_module,
        "_ensure_local_mojo_runtime",
        lambda config: pool_roots.append(config.paths.root),
    )
    monkeypatch.setattr(
        local_client_module,
        "_publish_local_mojo_runtime_state",
        lambda config: published_roots.append(config.paths.root),
    )

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="local-test", connector_type="local_repo", config={"path": str(tmp_path)})
    result = client.simulate_repository(connector["id"], {"decision_plan_id": "plan-1"})

    assert result["simulation_id"] == "sim-1"
    assert pool_roots == [(tmp_path / ".codna").resolve()]
    assert published_roots == [(tmp_path / ".codna").resolve()]
    assert captured == {
        "fallback_mode": "deny",
        "request": {"decision_plan_id": "plan-1"},
    }


def test_local_client_fail_closes_high_risk_simulation_apply(tmp_path, monkeypatch) -> None:
    apply_requests = []
    simulation_responses = [
        {
            "validated_inputs": {"simulation_id": "sim-high"},
            "recommended_action": "apply_patch",
            "metrics": {
                "probability_of_loss": 0.9548,
                "var_95": -50.15,
            },
            "score_breakdown": {"apply_gate": {"passed": True}},
        },
        {
            "validated_inputs": {"simulation_id": "sim-low"},
            "recommended_action": "apply_patch",
            "metrics": {
                "probability_of_loss": 0.17,
                "var_95": -6.87,
            },
            "score_breakdown": {"apply_gate": {"passed": True}},
        },
    ]

    async def simulate_repository_decision(*, connector, request):
        assert connector.id
        assert request.payload["decision_plan_id"] == "plan-1"
        return _Dumpable(simulation_responses.pop(0))

    async def apply_repository_decision(*, connector, request):
        assert connector.id
        apply_requests.append(dict(request.payload))
        return _Dumpable({"apply_result_id": f"apply-{len(apply_requests)}"})

    core = SimpleNamespace(
        simulate_repository_decision=simulate_repository_decision,
        apply_repository_decision=apply_repository_decision,
    )
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))
    monkeypatch.setattr(local_client_module, "_ensure_local_mojo_runtime", lambda _config: None)
    monkeypatch.setattr(local_client_module, "_publish_local_mojo_runtime_state", lambda _config: None)

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="local-test",
        connector_type="local_repo",
        config={"path": str(tmp_path)},
    )
    high = client.simulate_repository(connector["id"], {"decision_plan_id": "plan-1"})

    with pytest.raises(local_client_module.LocalCodnaClientError) as excinfo:
        client.apply_repository(
            connector["id"],
            {
                "mode": "local_branch",
                "write_permission": True,
                "decision_plan_id": "plan-1",
                "simulation_id": high["validated_inputs"]["simulation_id"],
            },
        )

    assert excinfo.value.code == "repository_apply_gate_failed"
    assert excinfo.value.details["simulation_id"] == "sim-high"
    assert excinfo.value.details["probability_of_loss"] == 0.9548
    assert excinfo.value.details["var_95"] == -50.15
    assert apply_requests == []

    patch_only = client.apply_repository(
        connector["id"],
        {
            "mode": "patch_only",
            "decision_plan_id": "plan-1",
            "simulation_id": high["validated_inputs"]["simulation_id"],
        },
    )
    assert patch_only["apply_result_id"] == "apply-1"
    assert apply_requests == [
        {
            "mode": "patch_only",
            "decision_plan_id": "plan-1",
            "simulation_id": "sim-high",
        }
    ]

    low = client.simulate_repository(connector["id"], {"decision_plan_id": "plan-1"})
    local_branch = client.apply_repository(
        connector["id"],
        {
            "mode": "local_branch",
            "write_permission": True,
            "decision_plan_id": "plan-1",
            "simulation_id": low["validated_inputs"]["simulation_id"],
        },
    )
    assert local_branch["apply_result_id"] == "apply-2"
    assert apply_requests[-1] == {
        "mode": "local_branch",
        "write_permission": True,
        "decision_plan_id": "plan-1",
        "simulation_id": "sim-low",
    }
