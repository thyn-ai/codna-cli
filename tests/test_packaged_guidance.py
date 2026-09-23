from __future__ import annotations

from pathlib import Path

from codna import packaged_agent_runner as par
from codna.packaged_repository_advanced import PackagedAgentRunRequest


def _req(**kw):
    base = dict(repository_id="r", snapshot_id="s", repo_root=Path("/no/such/repo"),
                issue_text="fix the bug", model="m", signals={}, evidence_bundle={}, snapshot={})
    base.update(kw)
    return PackagedAgentRunRequest(**base)


def test_guidance_from_signal_folded_into_cline_task():
    r = _req(signals={"project_guidance": "Test cmd: pytest -q"})
    payload = par._sidecar_payload(r, Path("/tmp"))
    task = payload["task_spec"]["issue_text"]
    assert "fix the bug" in task and "PROJECT GUIDANCE" in task and "pytest -q" in task
    assert payload["injected_context"]["project_guidance"] == "Test cmd: pytest -q"


def test_no_guidance_leaves_task_clean():
    payload = par._sidecar_payload(_req(signals={}), Path("/tmp"))
    assert payload["task_spec"]["issue_text"] == "fix the bug"
    assert "project_guidance" not in payload["injected_context"]


def test_guidance_read_from_repo_root_when_not_signalled(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Do not touch vendored code.", encoding="utf-8")
    payload = par._sidecar_payload(_req(repo_root=tmp_path, signals={}), tmp_path)
    assert "Do not touch vendored code." in payload["task_spec"]["issue_text"]


def test_localized_payload_limits_fast_path_writes_to_evidence_files():
    payload = par._sidecar_payload(
        _req(
            issue_text="1 failing test(s): tests.test_calc::test_add",
            evidence_bundle={
                "suspect_files": ["calc.py", "tests/test_calc.py", "../escape.py"],
                "evidence_items": [{"file_path": "calc.py"}, {"file_path": "/tmp/nope.py"}],
            },
        ),
        Path("/tmp"),
    )

    assert payload["allowed_write_paths"] == ["calc.py", "tests/test_calc.py"]
    assert "tests.test_calc::test_add" in payload["task_spec"]["issue_text"]
    assert "test identifiers, not file paths" in payload["task_spec"]["issue_text"]
    assert "do not leave duplicate unreachable code behind" in payload["task_spec"]["issue_text"]


def test_provider_qualified_model_populates_sidecar_provider_and_model(monkeypatch):
    monkeypatch.delenv("ALGENTA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("CODNA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("ALGENTA_AGENT_MODEL", raising=False)
    monkeypatch.delenv("CODNA_AGENT_MODEL", raising=False)

    payload = par._sidecar_payload(_req(model="openai-native/gpt-5.5"), Path("/tmp"))

    assert payload["provider"] == "openai-native"
    assert payload["model"] == "gpt-5.5"


def test_default_repository_model_does_not_force_agent_provider(monkeypatch):
    monkeypatch.delenv("ALGENTA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("CODNA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("ALGENTA_AGENT_MODEL", raising=False)
    monkeypatch.delenv("CODNA_AGENT_MODEL", raising=False)

    payload = par._sidecar_payload(_req(model="repository.verified_agentic_v1"), Path("/tmp"))

    assert "provider" not in payload
    assert "model" not in payload


def test_review_payload_uses_read_only_sidecar_profile():
    payload = par._sidecar_payload(
        _req(
            task_kind="review",
            evidence_bundle={"suspect_files": ["calc.py"]},
            signals={"review_max_iterations": 3, "review_timeout_s": 45},
        ),
        Path("/tmp"),
    )

    assert payload["task_kind"] == "review"
    assert payload["task_spec"]["task_kind"] == "review"
    assert payload["approval_profile"] == "repository_review"
    assert payload["allowed_write_paths"] == []
    assert payload["allowed_write_scope"] == []
    assert payload["limits"] == {"maxIterations": 3, "timeoutMs": 45000}


def test_prepare_workspace_excludes_repo_local_runtime_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    runtime_root = repo / "runtime"
    nested = runtime_root / "repository-intelligence" / "packaged"
    nested.mkdir(parents=True)
    (nested / "state.json").write_text("{}", encoding="utf-8")

    workspace = par._prepare_workspace(
        repo_root=repo,
        runtime_root=runtime_root,
        repository_id="r",
        snapshot_id="s",
    )
    try:
        assert (workspace.path / "calc.py").is_file()
        assert not (workspace.path / "runtime").exists()
    finally:
        par._cleanup_workspace(repo, workspace)


# --- a review is delivered as a review, never as a repository issue to fix ---------------------
def test_review_payload_carries_the_prompt_verbatim_and_no_fix_evidence(tmp_path):
    """The sidecar's `task_spec.issue_text` path is the FIX path (buildPrompt wraps it in 'Fix this
    repository issue ... with your file-editing tools, then stop', mines TARGET FILES, and the system
    prompt then says 'edit that file directly'). A review must never travel through it
    (thyn-ai/test-codna-app-e2e#16, 2026-09-19: the review model narrated an edit instead of
    returning findings JSON)."""
    (tmp_path / "AGENTS.md").write_text("Always use tabs.\n", encoding="utf-8")   # would become fix-worded guidance
    review_prompt = "Review the following pull-request diff ...\n----- BEGIN DIFF -----\n+x\n----- END DIFF -----"
    payload = par._sidecar_payload(_req(repo_root=tmp_path, issue_text=review_prompt, task_kind="review", signals={}), tmp_path)
    assert payload["prompt"] == review_prompt                      # verbatim user turn
    assert "issue_text" not in payload["task_spec"]                # no fix-path input at all
    assert payload["task_spec"]["task_kind"] == "review"
    assert payload["injected_context"] == {}
    assert payload["allowed_write_paths"] == [] and payload["allowed_write_scope"] == []
    assert payload["approval_profile"] == "repository_review"
    assert "PROJECT GUIDANCE" not in payload["prompt"]             # review_findings already carries guidance


def test_fix_payload_is_unchanged_by_the_review_split(tmp_path):
    payload = par._sidecar_payload(_req(repo_root=tmp_path, issue_text="Fix the bug in a.py", signals={}), tmp_path)
    assert "prompt" not in payload
    assert payload["task_spec"]["issue_text"].startswith("Fix the bug in a.py")
    assert isinstance(payload["injected_context"], dict)
