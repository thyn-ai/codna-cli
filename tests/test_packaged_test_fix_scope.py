from __future__ import annotations

from pathlib import Path

import pytest

from codna.packaged_agent_runner import _allowed_write_paths
from codna.packaged_repository_advanced import (
    PackagedAgentRunRequest,
    PackagedAgentRunResult,
    PackagedRepositoryAdvanced,
    PackagedRepositoryAdvancedError,
)
from codna.packaged_repository_backend import FileEntry, _score_entry


def test_test_driven_ranking_demotes_test_files_below_source() -> None:
    terms = {"calc", "test_calc", "add"}
    source = FileEntry(
        path="calc.py",
        size_bytes=32,
        language="python",
        token_estimate=8,
        sample="def add(a, b):\n    return a - b\n",
        sample_sha256="src",
    )
    test = FileEntry(
        path="tests/test_calc.py",
        size_bytes=96,
        language="python",
        token_estimate=24,
        sample="from calc import add\n\nassert add(2, 3) == 5\n",
        sample_sha256="test",
    )

    assert _score_entry(source, terms, set(), test_driven=True) > _score_entry(
        test,
        terms,
        set(),
        test_driven=True,
    )


def test_failing_test_write_scope_excludes_test_files_when_source_is_available(tmp_path: Path) -> None:
    request = PackagedAgentRunRequest(
        repository_id="rid",
        snapshot_id="snap",
        repo_root=tmp_path,
        issue_text="tests are failing with AssertionError",
        model="repository.verified_agentic_v1",
        signals={"issue_text": "tests are failing with AssertionError"},
        evidence_bundle={
            "suspect_files": ["tests/test_calc.py", "calc.py"],
            "evidence_items": [{"file_path": "tests/test_calc.py"}, {"file_path": "calc.py"}],
        },
        snapshot={},
    )

    assert _allowed_write_paths(request) == ["calc.py"]


def test_test_only_patch_is_rejected_for_auto_discovered_failing_tests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    advanced = PackagedRepositoryAdvanced(
        runtime_root=tmp_path / "runtime",
        agent_runner=lambda _request: _agent_result(_test_only_patch()),
    )

    with pytest.raises(PackagedRepositoryAdvancedError) as exc:
        advanced.create_repository_decision_plan(
            repository_id="rid",
            request={
                "snapshot_id": "snap",
                "signals": {
                    "issue_text": "tests are failing with AssertionError: -1 != 5",
                    "failing_tests": [],
                },
                "workspace_evidence_bundle_ref": "bundle",
            },
            snapshot={"repository_id": "rid", "snapshot_id": "snap", "_repo_root_path": str(repo)},
            evidence_bundle={"raw_repo_token_estimate": 100, "evidence_bundle_token_count": 10},
        )

    assert exc.value.code == "local_repository_agent_test_only_patch"
    assert exc.value.details["changed_files"] == ["tests/test_calc.py"]


def test_explicit_test_expectation_request_allows_test_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    advanced = PackagedRepositoryAdvanced(
        runtime_root=tmp_path / "runtime",
        agent_runner=lambda _request: _agent_result(_test_only_patch()),
    )

    plan = advanced.create_repository_decision_plan(
        repository_id="rid",
        request={
            "snapshot_id": "snap",
            "signals": {"issue_text": "update the test expectation because the API intentionally changed"},
            "workspace_evidence_bundle_ref": "bundle",
        },
        snapshot={"repository_id": "rid", "snapshot_id": "snap", "_repo_root_path": str(repo)},
        evidence_bundle={"raw_repo_token_estimate": 100, "evidence_bundle_token_count": 10},
    )

    assert plan["changed_files"] == ["tests/test_calc.py"]


def _agent_result(patch_diff: str) -> PackagedAgentRunResult:
    return PackagedAgentRunResult(
        status="success",
        terminal_state="success",
        agent_run_id="run-1",
        session_id="session-1",
        text="patched",
        files_changed=["tests/test_calc.py"],
        telemetry={},
        artifacts={},
        runtime={},
        patch_diff=patch_diff,
    )


def _test_only_patch() -> str:
    return """diff --git a/tests/test_calc.py b/tests/test_calc.py
index 1111111..2222222 100644
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -1,2 +1,2 @@
 def test_add():
-    assert add(2, 3) == 5
+    assert add(2, 3) == -1
"""
