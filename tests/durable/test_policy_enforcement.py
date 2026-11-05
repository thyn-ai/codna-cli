"""The WORKER consumer of `.codna/policy.yaml` — the second of the two consumers.

The worker and the sidecar must reach the SAME verdict from the SAME file. This suite pins the
worker's half: `codna.durable.run_cli._require_repository_policy` refuses a run before Temporal
history exists, and the create-session spec marks the run `policy: {required: true}` so the sidecar
cannot fall back to a permissive profile.
"""
from __future__ import annotations

from pathlib import Path

import importlib.util

import pytest

# `codna.durable.cli` MUST be imported before `run_cli`: `run_cli` imports `cli` at module scope, and
# `cli` reads constants back from `run_cli`. Importing `run_cli` first starts the cycle from the wrong
# end and fails with "partially initialized module". Every other durable test enters through `cli`.
from codna.durable import cli as _durable_cli  # noqa: F401  (import-order contract, above)
from codna.durable import run_cli
from codna.durable.contracts import TaskContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "policy"


def test_a_valid_policy_permits_a_run(tmp_path: Path) -> None:
    (tmp_path / ".codna").mkdir()
    (tmp_path / ".codna" / "policy.yaml").write_text(
        (FIXTURES / "valid.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_cli._require_repository_policy(tmp_path)  # must not raise


def test_a_missing_policy_refuses_a_run(tmp_path: Path) -> None:
    """Unattended runs are governed: no policy file is a denial, not a permissive default."""
    with pytest.raises(Exception) as caught:
        run_cli._require_repository_policy(tmp_path)
    assert getattr(caught.value, "code", "") == "policy_missing"


@pytest.mark.parametrize(
    "fixture",
    ["permissive-looking-invalid", "unknown-key-nested", "wrong-type", "empty", "duplicate-key", "unparseable"],
)
def test_a_bad_policy_refuses_a_run_with_the_readers_own_code(tmp_path: Path, fixture: str) -> None:
    """The worker reports the SAME code the sidecar would — one implementation, two callers."""
    from codna.policy_document import policy_verdict

    (tmp_path / ".codna").mkdir()
    path = tmp_path / ".codna" / "policy.yaml"
    path.write_text((FIXTURES / f"{fixture}.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    expected = policy_verdict(path, required=True).code
    with pytest.raises(Exception) as caught:
        run_cli._require_repository_policy(tmp_path)
    assert getattr(caught.value, "code", "") == expected
    assert expected != "policy_ok"


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is None, reason="the `durable` extra is not installed")
def test_the_create_session_spec_requires_a_policy() -> None:
    """The sidecar gets `required: true`, so it cannot silently proceed without a policy."""
    from codna.durable.activity_support import _session_spec

    context = TaskContext(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        repo_dir="/repo",
        base_branch="main",
        provider=None,
        model=None,
    )
    spec = _session_spec(context, read_only=False, working_tree="/repo")
    assert spec["policy"] == {"required": True}


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is None, reason="the `durable` extra is not installed")
def test_a_read_only_task_still_requires_a_policy() -> None:
    """Read-only narrows the write scope; it does not exempt the run from governance."""
    from codna.durable.activity_support import _session_spec

    context = TaskContext(
        run_id="run-2",
        task_id="task-2",
        task_number=1,
        repo_dir="/repo",
        base_branch="main",
    )
    spec = _session_spec(context, read_only=True, working_tree="/repo")
    assert spec["allowed_write_paths"] == []
    assert spec["policy"] == {"required": True}
