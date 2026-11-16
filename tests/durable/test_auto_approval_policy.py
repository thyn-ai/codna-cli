"""Policy-controlled unattended completion for verified local-only work."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from codna.durable.contracts import TaskSpec
from codna.durable.state import TaskState
from codna.policy_document import policy_verdict

PLAN = "refs/plans/1"
POLICY_DIGEST = "a" * 64


def _policy(tmp_path: Path, mode: str = "auto_for_low_risk") -> Path:
    target = tmp_path / "policy.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tools": {"allow": ["read_files"]},
                "approval": {"mode": mode},
            }
        ),
        encoding="utf-8",
    )
    return target


def _spec(repo_dir: str = "/tmp/repo") -> TaskSpec:
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="Build a local app",
        repo_dir=repo_dir,
        verify_commands=("pytest",),
        auto_approve_local_only=True,
        approval_policy_digest=POLICY_DIGEST,
    )
    spec.validate()
    return spec


def _state(**overrides: Any) -> TaskState:
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "plan_ref": PLAN,
        "verify_ref": {
            "ref": "verify/task-1/abc",
            "passed": True,
            "command_digest": "c" * 64,
        },
        "review_verdict": "approved",
        "publish_skip": "no_remote",
    }
    payload.update(overrides)
    return TaskState.from_payload(payload)


def _run_gate(spec: TaskSpec, state: TaskState) -> tuple[dict[str, Any], dict[str, Any]]:
    from codna.durable.graph import wait_for_human

    captured: dict[str, Any] = {}

    def fake_interrupt(prompt: dict[str, Any]) -> dict[str, Any]:
        captured.update(prompt)
        return {"identity": "alice@example.com", "approved": True}

    fake_types = types.ModuleType("langgraph.types")
    fake_types.interrupt = fake_interrupt  # type: ignore[attr-defined]
    fake_langgraph = types.ModuleType("langgraph")
    fake_langgraph.types = fake_types  # type: ignore[attr-defined]
    fake_config = types.ModuleType("langgraph.config")
    fake_config.get_config = lambda: {"metadata": {"codna_task": spec.to_payload()}}  # type: ignore[attr-defined]
    replacements = {
        "langgraph": fake_langgraph,
        "langgraph.types": fake_types,
        "langgraph.config": fake_config,
    }
    saved = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        result = asyncio.run(wait_for_human(state.to_payload()))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return result, captured


def test_policy_defaults_to_human_and_accepts_the_explicit_exemption(tmp_path: Path) -> None:
    ordinary = _policy(tmp_path)
    ordinary.write_text("version: 1\ntools:\n  allow: [read_files]\n", encoding="utf-8")
    default = policy_verdict(ordinary)
    assert default.allowed is True
    assert default.permissions is not None
    assert default.permissions.approval_mode == "human"

    verdict = policy_verdict(_policy(tmp_path))
    assert verdict.allowed is True
    assert verdict.permissions is not None
    assert verdict.permissions.approval_mode == "auto_for_low_risk"


@pytest.mark.parametrize("body", [
    "version: 1\ntools:\n  allow: [read_files]\napproval:\n  mode: automatic\n",
    "version: 1\ntools:\n  allow: [read_files]\napproval:\n  mode: auto_for_low_risk\n  everything: true\n",
])
def test_unknown_approval_policy_values_fail_closed(tmp_path: Path, body: str) -> None:
    target = tmp_path / "policy.yaml"
    target.write_text(body, encoding="utf-8")
    assert policy_verdict(target).allowed is False


def test_task_spec_refuses_a_digest_without_an_explicit_grant() -> None:
    with pytest.raises(ValueError, match="digest requires"):
        TaskSpec(
            run_id="run-1", task_id="task-1", task_number=1,
            brief="Build a local app", repo_dir="/tmp/repo",
            approval_policy_digest=POLICY_DIGEST,
        ).validate()


def test_default_completed_review_still_interrupts_for_a_human() -> None:
    spec = TaskSpec(
        run_id="run-1", task_id="task-1", task_number=1,
        brief="Build a local app", repo_dir="/tmp/repo",
    )
    _, prompt = _run_gate(spec, _state())
    assert prompt["kind"] == "approve"


def test_explicit_policy_completes_local_verified_work_without_interrupt() -> None:
    result, prompt = _run_gate(_spec(), _state())
    assert prompt == {}
    assert result["pending_approval"] is None


def test_a_publishable_result_still_interrupts() -> None:
    state = _state(
        publish_skip=None,
        pr={"number": 1, "url": "https://github.com/example/repo/pull/1", "branch": "task"},
    )
    _, prompt = _run_gate(_spec(), state)
    assert prompt["kind"] == "approve"
    assert prompt["pr"]["number"] == 1


@pytest.mark.parametrize("overrides", [
    {"verify_ref": {"ref": "verify/task-1/abc", "passed": False, "command_digest": "c" * 64}},
    {"review_verdict": "rejected"},
    {"publish_skip": "remote_not_github"},
    {"budget": {"max_cost_usd": 1.0, "spent_usd": 1.0, "max_tokens": None, "spent_tokens": 0}},
])
def test_unverified_or_out_of_scope_work_still_interrupts(overrides: dict[str, Any]) -> None:
    _, prompt = _run_gate(_spec(), _state(**overrides))
    assert prompt["kind"] == "approve"


def test_auto_approval_ledger_record_is_not_a_human_approval() -> None:
    pytest.importorskip("temporalio")
    from codna.durable import task_workflow_graph

    spec = _spec()
    task = task_workflow_graph.TaskWorkflow.__new__(task_workflow_graph.TaskWorkflow)
    task._run_id = spec.run_id
    task._task_id = spec.task_id
    task._spec = spec
    task._state = _state().to_payload()
    task._approvals = []
    task._outcome = None
    records: list[dict[str, Any]] = []

    async def record(request: Any) -> None:
        records.append(request)

    task._record_auto_approval = record  # type: ignore[method-assign]
    decision = asyncio.run(task_workflow_graph.TaskWorkflow._auto_approve(task))
    assert decision == {"auto_approved": True}
    assert task._outcome == "approved_unpublished"
    assert task._approvals == [{
        "identity": None,
        "kind": "auto_approved",
        "turn_seq": 0,
        "reason": "policy_auto_approved_local_unpublished",
        "policy_digest": POLICY_DIGEST,
    }]
    assert records[0].kind == "auto_approved"
    assert records[0].actor == ""
    assert records[0].evidence["policy_digest"] == POLICY_DIGEST


def test_human_approval_still_records_a_named_human() -> None:
    pytest.importorskip("temporalio")
    from codna.durable import task_workflow_graph

    task = task_workflow_graph.TaskWorkflow.__new__(task_workflow_graph.TaskWorkflow)
    task._stopped = False
    task._interrupts = [{"kind": "approve", "park_reason": "review_not_approved"}]
    task._decision = {"identity": "alice@example.com", "approved": True}
    task._approvals = []
    task._state = _state().to_payload()
    task._task_id = "task-1"
    task._outcome = None
    notices: list[tuple[str, str]] = []

    async def notify(kind: str, message: str, *, actor: str = "") -> None:
        notices.append((kind, actor))

    task._notify = notify  # type: ignore[method-assign]

    async def decided(*args: Any, **kwargs: Any) -> None:
        return None

    original = task_workflow_graph.workflow.wait_condition
    task_workflow_graph.workflow.wait_condition = decided
    try:
        result = asyncio.run(task_workflow_graph.TaskWorkflow._await_decision(task))
    finally:
        task_workflow_graph.workflow.wait_condition = original
    assert result == {"identity": "alice@example.com", "approved": True}
    assert task._approvals[0]["kind"] == "approve"
    assert task._approvals[0]["identity"] == "alice@example.com"
    assert notices == [("approval.granted", "alice@example.com")]


def test_auto_approved_drive_never_waits_for_a_human() -> None:
    """Terminal auto-approval does not depend on its particular approved outcome spelling."""
    pytest.importorskip("temporalio")
    from codna.durable import task_workflow_graph

    task = task_workflow_graph.TaskWorkflow.__new__(task_workflow_graph.TaskWorkflow)
    task._stopped = False
    task._outcome = None
    task._auto_approved = False

    async def respect_pause() -> None:
        return None

    async def drain(_app: Any, _pending: Any) -> bool:
        task._auto_approved = True
        task._outcome = "approved"
        return True

    async def await_decision() -> dict[str, Any] | None:
        raise AssertionError("auto-approved work must not wait for a human")

    task._compile = lambda: object()  # type: ignore[method-assign]
    task._respect_pause = respect_pause  # type: ignore[method-assign]
    task._drain = drain  # type: ignore[method-assign]
    task._await_decision = await_decision  # type: ignore[method-assign]
    asyncio.run(task_workflow_graph.TaskWorkflow._drive(task, _state().to_payload()))
    assert task._outcome == "approved"


def test_real_notify_writes_auto_approved_not_human_approval(tmp_path: Path) -> None:
    """The durable ledger itself carries the policy reason and digest, with no invented actor."""
    pytest.importorskip("temporalio")
    from codna.durable.activities import configure_default_runtime
    from codna.durable.activity_steps import notify
    from codna.durable.activity_support import ActivityRuntime
    from codna.durable.contracts import NotifyRequest
    from codna.durable.ledger import RunLedger, verify_chain

    path = tmp_path / "ledger.ndjson"
    configure_default_runtime(ActivityRuntime(
        runs_root=tmp_path,
        sidecar_factory=lambda _owner: None,
        ledger_factory=lambda run_id: RunLedger(path, run_id=run_id),
    ))
    notify(NotifyRequest(
        run_id="run-1", task_id="task-1", kind="auto_approved",
        message="policy allowed unattended completion",
        evidence={"park_reason": "policy_auto_approved_local_unpublished", "policy_digest": POLICY_DIGEST},
    ))
    rows = [__import__("json").loads(line) for line in path.read_text().splitlines()]
    automatic = [row for row in rows if row["payload"].get("kind") == "auto_approved"]
    assert len(automatic) == 1
    assert automatic[0]["payload"]["evidence"]["policy_digest"] == POLICY_DIGEST
    assert automatic[0]["payload"]["evidence"]["park_reason"] == "policy_auto_approved_local_unpublished"
    assert "identity" not in automatic[0]["payload"]
    assert not any(row["payload"].get("kind") == "approval.granted" for row in rows)
    assert verify_chain(path) == []


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_real_local_workflow_finishes_and_ledgers_policy(tmp_path: Path, engine: str) -> None:
    """No approval Update: the real workflow, activities, Git repository and ledger finish."""
    pytest.importorskip("temporalio")
    pytest.importorskip("langgraph")
    from dataclasses import replace
    from tests.durable import fakes
    from tests.durable.test_task_workflow_time_skipping import (
        RUN_TIMEOUT, TASK_QUEUE, _result,
    )
    from codna.durable.workflows import workflow_class_for

    repo = fakes.make_repo(tmp_path / "repo", origin=None)
    harness = fakes.build_harness(repo, tmp_path / "runs")
    spec = replace(_spec(str(repo)), engine=engine)

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            workflow_class_for(engine).run,
            args=[spec.to_payload(), None],
            id=f"auto-approval-{engine}", task_queue=TASK_QUEUE, run_timeout=RUN_TIMEOUT,
        )
        return await _result(handle)

    completion = _with_worker_result(engine, scenario)
    assert completion.outcome == "approved_unpublished"
    assert completion.approved_by is None
    rows = harness.ledger("run-1").records()
    automatic = [row.payload for row in rows if row.payload.get("kind") == "auto_approved"]
    assert len(automatic) == 1
    assert automatic[0]["evidence"]["policy_digest"] == POLICY_DIGEST
    assert automatic[0]["evidence"]["park_reason"] == "policy_auto_approved_local_unpublished"
    assert not any(row.payload.get("kind") == "approval.requested" for row in rows)
    assert not any(row.payload.get("kind") == "approval.granted" for row in rows)


def _with_worker_result(engine: str, scenario: Any) -> Any:
    from tests.durable.test_task_workflow_time_skipping import _run, _with_worker
    return _run(_with_worker(engine, scenario))


def test_run_api_carries_the_policy_grant_and_digest(tmp_path: Path) -> None:
    """The API must not silently ignore an opt-in the CLI would honor."""
    from codna.durable.identity import AUTH_MODE_TOKEN, IdentityConfig, IdentityResolver
    from codna.durable.run_api import RunApiServer
    from codna.durable.run_api_support import RunApiSettings
    from codna.durable.siem import NullSiemSink
    from tests.durable.test_run_api import (
        ScriptedBackend, Running, TEST_IDENTITY, _config, _provider_spec, git_repo,
        load_or_create_token,
    )

    config = _config(tmp_path)
    backend = ScriptedBackend()
    local_token = load_or_create_token(config.paths.root / "run-api" / "run-api.token")
    settings = RunApiSettings(
        port=0, token_path=config.paths.root / "run-api" / "run-api.token",
        allowed_repo_roots=(tmp_path,),
    )
    active = RunApiServer(
        config=config, backend=backend, settings=settings,
        identity=IdentityResolver(IdentityConfig(
            mode=AUTH_MODE_TOKEN, local_token=local_token, local_identity=TEST_IDENTITY,
        )),
        siem=NullSiemSink(),
    ).start()
    server = Running(active, backend, config)
    try:
        repo = git_repo(tmp_path / "repo")
        policy = repo / ".codna" / "policy.yaml"
        policy.parent.mkdir()
        policy.write_text(
            "version: 1\ntools:\n  allow: [read_files]\napproval:\n  mode: auto_for_low_risk\n",
            encoding="utf-8",
        )
        status, body = server.json("POST", "/runs", body={
            "run_id": "run-auto-policy", "repo_dir": str(repo),
            "briefs": ["build local app"], "engine": "plain",
        })
        assert status == 202, body
        spec = _provider_spec(server.backend.started[0])
        assert spec.auto_approve_local_only is True
        assert spec.approval_policy_digest == policy_verdict(policy).policy_digest
    finally:
        server.server.stop()
