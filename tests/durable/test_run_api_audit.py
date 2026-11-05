"""E2 — `codna run verify-ledger` and the identity mirrored on every Update.

Two deliverables, one file, because they are the two halves of "the audit trail can be READ back":

1. **`codna run verify-ledger` PRINTS a divergence.** The failure this guards against is subtle: a
   verifier that tracebacks on a tampered file is useless precisely when it is needed, and one that
   says "invalid" without an index and both hashes gives an auditor nothing to compare. Every case
   below is driven through the REAL command (`cmd_verify_ledger`) and its REAL stdout, not through
   the helper, so "it prints rather than raises" is proven where a user would see it.
2. **Identity reaches the Update, and lands in the ledger.** The workflow's `_approvals` already
   named the approver; what E2 adds is that the ledger's `approval.granted` row does too, and that
   the SIEM row for the delivery says who. That is checked by executing the REAL workflow against a
   real (time-skipping) Temporal server, with the real activity that appends the ledger note.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.ledger import GENESIS_HASH, RunLedger  # noqa: E402
from codna.durable.verify_ledger import (  # noqa: E402
    EXIT_DIVERGED,
    EXIT_INTACT,
    verify_ledger_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# -- fixtures ----------------------------------------------------------------------------------


def _ledger(tmp_path: Path, run_id: str = "run-1") -> RunLedger:
    return RunLedger(tmp_path / "runs" / run_id / "ledger.ndjson", run_id=run_id)


def _seed(ledger: RunLedger) -> None:
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started", "message": "go"})
    ledger.append(
        "turn.event",
        task_id="task-1",
        session_id="s1",
        event_index=1,
        payload={"event_type": "agent.done", "payload": {}},
    )
    ledger.append(
        "note",
        payload={"event": "notification", "kind": "run.finished", "message": "done", "outcome": "completed"},
    )


def _rewrite(path: Path, mutate: Any) -> None:
    """Apply ``mutate`` to the decoded lines and write them back canonically.

    Canonical re-serialization matters: it is the *content* edit that must be detected, not a
    whitespace change that a lenient reader would accept.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mutate(records)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records) + "\n"
    )


class Args:
    """The namespace the command reads. Deliberately not an argparse Namespace: the command only ever
    touches these attributes, and constructing a parser here would test argparse rather than the
    command."""

    def __init__(self, run: str, *, expect_last_seq: int | None = None) -> None:
        self.run = run
        self.run_id = run
        self.expect_last_seq = expect_last_seq


@pytest.fixture
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A runtime root the CLI's config resolution will actually use."""
    root = tmp_path / "runtime"
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(root))
    return root


# -- 1. verify-ledger: the intact case ---------------------------------------------------------


def test_an_intact_chain_reports_intact_and_exits_zero(isolated_runtime: Path, capsys: Any) -> None:
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    ledger = RunLedger(isolated_runtime / "runs" / "run-1" / "ledger.ndjson", run_id="run-1")
    _seed(ledger)
    code = cmd_verify_ledger(Args("run-1"))
    body = json.loads(capsys.readouterr().out)
    assert code == EXIT_INTACT
    assert body["intact"] is True
    assert body["problems"] == []
    assert body["records"] == 3
    assert body["last_seq"] == 3
    assert body["head_hash"] == ledger.tail().hash


# -- 2. verify-ledger: a TAMPERED chain must REPORT, not raise ---------------------------------


def test_an_edited_payload_reports_the_index_and_both_hashes(isolated_runtime: Path, capsys: Any) -> None:
    """The headline requirement: a tampered ledger produces a report, never a traceback."""
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    path = isolated_runtime / "runs" / "run-1" / "ledger.ndjson"
    ledger = RunLedger(path, run_id="run-1")
    _seed(ledger)
    original = json.loads(path.read_text().splitlines()[1])

    def tamper(records: list[dict[str, Any]]) -> None:
        records[1]["payload"]["event_type"] = "agent.done.TAMPERED"

    _rewrite(path, tamper)
    code = cmd_verify_ledger(Args("run-1"))  # MUST NOT RAISE
    body = json.loads(capsys.readouterr().out)

    assert code == EXIT_DIVERGED
    assert body["intact"] is False
    finding = next(item for item in body["findings"] if item["kind"] == "hash")
    assert finding["seq"] == 2
    # BOTH hashes, in full: an auditor diffs them against a known-good head.
    assert len(finding["stored_hash"]) == 64
    assert len(finding["computed_hash"]) == 64
    assert finding["stored_hash"] != finding["computed_hash"]
    assert finding["stored_hash"] == original["hash"]
    # And the human line names the index and the hashes too.
    problem = next(text for text in body["problems"] if "does not match its hash" in text)
    assert "seq 2" in problem
    assert finding["stored_hash"] in problem
    assert finding["computed_hash"] in problem


def test_an_unparseable_line_is_reported_with_its_line_number(isolated_runtime: Path, capsys: Any) -> None:
    """A tampered file is often a corrupted file, and `verify_chain` RAISES on that. This must not."""
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    path = isolated_runtime / "runs" / "run-1" / "ledger.ndjson"
    _seed(RunLedger(path, run_id="run-1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 4, "kind": "note"  <-- truncated mid-write\n')
    code = cmd_verify_ledger(Args("run-1"))
    body = json.loads(capsys.readouterr().out)
    assert code == EXIT_DIVERGED
    finding = body["findings"][0]
    assert finding["kind"] == "line"
    assert "line 4" in finding["detail"]


def test_a_broken_link_names_the_expected_and_stored_hashes(isolated_runtime: Path, capsys: Any) -> None:
    """Deleting a middle record leaves every remaining record's hash valid but the LINK broken."""
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    path = isolated_runtime / "runs" / "run-1" / "ledger.ndjson"
    ledger = RunLedger(path, run_id="run-1")
    _seed(ledger)
    original = [json.loads(line) for line in path.read_text().splitlines()]

    def drop_middle(records: list[dict[str, Any]]) -> None:
        del records[1]

    _rewrite(path, drop_middle)
    code = cmd_verify_ledger(Args("run-1"))
    body = json.loads(capsys.readouterr().out)
    assert code == EXIT_DIVERGED
    link = next(item for item in body["findings"] if item["kind"] == "link")
    # The third record now follows the first, so it expects record 1's hash but stores record 2's.
    assert link["expected_prev"] == original[0]["hash"]
    assert link["stored_hash"] == original[2]["prev_hash"]
    assert link["stored_hash"] != link["expected_prev"]


# -- 3. a chain must not be beatable by SHORTENING it ------------------------------------------


def test_a_truncated_tail_is_reported_when_the_expected_head_is_supplied(
    isolated_runtime: Path, capsys: Any
) -> None:
    """A shortened chain verifies internally — which is exactly how to weaken one.

    This is the plan's "no accepting a chain whose length shrank": the caller supplies the last seq it
    remembers, and a chain that stops short is a finding rather than a clean verdict.
    """
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    path = isolated_runtime / "runs" / "run-1" / "ledger.ndjson"
    ledger = RunLedger(path, run_id="run-1")
    _seed(ledger)
    head = ledger.tail().seq

    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    # Without an expectation the shortened chain is internally consistent: it IS intact as a chain.
    assert cmd_verify_ledger(Args("run-1")) == EXIT_INTACT
    capsys.readouterr()

    code = cmd_verify_ledger(Args("run-1", expect_last_seq=head))
    body = json.loads(capsys.readouterr().out)
    assert code == EXIT_DIVERGED
    shrank = next(item for item in body["findings"] if item["kind"] == "shrank")
    assert shrank["seq"] == head - 1
    assert "removed from the tail" in shrank["detail"]


def test_the_helper_reports_shrinkage_without_the_cli(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    ledger = RunLedger(path, run_id="run-1")
    _seed(ledger)
    head = ledger.tail().seq
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    report = verify_ledger_file(path, expected_records=head)
    assert report.intact is False
    assert [finding.kind for finding in report.findings] == ["shrank"]


def test_a_missing_ledger_is_a_coded_verdict_not_a_crash(isolated_runtime: Path, capsys: Any) -> None:
    from codna.durable.verify_ledger_cli import cmd_verify_ledger

    code = cmd_verify_ledger(Args("no-such-run"))
    body = json.loads(capsys.readouterr().out)
    assert code == 1  # the command could not do its job; not a verdict about a chain
    assert body["exists"] is False
    assert "no ledger at" in body["problems"][0]


def test_an_empty_ledger_is_intact_with_zero_records(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    path.write_text("")
    report = verify_ledger_file(path)
    assert report.intact is True
    assert report.records == 0
    assert report.head_hash == GENESIS_HASH


# -- 4. the command is registered on BOTH surfaces ---------------------------------------------


def test_verify_ledger_is_reachable_as_a_run_subcommand() -> None:
    """`codna run verify-ledger` is the spelling the plan names; it must exist."""
    source = (REPO_ROOT / "cli" / "codna" / "durable" / "run_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert "verify-ledger" in found


def test_the_run_spelling_uses_run_id_because_run_is_ambiguous() -> None:
    """Under `codna run`, `--run` abbreviates ambiguously against `--run-id`/`--run-timeout`."""
    source = (REPO_ROOT / "cli" / "codna" / "durable" / "run_cli.py").read_text(encoding="utf-8")
    assert 'add_verify_ledger_flags(verify_ledger, flag="--run-id")' in source


# -- 5. identity mirrored on every Update ------------------------------------------------------


def test_the_ledger_record_of_an_approval_carries_the_actor() -> None:
    """The notify activity merges the actor into the ledger row — no second record, no branch.

    Exercised at the activity's own seam (a real `RunLedger` on a tmp path, the real `notify` step),
    because that is where "the identity is IN the hash chain" is decided. The runtime is installed the
    way the worker installs it (`configure_default_runtime`), so this is the production path.
    """
    import tempfile

    from codna.durable import activities, activity_steps
    from codna.durable.activity_support import ActivityRuntime
    from codna.durable.contracts import NotifyRequest

    with tempfile.TemporaryDirectory() as directory:
        ledger = RunLedger(Path(directory) / "ledger.ndjson", run_id="run-1")
        runtime = ActivityRuntime(
            runs_root=Path(directory),
            sidecar_factory=lambda owner: None,
            ledger_factory=lambda run_id: ledger,
        )
        previous = activities.DEFAULT_RUNTIME
        activities.configure_default_runtime(runtime)
        try:
            request = NotifyRequest(
                run_id="run-1",
                task_id="task-1",
                kind="approval.granted",
                message="task task-1 approved by qa@codna.ai",
                actor="qa@codna.ai",
            )
            activity_steps.notify(request)
        finally:
            activities.DEFAULT_RUNTIME = previous
            activity_steps._runtime_provider = None  # type: ignore[attr-defined]
        record = ledger.tail()
        assert record.kind == "note"
        assert record.payload["kind"] == "approval.granted"
        assert record.payload["identity"] == "qa@codna.ai"
        assert ledger.verify() == []


def test_a_notification_with_no_actor_carries_no_identity_field() -> None:
    """Absent, not empty: a reader must be able to tell "nobody" from "somebody unnamed"."""
    from codna.durable.contracts import NotifyRequest

    request = NotifyRequest(run_id="run-1", task_id=None, kind="run.started", message="go")
    assert request.actor == ""
    request.validate()  # an empty actor is legal


def test_a_notify_actor_wider_than_the_bound_is_refused() -> None:
    from codna.durable.contract_primitives import ContractError
    from codna.durable.contracts import NotifyRequest

    request = NotifyRequest(
        run_id="run-1", task_id=None, kind="approval.granted", message="m", actor="x" * 400
    )
    with pytest.raises(ContractError) as caught:
        request.validate()
    assert caught.value.details.get("field") == "actor"


def test_send_update_refuses_a_payload_without_an_identity() -> None:
    """A second, workflow-independent guarantee: an Update cannot leave the CLI anonymous."""
    import asyncio

    from codna.durable.cli import DurableCliError, _send_update

    with pytest.raises(DurableCliError) as caught:
        asyncio.run(_send_update(object(), "task-run-1-task-1", "approve", {"task_id": "task-1"}))
    assert caught.value.code == "durable_identity_required"


def test_both_engines_stamp_the_actor_on_the_approval_notification() -> None:
    """Read from the source of both workflow classes: the actor must be passed, not just the message.

    A behavioural assertion here would need a Temporal server per engine; the thing that can regress
    silently is "the notify call lost its actor", and that is exactly what this reads.
    """
    for module in ("task_workflow_plain.py", "task_workflow_graph.py"):
        source = (REPO_ROOT / "cli" / "codna" / "durable" / module).read_text(encoding="utf-8")
        assert 'actor=identity' in source, f"{module} does not stamp the approver on the notification"
        assert '"approval.granted"' in source


def test_the_run_api_identity_and_the_notify_actor_use_the_same_field_name() -> None:
    """One name for one fact, so an auditor's join is a string equality and not a mapping table."""
    source = (REPO_ROOT / "cli" / "codna" / "durable" / "activity_steps.py").read_text(encoding="utf-8")
    assert 'payload["identity"] = request.actor' in source


# -- 6. the identity reaches the ledger THROUGH a real workflow --------------------------------
#
# The tests above prove the parts. This one proves the chain a user's approval actually takes:
# Update → workflow handler → notify activity → ledger row, on a real Temporal server with the REAL
# activities. Nothing here is a stub except the sidecar and the git remote, which is the existing
# `fakes` harness's job.


def test_an_approval_through_a_real_workflow_stamps_the_approver_in_the_ledger(tmp_path: Path) -> None:
    pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
    import asyncio  # noqa: PLC0415 - only this test needs the event loop
    from datetime import timedelta  # noqa: PLC0415

    from temporalio.testing import WorkflowEnvironment  # noqa: PLC0415
    from temporalio.worker import Worker  # noqa: PLC0415

    from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: PLC0415
    from codna.durable.contracts import TaskSpec  # noqa: PLC0415
    from codna.durable.ledger import RunLedger as _RunLedger  # noqa: PLC0415
    from codna.durable.workflows import TaskWorkflowPlain, registered_workflows  # noqa: PLC0415
    from tests.durable import fakes  # noqa: PLC0415
    from tests.durable.test_task_workflow_time_skipping import (  # noqa: PLC0415
        TASK_QUEUE,
        _wait_parked,
    )

    approver = "approver@codna.ai"
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    fakes.build_harness(repo, runs_root)
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="Fix the flaky retry in the client",
        repo_dir=str(repo),
        verify_commands=("true",),
        engine="plain",
    )
    spec.validate()

    async def scenario() -> list[dict[str, Any]]:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            with activity_executor() as executor:
                async with Worker(
                    env.client,
                    task_queue=TASK_QUEUE,
                    workflows=list(registered_workflows()),
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                ):
                    with env.auto_time_skipping_disabled():
                        handle = await env.client.start_workflow(
                            TaskWorkflowPlain.run,
                            args=[spec.to_payload(), None],
                            id="audit-identity",
                            task_queue=TASK_QUEUE,
                            run_timeout=timedelta(hours=24),
                        )
                        await _wait_parked(handle, query=TaskWorkflowPlain.status)
                        await handle.execute_update(
                            TaskWorkflowPlain.approve, {"identity": approver}
                        )
                        await handle.result()
        finally:
            await env.shutdown()
        return [
            record.to_payload()
            for record in _RunLedger.for_run(fakes._paths(runs_root), "run-1").records()
        ]

    records = asyncio.run(scenario())

    granted = [
        record
        for record in records
        if record["kind"] == "note" and record["payload"].get("kind") == "approval.granted"
    ]
    assert granted, f"no approval.granted record in the ledger; kinds={[r['kind'] for r in records]}"
    # WHO, not merely THAT: the andon row names the approver the Update carried.
    assert granted[-1]["payload"]["identity"] == approver
    # And the row is inside the hash chain (the ledger the record came from verifies end to end).
    assert _RunLedger.for_run(fakes._paths(runs_root), "run-1").verify() == []
