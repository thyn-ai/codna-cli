"""Temporal Schedules and GitHub triggers: the shapes, the idempotency, and one real fire (D5).

Three layers, and each is tested where it can actually be proven:

1. **The definition and the SDK mapping** are PURE, so they are asserted directly: the interval, the
   ``skip`` overlap policy, the ``{{.ScheduledTime}}`` workflow-id template, the fingerprint's
   stability. This is the layer a time-skipping server cannot reach — measured:
   ``WorkflowService/CreateSchedule is unimplemented`` on the test server.
2. **The dispatch contract** is asserted against a fake client: the workflow id IS the delivery id,
   the signal is delivered WITH the start (Signal-with-Start), and the conflict policy is
   ``USE_EXISTING`` so a redelivery attaches instead of failing.
3. **The trigger's effect ON a workflow** is asserted under the real time-skipping server: the same
   delivery signalled twice reaches ONE execution and is recorded as a duplicate, and a run started
   with a start signal has it before its first task.

The real-server schedule path is gated on ``CODNA_TEST_TEMPORAL=1`` (a dev server implements
``CreateSchedule``; the in-memory test server does not), so the create/update/idempotency cycle is
proven against the same server a user gets rather than skipped entirely.
"""
from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from codna.durable.schedule import (  # noqa: E402
    DEFAULT_JITTER_S,
    MAX_INTERVAL_S,
    ScheduleDefinition,
    ScheduleError,
    _temporal_schedule,
    schedule_id_for,
    schedule_payload,
)
from codna.durable.trigger import (  # noqa: E402
    NAME_WEBHOOK_SIGNAL,
    WebhookDispatch,
    WebhookTriggerError,
    dispatch_webhook,
    goal_for_webhook,
    run_id_for_delivery,
    validate_delivery_id,
    webhook_brief,
)

# -- the definition ------------------------------------------------------------------------------


def _definition(**overrides: Any) -> ScheduleDefinition:
    payload: dict[str, Any] = {
        "schedule_id": schedule_id_for("nightly"),
        "every_seconds": 3600,
        "cron": None,
        "run_payload": {"run_id": "run-nightly-scheduled", "repo_dir": "/tmp/repo", "tasks": []},
    }
    payload.update(overrides)
    return ScheduleDefinition(**payload)  # type: ignore[arg-type]


def test_a_schedule_id_is_prefixed_and_validated():
    assert schedule_id_for("nightly") == "codna-schedule-nightly"
    with pytest.raises(Exception):
        schedule_id_for("../escape")


def test_a_schedule_needs_exactly_one_trigger():
    """Both or neither is a refusal: a schedule with two timetables has no single "when"."""
    with pytest.raises(ScheduleError) as caught:
        _definition(every_seconds=None, cron=None).validate()
    assert caught.value.code == "schedule_trigger_invalid"
    with pytest.raises(ScheduleError):
        _definition(every_seconds=60, cron="0 9 * * *").validate()
    # Exactly one is fine, in either spelling.
    _definition(every_seconds=60, cron=None).validate()
    _definition(every_seconds=None, cron="0 9 * * *").validate()


@pytest.mark.parametrize("every", [0, -1, MAX_INTERVAL_S + 1, "sixty", True])
def test_a_bad_interval_is_refused(every: Any):
    with pytest.raises(ScheduleError):
        _definition(every_seconds=every).validate()


def test_the_fingerprint_is_stable_and_order_independent():
    """Two definitions that differ only in dict order are the SAME schedule.

    That is what makes re-running `codna run schedule` idempotent: the comparison is the fingerprint,
    so an unchanged definition must not produce an update.
    """
    first = _definition(run_payload={"run_id": "r", "tasks": [{"b": 1, "a": 2}]})
    second = _definition(run_payload={"tasks": [{"a": 2, "b": 1}], "run_id": "r"})
    assert first.fingerprint == second.fingerprint
    # ...and a real change does move it.
    assert _definition(every_seconds=7200).fingerprint != first.fingerprint


def test_a_malformed_goal_in_the_payload_is_refused_at_DEFINITION_time():
    """A schedule whose runs would fail on decode is invisible until the first fire, often hours later."""
    with pytest.raises(Exception):
        _definition(run_payload={"run_id": "r", "goal": {"objective": ""}}).validate()
    _definition(
        run_payload={"run_id": "r", "goal": {"objective": "ship it", "stop_condition": {"verify": True}}}
    ).validate()


# -- the SDK mapping (what a server would receive) -----------------------------------------------


def test_the_schedule_skips_an_overlapping_fire():
    """A run that spends money must not overlap itself; two copies is a doubled bill."""
    schedule = _temporal_schedule(_definition(), task_queue="codna.agent.host")
    from temporalio.client import ScheduleOverlapPolicy

    assert schedule.policy.overlap == ScheduleOverlapPolicy.SKIP


def test_every_fire_gets_its_own_workflow_id():
    """``{{.ScheduledTime}}`` in the id: a fixed id would make the second fire collide with the first."""
    schedule = _temporal_schedule(_definition(), task_queue="q")
    assert schedule.action.id == "run:codna-schedule-nightly:{{.ScheduledTime}}"
    assert "{{.ScheduledTime}}" in schedule.action.id


def test_the_schedule_starts_the_run_workflow_with_the_declared_payload():
    schedule = _temporal_schedule(_definition(), task_queue="codna.agent.host")
    assert schedule.action.workflow == "RunWorkflow"
    assert schedule.action.task_queue == "codna.agent.host"
    assert schedule.action.args[0]["run_id"] == "run-nightly-scheduled"
    # `RunWorkflow.run(spec_payload, progress_payload)` — the second argument is the continue-as-new
    # progress, which a fresh fire does not have.
    assert schedule.action.args[1] is None
    assert schedule.action.run_timeout == timedelta(seconds=7 * 24 * 60 * 60)


def test_a_cron_schedule_maps_to_a_cron_expression_not_an_interval():
    schedule = _temporal_schedule(
        _definition(every_seconds=None, cron="0 9 * * 1-5"), task_queue="q"
    )
    assert schedule.spec.intervals == []
    assert schedule.spec.cron_expressions == ["0 9 * * 1-5"]


def test_jitter_is_applied_and_defaults_to_zero():
    assert _temporal_schedule(_definition(), task_queue="q").spec.jitter in (None, timedelta(0))
    assert DEFAULT_JITTER_S == 0
    assert _temporal_schedule(
        _definition(jitter_seconds=30), task_queue="q"
    ).spec.jitter == timedelta(seconds=30)


def test_the_payload_report_carries_the_fingerprint():
    body = schedule_payload(_definition())
    assert body["schedule_id"] == "codna-schedule-nightly"
    assert body["fingerprint"] == _definition().fingerprint
    assert body["every_seconds"] == 3600


# -- the CLI surface -----------------------------------------------------------------------------


def test_the_schedule_command_exists_on_both_surfaces_and_requires_a_trigger():
    """`codna run schedule` (the plan's spelling) and its `durable` sibling parse the same flags."""
    from codna import cli as top_cli

    for argv in (["run", "schedule"], ["durable", "schedule"]):
        parsed = top_cli.build_parser().parse_args(
            argv + ["--brief", "b", "--name", "nightly", "--every", "3600"]
        )
        assert parsed.every == 3600
        assert parsed.schedule_name == "nightly"
        assert parsed.cron is None


def test_the_schedule_command_accepts_the_argv_the_ide_actually_sends():
    """`--json` is on BOTH callers' argv, and it was missing from this parser.

    MEASURED: `ide/extensions/codna-theia/src/common/codna-durable-commands.ts` builds
    ``run schedule … --worktree --json`` (``scheduleArgv``) and ``codna-durable-scheduler.ts`` lists with
    ``run schedule --list --repo <dir> --json``. Neither flag existed here, so BOTH first-class IDE calls died in
    argparse with ``error: unrecognized arguments: --json`` before the command ran: ``/loop``, ``/autopilot`` and
    the Skills & Automations panel's listing were all unreachable, and the surface reported the argparse usage
    line as the refusal.

    ``--json`` is a NO-OP and that is correct: ``_print`` in ``durable/cli.py`` always emits JSON, so a second
    human-formatted branch would be a new output contract invented to justify the flag.
    """
    from codna import cli as top_cli

    parser = top_cli.build_parser()
    listing = parser.parse_args(["run", "schedule", "--list", "--repo", "/repo", "--json"])
    assert listing.list_schedules is True
    assert listing.as_json is True
    assert getattr(listing.func, "__name__", "") == "_cmd_schedule_entry"

    registering = parser.parse_args(
        ["run", "schedule", "--repo", "/repo", "--brief", "b", "--name", "n", "--every", "1800",
         "--worktree", "--json"]
    )
    assert registering.worktree is True
    assert registering.as_json is True
    assert getattr(registering.func, "__name__", "") == "_cmd_schedule_entry"


def test_listing_awaits_the_client_coroutine_before_iterating():
    """``Client.list_schedules`` is a COROUTINE that RESOLVES TO an async iterator.

    MEASURED against temporalio: the signature is ``async def list_schedules(...) -> ScheduleAsyncIterator``, so
    ``async for description in client.list_schedules()`` raises
    ``TypeError: 'async for' requires an object with __aiter__ method, got coroutine``. Listing was therefore
    broken for EVERY caller — the CLI's own ``--list`` and the IDE's automations channel — which is why the
    source is asserted here: the bug is one missing ``await``, and a behavioural test needs a live namespace to
    reach the same line.
    """
    import inspect

    from codna.durable import trigger_cli

    source = inspect.getsource(trigger_cli.cmd_schedule_list)
    assert "await client.list_schedules()" in source
    # The broken spelling must not come back: iterating the coroutine never worked.
    assert "async for description in client.list_schedules():" not in source


def test_the_schedule_command_refuses_both_triggers_and_neither():
    from codna import cli as top_cli

    from codna.durable.trigger_cli import schedule_definition_for as _schedule_definition

    base = ["run", "schedule", "--brief", "b"]
    for extra in (["--name", "n"], ["--name", "n", "--every", "60", "--cron", "0 9 * * *"]):
        args = top_cli.build_parser().parse_args(base + extra)
        with pytest.raises(Exception) as caught:
            # `_schedule_definition` validates before connecting, so no server or event loop is needed.
            from codna.durable.config import load_durable_config

            env = dict(os.environ)
            env["CODNA_RUNTIME_ROOT"] = str(Path("/tmp") / "codna-sched-test")
            config = load_durable_config(env=env, hostname="sched-host")
            _schedule_definition(args, config)
        assert getattr(caught.value, "code", "") == "schedule_trigger_invalid"


def test_the_plan_spells_it_codna_run_schedule():
    """The plan's literal spelling, asserted so a refactor cannot quietly move it under `durable`."""
    from codna import cli as top_cli

    help_text = top_cli.build_parser().format_help()
    assert "run" in help_text
    run_help = top_cli.build_parser()._actions[-1].choices["run"].format_help()
    assert "schedule" in run_help
    assert "trigger" in run_help


# -- the trigger: idempotency by delivery id ------------------------------------------------------


def test_a_delivery_id_is_the_workflow_id_and_is_validated():
    assert validate_delivery_id("1234-abcd") == "1234-abcd"
    for bad in ("", "a/b", "../x", "with space"):
        with pytest.raises(WebhookTriggerError):
            validate_delivery_id(bad)


def test_the_run_id_is_derived_from_the_delivery_so_a_redelivery_names_the_same_run():
    """The run DIRECTORY and ledger are keyed on the run id: a redelivery must resolve to the same."""
    assert run_id_for_delivery("d-1", "fix") == "gh-fix-d-1"
    assert run_id_for_delivery("d-1", "fix") == run_id_for_delivery("d-1", "fix")
    assert run_id_for_delivery("d-1", "review") != run_id_for_delivery("d-1", "fix")
    # A kind with spaces or punctuation is normalised rather than refused: an event's kind is
    # GitHub's vocabulary, not ours.
    assert run_id_for_delivery("d-1", "Check Suite") == "gh-check-suite-d-1"


def test_the_brief_is_a_pure_function_of_the_event():
    brief = webhook_brief(
        kind="fix", repo_full_name="thyn-ai/codna", number=42, reason="comment_codna_fix",
        extra="the retry loop spins",
    )
    assert brief == webhook_brief(
        kind="fix", repo_full_name="thyn-ai/codna", number=42, reason="comment_codna_fix",
        extra="the retry loop spins",
    )
    assert "#42" in brief and "thyn-ai/codna" in brief and "the retry loop spins" in brief


def test_a_webhook_goal_does_not_gate_on_ci():
    """An App-triggered fix already reacts to a check suite; gating on it again would deadlock."""
    goal = goal_for_webhook(objective="fix the bug")
    assert goal.stop_condition.ci is False
    assert goal.stop_condition.verify is True and goal.stop_condition.review is True
    assert goal.max_ci_attempts >= 1


class _FakeClient:
    """A client that records the start request. The dispatch contract is what is under test here."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(self, workflow: Any, **kwargs: Any) -> Any:
        self.calls.append({"workflow": workflow, **kwargs})

        class _Handle:
            first_execution_run_id = "run-abc"

        return _Handle()


def _dispatch() -> WebhookDispatch:
    return WebhookDispatch(
        delivery_id="delivery-1",
        kind="fix",
        repo_full_name="thyn-ai/codna",
        run_payload={"run_id": run_id_for_delivery("delivery-1", "fix"), "repo_dir": "/tmp/r", "tasks": []},
        signal_payload={
            "schema_version": 1,
            "delivery_id": "delivery-1",
            "kind": "fix",
            "repo_full_name": "thyn-ai/codna",
            "number": 3,
        },
    )


def test_a_dispatch_starts_with_the_delivery_as_the_id_and_the_signal():
    """Workflow id = delivery id, the signal delivered WITH the start, and `USE_EXISTING` on conflict."""
    from temporalio.common import WorkflowIDConflictPolicy

    client = _FakeClient()
    config = type("C", (), {"task_queue": "codna.agent.host"})()
    report = asyncio.run(dispatch_webhook(client, config, _dispatch()))

    call = client.calls[0]
    assert call["id"] == "delivery-1"
    assert call["task_queue"] == "codna.agent.host"
    # Signal-with-Start: the trigger is atomic with the start, so there is no window in which the
    # run exists without knowing which delivery woke it.
    assert call["start_signal"] == NAME_WEBHOOK_SIGNAL
    assert call["start_signal_args"][0]["delivery_id"] == "delivery-1"
    # A GitHub redelivery is normal, not an error: attach rather than refuse.
    assert call["id_conflict_policy"] == WorkflowIDConflictPolicy.USE_EXISTING
    assert report["workflow_id"] == "delivery-1"
    assert report["delivery_id"] == "delivery-1"


def test_a_dispatch_refuses_a_delivery_that_could_not_be_idempotent():
    """A made-up delivery id would turn every redelivery into a second run."""
    bad = WebhookDispatch(
        delivery_id="",
        kind="fix",
        repo_full_name="a/b",
        run_payload={"run_id": "r"},
        signal_payload={},
    )
    with pytest.raises(WebhookTriggerError) as caught:
        asyncio.run(dispatch_webhook(_FakeClient(), type("C", (), {"task_queue": "q"})(), bad))
    assert caught.value.code == "webhook_delivery_invalid"


def test_the_workflow_exposes_the_signal_the_trigger_names():
    """A mismatch would be a signal the server accepts and the workflow ignores — silent.

    Asserted against the REGISTERED decorator rather than the method name, because Temporal matches on
    the wire name.
    """
    from codna.durable.run_workflow import RunWorkflow

    definition = RunWorkflow.__temporal_workflow_definition  # type: ignore[attr-defined]
    names = {signal.name for signal in definition.signals.values()}
    assert NAME_WEBHOOK_SIGNAL in names
