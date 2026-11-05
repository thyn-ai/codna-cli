"""The guard for `tests/durable/approval_driver.py`: a hang is never an acceptable failure mode.

That sentence is the whole reason this file exists. The bug it pins was not a wrong assertion — it was
a failure MODE. A driver with a blanket ``except Exception`` swallowed ``AttributeError`` from
``None.status``, so no approval was ever sent, every child stayed parked at its ``interrupt()``, the run
sat RUNNING, and the caller died on a 90-120s timeout with nothing pointing at the cause. That reads as
a product defect and sends whoever debugs it next into the wrong subsystem (it cost a full round on
"LangGraph concurrency is broken", which was never true).

Two properties are proven here, and both were absent before:

* **a wrong approval class is REFUSED before any await** — the offender is named and no query is
  issued, so the failure is at the call that got it wrong;
* **only the three enumerated benign outcomes are absorbed** — not-started/closed (``RPCError``
  ``NOT_FOUND``) and an already-recorded decision (``WorkflowUpdateFailedError`` whose cause says the
  task "is not waiting on a decision"). Every other RPC status, a validator refusal with a different
  reason, and any plain ``AttributeError``/``TypeError``/``NameError`` propagate.

The fake-client tests need no server, because the contract under test is the helper's own. The last
test drives the REAL workflow, because "the narrowing did not break the race" can only be shown
against the engine that produces it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from datetime import timedelta  # noqa: E402

from temporalio.client import WorkflowUpdateFailedError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.service import RPCError, RPCStatusCode  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: E402
from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.workflows import TaskWorkflowPlain, registered_workflows  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import (  # noqa: E402
    approve_all_parked,
    approve_if_parked,
)

TASK_QUEUE = "codna-durable-approval-driver-tests"
RUN_TIMEOUT = timedelta(hours=24)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Handle:
    """A handle that records what the driver asked it to do, and answers as scripted."""

    def __init__(self, *, query_result: Any = None, query_error: Any = None,
                 update_error: Any = None) -> None:
        self.query_result = query_result
        self.query_error = query_error
        self.update_error = update_error
        self.queries: list[Any] = []
        self.updates: list[tuple[Any, Any]] = []

    async def query(self, name: Any) -> Any:
        self.queries.append(name)
        if self.query_error is not None:
            raise self.query_error
        return self.query_result

    async def execute_update(self, name: Any, payload: Any) -> Any:
        self.updates.append((name, payload))
        if self.update_error is not None:
            raise self.update_error
        return {"applied": True}


class _Client:
    """A client whose only job is to hand back one scripted handle."""

    def __init__(self, handle: _Handle) -> None:
        self.handle = handle
        self.requested: list[str] = []

    def get_workflow_handle(self, workflow_id: str) -> _Handle:
        self.requested.append(workflow_id)
        return self.handle


def _rpc(status: RPCStatusCode, message: str = "scripted") -> RPCError:
    return RPCError(message, status, b"")


# -- property 1: a wrong class fails AT THE CALL, before any await ---------------------------


@pytest.mark.parametrize(
    "wrong",
    [None, object(), "TaskWorkflowPlain", TaskWorkflowPlain.status],
    ids=["none", "live-object", "string", "unbound-function"],
)
def test_a_wrong_approval_class_is_refused_before_any_await(wrong: Any) -> None:
    """``None`` here is the exact bug that produced the 120s hang, so it is the first case.

    The assertion is about WHEN, not only that it raises: no query may have been issued. A driver that
    raised only after calling ``wrong.status`` would already have turned ``AttributeError`` into the
    failure mode this file exists to forbid.
    """
    handle = _Handle(query_result={"awaiting_approval": True})
    client = _Client(handle)

    with pytest.raises(TypeError) as caught:
        _run(approve_if_parked(client, "task-1", wrong, identity="guard@codna.ai"))

    assert "child workflow class" in str(caught.value)
    assert repr(wrong) in str(caught.value) or type(wrong).__name__ in str(caught.value)
    assert client.requested == [], "the driver asked for a handle from a class it had not accepted"
    assert handle.queries == [], "the driver queried with a class it had not accepted"


def test_a_class_without_the_handlers_is_refused_and_says_which_one_is_missing() -> None:
    """A class is not enough: it must expose BOTH handlers. Rendered as a refusal, not an AttributeError."""

    class OnlyStatus:
        status = staticmethod(lambda: None)

    handle = _Handle(query_result={"awaiting_approval": True})
    client = _Client(handle)

    with pytest.raises(TypeError) as caught:
        _run(approve_if_parked(client, "task-1", OnlyStatus, identity="guard@codna.ai"))

    assert "cannot drive an approval" in str(caught.value)
    assert "'approve'" in str(caught.value)
    assert handle.queries == []


def test_the_fan_out_helper_refuses_a_wrong_class_too() -> None:
    """`approve_all_parked` is a second entry point, so it is guarded independently.

    Otherwise a caller could pass a wrong class to the fan-out and get the old failure mode back
    without ever touching `approve_if_parked`'s guard.
    """
    handle = _Handle(query_result={"awaiting_approval": True})
    client = _Client(handle)

    with pytest.raises(TypeError):
        _run(approve_all_parked(client, ["task-1", "task-2"], None, identity="guard@codna.ai"))
    assert handle.queries == []


# -- property 2: a harness BUG propagates; only the three benign answers are absorbed ----------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (AttributeError("'NoneType' object has no attribute 'status'"), AttributeError),
        (TypeError("approve() takes 2 positional arguments"), TypeError),
        (NameError("name 'TaskWorkflow' is not defined"), NameError),
        (KeyError("awaiting_approval"), KeyError),
        (RuntimeError("the harness lost the handle"), RuntimeError),
    ],
)
def test_a_programming_error_in_the_driver_propagates(error: Exception, expected: type) -> None:
    """THE regression: under the old blanket ``except Exception`` every one of these was swallowed.

    A swallowed one is not a red test — it is zero approvals and a timeout, so the test fails 90
    seconds later naming nothing. Asserting the type is how "fails at the call" is made mechanical.
    """
    handle = _Handle(query_error=error)
    client = _Client(handle)

    with pytest.raises(expected) as caught:
        _run(approve_if_parked(client, "task-1", TaskWorkflowPlain, identity="guard@codna.ai"))

    assert caught.value is error


@pytest.mark.parametrize(
    "status",
    [
        RPCStatusCode.UNAVAILABLE,
        RPCStatusCode.DEADLINE_EXCEEDED,
        RPCStatusCode.PERMISSION_DENIED,
        RPCStatusCode.INTERNAL,
    ],
)
def test_an_environment_rpc_failure_is_not_mistaken_for_a_benign_absence(
    status: RPCStatusCode,
) -> None:
    """Only NOT_FOUND means "not there". A down server must not read as "no child to approve".

    This is the specific over-narrowing risk of catching ``RPCError`` alone: UNAVAILABLE on a query
    would otherwise look exactly like a child that has not started, and the run would sit until the
    timeout again — same hang, a narrower ``except``.
    """
    handle = _Handle(query_error=_rpc(status))
    client = _Client(handle)

    with pytest.raises(RPCError) as caught:
        _run(approve_if_parked(client, "task-1", TaskWorkflowPlain, identity="guard@codna.ai"))
    assert caught.value.status == status


def test_a_validator_refusal_that_is_not_the_race_propagates() -> None:
    """The double-approval type is ALSO how a malformed payload is refused, so the cause is inspected.

    ``WorkflowUpdateFailedError`` is raised for any validator refusal. Absorbing the type wholesale
    would mean a harness that sent ``{"identity": ""}`` waited out the timeout in silence — the same
    hazard wearing the "legitimate race" comment.
    """
    refusal = WorkflowUpdateFailedError(
        ApplicationError("ValueError: approve: 'identity' must be a non-empty string")
    )
    handle = _Handle(query_result={"awaiting_approval": True}, update_error=refusal)
    client = _Client(handle)

    with pytest.raises(WorkflowUpdateFailedError) as caught:
        _run(approve_if_parked(client, "task-1", TaskWorkflowPlain, identity=""))

    assert caught.value is refusal
    assert handle.updates, "the update was not attempted, so this proves nothing about its refusal"


def test_the_benign_races_are_absorbed_and_reported_as_not_approved() -> None:
    """The positive control: the two races the old ``except`` was FOR still return quietly.

    Without this, "everything propagates" would also pass — proving the narrowing by deleting the
    behaviour it exists to keep.
    """
    not_started = _Handle(query_error=_rpc(RPCStatusCode.NOT_FOUND, "Execution not found"))
    assert _run(approve_if_parked(_Client(not_started), "t", TaskWorkflowPlain, identity="a@b.c")) is False
    assert not_started.updates == []

    # Closed between the query and the update: the status said parked, then the update found nothing.
    closed_late = _Handle(
        query_result={"awaiting_approval": True},
        update_error=_rpc(RPCStatusCode.NOT_FOUND, "update ... not found"),
    )
    assert _run(approve_if_parked(_Client(closed_late), "t", TaskWorkflowPlain, identity="a@b.c")) is False

    # The concurrent-iteration race: the validator says the decision is already recorded.
    already = _Handle(
        query_result={"awaiting_approval": True},
        update_error=WorkflowUpdateFailedError(
            ApplicationError("ValueError: approve: this task is not waiting on a decision")
        ),
    )
    assert _run(approve_if_parked(_Client(already), "t", TaskWorkflowPlain, identity="a@b.c")) is False

    # And the ordinary case still sends exactly one working Update.
    parked = _Handle(query_result={"awaiting_approval": True})
    assert _run(approve_if_parked(_Client(parked), "t", TaskWorkflowPlain, identity="a@b.c")) is True
    assert parked.updates == [(TaskWorkflowPlain.approve, {"identity": "a@b.c"})]


def test_an_unparked_child_is_left_alone() -> None:
    """A running child that is not asking for a decision must not be approved into one."""
    running = _Handle(query_result={"awaiting_approval": False})
    client = _Client(running)

    assert _run(approve_if_parked(client, "t", TaskWorkflowPlain, identity="a@b.c")) is False
    assert running.updates == []


# -- the real race, against the real workflow -------------------------------------------------


def test_the_real_double_approval_race_is_absorbed_not_raised(tmp_path: Path):
    """The narrowing against the ENGINE: two concurrent drivers, one approval, no exception.

    The fake-client tests above are the contract; this one is the proof that the contract matches what
    Temporal actually answers. Two `approve_if_parked` calls race on ONE parked child — the second
    must report ``False`` (someone else won) rather than raise, which is precisely the race the old
    comment claimed to handle. If the narrowing had removed it, this test would fail loudly instead of
    the whole suite hanging, which is the point: same protection, a failure that explains itself.
    """
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
        base_branch="main",
        verify_commands=("true",),
        engine="plain",
    )

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            TaskWorkflowPlain.run,
            args=[spec.to_payload(), None],
            id="driver-race",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        for _ in range(600):
            if (await handle.query(TaskWorkflowPlain.status)).get("awaiting_approval"):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the task never parked on a decision")

        results = await asyncio.gather(
            approve_if_parked(client, "driver-race", TaskWorkflowPlain, identity="one@codna.ai"),
            approve_if_parked(client, "driver-race", TaskWorkflowPlain, identity="two@codna.ai"),
        )
        completion = await asyncio.wait_for(handle.result(), timeout=60)
        return {"results": results, "completion": completion}

    async def wrapper() -> dict[str, Any]:
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
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    result = _run(wrapper())

    # Neither call raised (the gather would have propagated), the task completed, and at most one
    # Update was applied — the loser learned it had lost instead of dying on it.
    assert result["completion"].outcome == "approved", result["completion"]
    assert sorted(result["results"]) == [False, True], result["results"]
