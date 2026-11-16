"""DEFECT 3: cancelling a run must TERMINATE a parked task, and must never approve it.

── THE LEDGER THIS FILE IS WRITTEN FROM ─────────────────────────────────────────────────────────

``~/.codna/runs/run-cancelrepro-before/ledger.ndjson``, verbatim:

    seq 404  kind=review   payload={verdict:"approved", sha:"1686ad67...", conclusion:"success"}
    seq 405  kind=note     publish.skipped (no origin remote)
    seq 406  kind=note     kind=approval.requested
                           message="task task-1 is waiting on a human (review_not_approved)"
    seq 407  kind=note     {action:"cancel", event:"control.requested", identity:"angel"}
    seq 408  kind=note     kind=approval.granted, "task task-1 approved by angel"
    seq 410  kind=note     task.finished: approved_unpublished

seq 404 → 406 is DEFECT 2 and is pinned in `test_park_reason.py` (the gate's NAME said the review
failed while the review record said it passed — the park itself is correct).

seq 407 → 408 → 410 is this file. The run was cancelled and stayed NON-TERMINAL until a human
approval arrived 112 seconds later, after which it finished ``approved_unpublished``: the cancel did
nothing and an approval did everything. Three separate facts had to hold for that ledger, and each of
the three is a test below:

1. **The run never told the child.** `RunWorkflow` awaited the child with a bare ``await handle``, and
   a parked child waits on ``self._decision is not None or self._stopped`` — an attribute of a
   DIFFERENT workflow object, in a different execution. The run-level flag was invisible to the only
   predicate that mattered, so the child waited for a decision nobody was going to give. The fix is
   `stop_delivery.py`; `test_a_cancelled_run_terminates_a_parked_task` is its regression.

2. **Nothing about cancel approved anything.** ``cancel`` is a Signal (route ``stop``) and ``approve``
   is an Update; they reach different receivers and no ``stop`` handler writes an approval. seq 408
   is a SEPARATE, HUMAN act — it carries ``identity: "angel"``, and it is recorded as
   ``approval.granted`` rather than ``auto_approved``, so no stop path synthesized it.
   `test_cancel_never_produces_an_approval` asserts the absence end to end, and
   `test_cancel_is_a_signal_and_approve_is_an_update` pins the two wire routes apart.

3. **Why it unblocked a run the cancel should have ended first.** The approval was ACCEPTED because
   the child was still parked when it arrived: the cancel had not reached the wait yet. That is the
   race, and it is why (1) is the whole fix — once the stop is delivered the child ends itself and a
   late approval has no gate to arrive at. `test_an_approval_after_the_stop_is_refused_not_applied`
   asserts the refusal, so "the cancel won" does not depend on the human being slow.

Every test here runs against a real Temporal test server with the production workflows and the REAL
activities (a fake sidecar, test runner and reviewer — see `tests/durable/fakes.py`). The stop travels
from one workflow execution to a CHILD workflow, which is exactly the seam a unit test cannot exercise.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

# The offline CI job (`cli secure subsystem`) runs `pytest tests` WITHOUT the durable extra, and a
# module-level `temporalio` import aborts COLLECTION for the whole job. Gated exactly as every
# sibling file in this directory gates it.
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

import asyncio  # noqa: E402

from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.run_api_http import SIGNALS, TASK_UPDATES  # noqa: E402
from codna.durable.workflows import (  # noqa: E402
    RunWorkflow,
    TaskWorkflow,
    TaskWorkflowPlain,
)
from tests.durable import fakes  # noqa: E402
from tests.durable.test_task_workflow_time_skipping import (  # noqa: E402
    POLL_INTERVAL_S,
    QUERY_RPC_TIMEOUT_S,
    RUN_TIMEOUT,
    TASK_QUEUE,
    _query,
    _result,
    _run,
    _spec,
    _update,
    _wait_parked,
    _with_worker,
)

# A ceiling on a HANG, not a tolerance. Before the fix the run NEVER ended (it waited for an approval
# that was never coming), so this bound is what turns "the cancel was silently ignored" into a named
# failure instead of a global-timeout traceback that names nothing.
STOPPED_TIMEOUT_S = 45.0
# A validator refusal is local to the child's workflow task; nothing here crosses a network that
# needs to be slow.
REFUSAL_TIMEOUT_S = 10.0


def _parked_spec(repo: Path, *, engine: str) -> TaskSpec:
    """A task that MUST park: a passing verify, a clean review, and a pull request opened.

    Published work is not eligible for the policy's unattended completion (`auto_approval_eligible`
    requires ``publish_skip == "no_remote"`` AND ``pr is None``), and this spec carries no policy
    grant anyway — so it cannot complete itself and the gate is genuinely reached.
    """
    return _spec(repo_dir=str(repo), engine=engine)


def _task_class(engine: str) -> Any:
    return TaskWorkflowPlain if engine == "plain" else TaskWorkflow


def _status_query(engine: str) -> Any:
    return _task_class(engine).status


def _parked_reason(status: dict[str, Any]) -> str:
    """The reason an engine reports for its gate, from whichever field that engine publishes.

    The plain engine reports ``parked_on``; the graph engine reports the interrupt payload it armed,
    whose ``park_reason`` is the same vocabulary. Read here so a test asserts the REASON and not one
    engine's field name — the two engines agreeing on the value is the point of the shared module.
    """
    if isinstance(status.get("parked_on"), str):
        return str(status["parked_on"])
    interrupts = status.get("interrupts") or ()
    if interrupts and isinstance(interrupts[0], dict):
        return str(interrupts[0].get("park_reason") or "")
    pending = status.get("pending_approval") or {}
    return str(pending.get("reason") or "")


def _run_payload(repo: Path, spec: TaskSpec) -> dict[str, Any]:
    """A minimal valid `RunWorkflow` input. One task, the engine the spec names."""
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [spec.to_payload()],
        "engine": spec.engine,
        "max_parallel_tasks": 1,
    }


def _rows(harness: fakes.Harness) -> list[Any]:
    """Every ledger record for the run, in order. `record.kind` is the ledger's own field — a
    `review` or `task.finished` record is NOT a `note`, so the kind is read from the RECORD."""
    return list(harness.ledger("run-1").records())


def _of_kind(rows: list[Any], kind: str) -> list[Any]:
    """Records of one ledger kind.

    A notification names itself in ``payload.kind`` (``approval.granted``, ``task.finished``); the
    structural records (``review``, ``verify``, ``checkpoint``) carry it on the RECORD. Both spellings
    are honoured so a caller can ask for ``"review"`` and ``"approval.granted"`` by the same name.
    """
    return [row for row in rows if row.kind == kind or row.payload.get("kind") == kind]


async def _wait_parked_child(client: Any, workflow_id: str, query: Any) -> dict[str, Any]:
    """Wait for the CHILD task to exist and report it is parked, with a wall-clock ceiling.

    A separate helper from `_wait_parked` because the child is started by the RUN, so the very first
    query can arrive before the child's execution exists — the server answers NOT_FOUND, which is a
    fact about timing rather than a wedged workflow. Tolerated for a bounded window, then reported.
    """
    deadline = time.monotonic() + STOPPED_TIMEOUT_S
    handle = client.get_workflow_handle(workflow_id)
    last_error: str = ""
    while time.monotonic() < deadline:
        try:
            status = await _query(handle, query, timeout_s=QUERY_RPC_TIMEOUT_S)
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - the child has not started yet
            last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(POLL_INTERVAL_S)
            continue
        if status.get("awaiting_approval"):
            return status
        await asyncio.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"the child task {workflow_id!r} never reported it was parked "
        f"(last error={last_error!r}). The run must actually reach the gate for a cancel test to "
        "prove anything about unblocking it."
    )


async def _wait_stopped(run: Any, *, timeout_s: float = STOPPED_TIMEOUT_S) -> Any:
    """Wait for the RUN to reach a terminal state, with a wall-clock ceiling.

    Polled rather than ``await handle.result()`` so a run that never ends fails with a message naming
    THIS wait — the failure mode the defect actually had.
    """
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                "the run never reached a terminal state after its cancel "
                f"(timeout={timeout_s:g}s, last status={last!r}). A cancelled run that keeps waiting "
                "is the silent stall: a run blocked on a parked child nobody told to stop."
            )
        last = await _query(run, RunWorkflow.status, timeout_s=min(QUERY_RPC_TIMEOUT_S, remaining))
        if last.get("stopped"):
            break
        await asyncio.sleep(POLL_INTERVAL_S)
    # `stopped` flips in the signal handler; the completion is what proves the claim was carried out.
    return await _result(run, timeout_s=STOPPED_TIMEOUT_S)


# ── DEFECT 3(a): the cancel must reach the parked child ─────────────────────────────────────────


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_a_cancelled_run_terminates_a_parked_task(tmp_path: Path, engine: str):
    """THE REGRESSION FOR DEFECT 3. Fails before the fix: the run sits non-terminal forever.

    The ledger's shape, reproduced: the task walks its whole route to the approval gate, the run is
    cancelled with NO approval, and the run must reach a terminal ``stopped`` — with a ``task.finished``
    whose outcome is ``stopped``, never ``approved``.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, repo / "runs")
    spec = _parked_spec(repo, engine=engine)

    async def scenario(client: Any) -> dict[str, Any]:
        run = await client.start_workflow(
            RunWorkflow.run,
            args=[_run_payload(repo, spec)],
            id="run:run-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        # It must genuinely PARK first, or this proves nothing about unblocking a park.
        parked = await _wait_parked_child(
            client, f"task-run-1-{spec.task_id}", _status_query(engine)
        )
        # The exact Signal `POST /runs/:id/cancel` sends (`SIGNALS["cancel"]`), and the ONLY act here.
        await run.signal(SIGNALS["cancel"])
        completion = await _wait_stopped(run)
        return {"parked": parked, "completion": completion}

    result = _run(_with_worker(engine, scenario))

    assert result["parked"]["awaiting_approval"] is True, result["parked"]
    assert result["completion"].outcome == "stopped", result["completion"]
    rows = _rows(harness)
    finished = _of_kind(rows, "task.finished")
    assert finished, f"no task.finished was recorded: {[r.kind for r in rows]}"
    # The task's own ending says `stopped`, and it is ALSO reported as unfinished — the two records
    # that make "this task ended without its work" legible instead of reading as a success.
    assert "stopped" in finished[-1].payload["message"], finished[-1].payload
    assert _of_kind(rows, "task.unfinished"), "a stopped task must also be reported as unfinished"
    terminal = _of_kind(rows, "run.finished")[-1].payload
    assert terminal.get("outcome") == "stopped", terminal
    # AND NOBODY APPROVED IT. This is the half that makes it a cancel rather than a rescue.
    assert not _of_kind(rows, "approval.granted"), "a cancel must never approve work"


# ── DEFECT 3(b): cancel carries no approval, and cannot reach an approval handler ──────────────


def test_cancel_is_a_signal_and_approve_is_an_update():
    """`cancel` and `approve` are different commands that cannot reach the same handler.

    Read from the surface's own tables rather than from prose. `cancel` is a run-level Signal
    (``stop``); `approve` is a task-level Update. There is therefore no route by which a cancel
    becomes an approval, which is why seq 408 must be a human act and not a consequence of seq 407.
    """
    assert SIGNALS["cancel"] == "stop"
    assert "cancel" not in TASK_UPDATES
    assert "approve" in TASK_UPDATES
    assert "approve" not in SIGNALS


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_cancel_never_produces_an_approval(tmp_path: Path, engine: str):
    """The ledger a cancelled run leaves has no approval in it — end to end, both engines.

    The narrow version of (a): the ONLY act performed is the cancel, and the resulting ledger carries
    no ``approval.granted``, no ``auto_approved``, and no task summary claiming success.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, repo / "runs")
    spec = _parked_spec(repo, engine=engine)

    async def scenario(client: Any) -> Any:
        run = await client.start_workflow(
            RunWorkflow.run,
            args=[_run_payload(repo, spec)],
            id="run:run-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked_child(client, f"task-run-1-{spec.task_id}", _status_query(engine))
        await run.signal(SIGNALS["cancel"])
        return await _wait_stopped(run)

    completion = _run(_with_worker(engine, scenario))

    assert completion.outcome == "stopped", completion
    rows = _rows(harness)
    assert not _of_kind(rows, "approval.granted"), [r.kind for r in rows]
    assert not _of_kind(rows, "auto_approved"), [r.kind for r in rows]
    assert all(summary.outcome != "approved" for summary in completion.tasks), completion.tasks


# ── DEFECT 3(c): why a late approval unblocked a cancelled run ─────────────────────────────────


def test_an_approval_after_the_stop_is_refused_not_applied(tmp_path: Path):
    """A LATE approval cannot turn a cancelled run into a completed one.

    This is the mechanism behind seq 408. The approval in that ledger was ACCEPTED because the child
    was still parked when it arrived — the cancel had not reached the wait. With the stop delivered
    the child clears ``_parked_on`` and ends, so the approval's VALIDATOR refuses it ("this task is
    not waiting on a decision") instead of applying it. Asserted through the real Update path, so the
    refusal is the product's own and not a test's assumption.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, repo / "runs")
    spec = _parked_spec(repo, engine="plain")

    async def scenario(client: Any) -> dict[str, Any]:
        run = await client.start_workflow(
            RunWorkflow.run,
            args=[_run_payload(repo, spec)],
            id="run:run-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked_child(client, f"task-run-1-{spec.task_id}", TaskWorkflowPlain.status)
        await run.signal(SIGNALS["cancel"])
        completion = await _wait_stopped(run)
        # The child has ENDED by now, so the approval has no gate to arrive at.
        task = client.get_workflow_handle(f"task-run-1-{spec.task_id}")
        try:
            await _update(
                task, TaskWorkflowPlain.approve, {"identity": "angel"}, timeout_s=REFUSAL_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001 - ANY refusal is the answer; applying it is the failure
            refused = True
        else:
            refused = False
        return {"completion": completion, "refused": refused}

    result = _run(_with_worker("plain", scenario))

    assert result["completion"].outcome == "stopped", result["completion"]
    assert result["refused"] is True, "a late approval was applied to a task that had already stopped"
    assert not _of_kind(_rows(harness), "approval.granted")


# ── DEFECT 2 at the workflow level: the gate's own ledger row ───────────────────────────────────


@pytest.mark.parametrize("engine", ["plain", "langgraph"])
def test_an_approved_task_parks_for_RELEASE_not_for_a_failed_review(tmp_path: Path, engine: str):
    """DEFECT 2's ledger contradiction, asserted where it was read: the approval.requested row.

    `run-cancelrepro-before` seq 404 says the review APPROVED and seq 406 says the gate parked on
    ``review_not_approved``. Both halves of that are asserted here from the SAME ledger: the review
    record approves, and the gate's own message agrees with it. The task still parks — releasing
    reviewed work is a human's decision — so this also pins that the fix did not delete the gate.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, repo / "runs")
    spec = _parked_spec(repo, engine=engine)

    async def scenario(client: Any) -> dict[str, Any]:
        task = await client.start_workflow(
            _task_class(engine).run,
            args=[spec.to_payload(), None],
            id="task-run-1-task-1",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        parked = await _wait_parked(task, query=_status_query(engine))
        # A decision is given so the workflow ends and its ledger is complete.
        await _update(task, _task_class(engine).approve, {"identity": "alice@example.com"})
        completion = await _result(task)
        return {"parked": parked, "completion": completion}

    result = _run(_with_worker(engine, scenario))

    rows = _rows(harness)
    review = _of_kind(rows, "review")[-1]
    assert review.payload["verdict"] == "approved", review.payload
    requested = _of_kind(rows, "approval.requested")[-1].payload
    # THE CONTRADICTION, ASSERTED DIRECTLY: an approved review must not be accompanied by a park
    # reason claiming the review was not approved.
    assert "review_not_approved" not in requested["message"], (
        "the gate parked claiming the review was not approved while the review record says it was: "
        f"{requested['message']!r}"
    )
    assert "ready_for_approval" in requested["message"], requested["message"]
    # The gate still EXISTS, and the work still finishes on the human's decision.
    assert _parked_reason(result["parked"]) == "ready_for_approval", result["parked"]
    assert result["completion"].outcome in ("approved", "approved_unpublished"), result["completion"]


# ── FINDING 1: the in-flight handle must survive a NESTED `_run_task` ──────────────────────────
#
# A review bot initially dismissed this as impossible, which is why the reasoning is recorded here.
# `RunWorkflow._run_task` sets `_in_flight_handle`, and `CiFixLoop._run_ci_fix` calls `_run_task`
# AGAIN for a "fix CI" child while the run is already inside the ENGINE child's await — so the field
# has two live owners at once. A `finally: self._in_flight_handle = None` in the inner call destroys
# the OUTER child's handle, and a stop arriving after the fix then has nothing to deliver to: the
# parked outer task waits for an approval that was cancelled. These tests pin the save/restore.


class _RecordingHandle:
    """A handle that stands in for a child workflow in a test with no server."""

    def __init__(self, workflow_id: str) -> None:
        self.id = workflow_id
        self.signals: list[str] = []

    def done(self) -> bool:
        return False

    async def signal(self, name: str) -> None:
        self.signals.append(name)


def test_a_nested_run_task_restores_the_outer_in_flight_handle():
    """After a nested `_run_task` returns, `_in_flight_handle` is the OUTER handle, not `None`.

    Driven through the REAL `_run_task`/`_await_in_flight` bookkeeping, with the two sandbox primitives
    it touches (`workflow.start_child_workflow`, `workflow.wait_condition`) replaced by fakes. The bug
    lives in that bookkeeping, so a helper-level test would not reach it.
    """
    import contextlib

    from codna.durable import run_workflow as run_module

    outer = _RecordingHandle("task-run-1-task-1")
    inner = _RecordingHandle("task-run-1-task-1-ci1")
    starts = iter((outer, inner))

    async def fake_start_child(*_args: Any, **_kwargs: Any) -> Any:
        return next(starts)

    async def fake_wait_condition(predicate: Any) -> None:
        # The real one parks until the predicate holds and is re-evaluated when a signal is applied.
        # Here nothing ever completes, so it parks forever — exactly the state a stop interrupts.
        while not predicate():
            await asyncio.sleep(0)
        await asyncio.sleep(0)

    workflow_run = RunWorkflow()
    spec = _spec(repo_dir="/tmp/codna-nesting-repo", engine="plain")
    workflow_run._spec = run_module.RunSpec(
        run_id="run-1", repo_dir=spec.repo_dir, base_branch="main", tasks=(spec,)
    )

    async def with_fakes() -> Any:
        original_start = run_module.workflow.start_child_workflow
        original_wait = run_module.workflow.wait_condition
        run_module.workflow.start_child_workflow = fake_start_child  # type: ignore[assignment]
        run_module.workflow.wait_condition = fake_wait_condition  # type: ignore[assignment]
        try:
            outer_task = asyncio.ensure_future(workflow_run._run_task(spec, remaining_usd=None))
            for _ in range(100):
                if workflow_run._in_flight_handle is outer:
                    break
                await asyncio.sleep(0)
            assert workflow_run._in_flight_handle is outer, "the outer await never claimed the handle"

            inner_task = asyncio.ensure_future(workflow_run._run_task(spec, remaining_usd=None))
            for _ in range(100):
                if workflow_run._in_flight_handle is inner:
                    break
                await asyncio.sleep(0)
            assert workflow_run._in_flight_handle is inner, "the nested call did not claim the handle"

            # The nested fix finishes: the field must fall back to the OUTER handle, never to None.
            inner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await inner_task
            restored = workflow_run._in_flight_handle
            outer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await outer_task
            return restored
        finally:
            run_module.workflow.start_child_workflow = original_start  # type: ignore[assignment]
            run_module.workflow.wait_condition = original_wait  # type: ignore[assignment]

    restored = _run(with_fakes())
    assert restored is outer, (
        "the nested `_run_task` clobbered the outer in-flight handle, so a stop arriving after a CI "
        f"fix would have nothing to deliver to (got {restored!r})"
    )
