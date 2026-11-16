"""How a run's stop reaches a task that is PARKED on an approval — a LEAF module.

── THE MEASURED DEFECT ──────────────────────────────────────────────────────────────────────────

`POST /runs/<id>/cancel` signals the RUN workflow's `stop`, which sets `RunWorkflow._stopped`. A task
parked in `wait_for_human` suspends on

    workflow.wait_condition(lambda: self._decision is not None or self._stopped)

and nothing forwarded the run's stop into the child, so the child's `_stopped` stayed `False` and it
waited for an approval the user had already cancelled. Measured on a real run
(`~/.codna/runs/run-cancelrepro-before`): the cancel was accepted (`HTTP 200`, ledger row
`control.requested`), the run's status query reported `stopped: true`, and the run stayed NON-TERMINAL
past 100 seconds — the child's status was `awaiting_approval: true`, `stopped: false`. It advanced only
once a human approved it, and then finished `completed`: the cancel was silently ignored. That is the
"no silent stall" requirement broken.

── WHY A SIGNAL AND NOT A CANCELLATION SCOPE ───────────────────────────────────────────────────

Temporal's cancellation propagates to child workflows through the scope their START is in, so
CANCELLING THE RUN'S WORKFLOW would reach the child. That is not what this product's stop is: a stop
is cooperative and terminal-reporting — the run and the task write their own `stopped` rows and keep
their episodes, which a cancelled execution's code never gets to do. So the flag stays the mechanism
(it is what the existing wait predicate already reads) and this module makes it VISIBLE to the waiter:
the stopped child is sent the signal its own `stop` handler already answers.

── WHY THE HANDLER DOES NOT SEND, AND WHAT WAKES THE RUN INSTEAD ───────────────────────────────

The previous implementation — `asyncio.create_task(handle.signal(...))` from `RunWorkflow.stop` — is
what this module replaced. **Measured, honestly: it DID replay.** `tests/durable/test_stop_replay.py`
replays a real cancelled-run history and passes against that implementation too, so the review's claim
that this path was already nondeterministic was NOT reproduced. What could not be established is the
opposite: nothing in the SDK's contract makes it SAFE.

* `asyncio.create_task` from a synchronous signal handler is not gated on a Temporal event. The task it
  spawns is resumed whenever the workflow event loop gets to it, so whether its
  `SignalExternalWorkflowExecutionInitiated` command lands in THIS workflow task or the next is a
  property of scheduling the history does not pin. It happening to land identically is not a guarantee.
* There is no second chance if that ever changes. A divergent command sequence fails the workflow
  task, Temporal retries it against the same history, and the run is wedged permanently.

So the handler mutates STATE and nothing else (`RunWorkflow.stop` sets `_stopped`), and the main
coroutine drives the send. What makes that sufficient is `workflow.wait_condition`: it IS re-evaluated
when a signal is applied, so the run does not sit unreachable inside `await handle`. `_await_in_flight`
waits on ``handle.done() or self._stopped`` — no command is emitted by that wait — and, once stopped,
records the target and `await`s the send itself. The class of risk above disappears rather than being
argued away, and the replay test is the evidence for the NEW shape (it proves the commands are still
emitted, in an order a replay reproduces).

**THE REPLAY INVARIANT, as an argument rather than a wish.** A replayed execution must issue the same
commands in the same order. Here that reduces to three facts, all of them true:

1. WHICH children are sent a stop is a pure function of history: the children the run started (in start
   order) that had not completed at the boundary where the stop was observed. Every input to that
   decision — the start commands, the child's completion, the `WorkflowExecutionSignaled` event — is a
   recorded event.
2. WHEN the `SignalExternalWorkflowExecutionInitiated` command lands is the same in replay, because
   the send is on the main coroutine: it is visited at the same boundary, after the same awaits, in
   both executions. Nothing is deferred to a task whose scheduling the history does not pin.
3. A run recorded BEFORE this code replays unchanged: its history contains no stop signal, so
   `_stopped` never becomes true, no send is ever issued, and the wait resolves on the child's own
   completion exactly as recorded.

`tests/durable/test_stop_replay.py` is the evidence for (2) and (3): it records a real cancelled run,
replays its history through the SDK Replayer and asserts no nondeterminism error.

No clock, no I/O, no randomness: importable from the sandbox (covered by
`tests/durable/test_import_hygiene.py`).
"""
from __future__ import annotations

from typing import Any, Final

from temporalio import workflow

__all__ = ["STOP_SIGNAL", "StopDelivery"]

#: The signal a parent sends a parked child to make a run's stop visible to its wait predicate.
#:
#: A NAME rather than a method reference: the run cannot import a task workflow class (that choice is
#: made lazily by `workflow_class_for`, and `run_workflow` is what imports it), and looking a handler
#: up by name is exactly what the SDK does for a string signal. Both task engines declare `stop`, and
#: `tests/durable/test_stop_delivery.py` asserts the name they declare is this one.
STOP_SIGNAL: Final = "stop"


class StopDelivery:
    """Collect the child workflows a stop must reach, and send it to each ONCE, in record order.

    **Recording is synchronous; sending is not.** A boundary that observes a stop calls `record` (a
    list append, so it is safe to call from anywhere including a signal handler) and then
    `send_pending` from the MAIN coroutine, which awaits `handle.signal(...)` for each recorded target.
    The awaits are what make the emitted command sequence the same on replay (module docstring).

    **One delivery per child WORKFLOW ID.** A second stop signal is the same instruction; re-sending
    would be a second `SignalExternalWorkflow` command for a history that recorded one. Keyed by id (a
    fix-CI attempt has its own) so two addresses cannot be confused.

    **A failed delivery is reported, never fatal.** A child that already closed (a task whose ending
    raced the stop) answers the signal with `NOT_FOUND`, which Temporal reports back as a
    `SignalExternalWorkflowExecutionFailed` event and the SDK raises from the signal coroutine. There
    is nothing left to stop, so that is logged and absorbed — raising would fail the run, and a stop is
    the worst possible moment to fail a run.
    """

    def __init__(self) -> None:
        #: Targets recorded but not yet sent, in the order a stop asked for them. Workflow state, so
        #: it survives a workflow-task boundary and is identical on replay.
        self._pending: list[tuple[str, Any]] = []
        #: Targets a send has been issued for, in send order. Also `status`'s view of delivery.
        self._sent: list[str] = []

    def record(self, handle: Any) -> bool:
        """Record that ``handle`` must be told to stop. ``False`` if it is not a target.

        Not a target: a handle with no workflow id, one already sent, one already recorded, or one
        that has already COMPLETED (`done()` is resolved from history, so this test is deterministic
        and a finished child would answer the signal with `NOT_FOUND` for no reason).
        """
        workflow_id = str(getattr(handle, "id", "") or "")
        if not workflow_id or workflow_id in self._sent:
            return False
        done = getattr(handle, "done", None)
        if done is not None and done():
            return False
        if any(recorded == workflow_id for recorded, _ in self._pending):
            return False
        self._pending.append((workflow_id, handle))
        return True

    def pending(self) -> tuple[str, ...]:
        """The targets recorded but not yet sent, in record order. For `status` and for a test."""
        return tuple(workflow_id for workflow_id, _ in self._pending)

    def sent(self) -> tuple[str, ...]:
        """The targets a stop has been sent to, in send order. For `status`."""
        return tuple(self._sent)

    async def send_pending(self) -> None:
        """Send `STOP_SIGNAL` to every recorded target, in record order. MAIN coroutine only.

        The queue is taken and cleared BEFORE the first await, so a re-entrant call (a stop signal
        applied in the middle of a send) cannot send the same target twice: `_sent` is appended as
        each send is ISSUED, not as it resolves.
        """
        recorded = self._pending
        self._pending = []
        for workflow_id, handle in recorded:
            if workflow_id in self._sent:
                continue
            self._sent.append(workflow_id)
            await self._send(handle, workflow_id)

    async def _send(self, handle: Any, workflow_id: str) -> None:
        try:
            await handle.signal(STOP_SIGNAL)
        except Exception as exc:  # noqa: BLE001 - "already gone" is the answer, not a failure
            workflow.logger.warning(
                "stop delivery to %s did not reach it (%s); nothing is left to stop",
                workflow_id,
                type(exc).__name__,
            )


class StopRouting:
    """The run's side of the delivery: remembers the awaited child and tells the whole window.

    A MIXIN rather than free functions because the state it guards (the handle currently being
    awaited) is the run's own, and the run is the only object that knows which child it is blocked on.
    Keeping it beside `StopDelivery` also keeps the rationale in one file: `run_workflow.py` is held
    under this package's 1000-line modularity ceiling (`tests/test_modularity.py`), and the alternative
    to moving it here would be raising a ceiling that exists to force exactly this split.

    Subclasses set `_in_flight_handle` and `_stop_delivery` in their own `__init__`.
    """

    _in_flight_handle: Any | None
    _stop_delivery: StopDelivery

    def _record_stop_for_in_flight(self) -> None:
        """Record a stop for the child the run is awaiting, if any."""
        handle = self._in_flight_handle
        if handle is not None:
            self._stop_delivery.record(handle)

    async def _await_in_flight(
        self, handle: Any, *, window: tuple[Any, ...] = ()
    ) -> Any:
        """Await a child while staying REACHABLE by a stop, then deliver the stop and await it again.

        **Why not a bare ``await handle``.** A run blocked inside ``await handle`` is not resumed by a
        signal, so the handler would have to spawn work — the shape this replaces (module docstring).
        ``workflow.wait_condition`` IS re-evaluated when a signal is applied, so the handler stays a
        pure state mutation and the run wakes HERE, on the main coroutine. The condition emits no
        command; the only commands added are the sends below.

        ``window`` is the rest of a concurrent window, whose handles are recorded too so a stop cannot
        leave them parked. Ordering is start order plus record order, so both are replay-stable.

        **The handle is SAVED AND RESTORED, never nulled.** This method nests in practice: a goal-mode
        CI fix (`CiFixLoop._run_ci_fix`) awaits its own child while the run is already inside the
        OUTER child's await. Clearing the field on the way out would destroy the outer handle, and a
        stop arriving after a CI fix would then have nothing to deliver to — the stall this module
        exists to close.
        """
        previous_handle = self._in_flight_handle
        self._in_flight_handle = handle
        try:
            await workflow.wait_condition(lambda: handle.done() or self._stopped)
            if self._stopped:
                self._record_stop_for_in_flight()
                for other in window:
                    self._stop_delivery.record(other)
                await self._stop_delivery.send_pending()
            return await handle
        finally:
            self._in_flight_handle = previous_handle

    async def _stop_in_flight(self) -> None:
        """Record the in-flight child (a no-op when none), then send every recorded stop and wait."""
        self._record_stop_for_in_flight()
        await self._stop_delivery.send_pending()

    async def _stop_window(self, pending: list[tuple[int, Any]]) -> None:
        """Send the stop to every child still in the concurrent window, then wait for the sends.

        The window can hold more than the one child being awaited, so a stop that only reached the
        awaited handle would leave the rest parked. They are TOLD too, and the run waits for the
        signals to be sent before it reports `stopped` — "the cancel was accepted" must mean every
        child heard it, not just the one the loop happened to be on.
        """
        self._record_stop_for_in_flight()
        for _index, handle in pending:
            self._stop_delivery.record(handle)
        await self._stop_delivery.send_pending()
