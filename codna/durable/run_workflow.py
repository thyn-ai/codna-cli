"""The run: one child task per task, a run-level budget, and continue-as-new.

**Continue-as-new is how a long run stays replayable.** Temporal history is finite; a run with thirty
tasks and ten turns each would pass the 10,240-event ceiling, become slow to replay, and eventually be
rejected. So the workflow carries its progress in its INPUT (`RunProgress`) and restarts itself with
``continue_as_new`` when either bound is crossed — every 10 completed tasks, or once this execution's
history passes 5,000 events. The input is small and fixed-size, so the restart cannot itself grow
without bound.

**Continue-as-new is how a long run stays replayable.** Temporal history is finite; a run with thirty
tasks and ten turns each would pass the 10,240-event ceiling, become slow to replay, and eventually be
rejected. So the workflow carries its progress in its INPUT (`RunProgress`) and restarts itself with
``continue_as_new`` when either bound is crossed — every 10 completed tasks, or once this execution's
history passes 5,000 events. The input is small and fixed-size, so the restart cannot itself grow
without bound. (The keyword-`args` requirement is stated where the call is, because that is where an
edit would break it.)

**`remaining_usd` is the run's cap minus what its children reported.** A task that exhausts its own cap
parks on a human; a run whose TOTAL is exhausted stops starting tasks and returns ``budget_exhausted``.
Two different questions — "may this task spend more?" vs. "may this run buy another task?" — which is
why the budget is checked in both places.

**A stop is DELIVERED to the in-flight child.** A run cannot return until its `await` on the child
returns, and a child parked in `wait_for_human` only returns when something sets ITS stop. The run
keeps that handle and delivers `STOP_SIGNAL` before reporting `stopped`; `stop_delivery.py` carries
the rationale and the measured stall this closes.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping

from temporalio import workflow
from temporalio.workflow import HandlerUnfinishedPolicy

from .contracts import (
    NAME_NOTIFY,
    NAME_RECALL,
    NAME_WORKTREE_CLEANUP,
    NotifyRequest,
    RecallRequest,
    RunCompletion,
    RunOutcome,
    RunProgress,
    RunSpec,
    TaskCompletion,
    TaskSpec,
    TaskSummary,
    WorktreeCleanupRequest,
)
from .run_contracts import NOTIFICATION_KIND_RUN_FINISHED, run_terminal_evidence
from .run_reporting import report_publish_skips
from .stop_delivery import StopDelivery, StopRouting
from .ci_fix_loop import CiFixLoop
from .goal import (
    EpisodeRecap,
    EpisodeRef,
    GoalEvidence,
    StopDecision,
    evaluate_stop,
    format_objective_section,
)
from .workflow_support import (
    history_length,
    progress_from,
    remaining_tasks,
    run_budget_left,
    run_spec_from,
    workflow_activity,
)

def workflow_class_for(engine: str) -> Any:
    """The workflow class for an engine name, refusing an unrecognized one.

    Imports the two engines lazily: `workflows.py` imports this module, so a module-level import of
    them here would close a cycle. No default on a typo — the two engines produce different history,
    and a run started under the wrong one cannot be replayed by the worker that picks it up.
    """
    from .contracts import GRAPH_ENGINES

    if engine == "langgraph":
        from .task_workflow_graph import TaskWorkflow

        return TaskWorkflow
    if engine == "plain":
        from .task_workflow_plain import TaskWorkflowPlain

        return TaskWorkflowPlain
    raise ValueError(f"unknown graph engine {engine!r}; expected one of {', '.join(GRAPH_ENGINES)}")


__all__ = ["RunWorkflow", "workflow_class_for"]


# -- the run -----------------------------------------------------------------------------


@workflow.defn(name="RunWorkflow")
class RunWorkflow(StopRouting, CiFixLoop):
    """A run: one child task per task, a run-level budget, and continue-as-new.

    **Continue-as-new is how a long run stays replayable.** Temporal history is finite; a run with
    thirty tasks and ten turns each would pass the 10,240-event ceiling, become slow to replay, and
    eventually be rejected. So the workflow carries its progress in its INPUT (`RunProgress`:
    completed summaries, spend, generation) and restarts itself with ``continue_as_new`` when either
    bound is crossed — every 10 completed tasks, or once this execution's history passes 5,000
    events. The input is small and fixed-size, so the restart cannot itself grow without bound.

    ``remaining_usd`` is the run's cap minus what its children reported. A task that exhausts its own
    cap parks on a human; a run whose TOTAL is exhausted stops starting tasks and returns
    ``budget_exhausted``. Two different questions — "may this task spend more?" vs. "may this run buy
    another task?" — which is why the budget is checked in both places.
    """

    def __init__(self) -> None:
        self._spec: RunSpec | None = None
        self._progress = RunProgress()
        self._paused = False
        self._stopped = False
        self._completed: list[TaskSummary] = []
        self._spent: float = 0.0
        # D5 goal mode. Empty/None/0 for a task-mode run, so nothing below runs and no activity is
        # scheduled that a task-mode history does not already have.
        self._episodes: list[EpisodeRef] = []
        self._ci_state: str | None = None
        self._ci_attempts: int = 0
        self._goal_reason: str = ""
        # WHY the CI loop ended, when it ended red. `""` while green or never asked. Carried so the
        # completion can say "three attempts" or "out of money" — two very different next steps.
        self._ci_stop: str = ""
        # D5 trigger state: the delivery ids this execution has seen, and the duplicate it recorded.
        # A list rather than a set because it is reported in `status` and a set's order would be a
        # dict-order accident of the runtime rather than a fact about the deliveries.
        self._deliveries: list[str] = []
        self._duplicate_deliveries: list[str] = []
        self._webhook_payloads: list[dict[str, Any]] = []
        # The child this run is AWAITING, so a stop can reach a child parked on an approval.
        # `stop_delivery.py` owns the rationale and the measured stall this closes. SAVED AND
        # RESTORED around each await: a goal-mode CI fix runs a NESTED `_run_task` inside the outer
        # child's await, and nulling on the way out would destroy the outer handle.
        self._in_flight_handle: Any | None = None
        self._stop_delivery = StopDelivery()

    @workflow.run
    async def run(
        self, spec_payload: Mapping[str, Any], progress_payload: Mapping[str, Any] | None = None
    ) -> RunCompletion:
        spec = run_spec_from(spec_payload)
        self._spec = spec
        progress = progress_from(progress_payload)
        self._progress = progress
        self._completed = list(progress.completed)
        self._spent = float(progress.spent_usd)
        self._episodes = list(progress.episodes)
        self._ci_state = progress.ci_state
        self._ci_attempts = int(progress.ci_attempts)
        self._goal_reason = progress.goal_reason

        remaining = remaining_tasks(spec, self._completed)
        # THE OPENING NOTIFICATION CARRIES THE CHECKOUT. It is the one row guaranteed to exist for a run
        # that got far enough to be listed, so `GET /runs` reads the repository from here — see
        # `repo_dir_of` in `run_repo`. Attached to THIS call rather than a new activity: a run's
        # history replays its command sequence, so an extra activity would make every existing history
        # diverge.
        await self._notify(
            "run.started",
            f"run {spec.run_id}: {len(remaining)} task(s) to go",
            repo_dir=spec.repo_dir,
        )
        await self._report_capabilities(spec)

        try:
            return await self._run_tasks(spec, spec_payload, remaining)
        except Exception:
            # A run that FAILED is still a run whose worktrees must not leak. `Exception`, not
            # `BaseException`: `continue_as_new` raises a `BaseException`, and that is not an ending —
            # the next generation keeps the same run directory and may still work in those checkouts.
            #
            # The terminal is written BEFORE the sweep, and before the re-raise, for two reasons: a
            # ledger has to say how the run ended (a run that only swept its checkouts would leave the
            # last record a turn terminal, and the list route would have to DERIVE a verdict from it),
            # and the ORDER is what a reader's tail sees — the ending first, then the sweep it caused.
            await self._finish("failed", f"run {spec.run_id}: failed, see the run's error")
            await self._finalize("failed")
            raise

    async def _finish(
        self, outcome: str, message: str, *, task_id: str | None = None
    ) -> None:
        """Write the run's OWN terminal, carrying the outcome the exit took.

        Every exit from a run goes through here, which is what makes "a listed run says how it ended" a
        property of the code path rather than of one branch somebody remembered.

        ``task_id`` names the task the ending came from. It is not decoration: it is what lets a reader
        match the terminal to that task's OWN verdict in a ledger written before the terminal carried an
        outcome, where a "fix CI" attempt's verdict sits in the same window.

        The run's WORKER can still die without reaching this (a killed process has no exit to run), so a
        ledger ending at a turn terminal remains a real shape; `run_verdict` derives a verdict for it.
        This makes that derivation the exception rather than the rule.
        """
        await self._notify(NOTIFICATION_KIND_RUN_FINISHED, message, task_id=task_id, outcome=outcome)

    async def _run_tasks_concurrently(
        self, spec: RunSpec, spec_payload: Mapping[str, Any], remaining: tuple[TaskSpec, ...]
    ) -> RunCompletion:
        """Run up to ``max_parallel_tasks`` tasks at once, keeping the caller's task ORDER in the result.

        **Why a window and not ``asyncio.gather`` over everything.** A run is a list of briefs against
        one repository. Unbounded fan-out would start every paid turn in a burst, and a 50-task run
        would open 50 streams against one sidecar. ``max_parallel_tasks`` is the bound, and it is the
        one the contract validates (it REQUIRES ``worktree``, since two tasks cannot share a checkout).

        **Order is by START, not by completion.** `self._completed` is what the completion and the
        ledger report, and a caller who wrote briefs A, B, C must not see them reported C, A, B because
        C happened to finish first. So a finished child's summary is placed at ITS index. That also
        keeps `remaining_tasks`/`RunProgress` correct across a continue-as-new, which keys by task id.

        **Goal mode is refused here rather than half-supported.** The goal loop is a sequence of
        boundaries — each verdict computed over "every task so far" and each CI gate reading the run's
        PR — so running two tasks across one boundary would make the verdict describe a state that
        never existed. `RunSpec.validate` already refuses the combination; this assertion is the code
        path agreeing with the contract.
        """
        if spec.goal is not None:  # pragma: no cover - refused by RunSpec.validate
            raise ValueError("goal mode is sequential: max_parallel_tasks must be 1")
        window = max(1, int(spec.max_parallel_tasks))
        pending: list[tuple[int, Any]] = []
        started = 0
        next_index = 0
        # The task the run is currently ON. A stop lands wherever the loop happens to be, so the
        # terminal names the task in the window rather than the next one nobody started.
        in_flight: str | None = None
        while next_index < len(remaining) or pending:
            await self._respect_pause()
            if self._stopped:
                await self._stop_window(pending)
                await self._finish(
                    "stopped",
                    f"run {spec.run_id}: stopped with {len(self._completed)} task(s) done",
                    task_id=in_flight,
                )
                await self._finalize("stopped")
                return self.completion("stopped")
            # Fill the window. Every guard the sequential loop applies is applied here too, in the
            # same order, so a concurrent run cannot spend money a sequential one would have refused.
            while next_index < len(remaining) and len(pending) < window:
                task = remaining[next_index]
                budget_left = run_budget_left(spec, self._spent)
                if budget_left is not None and budget_left <= 0:
                    await self._notify(
                        "note",
                        f"run {spec.run_id}: budget exhausted, not starting {task.task_id}",
                    )
                    await self._finish(
                        "budget_exhausted",
                        f"run {spec.run_id}: budget exhausted before {task.task_id}",
                        task_id=task.task_id,
                    )
                    await self._finalize("budget_exhausted")
                    return self.completion("budget_exhausted")
                # `start_child_workflow` is a COROUTINE: awaiting it STARTS the child and returns a
                # handle whose own `__await__` resolves at completion. So awaiting the start does not
                # serialise anything — every child in the window is running from the moment it was
                # started, and the handles are awaited later. (Not awaiting it, as an earlier draft
                # did, leaves a never-awaited coroutine and starts nothing.)
                pending.append(
                    (
                        next_index,
                        await workflow.start_child_workflow(
                            workflow_class_for(task.engine).run,
                            args=[task.to_payload(), budget_left],
                            id=f"task-{spec.run_id}-{task.task_id}",
                        ),
                    )
                )
                next_index += 1
                started += 1
                in_flight = task.task_id
            index, handle = pending.pop(0)
            # The awaited child is this run's in-flight work, so a stop can reach it while it is
            # parked — see `stop_delivery.py`. `_await_in_flight` keeps the run REACHABLE by the stop
            # and records the REST OF THE WINDOW too, since those children are just as parked.
            child = await self._await_in_flight(handle, window=tuple(other for _i, other in pending))
            self._spent += child.cost_usd
            self._completed.append(child.summary())
            in_flight = None
            # A child that returned BECAUSE it was told to stop leaves the run stopped, not completed:
            # its summary is kept, and the rest of the window is told so none of them keeps waiting.
            if self._stopped:
                await self._stop_window(pending)
                await self._report_publish_skips(spec)
                await self._finish(
                    "stopped",
                    f"run {spec.run_id}: stopped with {len(self._completed)} task(s) done",
                    task_id=_last_task_id(remaining),
                )
                await self._finalize("stopped")
                return self.completion("stopped")
            if spec.idle_between_tasks_s:
                await workflow.sleep(timedelta(seconds=spec.idle_between_tasks_s))

        # ORDER BY START. `self._completed` is what the completion and the ledger report, and a caller
        # who wrote briefs A, B, C must not read them back as C, A, B because C finished first. Kept by
        # task id (never by position) so `RunProgress`/`remaining_tasks` agree across a continue-as-new.
        order = {task.task_id: position for position, task in enumerate(remaining)}
        self._completed.sort(key=lambda summary: order.get(summary.task_id, len(order)))
        await self._report_publish_skips(spec)
        await self._finish(
            "completed",
            f"run {spec.run_id}: {len(self._completed)} task(s) done",
            task_id=_last_task_id(remaining),
        )
        await self._finalize("completed")
        return self.completion("completed")

    async def _report_publish_skips(self, spec: RunSpec) -> None:
        """The run's one summary of tasks that could not publish; `run_reporting.py` explains why."""
        await report_publish_skips(self._completed, spec.run_id, notify=self._notify)

    async def _run_tasks(
        self, spec: RunSpec, spec_payload: Mapping[str, Any], remaining: tuple[TaskSpec, ...]
    ) -> RunCompletion:
        """Start each remaining task in order, respecting pause, stop and the run's budget.

        Split out of `run` so the failure path can be wrapped once: `run` is the one place that knows
        both "this is the whole run" and "it is over", which is exactly what a finalisation point needs
        and what a helper per outcome would duplicate.

        **Goal mode adds a boundary, not a second loop.** At each task boundary the run evaluates the
        stop condition over all the evidence gathered so far (plus the CI gate, when the condition
        requires it), records the verdict in the ledger, and uses it for TWO things:

        * it starts a "fix CI" task while the gate is red and attempts remain, and
        * it decides the run's OUTCOME once the task list is exhausted.

        It does NOT abandon caller-supplied tasks. ``spec.tasks`` is the plan, and a run that
        discarded tasks 2 and 3 because task 1 happened to pass its own checks would be a run that
        silently did less than it was asked. What the condition bounds is when the run may CALL ITSELF
        DONE: exhausted tasks with unmet evidence finish ``goal_unmet``, never ``completed``.

        That evaluation is a pure function of `TaskSummary` values — workflow state — so a replay
        reaches the same decision without an activity or a clock.
        """
        progress = self._progress
        if spec.max_parallel_tasks > 1:
            # D7/MULTITASK: the concurrent path. Kept as a SEPARATE method rather than folded into the
            # loop below, because the loop below is what every recorded history replays through: its
            # command sequence (one child started, awaited, then the next) must not change by so much
            # as an extra `workflow.wait_condition` for a run that never asked for parallelism.
            return await self._run_tasks_concurrently(spec, spec_payload, remaining)
        started = 0
        for index, task in enumerate(remaining):
            await self._respect_pause()
            if self._stopped:
                # A stop that lands BETWEEN children has no child to tell: `_in_flight_handle` was
                # cleared when the last await returned, so this delivery is a no-op.
                await self._stop_in_flight()
                await self._finish(
                    "stopped",
                    f"run {spec.run_id}: stopped with {len(self._completed)} task(s) done",
                    task_id=task.task_id,
                )
                await self._finalize("stopped")
                return self.completion("stopped")
            if self._should_continue_as_new(started):
                # Keyword `args`: `continue_as_new`'s signature is ``(arg=..., *, args=[])``, so a
                # second POSITIONAL argument raises TypeError — which fails the workflow task, and
                # Temporal then retries it forever. The run wedges with children 3 and 4 never
                # started, and `status` times out. Measured while building D2.3; `args=[...]` is the
                # correct form for a multi-parameter ``run``.
                workflow.continue_as_new(
                    args=[
                        spec_payload,
                        RunProgress(
                            completed=tuple(self._completed),
                            spent_usd=self._spent,
                            generations=progress.generations + 1,
                            outcome=progress.outcome,
                            episodes=tuple(self._episodes),
                            ci_state=self._ci_state,
                            ci_attempts=self._ci_attempts,
                            goal_reason=self._goal_reason,
                        ).to_payload(),
                    ]
                )
            budget_left = run_budget_left(spec, self._spent)
            if budget_left is not None and budget_left <= 0:
                await self._notify(
                    "note", f"run {spec.run_id}: budget exhausted, not starting {task.task_id}"
                )
                await self._finish(
                    "budget_exhausted",
                    f"run {spec.run_id}: budget exhausted before {task.task_id}",
                    task_id=task.task_id,
                )
                await self._finalize("budget_exhausted")
                return self.completion("budget_exhausted")
            if self._goal_exhausted():
                await self._finish(
                    "goal_unmet",
                    f"run {spec.run_id}: goal exhausted before {task.task_id} "
                    f"({self._goal_reason})",
                    task_id=task.task_id,
                )
                await self._finalize("goal_unmet")
                return self.completion("goal_unmet")
            planned = await self._prepare_plan_context(task) if spec.goal is not None else task
            child = await self._run_task(planned, remaining_usd=budget_left)
            self._completed.append(child.summary())
            self._spent += child.cost_usd
            started += 1
            # A child that returned BECAUSE it was told to stop ends the run stopped: the cancel the
            # caller already had accepted must not turn into `completed` just because the task managed
            # to finish its own bookkeeping on the way out.
            if self._stopped:
                # A stop that arrived while the child was awaited was RECORDED by the handler and not
                # yet sent (the handler never sends). Send it here, from this coroutine, before the run
                # reports `stopped` — that is the invariant "the cancel was accepted means it was
                # delivered".
                await self._stop_in_flight()
                await self._report_publish_skips(spec)
                await self._finish(
                    "stopped",
                    f"run {spec.run_id}: stopped with {len(self._completed)} task(s) done",
                    task_id=task.task_id,
                )
                await self._finalize("stopped")
                return self.completion("stopped")
            if spec.goal is not None:
                await self._after_boundary(task, child, remaining=len(remaining) - index - 1)
            if spec.idle_between_tasks_s:
                await workflow.sleep(timedelta(seconds=spec.idle_between_tasks_s))

        if spec.goal is not None:
            # The list is exhausted: the condition decides whether the run did what it was asked.
            # `_after_boundary` has already recorded the verdict and run the CI gate to its bound.
            if self._goal_met():
                await self._report_publish_skips(spec)
                await self._finish(
                    "completed",
                    f"run {spec.run_id}: goal met after {len(self._completed)} task(s) "
                    f"({self._goal_reason})",
                    task_id=_last_task_id(remaining),
                )
                await self._finalize("completed")
                return self.completion("completed")
            await self._finish(
                "goal_unmet",
                f"run {spec.run_id}: tasks exhausted with the goal UNMET ({self._goal_reason})",
                task_id=_last_task_id(remaining),
            )
            await self._finalize("goal_unmet")
            return self.completion("goal_unmet")

        await self._report_publish_skips(spec)
        await self._finish(
            "completed",
            f"run {spec.run_id}: {len(self._completed)} task(s) done",
            task_id=_last_task_id(remaining),
        )
        # AFTER the "finished" notification, so a reader of the ledger sees the run complete before its
        # checkouts disappear — and so the sweep is the last thing that happens, never something a task
        # could still be using.
        await self._finalize("completed")
        return self.completion("completed")

    async def _report_capabilities(self, spec: RunSpec) -> None:
        """Record what the engine path DOES and does not carry, ONCE, at the start of a run (D5/D7).

        The sentences and the policy verdict live in `capability_report` (split out at the
        repository's line ceiling). This method only supplies the workflow's own notifier, so the
        report cannot grow a second way of reaching the ledger.
        """
        from .capability_report import report_capabilities

        await report_capabilities(spec, self._notify, logger=workflow.logger)

    async def _finalize(self, outcome: str) -> None:
        """THE run-level finalisation point: reclaim this run's per-task worktrees (D5).

        Every terminal exit from `run` goes through here — completed, stopped, budget_exhausted — which
        is what makes "a finished run leaks no worktree" a property of the code path rather than of an
        operator remembering. It is NOT reached by `continue_as_new`: that is not an ending, the next
        generation owns the same run directory and must keep the worktrees it may still work in.

        Two deliberate properties:

        * **Only a run that opted in is swept.** ``spec.worktree`` gates it, so a run that never created
          a worktree schedules NOTHING new. That is what keeps this addition replay-safe: an in-flight
          run's history has no cleanup activity, and replay must not produce one.
        * **A failed sweep never fails the run.** The tasks' commits are on their branches in the shared
          repository, so the run's RESULT is already complete; `worktree_cleanup` records a failure as a
          ledger note and returns. Raising here would turn a good run red over a directory that would
          not delete.
        """
        spec = self._spec
        if spec is None:  # pragma: no cover - set before any work
            return
        if not spec.worktree:
            return
        try:
            await workflow_activity(
                NAME_WORKTREE_CLEANUP,
                WorktreeCleanupRequest(
                    run_id=spec.run_id,
                    repo_dir=spec.repo_dir,
                    reason=f"run {outcome}",
                    keep=False,
                ),
            )
        except Exception:  # noqa: BLE001 - tidying must never un-finish a finished run
            workflow.logger.warning("run %s: reclaiming worktrees failed", spec.run_id)

    def _should_continue_as_new(self, started: int) -> bool:
        """Whether this execution is big enough to be replaced. Deterministic in workflow state."""
        spec = self._spec
        if spec is None:  # pragma: no cover
            return False
        if started >= spec.continue_as_new_tasks:
            return True
        return history_length() >= spec.continue_as_new_history_events

    async def _run_task(self, task: TaskSpec, *, remaining_usd: float | None) -> TaskCompletion:
        """Run one child task and hand back its completion.

        The child's workflow id is ``task-<run>-<task>``. A "fix CI" attempt needs no suffix: its
        ``task_id`` is already its own (``task-1-ci1``), while its WORKTREE is its parent's — which is
        exactly the split ``TaskSpec.worktree_task_id`` exists to express.

        Started and awaited separately so ``_in_flight_handle`` names the child for the whole time the
        run is blocked on it; the two calls are command-identical to ``execute_child_workflow``, so
        recorded histories replay unchanged. `stop_delivery.py` explains why that handle is the route
        a cancel needs.

        **The handle is SAVED AND RESTORED, never nulled.** This method is nested in practice: a
        goal-mode CI fix (``CiFixLoop._run_ci_fix``) calls it while the run is already inside the
        OUTER task's ``await handle``. Setting the field to ``None`` on the way out would clear the
        outer child's handle, and a stop arriving after a CI fix would then have nothing to deliver
        to — the exact stall `stop_delivery.py` exists to close. The previous value is restored
        instead, so each nesting level owns the field only for its own await.
        """
        spec = self._spec
        assert spec is not None
        handle = await workflow.start_child_workflow(
            workflow_class_for(task.engine).run,
            args=[task.to_payload(), remaining_usd],
            id=f"task-{spec.run_id}-{task.task_id}",
        )
        # A stop that landed between the start and here is delivered before the first await, so the
        # child never parks for a decision nobody is going to give. `_await_in_flight` owns the
        # save/restore of the handle, including for this nesting level.
        return await self._await_in_flight(handle)

    async def _respect_pause(self) -> None:
        if not self._paused:
            return
        await self._notify("note", f"run {self._spec.run_id if self._spec else '?'} paused")
        await workflow.wait_condition(lambda: not self._paused or self._stopped)

    async def _notify(
        self,
        kind: str,
        message: str,
        *,
        repo_dir: str = "",
        outcome: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Report one thing the run did.

        ``outcome`` is for the records that ARE an ending — `run.finished`, `task.finished`,
        `task.unfinished` — and rides in ``evidence``, which `activity_steps.notify` lifts onto the
        ledger row's top level. It defaults to None because most notifications are progress; a run that
        ends its own ledger writes one.

        ``run_terminal_evidence`` refuses a value outside the closed vocabulary, and that refusal is
        swallowed below like every other notify failure: the run's own work must never be failed by its
        audit row. ``task_id`` is empty for run-level rows, which is the shape every existing row has.
        """
        spec = self._spec
        if spec is None:  # pragma: no cover
            return
        try:
            evidence = {} if outcome is None else run_terminal_evidence(outcome)
            await workflow_activity(
                NAME_NOTIFY,
                NotifyRequest(
                    run_id=spec.run_id,
                    task_id=task_id,
                    kind=kind,
                    message=message,
                    repo_dir=repo_dir,
                    evidence=evidence,
                ),
            )
        except Exception:  # noqa: BLE001 - a notification must never fail a run
            workflow.logger.warning("run %s: notify(%s) failed", spec.run_id, kind)

    async def _notify_structured(
        self, kind: str, evidence: Mapping[str, Any], *, task_id: str | None
    ) -> None:
        """A record-shaped notification: the ledger row carries the evidence, not just a message.

        Used for the goal verdict, whose CHECKS an auditor must be able to read back. A plain
        ``_notify("note", "...")`` would leave the verdict as prose, and "the run said it was done"
        would only be checkable by re-deriving it. A failure is still non-fatal, for the same reason
        every other notification is: a broken audit sink must not un-finish a run.
        """
        spec = self._spec
        if spec is None:  # pragma: no cover
            return
        event = str(evidence.get("event") or kind)
        try:
            await workflow_activity(
                NAME_NOTIFY,
                NotifyRequest(
                    run_id=spec.run_id,
                    task_id=task_id,
                    kind=kind,
                    # The message is the human summary; the evidence is the machine-checkable record.
                    message=f"{event}: {_evidence_message(evidence)}",
                    evidence={str(key): value for key, value in evidence.items()},
                ),
            )
        except Exception:  # noqa: BLE001 - a record must never fail a run
            workflow.logger.warning("run %s: structured notify(%s) failed", spec.run_id, kind)

    def completion(self, outcome: RunOutcome) -> RunCompletion:
        spec = self._spec
        assert spec is not None
        unpublished = sum(1 for summary in self._completed if summary.unpublished)
        return RunCompletion(
            run_id=spec.run_id,
            tasks=tuple(self._completed),
            spent_usd=self._spent,
            generations=self._progress.generations,
            outcome=outcome,
            goal_reason=self._goal_reason,
            ci_attempts=self._ci_attempts,
            unpublished_tasks=unpublished,
        )

    # -- goal mode (D5) -------------------------------------------------------------------
    #
    # Everything here is a pure function of workflow state plus ONE activity call per boundary. The
    # verdict itself is never computed by an activity: a stop condition decided by a remote process
    # could not be replayed, and "the run said it was done" is the one claim a durable run must be
    # able to re-derive from its own history.

    def _evidence(self) -> tuple[GoalEvidence, ...]:
        """The evidence every completed task reported, in completion order."""
        return tuple(
            GoalEvidence(
                task_id=summary.task_id,
                outcome=summary.outcome,
                verify_passed=summary.verify_passed,
                review_verdict=summary.review_verdict,
            )
            for summary in self._completed
        )

    def _decision(self) -> StopDecision:
        spec = self._spec
        assert spec is not None and spec.goal is not None
        return evaluate_stop(
            spec.goal.stop_condition, self._evidence(), ci_state=self._ci_state
        )

    def _goal_met(self) -> bool:
        spec = self._spec
        if spec is None or spec.goal is None:
            return False
        decision = self._decision()
        self._goal_reason = decision.summary()
        if not decision.satisfied and self._ci_stop:
            self._goal_reason = f"{self._goal_reason} ({self._ci_stop})"
        return decision.satisfied

    def _goal_exhausted(self) -> bool:
        """Whether the goal has spent its whole allowance of attempted work.
        THE hard stop, checked BEFORE a task is started rather than after it finishes: a goal run
        whose tasks keep failing must not buy one more. ``max_tasks`` counts the tasks the run has
        ACTUALLY started (completed + this one), so a task that failed still spends the allowance —
        otherwise a repo whose verify always fails would loop forever inside one "task".
        """
        spec = self._spec
        if spec is None or spec.goal is None:
            return False
        started = len(self._completed) + 1
        if started > spec.goal.max_tasks:
            self._goal_reason = f"goal_unmet: max_tasks={spec.goal.max_tasks} reached"
            return True
        if self._ci_attempts >= spec.goal.max_ci_attempts and self._ci_unresolved():
            self._goal_reason = (
                f"goal_unmet: ci still red after {self._ci_attempts} attempt(s) "
                f"(max_ci_attempts={spec.goal.max_ci_attempts})"
            )
            return True
        return False

    def _ci_unresolved(self) -> bool:
        """Whether the CI gate is required and NOT green. The second independent stop bound."""
        spec = self._spec
        if spec is None or spec.goal is None or not spec.goal.stop_condition.ci:
            return False
        return self._ci_state != "passed"

    async def _prepare_plan_context(self, task: TaskSpec) -> TaskSpec:
        """THIS task's spec with the goal block its plan turn must see.

        ONE activity call (``recall``) and then a PURE formatting step. The recall returns references
        only; the recap text it produces is a function of the episode files, so a replay that reads
        the same run directory formats the same prompt.

        A task that already carries a goal context keeps it: this is additive, and a caller that
        built the prompt itself (a test, an operator resuming a task) is not overridden.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        if task.goal_context:
            return task
        refs = tuple(episode.to_payload() for episode in self._episodes)
        recalled = await workflow_activity(NAME_RECALL, RecallRequest(run_id=spec.run_id, refs=refs))
        block = format_objective_section(
            spec.goal.objective, [_recap_from_recalled(item) for item in recalled.episodes]
        )
        return task.with_goal_context(block)

    async def _recall_episode_ref(self, task: TaskSpec, child: TaskCompletion) -> EpisodeRef | None:
        """The episode reference a finished task reported, or None if it recorded none.

        Taken from the COMPLETION — not by calling `remember_episode` again, and not by deriving
        `episodes/<task>.json` in workflow code. Both alternatives would create a second truth: a
        second call appends a second ledger note for one episode, and a derived path is a second copy
        of the activity's naming rule.

        No activity call, so this is replay-safe by construction: it reads the child's result.
        """
        path = getattr(child, "episode_path", "") or ""
        digest = getattr(child, "episode_digest", "") or ""
        if not path or not digest:
            return None
        try:
            return EpisodeRef(path=path, digest=digest, task_id=task.task_id)
        except Exception:  # noqa: BLE001 - an unusable reference is reported, not trusted
            workflow.logger.warning("run: %s's episode reference is unusable", task.task_id)
            return None

    async def _after_boundary(
        self, task: TaskSpec, child: TaskCompletion, *, remaining: int = 0
    ) -> None:
        """Record the task's episode, run the CI gate, and record the run's verdict at this boundary.

        Three distinct things happen here, in this order:

        1. **Recall bookkeeping.** The just-finished task's episode is already on disk (its own
           `remember_episode` wrote it), and the path/digest come back on the COMPLETION — so nothing
           is derived here and no second episode note is written.
        2. **The CI gate.** Read ONLY when the stop condition requires it, so a goal that does not
           gate on CI schedules no GitHub read at all.
        3. **The verdict.** ``evaluate_stop`` over every task so far, recorded as a ledger note. The
           verdict is recomputed at the END of the task list too (it is pure), but recording it at
           each boundary is what makes the run's progress auditable while it is still running —
           ``remaining`` says how many tasks had not started yet when this verdict was taken.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        episode = await self._recall_episode_ref(task, child)
        if episode is not None:
            self._episodes.append(episode)
        if spec.goal.stop_condition.ci:
            await self._ci_gate(task, child)
        decision = self._decision()
        self._goal_reason = decision.summary()
        if not decision.satisfied and self._ci_stop:
            self._goal_reason = f"{self._goal_reason} ({self._ci_stop})"
        await self._record_verdict(decision, remaining=remaining, task_id=task.task_id)

    async def _record_verdict(self, decision: StopDecision, *, remaining: int, task_id: str) -> None:
        """One ledger note per boundary: what the run had PROVEN, not merely how much it spent."""
        spec = self._spec
        assert spec is not None
        await self._notify_structured(
            "goal.verdict",
            {
                "event": "goal.verdict",
                "satisfied": decision.satisfied,
                "reason": decision.reason,
                "checks": [list(entry) for entry in decision.checks],
                "tasks_done": len(self._completed),
                "tasks_remaining": remaining,
                "ci_state": self._ci_state,
                "ci_attempts": self._ci_attempts,
            },
            task_id=task_id,
        )


    @workflow.query
    def status(self) -> dict[str, Any]:
        spec = self._spec
        return {
            "run_id": spec.run_id if spec else None,            "tasks_total": len(spec.tasks) if spec else 0,
            "tasks_done": len(self._completed),
            "spent_usd": self._spent,
            "generations": self._progress.generations,
            "paused": self._paused,
            "stopped": self._stopped,
            "completed": [summary.__dict__ for summary in self._completed],
            # HOW MANY tasks finished their work WITHOUT publishing it (no GitHub remote to open a
            # pull request against). Reported beside the summaries rather than only inside them,
            # because it is the run-level question "did this run open anything?" and a UI that shows
            # a green run must be able to say why there is no pull request to click.
            "unpublished_tasks": sum(1 for summary in self._completed if summary.unpublished),
            # Goal mode. Reported even for a task-mode run so a client has ONE shape: `goal` is
            # null, and the two evidence fields are empty rather than absent.
            "goal": spec.goal.to_payload() if spec is not None and spec.goal is not None else None,
            "goal_reason": self._goal_reason,
            "ci_state": self._ci_state,
            "ci_attempts": self._ci_attempts,
            "episodes": [episode.to_payload() for episode in self._episodes],
            # D5 triggers: which deliveries this run has been told about, and which were repeats. A
            # redelivery is visible here rather than only in the ledger, so a UI can say "GitHub sent
            # this twice" without reading the event stream.
            "deliveries": list(self._deliveries),
            "duplicate_deliveries": list(self._duplicate_deliveries),
        }

    @workflow.signal
    def pause(self) -> None:
        self._paused = True

    @workflow.signal
    def resume(self) -> None:
        self._paused = False

    @workflow.signal(unfinished_policy=HandlerUnfinishedPolicy.ABANDON)
    def stop(self) -> None:
        """Stop the run — and RECORD that the child it is blocked on must be told.

        **This handler mutates state only.** It is a SYNCHRONOUS Temporal signal handler, so it must
        not await (an await would suspend the handler until the CHILD's signal resolved, and the child
        is waiting for this run) and must not spawn work (`asyncio.create_task` here is not gated on a
        Temporal event, so the command it eventually emits lands at a point a replay has no reason to
        reproduce). The send is driven from the main coroutine instead, at the boundary this signal
        always wakes: `_run_task`'s await on the child, or the loop's next `_stop_in_flight`. See
        `stop_delivery.py` for the replay argument in full.

        ``ABANDON`` says what happens if the run exits while a delivery is still sending: it is
        abandoned, not warned about. The sends are awaited by `_run_task`/`_stop_window` before the
        run reports `stopped`.
        """
        self._stopped = True
        self._paused = False

    @workflow.signal(name="webhook")
    def webhook(self, payload: Mapping[str, Any]) -> None:
        """A GitHub delivery for this run (D5): recorded, and recognised as a duplicate when it is one.

        The workflow id IS the delivery id, so a redelivered event with ``USE_EXISTING`` reaches THIS
        execution — the one already working — rather than starting a second. What must not happen is
        the signal being treated as fresh work: that would make a delivery GitHub retried twice spend
        twice.

        So the first delivery's id is remembered and every later one is recorded as a duplicate. The
        TODAY check is the id, not a clock: two signals that carry the same delivery id are the same
        event by definition, and comparing arrival times would make the answer depend on when the
        server happened to deliver them.

        Nothing is queued. A signal cannot start work in a run that has already finished its task
        list, and pretending otherwise (an in-memory queue the workflow would drain) would be a queue
        with none of the durability the queue it replaces had. What the signal BUYS is the record:
        ``webhook.received`` / ``webhook.duplicate`` in the ledger, so "GitHub sent this twice" is
        visible.
        """
        delivery_id = str(payload.get("delivery_id") or "").strip() if isinstance(payload, Mapping) else ""
        if not delivery_id:
            workflow.logger.warning("run: webhook signal carried no delivery_id; ignoring it")
            return
        if delivery_id in self._deliveries:
            self._duplicate_deliveries.append(delivery_id)
            return
        self._deliveries.append(delivery_id)
        self._webhook_payloads.append(dict(payload))


# -- helpers (module level: workflow code must not close over methods) -------------------------


def _recap_from_recalled(item: Any) -> EpisodeRecap:
    """A recalled episode as the pure formatter's input. One conversion, so both engines' prompts agree."""
    return EpisodeRecap(
        task_id=item.task_id,
        outcome=item.outcome,
        brief=item.brief,
        verify_passed=item.verify_passed,
        review_verdict=item.review_verdict,
        pr_url=item.pr_url,
        plan_ref=item.plan_ref,
    )

# The two PURE helpers below are imported from `workflow_support`, where they live beside the rest
# of the state arithmetic. Re-exported so a caller that reads them from this module (the CI-gate
# tests do) keeps working, and so "a pure helper belongs in the shared module" is one rule.
from .workflow_support import _evidence_message, _pr_coordinates  # noqa: E402,F401


def _last_task_id(tasks: tuple[TaskSpec, ...]) -> str | None:
    """The id of the last task a run's terminal reports on, or None when there is none.

    A run's terminal names the task its exit came from, so a reader can match it to THAT task's own
    verdict in a ledger written before the terminal carried an outcome. ``None`` rather than an
    exception for an empty list: a resumed run can arrive with every task already `continue_as_new`-ed
    away, and an ending is not an error just because this execution started none of them.
    """
    return tasks[-1].task_id if tasks else None
