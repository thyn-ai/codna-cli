"""The run: one child task per task, a run-level budget, and continue-as-new.

**Continue-as-new is how a long run stays replayable.** Temporal history is finite; a run with thirty
tasks and ten turns each would pass the 10,240-event ceiling, become slow to replay, and eventually be
rejected. So the workflow carries its progress in its INPUT (`RunProgress`) and restarts itself with
``continue_as_new`` when either bound is crossed — every 10 completed tasks, or once this execution's
history passes 5,000 events. The input is small and fixed-size, so the restart cannot itself grow
without bound.

**Why the arguments are keyword `args`.** ``workflow.continue_as_new``'s signature is
``(arg=..., *, args=[...])``, so passing a SECOND positional argument raises ``TypeError`` — which
fails the workflow task, and Temporal then retries it forever. The run wedges with its later tasks
never started, and every query on it times out. Measured while building D2.3; `args=[...]` is the only
correct form for a multi-parameter ``run``.

**`remaining_usd` is the run's cap minus what its children reported.** A task that exhausts its own cap
parks on a human; a run whose TOTAL is exhausted stops starting tasks and returns ``budget_exhausted``.
Two different questions — "may this task spend more?" vs. "may this run buy another task?" — which is
why the budget is checked in both places.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping

from temporalio import workflow

from .contracts import (
    NAME_NOTIFY,
    NAME_RECALL,
    NAME_WATCH_CI,
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
    WatchCiRequest,
    WorktreeCleanupRequest,
)
from .goal import (
    EpisodeRecap,
    EpisodeRef,
    GoalEvidence,
    StopDecision,
    build_ci_fix_brief,
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
class RunWorkflow:
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
        await self._notify("run.started", f"run {spec.run_id}: {len(remaining)} task(s) to go")
        await self._report_capabilities(spec)

        try:
            return await self._run_tasks(spec, spec_payload, remaining)
        except Exception:
            # A run that FAILED is still a run whose worktrees must not leak. `Exception`, not
            # `BaseException`: `continue_as_new` raises a `BaseException`, and that is not an ending —
            # the next generation keeps the same run directory and may still work in those checkouts.
            await self._finalize("failed")
            raise

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
        while next_index < len(remaining) or pending:
            await self._respect_pause()
            if self._stopped:
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
            index, handle = pending.pop(0)
            child = await handle
            self._spent += child.cost_usd
            self._completed.append(child.summary())
            if spec.idle_between_tasks_s:
                await workflow.sleep(timedelta(seconds=spec.idle_between_tasks_s))

        # ORDER BY START. `self._completed` is what the completion and the ledger report, and a caller
        # who wrote briefs A, B, C must not read them back as C, A, B because C finished first. Kept by
        # task id (never by position) so `RunProgress`/`remaining_tasks` agree across a continue-as-new.
        order = {task.task_id: position for position, task in enumerate(remaining)}
        self._completed.sort(key=lambda summary: order.get(summary.task_id, len(order)))
        await self._report_publish_skips(spec)
        await self._notify(
            "run.finished", f"run {spec.run_id}: {len(self._completed)} task(s) done"
        )
        await self._finalize("completed")
        return self.completion("completed")

    async def _report_publish_skips(self, spec: RunSpec) -> None:
        """Record ONCE, at the end, which tasks finished without publishing — and why.

        The task's own ledger note (`publish.skipped`, written by the publish step) says what
        happened to it; this summary is what makes the RUN's ledger answer "did this run open
        anything?" without a reader walking every task's records. A run whose tasks all published
        writes nothing new, so an existing history replays with the exact record set it had.

        A notification, never a failure: the work is done, and the reason (no GitHub remote) is a
        fact about the repository rather than a defect in the run.
        """
        skipped = tuple(summary for summary in self._completed if summary.unpublished)
        if not skipped:
            return
        names = ", ".join(summary.task_id for summary in skipped)
        await self._notify(
            "note",
            f"run {spec.run_id}: {len(skipped)} task(s) finished without a pull request "
            f"({names}): the repository has nothing to publish to (no GitHub remote); the commits "
            "are on each task's branch.",
        )

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
                await self._finalize("budget_exhausted")
                return self.completion("budget_exhausted")
            if self._goal_exhausted():
                await self._finalize("goal_unmet")
                return self.completion("goal_unmet")
            planned = await self._prepare_plan_context(task) if spec.goal is not None else task
            child = await self._run_task(planned, remaining_usd=budget_left)
            self._completed.append(child.summary())
            self._spent += child.cost_usd
            started += 1
            if spec.goal is not None:
                await self._after_boundary(task, child, remaining=len(remaining) - index - 1)
            if spec.idle_between_tasks_s:
                await workflow.sleep(timedelta(seconds=spec.idle_between_tasks_s))

        if spec.goal is not None:
            # The list is exhausted: the condition decides whether the run did what it was asked.
            # `_after_boundary` has already recorded the verdict and run the CI gate to its bound.
            if self._goal_met():
                await self._report_publish_skips(spec)
                await self._notify(
                    "run.finished",
                    f"run {spec.run_id}: goal met after {len(self._completed)} task(s) "
                    f"({self._goal_reason})",
                )
                await self._finalize("completed")
                return self.completion("completed")
            await self._notify(
                "note",
                f"run {spec.run_id}: tasks exhausted with the goal UNMET ({self._goal_reason})",
            )
            await self._finalize("goal_unmet")
            return self.completion("goal_unmet")

        await self._report_publish_skips(spec)
        await self._notify("run.finished", f"run {spec.run_id}: {len(self._completed)} task(s) done")
        # AFTER the "finished" notification, so a reader of the ledger sees the run complete before its
        # checkouts disappear — and so the sweep is the last thing that happens, never something a task
        # could still be using.
        await self._finalize("completed")
        return self.completion("completed")

    async def _report_capabilities(self, spec: RunSpec) -> None:
        """Record what the engine path DOES and does not carry, ONCE, at the start of a run (D5).

        MCP is reported in BOTH directions, and the two notes mean different things: an operator who
        configured servers learns from the run which ones were handed to the engine, and an operator
        who configured none reads that the field exists rather than assuming their IDE-level server
        was consulted. A one-sided note (only "not attached") cannot distinguish "carried" from
        "silently dropped", which is the confusion this whole area exists to remove.

        Deliberately a notification, not a failure: a run must not refuse to start because one
        optional integration is unavailable, and the information is only actionable for an operator
        who expected it.
        """
        from .rules import mcp_attachment_state
        from .swarm import swarm_attachment_state

        # Built from the RUN's own tasks, not from an empty context: a note that said "no servers
        # configured" for a run whose tasks all configured one would be the exact wrong-audit-note this
        # reporting exists to avoid. A run's tasks may differ, so the union is what was carried.
        servers = tuple(
            server
            for task in spec.tasks
            for server in task.mcp_servers
            if server.get("disabled") is not True
        )
        attachment = mcp_attachment_state(type("_RunMcpView", (), {"mcp_servers": servers})())
        if attachment.attached:
            await self._notify(
                "note",
                f"run {spec.run_id}: MCP is attached — servers: {', '.join(attachment.servers)}",
            )
        else:
            await self._notify("note", f"run {spec.run_id}: MCP is not attached — {attachment.reason}")
        # D7/SWARM: the same two-way report for delegation, and for the same reason — an operator who
        # asked for a swarm learns from the run that it was carried, and one who did not reads that the
        # field exists rather than assuming their expectation was silently ignored.
        swarm = swarm_attachment_state(
            type(
                "_RunSwarmView",
                (),
                {
                    "enable_spawn_agent": any(task.enable_spawn_agent for task in spec.tasks),
                    "enable_agent_teams": any(task.enable_agent_teams for task in spec.tasks),
                },
            )()
        )
        if swarm.attached:
            carried = [
                name
                for name, on in (
                    ("enable_spawn_agent", swarm.spawn),
                    ("enable_agent_teams", swarm.teams),
                )
                if on
            ]
            await self._notify(
                "note",
                f"run {spec.run_id}: delegation is attached — {', '.join(carried)}",
            )
        else:
            await self._notify(
                "note", f"run {spec.run_id}: delegation is not attached — {swarm.reason}"
            )

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
        exactly the split `TaskSpec.worktree_task_id` exists to express.
        """
        spec = self._spec
        assert spec is not None
        return await workflow.execute_child_workflow(
            workflow_class_for(task.engine).run,
            args=[task.to_payload(), remaining_usd],
            id=f"task-{spec.run_id}-{task.task_id}",
        )

    async def _respect_pause(self) -> None:
        if not self._paused:
            return
        await self._notify("note", f"run {self._spec.run_id if self._spec else '?'} paused")
        await workflow.wait_condition(lambda: not self._paused or self._stopped)

    async def _notify(self, kind: str, message: str) -> None:
        spec = self._spec
        if spec is None:  # pragma: no cover
            return
        try:
            await workflow_activity(
                NAME_NOTIFY,
                NotifyRequest(run_id=spec.run_id, task_id=None, kind=kind, message=message),
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

    async def _ci_gate(self, task: TaskSpec, child: TaskCompletion) -> None:
        """Read CI, and while it is red start ONE fix task and read again — bounded twice over.

        **This is the loop, and it has two independent exits.** ``max_ci_attempts`` bounds how many
        fix tasks one red PR may spend; the RUN's budget bounds how much money it may spend. Either
        alone stops it: an attempt cap with no budget cap would spend the whole run on one PR, and a
        budget cap with no attempt cap would keep buying attempts until the run died. Both are
        checked before every fix task, so the loop cannot outlive either.

        A ``failed`` CI state with no attempts left is not retried and not an error: the state stays
        ``failed``, `evaluate_stop` reports ``ci=failed``, and ``self._ci_stop`` says WHICH bound
        ended the loop — the difference between "we tried three times" and "we ran out of money".

        ``infrastructure``, ``pending`` and ``unknown`` never start a fix task. The first two are not
        this run's code; the third means the rollup could not be read, and patching code to satisfy a
        check we could not see is the failure `webhook_ci_triage` exists to prevent.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        while True:
            pr_number, head_sha = _pr_coordinates(child)
            if pr_number is None:
                # No pull request on this task, so there is no head for CI to judge. Leaving the state
                # alone is deliberate: `evaluate_stop` then reports `ci=failed`, the honest answer for
                # a goal that requires a green PR and has none.
                return
            outcome = await workflow_activity(
                NAME_WATCH_CI,
                WatchCiRequest(
                    run_id=spec.run_id,
                    repo_dir=spec.repo_dir,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    base_branch=spec.base_branch,
                ),
            )
            self._ci_state = outcome.state
            if outcome.state != "failed":
                return
            if self._ci_attempts >= spec.goal.max_ci_attempts:
                self._ci_stop = (
                    f"ci still red after {self._ci_attempts} attempt(s) "
                    f"(max_ci_attempts={spec.goal.max_ci_attempts})"
                )
                await self._notify(
                    "note",
                    f"run {spec.run_id}: CI still red on PR #{pr_number} after "
                    f"{self._ci_attempts} fix attempt(s); not starting another "
                    f"(max_ci_attempts={spec.goal.max_ci_attempts})",
                )
                return
            budget_left = run_budget_left(spec, self._spent)
            if budget_left is not None and budget_left <= 0:
                self._ci_stop = (
                    f"ci still red and the run budget is exhausted "
                    f"({self._ci_attempts} attempt(s) of {spec.goal.max_ci_attempts})"
                )
                await self._notify(
                    "note",
                    f"run {spec.run_id}: CI fix loop stopped by the run budget, not by the attempt cap",
                )
                return
            child = await self._run_ci_fix(
                task, child, pr_number=pr_number, head_sha=head_sha, checks=outcome.checks,
                remaining_usd=budget_left,
            )

    async def _run_ci_fix(
        self,
        task: TaskSpec,
        child: TaskCompletion,
        *,
        pr_number: int,
        head_sha: str | None,
        checks: Any,
        remaining_usd: float | None,
    ) -> TaskCompletion:
        """Start ONE "fix CI" task on the same branch and consolidate its result into the run.

        The fix task is a REAL child task: its own plan/implement/verify/review/PR steps, against the
        same worktree and the same pull request. That is what makes the next CI read describe the
        fix — anything cheaper (a turn with a synthetic prompt) would not produce a commit on the
        branch the check suite is running against.
        """
        spec = self._spec
        assert spec is not None and spec.goal is not None
        attempt = self._ci_attempts + 1
        self._ci_attempts = attempt
        brief = build_ci_fix_brief(
            pr_number=pr_number, head_sha=head_sha, checks=checks, attempt=attempt
        )
        fix = task.with_ci_fix(
            task_id=f"{task.task_id}-ci{attempt}", brief=brief, goal_context=task.goal_context
        )
        try:
            completion = await self._run_task(fix, remaining_usd=remaining_usd)
        except Exception as exc:  # noqa: BLE001 - a failed fix attempt is an ANSWER, not a crash
            # A fix that cannot even complete (its turn failed, its verify blew up) must not turn the
            # RUN red: the goal is merely still unmet, and the CI loop's bound decides whether to try
            # again. Escaping here would lose the run's whole result over one repair attempt — and the
            # attempt has already been counted, so the loop cannot spin on it.
            await self._notify(
                "note",
                f"run {spec.run_id}: fix CI attempt {attempt} FAILED "
                f"({type(exc).__name__}) on PR #{pr_number}; counting it against max_ci_attempts",
            )
            return child
        self._completed.append(completion.summary())
        self._spent += completion.cost_usd
        episode = await self._recall_episode_ref(fix, completion)
        if episode is not None:
            self._episodes.append(episode)
        await self._notify(
            "note",
            f"run {spec.run_id}: fix CI attempt {attempt} finished ({completion.outcome}) "
            f"on PR #{pr_number}",
        )
        return completion

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

    @workflow.signal
    def stop(self) -> None:
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


def _pr_coordinates(child: Any) -> tuple[int | None, str | None]:
    """A finished task's pull request: ``(number, head_sha)``, either of which may be None.

    Read from the state rather than from the `TaskSummary`, because the summary carries the URL (what
    a human reads) and the CI gate needs the NUMBER (what the API takes); deriving one from the other
    would mean parsing a URL in workflow code.

    The head sha falls back to the checkpoint's commit when the PR ref did not record one: that is
    the commit this run pushed, which is the commit CI judges. A PR opened without a head sha (an
    older spec, or a provider that omits it) therefore still gates on the right commit.
    """
    state = getattr(child, "state", None)
    if not isinstance(state, dict):
        return None, None
    pr = state.get("pr")
    number: int | None = None
    head_sha: str | None = None
    if isinstance(pr, dict):
        raw = pr.get("number")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            number = raw
        if isinstance(pr.get("head_sha"), str) and pr["head_sha"]:
            head_sha = str(pr["head_sha"])
    if head_sha is None:
        checkpoint = state.get("checkpoint_ref")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("sha"), str):
            head_sha = str(checkpoint["sha"])
    return number, head_sha



def _evidence_message(evidence: Mapping[str, Any]) -> str:
    """A one-line human summary of a structured notification, from the evidence itself.

    Deterministic in the evidence, so the ledger row replays to the same string. Kept small: the
    evidence is the record, this is the sentence a reader skims.
    """
    parts: list[str] = []
    for key in ("satisfied", "reason", "tasks_done", "tasks_remaining", "ci_state", "ci_attempts"):
        if key in evidence:
            parts.append(f"{key}={evidence[key]}")
    checks = evidence.get("checks")
    if isinstance(checks, list) and checks:
        parts.append(
            "checks="
            + ",".join(
                f"{entry[0]}-{entry[1]}"
                for entry in checks
                if isinstance(entry, list) and len(entry) == 2
            )
        )
    return " ".join(parts) or "notified"
