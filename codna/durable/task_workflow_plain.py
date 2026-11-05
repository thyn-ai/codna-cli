"""The plain engine: the task as ordinary Temporal workflow code — the reference semantics.

`TaskWorkflowPlain` is the SAME six steps the LangGraph graph runs, written as a cursor over the pure
routers in `routing.py` with `workflow.wait_condition` approvals. It exists for three reasons the plan
depends on:

1. **Fallback.** ``graph_engine=plain`` runs when the LangGraph plugin is unavailable or its
   Public-Preview behaviour changes.
2. **Reference.** When the graph's behaviour is in doubt, this is the readable statement of what is
   supposed to happen — no plugin, no ``interrupt()``, no checkpointer.
3. **Parity.** The task suite runs the same brief through both and compares the route, the outcome and
   the reference set.

A cursor (`node = self._next(...)`) rather than an `if/elif` chain over the state: the latter is how
the two engines would silently diverge, because the plain version would re-derive "where am I" from
the state and could reach a different conclusion than the graph's edges for the same state.

**A park is terminal.** `wait_for_human` is the graph's last node, so this engine returns at the same
point. Making one engine continue after a park and not the other would be a divergence, and making the
graph continue would need a new node (a `GRAPH_VERSION` bump).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from temporalio import workflow

from .contracts import (
    MAX_STEER_NOTES,
    NAME_CHECKPOINT,
    NAME_EPISODE,
    NAME_NOTIFY,
    NAME_OPEN_PR,
    NAME_PLAN,
    NAME_REVIEW,
    NAME_TURN,
    NAME_VERIFY,
    EpisodeRequest,
    NotifyRequest,
    TaskCompletion,
    TaskContext,
    TaskOutcome,
    TaskSpec,
    implement_prompt,
    pr_title,
)
from .routing import after_plan, after_review, after_verify
from .state import TaskState
from .workflow_support import (
    adopt,
    task_completion,
    approver_identity,
    initial_state,
    steer_note,
    park_approval,
    park_reason,
    raised_cap,
    run_id_of,
    spec_from,
    workflow_activity,
)

__all__ = ["TaskWorkflowPlain"]


# -- the plain (reference) engine ---------------------------------------------------------


@workflow.defn(name="TaskWorkflowPlain")
class TaskWorkflowPlain:
    """The task as plain workflow code — the reference semantics the graph engine must match.

    A cursor (`next_node`) walks the same six steps the graph's edges do, applying the SAME pure
    router after each one. That is deliberate: writing the control flow as an `if` chain over the
    state, rather than as an explicit cursor, is how the two engines would silently diverge — the
    plain version would re-derive "where am I" from the state and could reach a different conclusion
    than the graph's edges for the same state.
    """

    def __init__(self) -> None:
        self._run_id: str = ""
        self._task_id: str = ""
        self._state: dict[str, Any] = {}
        self._spec: TaskSpec | None = None
        self._steer: list[str] = []
        self._paused = False
        self._stopped = False
        self._decision: dict[str, Any] | None = None
        self._approvals: list[dict[str, Any]] = []
        self._nodes_run: list[str] = []
        self._parked_on: str | None = None
        self._outcome: TaskOutcome | None = None
        self._spend: list[float] = [0.0]

    # -- the run -------------------------------------------------------------------

    @workflow.run
    async def run(
        self, spec_payload: Mapping[str, Any], remaining_usd: float | None = None
    ) -> TaskCompletion:
        spec = spec_from(spec_payload)
        self._spec = spec
        self._run_id, self._task_id = spec.run_id, spec.task_id
        state = initial_state(spec, remaining_usd)
        self._state = state.to_payload()
        await self._notify("task.started", f"task {spec.task_id} started (plain engine)")

        state = await self._walk(state)
        self._state = state.to_payload()
        outcome = self._outcome or "rejected"
        episode = await self._remember(state, outcome)
        await self._notify("task.finished", f"task {self._task_id} finished: {outcome}")
        return task_completion(
            state,
            outcome,
            self._approvals,
            "plain",
            self._spend[0],
            episode_path=episode.path,
            episode_digest=episode.digest,
        )

    async def _walk(self, state: TaskState) -> TaskState:
        """Walk the graph's edges as plain code, one step per activity call."""
        spec = self._spec
        assert spec is not None
        node = "plan"
        while True:
            await self._respect_pause()
            if self._stopped:
                self._outcome = "stopped"
                return state
            if node == "wait_for_human":
                # A park is terminal: the graph's approval node is its last, so both engines stop
                # here. The decision is recorded, the task finishes, and the human's answer is in
                # `approvals` for the ledger and the UI.
                return await self._park(state)
            state = await self._step(node, state)
            self._nodes_run.append(node)
            self._state = state.to_payload()
            node = self._next(node, state, spec)

    def _next(self, node: str, state: TaskState, spec: TaskSpec) -> str:
        """The next node, from the same routers the graph's conditional edges use."""
        if node == "plan":
            return after_plan(state, max_attempts=spec.max_attempts)
        if node == "implement":
            return "verify"  # the graph's plain edge
        if node == "verify":
            return after_verify(state, max_attempts=spec.max_attempts)
        if node == "review":
            return after_review(state, max_attempts=spec.max_attempts)
        if node == "open_pr":
            return "wait_for_human"  # the graph's plain edge
        raise ValueError(f"unknown node {node!r}")  # pragma: no cover - the cursor is closed

    async def _step(self, node: str, state: TaskState) -> TaskState:
        spec = self._spec
        assert spec is not None
        context = TaskContext.from_spec(spec)
        steer = tuple(self._steer)
        if node == "plan":
            outcome = await workflow_activity(
                NAME_PLAN,
                context.plan_request(
                    spec.brief, state.to_payload(), steer, spec.goal_context
                ),
            )
            return adopt(state, outcome, self._spend)
        if node == "implement":
            outcome = await workflow_activity(
                NAME_TURN,
                context.turn_request(implement_prompt(spec.brief, steer), state.to_payload(), steer),
            )
            moved = adopt(state, outcome, self._spend)
            # The checkpoint names the turn that just advanced, so it must receive the POST-turn
            # state. One commit per `turn_seq`, exactly as the graph's implement step does.
            checkpoint = await workflow_activity(NAME_CHECKPOINT, context.checkpoint_request(moved.to_payload()))
            return adopt(moved, checkpoint, self._spend)
        if node == "verify":
            outcome = await workflow_activity(
                NAME_VERIFY, context.verify_request(state.to_payload(), spec.verify_commands)
            )
            return adopt(state, outcome, self._spend)
        if node == "review":
            outcome = await workflow_activity(NAME_REVIEW, context.review_request(state.to_payload()))
            return adopt(state, outcome, self._spend)
        if node == "open_pr":
            outcome = await workflow_activity(
                NAME_OPEN_PR, context.pull_request_request(pr_title(spec), state.to_payload())
            )
            return adopt(state, outcome, self._spend)
        raise ValueError(f"unknown node {node!r}")  # pragma: no cover

    async def _park(self, state: TaskState) -> TaskState:
        """Suspend until a human decides.

        The suspension is a `wait_condition` on workflow state, so it survives a worker restart with
        nothing but history: no polling, no timer, no external store. `self._parked_on` keeps WHY we
        are waiting, which is what `status` reports to the IDE's "needs input" card.
        """
        reason = park_reason(state)
        self._parked_on = reason
        self._decision = None
        state = state.awaiting(park_approval(state, reason))
        self._state = state.to_payload()
        await self._notify("approval.requested", f"task {self._task_id} is waiting on a human ({reason})")
        await workflow.wait_condition(lambda: self._decision is not None or self._stopped)
        self._parked_on = None
        if self._stopped:
            self._outcome = "stopped"
            return state
        decision = self._decision or {}
        approved = decision.get("approved") is True
        identity = str(decision.get("identity") or "")
        self._approvals.append(
            {
                "identity": identity,
                "kind": "approve" if approved else "reject",
                "turn_seq": state.turn_seq,
                "reason": reason,
            }
        )
        if not approved:
            self._outcome = "rejected"
            await self._notify(
                "task.finished", f"task {self._task_id} rejected by {identity}", actor=identity
            )
            return state
        await self._notify(
            "approval.granted", f"task {self._task_id} approved by {identity}", actor=identity
        )
        # The OUTCOME is what happened to the task, not what the human said: a task parked on an
        # exhausted budget finished budget_exhausted, and saying "approved" would hide the reason it
        # stopped.
        self._outcome = "budget_exhausted" if reason == "budget_exhausted" else "approved"
        return state.resumed()

    async def _respect_pause(self) -> None:
        """Hold at a step boundary while paused. A pause never interrupts a paid turn."""
        if not self._paused:
            return
        await self._notify("note", f"task {self._task_id} paused")
        await workflow.wait_condition(lambda: not self._paused or self._stopped)

    async def _remember(self, state: TaskState, outcome: str) -> Any:
        """Record this task's episode and hand back the OUTCOME (path + digest).

        The returned reference is what the run carries across continue-as-new, so it must come from
        the activity that wrote the file — deriving it here would be a second copy of the naming
        rule. A failure is reported and yields an empty outcome: bookkeeping must not un-complete a
        task, and the run then knows this task contributed no episode.
        """
        from .contracts import EpisodeOutcome

        spec = self._spec
        if spec is None:  # pragma: no cover - set before any step
            return EpisodeOutcome(path="", digest="", ledger_seq=None)
        try:
            return await workflow_activity(
                NAME_EPISODE,
                EpisodeRequest(
                    state=state.to_payload(),
                    context=TaskContext.from_spec(spec),
                    outcome=outcome,
                    # The plan-side recap reads this, so a later task knows what this one was.
                    brief=spec.brief,
                ),
            )
        except Exception:  # noqa: BLE001 - bookkeeping must not un-complete a task
            workflow.logger.warning("task %s: recording the episode failed", self._task_id)
            return EpisodeOutcome(path="", digest="", ledger_seq=None)

    async def _notify(self, kind: str, message: str, *, actor: str = "") -> None:
        try:
            await workflow_activity(
                NAME_NOTIFY,
                NotifyRequest(
                    run_id=self._run_id,
                    task_id=self._task_id,
                    kind=kind,
                    message=message,
                    actor=actor,
                ),
            )
        except Exception:  # noqa: BLE001 - a notification must never fail a task
            workflow.logger.warning("task %s: notify(%s) failed", self._task_id, kind)

    # -- handlers ------------------------------------------------------------------

    @workflow.query
    def status(self) -> dict[str, Any]:
        state = self._safe_state()
        return {
            "engine": "plain",
            "run_id": self._run_id,
            "task_id": self._task_id,
            "turn_seq": state.turn_seq,
            "paused": self._paused,
            "stopped": self._stopped,
            "awaiting_approval": self._parked_on is not None and self._decision is None,
            "parked_on": self._parked_on,
            "pending_approval": state.pending_approval.to_payload() if state.pending_approval else None,
            # D4: WHAT the agent is waiting to be allowed to do. Present exactly when the turn
            # parked on an ask, so a UI can show the question instead of a generic park.
            "ask": state.ask.to_payload() if state.ask else None,
            "budget": state.budget.to_payload(),
            "budget_exhausted": state.budget.exhausted,
            "review_verdict": state.review_verdict,
            "verify_passed": state.verify_ref.passed if state.verify_ref else None,
            "pr": state.pr.to_payload() if state.pr else None,
            "approvals": list(self._approvals),
            "nodes_run": list(self._nodes_run),
            "outcome": self._outcome,
            "state": state.to_payload() if self._state else None,
        }

    def _safe_state(self) -> TaskState:
        """The current state, or a minimal valid placeholder before the first task has run.

        A query can arrive before the workflow task decodes its input. Returning a TaskState built
        from the execution identity keeps `status` total and valid rather than raising at whoever
        asked.
        """
        if self._state:
            return TaskState.from_payload(self._state)
        info = workflow.info()
        return TaskState(run_id=run_id_of(info.workflow_id), task_id=info.workflow_id)

    @workflow.update
    def approve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Approve the pending decision, recording WHO approved."""
        identity = approver_identity(payload, handler="approve")
        self._decision = {"identity": identity, "approved": True, "kind": "approve"}
        return {"applied": True, "identity": identity, "turn_seq": self._safe_state().turn_seq}

    @approve.validator
    def _validate_approve(self, payload: Mapping[str, Any]) -> None:
        """Refuse an anonymous or ill-timed approval BEFORE it is applied.

        A validator, not a check inside the handler: an Update handler that raises FAILS THE
        WORKFLOW TASK, and Temporal then retries that task forever — the run wedges, every query times
        out, and the caller sees "Timeout expired" rather than the reason. Measured, not assumed: this
        is exactly what happened before the validators existed. A validator's exception is returned to
        the caller as a failure and recorded as a rejected update, which is what "refused" means.
        """
        approver_identity(payload, handler="approve")
        if self._parked_on is None or self._decision is not None:
            raise ValueError("approve: this task is not waiting on a decision")

    @workflow.update
    def reject(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Refuse the pending decision, ending the task. Records who refused."""
        identity = approver_identity(payload, handler="reject")
        self._decision = {"identity": identity, "approved": False, "kind": "approve"}
        return {"applied": True, "identity": identity}

    @reject.validator
    def _validate_reject(self, payload: Mapping[str, Any]) -> None:
        approver_identity(payload, handler="reject")
        if self._parked_on is None or self._decision is not None:
            raise ValueError("reject: this task is not waiting on a decision")

    @workflow.update
    def steer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Add a steering note the NEXT turn carries. Never interrupts a running turn.

        Identity is required for the same reason an approval needs it: a note that steers an agent's
        work is an auditable act. The note is bounded here and secret-checked at the contract
        boundary, so a steer cannot push a credential into history.
        """
        identity = approver_identity(payload, handler="steer")
        note = steer_note(payload)
        self._steer.append(note)
        self._approvals.append(
            {"identity": identity, "kind": "steer", "turn_seq": self._safe_state().turn_seq}
        )
        return {"applied": True, "identity": identity, "notes": len(self._steer)}

    @steer.validator
    def _validate_steer(self, payload: Mapping[str, Any]) -> None:
        approver_identity(payload, handler="steer")
        steer_note(payload)
        if len(self._steer) >= MAX_STEER_NOTES:
            raise ValueError(f"steer: at most {MAX_STEER_NOTES} notes are carried into a turn")

    @workflow.update
    def set_budget(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Raise the task's cost cap. Raises only; a lower cap would retroactively blow the budget."""
        identity = approver_identity(payload, handler="set_budget")
        state = self._safe_state()
        raised = raised_cap(state.budget.max_cost_usd, payload.get("max_cost_usd"))
        self._state = replace(state, budget=replace(state.budget, max_cost_usd=raised)).to_payload()
        self._approvals.append(
            {
                "identity": identity,
                "kind": "set_budget",
                "max_cost_usd": raised,
                "turn_seq": state.turn_seq,
            }
        )
        return {"applied": True, "identity": identity, "max_cost_usd": raised}

    @set_budget.validator
    def _validate_set_budget(self, payload: Mapping[str, Any]) -> None:
        approver_identity(payload, handler="set_budget")
        raised_cap(self._safe_state().budget.max_cost_usd, payload.get("max_cost_usd"))

    @workflow.signal
    def pause(self) -> None:
        """Hold at the next step boundary. Never interrupts a paid turn."""
        self._paused = True

    @workflow.signal
    def resume(self) -> None:
        self._paused = False

    @workflow.signal
    def stop(self) -> None:
        """Stop at the next boundary. Distinct from cancel: a stop still records the episode."""
        self._stopped = True
        self._paused = False

    @workflow.query
    def approvals(self) -> list[dict[str, Any]]:
        return list(self._approvals)


