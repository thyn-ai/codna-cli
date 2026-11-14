"""The graph engine: the task driven by the LangGraph graph under `LangGraphPlugin`.

The plugin runs the graph; five of its nodes are activities (`graph.NODE_EXECUTION`) and the approval
node is an in-workflow `interrupt()`. This workflow hands the graph a state and a config, then resumes
the interrupt with the human's decision.

**Superstep driving.** The graph is advanced with ``astream`` and the workflow acts BETWEEN supersteps.
One ``ainvoke`` would hand control back only at the interrupt, so ``pause``, ``stop`` and ``steer``
could not take effect until the task had already spent its money — the plain engine honours them at
every step boundary, so this one must too.

**The stream is closed on park.** LangGraph raises ``GeneratorExit`` into it on early exit, and a
stream left open across a workflow suspension would be resumed under a different workflow task — the
concrete bug the D2.2 probe found, so the close is not optional bookkeeping.

**State equality with the plain engine.** The graph's final state is a `TaskState` payload (the nodes
return exactly that), so both engines return the same shape and the parity test compares them field by
field.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from temporalio import workflow

from .contracts import (
    MAX_STEER_NOTES,
    NAME_EPISODE,
    NAME_NOTIFY,
    EpisodeRequest,
    NotifyRequest,
    TaskCompletion,
    TaskContext,
    TaskOutcome,
    TaskSpec,
)
from .state import TaskState
from .parking import park_reason
from .approval_policy import AUTO_APPROVAL_REASON, auto_approval_eligible
from .workflow_support import (
    approved_outcome,
    task_completion,
    approver_identity,
    initial_state,
    steer_note,
    raised_cap,
    run_id_of,
    spec_from,
    workflow_activity,
)

__all__ = ["TaskWorkflow"]


def resume_command(decision: Mapping[str, Any]) -> Any:
    """The ``Command(resume=...)`` that completes the graph's approval node.

    ``langgraph`` is imported HERE, not at module scope: this module is registered by a worker that
    may run only the plain engine, and a module-level langgraph import would make the plain path
    depend on the extra it exists to replace.
    """
    from langgraph.types import Command

    return Command(resume=dict(decision))


# -- the graph engine ---------------------------------------------------------------------


@workflow.defn(name="TaskWorkflow")
class TaskWorkflow:
    """The same task, driven by the LangGraph graph under `LangGraphPlugin`.

    The plugin runs the graph; five of its nodes are activities and the approval node is an
    in-workflow `interrupt()`. This workflow's job is to hand the graph a state and a config, then
    resume the interrupt with the human's decision.

    **Superstep driving.** The graph is advanced with ``astream`` and the workflow acts BETWEEN
    supersteps. One ``ainvoke`` would hand control back only at the interrupt, so ``pause``,
    ``stop`` and ``steer`` could not take effect until the task had already spent its money — the
    plain engine honours them at every step boundary, so the graph engine must too.
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
        self._interrupts: list[dict[str, Any]] = []
        self._nodes_run: list[str] = []
        self._outcome: TaskOutcome | None = None
        self._auto_approved = False
        self._spend: list[float] = [0.0]

    @workflow.run
    async def run(
        self, spec_payload: Mapping[str, Any], remaining_usd: float | None = None
    ) -> TaskCompletion:
        spec = spec_from(spec_payload)
        self._spec = spec
        self._run_id, self._task_id = spec.run_id, spec.task_id
        state = initial_state(spec, remaining_usd)
        self._state = state.to_payload()
        await self._notify("task.started", f"task {spec.task_id} started (langgraph engine)")

        await self._drive(state.to_payload())
        final = TaskState.from_payload(self._state)
        # The fallback (`self._outcome` never set) is the same rule as an approval: a task whose
        # review approved finished on success, and whether it was PUBLISHED is read from the state by
        # the SHARED helper. `pr is not None` used to stand in for "approved", which would call a
        # legitimately unpublished task rejected — the defect this change removes.
        outcome = self._outcome or (
            approved_outcome(final) if final.review_verdict == "approved" else "rejected"
        )
        episode = await self._remember(final, outcome)
        await self._notify("task.finished", f"task {self._task_id} finished: {outcome}")
        return task_completion(
            final,
            outcome,
            self._approvals,
            "langgraph",
            self._spend[0],
            episode_path=episode.path,
            episode_digest=episode.digest,
        )

    async def _drive(self, initial: Mapping[str, Any]) -> None:
        """Run the graph superstep by superstep, resuming each interrupt with a human's decision."""
        app = self._compile()
        pending: Any = dict(initial)
        while True:
            await self._respect_pause()
            if self._stopped:
                self._outcome = "stopped"
                return
            interrupted = await self._drain(app, pending)
            if self._auto_approved:
                return
            if not interrupted:
                return
            decision = await self._await_decision()
            if decision is None:
                return
            pending = resume_command(decision)

    async def _drain(self, app: Any, pending: Any) -> bool:
        """Advance the graph until it parks or ends. Returns whether it parked.

        The stream is CLOSED explicitly on park. LangGraph raises ``GeneratorExit`` into it on early
        exit, and a stream left open across a workflow suspension would be resumed under a different
        workflow task — the concrete bug the D2.2 probe found, so the close is not optional.
        """
        stream = app.astream(pending, self._graph_config(), stream_mode="updates")
        interrupted = False
        try:
            async for chunk in stream:
                for node_name, value in (chunk or {}).items():
                    if node_name == "__interrupt__":
                        await self._record_interrupts(value)
                        interrupted = True
                        continue
                    self._nodes_run.append(str(node_name))
                    if isinstance(value, Mapping) and value.get("auto_approved") is True:
                        await self._auto_approve()
                        continue
                    self._absorb(value)
                if interrupted:
                    break
        finally:
            await stream.aclose()
        return interrupted

    async def _auto_approve(self) -> dict[str, bool]:
        """Record a policy grant, never a fabricated human identity or `approval.granted`."""
        spec = self._spec
        if spec is None:
            raise ValueError("auto approval requires a validated task policy")
        current = TaskState.from_payload(self._state)
        if not auto_approval_eligible(current, enabled=spec.auto_approve_local_only):
            raise ValueError("auto approval requires verified, reviewed local-only evidence")
        self._approvals.append({
            "identity": None,
            "kind": "auto_approved",
            "turn_seq": current.turn_seq,
            "reason": AUTO_APPROVAL_REASON,
            "policy_digest": spec.approval_policy_digest,
        })
        self._outcome = approved_outcome(current)
        await self._record_auto_approval(NotifyRequest(
            run_id=self._run_id,
            task_id=self._task_id,
            kind="auto_approved",
            message=(
                f"task {self._task_id} completed unattended under policy "
                f"{AUTO_APPROVAL_REASON}: no remote, verify passed, review approved"
            ),
            evidence={
                "policy_digest": spec.approval_policy_digest,
                "policy_mode": "auto_for_low_risk",
                "park_reason": AUTO_APPROVAL_REASON,
                "verify_passed": True,
                "review_verdict": current.review_verdict,
                "publish_skip": current.publish_skip,
            },
        ))
        self._auto_approved = True
        return {"auto_approved": True}

    async def _record_auto_approval(self, request: NotifyRequest) -> None:
        """The same durable notification boundary as other audit events, without swallowing failure."""
        await workflow_activity(NAME_NOTIFY, request)

    def _absorb(self, value: Any) -> None:
        """Adopt a node's state payload, accumulating the spend it added.

        The node returns a full `TaskState` payload built from the one it was handed, so the newest
        payload wins. Spend is a DELTA against the previous payload's budget, which is what keeps the
        run-level total from double counting across turns.
        """
        if not isinstance(value, Mapping):
            return
        try:
            current = TaskState.from_payload(value)
            previous = TaskState.from_payload(self._state) if self._state else None
        except ValueError:
            # A node that returned something undecodable is a bug in that node; the workflow keeps the
            # last valid state rather than propagating half a payload, and the run parks on the first
            # step that cannot proceed (the routing functions see the state they trust).
            workflow.logger.error("task %s: a graph node returned an invalid state", self._task_id)
            return
        self._state = current.to_payload()
        if previous is not None and current.budget.spent_usd > previous.budget.spent_usd:
            self._spend[0] += current.budget.spent_usd - previous.budget.spent_usd

    async def _record_interrupts(self, value: Any) -> None:
        """Arm the gate, then keep the interrupt payloads queryable.

        **Arming happens HERE, before any await, and that ordering is the fix for a real wedge.**
        ``status`` reports ``awaiting_approval`` as soon as ``self._interrupts`` is non-empty, and the
        ``approve`` validator accepts an Update on exactly that condition. Recording the interrupts
        and then awaiting ``approval.requested`` therefore publishes a gate that is open to Updates
        while ``_await_decision`` has not run yet. An Update processed in that window is ACCEPTED by
        the validator and writes ``self._decision`` — which ``_await_decision`` used to clear on
        entry, discarding a decision the caller was already told was applied. The workflow then
        waited forever for a decision that had arrived, and the run only failed on the caller's
        timeout (measured in CI, never locally: the window is the width of one notify round-trip,
        so it opens on a loaded runner and not on an idle one).

        Clearing the decision and setting the interrupts in the same synchronous stretch — no await
        between them — makes "parked" and "ready to accept" the same instant, and makes an accepted
        decision survive. `TaskWorkflowPlain` has always had this ordering (``_park`` clears the
        decision before it notifies and before it waits); this engine now matches it.
        """
        self._decision = None
        self._interrupts = []
        for item in value or ():
            payload = getattr(item, "value", None)
            if isinstance(payload, Mapping):
                self._interrupts.append(dict(payload))
        if self._interrupts:
            await self._notify(
                "approval.requested",
                f"task {self._task_id} is waiting on a human ({self._interrupts[0].get('kind')})",
            )

    async def _await_decision(self) -> dict[str, Any] | None:
        """Suspend until a human decides. Returns the resume value, or None when refused/stopped.

        The decision is CLEARED when it is consumed, not on entry. Clearing on entry discarded an
        Update the validator had already accepted: the gate opens in `_record_interrupts` (which
        necessarily awaits the ``approval.requested`` activity), so a decision can be recorded before
        this coroutine is ever entered. See `_record_interrupts` for the measured wedge.
        """
        await workflow.wait_condition(lambda: self._decision is not None or self._stopped)
        parked = dict(self._interrupts[0]) if self._interrupts else {}
        self._interrupts = []
        if self._stopped:
            self._outcome = "stopped"
            return None
        decision = self._decision or {}
        self._decision = None
        identity = str(decision.get("identity") or "")
        approved = decision.get("approved") is True
        # The reason is the one the graph recorded when it parked (`park_reason`), not a re-derivation
        # here. `open_pr_ready` used to be the value for ANY non-budget park, which made an approval
        # granted after three failed verifies indistinguishable in the audit trail from one granted on
        # work that passed. Re-deriving it here would be a second copy of the rule; the prompt is
        # already the channel that carries it, and `budget_exhausted` stays a separate field because
        # it is a property of the budget rather than of the parked step.
        #
        # The prompt is absent when this workflow is REPLAYING a park recorded before the prompt
        # carried the reason, and a missing key must not become `reason: ""` — an empty string is not
        # in the vocabulary `park_reason` defines, and an approval record with no reason is worse than
        # the `open_pr_ready` it replaced. So the fallback derives it from the replayed STATE with the
        # same shared function: that is the rule itself, not a second copy of it, and it is
        # deterministic on replay because `park_reason` is a pure function of `TaskState`.
        current = TaskState.from_payload(self._state)
        parked_reason = str(parked.get("park_reason") or "") or park_reason(current)
        self._approvals.append(
            {
                "identity": identity,
                "kind": "approve" if approved else "reject",
                "turn_seq": current.turn_seq,
                "reason": "budget_exhausted" if parked.get("budget_exhausted") else parked_reason,
            }
        )
        if not approved:
            self._outcome = "rejected"
            await self._notify(
                "task.finished", f"task {self._task_id} rejected by {identity}", actor=identity
            )
            return None
        await self._notify(
            "approval.granted", f"task {self._task_id} approved by {identity}", actor=identity
        )
        if parked.get("budget_exhausted"):
            self._outcome = "budget_exhausted"
        else:
            # The SHARED rule, so an unpublished task is `approved_unpublished` on both engines.
            self._outcome = approved_outcome(TaskState.from_payload(self._state))
        return {"identity": identity, "approved": True}

    def _compile(self) -> Any:
        """Compile a FRESH graph for this run.

        Fresh per run, not per class: constructing the plugin mutates the graph (spike finding 2), so
        a shared graph object works for the first plugin and raises for the second. Inside a workflow
        the graph is rebuilt on every replay, which is why it must derive from code and constants
        only — never from anything it observed.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        from temporalio.contrib.langgraph import graph as langgraph_graph

        from .graph import GRAPH_NAME

        return langgraph_graph(GRAPH_NAME).compile(checkpointer=InMemorySaver())

    def _graph_config(self) -> dict[str, Any]:
        """The LangGraph config: a deterministic thread id, the spec, and the current steering.

        The thread id is the Temporal RUN id, not the workflow id: a continued-as-new execution must
        not resume the previous one's checkpoint. The spec and steering ride in ``metadata``, which
        is the channel the plugin round-trips to activity-bound nodes.
        """
        spec = self._spec
        assert spec is not None
        return {
            "configurable": {"thread_id": workflow.info().run_id},
            "metadata": {
                "codna_task": spec.to_payload(),
                "codna_steer": list(self._steer),
            },
        }

    async def _respect_pause(self) -> None:
        if not self._paused:
            return
        await self._notify("note", f"task {self._task_id} paused")
        await workflow.wait_condition(lambda: not self._paused or self._stopped)

    async def _remember(self, state: TaskState, outcome: str) -> Any:
        """Record this task's episode and hand back its reference (path + digest). See the plain engine."""
        from .contracts import EpisodeOutcome

        spec = self._spec
        if spec is None:  # pragma: no cover
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
            "engine": "langgraph",
            "run_id": self._run_id,
            "task_id": self._task_id,
            "turn_seq": state.turn_seq,
            "paused": self._paused,
            "stopped": self._stopped,
            "awaiting_approval": bool(self._interrupts) and self._decision is None,
            "interrupts": list(self._interrupts),
            "pending_approval": state.pending_approval.to_payload() if state.pending_approval else None,
            # D4: WHAT the agent is waiting to be allowed to do. Present exactly when the turn
            # parked on an ask, so a UI can show the question instead of a generic park.
            "ask": state.ask.to_payload() if state.ask else None,
            "budget": state.budget.to_payload(),
            "budget_exhausted": state.budget.exhausted,
            "review_verdict": state.review_verdict,
            "verify_passed": state.verify_ref.passed if state.verify_ref else None,
            "pr": state.pr.to_payload() if state.pr else None,
            # WHY there is no pull request, when there is none ON PURPOSE. See the plain engine's
            # `status` for the reason the two must be reported together.
            "publish_skip": state.publish_skip,
            "approvals": list(self._approvals),
            "nodes_run": list(self._nodes_run),
            "outcome": self._outcome,
            "state": state.to_payload() if self._state else None,
        }

    def _safe_state(self) -> TaskState:
        if self._state:
            return TaskState.from_payload(self._state)
        info = workflow.info()
        return TaskState(run_id=run_id_of(info.workflow_id), task_id=info.workflow_id)

    @workflow.update
    def approve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        identity = approver_identity(payload, handler="approve")
        self._decision = {"identity": identity, "approved": True, "kind": "approve"}
        return {"applied": True, "identity": identity, "turn_seq": self._safe_state().turn_seq}

    @approve.validator
    def _validate_approve(self, payload: Mapping[str, Any]) -> None:
        """Refuse an anonymous or ill-timed approval BEFORE it is applied.

        A validator, not a check in the handler: an Update handler that raises FAILS THE WORKFLOW TASK
        and Temporal retries it forever — the run wedges and every query times out (measured). A
        validator's exception goes back to the caller as a rejected update, which is what "refused"
        means.
        """
        approver_identity(payload, handler="approve")
        if not self._interrupts or self._decision is not None:
            raise ValueError("approve: this task is not waiting on a decision")

    @workflow.update
    def reject(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        identity = approver_identity(payload, handler="reject")
        self._decision = {"identity": identity, "approved": False, "kind": "approve"}
        return {"applied": True, "identity": identity}

    @reject.validator
    def _validate_reject(self, payload: Mapping[str, Any]) -> None:
        approver_identity(payload, handler="reject")
        if not self._interrupts or self._decision is not None:
            raise ValueError("reject: this task is not waiting on a decision")

    @workflow.update
    def steer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Add a steering note. The next superstep's config carries it, so the graph sees it between
        nodes — the same boundary the plain engine honours."""
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
        self._paused = True

    @workflow.signal
    def resume(self) -> None:
        self._paused = False

    @workflow.signal
    def stop(self) -> None:
        self._stopped = True
        self._paused = False

    @workflow.query
    def approvals(self) -> list[dict[str, Any]]:
        return list(self._approvals)


