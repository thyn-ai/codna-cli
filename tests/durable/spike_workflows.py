"""The spike's workflow and stub nodes, in their own module ON PURPOSE.

The D2.0 spike found a third sharp edge that the plan did not anticipate: Temporal re-imports a
workflow's defining module inside the workflow sandbox, so anything that module imports must be
importable there. A workflow class defined in a `pytest` test module therefore fails at worker
construction with

    RestrictedWorkflowAccessError: Cannot access urllib.request.Request.__mro_entries__
    from inside a workflow
    → RuntimeError: Failed validating workflow <T>

because importing `pytest` transitively touches `urllib.request`. That is not a test-only
curiosity: it is the rule `codna.durable.workflows` (D2.3) must follow — no `httpx`, no
`sidecar_client`, no `subprocess`, nothing that resolves a URL, at module scope. Activities may
import all of it; the workflow module may not.

So this file imports only what the sandbox allows: stdlib (`json`, `dataclasses`, `datetime`,
`typing`), `langgraph`, `temporalio`, and `codna.durable.{graph,routing,state}`.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from temporalio import workflow as temporal_workflow
from temporalio.contrib.langgraph import graph as langgraph_graph

from codna.durable.graph import GRAPH_NAME, task_state_from_graph_state

TASK_QUEUE = "codna-spike"
RUN_TIMEOUT = timedelta(minutes=5)


# -- stub nodes -------------------------------------------------------------------------
#
# Module level, not closures: the plugin derives its task id from `__module__.__qualname__` and
# raises "Tasks must be importable from a named module" for anything defined in `__main__` or
# locally. These return ONLY `TaskState` fields — the graph state is a TaskState payload, and a
# node that invented its own key would be carrying unvalidated state nothing ever cleans up.


def initial_state(**overrides: Any) -> dict[str, Any]:
    """A FRESH state payload per call.

    Fresh because LangGraph writes its own keys into the dict it is handed (`__interrupt__`), so a
    graph state dict is mutated in place — and the `Interrupt` object inside is not JSON. Reusing
    one dict leaks it into the next call, which then fails the state decoder. D2.3's workflow must
    pass its own copy for the same reason.
    """
    import json

    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "run-1",
        "task_id": "task-1",
        "turn_seq": 0,
        "session_id": None,
        "last_turn_key": None,
        "last_event_index": 0,
        "plan_ref": None,
        "checkpoint_ref": None,
        "verify_ref": None,
        "review_verdict": None,
        "pr": None,
        "budget": {
            "max_cost_usd": None,
            "spent_usd": 0.0,
            "max_tokens": None,
            "spent_tokens": 0,
            "unpriced_tokens": 0,
        },
        "pending_approval": None,
        "attempts": {"plan": 0, "implement": 0, "verify": 0, "review": 0, "open_pr": 0},
    }
    state.update(overrides)
    return json.loads(json.dumps(state))  # deep copy: no shared nested dicts across tests


async def stub_plan(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "plan_ref": f"refs/plans/{state['task_id']}"}


async def stub_implement(state: dict[str, Any]) -> dict[str, Any]:
    attempts = dict(state["attempts"])
    attempts["implement"] = attempts["implement"] + 1
    return {**state, "attempts": attempts}


async def stub_verify(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "verify_ref": {"ref": "verify/1", "passed": True, "command_digest": "a" * 64}}


async def stub_review(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "review_verdict": "approved"}


async def stub_open_pr(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "pr": {"number": 1, "url": "https://github.com/o/r/pull/1", "branch": "codna/run-1"}}


async def approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """The approval node: `interrupt()`, refusing a resume value without an identity."""
    decision = interrupt({"kind": "approve", "task_id": state["task_id"]})
    if not isinstance(decision, dict):
        raise ValueError("approval resume value must be an object")
    identity = decision.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("approval resume value must carry a non-empty 'identity'")
    if decision.get("approved") is not True:
        raise ValueError(f"task rejected by {identity}")
    return {**state, "pending_approval": None}


STUB_NODES = {
    "plan": stub_plan,
    "implement": stub_implement,
    "verify": stub_verify,
    "review": stub_review,
    "open_pr": stub_open_pr,
    "wait_for_human": approval_node,
}


def interrupt_payloads(result: Any) -> list[dict[str, Any]]:
    """Extract interrupt payloads from the shape LangGraph 1.x returns.

    LangGraph 1.x RETURNS ``{"__interrupt__": (Interrupt(...),)}`` rather than raising
    `GraphInterrupt` (spike finding 4). Both shapes are read, because a middle-of-node interrupt in
    another minor can surface either way and the difference is invisible until a run parks.
    """
    if isinstance(result, dict) and result.get("__interrupt__"):
        return [dict(item.value) for item in result["__interrupt__"]]
    return []


@temporal_workflow.defn(name="SpikeTaskWorkflow")
class SpikeTaskWorkflow:
    """The smallest workflow that exercises all four spike facts.

    Mirrors what `TaskWorkflow` (D2.3) will do: compile the plugin-provided graph with
    `InMemorySaver`, run it, park on the returned `__interrupt__`, wait for an Update, resume with
    `Command(resume=...)`, and record the approver's identity in workflow state (not in the graph
    state, which stays a clean `TaskState` payload).
    """

    def __init__(self) -> None:
        self._thread_id = temporal_workflow.info().workflow_id
        self._app = langgraph_graph(GRAPH_NAME).compile(checkpointer=InMemorySaver())
        self.interrupts: list[dict[str, Any]] = []
        self.approval: dict[str, Any] | None = None
        self.approvals_applied: list[str] = []

    @temporal_workflow.run
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        config = {"configurable": {"thread_id": self._thread_id}}
        result = await self._app.ainvoke(state, config)
        self.interrupts = interrupt_payloads(result)
        await temporal_workflow.wait_condition(lambda: self.approval is not None)
        resumed = await self._app.ainvoke(Command(resume=self.approval), config)
        return task_state_from_graph_state(resumed).to_payload()

    @temporal_workflow.query
    def status(self) -> dict[str, Any]:
        return {
            "interrupts": self.interrupts,
            "awaiting_approval": self.approval is None and bool(self.interrupts),
        }

    @temporal_workflow.query
    def approvals(self) -> list[str]:
        return list(self.approvals_applied)

    @temporal_workflow.update
    def approve(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.approval is not None:
            raise ValueError("this run has already been approved")
        self.approval = decision
        self.approvals_applied.append(str(decision.get("identity")))
        return decision
