"""The durable workflows: both task engines, the run, and the one place they are registered.

Three workflow classes, registered together, selected by ``graph_engine``:

* `TaskWorkflowPlain` (`task_workflow_plain.py`) — the same six steps as plain workflow code, with
  `workflow.wait_condition` approvals. The **reference semantics**: readable as a sequence, no plugin,
  no checkpointer, and what the LangGraph engine is checked against. Runs when ``graph_engine=plain``.
* `TaskWorkflow` (`task_workflow_graph.py`) — the same task driven by the graph from
  `graph.build_task_graph`, where five nodes are activities and the approval is a LangGraph
  `interrupt()` resumed by a Temporal Update.
* `RunWorkflow` (`run_workflow.py`) — one child task per task, a run-level budget, and continue-as-new.

This module is the REGISTRATION and re-export surface. It used to hold all three classes and crossed
the repository's 1000-line ceiling (`test_modularity.py`); the split is by concern — one engine per
file — so the header can stay the place a reader learns how the pieces fit together. Everything is
re-exported here, so a caller (the worker in D2.4, the tests) imports one module.

**Parity is the point, not a nicety.** Both engines schedule the same activities, in the same order,
from the same `graph.STEPS` table, and both route with the same pure functions from `routing.py`. The
task suite asserts the same brief completes under both with the same observable route and outcome. The
only things that differ are *how* an approval suspends (an Update-resumed interrupt vs. a
`wait_condition`) and *what* the two engines put in history.

**No wall clock, no randomness, no I/O in any of these modules.** Temporal replays a workflow by
re-running it, so anything impure becomes a non-determinism error on replay — a stuck run in
production, not a red test. Timestamps come from `workflow.now()`, ids from state, every timeout is a
constant. The imports are `temporalio`, sandbox-clean `codna.durable` leaves, and `langgraph` only
LAZILY (inside a function) — `tests/durable/test_import_hygiene.py` guards that, because the sandbox
refuses `urllib.request` and a workflow that reaches it through `config.py` fails at WORKER
CONSTRUCTION.

**Approvals are Updates, and they record identity.** ``approve``/``reject``/``steer``/``set_budget``
are Updates whose REJECTIONS live in validators, not in the handlers: an Update handler that raises
fails the workflow task and Temporal retries it forever (measured while building D2.3 — the run wedges
and every query times out). A validator's exception goes back to the caller as a rejected update, which
is what "refused" means.
"""
from __future__ import annotations

from typing import Any, Final

from .contracts import GRAPH_ENGINES
from .run_workflow import RunWorkflow
from .task_workflow_graph import TaskWorkflow
from .task_workflow_plain import TaskWorkflowPlain

__all__ = [
    "APPROVAL_UPDATE_NAMES",
    "CONTINUE_AS_NEW_TASKS",
    "HISTORY_EVENTS_LIMIT",
    "RunWorkflow",
    "TaskWorkflow",
    "TaskWorkflowPlain",
    "registered_workflows",
    "workflow_class_for",
]

# Continue-as-new thresholds — the plan's numbers, and its exit criterion (history < 10,240 events).
CONTINUE_AS_NEW_TASKS: Final = 10
HISTORY_EVENTS_LIMIT: Final = 5000

APPROVAL_UPDATE_NAMES: Final[tuple[str, ...]] = ("approve", "reject", "steer", "set_budget")


def workflow_class_for(engine: str) -> Any:
    """The workflow class for an engine name, refusing an unrecognized one.

    No default on a typo: the two engines produce different history, and a run started under the wrong
    one cannot be replayed by the worker that picks it up.
    """
    if engine == "langgraph":
        return TaskWorkflow
    if engine == "plain":
        return TaskWorkflowPlain
    raise ValueError(f"unknown graph engine {engine!r}; expected one of {', '.join(GRAPH_ENGINES)}")


def registered_workflows() -> tuple[Any, ...]:
    """Both task engines plus the run, for ``Worker(workflows=...)`` (D2.4).

    One tuple defined here, so the worker cannot register one engine and forget the other — which
    would leave a run started under the missing engine waiting until its task-queue timeout.
    """
    return (RunWorkflow, TaskWorkflow, TaskWorkflowPlain)
