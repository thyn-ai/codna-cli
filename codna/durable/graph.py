"""The task graph, its node→activity mapping, and the spike findings that make it safe to ship.

D2.0 spike — GO. Executed 2026-10-26 against the real installed libraries
(``temporalio`` 1.34.0, ``langgraph`` 1.2.17, ``langchain-core`` 1.7.0) on a real Temporal test
server (``WorkflowEnvironment.start_time_skipping()``), not reasoned about. The four facts the plan
required:

1. **A `StateGraph` whose nodes are Temporal activities runs inside a workflow via
   `LangGraphPlugin`.** GO. A node registered with ``metadata={"execute_in": "activity"}`` became a
   real ``ActivityTaskScheduled`` in history (activity type ``<graph>.<node>``, executed by the
   worker's activity pool) while a node registered ``execute_in="workflow"`` ran inline in the
   workflow. The workflow completed and returned the graph's final state.
   *Two hard requirements surfaced that the plugin's docs do not state:*
   (a) every activity node needs ``start_to_close_timeout`` — without it the workflow task fails
   with ``ValueError: Activity must have start_to_close_timeout or schedule_to_close_timeout``, so
   `build_task_graph` sets it via ``default_activity_options``;
   (b) node functions must be importable from a **named module** — the plugin derives a task id
   from ``__module__.__qualname__`` and refuses ``__main__`` or a closure, so every node here is a
   module-level function in this file.

2. **The plugin must be on the Worker AND is required for the Replayer.** GO. ``Replayer`` accepts
   ``plugins=`` and replays the recorded history with ``replay_failure is None``; the *same*
   history with no plugin fails at construction with
   ``RuntimeError: Failed validating workflow <T>``, because the plugin is what rewrites the nodes'
   ``afunc`` into activity calls. This is the single most dangerous fact in the spike: a replay
   suite that forgets the plugin appears to pass (it fails loudly here, but a hosted replay service
   would simply report nothing), so `replay_history` below takes the plugin as a required argument.
   *Second sharp edge:* ``LangGraphPlugin.__init__`` **mutates the graph in place** — it strips
   ``execute_in`` out of each node's metadata and replaces ``afunc``. Constructing a second plugin
   over the same graph object therefore raises
   ``ValueError: Node <graph>.<node> is missing required 'execute_in'``. Worker and Replayer must
   each get a **fresh** graph from `build_task_graph()`; a module-level singleton graph is a bug
   that only appears when the second component starts.

3. **`InMemorySaver` is mandatory.** GO, and narrower than "mandatory": a graph with no interrupt
   runs fine without a checkpointer, but `Command(resume=...)` raises
   ``RuntimeError: Cannot use Command(resume=...) without checkpointer``. Since the human-approval
   node is an `interrupt()`, the graph MUST be compiled with a checkpointer, and it must be
   `InMemorySaver`: Temporal owns durability, the checkpointer only has to survive within the
   replay of one workflow, and a persistent checkpointer would put workflow state outside Temporal
   (unreplayable, and a second source of truth for "where was I").

4. **In-workflow `interrupt()` + Update-resume works for the human-approval node.** GO, with a
   correction to the plan's wording. `interrupt()` must run in a node marked
   ``execute_in="workflow"`` — in an activity the interrupt would be raised on the activity side,
   where there is no graph execution to suspend. And in LangGraph 1.x the interrupt does **not**
   raise `GraphInterrupt` out of ``ainvoke``; it *returns* the state with an ``__interrupt__`` key
   holding `Interrupt` objects. `wait_for_human` handles both shapes (returned and raised) because
   a middle-of-node interrupt in an older/newer minor can surface either way. Resuming with
   ``Command(resume=<value>)`` on the same ``thread_id`` completes the node, and the value reaches
   the node as `interrupt()`'s return value — which is how the approver's identity is proven.

First replay evidence (recorded, and re-recordable by the spike test): a 19-event history
(``WorkflowExecutionStarted`` → ``ActivityTaskScheduled/Started/Completed`` for ``codna.task.plan``
→ ``WorkflowExecutionSignaled`` for the approval Update → ``WorkflowExecutionCompleted``) replays to
``replay_failure=None`` with the plugin. Without the plugin the Replayer cannot even validate the
workflow: the plugin also patches the sandbox's passthrough modules.

Two further requirements the spike found, both of which would otherwise surface only in D2.3:

5. **A conditional edge must be `async`.** LangGraph runs a *synchronous* conditional edge through
   ``loop.run_in_executor``, and Temporal's deterministic event loop does not implement it — the
   workflow task dies with a bare ``NotImplementedError`` inside the branch, on the first route
   decision, with nothing pointing at the edge. The routers in `routing.py` stay pure and sync (the
   plain engine calls them directly); the `async def decide` adapter in `build_task_graph` is what
   satisfies the plugin.
6. **A workflow's module may not import anything that reaches `urllib.request`.** The sandbox
   re-imports the workflow's module and refuses ``urllib.request``, so a workflow module that
   imports `config.py` (→ `codna.runtime.config` → `httpx` → `urllib.request`) fails at worker
   construction with ``RestrictedWorkflowAccessError``. `GRAPH_VERSION` therefore lives in
   `version.py`, a leaf module. `tests/durable/test_import_hygiene.py` is the guard.

Write rule inherited from the spike: **`GRAPH_VERSION` changes ⇒ new `build_id` ⇒ new pinned worker
version ⇒ re-record the replay histories.** That is why the version lives beside the graph and why
the replay test asserts the histories were recorded for the version it is running.

## D2.2: the activity bindings, and the correction the plan needed

The plan says each node should "call `workflow.execute_activity(..., start_to_close_timeout=...)`".
That is what the **plain** engine does. It is not something the graph can do, and that is measured,
not assumed:

    temporalio.workflow._exceptions._NotInWorkflowEventLoopError: Not in workflow event loop

An ``execute_in="activity"`` node does not run *beside* an activity, it runs **inside one**
(`_plugin.wrap_activity` registers the node function as the activity's body). There is no workflow
runtime on that side, so `workflow.execute_activity` raises before it can schedule anything.

So the five doing-nodes are **activities whose bodies are the real work**: each builds its request
from the graph state plus the config metadata, and hands it to `activities.py`. The
`start_to_close_timeout` the plan asks for is still enforced — one level up, where the plugin
schedules that node through `workflow.execute_activity(node_body, start_to_close_timeout=…)` from
`_plugin_activity_options()`, which is exactly where a missing timeout fails a workflow task.

`_run_step` therefore does ONE thing and has no branch: the node IS an activity, so it runs the step
bodies itself, off the event loop. There is no "schedule it instead" path to choose between — an
`execute_in="activity"` node has no workflow runtime to schedule from. The other way an activity is
started is `TaskWorkflowPlain`, which schedules the same registered functions from real workflow code;
both paths call the SAME implementations from the same `STEPS` table, which is what makes "the same
brief completes under both engines" a tested fact rather than an aspiration.

`STEPS` is the shared truth between the two engines: the same activity names, the same request
builders and the same timeouts, so "the same brief completes under both engines" is a tested fact
rather than an aspiration.
"""
from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable, Final, Literal, Mapping

from .contracts import (
    ACTIVITY_HEARTBEAT_TIMEOUT_S,
    ACTIVITY_TIMEOUT_S,
    ACTIVITY_TURN_TIMEOUT_S,
    ACTIVITY_VERIFY_TIMEOUT_S,
    HEARTBEAT_INTERVAL_S,
    NAME_CHECKPOINT,
    NAME_DELEGATION_POLICY,
    NAME_EPISODE,
    NAME_NOTIFY,
    NAME_OPEN_PR,
    NAME_PLAN,
    NAME_RECALL,
    NAME_REVIEW,
    NAME_TURN,
    NAME_VERIFY,
    NAME_WATCH_CI,
    NAME_WORKTREE_CLEANUP,
    CheckpointOutcome,
    CheckpointRequest,
    NodeContext,
    PlanRequest,
    PullRequestOutcome,
    PullRequestRequest,
    ReviewRequest,
    TaskSpec,
    TurnRequest,
    VerifyRequest,
    implement_prompt,
    task_spec_from_config,
)
from .parking import park_reason
from .routing import DEFAULT_MAX_ATTEMPTS, Route, after_plan, after_review, after_verify
from .state import TaskState
from .version import GRAPH_VERSION

__all__ = [
    "ACTIVITY_TIMEOUT_S",
    "CONFIG_STEER",
    "CONFIG_TASK_SPEC",
    "GRAPH_INTERNAL_KEYS",
    "GRAPH_NAME",
    "GRAPH_VERSION",
    "NODE_EXECUTION",
    "STEP_TIMEOUT_S",
    "STEPS",
    "TASK_NODES",
    "TaskGraph",
    "build_plugin",
    "build_task_graph",
    "replay_history",
    "task_spec_from_config",
    "task_state_from_graph_state",
]

GRAPH_NAME: Final = "codna.task"

NodeName = Literal["plan", "implement", "verify", "review", "open_pr", "wait_for_human"]

TASK_NODES: Final[tuple[NodeName, ...]] = (
    "plan",
    "implement",
    "verify",
    "review",
    "open_pr",
    "wait_for_human",
)

# The mapping the spike proved: the five doing-steps cross the activity boundary (so they are
# retried, heartbeated and durable), and only the approval suspension runs in-workflow (it must:
# `interrupt()` needs an executing graph, and a suspension is not work to retry).
NODE_EXECUTION: Final[dict[NodeName, Literal["activity", "workflow"]]] = {
    "plan": "activity",
    "implement": "activity",
    "verify": "activity",
    "review": "activity",
    "open_pr": "activity",
    "wait_for_human": "workflow",
}

# The metadata keys the workflow puts in the graph's config. They ride in the config rather than the
# state because the state is a validated `TaskState` payload with a closed field set: a `brief` in
# it would be a field nothing validates, growing with the request, in the one place that must not.
CONFIG_TASK_SPEC: Final = "codna_task"
CONFIG_STEER: Final = "codna_steer"

# A transport-level retry bound, distinct from the workflow's SEMANTIC bound (verify-fail →
# implement, review-reject → implement, both capped by `max_attempts`). An activity failing the same
# way four times will not succeed on the fifth, and an unbounded server-side retry means the workflow
# never learns the turn is dead — the run would look alive forever.
#
# IT LIVES HERE, BESIDE `STEP_TIMEOUT_S`, because BOTH engines must read it: `workflow_activity` in
# `workflow_support.py` (plain) and `_plugin_activity_options` below (graph). It used to live only in
# `workflow_support.py`, which the plugin path could not read, so the graph engine silently ran with
# Temporal's default policy — no attempt cap at all.
ACTIVITY_RETRIES: Final = 3

# Per-activity `start_to_close_timeout`, seconds. The one table both engines read — including the two
# bookkeeping activities, which are scheduled from workflow code (habitually the same way) and would
# otherwise be scheduled with no timeout, the exact failure the spike proved.
STEP_TIMEOUT_S: Final[dict[str, int]] = {
    NAME_PLAN: ACTIVITY_TURN_TIMEOUT_S,
    NAME_TURN: ACTIVITY_TURN_TIMEOUT_S,
    NAME_CHECKPOINT: ACTIVITY_TIMEOUT_S,
    NAME_VERIFY: ACTIVITY_VERIFY_TIMEOUT_S,
    NAME_REVIEW: ACTIVITY_TURN_TIMEOUT_S,
    NAME_OPEN_PR: ACTIVITY_TIMEOUT_S,
    NAME_EPISODE: ACTIVITY_TIMEOUT_S,
    NAME_NOTIFY: ACTIVITY_TIMEOUT_S,
    # Run-level bookkeeping, scheduled by `RunWorkflow` rather than by a graph node — here for the same
    # reason `episode`/`notify` are: one timeout table, so a newly scheduled activity cannot be
    # scheduled without one (the D2.0 spike's failure mode). It does NOT become a graph step; the task
    # graph's node sequence is unchanged, which is what keeps the replay cassette valid.
    NAME_WORKTREE_CLEANUP: ACTIVITY_TIMEOUT_S,
    # D5 goal mode: both are run-level, scheduled by `RunWorkflow` at a task boundary. Neither is a
    # graph node — the goal loop lives above the task, so the graph's node sequence (and therefore
    # every recorded cassette) is unchanged.
    NAME_RECALL: ACTIVITY_TIMEOUT_S,
    NAME_WATCH_CI: ACTIVITY_TIMEOUT_S,
    # One policy read: `policy_verdict` parses a small committed YAML file. The ordinary ceiling is
    # generous; it is NOT a graph node, so the node sequence (and every recorded cassette) is
    # unchanged.
    NAME_DELEGATION_POLICY: ACTIVITY_TIMEOUT_S,
}

def _plugin_activity_options() -> dict[str, Any]:
    """The `default_activity_options` the graph engine hands to `LangGraphPlugin`.

    WHY A FUNCTION AND NOT A MODULE CONSTANT. `graph.py` is a workflow-facing module and must
    import cleanly inside Temporal's sandbox, where `temporalio` is not importable at module scope
    (`tests/durable/test_import_hygiene.py` proved it: a module-level RetryPolicy made
    `import codna.durable.graph` fail with `ModuleNotFoundError: No module named 'temporalio'`).
    Building the policy here keeps the import lazy, at the one point a worker actually constructs
    the plugin.

    Both timeouts are mandatory: without `start_to_close_timeout` the plugin schedules an activity
    the SDK refuses and the workflow task fails with a bare ValueError (spike finding 1a). The
    heartbeat is generous relative to the turn's 10s beat — one missed beat must not retry a paid
    turn.

    The RETRY BOUND is the part that used to be missing. `workflow_activity` (the plain engine)
    passes `retry_policy=_retry_policy()`, but this path passed only the two timeouts, so Temporal
    applied its DEFAULT policy — no attempt cap. MEASURED in the installed app: a
    `codna.task.plan` activity that could never succeed (403 from the policy reader) retried 27
    times over 30 minutes while the run's ledger stayed at 4 records, and `codna durable control
    --run ... stop` did not end it. Both engines now read `ACTIVITY_RETRIES`.
    """
    from temporalio.common import RetryPolicy

    return {
        "start_to_close_timeout": timedelta(seconds=ACTIVITY_TURN_TIMEOUT_S),
        "heartbeat_timeout": timedelta(seconds=ACTIVITY_HEARTBEAT_TIMEOUT_S),
        "retry_policy": RetryPolicy(maximum_attempts=ACTIVITY_RETRIES),
    }

_GRAPH_KEY: Final = GRAPH_NAME

# LangGraph writes its own bookkeeping into the state dict. `__interrupt__` appears once a run has
# parked, and — this bit is easy to miss — the object in it is an `Interrupt`, which is not JSON. So
# a graph state is NOT a `TaskState` payload: it is a payload plus LangGraph's keys. Routing and
# every node therefore go through `task_state_from_graph_state`, which drops exactly these keys and
# then validates, rather than relaxing the decoder (a decoder that ignores unknown keys would also
# ignore a field a future change forgot to validate).
GRAPH_INTERNAL_KEYS: Final[frozenset[str]] = frozenset({"__interrupt__"})


def task_state_from_graph_state(state: Mapping[str, Any]) -> TaskState:
    """Decode a LangGraph state dict into a `TaskState`, dropping LangGraph's own keys."""
    return TaskState.from_payload(
        {key: value for key, value in state.items() if key not in GRAPH_INTERNAL_KEYS}
    )


# -- the step table both engines share ---------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One activity inside a doing-step: the registered name, how to build its request, its result.

    ``build`` takes the node context (task spec + steering) and the state accumulated so far *within
    this step*, so the checkpoint after a turn sees the turn's new ``turn_seq`` — which is what makes
    "one checkpoint per turn_seq" true rather than incidental.
    """

    activity: str
    build: Callable[[NodeContext, Mapping[str, Any]], Any]
    result_type: type
    heartbeat: bool = False


def _plan_request(context: NodeContext, state: Mapping[str, Any]) -> PlanRequest:
    return context.plan_request(state)


def _turn_request(context: NodeContext, state: Mapping[str, Any]) -> TurnRequest:
    return context.turn_request(state, implement_prompt(context.spec.brief, context.steer))


def _checkpoint_request(context: NodeContext, state: Mapping[str, Any]) -> CheckpointRequest:
    return context.checkpoint_request(state)


def _verify_request(context: NodeContext, state: Mapping[str, Any]) -> VerifyRequest:
    return context.verify_request(state)


def _review_request(context: NodeContext, state: Mapping[str, Any]) -> ReviewRequest:
    return context.review_request(state)


def _pull_request_request(context: NodeContext, state: Mapping[str, Any]) -> PullRequestRequest:
    return context.pull_request_request(state)


STEPS: Final[dict[str, tuple[Step, ...]]] = {
    "plan": (Step(NAME_PLAN, _plan_request, PlanRequest),),
    # A turn, then its checkpoint: one commit per `turn_seq`, taken right after the edits exist. Two
    # activities because the graph cannot express "and then commit" without a new node, and a new
    # node is a history change (⇒ a `GRAPH_VERSION` bump).
    "implement": (
        Step(NAME_TURN, _turn_request, TurnRequest, heartbeat=True),
        Step(NAME_CHECKPOINT, _checkpoint_request, CheckpointOutcome),
    ),
    "verify": (Step(NAME_VERIFY, _verify_request, VerifyRequest),),
    "review": (Step(NAME_REVIEW, _review_request, ReviewRequest),),
    "open_pr": (Step(NAME_OPEN_PR, _pull_request_request, PullRequestOutcome),),
}


def step_timeout_s(activity: str) -> int:
    """Seconds of ``start_to_close_timeout`` for an activity name, refusing an unknown one."""
    try:
        return STEP_TIMEOUT_S[activity]
    except KeyError as exc:
        raise ValueError(
            f"no step timeout for activity {activity!r}; known: {', '.join(sorted(STEP_TIMEOUT_S))}"
        ) from exc


@dataclass(frozen=True)
class NodeSpec:
    """A node's identity and routing contract, independent of LangGraph.

    Kept as data so the plain workflow can iterate the same sequence without importing langgraph
    (the plain engine must run when the extra is missing — that is the whole point of the fallback).
    """

    name: NodeName
    route: Callable[[TaskState], Route] | None = None


@dataclass(frozen=True)
class TaskGraph:
    """The compiled-for-Temporal graph plus the plugin that wraps it.

    A pair, not a graph: constructing the plugin mutates the graph (spike finding 2), so the two
    must be built together and neither may be reused for a second component.
    """

    graph: Any
    plugin: Any
    graph_version: int = GRAPH_VERSION

    @property
    def activity_names(self) -> tuple[str, ...]:
        """The activity type names this plugin registers, as they appear in history."""
        return tuple(
            f"{GRAPH_NAME}.{name}"
            for name, execution in NODE_EXECUTION.items()
            if execution == "activity"
        )


# -- node functions ---------------------------------------------------------------------
#
# Module level, not closures: the plugin derives its task id from `__module__.__qualname__` and
# refuses a function it cannot name (spike finding 1b). Each takes and returns the graph's state,
# which is the serializable `TaskState` payload — never a live object.
#
# The five doing-nodes are ACTIVITY BODIES (see the module docstring): they run inside the activity
# the plugin registered, so `workflow.execute_activity` is unavailable to them and `_run_step` calls
# the implementations directly. `wait_for_human` stays in-workflow because `interrupt()` needs an
# executing graph.


def _graph_config() -> Mapping[str, Any]:
    """The LangGraph config this node is running under.

    `get_config()` rather than a `config=` parameter: the plugin round-trips the config across the
    activity boundary as a contextvar (verified), and a node that declares no config parameter keeps
    LangGraph's signature inspection quiet. Imported here, not at module scope, so importing `graph`
    stays sandbox-clean (`langgraph` is passed through by the plugin, but a bare import of it before
    the plugin exists is needless risk).
    """
    from langgraph.config import get_config

    config = get_config()
    if not isinstance(config, Mapping):
        raise ValueError(
            "a task-graph node must run under a LangGraph config carrying the task spec "
            f"(config['metadata']['{CONFIG_TASK_SPEC}'])."
        )
    return config


def _node_context(config: Mapping[str, Any] | None = None) -> NodeContext:
    """The task spec and steering for this node, from the config's metadata."""
    active = _graph_config() if config is None else config
    spec = task_spec_from_config(active)
    steer: tuple[str, ...] = ()
    metadata = active.get("metadata")
    if isinstance(metadata, Mapping):
        raw = metadata.get(CONFIG_STEER)
        if isinstance(raw, (list, tuple)):
            steer = tuple(str(note) for note in raw if str(note).strip())
    return NodeContext(spec=spec, steer=steer)


async def plan(state: dict[str, Any]) -> dict[str, Any]:
    """Plan the task: a read-only turn (`codna.step.plan`)."""
    return await _run_step("plan", state)


async def implement(state: dict[str, Any]) -> dict[str, Any]:
    """Implement: the heartbeating paid turn, then its checkpoint (`run_cline_turn`, `checkpoint_turn`)."""
    return await _run_step("implement", state)


async def verify(state: dict[str, Any]) -> dict[str, Any]:
    """Verify under the sandbox (`codna.step.verify`)."""
    return await _run_step("verify", state)


async def review(state: dict[str, Any]) -> dict[str, Any]:
    """Review the checkpointed diff (`codna.step.review`)."""
    return await _run_step("review", state)


async def open_pr(state: dict[str, Any]) -> dict[str, Any]:
    """Open (or find) the pull request (`codna.step.open_pr`)."""
    return await _run_step("open_pr", state)


async def _run_step(name: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Run a doing-step's activities in order, folding each result into the graph state.

    The bodies are called DIRECTLY, off the event loop, because a graph node runs inside an activity
    — there is no workflow runtime to schedule from (see the module docstring). The imported
    implementation is the same function the plugin's activity wraps, so what the graph runs and what
    `TaskWorkflowPlain` schedules are literally the same code.

    Heartbeating is promoted for the paid turn: while the body runs on an executor thread, a task on
    THIS loop beats every `HEARTBEAT_INTERVAL_S`, which is what lets the workflow's
    `heartbeat_timeout` notice a dead worker mid-turn rather than at the 60-minute ceiling.

    The state is copied, never mutated: LangGraph keeps the dict it was handed (and writes
    ``__interrupt__`` into it), so mutating in place would leak an `Interrupt` object — which is not
    JSON — into the next node's payload.
    """
    context = _node_context()
    merged = dict(state)
    for step in STEPS[name]:
        request = step.build(context, merged)
        outcome = await _run_step_with_heartbeat(step.activity, request)
        merged.update(outcome.state)
    return merged


async def _run_step_with_heartbeat(activity_name: str, request: Any) -> Any:
    """Await one step body on an executor thread while beating from THIS loop.

    **The beat must be an asyncio task, not a thread.** `codna.task.<step>` is an ASYNC activity
    (temporalio's langgraph contrib wraps the node in `async def`), and `activity.heartbeat()` in an
    async activity resolves the running event loop and schedules `_heartbeat_async` on it. Every
    THREAD-based beat therefore failed, in two different ways, and `_beat_once` swallowed both:

      * a bare `threading.Thread`          -> RuntimeError('Not in activity context')
      * `threading.Thread(ctx.run(...))`   -> RuntimeError('no running event loop')
      * a task created on the activity's   -> works
        own loop

    A thread can therefore never beat for an async activity, whatever context it carries. MEASURED
    against the real dev server: `loop -> ok`, `executor_thread_copied -> 'no running event loop'`,
    `nested_thread_bare -> 'Not in activity context'`. The consequence was a wrapper that never beat:
    a turn that streamed longer than `ACTIVITY_HEARTBEAT_TIMEOUT_S` (120s) failed the workflow with
    "activity Heartbeat timeout", discarding work the user had already paid for.

    So this awaits the body and the beat loop CONCURRENTLY, cancelling the beat as soon as the body
    returns or fails. `asyncio` is imported HERE rather than at module scope on purpose: importing it
    at module scope pulls `socket` into `sys.modules`, which the workflow sandbox refuses
    (`tests/durable/test_import_hygiene.py`).
    """
    import asyncio

    from . import activities as activity_module

    body = activity_module.impl_for(activity_name)
    beat = asyncio.ensure_future(_beat_loop(activity_name))
    try:
        return await asyncio.to_thread(body, request)
    finally:
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass


async def _beat_loop(activity_name: str) -> None:
    """Beat every `HEARTBEAT_INTERVAL_S` until cancelled. Runs on the activity's own loop."""
    import asyncio

    while True:
        _beat_once({"activity": activity_name})
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)


def _beat_around(body: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a SYNCHRONOUS body so it proves liveness while the work runs on another thread.

    Kept for callers that run a body on a thread from a context where `activity.heartbeat()` is
    served synchronously (a sync activity). The graph path no longer uses it: its activities are
    ASYNC, and a beat from any thread fails there — see `_run_step_with_heartbeat`, which is what
    the graph actually calls now.

    The thread must run inside a COPY of the caller's context: a bare `threading.Thread` starts with
    an EMPTY `contextvars` context, and `temporalio.activity.heartbeat` reads the activity from
    exactly that contextvar, so a bare thread's beat raises `RuntimeError('Not in activity
    context')`. MEASURED in a sync activity: `activity_thread -> ok`,
    `bare_thread -> 'Not in activity context'`, `copied_context_thread -> ok`.
    """

    def wrapped(request: Any) -> Any:
        stop = threading.Event()
        context = contextvars.copy_context()

        def beat() -> None:
            while not stop.is_set():
                context.run(_beat_once, {"activity": getattr(body, "__name__", "activity")})
                stop.wait(HEARTBEAT_INTERVAL_S)

        worker = threading.Thread(target=beat, name="codna-heartbeat", daemon=True)
        worker.start()
        try:
            return body(request)
        finally:
            stop.set()
            worker.join(timeout=1.0)

    return wrapped


def _beat_once(details: Mapping[str, Any]) -> None:
    """Heartbeat if there is an activity context; do nothing in a test or a plain call."""
    from temporalio import activity as temporal_activity

    try:
        temporal_activity.heartbeat(dict(details))
    except RuntimeError:
        return


async def wait_for_human(state: dict[str, Any]) -> dict[str, Any]:
    """Suspend until a human approves, then clear the pending approval.

    In-workflow (spike finding 4). The resume value carries the approver's identity — this node
    REFUSES a resume value without one (and a false `approved`), so an approval can never be
    mistaken for "somebody clicked something". The identity itself is not written into the graph
    state: the graph state is exactly a `TaskState` payload, the workflow records the identity from
    the Update it executed, and the ledger records it as `approval.granted`. That keeps state
    validated, bounded, and free of a field nothing would ever clean up.

    The `interrupt()` payload is a plain dict of references so it is serializable and shows up
    readably in the Temporal UI ("what is this run waiting for"), which is the whole reason the
    approval is modelled as an interrupt rather than a bare signal.
    """
    from langgraph.types import interrupt

    current = task_state_from_graph_state(state)
    prompt = {
        "kind": "approve",
        "run_id": current.run_id,
        "task_id": current.task_id,
        "turn_seq": current.turn_seq,
        "plan_ref": current.plan_ref,
        "verify_ref": current.verify_ref.ref if current.verify_ref else None,
        "review_verdict": current.review_verdict,
        "pr": current.pr.to_payload() if current.pr else None,
        # WHY there is no pull request, when there is none on purpose: a human being asked to approve
        # the change must be able to read "the repository has no GitHub remote, so publishing was
        # skipped" instead of "pr: null" and having to guess.
        "publish_skip": current.publish_skip,
        "budget_exhausted": current.budget.exhausted,
        # WHY the run parked, in the shared vocabulary `park_reason` already defines. Without this the
        # graph engine cannot tell "the work is ready for a pull request" apart from "verification
        # failed three times and the attempt budget is spent", and the approval it records says
        # `open_pr_ready` for both. Measured (`run-20261113T225225Z`): three verify records with
        # `passed: false` and `returncode: 127`, then an approval whose recorded reason was
        # `open_pr_ready` — so the human gate reported the work as ready when its own verify had never
        # passed. The prompt is the only channel that survives into `_await_decision`, which is why
        # the reason travels here rather than being re-derived there.
        "park_reason": park_reason(current),
    }
    decision = interrupt(prompt)
    if not isinstance(decision, dict):
        raise ValueError(
            "wait_for_human resume value must be an object carrying the approver's identity "
            f"(got {type(decision).__name__})."
        )
    identity = decision.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("wait_for_human resume value must carry a non-empty 'identity'.")
    approved = decision.get("approved")
    if approved is not True:
        # A refusal ends the task rather than looping: re-planning on "no" would be re-asking the
        # same person the same question, and the plan's approval is a gate, not a retry budget.
        raise ValueError(f"task rejected by {identity}")
    # Only `TaskState` fields are returned. The graph's state IS a `TaskState` payload — a node that
    # invented its own key would be carrying data that never gets validated and never gets cleaned
    # up, so the identity is not stored here: the workflow holds it from the Update it executed, and
    # the ledger records it as `approval.granted`.
    return {**state, "pending_approval": None}


# -- graph construction -----------------------------------------------------------------


# The production node implementations, resolved at call time by `build_task_graph`. Defined here,
# after every function, so the mapping cannot reference a name that does not exist yet.
_DEFAULT_NODES: Final[dict[NodeName, Callable[..., Any]]] = {
    "plan": plan,
    "implement": implement,
    "verify": verify,
    "review": review,
    "open_pr": open_pr,
    "wait_for_human": wait_for_human,
}


def build_task_graph(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    nodes: Mapping[NodeName, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] | None = None,
) -> "TaskGraph":
    """Build a FRESH graph + plugin pair. Call once per Worker and once per Replayer.

    Fresh because `LangGraphPlugin.__init__` consumes the graph's node metadata (spike finding 2):
    a shared graph object works for the first plugin and raises for the second. Returning both from
    one function makes that impossible to get wrong, and `TaskGraph` is intentionally not cached.

    ``nodes`` overrides the node implementations without changing the graph's shape, activity names
    or routing. The activity name is derived from the graph and node names
    (``codna.task.<node>``), never from the function, so an override is wire-compatible with the
    real one — which is what lets the spike prove the mapping with stub activities. An override
    still has to be a module-level function: the plugin derives its task-cache id from
    ``__module__.__qualname__`` and refuses anything it cannot name.

    Shape: ``plan → implement → verify → review → open_pr → wait_for_human → END`` with three
    conditional edges. The conditional edges are the *pure* routing functions, called with the
    decoded `TaskState`, so the graph and the plain workflow cannot diverge in their decisions.
    """
    from langgraph.graph import END as LANGGRAPH_END
    from langgraph.graph import START, StateGraph

    from temporalio.contrib.langgraph import LangGraphPlugin

    implementations = dict(nodes or {})
    for name in implementations:
        if name not in NODE_EXECUTION:
            raise ValueError(f"unknown task node {name!r}; expected one of {', '.join(TASK_NODES)}")

    def _resolve(name: NodeName) -> Callable[..., Any]:
        return implementations.get(name) or _DEFAULT_NODES[name]

    def _route(router: Callable[..., Route]) -> Callable[[dict[str, Any]], Awaitable[str]]:
        """Adapt a pure router to a LangGraph conditional edge.

        **Must be `async`.** A *synchronous* conditional edge is silently run through
        ``loop.run_in_executor`` by LangGraph, and Temporal's deterministic event loop does not
        implement ``run_in_executor`` — the workflow task dies with a bare ``NotImplementedError``
        inside the branch. This is spike finding 5 and it is invisible until a graph actually takes
        a branch, i.e. it would have appeared in D2.3 with no obvious cause. The router itself stays
        pure and sync (`routing.py`); only this adapter is async.

        The state is decoded (validating it: a corrupted state must fail here, loudly, rather than
        route somewhere plausible) and the router's own `max_attempts` is pinned so replay sees the
        same ceiling it originally used.
        """

        async def decide(state: dict[str, Any]) -> str:
            return router(task_state_from_graph_state(state), max_attempts=max_attempts)

        decide.__name__ = f"route_{router.__name__}"
        return decide

    graph = StateGraph(dict)
    for name in TASK_NODES:
        graph.add_node(name, _resolve(name), metadata=_node_metadata(name))

    # `wait_for_human` is the park node: every "needs a human" outcome lands there, which keeps the
    # routing functions total (they never throw) and gives the UI exactly one state to render.
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _route(after_plan),
        {"implement": "implement", "plan": "plan", "wait_for_human": "wait_for_human"},
    )
    graph.add_conditional_edges(
        "verify",
        _route(after_verify),
        {"review": "review", "implement": "implement", "wait_for_human": "wait_for_human"},
    )
    graph.add_conditional_edges(
        "review",
        _route(after_review),
        {"open_pr": "open_pr", "implement": "implement", "wait_for_human": "wait_for_human"},
    )
    graph.add_edge("implement", "verify")
    graph.add_edge("open_pr", "wait_for_human")
    graph.add_edge("wait_for_human", LANGGRAPH_END)

    plugin = LangGraphPlugin(
        graphs={_GRAPH_KEY: graph},
        default_activity_options=_plugin_activity_options(),
    )
    return TaskGraph(graph=graph, plugin=plugin)


def _node_metadata(name: NodeName) -> dict[str, Any]:
    """Node metadata: `execute_in` (required) plus a static summary for the Temporal UI."""
    return {
        "execute_in": NODE_EXECUTION[name],
        "summary": f"{name} ({NODE_EXECUTION[name]})",
    }


def build_plugin(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    nodes: Mapping[NodeName, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] | None = None,
) -> Any:
    """Just the plugin, for a caller that only needs the activities + interceptor (the Replayer).

    Still builds a fresh graph internally — see `build_task_graph`.
    """
    return build_task_graph(max_attempts=max_attempts, nodes=nodes).plugin


async def replay_history(
    *,
    workflow_cls: type,
    history: Any,
    plugin: Any = None,
    nodes: Mapping[NodeName, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] | None = None,
) -> Any:
    """Replay one recorded history with the plugin installed. Returns the `WorkflowReplayResult`.

    ``plugin`` defaults to a fresh one, but is accepted explicitly because passing *no* plugin is
    the failure mode the spike found: without it the Replayer cannot even validate the workflow. A
    caller that wants to prove that must construct its own plugin-less Replayer; this helper always
    supplies one, so a green replay here means "the graph still produces the same commands".
    """
    from temporalio.worker import Replayer

    active_plugin = plugin or build_plugin(nodes=nodes)
    replayer = Replayer(
        workflows=[workflow_cls],
        plugins=[active_plugin],
        build_id=None,
    )
    return await replayer.replay_workflow(history)


# `TaskSpec` is re-exported for callers that only import the graph; the import above also proves the
# contracts module is sandbox-clean, which is what lets workflows import both freely.
_ = (TaskSpec, DEFAULT_MAX_ATTEMPTS, CheckpointRequest)
