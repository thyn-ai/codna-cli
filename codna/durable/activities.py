"""The durable activities: registration, the process-wide runtime, and the public surface.

Every step that spends money, touches the disk or talks to the network is an activity, and every
activity here obeys the same three rules — because Temporal retries them, and a retry must be a
retry, never a second effect:

1. **Idempotent.** A paid turn is keyed by ``turn_key = f"{run}:{task}:{seq}"``, so a retried turn is
   answered with ``replayed: true`` instead of buying a new one. A checkpoint finds the commit it
   already made and re-points its refs. A pull request is looked up by branch before one is opened.
   The ledger dedupes on ``(session_id, event_index)``.
2. **Bounded.** Every activity is registered with a ``start_to_close_timeout`` (the D2.0 spike proved
   a missing timeout fails the workflow task) and the long ones heartbeat, so a worker that dies
   mid-turn is retried in seconds rather than at the timeout.
3. **Explicit about failure.** A failure a retry cannot fix raises `ActivityError` with a stable code.
   The workflow turns that into a parked run; the ledger says why.

This module is the REGISTRATION half. The implementations live in `activity_steps.py` and their
dependencies in `activity_support.py`; the split was forced by the repository's 1000-line ceiling
(`test_modularity.py`) and is by concern, not by size. Everything stays importable from here, so a
caller (the worker in D2.4, the graph's nodes, the tests) has one module to import.

**Why a graph node calls these functions directly.** An ``execute_in="activity"`` LangGraph node runs
*inside* an activity: `workflow.execute_activity` there raises ``_NotInWorkflowEventLoopError``
(measured, not assumed). So `graph.py`'s five doing-nodes are thin activities whose body is the step
function, and `TaskWorkflowPlain` schedules the same functions with `workflow.execute_activity` under
their registered names. One implementation, two call paths, one set of semantics.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Final, Mapping

from temporalio import activity

from . import activity_steps as _steps
from .activity_support import (
    HEARTBEAT_INTERVAL_S,
    ActivityError,
    ActivityRuntime,
    beat_now,
    default_heartbeat,
)
from .contracts import (
    ACTIVITY_TIMEOUT_S,
    ACTIVITY_TURN_TIMEOUT_S,
    ACTIVITY_VERIFY_TIMEOUT_S,
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
)
from . import goal_steps as _goal

__all__ = [
    "ACTIVITIES",
    "ACTIVITY_NAMES",
    "HEARTBEAT_INTERVAL_S",
    "ActivityError",
    "ActivityRuntime",
    "activity_executor",
    "beat_now",
    "body_for",
    "checkpoint_turn",
    "configure_default_runtime",
    "default_heartbeat",
    "impl_for",
    "notify",
    "open_pr",
    "plan_task",
    "remember_episode",
    "review_task",
    "run_cline_turn",
    "runtime_for",
    "timeout_for",
    "verify_task",
    "watch_ci",
    "recall_episodes",
    "worktree_cleanup",
]

# -- the process-wide runtime --------------------------------------------------------------
#
# The registered activity functions take exactly their request (Temporal inspects the signature, and
# a second parameter would have to be decodable from the payload), so the runtime is injected
# out-of-band. `activity_steps` reads it through the provider installed below.

DEFAULT_RUNTIME: ActivityRuntime | None = None


def configure_default_runtime(runtime: ActivityRuntime) -> ActivityRuntime:
    """Install the process-wide runtime the registered activities (and the graph's nodes) use.

    The worker calls this once at startup (D2.4). Explicit rather than lazy, so an activity that runs
    with neither a runtime nor a configured default fails loudly instead of inventing one.
    """
    global DEFAULT_RUNTIME
    DEFAULT_RUNTIME = runtime
    _steps._runtime_provider = _use  # type: ignore[attr-defined]  # the step layer's accessor
    return runtime


def _use() -> ActivityRuntime:
    if DEFAULT_RUNTIME is None:
        raise ActivityError(
            "no ActivityRuntime is configured: the worker must call "
            "codna.durable.activities.configure_default_runtime (or pass one explicitly).",
            code="activity_runtime_missing",
            details={},
        )
    return DEFAULT_RUNTIME


def runtime_for(
    runtime: ActivityRuntime | None = None, *, cancelled: Callable[[], bool] | None = None
) -> ActivityRuntime:
    """The runtime an activity uses, optionally with a thread-safe cancellation probe bound in.

    A copy, not a mutation: the workflow thread binds `cancelled` per call, and mutating the shared
    default would race two concurrent activities of the same task.
    """
    base = runtime if runtime is not None else _use()
    if cancelled is None:
        return base
    return replace(base, cancelled=cancelled)


# -- the step functions --------------------------------------------------------------------
#
# Re-exported so callers import one module. Each one is the exact object the registered activity
# wraps, which is what makes the graph's direct call and the plain engine's scheduled call the same
# code path rather than two lookalikes.

plan_task = _steps.plan_task
run_cline_turn = _steps.run_cline_turn
checkpoint_turn = _steps.checkpoint_turn
verify_task = _steps.verify_task
review_task = _steps.review_task
open_pr = _steps.open_pr
remember_episode = _steps.remember_episode
notify = _steps.notify
delegation_policy = _steps.delegation_policy
worktree_cleanup = _steps.worktree_cleanup
# D5 goal mode. Re-exported on the same rule as the step bodies: the registered activity and the
# function a test calls must be one object.
watch_ci = _goal.watch_ci
recall_episodes = _goal.recall_episodes


# -- registration --------------------------------------------------------------------------
#
# One wrapper per activity, each taking exactly its request. The registered name is independent of
# the module path so a rename cannot orphan in-flight runs, and `ACTIVITIES` is what the worker
# passes to `Worker(activities=...)`.

plan_task_activity = activity.defn(name=NAME_PLAN)(plan_task)
run_cline_turn_activity = activity.defn(name=NAME_TURN)(run_cline_turn)
checkpoint_turn_activity = activity.defn(name=NAME_CHECKPOINT)(checkpoint_turn)
verify_task_activity = activity.defn(name=NAME_VERIFY)(verify_task)
review_task_activity = activity.defn(name=NAME_REVIEW)(review_task)
open_pr_activity = activity.defn(name=NAME_OPEN_PR)(open_pr)
remember_episode_activity = activity.defn(name=NAME_EPISODE)(remember_episode)
notify_activity = activity.defn(name=NAME_NOTIFY)(notify)
delegation_policy_activity = activity.defn(name=NAME_DELEGATION_POLICY)(
    delegation_policy
)
worktree_cleanup_activity = activity.defn(name=NAME_WORKTREE_CLEANUP)(worktree_cleanup)
watch_ci_activity = activity.defn(name=NAME_WATCH_CI)(watch_ci)
recall_episodes_activity = activity.defn(name=NAME_RECALL)(recall_episodes)

ACTIVITY_NAMES: Final[tuple[str, ...]] = (
    NAME_PLAN,
    NAME_TURN,
    NAME_CHECKPOINT,
    NAME_VERIFY,
    NAME_REVIEW,
    NAME_OPEN_PR,
    NAME_EPISODE,
    NAME_NOTIFY,
    NAME_DELEGATION_POLICY,
    NAME_WORKTREE_CLEANUP,
    NAME_WATCH_CI,
    NAME_RECALL,
)

ACTIVITIES: Final[tuple[Any, ...]] = (
    plan_task_activity,
    run_cline_turn_activity,
    checkpoint_turn_activity,
    verify_task_activity,
    review_task_activity,
    open_pr_activity,
    remember_episode_activity,
    notify_activity,
    delegation_policy_activity,
    worktree_cleanup_activity,
    watch_ci_activity,
    recall_episodes_activity,
)

# The registered activity behind each name, for the two call paths that invoke one directly: the
# graph's activity nodes, and `TaskWorkflowPlain`'s `workflow.execute_activity(name, ...)`.
_BODIES: Final[dict[str, Callable[[Any], Any]]] = {
    NAME_PLAN: plan_task_activity,
    NAME_TURN: run_cline_turn_activity,
    NAME_CHECKPOINT: checkpoint_turn_activity,
    NAME_VERIFY: verify_task_activity,
    NAME_REVIEW: review_task_activity,
    NAME_OPEN_PR: open_pr_activity,
    NAME_EPISODE: remember_episode_activity,
    NAME_NOTIFY: notify_activity,
    NAME_DELEGATION_POLICY: delegation_policy_activity,
    NAME_WORKTREE_CLEANUP: worktree_cleanup_activity,
    NAME_WATCH_CI: watch_ci_activity,
    NAME_RECALL: recall_episodes_activity,
}


def body_for(name: str) -> Callable[[Any], Any]:
    """The registered activity for a name — the same object `Worker(activities=...)` takes.

    ``activity.defn`` returns the function it decorates, so this is one object for every call path:
    `Worker` registers it, `workflow.execute_activity` schedules it by name, and the graph's nodes
    call its implementation. Which is what makes "same brief, both engines" a property rather than a
    coincidence.
    """
    try:
        return _BODIES[name]
    except KeyError as exc:
        raise ActivityError(
            f"unknown activity {name!r}",
            code="activity_unknown",
            details={"name": name, "known": sorted(_BODIES)},
        ) from exc


def impl_for(name: str) -> Callable[[Any], Any]:
    """The UNWRAPPED implementation behind an activity name.

    For a caller that runs the body itself inside an activity (the graph's nodes): the decorated
    function is the same callable, but naming the intent here keeps a future decorator change from
    silently altering the direct-call path.
    """
    return body_for(name)


# The one place a workflow learns an activity's timeout. `graph.STEP_TIMEOUT_S` is asserted equal to
# this by tests, so the two engines cannot drift.
_TIMEOUTS: Final[dict[str, int]] = {
    NAME_PLAN: ACTIVITY_TURN_TIMEOUT_S,
    NAME_TURN: ACTIVITY_TURN_TIMEOUT_S,
    NAME_CHECKPOINT: ACTIVITY_TIMEOUT_S,
    NAME_VERIFY: ACTIVITY_VERIFY_TIMEOUT_S,
    NAME_REVIEW: ACTIVITY_TURN_TIMEOUT_S,
    NAME_OPEN_PR: ACTIVITY_TIMEOUT_S,
    NAME_EPISODE: ACTIVITY_TIMEOUT_S,
    NAME_NOTIFY: ACTIVITY_TIMEOUT_S,
    NAME_DELEGATION_POLICY: ACTIVITY_TIMEOUT_S,
    # A sweep of a run's worktrees: one `git worktree remove` per task, each verified against git's own
    # list. It is not a paid step, and a run's task count is bounded, so the ordinary activity ceiling
    # is right; a larger one would only hide a wedged worker behind a longer wait.
    NAME_WORKTREE_CLEANUP: ACTIVITY_TIMEOUT_S,
    # A CI read is a handful of GitHub requests (`failing_required_checks` walks the rollup, at most
    # `_ROLLUP_PAGES_MAX` pages) and no repository work, so the ordinary ceiling is generous.
    NAME_WATCH_CI: ACTIVITY_TIMEOUT_S,
    # Recall reads at most `MAX_EPISODES` small files off the run directory. Bounded by `MAX_EPISODES`
    # in the activity itself, so this is the outer limit for a large episode set on a slow disk.
    NAME_RECALL: ACTIVITY_TIMEOUT_S,
}


def timeout_for(name: str) -> int:
    """Seconds of ``start_to_close_timeout`` for an activity, refusing an unknown name."""
    try:
        return _TIMEOUTS[name]
    except KeyError as exc:
        raise ActivityError(
            f"no timeout is registered for activity {name!r}",
            code="activity_timeout_unknown",
            details={"name": name, "known": sorted(_TIMEOUTS)},
        ) from exc


def activity_executor(max_workers: int = 4) -> Any:
    """The thread pool a worker must run these activities on.

    Every activity here is a synchronous function — it does blocking work (a streamed HTTP turn, a
    subprocess, `git`, a file write) — and Temporal refuses to register a sync activity without an
    executor: ``Activity codna.step.plan is not async so an activity_executor must be present``.
    Running them on the event loop instead would stall every workflow task in the process for the
    length of a paid turn.

    Kept here so the worker (D2.4) and the tests use the SAME pool shape rather than each inventing
    one. Bounded well below the default: each slot can hold an open sidecar stream, and the plan caps
    a worker at two concurrent activities anyway.
    """
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="codna-activity")


def run_dir(runs_root: Path, run_id: str) -> Path:
    """The run's directory, created if needed. Shared by the steps that write a file beside it."""
    from .config import validate_run_id

    directory = Path(runs_root) / validate_run_id(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


_ = (default_heartbeat, beat_now, run_dir, Mapping)
