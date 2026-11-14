"""Everything the three workflows share: the activity call, the state arithmetic, and the parsers.

Split from `workflows.py` when that module crossed the repository's 1000-line ceiling
(`test_modularity.py`). This is the file both engines import, which is exactly why it is its own
module: `workflows.py` re-exports it, and `task_workflow_plain.py` / `task_workflow_graph.py` import
it directly. A helper both engines use lives HERE, once.

Every function in this module is a pure function of its arguments EXCEPT:

* `_execute` — schedules an activity. It carries the timeout and retry policy, and it is the ONLY way
  workflow code schedules one of this package's activities, because the D2.0 spike proved an activity
  scheduled without a ``start_to_close_timeout`` fails the workflow task.
* `_history_length` — reads the SDK's own event counter, which is replayable (it is derived from the
  events consumed, not from a clock).

Nothing here touches the wall clock, a random source, or I/O. Temporal replays a workflow by
re-running it, so an impurity here becomes a non-determinism error in production.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Final, Mapping

from temporalio import workflow

from .contracts import (
    ACTIVITY_HEARTBEAT_TIMEOUT_S,
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
    RunProgress,
    TaskOutcome,
    TaskSpec,
    TaskSummary,
)
from .graph import ACTIVITY_RETRIES, STEP_TIMEOUT_S
# The park vocabulary lives in a LEAF module (`parking`) so the workflow-facing `graph.py` can read it
# without importing this one — whose `codna.runtime` reach the workflow sandbox refuses. Re-exported
# here because every existing caller imports it from this module.
from .parking import park_kind, park_reason
from .state import Budget, PendingApproval, TaskState

__all__ = [
    "ACTIVITY_NAMES",
    "BOOKKEEPING_ACTIVITIES",
    "approved_outcome",
    "park_kind",
    "park_reason",
    "workflow_activity",
    "task_completion",
    "history_length",
    "approver_identity",
    "initial_state",
    "steer_note",
    "progress_from",
    "raised_cap",
    "run_budget_left",
    "run_id_of",
    "run_spec_from",
    "spec_from",
]

# The activities a workflow schedules. `STEP_TIMEOUT_S` (in `graph.py`) is the one timeout table, and
# it covers the bookkeeping activities too — a `notify` scheduled with no timeout is exactly the
# failure the spike found, and it is invisible because the notification just silently never happens.
ACTIVITY_NAMES: Final[tuple[str, ...]] = (
    NAME_PLAN,
    NAME_TURN,
    NAME_CHECKPOINT,
    NAME_VERIFY,
    NAME_REVIEW,
    NAME_OPEN_PR,
    NAME_EPISODE,
    NAME_NOTIFY,
    # Run-level, but listed here so `activity_timeout` can answer for it: `RunWorkflow` schedules it
    # through this module's helper like every other activity, and a name missing from this tuple is a
    # `ValueError` at schedule time rather than a silently untimed step.
    NAME_WORKTREE_CLEANUP,
    # D5 goal mode: scheduled by `RunWorkflow` at a task boundary, never by a graph node.
    NAME_RECALL,
    NAME_WATCH_CI,
)

BOOKKEEPING_ACTIVITIES: Final[frozenset[str]] = frozenset(
    {NAME_EPISODE, NAME_NOTIFY, NAME_WORKTREE_CLEANUP, NAME_RECALL, NAME_WATCH_CI}
)

# The transport-level retry bound lives in `graph.py` beside `STEP_TIMEOUT_S`, because BOTH engines
# read it: `workflow_activity` here (plain) and `_plugin_activity_options()` there (the graph engine,
# which could not import this module). Keeping a second copy here is what let the two drift.


def activity_timeout(name: str) -> timedelta:
    """The ``start_to_close_timeout`` for a registered activity, from the one shared table."""
    try:
        return timedelta(seconds=STEP_TIMEOUT_S[name])
    except KeyError as exc:
        raise ValueError(
            f"no step timeout for activity {name!r}; known: {', '.join(sorted(STEP_TIMEOUT_S))}"
        ) from exc


def result_type(name: str) -> type:
    """The contract type an activity returns, for decode-by-name.

    Declared per activity rather than inferred: with no ``result_type`` the SDK hands back a plain
    dict, which would let a workflow work with an UNVALIDATED mapping — the opposite of what
    `contracts.py` exists to guarantee.
    """
    from .activity_responses import (
        ActivityOutcome,
        CheckpointOutcome,
        EpisodeOutcome,
        NotifyOutcome,
        PullRequestOutcome,
    )
    from .contracts import DelegationPolicyOutcome, WorktreeCleanupOutcome
    from .goal_contracts import RecallOutcome, WatchCiOutcome

    if name == NAME_DELEGATION_POLICY:
        return DelegationPolicyOutcome
    if name == NAME_CHECKPOINT:
        return CheckpointOutcome
    if name == NAME_OPEN_PR:
        return PullRequestOutcome
    if name == NAME_EPISODE:
        return EpisodeOutcome
    if name == NAME_NOTIFY:
        return NotifyOutcome
    if name == NAME_WORKTREE_CLEANUP:
        return WorktreeCleanupOutcome
    if name == NAME_WATCH_CI:
        return WatchCiOutcome
    if name == NAME_RECALL:
        return RecallOutcome
    return ActivityOutcome


def _retry_policy() -> Any:
    from temporalio.common import RetryPolicy

    return RetryPolicy(maximum_attempts=ACTIVITY_RETRIES)


def workflow_activity(name: str, request: Any) -> Any:
    """Schedule one of this package's activities.

    Every call carries a timeout (the spike proved a missing one fails the workflow task) and an
    explicit ``result_type`` (so the response is the validated contract type, not a dict). This is the
    only place workflow code is allowed to call `workflow.execute_activity`, so neither bound can be
    forgotten in one place and remembered in another.
    """
    return workflow.execute_activity(
        name,
        request,
        start_to_close_timeout=activity_timeout(name),
        heartbeat_timeout=(
            timedelta(seconds=ACTIVITY_HEARTBEAT_TIMEOUT_S) if name == NAME_TURN else None
        ),
        retry_policy=_retry_policy(),
        result_type=result_type(name),
    )


def adopt(state: TaskState, outcome: Any, spend: list[float]) -> TaskState:
    """Adopt an activity outcome: its state payload, plus its cost delta into `spend`.

    The returned state already carries the turn's spend (it must, so `spent_usd` stays monotone
    across retries). The DELTA is accumulated separately because it is what the RUN's budget sums:
    summing budgets would double count, summing deltas cannot.
    """
    spend[0] += float(getattr(outcome, "cost_usd", 0.0))
    return TaskState.from_payload(outcome.state)


def initial_state(spec: TaskSpec, remaining_usd: float | None) -> TaskState:
    """A task's starting state, its budget capped by what the run has left."""
    from dataclasses import replace

    capped = spec.with_remaining_budget(remaining_usd)
    base = TaskState.from_payload(spec.initial_state())
    return replace(
        base,
        budget=Budget(
            max_cost_usd=capped.max_cost_usd,
            spent_usd=0.0,
            max_tokens=capped.max_tokens,
            spent_tokens=0,
            unpriced_tokens=0,
        ),
    )


def approved_outcome(state: TaskState) -> TaskOutcome:
    """The outcome of a task whose review APPROVED — `approved`, or `approved_unpublished`.

    PURE, and shared by both engines: the graph engine and the plain engine must not disagree about
    whether an unpublished task succeeded, and the only state that can answer is the one the publish
    step returned.

    ``publish_skip`` set means publishing was SKIPPED (a repository with no GitHub remote, or a
    remote that is not GitHub): the work is done — committed, verified and approved — and there was
    nothing to open a pull request against. That is ``approved_unpublished`` rather than ``approved``,
    because reporting it as a plain approval would let a reader claim a pull request exists when none
    does. A REFUSAL (a GitHub remote with no token) never reaches here: it is an activity failure, so
    the task fails loudly instead of finishing as if publishing had been unnecessary.

    A state with neither a pull request nor a skip is still ``approved`` here: this function is only
    called on the approved path, and the older histories it must replay reached that point with a
    ``pr`` (or with the old hard failure, which never produced a completion at all).
    """
    return "approved_unpublished" if state.publish_skip is not None else "approved"


def approver_identity(payload: Any, *, handler: str) -> str:
    """The approver's identity from an Update payload, refusing an anonymous one.

    An approval without a name is worthless as an audit record and invites a UI that posts one
    because it is easier. Checked in a VALIDATOR before the decision is recorded, so a malformed
    Update leaves the run exactly as it was.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{handler}: expected an object with an 'identity' field")
    raw = payload.get("identity")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{handler}: 'identity' must be a non-empty string (who is deciding?)")
    return raw.strip()


def steer_note(payload: Any) -> str:
    """A steering note from an Update payload, bounded and non-empty."""
    from .contracts import MAX_STEER_NOTE_CHARS

    if not isinstance(payload, Mapping):
        raise ValueError("steer: expected an object with a 'note' field")
    raw = payload.get("note")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("steer: 'note' must be a non-empty string")
    if len(raw) > MAX_STEER_NOTE_CHARS:
        raise ValueError(f"steer: 'note' is over the {MAX_STEER_NOTE_CHARS}-character limit")
    return raw.strip()


def raised_cap(current: float | None, raw: Any) -> float:
    """A new cost cap from an Update, refusing anything that is not a positive number.

    A cap can only be RAISED: a lower number would retroactively put the task over budget, which parks
    every run that is already inside its ceiling.
    """
    if not isinstance(raw, (int, float)) or isinstance(raw, bool) or float(raw) <= 0:
        raise ValueError("set_budget: 'max_cost_usd' must be a positive number")
    value = float(raw)
    return value if current is None else max(current, value)


def task_completion(
    state: TaskState,
    outcome: str,
    approvals: list[dict[str, Any]],
    engine: str,
    cost_usd: float,
    *,
    episode_path: str = "",
    episode_digest: str = "",
) -> Any:
    """A task's result, with the approving identity from the LAST approval.

    ``episode_path``/``episode_digest`` are the `remember_episode` outcome the task recorded. They
    travel on the completion because the run needs them to build its episode references, and asking
    the activity again would write a second note for one episode.
    """
    from .contracts import TaskCompletion

    approved_by = next(
        (str(item["identity"]) for item in reversed(approvals) if item.get("kind") == "approve"),
        None,
    )
    return TaskCompletion(
        run_id=state.run_id,
        task_id=state.task_id,
        outcome=outcome,
        state=state.to_payload(),
        approved_by=approved_by,
        cost_usd=cost_usd,
        turn_seq=state.turn_seq,
        checkpoints=(state.checkpoint_ref.ref,) if state.checkpoint_ref else (),
        engine=engine,
        episode_path=episode_path,
        episode_digest=episode_digest,
    )


def spec_from(payload: Mapping[str, Any]) -> TaskSpec:
    """Decode a task spec from a workflow argument, refusing a malformed one before any work starts."""
    spec = TaskSpec(
        schema_version=int(payload.get("schema_version", 1)),
        run_id=payload.get("run_id"),  # type: ignore[arg-type]
        task_id=payload.get("task_id"),  # type: ignore[arg-type]
        task_number=payload.get("task_number"),  # type: ignore[arg-type]
        brief=payload.get("brief"),  # type: ignore[arg-type]
        repo_dir=payload.get("repo_dir"),  # type: ignore[arg-type]
        base_branch=payload.get("base_branch", "main"),
        verify_commands=tuple(payload.get("verify_commands") or ()),
        max_attempts=int(payload.get("max_attempts", 3)),
        max_cost_usd=payload.get("max_cost_usd"),
        max_tokens=payload.get("max_tokens"),
        engine=payload.get("engine", "langgraph"),
        # Absent in a payload recorded before the provider fields existed, which is why these
        # default to None rather than being required: a run already in flight must still replay.
        provider=payload.get("provider"),
        model=payload.get("model"),
        base_url=payload.get("base_url"),
        provider_key_source=payload.get("provider_key_source"),
        # Absent in a payload recorded before per-task worktrees existed, hence the default: such a
        # task worked in the shared checkout and must keep doing so, which is what `False` means.
        worktree=bool(payload.get("worktree", False)),
        # Absent in a payload recorded before goal mode existed: `""` is "no goal context", which is
        # exactly what a task-mode run's plan turn has always had.
        goal_context=str(payload.get("goal_context") or ""),
        worktree_task_id=str(payload.get("worktree_task_id") or ""),
        # Absent in a payload recorded before project rules existed: `False` is what that history
        # meant, so such a run replays with the prompts it actually sent.
        attach_project_rules=bool(payload.get("attach_project_rules", False)),
        # Absent in a payload recorded before user-defined subagents existed: `False` is what that
        # history meant, so such a run replays the sessions it actually recorded.
        attach_subagents=bool(payload.get("attach_subagents", False)),
        # Absent before MCP could be carried: `()` is exactly what such a run sent, so it replays with
        # the sessions it recorded rather than gaining an MCP attachment it never had.
        mcp_servers=_recorded_mcp_servers(payload.get("mcp_servers")),
        # D7/SWARM: absent before delegation could be carried, and `False` is exactly what such a run
        # sent (it sent neither field), so its replay creates the same sessions it recorded.
        enable_spawn_agent=bool(payload.get("enable_spawn_agent", False)),
        enable_agent_teams=bool(payload.get("enable_agent_teams", False)),
        # Absent before run-from-a-branch existed: `""` is "my own branch", so such a run replays
        # adopting nothing rather than being pointed at a branch it never used.
        worktree_branch=str(payload.get("worktree_branch") or ""),
        auto_approve_local_only=payload.get("auto_approve_local_only", False),
        approval_policy_digest=payload.get("approval_policy_digest", ""),
    )
    spec.validate()
    return spec


def _recorded_mcp_servers(raw: Any) -> tuple[dict[str, Any], ...]:
    """A recorded ``mcp_servers`` list, re-validated through the one normalizer (see `mcp.py`)."""
    from .mcp import normalize_mcp_servers

    return normalize_mcp_servers(raw)


def run_spec_from(payload: Mapping[str, Any]) -> Any:
    from .contracts import RunSpec
    from .goal import EpisodeRef, GoalSpec

    spec = RunSpec(
        schema_version=int(payload.get("schema_version", 1)),
        run_id=payload.get("run_id"),  # type: ignore[arg-type]
        repo_dir=payload.get("repo_dir"),  # type: ignore[arg-type]
        base_branch=payload.get("base_branch", "main"),
        tasks=tuple(spec_from(task) for task in payload.get("tasks") or ()),
        max_cost_usd=payload.get("max_cost_usd"),
        engine=payload.get("engine", "langgraph"),
        idle_between_tasks_s=float(payload.get("idle_between_tasks_s") or 0.0),
        continue_as_new_tasks=int(payload.get("continue_as_new_tasks") or 10),
        continue_as_new_history_events=int(payload.get("continue_as_new_history_events") or 5000),
        # Absent in a payload recorded before per-task worktrees existed: such a run never created a
        # worktree, so `False` is not merely a default, it is what that history MEANS.
        worktree=bool(payload.get("worktree", False)),
        # Absent in a payload recorded before goal mode existed: `None`/empty means a task-mode run,
        # which is exactly what such a history is. Decoding it as "a goal with no stop condition"
        # would change its behaviour retroactively.
        goal=GoalSpec.from_payload(payload.get("goal")),
        episodes=tuple(
            EpisodeRef.from_payload(item) for item in payload.get("episodes") or ()
        ),
        ci_state=payload.get("ci_state"),
        ci_attempts=int(payload.get("ci_attempts") or 0),
        # D7/MULTITASK: absent means `1`, which is exactly the strictly sequential run such a history
        # recorded — decoding it as "run everything at once" would change its behaviour retroactively.
        max_parallel_tasks=int(payload.get("max_parallel_tasks") or 1),
    )
    spec.validate()
    return spec


def progress_from(payload: Mapping[str, Any] | None) -> RunProgress:
    return RunProgress.from_payload(dict(payload) if payload else None)


def run_budget_left(spec: Any, spent: float) -> float | None:
    if spec.max_cost_usd is None:
        return None
    return float(spec.max_cost_usd) - spent


def history_length() -> int:
    """This execution's history length, from the SDK's own counter.

    There is no public accessor in ``temporalio`` 1.34, so this reaches the runtime the SDK keeps on
    the workflow loop. Read defensively: a future SDK that moves it must not break a run, and a wrong
    value costs an early continue-as-new, never correctness. Unlike a clock or a byte count anyone
    measures, it is derived from the events this execution consumed, so it is replayable.
    """
    try:
        from temporalio.workflow import _Runtime

        return int(_Runtime.current().workflow_get_current_history_length())
    except Exception:  # noqa: BLE001 - see docstring
        return 0


def run_id_of(workflow_id: str) -> str:
    """Derive a run id from a workflow id, for a `status` query that arrives before the first task.

    ``run:<id>`` and ``task-<run>-<task>`` are the id shapes the CLI starts. Purely lexical: no
    payload decode, no I/O, replay-safe.
    """
    return workflow_id.split(":", 1)[1] if ":" in workflow_id else workflow_id


def park_approval(state: TaskState, reason: str) -> PendingApproval:
    return PendingApproval(
        kind=park_kind(state), reason=reason, requested_at_index=state.last_event_index
    )


def remaining_tasks(spec: Any, completed: list[TaskSummary]) -> tuple[TaskSpec, ...]:
    """The tasks not yet done, in order. Keyed by id, never by position."""
    return spec.remaining(tuple(completed))
