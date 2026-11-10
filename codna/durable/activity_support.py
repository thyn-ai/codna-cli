"""The runtime, the error and the paid-turn machinery every durable activity shares.

Split from `activities.py` when that module crossed the repository's 1000-line ceiling
(`test_modularity.py`). The boundary is "what an activity body needs" vs. "what a body does":

* this module owns the INJECTED DEPENDENCIES (`ActivityRuntime`), the error type the whole step
  layer raises (`ActivityError`), request decoding/validation, the git runner, and the one piece of
  machinery that must never be re-implemented per step — `drive_turn`, which buys at most one paid
  turn per ``turn_key``;
* `activity_steps.py` owns the eight step implementations;
* `activities.py` owns registration and the process-wide runtime, and re-exports both.

`drive_turn` lives HERE rather than in `activity_steps.py` for the same reason git does: it is a
cross-cutting dependency of a step body, and keeping it in the shipped runtime module makes its
contract — one paid turn per ``(run, task, turn_seq)`` — impossible to bypass by adding a new step
that posts to the sidecar itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from temporalio import activity

from .contracts import HEARTBEAT_INTERVAL_S
from .ledger import LedgerRecord
# `_project_rules` is imported for RE-EXPORT: `test_goal_rules.py` exercises it directly, and it
# has always been reachable from this module. `_session_spec` is what the two call sites below use.
from .session_spec import _project_rules, _session_spec  # noqa: F401
from .state import TaskState, TaskStateError
from .worktree import GIT_ENV_DROP

__all__ = [
    "ActivityError",
    "ActivityRuntime",
    "HEARTBEAT_INTERVAL_S",
    "RESUMED_EVENT",
    "TurnOutcome",
    "beat_now",
    "decode_request",
    "default_heartbeat",
    "drive_turn",
    "ensure_checkout",
    "git",
    "git_ok",
    "owner_for",
    "redact",
    "working_dir",
]


class ActivityError(RuntimeError):
    """A durable step failed in a way a retry cannot fix. Always carries a stable code."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass
class ActivityRuntime:
    """Every external dependency an activity has, injected.

    This is what lets the activities run in a test with no sidecar, no network and no git remote —
    and it is how the worker injects the keystore-resolved token (D2.4), so no activity reads a
    provider key itself. Fields are explicit rather than a bag of kwargs: a new dependency is a
    visible change to this contract, not a hidden `os.environ` read buried in an activity.
    """

    runs_root: Path
    # Takes the OWNER of the run whose turn is about to run. The engine stamps a session with its
    # owner at CREATE time and refuses any later request whose `x-session-owner` disagrees, so the
    # client that starts a turn MUST be built with the same owner the session was created with.
    # A no-argument factory cannot express that: it would default to "anonymous" and be refused
    # with `session_owner_mismatch` on its own session's first turn. The owner is the run id.
    sidecar_factory: Callable[[str], Any]
    ledger_factory: Callable[[str], Any]
    clock: Callable[[], float] = time.time
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None
    log: Callable[[str], None] | None = None
    # Resolved lazily so importing this module never needs credentials. Returns a token or None.
    github_token: Callable[[], str | None] | None = None
    # Cancellation, as a plain callable: a body runs on a worker thread, where Temporal's activity
    # context (a contextvar) is absent, so the workflow thread's probe is what the body sees.
    cancelled: Callable[[], bool] | None = None
    # Test seams for the three steps that would otherwise shell out or call a model:
    run_tests: Callable[[Any, tuple[str, ...]], Any] | None = None
    run_review: Callable[[Any, str], Any] | None = None
    push_and_open: Callable[[Any, str], Mapping[str, Any]] | None = None
    git: Callable[[str, Sequence[str]], str] | None = None
    # D5 goal mode: the CI gate's seam. A callable taking a `WatchCiRequest` and returning a
    # `WatchCiOutcome`, so a test (and a host with another CI provider) drives the loop without a
    # GitHub token — while the REAL body still runs, writing its ledger note and classification.
    watch_ci: Callable[[Any], Any] | None = None

    def emit(self, event: Mapping[str, Any]) -> None:
        if self.log is not None:
            self.log(json.dumps(dict(event), sort_keys=True, default=str))

    def beat(self, details: Mapping[str, Any]) -> None:
        (self.heartbeat or default_heartbeat)(details)

    def is_cancelled(self) -> bool:
        """Whether this activity was cancelled. On a worker thread the probe is the bound one."""
        if self.cancelled is not None:
            return bool(self.cancelled())
        return _temporal_cancelled()

    def ledger(self, run_id: str) -> Any:
        """The run's ledger. One indirection so every step opens it the same way."""
        return self.ledger_factory(run_id)


def default_heartbeat(details: Mapping[str, Any]) -> None:
    """Heartbeat through Temporal when there is an activity context; no-op otherwise.

    These functions also run in tests, where there may be no activity context at all.
    `activity.heartbeat` raises there, and a heartbeat that raises would fail a step that is
    otherwise fine, so the absence of a context is detected rather than assumed.
    """
    try:
        activity.heartbeat(dict(details))
    except RuntimeError:
        return


def beat_now(details: Mapping[str, Any]) -> None:
    """Heartbeat from a workflow thread while a body runs on a worker thread (the graph's nodes)."""
    default_heartbeat(details)


def _temporal_cancelled() -> bool:
    try:
        return bool(activity.is_cancelled())
    except RuntimeError:
        return False


def decode_request(request: Any, label: str) -> TaskState:
    """Validate a request's contract checks and return its decoded `TaskState`.

    Runs on EVERY attempt, including a replayed one: a retry that receives a payload this build
    cannot validate must fail here rather than three steps later.
    """
    try:
        payload = request.validate()
    except (TypeError, ValueError) as exc:
        raise ActivityError(
            f"{label}: invalid request: {exc}",
            code=str(getattr(exc, "code", "activity_request_invalid")),
            details={"cause": getattr(exc, "details", None) or {}},
        ) from exc
    try:
        return TaskState.from_payload(payload)
    except TaskStateError as exc:
        raise ActivityError(
            f"{label}: invalid task state: {exc}",
            code="activity_state_invalid",
            details={"cause": exc.as_dict()["error"]},
        ) from exc


# -- git --------------------------------------------------------------------------------

# Environment variables that would redirect git away from the checkout we were given. ONE list, in
# `worktree.py` (which needs it for the same reason), so the two git runners cannot drift apart.
_GIT_ENV_DROP: Final[tuple[str, ...]] = GIT_ENV_DROP


def redact(text: str) -> str:
    """Strip an embedded credential from a URL: ``https://x-access-token:<t>@host`` → ``https://host``."""
    marker = "x-access-token:"
    index = text.find(marker)
    if index < 0:
        return text
    at = text.find("@", index)
    if at < 0:
        return text
    scheme_end = text.find("://")
    prefix = text[: scheme_end + 3] if scheme_end >= 0 else ""
    return f"{prefix}{text[at + 1 :]}"


def _default_git(repo_dir: str, args: Sequence[str]) -> str:
    """Run one git command in ``repo_dir`` and return stdout, or raise with git's own message.

    A cleaned environment on purpose: a worker inherits its launcher's environment, and an inherited
    ``GIT_DIR`` would silently point these commits at an unrelated repository. Arguments are
    redacted before they reach an error or a log: a caller may pass an authenticated push URL.
    """
    env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}
    try:
        completed = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ActivityError(
            f"git could not run in {repo_dir}: {exc}",
            code="activity_git_unavailable",
            details={"repo_dir": repo_dir, "error": str(exc)},
        ) from exc
    if completed.returncode != 0:
        raise ActivityError(
            f"git {redact(' '.join(args[:2]))} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[:400]}",
            code="activity_git_failed",
            details={
                "repo_dir": repo_dir,
                "args": [redact(arg) for arg in args],
                "returncode": completed.returncode,
            },
        )
    return completed.stdout


def git(runtime: ActivityRuntime, repo_dir: str, args: Sequence[str]) -> str:
    return (runtime.git or _default_git)(repo_dir, args)


def git_ok(runtime: ActivityRuntime, repo_dir: str, args: Sequence[str]) -> str | None:
    """Run git where a non-zero exit is an ANSWER ("does this ref exist?"), not a failure."""
    try:
        return git(runtime, repo_dir, args)
    except ActivityError:
        return None


# -- the task's working directory (D5) ----------------------------------------------------
#
# EVERY repo-touching step asks these two functions where to work, and nothing else. That is the whole
# threading story: `TaskContext.repo_dir` stays the SHARED checkout (the repository a worktree's objects
# and refs belong to), `working_dir(context)` is the tree a step should edit, and `ensure_checkout`
# makes the latter exist. A step that used `context.repo_dir` directly would silently edit the shared
# tree for a worktree-backed task — which is the bug this indirection exists to make impossible.


def _checkout_id(context: Any) -> str:
    """The task id whose CHECKOUT this context works in (D5 goal mode's fix-CI task shares its parent's).

    Read off the context, never remembered: a retried activity derives the same path from its own
    request. The fallback keeps every pre-D5 context (and every context built without the field)
    meaning "my own worktree", which is what it always did.
    """
    return getattr(context, "worktree_task_id", "") or context.task_id


def working_dir(runtime: ActivityRuntime, context: Any) -> str:
    """The tree this task's steps work in: its own worktree, or the shared checkout.

    Derived, never remembered, so a retried activity computes the same answer from its own request.
    ``context.worktree`` False is the pre-D5 behaviour and returns ``repo_dir`` unchanged.
    """
    if not getattr(context, "worktree", False):
        return context.repo_dir
    from .worktree import worktree_path_for

    return str(
        worktree_path_for(runtime.runs_root, context.run_id, _checkout_id(context))
    )


def ensure_checkout(runtime: ActivityRuntime, context: Any) -> str:
    """``working_dir`` for this task, CREATED if the task asked for a worktree.

    Called at the START of every repo-touching step rather than once, because a worker may pick up a
    task whose activity was already started on another worker (or whose worktree was reclaimed by an
    operator): the step that runs must be able to rely on the tree existing, and only the step that
    runs knows it is running. Idempotent, so the second call in a turn costs one ``git worktree list``.

    Refuse-with-a-code on every failure: a task that cannot get its own checkout must not quietly fall
    back to the shared one, where its commits would land on another task's tree.
    """
    from .worktree import WorktreeError, create_task_worktree

    base = context.repo_dir
    if not getattr(context, "worktree", False):
        return base
    try:
        made = create_task_worktree(
            repo_dir=base,
            runs_root=runtime.runs_root,
            run_id=context.run_id,
            task_id=_checkout_id(context),
            task_number=context.task_number,
            base_branch=context.base_branch,
            adopt_branch=getattr(context, "worktree_branch", ""),
        )
    except WorktreeError as exc:
        raise ActivityError(
            f"could not prepare the worktree for task {context.task_id}: {exc}",
            code="activity_worktree_unavailable",
            details={"cause_code": exc.code, **exc.details},
        ) from exc
    return str(made.path)


# -- the paid turn ------------------------------------------------------------------------
#
# WHERE THE MONEY IS, SO WHERE THE IDEMPOTENCY HAS TO BE EXACT.
#
# WHAT A RETRY ACTUALLY RECEIVES. A Temporal activity retry is handed the SAME input from history,
# so it sees the *pre-turn* `TaskState`: the mutation that records "this turn started" is produced by
# the attempt being retried and therefore cannot be in the retry's input. Any fact a retry must know
# has to live somewhere it can READ — the run's append-only ledger. That is why the durable record of
# a started turn is a `turn.started` LEDGER record, written the moment the engine accepts the POST.
#
# WHY THE ENGINE'S `turn_key` IS NOT ENOUGH ON ITS OWN. `POST /v1/sessions/:id/turns` replays a
# repeated `x-codna-turn-key` instead of billing again — but the engine's index is keyed by SESSION
# (`server/state.ts`: `turnKeyIndex: Map<sessionId, Map<key, record>>`; `turn-key.ts::lookupTurnKey`
# reads `state.turnKeyIndex.get(sessionId)?.get(key)`). And `resume_from` returns the same session id
# only while the engine's own process still holds it: `run-server.ts::handleCreateSession` calls
# `resumableSession(...)`, answers 200 with the SAME id when it can, and otherwise mints a NEW id and
# sets `resume_from_ignored: true`. So after a sidecar restart the same key lands on a fresh, empty
# index and buys a second turn — precisely the failure `turn_key_rescoped` exists to detect. The
# durable side therefore never relies on the replay index for the money guarantee:
#
# 1. **Look before you POST.** A `turn.started` already recorded for this `turn_key` means the turn
#    was already bought. Do not POST.
# 2. **Finished turns come from the ledger.** If the recorded turn also has its `turn.finished`, the
#    attempt returns that record's spend and event index without touching the engine at all.
# 3. **Unfinished turns are re-attached, never re-bought.** The attempt streams the recorded turn id
#    from the ledger's high-water `event_index` for that session.
# 4. **A lost session is a loud failure.** If the recorded turn never finished and its session can no
#    longer be resumed, the attempt raises `activity_turn_unobservable`. Starting a new paid turn
#    instead would be silent double billing; that is the one thing this module exists to prevent.

# Recorded as a `note` when an attempt RESUMED rather than started. Never a second `turn.started`:
# the soak's duplicate checks are exactly "more than one start under one key", so recording a resume
# as a start would make a healthy retry look like the failure those checks exist to catch.
RESUMED_EVENT: Final = "turn.resumed_from_ledger"


@dataclass(frozen=True)
class TurnOutcome:
    """What one turn attempt produced: identity, stream position, and the spend DELTA to fold in."""

    session_id: str
    turn_id: str
    event_index: int
    cost_usd: float = 0.0
    tokens: int = 0
    priced: bool = True
    # True when this attempt did not start the turn: it resumed an engine replay or a ledger record.
    resumed: bool = False
    # The engine's own `finish_reason` for the turn. `"ask"` is the one value that changes what the
    # workflow does: the agent stopped because a tool call needed a human, so there is work left that
    # only a decision can release. Every other value means the turn ran to its own end.
    finish_reason: str = "completed"
    # The ask the turn parked on, as the engine emitted it (`ask_id`, `kind`, `tool`, `prompt`).
    # Present exactly when `finish_reason == "ask"`.
    ask: Mapping[str, Any] | None = None


def owner_for(context: Any) -> str:
    """The sidecar owner scope: a durable run is its own owner, so it cannot re-attach to a chat."""
    return f"codna-run-{context.run_id}"


@dataclass(frozen=True)
class _RecordedTurn:
    session_id: str
    turn_id: str
    high_water: int


def _scan_ledger(ledger: Any, turn_key: str) -> tuple[_RecordedTurn | None, LedgerRecord | None]:
    """One pass over the ledger: the recorded turn for ``turn_key``, plus its completion if any.

    ONE pass on purpose. `RunLedger.records()` re-parses the whole NDJSON file and a run writes one
    line per streamed event, so scanning once per fact would make a long run quadratic in its own
    ledger. Everything a retry needs — was this key bought, in which session, how far did its stream
    get, and did it finish — comes out of this single traversal.

    The FIRST `turn.started` for the key wins: it is the attempt that actually bought the turn, and
    later records under the same key would describe the same turn.

    The high-water floor is scoped to the recorded turn's SESSION. Event indices are per session, so
    a floor taken across sessions could sit above this turn's real position and silently skip events
    — the one error a resume must never make.

    **The floor steps back below a TERMINAL event, because `?after=` is EXCLUSIVE.** An attempt can
    append the turn's terminal event and then die before its own bookkeeping (measured: the turn
    aborted on a model loop, `session.turn_aborted` was ledgered, and the activity raised one line
    before writing `turn.finished`). A retry then re-attached with ``?after=<that terminal's index>``
    — asking the sidecar for everything AFTER the event that ends the turn. Nothing more was ever
    emitted, so the stream followed no live events and blocked until the 60-minute activity ceiling,
    heartbeating happily: the run looked alive forever and its ledger never grew again.
    A floor of ``index - 1`` replays exactly that one terminal, which is what lets the retry observe
    the ending and fail by name instead of hanging.
    """
    from .sidecar_client import SSE_TERMINAL_TURN_EVENTS

    recorded: _RecordedTurn | None = None
    completion: LedgerRecord | None = None
    high_water = 0
    terminal_index: int | None = None
    for record in ledger.records():
        if recorded is None:
            if record.kind != "turn.started":
                continue
            payload = record.payload
            if payload.get("turn_key") != turn_key:
                continue
            turn_id = payload.get("turn_id")
            session_id = payload.get("session_id")
            if not isinstance(turn_id, str) or not turn_id:
                continue
            if not isinstance(session_id, str) or not session_id:
                continue
            # `turn.started` precedes every event of its own turn, so from here the session is known
            # and the floor can be scoped to it.
            recorded = _RecordedTurn(session_id=session_id, turn_id=turn_id, high_water=0)
            continue
        if record.session_id == recorded.session_id and record.event_index is not None:
            high_water = max(high_water, record.event_index)
            if record.kind == "turn.event":
                event_payload = record.payload
                if (
                    isinstance(event_payload, Mapping)
                    and event_payload.get("event_type") in SSE_TERMINAL_TURN_EVENTS
                ):
                    terminal_index = record.event_index
        if record.kind == "turn.finished" and record.payload.get("turn_id") == recorded.turn_id:
            completion = record
    if recorded is None:
        return None, None
    # Only when the terminal is the LAST thing this turn recorded: then it is the floor, `?after=`
    # would skip it, and replaying one event is both safe and necessary. A terminal followed by
    # later events means the stream was already reading past it, so the floor is left alone.
    if completion is None and terminal_index is not None and terminal_index == high_water:
        high_water = max(0, high_water - 1)
    return _RecordedTurn(recorded.session_id, recorded.turn_id, high_water), completion


def _note_resume(
    ledger: Any, task_id: str, *, turn_key: str, turn_id: str, source: str
) -> None:
    """Record that this attempt resumed instead of starting, as a note (never a second start)."""
    ledger.append(
        "note",
        task_id=task_id,
        payload={
            "event": RESUMED_EVENT,
            "turn_key": turn_key,
            "turn_id": turn_id,
            "source": source,
        },
    )


def _ensure_session(
    runtime: ActivityRuntime,
    client: Any,
    context: Any,
    state: TaskState,
    *,
    read_only: bool,
    working_tree: str,
) -> str:
    """Create the session, or resume the one the state remembers.

    ``resume_from`` keeps ONE conversation across a task's turns. Read-only is enforced by the ENGINE
    (an empty write allow-list), not by asking the model nicely: the prompt's instruction is a
    courtesy, the allow-list is the control.

    ``working_tree`` is the tree the ENGINE is told to edit — the task's own worktree when it has one.
    It is a parameter rather than ``context.repo_dir`` because that distinction is exactly the D5 bug
    this threading exists to prevent: a session pointed at the shared checkout would edit another
    task's tree no matter where our own commits landed.
    """
    session = client.create_session(
        working_dir=working_tree,
        resume_from=state.session_id,
        spec=_session_spec(context, read_only=read_only, working_tree=working_tree),
    )
    # `resume_from_ignored` is NOT the only way continuity can move to a new id, so it is not the test.
    # The engine answers a create for a session its process no longer holds by RECOVERING the persisted
    # transcript and seeding a NEW session from it: that reply is `resumed: true` with
    # `resume_from_ignored: false` and a DIFFERENT `session_id` (measured against a real kill -9 restart).
    # Keying the notice off `resume_from_ignored` alone therefore hid exactly the case a restart produces.
    # The honest test is the id itself: any change means the conversation continued elsewhere, and a reader
    # of the run must be able to see that rather than infer it from a mismatched session id three turns on.
    if state.session_id is not None and session.session_id != state.session_id:
        runtime.emit(
            {
                "event": "turn.session_restarted",
                "run_id": context.run_id,
                "task_id": context.task_id,
                "previous_session_id": state.session_id,
                "session_id": session.session_id,
                # Whether the engine carried the transcript over, so "the conversation survived, in a new
                # session" is distinguishable from "the conversation was lost".
                "resumed_from_history": bool(session.resumed),
                "resume_from_ignored": bool(session.resume_from_ignored),
            }
        )
    return session.session_id


def _abort(client: Any, session_id: str) -> None:
    try:
        client.abort(session_id)
    except Exception as exc:  # noqa: BLE001 - reported, then the cancellation is re-raised
        raise ActivityError(
            f"could not abort session {session_id} after cancellation: {exc}",
            code="activity_abort_failed",
            details={"session_id": session_id, "error": str(exc)},
        ) from exc


def _close(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class _Spend:
    """One turn's spend, as a DELTA. The workflow folds it in, so replay cannot double count."""

    cost_usd: float = 0.0
    tokens: int = 0
    priced: bool = True


def _token_total(usage: Mapping[str, Any]) -> int:
    """Sum the token fields the engine may report, tolerating any of them being absent."""
    total = 0
    for key in (
        "inputTokens",
        "outputTokens",
        "cacheReadTokens",
        "cacheWriteTokens",
        "tokens_in_uncached",
        "tokens_out",
    ):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def _spend_from_event(payload: Mapping[str, Any], *, state: TaskState) -> _Spend:
    """Per-turn spend from the terminal event's CUMULATIVE usage.

    The sidecar reports cumulative session usage on the terminal event (verified in `execution.ts`),
    so this turn's delta is that snapshot minus what the state already accounts for. A payload with
    no cost at all is recorded as UNPRICED rather than free: a run that cannot price a turn must be
    able to say "this much is unknown" (`Budget.unpriced_tokens`).
    """
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return _Spend(0.0, 0, False)
    raw_cost = usage.get("totalCost", usage.get("total_cost"))
    priced = isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool) and raw_cost > 0
    cost_delta = max(0.0, float(raw_cost) - float(state.budget.spent_usd)) if priced else 0.0
    token_delta = max(0, _token_total(usage) - int(state.budget.spent_tokens))
    return _Spend(cost_delta, token_delta, priced)


def _resume_finished(
    runtime: ActivityRuntime,
    ledger: Any,
    task_id: str,
    state: TaskState,
    recorded: _RecordedTurn,
    finished: LedgerRecord,
    phase: str,
) -> TurnOutcome:
    """The turn is already complete: return what the ledger recorded. No engine, no engine spend.

    Reporting the recorded cost is correct and NOT double counting. This path runs only when an
    attempt died after its ``turn.finished`` append but before the workflow adopted its result, so
    that cost was never folded into the run's budget: `adopt` sums the outcome's delta, and there was
    no outcome to sum. Returning zero would under-report a turn the user paid for.
    """
    payload = finished.payload
    recorded_ask = payload.get("ask")
    outcome = TurnOutcome(
        session_id=recorded.session_id,
        turn_id=recorded.turn_id,
        event_index=finished.event_index if finished.event_index is not None else state.last_event_index,
        cost_usd=float(payload.get("cost_usd") or 0.0),
        tokens=int(payload.get("tokens") or 0),
        priced=bool(payload.get("priced", True)),
        resumed=True,
        # A turn that ended on a question ENDS ON A QUESTION for every attempt, including one that
        # only ever reads the ledger. Dropping it here would make a crash-retried ask look like a
        # completed turn — the run would try to verify work that never happened.
        finish_reason=str(payload.get("finish_reason") or "completed"),
        ask=dict(recorded_ask) if isinstance(recorded_ask, Mapping) else None,
    )
    turn_key = f"{state.run_id}:{state.task_id}:{state.turn_seq + 1}"
    _note_resume(ledger, task_id, turn_key=turn_key, turn_id=outcome.turn_id, source="turn.finished")
    runtime.emit(
        {
            "event": RESUMED_EVENT,
            "run_id": state.run_id,
            "task_id": task_id,
            "phase": phase,
            "turn_id": outcome.turn_id,
            "turn_key": turn_key,
            "source": "turn.finished",
            "cost_usd": outcome.cost_usd,
        }
    )
    return outcome


def _resume_in_flight(
    runtime: ActivityRuntime,
    client: Any,
    ledger: Any,
    task_id: str,
    context: Any,
    state: TaskState,
    recorded: _RecordedTurn,
    turn_key: str,
    phase: str,
    notes: Sequence[str],
    *,
    read_only: bool,
    working_tree: str,
) -> TurnOutcome:
    """A turn was bought but never finished: re-attach to it, or fail without buying another.

    The session check is what makes this honest. `create_session(resume_from=…)` returns the SAME id
    only while the engine still holds the session; when it reports ``resume_from_ignored`` (or a
    different id) the recorded turn id no longer exists in that engine, so the turn cannot be
    observed — and starting a new one would bill the same work twice under one ``turn_key``. That is
    a hard failure, not a retry.
    """
    session = client.create_session(
        working_dir=working_tree,
        resume_from=recorded.session_id,
        spec=_session_spec(context, read_only=read_only, working_tree=working_tree),
    )
    if session.session_id != recorded.session_id or session.resume_from_ignored:
        raise ActivityError(
            f"turn {recorded.turn_id} was started in session {recorded.session_id}, which no "
            "longer exists; it cannot be observed and must not be started again.",
            code="activity_turn_unobservable",
            details={
                "turn_id": recorded.turn_id,
                "turn_key": turn_key,
                "session_id": recorded.session_id,
                "current_session_id": session.session_id,
            },
        )
    _note_resume(ledger, task_id, turn_key=turn_key, turn_id=recorded.turn_id, source="turn.started")
    runtime.emit(
        {
            "event": RESUMED_EVENT,
            "run_id": state.run_id,
            "task_id": task_id,
            "phase": phase,
            "turn_id": recorded.turn_id,
            "turn_key": turn_key,
            "session_id": recorded.session_id,
            "source": "turn.started",
            "steer": list(notes),
        }
    )
    # The re-attach floor is the LEDGER's high-water mark, not the state's: a retry from history
    # carries the pre-turn state, whose `last_event_index` predates everything this turn emitted.
    return _stream(
        runtime,
        client,
        ledger,
        task_id,
        state,
        session_id=recorded.session_id,
        turn_id=recorded.turn_id,
        phase=phase,
        start_from=max(recorded.high_water, 0),
        resumed=True,
    )


def _stream(
    runtime: ActivityRuntime,
    client: Any,
    ledger: Any,
    task_id: str,
    state: TaskState,
    *,
    session_id: str,
    turn_id: str,
    phase: str,
    start_from: int,
    resumed: bool,
) -> TurnOutcome:
    """Consume the turn's stream to its terminal event, ledgering and heartbeating as it arrives."""
    last_index = int(start_from)
    spend = _Spend()
    terminal: str | None = None
    finish_reason: str = "completed"
    ask: Mapping[str, Any] | None = None
    last_beat = runtime.clock()
    for event in client.stream_turn(
        session_id,
        turn_id,
        after=last_index,
        should_reconnect=lambda _index: not runtime.is_cancelled(),
    ):
        if runtime.is_cancelled():
            _abort(client, session_id)
            raise ActivityError(
                f"turn {turn_id} was cancelled; the sidecar was told to abort it.",
                code="activity_cancelled",
                details={"session_id": session_id, "turn_id": turn_id, "event_index": last_index},
            )
        if event.event_index is not None:
            last_index = max(last_index, event.event_index)
            ledger.append_events(
                [
                    {
                        "event_index": event.event_index,
                        "event_type": event.event_type,
                        "payload": event.payload,
                    }
                ],
                task_id=task_id,
                session_id=session_id,
            )
        if event.is_terminal:
            terminal = event.event_type
            spend = _spend_from_event(event.payload, state=state)
            # D4: did this turn stop because a HUMAN owes it an answer? The engine says so on its
            # terminal event (`finish_reason: "ask"` + the ask itself). Read here, at the one place a
            # turn ends, so no caller has to infer it from the absence of work.
            raw_reason = event.payload.get("finish_reason")
            if isinstance(raw_reason, str) and raw_reason:
                finish_reason = raw_reason
            raw_ask = event.payload.get("ask")
            if isinstance(raw_ask, Mapping):
                ask = dict(raw_ask)
        now = runtime.clock()
        if now - last_beat >= HEARTBEAT_INTERVAL_S:
            last_beat = now
            runtime.beat(
                {
                    "task_id": task_id,
                    "phase": phase,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "event_index": last_index,
                    "cost_usd": spend.cost_usd,
                }
            )
    if terminal is None:
        # `stream_turn` reconnects on a drop, so reaching here means it either gave up
        # (`sidecar_stream_lost`) or cancellation stopped the retries. Both are "we do not know how
        # the turn ended", which is not success.
        raise ActivityError(
            f"turn {turn_id} ended without a terminal event.",
            code="activity_turn_unterminated",
            details={"session_id": session_id, "turn_id": turn_id, "event_index": last_index},
        )
    if terminal != "session.turn_completed":
        raise ActivityError(
            f"turn {turn_id} did not complete: {terminal}",
            code="activity_turn_failed",
            details={"session_id": session_id, "turn_id": turn_id, "terminal": terminal},
        )
    ledger.append(
        "turn.finished",
        task_id=task_id,
        session_id=session_id,
        event_index=last_index,
        payload={
            "turn_id": turn_id,
            "terminal": terminal,
            "cost_usd": spend.cost_usd,
            "tokens": spend.tokens,
            "priced": spend.priced,
            "phase": phase,
            # Recorded so a resumed attempt and the run's audit trail both know the turn ended on a
            # question rather than on completion — including the ask id, which is what a human's answer
            # would have been posted against.
            "finish_reason": finish_reason,
            **({"ask": dict(ask)} if ask is not None else {}),
        },
        allow_duplicate_event=True,
    )
    return TurnOutcome(
        session_id=session_id,
        turn_id=turn_id,
        event_index=last_index,
        cost_usd=spend.cost_usd,
        tokens=spend.tokens,
        priced=spend.priced,
        resumed=resumed,
        finish_reason=finish_reason,
        ask=ask,
    )


def _start_turn(
    runtime: ActivityRuntime,
    client: Any,
    ledger: Any,
    task_id: str,
    context: Any,
    state: TaskState,
    *,
    prompt: str,
    read_only: bool,
    phase: str,
    notes: Sequence[str],
    turn_key: str,
    working_tree: str,
) -> TurnOutcome:
    """No recorded turn: create/resume the session, POST once, and record the start immediately."""
    session_id = _ensure_session(
        runtime, client, context, state, read_only=read_only, working_tree=working_tree
    )
    # The TURN carries its own scope, because the plan and implement turns share ONE session and the
    # session is created READ-ONLY (an unattended run must not be writable by default). So:
    #   * a read-only turn sends `[]` (write nothing);
    #   * a writable turn sends `None`, i.e. JSON `null`, which explicitly lifts the session's
    #     read-only scope. Sending nothing would INHERIT it and leave the implement turn unable to work.
    start = client.start_turn(
        session_id,
        prompt,
        turn_key=turn_key,
        allowed_write_paths=[] if read_only else None,
    )
    ledger.append(
        "turn.started",
        task_id=task_id,
        payload={
            "turn_id": start.turn_id,
            "turn_key": turn_key,
            "replayed": bool(start.replayed),
            # The session the turn ran in. `append` refuses a session_id without an event_index
            # (they travel together on `turn.event` records), so the id goes in the payload where it
            # is still searchable — and where a soak can tell "the same key replayed" from "the
            # session was lost and the key got a fresh index" (see `soak_report`).
            "session_id": session_id,
            "phase": phase,
            "read_only": read_only,
            "steer": list(notes),
        },
    )
    return _stream(
        runtime,
        client,
        ledger,
        task_id,
        state,
        session_id=session_id,
        turn_id=start.turn_id,
        phase=phase,
        start_from=int(state.last_event_index),
        resumed=bool(start.replayed),
    )


def drive_turn(
    runtime: ActivityRuntime,
    context: Any,
    state: TaskState,
    *,
    prompt: str,
    read_only: bool,
    phase: str,
    notes: Sequence[str] = (),
    working_tree: str | None = None,
) -> TurnOutcome:
    """Run one paid turn to completion, buying at most one turn per ``turn_key``.

    The sequence, and why each step sits where it does:

    1. **Look for a recorded turn.** The ledger is the only thing a retry can read (see the section
       header above), so the check happens before the session is touched.
    2. **Create or resume the session** from ``state.session_id``. A resume the engine could not
       honour still runs the turn — the run survives, continuity does not — and the loss is recorded
       instead of hidden.
    3. **POST with ``turn_key``** only when nothing was recorded. ``seq`` is the state's next
       ``turn_seq``, so a retry of THIS activity reuses the key.
    4. **Record ``turn.started`` as soon as the engine accepts**, before any event is read: the crash
       this protects against is the one that happens mid-turn.
    5. **Stream with ``?after=``**, reconnecting from the highest index actually delivered. A dropped
       socket is not a failed turn.
    6. **Append as events arrive**, not at the end: buffering the turn would lose exactly the part
       that matters.
    7. **Heartbeat every 10s** with ``{event_index, cost_usd}``.
    8. **Cancel → abort.** Closing the stream is an unsubscribe; a cancelled activity must tell the
       sidecar to stop, or the turn keeps spending after the user pressed Stop.

    The engine's own replay path still covers the narrow window this leaves open: a crash BETWEEN the
    POST and the ledger append leaves no record, so the retry POSTs and the engine replays (if it
    still holds the session) rather than charging twice.

    ``working_tree`` is the tree the ENGINE edits (D5): the task's own worktree when it has one, else
    the shared checkout. It defaults to ``repo_dir`` so every existing caller keeps its behaviour, and
    a caller that owns a worktree passes it explicitly rather than letting this function guess.
    """
    tree = working_tree if working_tree is not None else context.repo_dir
    ledger = runtime.ledger(state.run_id)
    turn_key = f"{state.run_id}:{state.task_id}:{state.turn_seq + 1}"
    # Built WITH the run's owner, because the session this turn drives was created under it. The
    # engine stamps a session's owner at CREATE time and refuses a later request whose
    # `x-session-owner` disagrees, so a client built without one sends "anonymous" and every turn on a
    # durable run's own session is rejected (403 session_owner_mismatch) — measured against the real
    # sidecar. The owner has one definition, `owner_for`, used here and by `_ensure_session`.
    client = runtime.sidecar_factory(owner_for(context))
    try:
        recorded, finished = _scan_ledger(ledger, turn_key)
        if recorded is not None and finished is not None:
            return _resume_finished(
                runtime, ledger, state.task_id, state, recorded, finished, phase
            )
        if recorded is not None:
            return _resume_in_flight(
                runtime,
                client,
                ledger,
                state.task_id,
                context,
                state,
                recorded,
                turn_key,
                phase,
                notes,
                read_only=read_only,
                working_tree=tree,
            )
        return _start_turn(
            runtime,
            client,
            ledger,
            state.task_id,
            context,
            state,
            prompt=prompt,
            read_only=read_only,
            phase=phase,
            notes=notes,
            turn_key=turn_key,
            working_tree=tree,
        )
    finally:
        _close(client)


# `replace` is re-exported because every step builds a modified `TaskState` and importing it from
# here keeps the steps' import lists to one module.
__all__ = [*__all__, "replace"]
