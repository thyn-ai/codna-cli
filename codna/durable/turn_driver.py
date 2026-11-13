"""The paid-turn machinery: one billed turn per ``(run, task, turn_seq)``, never two.

Split out of `activity_support.py` when the turn-loop guard pushed that module over the repository's
1000-line ceiling (`test_modularity.py`). The boundary is a real one and not a size dodge:

* `activity_support` answers "what can an activity DEPEND on" — the injected runtime, the error type,
  the request decoder, git, and the checkout a step works in;
* this module answers "how is a turn BOUGHT", which is the money guarantee: the ledger is consulted
  before any POST, a recorded turn is re-attached rather than re-bought, and a lost session is a loud
  failure instead of a second bill.

`drive_turn` is re-exported by `activity_support`, which is the module every step already imports, so
the contract stays reachable from one place and no caller had to change.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final, Mapping, Sequence

from .activity_support import (
    HEARTBEAT_INTERVAL_S,
    ActivityError,
    ActivityRuntime,
    owner_for,
)
from .ledger import LedgerRecord
from .session_spec import _session_spec
from .state import TaskState
from .turn_guard import TOOL_LOOP_EVENT, TurnGuard

__all__ = [
    "RESUMED_EVENT",
    "TurnOutcome",
    "drive_turn",
]


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


@dataclass(frozen=True)
class _RecordedTurn:
    session_id: str
    turn_id: str
    high_water: int
    #: The tool-loop verdict an earlier attempt recorded for this turn, if any. Carried out of the
    #: SAME traversal that found the turn: a retry that re-attached to a turn the guard aborted would
    #: otherwise report `activity_turn_failed` and lose the reason (the bad call) this verdict exists
    #: to preserve. Read here rather than in a second `ledger.records()` call, which would break the
    #: one-pass guarantee the docstring below promises — and cost a full NDJSON re-parse per retry.
    loop_failure: Mapping[str, Any] | None = None


def _scan_ledger(ledger: Any, turn_key: str) -> tuple[_RecordedTurn | None, LedgerRecord | None]:
    """One pass over the ledger: the recorded turn for ``turn_key``, plus its completion if any.

    ONE pass on purpose. `RunLedger.records()` re-parses the whole NDJSON file and a run writes one
    line per streamed event (the measured runs had 3131+), so scanning once per fact would make a
    long run quadratic in its own ledger. Everything a retry needs — was this key bought, in which
    session, how far did its stream get, did it finish, and was it judged a tool loop — comes out of
    this single traversal. The last of those belongs here rather than in a second helper: a turn the
    guard stopped is aborted, so the retry must report THAT verdict instead of the abort it caused,
    and reading it separately paid for a second full re-parse on every activity retry.

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
    loop_failure: Mapping[str, Any] | None = None
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
        elif record.kind == "note":
            note = record.payload
            if note.get("event") == TOOL_LOOP_EVENT and note.get("turn_id") == recorded.turn_id:
                # The LAST verdict wins, matching the single-slot retry semantics: a turn gets one
                # verdict, and a later record describes a later judgement of the same turn.
                loop_failure = dict(note)
    if recorded is None:
        return None, None
    # Only when the terminal is the LAST thing this turn recorded: then it is the floor, `?after=`
    # would skip it, and replaying one event is both safe and necessary. A terminal followed by
    # later events means the stream was already reading past it, so the floor is left alone.
    if completion is None and terminal_index is not None and terminal_index == high_water:
        high_water = max(0, high_water - 1)
    return (
        _RecordedTurn(recorded.session_id, recorded.turn_id, high_water, loop_failure),
        completion,
    )


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
    # WHY the turn was aborted, when the engine said so on its terminal event: the `abort` block
    # (`{kind, reason}`) that distinguishes a fail-closed guard refusal (`control_stop`) from a
    # cancellation (`caller_abort`). Before the engine carried it, the reason a write-scope hook
    # supplied was computed and then dropped, so an aborted task reached this layer as a bare
    # terminal name with nothing to report to the user.
    abort: Mapping[str, Any] | None = None
    # ONE guard per turn, created here: the streak it counts is a property of THIS turn, and a
    # resumed attempt re-reads the events it already ledgered (`?after=`), so a fresh count is the
    # true one. See `turn_guard` for the measured defect and the threshold.
    guard = TurnGuard()
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
            # Read at the terminal, like `ask`: the reason exists only for the turn that was stopped,
            # so carrying it any other way would let one turn's refusal explain another's ending.
            raw_abort = event.payload.get("abort")
            abort = dict(raw_abort) if isinstance(raw_abort, Mapping) else None
        # A REPEATED IDENTICAL TOOL FAILURE STOPS THE TURN HERE. A turn that hands a tool the same
        # arguments and gets the same deterministic error back is not going to succeed on the next
        # iteration — waiting for the deadline only burns the budget and replaces the real cause with
        # "turn exceeded timeout budget". MEASURED: `run-20261112T232923Z` spent its whole 600 s on ten
        # identical `editor` failures (`old_text` required) and died on the deadline, so verify and
        # review never ran. See `turn_guard` for the numbers and for what is deliberately NOT counted.
        loop = guard.observe(event.event_type, event.payload)
        if loop is not None:
            # THE VERDICT IS LEDGERED BEFORE THE ABORT, and the ordering is the point. An activity
            # retry re-reads the ledger: with the abort first, a retry would re-attach to an ABORTED
            # turn and report `activity_turn_failed` — replacing "the editor call was wrong" with "the
            # turn stopped", which is exactly the substitution this fix exists to prevent. Recorded
            # first, the retry finds this record (see `_scan_ledger`'s `loop_failure`) and reports the
            # SAME code.
            # `session_id` and `event_index` travel together (the ledger's dedupe key is the pair), and
            # the index recorded is the LAST event this attempt saw — the honest position a retry would
            # resume from.
            #
            # `allow_duplicate_event=True` IS REQUIRED, and its absence silently dropped this record
            # (verified: the note never reached the ledger). A ``turn.event`` already holds that
            # `(session_id, event_index)` pair, and the dedupe key cannot tell "the same event again"
            # from "a different KIND of record about that event". This is exactly the case the flag
            # documents — a verdict ABOUT an event, not a re-delivery of it — and without the record
            # the retry reports the abort this verdict caused instead of the verdict.
            ledger.append(
                "note",
                task_id=task_id,
                session_id=session_id,
                event_index=last_index,
                allow_duplicate_event=True,
                payload={
                    "event": TOOL_LOOP_EVENT,
                    "turn_id": turn_id,
                    **loop.as_details(),
                },
            )
            # Abort the engine's run before raising: the sandbox is charged per token, and a turn we
            # have judged unsalvageable must not keep generating after we stop listening.
            try:
                client.abort(session_id)
            except Exception as exc:  # noqa: BLE001 - the loop failure is the outcome; this is context
                loop_abort_error = str(exc)
            else:
                loop_abort_error = None
            raise ActivityError(
                f"turn {turn_id} repeated the same {loop.tool_name} failure {loop.count} times "
                f"({loop.error}); stopping instead of spending the rest of the budget on it.",
                code="activity_turn_tool_loop",
                details={
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "event_index": last_index,
                    **loop.as_details(),
                    **({"abort_error": loop_abort_error} if loop_abort_error else {}),
                },
            )
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
        # The engine's abort reason belongs on the failure, not only in a log. `terminal` names the
        # event; `abort.kind` says whether the work was refused on purpose or cancelled, and
        # `abort.reason` quotes the refusal (e.g. a write-scope violation) so the run's failure is
        # legible without replaying the stream. `reason` is promoted alongside `abort` because that
        # is the field every existing ActivityError consumer already reads.
        abort_reason = abort.get("reason") if abort is not None else None
        raise ActivityError(
            f"turn {turn_id} did not complete: {terminal}",
            code="activity_turn_failed",
            details={
                "session_id": session_id,
                "turn_id": turn_id,
                "terminal": terminal,
                # The whole block is preserved (so a field added to `abort` later is not lost here)…
                **({"abort": dict(abort)} if abort is not None else {}),
                # …and `kind`/`reason` are promoted, because those are the two a caller branches on and
                # `reason` is the field every existing ActivityError consumer already reads.
                **({"kind": abort["kind"]} if abort is not None and isinstance(abort.get("kind"), str) else {}),
                **({"reason": abort_reason} if isinstance(abort_reason, str) and abort_reason else {}),
            },
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
            # WAS THIS TURN ALREADY JUDGED A LOOP? Then the retry must report THAT, not the abort the
            # judgement caused. `_scan_ledger` carries the verdict out of the same traversal that
            # found the turn; without it the same turn would come back as `activity_turn_failed`, and
            # the timeout-style substitution this fix removes would simply move one layer down.
            loop = recorded.loop_failure
            if loop is not None:
                raise ActivityError(
                    f"turn {recorded.turn_id} repeated the same {loop.get('tool')} failure "
                    f"{loop.get('count')} times ({loop.get('error')}); it was stopped and must not be "
                    "re-run, because it would make the same call again.",
                    code="activity_turn_tool_loop",
                    details={
                        "session_id": recorded.session_id,
                        "turn_id": recorded.turn_id,
                        "event_index": recorded.high_water,
                        **{key: loop[key] for key in ("tool", "error", "count") if key in loop},
                    },
                )
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


# `replace` stays importable from here because it was importable from `activity_support` before this
# module was split out of it, and `dataclasses.replace` is what every step uses to build a modified
# `TaskState`. It is imported at module scope above, so this only keeps the NAME reachable — removing
# it would turn a working import into an ImportError for a caller this change did not edit.
__all__ = [*__all__, "replace"]
