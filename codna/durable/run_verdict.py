"""How a run ENDED, read from its ledger — the verdict `GET /runs` and `GET /runs/:id` report.

**Why this is its own module.** Two facts the routes need — "is this run over?" and "how did it end?" —
are both read from the TAIL of an append-only ledger, and neither is about HTTP. `run_api_support` holds
the transport's primitives and had reached the repository's 1000-line ceiling; the same split that put
`run_repo` (the HEAD reader) in its own file puts the TAIL reader here, so a reviewer asking "can a
finished run ever be reported as open, or a success as a failure?" reads one file with nothing else in it.

**The one rule that took two attempts to get right.** A run's terminal is `run.finished`, and the
outcome is what that record says. What it said used to be nothing at all: the notification activity
carried a message and no verdict, so every completed run read `outcome: null` on the LIST while its
DETAIL row — which reads the workflow's result, not the ledger — read `completed`. The writer now
records the outcome (see `run_contracts.run_terminal_evidence`), and the reader below prefers it.

For the ledgers already on disk, written before that, this falls back to the newest STRUCTURED verdict
in the same bounded window — `episode.remembered`, `task.finished`, `task.unfinished` — matched to the
terminal's own task, and reported only when that match is unambiguous. It never parses prose. "The
workflow ended, so it must have completed" is exactly the inference this module refuses: the endings
that used to write no terminal were the ones whose outcome is not success, so an ending with no
evidence is reported as `None` rather than as success.

**Every read is bounded.** The answer is in the final records of a file that can hold a hundred thousand
of them, so each entry point takes a 64 KiB / 32-record window rather than scanning a run's history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Final

from .ledger import GENESIS_HASH, LedgerError, LedgerRecord, compute_hash

__all__ = [
    "ACTIVITY_TURN_TIMEOUT_S",
    "INERT_EVENT_TYPES",
    "LEDGER_OUTCOMES",
    "NOTIFICATION_KIND_TASK_FINISHED",
    "NOTIFICATION_KIND_TASK_UNFINISHED",
    "STALLED_AFTER_S",
    "TERMINAL_EVENT",
    "TERMINAL_EVENT_TYPES",
    "TURN_ABORTED_EVENT_TYPES",
    "TURN_COMPLETED_EVENT",
    "TURN_FAILED_EVENT_TYPES",
    "TURN_TERMINAL_EVENT_TYPES",
    "TURN_VERDICT_EVENT_TYPES",
    "event_frame_payload",
    "event_type_of",
    "last_seq_of",
    "ledger_problems",
    "ledger_verdict",
    "outcome_of",
    "outcome_of_turn_terminal",
    "run_outcome_of_task_outcome",
    "terminal_verdict",
]

# The event that ends a run's stream, and the run's own final notification. Declared here rather than
# imported, so this module stays a leaf the transport can load without reaching the workflow layer.
TERMINAL_EVENT: Final = "run.finished"
TERMINAL_EVENT_TYPES: Final[frozenset[str]] = frozenset({TERMINAL_EVENT})

# -- turn terminals: not RUN terminals, but evidence that the run ended -------------------------
#
# The distinction the run list used to miss. `run.finished` is written only on the two `completed`
# exits of `RunWorkflow` (all tasks done, and goal met). Every other way a run stops — a task whose
# turn failed, a cancellation, a dead worker — leaves the ledger ending at a TURN terminal and
# nothing above it. Reading only `run.finished` therefore reported such a run as "not finished", and
# the sidebar animated it as Working indefinitely.
#
# The names and the membership are `sidecar_client.SSE_TERMINAL_TURN_EVENTS`, transcribed here (the
# same way the SSE constants above are) because a turn terminal is the SIDECAR's vocabulary, and
# `tests/durable/test_run_api.py` asserts the two sets are equal so they cannot drift.
TURN_COMPLETED_EVENT: Final = "session.turn_completed"
# The engine reported the turn as failed: a failure of the work, not of the harness.
TURN_FAILED_EVENT_TYPES: Final[frozenset[str]] = frozenset({"session.turn_failed"})
# The turn was stopped rather than lost. An INTERRUPTED turn is not a failure — the soak harness
# reports `turn_unpaired` as a warning for exactly this reason — so these resolve to `stopped`.
TURN_ABORTED_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"session.turn_aborted", "session.aborted", "session.stopped"}
)
# The turn terminals that are EVIDENCE about how the RUN ended.
#
# `session.turn_completed` is deliberately NOT here, and the omission is the point. A turn that
# completed is a task that did its job: it is work, not a verdict, and a run whose next task is still
# coming must not read as over because the previous turn ended cleanly. That is exactly why it is also
# absent from `INERT_EVENT_TYPES` — it is neither a verdict nor noise, it is progress.
#
# Listed in `TURN_TERMINAL_EVENT_TYPES` below only so `event_type_of`'s vocabulary is complete; nothing
# in the verdict path may treat it as an ending.
TURN_TERMINAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {TURN_COMPLETED_EVENT} | TURN_FAILED_EVENT_TYPES | TURN_ABORTED_EVENT_TYPES
)
# The subset a verdict may be DERIVED from: everything that says the turn did not finish its work.
TURN_VERDICT_EVENT_TYPES: Final[frozenset[str]] = TURN_FAILED_EVENT_TYPES | TURN_ABORTED_EVENT_TYPES

# -- a run TERMINAL that does not say how the run ended ------------------------------------------
#
# The outcome vocabulary is `run_contracts`' (`LedgerRunOutcome`), spelled here as literals because
# this module is a LEAF that `run_api_http` imports at load time — importing `run_contracts` would be
# harmless today and is exactly how a leaf stops being one. `tests/durable/test_run_api.py` asserts
# this set EQUALS `run_contracts.LEDGER_RUN_OUTCOMES`, so the copy cannot drift.
OUTCOME_COMPLETED: Final = "completed"
OUTCOME_FAILED: Final = "failed"
OUTCOME_STOPPED: Final = "stopped"
OUTCOME_BUDGET_EXHAUSTED: Final = "budget_exhausted"
LEDGER_OUTCOMES: Final[frozenset[str]] = frozenset(
    {OUTCOME_COMPLETED, OUTCOME_FAILED, OUTCOME_STOPPED, OUTCOME_BUDGET_EXHAUSTED, "goal_unmet"}
)

# `task.finished` names itself in `kind`, next to every other notification.
NOTIFICATION_KIND_TASK_FINISHED: Final = "task.finished"
# The record `remember_episode` appends with the task's outcome. NOT a notification — it is the
# structural verdict the ledgers on disk actually carry, so it is a first-class candidate here.
VERDICT_EVENT_EPISODE: Final = "episode.remembered"
# A task that ended WITHOUT its work, written beside `task.finished` so the two can be told apart
# without reading a message. It names itself in `kind`, like `task.finished` — every notification's
# `event` is the literal ``"notification"`` and the KIND is the specific fact — so a reader that
# compares `event` to this value is comparing the wrong field and can never match a real record.
NOTIFICATION_KIND_TASK_UNFINISHED: Final = "task.unfinished"
# The task-level "finished with nothing to publish" value — success with no pull request. Spelled here
# for the same leaf reason as the outcomes above.
TASK_OUTCOME_APPROVED_UNPUBLISHED: Final = "approved_unpublished"

# Records that are INERT when looking for a verdict: they neither record work nor give one.
#
#   * `turn.resumed_from_ledger` / `session.resumed` narrate a RETRY. An aborted turn is followed by
#     a resume, and the resume is not evidence the run is alive again — the work after it is.
#   * `control.requested` is the audit record of a human's signal. A pause on a finished run is
#     allowed and harmless and appends a note PAST the terminal; "the last record is a control note"
#     must not be read as "this run never finished".
#
# Anything NOT in this set is work, and work after a turn terminal means it was superseded.
INERT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"turn.resumed_from_ledger", "session.resumed", "control.requested"}
)

# -- how long a quiet ledger is read as "stalled" rather than "working" -------------------------
#
# Twice `contracts.ACTIVITY_TURN_TIMEOUT_S` (60 min), kept as a literal here because this module is
# a LEAF that `run_api_http` imports at load time: importing `contracts` would pull the provider,
# state and prompt layers into the HTTP surface for one integer. The value is pinned by
# `test_stalled_threshold_is_twice_the_longest_activity`, so it cannot drift from what it mirrors.
#
# WHY TWICE THE LONGEST ACTIVITY. `ACTIVITY_TURN_TIMEOUT_S` is Temporal's own outer ceiling for one
# activity — the longest a paid turn may run before it is abandoned. A live run that has written
# NOTHING for longer than two such ceilings has outlived the whole timeout-and-retry machinery: its
# activity was abandoned, its retries ran and ended, and no worker is producing this ledger. Waiting
# longer only keeps the lie up for longer. Nothing is lost by being this generous, because the
# verdict is re-derived on every read: a run that starts writing again leaves `stalled` on its own,
# with no state to clear.
ACTIVITY_TURN_TIMEOUT_S: Final = 60 * 60
STALLED_AFTER_S: Final = 2 * ACTIVITY_TURN_TIMEOUT_S


def event_type_of(record: LedgerRecord) -> str:
    """The ``event_type`` a frame carries. The IDE's parser drops a frame without a string one.

    Three shapes reach a ledger:

    * ``turn.event`` wraps the engine's own event, so its type is the engine's;
    * ``note`` is a notification or a control record. A notification names itself in ``kind``
      (``run.started``, ``run.finished``); a control record names itself in ``event``
      (``control.requested``);
    * everything else is named by its ledger kind (``task.finished``, ``approval.granted``, ...).
    """
    payload = record.payload
    if record.kind == "turn.event":
        raw = payload.get("event_type")
        return raw if isinstance(raw, str) and raw else "turn.event"
    if record.kind == "note":
        if payload.get("event") in (None, "", "notification"):
            kind = payload.get("kind")
            if isinstance(kind, str) and kind:
                return kind
        raw = payload.get("event")
        if isinstance(raw, str) and raw:
            return raw
        return "note"
    return record.kind


def event_frame_payload(record: LedgerRecord) -> dict[str, Any]:
    """The JSON object inside one ``data:`` line.

    ``event_index`` is the ledger's ``seq`` — the number ``?after=`` resumes from — so a client that
    persists the last index it saw reattaches exactly where it left off, with no second cursor to
    keep in sync. ``hash``/``prev_hash`` ride along so a client can verify a record it cached.
    """
    return {
        "event_index": record.seq,
        "event_type": event_type_of(record),
        "run_id": record.run_id,
        "task_id": record.task_id,
        "session_id": record.session_id,
        "kind": record.kind,
        "ts": record.ts,
        "payload": record.payload,
        "prev_hash": record.prev_hash,
        "hash": record.hash,
    }


def outcome_of(path: Path) -> str | None:
    """The run's outcome, read from the tail of the ledger. ``None`` when the run has not ended.

    A bounded read: the answer is in the final records, so scanning a year of ledger to find it
    would be work with no answer at the end of it. ``None`` is the honest answer for a truncated
    tail as well — a record that did not commit is not an outcome.

    TWO SHAPES OF "FINISHED", because the workflow writes two. ``run.finished`` is the run's own
    terminal and carries the outcome the workflow decided. A run whose task never got that far ends
    on a TURN terminal and writes nothing above it (see ``TURN_TERMINAL_EVENT_TYPES``). Both are
    read here, so a run that plainly ended stops being reported as open.

    A turn terminal has no ``outcome`` field, so one is DERIVED from WHICH terminal it was, in the
    vocabulary run-level outcomes already use (``RunOutcome`` in ``run_contracts``, plus the
    ``failed`` the workflow already finalises with). The distinction that matters is kept: a turn
    the engine FAILED is ``failed``; a turn that was stopped or interrupted is ``stopped``. An
    interrupted turn is not a failure — the same rule the soak harness applies by reporting
    ``turn_unpaired`` as a warning rather than a violation.
    """
    ended, outcome = terminal_verdict(path)
    return outcome if ended else None


def outcome_of_turn_terminal(event_type: str) -> str:
    """The run outcome implied by a turn terminal. Total over ``TURN_VERDICT_EVENT_TYPES``.

    ``failed`` for a turn the engine reported as failed; ``stopped`` for one that was aborted or
    interrupted. Both are non-``completed``, so both reach the sidebar as a run the user has to look
    at rather than as one still in progress.

    ``session.turn_completed`` is REFUSED rather than mapped: a completed turn is success, and folding
    it into "stopped" would make a turn that finished its job indistinguishable from one that was
    killed. A caller holding a completed turn has no verdict to derive, and this raises instead of
    inventing one.
    """
    if event_type in TURN_FAILED_EVENT_TYPES:
        return "failed"
    if event_type in TURN_ABORTED_EVENT_TYPES:
        return "stopped"
    raise ValueError(
        f"{event_type!r} is not a turn verdict: a turn that completed is progress, not an ending"
    )


def _terminal_verdict(records: list[LedgerRecord]) -> tuple[bool, str | None]:
    """``(ended, outcome)`` from a tail, newest first. The ONE place the terminal rules are applied.

    **A ``run.finished`` is final.** It is the run's own completion and nothing is scheduled after
    it, so the whole window is searched for one FIRST. That is what keeps the property the
    control-note case depends on: a pause recorded AFTER the run finished (allowed and harmless)
    cannot un-finish it, and neither can a stray record.

    **A turn terminal is superseded by work that followed it.** An aborted turn is not always an
    ending: a worker restarts the turn under the same ``turn_key``, and the ledger then reads
    ``session.aborted``, ``turn.resumed_from_ledger``, ``session.resumed``, ... The abort records
    what happened; it is not a verdict that outlives the resume. So when the run wrote no terminal
    of its own, this returns the NEWEST turn verdict, looking past the INERT records above: they
    neither confirm nor revive one. Any other record is work, and work supersedes a turn terminal.

    Measured against the 97 ledgers of a 24h soak: 52 end at a ``review`` record and 23 at
    ``session.aborted``. Reading "the newest terminal anywhere" would call 75 of them finished;
    reading "the newest verdict, superseded by work" calls finished exactly the ones that stopped.

    **A terminal that does not SAY how the run ended is asked of the run's own verdicts.** The
    paragraph above is why the answer must not be "the workflow closed, so completed": the ends that
    used to write no terminal were exactly the ends whose outcome is not success. When the terminal
    carries no ``outcome``, the newest STRUCTURED verdict in the same bounded window decides, and only
    a record that IS a verdict about the task the run finished on:

    * ``episode.remembered`` — the task's outcome, recorded when its episode was written;
    * ``task.finished`` with an outcome — the same fact, from the task's own notification;
    * ``task.unfinished`` — a task that ended WITHOUT its work.

    A run-level terminal names no task, and is answered ONLY when it is unambiguous: a single
    candidate, or several that all name one task, or the terminal names that task. Anything else is
    reported as ``None``. A task-scoped terminal answers only from its own task, so a goal run's
    "fix CI" attempt cannot speak for the parent. A run with nothing structured to say reports
    ``None``: "it ended, and the ledger does not say how" is a fact, not a gap to fill with a guess
    from the message prose.
    """
    found = False
    terminal_task_id: str | None = None
    for record in records:
        if event_type_of(record) != TERMINAL_EVENT:
            continue
        raw = record.payload.get("outcome")
        if isinstance(raw, str) and raw:
            return True, raw
        # The ending is a fact whether or not the cause was written down — the shape every ledger
        # written before the terminal carried an outcome has. `task_id` names the task the exit came
        # from, and is what lets the fallback below pick THAT task's verdict out of the window.
        found = True
        terminal_task_id = record.task_id
        break
    if found:
        return True, _structured_verdict(records, task_id=terminal_task_id)
    for record in records:
        event_type = event_type_of(record)
        if event_type in INERT_EVENT_TYPES:
            continue
        if event_type in TURN_VERDICT_EVENT_TYPES:
            return True, outcome_of_turn_terminal(event_type)
        # Any other record is work — a completed turn, a task starting, a tool call — that happened
        # AFTER an older turn verdict, so it supersedes it.
        return False, None
    return False, None


def _structured_verdict(records: list[LedgerRecord], *, task_id: str | None) -> str | None:
    """The newest run-level outcome the records STATE, for ``task_id``. Never parsed from prose.

    Our window is 32 records, and a real ledger's ending is short: on the 85 ledgers on this host a
    run's verdict sits at most 4 records from the end (``episode.remembered``, then the task's and the
    run's own notifications). So one pass is enough and no second, larger read is needed.

    Ordering is by the records as given (newest first) and the FIRST match wins, so a record that
    contradicts an older one supersedes it — the same rule the turn-terminal path applies.

    ``task_id`` None means the terminal was RUN-level. It may still be answered by a task's verdict,
    but only when that verdict is UNAMBIGUOUS: a single candidate, or several that all NAME one task.
    A multi-task run whose terminal does not name the task it ended on cannot be attributed — the
    newest verdict might belong to a task that finished EARLIER while the run ended on another, and
    "task A's approval is the run's outcome" is the misattribution this refuses. With two or more
    candidates that do not agree on a task, a run-level terminal reports ``None``: the ledger does
    not say which task the run ended on, so it does not say how the run ended either.

    **An unnamed candidate cannot vouch for another.** `task_id` is nullable and a record can predate
    the field, so verdict records that do NOT name their task exist. Two of them cannot be PROVEN to
    be the same task, and keying them on a shared placeholder (``task_id or ""``) let exactly those
    merge into one bucket — the ambiguity guard defeated by the input it exists for. `_naming_task`
    keeps `None`/``""`` as "names nothing", so several unnamed candidates answer ``None``.
    """
    candidates: list[tuple[str | None, str]] = []
    for record in records:
        if record.kind != "note":
            continue
        if task_id is not None and record.task_id != task_id:
            continue
        outcome = _verdict_of_note(record)
        if outcome is None:
            # A note that is not a verdict about a task (a park reason, a publish skip, a goal
            # verdict) neither answers here nor stops the search: a run's ending can hold several.
            continue
        candidates.append((_naming_task(record.task_id), outcome))
    if task_id is not None:
        # A task-scoped terminal: the candidates are that task's, newest first.
        return candidates[0][1] if candidates else None
    if not candidates:
        return None
    if len(candidates) == 1:
        # Exactly one verdict in the window: there is no second task it could be mistaken for, so
        # even an unnamed one speaks for the run.
        return candidates[0][1]
    # A run-level terminal with SEVERAL candidates. Unambiguous only when every candidate NAMES the
    # same task. An UNNAMED candidate (``None`` or ``""``) names nothing, so two of them cannot be
    # proven to speak for one task and must not share a bucket: a reader that keyed on
    # ``task_id or ""`` merged exactly those, and answered with one of two different tasks' verdicts.
    identity = candidates[0][0]
    if identity is None:
        return None
    if any(candidate_task != identity for candidate_task, _ in candidates):
        return None
    return candidates[0][1]


def _naming_task(task_id: str | None) -> str | None:
    """The task a record NAMES, or ``None`` when it names none.

    ``task_id`` is nullable on `LedgerRecord` and a record can predate the field, so ``None`` and the
    empty string are the SAME fact: this record does not say which task it belongs to. Collapsing
    them here makes two unnamed records two unknown identities rather than one shared one, which is
    what keeps them from being attributed to each other.
    """
    return task_id if task_id else None


def _verdict_of_note(record: LedgerRecord) -> str | None:
    """The outcome ONE note states about the task it belongs to, or ``None`` when it states none.

    The one place a record's verdict is read, so the ambiguity rule above sees the same answer the
    returned value comes from. Only two shapes state a task's ending:

    * ``episode.remembered`` — the task's outcome, in ``outcome``;
    * a ``task.finished`` notification carrying an ``outcome``;
    * a ``task.unfinished`` notification — the task ended WITHOUT its work.

    The kind is what tells the last two apart: a notification's ``event`` is always the literal
    ``"notification"`` and its KIND is the specific fact, so this compares ``kind``.
    """
    payload = record.payload
    if payload.get("event") == "notification":
        kind = payload.get("kind")
        if kind == NOTIFICATION_KIND_TASK_UNFINISHED:
            # Decided by its kind, BEFORE any outcome is looked for: a task that ended without its
            # work has no success to report. Checked first so it can never be skipped for lacking a
            # field the record does not carry.
            return OUTCOME_FAILED
        if kind != NOTIFICATION_KIND_TASK_FINISHED:
            return None
        raw = payload.get("outcome")
        if not isinstance(raw, str) or not raw:
            return None
        return run_outcome_of_task_outcome(raw)
    if payload.get("event") == VERDICT_EVENT_EPISODE:
        raw = payload.get("outcome")
        if not isinstance(raw, str) or not raw:
            return None
        return run_outcome_of_task_outcome(raw)
    return None


def run_outcome_of_task_outcome(outcome: str) -> str:
    """What a TASK's final outcome implies about the RUN it ran under.

    Total over `TaskOutcome`, and it never returns ``completed`` for a task that did not finish its
    work — the distinction the run list exists to make:

    * ``approved`` / ``approved_unpublished`` — the work was done and passed review. Either is the
      ``completed`` the workflow itself returns; whether a pull request was opened is a property of
      the task, and the run-level story already says it (`RunCompletion.unpublished_tasks`).
    * ``rejected`` and anything else — the run ended on work nobody approved, which is the ``failed``
      a workflow that raised also reports. Not ``completed``: that would claim the run did its job.
    * ``stopped`` / ``budget_exhausted`` — the ending has its own name, so each keeps it.
    """
    if outcome in ("approved", TASK_OUTCOME_APPROVED_UNPUBLISHED):
        return OUTCOME_COMPLETED
    if outcome == "stopped":
        return OUTCOME_STOPPED
    if outcome == "budget_exhausted":
        return OUTCOME_BUDGET_EXHAUSTED
    return OUTCOME_FAILED


def terminal_verdict(path: Path) -> tuple[bool, str | None]:
    """``(ended, outcome)`` for one ledger, from a SINGLE bounded tail read.

    The one entry point a route should use when it reports both facts, because two separate reads can
    observe two different file states: a record appended between them makes ``terminal: False`` sit
    beside ``outcome: "completed"``, and a response that contradicts itself is worse than either
    answer alone. ``outcome_of`` remains for a caller that wants only the cause.
    """
    return _terminal_verdict(_tail_records(path, limit=_OUTCOME_SCAN_RECORDS))


def ledger_verdict(
    path: Path, *, updated_ts: float, now: float | None = None
) -> dict[str, object]:
    """Everything a LIST read needs about one ledger, from a single bounded tail pass.

    One pass rather than one per question: the answers would otherwise be able to disagree if a
    record were appended between them, and a row that said "stalled" while carrying an outcome would
    be worse than either answer alone.

    * ``ended`` — a terminal is present, so the run is over. This is what stops a spinner.
    * ``outcome`` — what the ledger recorded, or what its turn terminal implies. ``None`` when it
      ended without saying why, which is a fact rather than a gap to fill.
    * ``stalled`` — no terminal AND nothing appended for ``STALLED_AFTER_S``. The honest third
      state: not finished, and not being worked on either.

    ``updated_ts`` is the ledger's mtime — the same value the list reports as the row's age, so the
    row's "3d" and this verdict cannot disagree about how long it has been quiet.
    """
    ended, outcome = _terminal_verdict(_tail_records(path, limit=_OUTCOME_SCAN_RECORDS))
    current = time.time() if now is None else now
    quiet_for_s = max(0.0, current - updated_ts)
    return {
        "ended": ended,
        "outcome": outcome,
        "stalled": (not ended) and quiet_for_s >= STALLED_AFTER_S,
        "quiet_for_s": round(quiet_for_s, 3),
    }


def last_seq_of(path: Path) -> int:
    """The highest ``seq`` in the ledger, read from the tail. Zero when the file is unreadable.

    The tail read can land on a partial final line (a writer mid-append), so this takes the highest
    ``seq`` across the window rather than assuming the last complete line is the last record.
    """
    records = _tail_records(path, limit=_OUTCOME_SCAN_RECORDS)
    return max((record.seq for record in records), default=0)


_TAIL_WINDOW_BYTES: Final = 64 * 1024
# How far back the outcome scan looks. A terminal is followed by at most a handful of control notes,
# so a small bound finds it without turning a summary into a full scan.
_OUTCOME_SCAN_RECORDS: Final = 32

def _tail_records(path: Path, *, limit: int) -> list[LedgerRecord]:
    """Up to ``limit`` complete records from the end of the file, reading only the last 64 KiB."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - _TAIL_WINDOW_BYTES))
            tail = handle.read()
    except OSError:
        return []
    if not tail:
        return []
    # A window that starts mid-record leaves a fragment at the front; it is dropped rather than
    # parsed, because its beginning is outside the window.
    chunks = tail.split(b"\n")
    start = 1 if (size > _TAIL_WINDOW_BYTES and not tail.startswith(b"\n")) else 0
    found: list[LedgerRecord] = []
    for raw in reversed(chunks[start:]):
        text = raw.strip()
        if not text:
            continue
        try:
            decoded = json.loads(text.decode("utf-8"))
            if not isinstance(decoded, dict):
                continue
            found.append(LedgerRecord.from_payload(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError, LedgerError):
            continue
        if len(found) >= limit:
            break
    return found


def ledger_problems(records: list[LedgerRecord]) -> list[str]:
    """Hash-chain and contiguity problems across the records a caller has read."""
    problems: list[str] = []
    expected_prev = GENESIS_HASH
    expected_seq = 1
    for record in records:
        if record.seq != expected_seq:
            problems.append(f"seq {record.seq} at position {expected_seq}")
        if record.prev_hash != expected_prev:
            problems.append(f"seq {record.seq}: prev_hash mismatch")
        recomputed = compute_hash(record.to_payload())
        if recomputed != record.hash:
            problems.append(f"seq {record.seq}: hash mismatch")
        expected_prev = record.hash
        expected_seq = record.seq + 1
    return problems