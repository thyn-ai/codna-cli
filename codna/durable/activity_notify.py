"""The `notify` step: one ledger row per thing a run reports.

Split out of `activity_steps` at the repository's 1000-line ceiling, on a seam that is a real STEP
rather than a slice taken for size — the same reason `publish_steps` lives on its own. Everything here
writes ONE record and returns its sequence number; nothing here reads the ledger, decides anything, or
calls another activity.

**Why the shape of this row matters beyond this file.** `GET /runs` decides whether a run is over, and
how it ended, by reading the run's terminal record — the one this step appends. `run_verdict` owns that
read. The two are separate modules only because the READ is a leaf the HTTP surface loads at startup
while this WRITE needs the activity contracts; both state the same payload keys, and `run_contracts`
declares those keys once so the pair cannot drift.

**The outcome is a top-level field, lifted out of `evidence`.** `evidence` exists for records an
auditor must be able to read back (a goal verdict's checks). A run's OUTCOME is not evidence in that
sense — it is the verdict about the record itself — and the reader looks for it at the payload's top
level, where a control record's `action` already lives. So the writer pops it out rather than leaving a
second address for the same fact.
"""
from __future__ import annotations

from typing import Any, Final

from .activity_support import ActivityRuntime
from .contracts import NotifyOutcome, NotifyRequest
from .run_contracts import (
    NOTIFICATION_KIND_RUN_FINISHED,
    NOTIFICATION_KIND_TASK_FINISHED,
    OUTCOME_PAYLOAD_KEY,
)

__all__ = ["notify"]

# Which notification kinds are worth a log line (a run-level event, never a per-token one).
_CONSOLE_KINDS: Final[frozenset[str]] = frozenset(
    {"run.started", "task.started", "task.finished", "approval.granted", "auto_approved", "run.finished"}
)

# Which notification kinds carry a VERDICT about themselves. Only these lift an `outcome` out of
# `evidence` onto the ledger row's top level (see `notify`); any other kind's evidence is left intact.
#
# `task.unfinished` is deliberately ABSENT. It is an ending, but its verdict is its KIND: the writer
# emits it only when a task ended without its work, and it carries no `outcome` (the task's own
# `task.finished`, written beside it, is where the value lives). Listing it here would describe a
# field nothing writes, and would pop a caller's `evidence["outcome"]` — silently discarding it rather
# than leaving it where it was put. The reader decides this kind by `kind` (see `run_verdict`), which
# is the whole answer.
_OUTCOME_KINDS: Final[frozenset[str]] = frozenset(
    {
        NOTIFICATION_KIND_RUN_FINISHED,
        NOTIFICATION_KIND_TASK_FINISHED,
    }
)


def notify(request: NotifyRequest, *, runtime: ActivityRuntime) -> NotifyOutcome:
    """Append a structured notification to the run's ledger and log it once.

    A notification IS a ledger record: the Run API renders the ledger, so there is no second
    channel and nothing to fan out. That is what makes `notify` unable to fail because nobody is
    listening — the property a webhook would not have.

    ``runtime`` is passed rather than resolved from `activities`, so this module imports nothing
    that imports it back: `activity_steps` re-exports this function and supplies the process-wide
    runtime, which is the shape every other extracted step follows.
    """
    active = runtime
    request.validate()
    evidence = dict(request.evidence)
    # THE OUTCOME IS LIFTED OUT OF THE EVIDENCE, not left nested in it. The reader looks for
    # `payload["outcome"]` on a record that ends something, and it has always been the payload top
    # level that carries one (a control record's `action`, a `goal.verdict`'s checks); a value buried
    # in `evidence` would be a second address for the same fact. Popped rather than copied, so the row
    # does not say it twice. Only for a record that IS an ending — an approval's evidence mentioning an
    # outcome describes a task it observed, so it stays where it came from.
    outcome = evidence.pop(OUTCOME_PAYLOAD_KEY, None) if request.kind in _OUTCOME_KINDS else None
    payload: dict[str, Any] = {
        "event": "notification",
        "kind": request.kind,
        "message": request.message,
        # A record-shaped notification carries its evidence in the same ledger row, so an auditor
        # reading `goal.verdict` sees the checks it was taken over. Always present (empty for a
        # message-only notification), so a reader needs no branch.
        "evidence": evidence,
    }
    if outcome is not None:
        # A single string, never a structure: the row is read by a tail scan that must not be able to
        # disagree with itself about how the run ended.
        payload[OUTCOME_PAYLOAD_KEY] = str(outcome)
    if request.actor:
        # Present only when a person caused this row, so an approval's ledger record names WHO
        # without a second record and without a branch a reader has to know about.
        payload["identity"] = request.actor
    if request.repo_dir:
        # THE CHECKOUT THIS RUN WORKS IN, recorded once by the run's opening notification. `GET /runs`
        # reads it back so a sidebar row names the repository a run belongs to rather than the folder
        # the reading window happens to have open. Absent for every other notification, so a
        # message-only row keeps the shape it had — the ABSENCE of the key is the honest answer for a
        # row that is not about a checkout.
        payload["repo_dir"] = request.repo_dir
    record = active.ledger(request.run_id).append(
        "note",
        task_id=request.task_id,
        payload=payload,
    )
    if request.kind in _CONSOLE_KINDS:
        active.emit(
            {
                "event": request.kind,
                "run_id": request.run_id,
                "task_id": request.task_id,
                "message": request.message,
            }
        )
    return NotifyOutcome(seq=record.seq if record is not None else None, kind=request.kind)
