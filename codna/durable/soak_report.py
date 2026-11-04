"""The soak invariant checker — PURE functions over ledger records and a seeded kill schedule.

WHY this module exists
======================
A durability soak is only worth running if "did the invariants hold?" is answerable without the
server, without the clock, and without re-reading a log by eye. So the *judgement* lives here as
pure functions: given decoded ledger records (or the raw payloads a JSONL line decodes to) and an
injected clock value, they return findings. The chaos loop in ``scripts/durable_soak.py`` and the
nightly workflow only *drive* processes and then call these functions.

WHAT IS CHECKED, AND WHY EACH IS ITS OWN FINDING
================================================
1. ``duplicate_turn_seq`` — two ``turn.started`` records for the same task claim the same
   ``turn_seq``. The checkpoint design promises "exactly one checkpoint per turn_seq", so a repeated
   ``turn_seq`` means a task ran a paid turn twice under one sequence number and the audit trail can
   no longer say which turn produced which edits.
2. ``duplicate_turn_key`` — the same ``turn_key`` was used by two *distinct* ``turn_id``s IN THE SAME
   SESSION. The engine's replay index is per session, so within one session a repeat must replay; if
   the turn ids differ, the replay path did not engage and the run was billed twice. This is the
   failure the whole ``turn_key`` design exists to prevent.
   ``turn_key_rescoped`` — the same key reached a DIFFERENT session. The run's session was re-created
   (a resume the engine could not honour), which gave the key a fresh index and so a genuinely new
   paid turn. Different cause, different fix, so a different code.
3. ``hash_chain_broken`` — ``prev_hash`` does not equal the previous record's ``hash``, or the
   record's stored ``hash`` does not equal the recomputed one. Either way the ledger is no longer
   tamper-evident, and any claim derived from it is unverifiable.
4. ``seq_gap`` — ``seq`` does not advance by exactly one. The ledger lost or duplicated a position,
   so "everything up to seq N" is not a complete account of the run.
5. ``turn_unpaired`` — a ``turn.started`` without its matching ``turn.finished`` (by ``turn_id``).
   Reported as a WARNING, not a failure: a soak kills workers mid-turn on purpose, so an interrupted
   turn is the experiment working. It is still named, because "started and never finished" also covers
   the case where the ledger lost the ending — which an operator should read, even when it is benign.

Each finding names the offending id. Findings are sorted deterministically (by code, then id) so a
report over the same records is byte-identical run to run — the determinism guarantee the rest of the
durable package holds itself to.

WHAT IS DELIBERATELY NOT HERE
=============================
No file reads, no subprocesses, no ``time.time()``, no network. The chaos loop owns I/O; the clock is
an argument. That split is what makes these checks testable in milliseconds and what keeps a soak's
verdict from depending on when it ran.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

# -- finding codes ------------------------------------------------------------------------------

DUPLICATE_TURN_SEQ: Final = "duplicate_turn_seq"
DUPLICATE_TURN_KEY: Final = "duplicate_turn_key"
TURN_KEY_RESCOPED: Final = "turn_key_rescoped"
HASH_CHAIN_BROKEN: Final = "hash_chain_broken"
SEQ_GAP: Final = "seq_gap"
TURN_UNPAIRED: Final = "turn_unpaired"

FINDING_CODES: Final[frozenset[str]] = frozenset(
    {
        DUPLICATE_TURN_SEQ,
        DUPLICATE_TURN_KEY,
        TURN_KEY_RESCOPED,
        HASH_CHAIN_BROKEN,
        SEQ_GAP,
        TURN_UNPAIRED,
    }
)

GENESIS_HASH: Final = "0" * 64


# Severity separates "the ledger is unsound" from "a turn was interrupted". A soak kills workers on
# purpose, so an interrupted turn is the EXPECTED cost of the experiment, not a violation of it. Folding
# the two together would make every honest soak fail — and, worse, would train a reader to ignore the
# finding that actually matters. ``hard`` findings fail the soak; ``warning`` findings are reported and
# counted but do not.
HARD: Final = "hard"
WARNING: Final = "warning"


@dataclass(frozen=True)
class Finding:
    """One invariant violation. ``subject`` is the id an operator needs to go look at.

    ``code`` is a stable machine token (never a sentence) so a CI job can assert on it; ``detail``
    carries the human explanation and any secondary ids. ``seq`` is the ledger position the finding
    was raised at (``None`` for whole-run findings), which is what makes a report sortable and lets a
    reader jump straight to the offending line. ``severity`` decides whether it fails the soak.
    """

    code: str
    subject: str
    seq: int | None
    detail: str
    severity: str = HARD

    @property
    def hard(self) -> bool:
        return self.severity == HARD

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "subject": self.subject,
            "seq": self.seq,
            "detail": self.detail,
            "severity": self.severity,
        }

    def render(self) -> str:
        """A single human line, stable across runs. Used by the CLI and the workflow summary."""
        where = "run" if self.seq is None else f"seq {self.seq}"
        marker = "" if self.hard else " [warning]"
        return f"{self.code}: {self.subject} ({where}){marker}: {self.detail}"


class SoakInvariantError(RuntimeError):
    """Raised when a caller asks for a verdict on input that is not a usable ledger.

    Invariants over malformed input are meaningless — a missing ``seq`` would be reported as a gap
    rather than as the decoding bug it is — so malformed input fails loudly instead of being folded
    into a finding.
    """

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


# -- canonical hashing (mirrors ledger.compute_hash, without importing the ledger) ---------------


def _canonical(payload: Mapping[str, Any]) -> str:
    """Canonical JSON: sorted keys, no whitespace, no NaN/Infinity.

    Duplicated from ``ledger._canonical`` on purpose: this module must be importable and testable
    without the ledger's filesystem/config dependencies, and the checker's whole job is to be an
    independent opinion about a ledger's integrity. Sharing the implementation would mean a bug in
    the writer hides itself from the checker. A test pins the two to the same bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 over canonical JSON with ``hash`` removed — the ledger's own definition."""
    body = {key: value for key, value in record.items() if key != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


# -- record accessors (tolerant of dicts and of ledger.LedgerRecord) -----------------------------


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an attribute-holding object (``LedgerRecord``).

    Both shapes appear in practice: ``load_ledger`` returns frozen dataclasses, while a soak reading
    raw JSONL lines has dicts. One accessor keeps the checks written once.
    """
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _require_seq(record: Any, position: int) -> int:
    value = _field(record, "seq")
    if isinstance(value, bool) or not isinstance(value, int):
        raise SoakInvariantError(
            f"record at position {position} has no integer seq",
            code="soak_record_invalid",
            details={"position": position, "seq": repr(value)},
        )
    return value


def _require_hash(record: Any, position: int) -> str:
    value = _field(record, "hash")
    if not isinstance(value, str) or len(value) != 64:
        raise SoakInvariantError(
            f"record at position {position} has no 64-char hash",
            code="soak_record_invalid",
            details={"position": position, "hash": repr(value)},
        )
    return value


def _payload(record: Any) -> Mapping[str, Any]:
    value = _field(record, "payload", {})
    return value if isinstance(value, Mapping) else {}


def _kind(record: Any) -> str:
    value = _field(record, "kind", "")
    return value if isinstance(value, str) else ""


# -- invariant checks ----------------------------------------------------------------------------


def check_duplicate_turn_seq(records: Sequence[Any]) -> list[Finding]:
    """More than one ``checkpoint`` record for one ``(run_id, task_id, turn_seq)``.

    ``turn_seq`` is recorded on the ``checkpoint`` record's payload — that is where the design's
    promise "exactly one checkpoint per turn_seq" is observable, and ``activity_steps`` asserts it by
    counting these records rather than trusting the call graph. ``turn.started`` carries the
    ``turn_key`` (which embeds the seq) but not the seq as a field, so a checkpoint duplicate is both
    the only honest signal and the one that matters: a second checkpoint means the audit ref for a
    turn can no longer name one commit.

    Scoped per ``(run_id, task_id)``: two tasks may legitimately both be on turn 1. The subject is
    ``run:task:turn_seq`` because that triple is what the checkpoint ref is derived from, so it is
    also the key an operator can hand to ``git show``.
    """
    seen: dict[tuple[str, str, int], int] = {}
    findings: list[Finding] = []
    for position, record in enumerate(records):
        if _kind(record) != "checkpoint":
            continue
        seq = _require_seq(record, position)
        payload = _payload(record)
        turn_seq = payload.get("turn_seq")
        if isinstance(turn_seq, bool) or not isinstance(turn_seq, int):
            continue  # a missing turn_seq is a schema problem the writer already validates
        run_id = str(_field(record, "run_id", ""))
        task_id = str(_field(record, "task_id", ""))
        key = (run_id, task_id, turn_seq)
        subject = f"{run_id}:{task_id}:{turn_seq}"
        if key in seen:
            findings.append(
                Finding(
                    code=DUPLICATE_TURN_SEQ,
                    subject=subject,
                    seq=seq,
                    detail=f"checkpoint at seq {seen[key]} already claimed this turn_seq",
                )
            )
        else:
            seen[key] = seq
    return findings


def check_duplicate_turn_key(records: Sequence[Any]) -> list[Finding]:
    """One ``turn_key`` bound to more than one ``turn_id`` — a replay that did not replay.

    A repeated ``turn_key`` with the *same* ``turn_id`` is the healthy case: the engine replayed and
    returned the original turn, which is exactly what the design promises. A differing ``turn_id``
    proves the replay path did not engage. Subject is the ``turn_key`` itself, since that is the
    string an operator searches for in the engine's logs.

    Two distinct causes, reported as distinct codes because the fix differs:

    * **same session** — the engine's turn-key index is PER SESSION (``state.turnKeyIndex`` is a map of
      session → key → record), so within one session a repeat must replay. A second ``turn_id`` here
      means the replay path itself is broken: a real bug in the engine's idempotency.
    * **different session** — the run re-created its session (a resume the engine could not honour),
      which gave the same key a fresh, empty index and therefore a genuinely new paid turn. The
      turn-key protection was defeated by losing continuity, not by a broken replay.

    Reporting both as ``duplicate_turn_key`` would send an operator hunting an engine bug when the
    cause was a dropped session. ``turn_key_rescoped`` names the second case explicitly.
    """
    first: dict[str, tuple[str, str]] = {}
    findings: list[Finding] = []
    for position, record in enumerate(records):
        if _kind(record) != "turn.started":
            continue
        seq = _require_seq(record, position)
        payload = _payload(record)
        turn_key = payload.get("turn_key")
        turn_id = payload.get("turn_id")
        if not isinstance(turn_key, str) or not turn_key:
            continue
        if not isinstance(turn_id, str) or not turn_id:
            continue
        # The session id lives at the record's top level on ``turn.event`` records and in the payload
        # on ``turn.started`` (``append`` refuses a session_id without an event_index, so the turn's
        # own record cannot carry it in the column). Accept either, or this check silently degrades
        # to "same key, different turn" for the one case the distinction exists to catch.
        session_id = str(_field(record, "session_id") or payload.get("session_id") or "")
        previous = first.get(turn_key)
        if previous is None:
            first[turn_key] = (turn_id, session_id)
            continue
        previous_turn, previous_session = previous
        if previous_turn == turn_id:
            continue  # the healthy replay: same key, same turn
        if session_id and previous_session and session_id != previous_session:
            findings.append(
                Finding(
                    code=TURN_KEY_RESCOPED,
                    subject=turn_key,
                    seq=seq,
                    detail=(
                        f"turn_key was first used by turn_id {previous_turn} in session "
                        f"{previous_session} and reused by {turn_id} in session {session_id} — the "
                        "session changed, so the key's per-session replay index no longer covered it"
                    ),
                )
            )
            continue
        findings.append(
            Finding(
                code=DUPLICATE_TURN_KEY,
                subject=turn_key,
                seq=seq,
                detail=f"turn_key was first used by turn_id {previous_turn} and reused by {turn_id}",
            )
        )
    return findings


def check_hash_chain(records: Sequence[Any]) -> list[Finding]:
    """``prev_hash`` must chain, and each stored ``hash`` must recompute to itself.

    Both halves are reported as ``hash_chain_broken`` but with distinct detail text: a broken link
    means a line was removed or reordered, while a bad hash means a line's content was edited. An
    operator needs to know which, because the first is an operational accident and the second is
    tampering or a canonical-JSON regression. Subject is the offending ``seq``.
    """
    findings: list[Finding] = []
    expected_prev = GENESIS_HASH
    for position, record in enumerate(records):
        seq = _require_seq(record, position)
        prev_hash = _field(record, "prev_hash")
        if prev_hash != expected_prev:
            findings.append(
                Finding(
                    code=HASH_CHAIN_BROKEN,
                    subject=str(seq),
                    seq=seq,
                    detail=f"prev_hash {prev_hash!r} does not match previous record's hash "
                    f"{expected_prev!r}",
                )
            )
        recomputed = compute_hash(_as_mapping(record))
        stored = _require_hash(record, position)
        if recomputed != stored:
            findings.append(
                Finding(
                    code=HASH_CHAIN_BROKEN,
                    subject=str(seq),
                    seq=seq,
                    detail=f"stored hash {stored!r} does not match recomputed {recomputed!r}",
                )
            )
        expected_prev = stored
    return findings


def _as_mapping(record: Any) -> dict[str, Any]:
    """A plain dict for hashing, from either a mapping or a ``LedgerRecord``-like object."""
    if isinstance(record, Mapping):
        return dict(record)
    to_payload = getattr(record, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        if isinstance(payload, Mapping):
            return dict(payload)
        raise SoakInvariantError(
            "record.to_payload() did not return a mapping",
            code="soak_record_invalid",
            details={"type": type(payload).__name__},
        )
    raise SoakInvariantError(
        "record is neither a mapping nor a ledger record",
        code="soak_record_invalid",
        details={"type": type(record).__name__},
    )


def check_seq_contiguity(records: Sequence[Any]) -> list[Finding]:
    """``seq`` must start at 1 and advance by exactly one. Subject is the position that broke it."""
    findings: list[Finding] = []
    expected = 1
    for position, record in enumerate(records):
        seq = _require_seq(record, position)
        if seq != expected:
            findings.append(
                Finding(
                    code=SEQ_GAP,
                    subject=str(seq),
                    seq=seq,
                    detail=f"seq is {seq} where {expected} was expected "
                    f"(position {position})",
                )
            )
        expected = seq + 1
    return findings


def check_turn_pairing(records: Sequence[Any], *, allow_open_turns: bool = False) -> list[Finding]:
    """Every ``turn.started`` needs a ``turn.finished`` with the same ``turn_id``.

    ``allow_open_turns`` exists for the live case: a soak snapshots a run *while* a turn is in flight,
    and reporting that as a violation would make the checker unusable mid-run. A finished run (or a
    replay after the fact) is checked strictly, because there an open turn is a genuine hole. Subject
    is the ``turn_id``, so the operator can search the engine's logs for it directly.
    """
    started: dict[str, int] = {}
    finished: set[str] = set()
    findings: list[Finding] = []
    for position, record in enumerate(records):
        kind = _kind(record)
        if kind not in {"turn.started", "turn.finished"}:
            continue
        seq = _require_seq(record, position)
        turn_id = _payload(record).get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            continue
        if kind == "turn.started":
            started.setdefault(turn_id, seq)
        else:
            finished.add(turn_id)
    if allow_open_turns:
        return findings
    for turn_id, seq in sorted(started.items(), key=lambda item: (item[1], item[0])):
        if turn_id not in finished:
            findings.append(
                Finding(
                    code=TURN_UNPAIRED,
                    subject=turn_id,
                    seq=seq,
                    detail="turn.started has no matching turn.finished",
                    # A warning, not a hard failure: a soak SIGKILLs workers mid-turn by design, so
                    # "the turn never finished" is the experiment working. It is still reported —
                    # a turn that never finished AND never failed means the ledger lost the ending.
                    severity=WARNING,
                )
            )
    return findings


# -- the report ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SoakReport:
    """The verdict over one run's records: findings, plus the facts a soak log wants to print."""

    run_id: str
    records: int
    turns_started: int
    turns_finished: int
    checkpoints: int
    findings: tuple[Finding, ...]
    checked_at: float | None

    @property
    def hard_findings(self) -> tuple[Finding, ...]:
        """The findings that make the ledger unsound. These fail the soak."""
        return tuple(finding for finding in self.findings if finding.hard)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Expected costs of the experiment (an interrupted turn). Reported, not fatal."""
        return tuple(finding for finding in self.findings if not finding.hard)

    @property
    def ok(self) -> bool:
        """True when no HARD invariant broke. Warnings do not make a soak fail — by design."""
        return not self.hard_findings

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "records": self.records,
            "turns_started": self.turns_started,
            "turns_finished": self.turns_finished,
            "checkpoints": self.checkpoints,
            "checked_at": self.checked_at,
            "hard_findings": len(self.hard_findings),
            "warnings": len(self.warnings),
            "findings": [finding.to_payload() for finding in self.findings],
        }

    def render(self) -> str:
        head = (
            f"soak {self.run_id}: {'ok' if self.ok else 'FAILED'} "
            f"({self.records} records, {self.turns_started} turns, "
            f"{self.turns_finished} finished, {self.checkpoints} checkpoints, "
            f"{len(self.hard_findings)} hard / {len(self.warnings)} warning)"
        )
        return "\n".join([head, *[finding.render() for finding in self.findings]])


def _count(records: Sequence[Any], kind: str) -> int:
    return sum(1 for record in records if _kind(record) == kind)


def _sort_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Deterministic order: by code, then seq (``None`` first), then subject.

    Sorting on the *code* first groups a class of failure together in a long report; the seq and
    subject tie-breakers keep two records that produce the same code in file order. Without this the
    order would follow dict iteration and the same ledger could render differently twice.
    """
    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.code,
                -1 if finding.seq is None else finding.seq,
                finding.subject,
            ),
        )
    )


def build_report(
    run_id: str,
    records: Sequence[Any],
    *,
    checked_at: float | None = None,
    allow_open_turns: bool = False,
) -> SoakReport:
    """Run every invariant over one run's records. Pure: same records and ``checked_at``, same report.

    ``checked_at`` is injected rather than read from the clock so a report is reproducible — a soak
    that prints a timestamp it generated itself cannot be diffed against a previous run.
    """
    if not isinstance(run_id, str) or not run_id:
        raise SoakInvariantError(
            "run_id must be a non-empty string",
            code="soak_input_invalid",
            details={"run_id": repr(run_id)},
        )
    if checked_at is not None and not isinstance(checked_at, (int, float)):
        raise SoakInvariantError(
            "checked_at must be a number or None",
            code="soak_input_invalid",
            details={"checked_at": repr(checked_at)},
        )
    findings: list[Finding] = []
    findings.extend(check_seq_contiguity(records))
    findings.extend(check_hash_chain(records))
    findings.extend(check_duplicate_turn_seq(records))
    findings.extend(check_duplicate_turn_key(records))
    findings.extend(check_turn_pairing(records, allow_open_turns=allow_open_turns))
    return SoakReport(
        run_id=run_id,
        records=len(records),
        turns_started=_count(records, "turn.started"),
        turns_finished=_count(records, "turn.finished"),
        checkpoints=_count(records, "checkpoint"),
        findings=_sort_findings(findings),
        checked_at=None if checked_at is None else float(checked_at),
    )


# -- the seeded kill schedule --------------------------------------------------------------------

# WHY a schedule and not random kills: a soak that kills at random times cannot be reproduced after a
# failure, so the one bug it finds is the one bug nobody can debug. The generator emits the *plan* of
# kill points up front, derived only from the seed and the count — same seed, same plan, forever.

# Fractions of the observed run used as the default spread of kill points. Chosen to land across the
# phases a worker actually occupies: early registration, mid-turn, at a checkpoint, and at the gate.
DEFAULT_KILL_FRACTIONS: Final[tuple[float, ...]] = (0.10, 0.35, 0.55, 0.75, 0.92)

# Bounds on the per-point jitter in seconds. Bounded so a schedule never produces a negative delay or
# one so large that a nightly job would exceed its own timeout waiting for the next kill.
MIN_JITTER_SECONDS: Final = 0.0
MAX_JITTER_SECONDS: Final = 2.0


@dataclass(frozen=True)
class KillPoint:
    """One planned interruption: how long to let the run breathe, then what to do."""

    index: int
    delay_seconds: float
    action: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "delay_seconds": round(self.delay_seconds, 6),
            "action": self.action,
        }


KILL_ACTIONS: Final[frozenset[str]] = frozenset({"sigkill", "sigterm"})


def generate_kill_points(
    seed: int,
    *,
    count: int = 5,
    fractions: Sequence[float] = DEFAULT_KILL_FRACTIONS,
    total_seconds: float = 300.0,
    actions: Sequence[str] = ("sigkill",),
    max_jitter_seconds: float = MAX_JITTER_SECONDS,
) -> list[KillPoint]:
    """A deterministic, seeded schedule of worker kills.

    ``seed`` alone drives the jitter, so ``generate_kill_points(7)`` returns the same list on any
    machine, in any Python 3.12, and at any time — the property that makes a soak failure
    reproducible. ``random.Random(seed)`` is used rather than module-level ``random`` so this module
    cannot be perturbed by another caller's draws.

    ``count`` is how many points to emit. When ``count`` differs from ``len(fractions)`` the
    fractions are resampled deterministically to ``count`` values spread evenly over ``[0, 1]``; the
    default constants are used only for the matching ``count``. That keeps "give me 12 kills" honest
    instead of silently truncating to five.

    Validation is strict and up front: a soak that ran for an hour because a negative delay slipped
    through is worse than one that refused to start.
    """
    _validate_schedule_inputs(
        count=count,
        fractions=fractions,
        total_seconds=total_seconds,
        actions=actions,
        max_jitter_seconds=max_jitter_seconds,
    )
    chosen = _resample_fractions(fractions, count)
    rng = random.Random(seed)
    points: list[KillPoint] = []
    for index, fraction in enumerate(chosen):
        base = total_seconds * fraction
        jitter = rng.uniform(MIN_JITTER_SECONDS, max_jitter_seconds)
        action = actions[index % len(actions)]
        points.append(KillPoint(index=index, delay_seconds=base + jitter, action=action))
    return points


def _validate_schedule_inputs(
    *,
    count: int,
    fractions: Sequence[float],
    total_seconds: float,
    actions: Sequence[str],
    max_jitter_seconds: float,
) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SoakInvariantError(
            "count must be a non-negative integer",
            code="soak_schedule_invalid",
            details={"count": repr(count)},
        )
    if not fractions:
        raise SoakInvariantError(
            "fractions must not be empty",
            code="soak_schedule_invalid",
            details={"fractions": []},
        )
    for fraction in fractions:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise SoakInvariantError(
                "every fraction must be a number",
                code="soak_schedule_invalid",
                details={"fraction": repr(fraction)},
            )
        if not 0.0 <= float(fraction) <= 1.0:
            raise SoakInvariantError(
                "fractions must lie within [0, 1]",
                code="soak_schedule_invalid",
                details={"fraction": repr(fraction)},
            )
    if isinstance(total_seconds, bool) or not isinstance(total_seconds, (int, float)):
        raise SoakInvariantError(
            "total_seconds must be a number",
            code="soak_schedule_invalid",
            details={"total_seconds": repr(total_seconds)},
        )
    if float(total_seconds) <= 0.0:
        raise SoakInvariantError(
            "total_seconds must be positive",
            code="soak_schedule_invalid",
            details={"total_seconds": repr(total_seconds)},
        )
    if isinstance(max_jitter_seconds, bool) or not isinstance(max_jitter_seconds, (int, float)):
        raise SoakInvariantError(
            "max_jitter_seconds must be a number",
            code="soak_schedule_invalid",
            details={"max_jitter_seconds": repr(max_jitter_seconds)},
        )
    if float(max_jitter_seconds) < MIN_JITTER_SECONDS:
        raise SoakInvariantError(
            "max_jitter_seconds must not be negative",
            code="soak_schedule_invalid",
            details={"max_jitter_seconds": repr(max_jitter_seconds)},
        )
    if not actions:
        raise SoakInvariantError(
            "actions must not be empty",
            code="soak_schedule_invalid",
            details={"actions": []},
        )
    for action in actions:
        if action not in KILL_ACTIONS:
            raise SoakInvariantError(
                f"unknown kill action {action!r}",
                code="soak_schedule_invalid",
                details={"action": repr(action), "allowed": sorted(KILL_ACTIONS)},
            )


def _resample_fractions(fractions: Sequence[float], count: int) -> list[float]:
    """``count`` fractions spread evenly over [0, 1], or the given ones when the lengths match.

    Evenly spread rather than seeded-random so the *shape* of a schedule is predictable to a human
    reading the plan, while the jitter (seeded) is what varies. A schedule whose points cluster in
    the first 10% would kill only registration and never exercise a checkpoint.
    """
    if count == len(fractions):
        return [float(fraction) for fraction in fractions]
    if count == 0:
        return []
    # ``count`` points from 1/(count+1) .. count/(count+1): no point lands at 0 or 1, so a kill is
    # never "before the run started" or "after it should have finished".
    step = 1.0 / (count + 1)
    return [round(step * (index + 1), 6) for index in range(count)]


def describe_schedule(points: Sequence[KillPoint]) -> str:
    """One deterministic line per point. Printed into the soak log so a failure is reproducible."""
    return "\n".join(
        f"kill[{point.index}] +{point.delay_seconds:.3f}s {point.action}" for point in points
    )
