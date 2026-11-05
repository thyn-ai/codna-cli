"""The run ledger — append-only, hash-chained, and the run's source of truth for what happened.

Every durable run writes one NDJSON file at ``~/.codna/runs/<run>/ledger.ndjson``. It is the only
record that survives everything else: Temporal history is bounded and eventually expires, the
sidecar keeps sessions in memory, and the worker's logs rotate. The ledger is what the IDE
reattaches to (``?after=``), what ``proofs/usage.json`` is reconciled against, and therefore what
an auditor reads.

Four properties are enforced, each of which has been the actual failure mode somewhere:

1. **Append-only with a hash chain.** Each record carries ``prev_hash`` = the previous record's
   ``hash``, and ``hash`` = SHA-256 over the record's canonical JSON with ``hash`` removed. Editing
   a past event changes its hash and breaks every link after it, which `verify_chain` reports as
   the exact index where the file diverges. This is tamper-*evidence*, not tamper-proofing: the
   file is on the user's own disk and a determined owner can rewrite the whole chain. It exists so
   an accidental or partial edit is detectable, and so the hosted deployment (E2) can sign the head.
2. **Dedupe on ``(session_id, event_index)``.** A durable turn is at-least-once: an activity that
   times out after the ledger append is retried and appends the same events again. The pair is the
   sidecar's own identity for an event, so re-appending is a no-op instead of a second copy.
3. **Contiguity.** A run's ``event_index`` must advance by exactly one. A gap means events were
   lost (a dropped SSE tail), which is a correctness problem for reattach and cost accounting, so
   it is reported rather than silently tolerated.
4. **A Parquet rollup at task boundaries.** NDJSON is the record; Parquet is the thing that can be
   scanned across a year of runs. The rollup is *derived* — always rebuildable from the NDJSON —
   and written atomically so a crash mid-rollup leaves the previous one intact.

Concurrency: a single worker process appends per run, and two appenders would interleave and break
the chain. `append` therefore takes an exclusive ``flock`` on the ledger file for the duration of
the append and re-reads the tail hash *under the lock*, which also makes the common case (two
activities for different tasks finishing at once) safe.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

from .config import DurablePaths, validate_run_id
from ..runtime.processes import ensure_directories

__all__ = [
    "GENESIS_HASH",
    "LedgerError",
    "LedgerRecord",
    "LedgerStats",
    "RunLedger",
    "compute_hash",
    "verify_chain",
]

GENESIS_HASH: Final = "0" * 64

# The record kinds a run writes. Kept as a closed set so a typo in an activity is caught at append
# time rather than becoming an unqueryable row six months later.
RECORD_KINDS: Final[frozenset[str]] = frozenset(
    {
        "run.started",
        "run.finished",
        "task.started",
        "task.finished",
        "turn.started",
        "turn.event",
        "turn.finished",
        "checkpoint",
        "verify",
        "review",
        "approval.requested",
        "approval.granted",
        "budget.updated",
        "note",
    }
)


class LedgerError(RuntimeError):
    """A ledger operation that must not be silently tolerated."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


def _canonical(payload: Mapping[str, Any]) -> str:
    """Canonical JSON for hashing: sorted keys, no whitespace, no NaN/Infinity.

    ``allow_nan=False`` matters — Python writes ``NaN``/``Infinity`` by default, which is not JSON,
    would not round-trip through any other reader of the ledger, and would make two processes
    disagree about a record's hash.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_hash(record: Mapping[str, Any]) -> str:
    """The record's hash: SHA-256 over canonical JSON with the ``hash`` field itself removed."""
    body = {key: value for key, value in record.items() if key != "hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LedgerRecord:
    """One ledger line, decoded and validated.

    ``seq`` is the ledger's own monotonic position (1-based). It is NOT the sidecar's
    ``event_index``: a turn can append many ledger records for one event index, and a run can
    record non-event facts (an approval, a checkpoint) between them. Keeping both is what lets
    `contiguity` be a property of the ledger while `dedupe` stays a property of the session.
    """

    seq: int
    kind: str
    ts: float
    run_id: str
    task_id: str | None
    session_id: str | None
    event_index: int | None
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "ts": self.ts,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "event_index": self.event_index,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "LedgerRecord":
        try:
            return cls(
                seq=int(raw["seq"]),
                kind=str(raw["kind"]),
                ts=float(raw["ts"]),
                run_id=str(raw["run_id"]),
                task_id=None if raw.get("task_id") is None else str(raw["task_id"]),
                session_id=None if raw.get("session_id") is None else str(raw["session_id"]),
                event_index=None if raw.get("event_index") is None else int(raw["event_index"]),
                payload=dict(raw.get("payload") or {}),
                prev_hash=str(raw["prev_hash"]),
                hash=str(raw["hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(
                f"malformed ledger record: {exc}",
                code="ledger_record_invalid",
                details={"error": str(exc)},
            ) from exc


@dataclass(frozen=True)
class LedgerStats:
    """What a rollup needs to know about a ledger, without loading it into memory twice."""

    records: int
    last_seq: int
    last_hash: str
    first_ts: float | None
    last_ts: float | None


def _iter_records(path: Path) -> Iterator[LedgerRecord]:
    """Yield decoded records, failing on a truncated final line.

    A partially written last line is the one case worth naming: it happens when a process is
    killed mid-append. Because appends are fsynced per line under a lock, it means the write did
    not commit, so the caller can truncate and retry — but it must *know*, which is why this
    raises instead of skipping.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"ledger line {line_number} is not valid JSON: {exc}",
                    code="ledger_line_invalid",
                    details={"line": line_number, "error": str(exc)},
                ) from exc
            if not isinstance(decoded, dict):
                raise LedgerError(
                    f"ledger line {line_number} is not an object",
                    code="ledger_line_invalid",
                    details={"line": line_number},
                )
            yield LedgerRecord.from_payload(decoded)


def verify_chain(path: Path) -> list[str]:
    """Verify the hash chain and contiguous ``seq`` values. Returns the problems found.

    Returns a report instead of raising because "is this ledger intact?" is a question with both
    answers, and the CLI (`codna run verify-ledger`) must print the divergence rather than crash
    on it. An empty list means the file is intact end to end.
    """
    problems: list[str] = []
    expected_prev = GENESIS_HASH
    expected_seq = 1
    for record in _iter_records(path):
        if record.seq != expected_seq:
            problems.append(f"seq {record.seq} at position {expected_seq} (expected {expected_seq})")
        if record.prev_hash != expected_prev:
            problems.append(
                f"seq {record.seq}: prev_hash {record.prev_hash[:12]}… does not match {expected_prev[:12]}…"
            )
        recomputed = compute_hash(record.to_payload())
        if recomputed != record.hash:
            problems.append(
                f"seq {record.seq}: hash mismatch (stored {record.hash[:12]}…, computed {recomputed[:12]}…)"
            )
        expected_prev = record.hash
        expected_seq += 1
    return problems


@contextlib.contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on the ledger file itself.

    Locking the data file (not a sidecar ``.lock``) means it cannot drift from the thing it
    protects: there is no window where the lock exists but the file was replaced.
    """
    ensure_directories(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunLedger:
    """Append-only ledger for one run.

    One instance per (process, run). Not thread-safe by itself — callers serialise appends, and
    cross-process safety comes from the file lock in `_exclusive`.
    """

    def __init__(self, path: Path, *, run_id: str, clock: Any = None) -> None:
        self._path = path
        self._run_id = validate_run_id(run_id)
        self._clock = clock or time.time
        self._last_hash: str | None = None
        self._last_seq: int = 0
        self._seen: set[tuple[str, int]] = set()
        self._loaded = False

    @classmethod
    def for_run(cls, paths: DurablePaths, run_id: str, *, clock: Any = None) -> "RunLedger":
        return cls(paths.ledger_path(run_id), run_id=run_id, clock=clock)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    # -- loading ------------------------------------------------------------------

    def _reload_tail_under_lock(self) -> None:
        """Refresh the cached tail from the FILE, under the lock, before computing `seq`.

        Called on every append. On the first call it is exactly `_ensure_loaded`; on later calls it is the
        difference between a linear chain and a fork, because another instance may have appended since this one
        loaded. The dedupe set is rebuilt too: an event this instance never saw may already be in the file, and
        honouring the idempotency key is the other half of what the lock is for.

        Deliberately re-reads rather than trusting in-memory state — the file is the source of truth and it is
        shared. The cost is one sequential read of a local append-only file, which is what `_iter_records` already
        does for the first load.
        """
        last_hash = GENESIS_HASH
        last_seq = 0
        seen: set[tuple[str, int]] = set()
        for record in _iter_records(self._path):
            last_hash = record.hash
            last_seq = record.seq
            if record.session_id is not None and record.event_index is not None:
                seen.add((record.session_id, record.event_index))
        self._last_hash = last_hash
        self._last_seq = last_seq
        self._seen = seen
        self._loaded = True

    # -- appending ----------------------------------------------------------------

    def append(
        self,
        kind: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        event_index: int | None = None,
        payload: Mapping[str, Any] | None = None,
        allow_duplicate_event: bool = False,
    ) -> LedgerRecord | None:
        """Append one record. Returns it, or ``None`` when it was a deduplicated event.

        ``allow_duplicate_event`` exists for the one legitimate case of the same event index
        recorded twice: a *different kind* of record about the same event (a ``turn.event`` and the
        ``turn.finished`` marker that summarises it). Those callers pass it explicitly, so the
        default stays strict.
        """
        if kind not in RECORD_KINDS:
            raise LedgerError(
                f"unknown ledger record kind {kind!r}",
                code="ledger_kind_invalid",
                details={"kind": kind, "allowed": sorted(RECORD_KINDS)},
            )
        if event_index is not None and (not isinstance(event_index, int) or isinstance(event_index, bool)):
            raise LedgerError(
                "event_index must be an integer or None",
                code="ledger_event_index_invalid",
                details={"event_index": repr(event_index)},
            )
        if (session_id is None) != (event_index is None):
            raise LedgerError(
                "session_id and event_index must be provided together: dedupe is keyed on the pair",
                code="ledger_dedupe_key_incomplete",
                details={"session_id": session_id, "event_index": event_index},
            )
        body = dict(payload or {})
        # Round-trip the payload through canonical JSON now, so an unserializable value fails here
        # (with the kind named) instead of after the hash has been computed and half-written.
        try:
            body = json.loads(_canonical(body))
        except (TypeError, ValueError) as exc:
            raise LedgerError(
                f"ledger payload for {kind!r} is not JSON-encodable: {exc}",
                code="ledger_payload_invalid",
                details={"kind": kind, "error": str(exc)},
            ) from exc

        with _exclusive(self._path):
            # THE TAIL MUST BE RE-READ UNDER THE LOCK, ON EVERY APPEND — not only the first.
            #
            # `_ensure_loaded` replays the file once and caches the tail, which is right for the FIRST append. But
            # the ledger is shared: two activities (or two processes) each hold their own instance, and the flock
            # serialises only the WRITE. An instance that loaded before its neighbour appended still believes it
            # owns the tail, so it computes `seq = cached + 1` from a STALE parent and both records land on the
            # same seq with the same prev_hash — forking a chain whose entire purpose is being tamper-evident.
            #
            # Measured: two instances, alternating appends -> seqs [1, 2, 2, 3] with a shared prev_hash, and the
            # official `verify_chain` reports it. Re-reading here costs one small tail read per append (the file is
            # already open under the lock) and makes concurrency safe by construction rather than by convention.
            self._reload_tail_under_lock()
            key = (session_id, event_index) if session_id is not None and event_index is not None else None
            if key is not None and key in self._seen and not allow_duplicate_event:
                return None
            record = LedgerRecord(
                seq=self._last_seq + 1,
                kind=kind,
                ts=float(self._clock()),
                run_id=self._run_id,
                task_id=task_id,
                session_id=session_id,
                event_index=event_index,
                payload=body,
                prev_hash=self._last_hash or GENESIS_HASH,
                hash="",
            )
            hashed = LedgerRecord(**{**record.__dict__, "hash": compute_hash(record.to_payload())})
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(hashed.to_payload()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._last_hash = hashed.hash
            self._last_seq = hashed.seq
            if key is not None:
                self._seen.add(key)
            return hashed

    def append_events(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        task_id: str | None = None,
        session_id: str,
    ) -> list[LedgerRecord]:
        """Append a batch of sidecar events, dropping ones already recorded.

        The batch comes from an SSE window that may overlap a previous attempt after a reconnect,
        so duplicates are the normal case here, not an error.
        """
        appended: list[LedgerRecord] = []
        for event in events:
            index = event.get("event_index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise LedgerError(
                    "every ledgered event needs an integer event_index",
                    code="ledger_event_index_invalid",
                    details={"event": {key: event[key] for key in sorted(event) if key != "payload"}},
                )
            record = self.append(
                "turn.event",
                task_id=task_id,
                session_id=session_id,
                event_index=index,
                payload={"event_type": event.get("event_type"), "payload": event.get("payload") or {}},
            )
            if record is not None:
                appended.append(record)
        return appended

    # -- reading ------------------------------------------------------------------

    def records(self) -> list[LedgerRecord]:
        return list(_iter_records(self._path))

    def tail(self) -> LedgerRecord | None:
        last: LedgerRecord | None = None
        for last in _iter_records(self._path):
            pass
        return last

    def stats(self) -> LedgerStats:
        count = 0
        last_seq = 0
        last_hash = GENESIS_HASH
        first_ts: float | None = None
        last_ts: float | None = None
        for record in _iter_records(self._path):
            count += 1
            last_seq = record.seq
            last_hash = record.hash
            if first_ts is None:
                first_ts = record.ts
            last_ts = record.ts
        return LedgerStats(
            records=count, last_seq=last_seq, last_hash=last_hash, first_ts=first_ts, last_ts=last_ts
        )

    def check_contiguity(self) -> list[str]:
        """Report gaps or repeats in the ledger's own ``seq`` (not the sidecar's event index)."""
        problems: list[str] = []
        expected = 1
        for record in _iter_records(self._path):
            if record.seq != expected:
                problems.append(f"expected seq {expected}, found {record.seq}")
                expected = record.seq
            expected += 1
        return problems

    def check_session_contiguity(self, session_id: str) -> list[str]:
        """Report gaps in the sidecar ``event_index`` sequence for one session.

        This is the property reattach depends on: `?after=n` assumes n+1 exists. A gap is reported
        per (task, session) because a run legitimately starts a new session per task.
        """
        indices: dict[str | None, list[int]] = {}
        for record in _iter_records(self._path):
            if record.session_id != session_id or record.event_index is None:
                continue
            indices.setdefault(record.task_id, []).append(record.event_index)
        problems: list[str] = []
        for task_id, values in indices.items():
            ordered = sorted(set(values))
            for previous, current in zip(ordered, ordered[1:]):
                if current != previous + 1:
                    problems.append(
                        f"session {session_id} task {task_id or '-'}: event_index {previous} → {current} "
                        f"skips {current - previous - 1}"
                    )
        return problems

    def verify(self) -> list[str]:
        """Hash chain + ledger seq integrity."""
        return verify_chain(self._path)

    # -- rollup -------------------------------------------------------------------

    def rollup_parquet(self, *, out_path: Path | None = None) -> Path:
        """Write the Parquet rollup of this ledger. Returns the file written.

        Derived data, so it is rebuilt from scratch and replaced atomically: a half-written rollup
        is never observable, and a stale one never accumulates. ``task.started``/``task.finished``
        boundaries are what callers hook this to, which is why it is cheap enough (one pass, a
        handful of columns) to run at every boundary.
        """
        target = out_path or self._path.with_name("ledger.parquet")
        ensure_directories(target.parent)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - pyarrow is a base dependency
            raise LedgerError(
                "the Parquet rollup needs pyarrow, which is a base codna dependency.",
                code="ledger_parquet_unavailable",
                details={"error": str(exc)},
            ) from exc

        rows: list[dict[str, Any]] = []
        for record in _iter_records(self._path):
            rows.append(
                {
                    "seq": record.seq,
                    "kind": record.kind,
                    "ts": record.ts,
                    "run_id": record.run_id,
                    "task_id": record.task_id,
                    "session_id": record.session_id,
                    "event_index": record.event_index,
                    "prev_hash": record.prev_hash,
                    "hash": record.hash,
                    "event_type": record.payload.get("event_type"),
                    # Cost is extracted into its own column so a year of runs can be summed without
                    # parsing JSON; the rest of the payload stays in the NDJSON, not duplicated here.
                    "cost_usd": _as_float(record.payload.get("cost_usd")),
                    "tokens": _as_float(record.payload.get("tokens")),
                }
            )

        schema = pa.schema(
            [
                ("seq", pa.int64()),
                ("kind", pa.string()),
                ("ts", pa.float64()),
                ("run_id", pa.string()),
                ("task_id", pa.string()),
                ("session_id", pa.string()),
                ("event_index", pa.int64()),
                ("prev_hash", pa.string()),
                ("hash", pa.string()),
                ("event_type", pa.string()),
                ("cost_usd", pa.float64()),
                ("tokens", pa.float64()),
            ]
        )
        table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
        tmp = target.with_suffix(target.suffix + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, target)
        return target


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_ledger(path: Path) -> tuple[list[LedgerRecord], list[str]]:
    """Load a ledger and its integrity report in one pass over the file.

    The CLI and the Run API both want "the records, and whether they are intact"; doing it in one
    read avoids the two-answer race where the file grows between calls.
    """
    records = list(_iter_records(path))
    problems: list[str] = []
    expected_prev = GENESIS_HASH
    expected_seq = 1
    for record in records:
        if record.seq != expected_seq:
            problems.append(f"seq {record.seq} at position {expected_seq} (expected {expected_seq})")
        if record.prev_hash != expected_prev:
            problems.append(f"seq {record.seq}: prev_hash does not match")
        if compute_hash(record.to_payload()) != record.hash:
            problems.append(f"seq {record.seq}: hash mismatch")
        expected_prev = record.hash
        expected_seq += 1
    return records, problems


def records_after(records: Sequence[LedgerRecord], after: int) -> list[LedgerRecord]:
    """Records with ``seq > after`` — the reattach query."""
    return [record for record in records if record.seq > after]
