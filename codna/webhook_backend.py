"""Queue backend selection for the Codna GitHub App webhook, behind ONE flag.

``CODNA_WEBHOOK_QUEUE_BACKEND`` picks what ``codna webhook serve`` runs on:

* ``sqlite`` (the default, and today's production path, byte for byte): the WAL-mode SQLite file
  on the machine's volume, :class:`codna.webhook_queue.WebhookQueue`.
* ``shadow``: SQLite stays authoritative for every claim; each accepted delivery is ALSO written
  to Postgres, and any Postgres failure is counted (``shadow_errors``) and never surfaced to
  GitHub. This is how the new backend earns trust in production before it owns anything.
* ``postgres``: :class:`codna.webhook_pg_queue.PostgresQueue` is authoritative, wrapped in a
  :class:`SpoolingQueue`: the SQLite file becomes a write-ahead SPOOL. A delivery is written to
  Postgres first; if that fails (the database is unreachable, slow, mid-failover) the row lands in
  the local file and a forwarder thread moves it across with the same delivery id as soon as
  Postgres answers again -- so GitHub always gets its 202 and no delivery is ever lost to a
  database outage. When Postgres has been unreachable longer than ``CODNA_WEBHOOK_DEGRADED_AFTER_S``
  the ingress may run ``review`` jobs from its own spool (the DEGRADED LANE, kinds listed in
  ``CODNA_WEBHOOK_DEGRADED_LANE``, empty disables it); it never runs a ``fix`` there, because a fix
  executes the repository's own tests and the ingress holds the App's secrets.

Switching backends never loses a job: on the first ``postgres`` boot every row still in flight
in the SQLite file is imported once (:meth:`PostgresQueue.import_sqlite_backlog`); switching back
to ``sqlite`` exports every in-flight Postgres row into the file (:func:`open_queue` with
``rollback_from_postgres``). Both directions are idempotent and covered by
``cli/tests/test_webhook_backend_spool.py``.

The row ids a :class:`SpoolingQueue` hands out are DISJOINT between the two stores: a spool row is
offset by :data:`SPOOL_ROW_OFFSET`, so ``complete(row_id)`` always knows which store to write.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

from .webhook import WebhookJob
from .webhook_queue import QueuedJob, WebhookQueue

BACKENDS = ("sqlite", "shadow", "postgres")
BACKEND_ENV = "CODNA_WEBHOOK_QUEUE_BACKEND"
SPOOL_ROW_OFFSET = 1 << 40  # spool row ids live far above any Postgres identity value


class QueueBackend(Protocol):
    """What the worker pool and the ingress need from a queue. Both queue classes satisfy it."""

    def enqueue(self, job: WebhookJob, *, delivery_id: str | None, priority: int = 0) -> bool: ...
    def claim(self, *, reserve_priority: bool = False) -> QueuedJob | None: ...
    def complete(self, row_id: int, *, status: str, result: Mapping[str, object] | None = None,
                 retry: bool = True, retry_after_s: float | None = None) -> None: ...
    def recover_stale(self) -> dict[str, int]: ...
    def row_state(self, row_id: int) -> tuple[str, int] | None: ...
    def counts(self) -> dict[str, int]: ...
    def recent(self, *, limit: int = 10) -> list[dict[str, Any]]: ...


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


def backend_name(environ: Mapping[str, str] | None = None) -> str:
    """The configured backend; anything unrecognised is refused loudly rather than guessed --
    a typo must never silently run production on the wrong queue."""
    source = os.environ if environ is None else environ
    name = (source.get(BACKEND_ENV) or "sqlite").strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={name!r}: expected one of {', '.join(BACKENDS)}")
    return name


def degraded_lane_kinds(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """Kinds the ingress may run from its spool while Postgres is down. ``fix`` is refused even
    if configured: the ingress must never execute a repository's tests."""
    source = os.environ if environ is None else environ
    raw = source.get("CODNA_WEBHOOK_DEGRADED_LANE", "review")
    kinds = {k.strip() for k in raw.split(",") if k.strip()}
    return frozenset(kinds - {"fix"})


# --- the job the current worker thread is running (for _job_env's CODNA_REVIEW_RUN_ID) ------------
_current = threading.local()


def set_current_job(row_id: int | None) -> None:
    """Called by the pool's process wrapper around each job so the env the CLI subprocess gets can
    carry the queue row id (``CODNA_REVIEW_RUN_ID``): the review body is stamped with it and a
    second run of the SAME job finds the stamp and does not post twice (review_github)."""
    _current.row_id = row_id


def current_job_id() -> int | None:
    return getattr(_current, "row_id", None)


# --- shadow: SQLite authoritative, Postgres dual-written ----------------------------------------
class ShadowQueue:
    """Every read and every claim goes to SQLite; ``enqueue`` also writes Postgres and counts what
    happened. ``shadow_drift`` is the operator's gate for Phase 2: it is the number of accepted
    deliveries whose Postgres twin does not exist."""

    def __init__(self, primary: WebhookQueue, shadow: Any) -> None:
        self._primary = primary
        self._shadow = shadow
        self.shadow_writes = 0
        self.shadow_errors = 0
        self.shadow_duplicates = 0
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._primary.path

    @property
    def primary(self) -> WebhookQueue:
        return self._primary

    @property
    def shadow(self) -> Any:
        return self._shadow

    def enqueue(self, job: WebhookJob, *, delivery_id: str | None, priority: int = 0, unless_pending: bool = False,
                check_run_id: int | None = None) -> bool:
        # check_run_id is accepted for signature parity with the Postgres queue and ignored: the
        # ingress never pre-creates a Check Run on this backend (SQLite runs the job and creates
        # the run at claim), so a caller can never hand one in.
        fresh = self._primary.enqueue(job, delivery_id=delivery_id, priority=priority, unless_pending=unless_pending)
        if fresh:
            try:
                twin = self._shadow.enqueue(job, delivery_id=delivery_id, priority=priority, unless_pending=unless_pending)
                with self._lock:
                    self.shadow_writes += 1
                    if not twin:
                        self.shadow_duplicates += 1
            except Exception as exc:  # noqa: BLE001 -- the shadow must never fail a delivery
                with self._lock:
                    self.shadow_errors += 1
                _log("shadow_enqueue_error", delivery_id=delivery_id, error=type(exc).__name__)
        return fresh

    def shadow_drift(self) -> int | None:
        """Accepted deliveries (SQLite) minus rows present in Postgres, over everything in the
        file. None when Postgres cannot be asked. Zero for three days is the Phase-2 gate."""
        try:
            ids = [row["delivery_id"] for row in self._primary.recent(limit=50) if row.get("delivery_id")]
            if not ids:
                return 0
            with self._shadow.connection() as conn:
                rows = conn.execute(
                    f"SELECT delivery_id FROM {self._shadow.schema}.jobs WHERE delivery_id = ANY(%s)", (ids,)
                ).fetchall()
            present = {r["delivery_id"] for r in rows}
            return sum(1 for i in ids if i not in present)
        except Exception:  # noqa: BLE001
            return None

    def __getattr__(self, name: str) -> Any:  # everything else is SQLite's, unchanged
        return getattr(self._primary, name)


# --- postgres: authoritative, with the SQLite file as a write-ahead spool ---------------------------
class SpoolingQueue:
    """See the module docstring. ``primary`` is the PostgresQueue, ``spool`` the SQLite file."""

    def __init__(self, primary: Any, spool: WebhookQueue, *, enqueue_timeout_s: float = 2.0,
                 forward_interval_s: float = 5.0, degraded_after_s: float = 60.0,
                 degraded_kinds: Iterable[str] = ("review",), clock: Callable[[], float] = time.monotonic) -> None:
        self._primary = primary
        self._spool = spool
        self._enqueue_timeout_s = float(enqueue_timeout_s)
        self._forward_interval_s = float(forward_interval_s)
        self._degraded_after_s = float(degraded_after_s)
        self._degraded_kinds = frozenset(degraded_kinds) - {"fix"}
        self._clock = clock
        self._down_since: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._forwarder: threading.Thread | None = None
        self.spooled = 0
        self.forwarded = 0
        self.degraded_claims = 0

    # -- state ------------------------------------------------------------------------------------
    @property
    def path(self) -> str:
        return self._spool.path

    @property
    def primary(self) -> Any:
        return self._primary

    @property
    def spool(self) -> WebhookQueue:
        return self._spool

    def _mark_down(self) -> None:
        with self._lock:
            if self._down_since is None:
                self._down_since = self._clock()
                _log("postgres_unreachable", since=self._down_since)

    def _mark_up(self) -> None:
        with self._lock:
            if self._down_since is not None:
                _log("postgres_reachable_again", down_for_s=round(self._clock() - self._down_since, 1))
            self._down_since = None

    def postgres_down_for_s(self) -> float:
        with self._lock:
            return 0.0 if self._down_since is None else max(0.0, self._clock() - self._down_since)

    def degraded(self) -> bool:
        return bool(self._degraded_kinds) and self.postgres_down_for_s() > self._degraded_after_s

    def spool_rows(self) -> int:
        counts = self._spool.counts()
        return sum(counts.get(k, 0) for k in ("queued", "retry", "running"))

    # -- enqueue: Postgres first, the file when it cannot answer ------------------------------------
    def enqueue(self, job: WebhookJob, *, delivery_id: str | None, priority: int = 0, unless_pending: bool = False,
                check_run_id: int | None = None) -> bool:
        # The late path (a Postgres write that answers after the timeout) retires the SPOOL copy,
        # so it must not look before that copy exists: it waits on this gate, which the timeout
        # branch below sets once its spool write is done (success or failure). Ordered, not timed.
        spool_written = threading.Event()
        try:
            fresh = _call_with_timeout(
                lambda: self._primary.enqueue(job, delivery_id=delivery_id, priority=priority, unless_pending=unless_pending,
                                              check_run_id=check_run_id),
                self._enqueue_timeout_s,
                on_late=lambda accepted: self._retire_spooled(delivery_id) if accepted else None,
                late_gate=spool_written)
        except Exception as exc:  # noqa: BLE001 -- the file takes it; GitHub still gets its 202
            self._mark_down()
            try:
                # The spool row does NOT carry the pre-created Check Run: the file has no owner for
                # it (no reaper sweeps the spool), and the ingress, finding the run bound to no
                # Postgres row, closes it at once (webhook_queued_check.bind_or_close). The worker
                # that eventually claims the forwarded row creates the run then, as it always has.
                fresh = self._spool.enqueue(job, delivery_id=delivery_id, priority=priority, unless_pending=unless_pending)
            finally:
                spool_written.set()
            if fresh:
                with self._lock:
                    self.spooled += 1
            _log("delivery_spooled", delivery_id=delivery_id, kind=job.kind, error=type(exc).__name__, fresh=fresh)
            return fresh
        self._mark_up()
        return fresh

    def _retire_spooled(self, delivery_id: str | None) -> None:
        """A Postgres write that answered AFTER the enqueue timeout succeeded after all, so the copy
        the timeout put in the spool must go before anything can claim it: two claimable rows for
        one delivery is the one thing the spool must never produce. Only a row still WAITING is
        retired; the degraded lane cannot have taken it (it opens after 60 s down; the pool's own
        bounds answer within 10 s), and a row it did take completes from the spool as usual."""
        if not delivery_id:
            return
        try:
            row_id = self._spool.find_in_flight(delivery_id)
            if row_id is not None and self._spool.mark_rows([row_id], status="migrated",
                                                             result={"forwarded_to": "postgres", "late_write": True}):
                with self._lock:
                    self.forwarded += 1
                _log("spool_retired_after_late_write", delivery_id=delivery_id)
        except Exception as exc:  # noqa: BLE001 -- the forwarder retires it on its next pass anyway
            _log("spool_retire_error", delivery_id=delivery_id, error=type(exc).__name__)
        self._mark_up()

    # -- forwarder: drain the spool into Postgres whenever it answers ------------------------------
    def forward_once(self) -> int:
        """Move every queued/retry spool row into Postgres (same delivery id; a twin that already
        exists just retires the spool row). Running spool rows -- the degraded lane's -- stay.
        Returns how many rows were retired from the spool."""
        moved = 0
        try:
            rows = [r for r in self._spool.export_rows() if r["status"] in ("queued", "retry")]
        except Exception as exc:  # noqa: BLE001
            _log("spool_read_error", error=type(exc).__name__)
            return 0
        for row in rows:
            try:
                self._primary.enqueue(row["job"], delivery_id=row["delivery_id"], attempts=int(row["attempts"]),
                                      src_sqlite_id=int(row["id"]))
            except Exception as exc:  # noqa: BLE001 -- still down: try again next tick
                self._mark_down()
                _log("spool_forward_error", row_id=row["id"], error=type(exc).__name__)
                break
            self._mark_up()
            self._spool.mark_rows([int(row["id"])], status="migrated", result={"forwarded_to": "postgres"})
            moved += 1
        if moved:
            with self._lock:
                self.forwarded += moved
        return moved

    def start_forwarder(self) -> None:
        if self._forwarder is not None:
            return
        self._forwarder = threading.Thread(target=self._forward_loop, name="codna-webhook-spool-forwarder", daemon=True)
        self._forwarder.start()

    def forwarder_alive(self) -> bool:
        return self._forwarder is not None and self._forwarder.is_alive()

    def _forward_loop(self) -> None:
        while not self._stop.wait(self._forward_interval_s):
            try:
                if self.spool_rows():
                    self.forward_once()
                elif self._down_since is not None and self._primary.ping():
                    self._mark_up()
            except Exception as exc:  # noqa: BLE001 -- the forwarder must outlive any one error
                _log("spool_forwarder_error", error=type(exc).__name__)

    def stop(self) -> None:
        self._stop.set()

    # -- claim / complete: Postgres, or the degraded lane -------------------------------------------
    def claim(self, *, reserve_priority: bool = False, **kwargs: Any) -> QueuedJob | None:
        try:
            qjob = self._primary.claim(reserve_priority=reserve_priority, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._mark_down()
            if not self.degraded():
                raise
            _log("degraded_lane_claim", kinds=sorted(self._degraded_kinds), error=type(exc).__name__)
            spooled = self._spool.claim_kinds(self._degraded_kinds)
            if spooled is None:
                return None
            with self._lock:
                self.degraded_claims += 1
            return QueuedJob(row_id=spooled.row_id + SPOOL_ROW_OFFSET, job=spooled.job,
                             delivery_id=spooled.delivery_id, attempts=spooled.attempts)
        self._mark_up()
        return qjob

    def complete(self, row_id: int, **kwargs: Any) -> None:
        if row_id >= SPOOL_ROW_OFFSET:
            self._spool.complete(row_id - SPOOL_ROW_OFFSET, **kwargs)
            return
        self._primary.complete(row_id, **kwargs)

    def row_state(self, row_id: int) -> tuple[str, int] | None:
        if row_id >= SPOOL_ROW_OFFSET:
            return self._spool.row_state(row_id - SPOOL_ROW_OFFSET)
        return self._primary.row_state(row_id)

    def has_pending(self, **kwargs: Any) -> bool:
        try:
            if self._primary.has_pending(**kwargs):
                return True
        except Exception:  # noqa: BLE001 -- Postgres down: the spool is the only place a row could wait
            self._mark_down()
        return self._spool.has_pending(**kwargs)

    def recover_stale(self) -> dict[str, int]:
        """Both stores: Postgres for this machine's own rows, the spool for degraded-lane rows a
        crash left running (the ingress is the only thing that ever runs them). Postgres being
        unreachable at boot is exactly the outage the spool exists for: its rows are recovered,
        the Postgres side is logged and left to the reaper (leases expire on their own), and the
        service comes up."""
        try:
            out = self._primary.recover_stale()
            self._mark_up()
        except Exception as exc:  # noqa: BLE001 -- Postgres may be unreachable at boot; the spool still recovers
            self._mark_down()
            _log("recover_stale_pg_error", error=type(exc).__name__)
            out = {"requeued": 0, "failed": 0}
        spooled = self._spool.recover_stale()
        return {"requeued": out["requeued"] + spooled["requeued"], "failed": out["failed"] + spooled["failed"]}

    def counts(self) -> dict[str, int]:
        try:
            out = dict(self._primary.counts())
        except Exception:  # noqa: BLE001 -- diagnostics must answer while Postgres is down
            out = {}
        out["spool_rows"] = self.spool_rows()
        return out

    def recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        try:
            return self._primary.recent(limit=limit)
        except Exception:  # noqa: BLE001
            return self._spool.recent(limit=limit)

    def __getattr__(self, name: str) -> Any:  # ops helpers, events, record_check_run, ... are Postgres's
        return getattr(self._primary, name)


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float, *, on_late: Callable[[Any], None] | None = None,
                       late_gate: threading.Event | None = None, late_gate_timeout_s: float = 30.0) -> Any:
    """Run ``fn`` on a helper thread and give up after ``timeout_s`` (the connection pool's own
    timeout is the usual bound; this is the belt for the braces: a stuck TCP connect must not hold
    GitHub's 10 s delivery window).

    A result that arrives AFTER the timeout is not discarded: ``on_late(value)`` runs on the helper
    thread so the caller can reconcile what it did meanwhile (the spool). Exactly ONE side owns the
    result, decided under a lock: the helper, on finishing, either hands the value to the caller
    (the caller has not given up yet) or takes the late path; the caller, on timing out, either
    finds the value already handed over (returned normally, no ``on_late``) or marks the call
    abandoned (raises, and the helper's eventual value goes to ``on_late``). No interleaving runs
    both paths or neither. ``late_gate`` orders the late path after whatever the caller does in its
    timeout branch (the spool write): ``on_late`` runs only once the gate is set (bounded by
    ``late_gate_timeout_s`` so a caller that never sets it cannot wedge the helper)."""
    box: dict[str, Any] = {}
    done = threading.Event()
    state_lock = threading.Lock()
    state = {"abandoned": False, "handed_over": False}

    def _run() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 -- re-raised on the caller's thread (never SystemExit/KeyboardInterrupt)
            box["error"] = exc
        with state_lock:
            late = state["abandoned"]
            if not late:
                state["handed_over"] = True
        done.set()
        if late and "value" in box and on_late is not None:
            if late_gate is not None:
                late_gate.wait(late_gate_timeout_s)
            try:
                on_late(box["value"])
            except Exception:  # noqa: BLE001 -- reconciliation is best-effort; the forwarder is the backstop
                pass

    threading.Thread(target=_run, name="codna-webhook-enqueue", daemon=True).start()
    if not done.wait(timeout_s):
        with state_lock:
            if not state["handed_over"]:
                state["abandoned"] = True
        if state["abandoned"]:
            raise TimeoutError(f"queue write did not answer within {timeout_s}s")
        done.wait()  # handed over a moment ago: the helper is about to set done
    if "error" in box:
        raise box["error"]
    if "value" not in box:  # the helper died without a result (a non-Exception exit): the spool takes the delivery
        raise RuntimeError("queue write produced no result")
    return box["value"]


# --- the factory ----------------------------------------------------------------------------------
def open_queue(queue_path: str, *, backend: str | None = None, database_url: str | None = None,
               environ: Mapping[str, str] | None = None, migrate_on_open: bool = False,
               rollback_from_postgres: bool = False, **pg_kwargs: Any) -> Any:
    """Build the queue the service runs on. ``sqlite`` needs nothing new. ``shadow`` and
    ``postgres`` need a database URL (``DATABASE_URL`` from ``fly mpg attach``); without one the
    boot REFUSES rather than quietly running on SQLite under a flag that says otherwise.

    ``postgres``: the SQLite backlog is imported once (see the module docstring) and a
    :class:`SpoolingQueue` is returned with its forwarder running. ``sqlite`` with
    ``rollback_from_postgres=True`` (the boot after a flag flip back) exports Postgres's in-flight
    rows into the file first."""
    source = os.environ if environ is None else environ
    name = backend or backend_name(source)
    sqlite = WebhookQueue(queue_path)
    if name == "sqlite":
        if rollback_from_postgres:
            url = database_url or _database_url(source)
            if url:
                from .webhook_pg_queue import PostgresQueue

                pg = PostgresQueue(url, **pg_kwargs)
                try:
                    _log("backend_rollback_export", **pg.export_backlog(sqlite))
                finally:
                    pg.close()
        return sqlite
    url = database_url or _database_url(source)
    if not url:
        raise RuntimeError(f"{BACKEND_ENV}={name} needs DATABASE_URL (run `fly mpg attach`, see infra/README-webhook-deploy.md)")
    from .webhook_pg_queue import PostgresQueue

    pg = PostgresQueue(url, **pg_kwargs)
    if migrate_on_open:
        from .webhook_pg_schema import migrate

        with pg.connection() as conn:
            migrate(conn, schema=pg.schema)
    else:
        from .webhook_pg_schema import check_compatible

        with pg.connection() as conn:
            check_compatible(conn, schema=pg.schema)
    if name == "shadow":
        return ShadowQueue(sqlite, pg)
    _log("backend_reconcile_shadow_twins", **pg.reconcile_shadow_twins(sqlite))
    _log("backend_import_sqlite_backlog", **pg.import_sqlite_backlog(sqlite))
    queue = SpoolingQueue(
        pg, sqlite,
        enqueue_timeout_s=float(source.get("CODNA_WEBHOOK_PG_ENQUEUE_TIMEOUT_S", "2")),
        degraded_after_s=float(source.get("CODNA_WEBHOOK_DEGRADED_AFTER_S", "60")),
        degraded_kinds=degraded_lane_kinds(source),
    )
    queue.start_forwarder()
    return queue


def _database_url(source: Mapping[str, str]) -> str | None:
    from .webhook_pg_schema import database_url

    return database_url(source)
