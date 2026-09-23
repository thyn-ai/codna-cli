"""Postgres-backed job queue for the Codna GitHub App webhook (backend ``postgres``).

Same contract as :class:`codna.webhook_queue.WebhookQueue` -- ``enqueue`` / ``claim`` /
``complete`` / ``recover_stale`` / ``row_state`` / ``counts`` / ``recent`` -- so the worker pool
runs unchanged on either. What the shared table adds, and why a SQLite file on one volume could
not: several machines claim from ONE queue (the hard blocker on horizontal scale), every running
row carries an ``owner`` and a lease that a heartbeat renews, a reaper re-queues the job of a
machine that died mid-run, per-tenant concurrency caps and a fairness tie-break stop one noisy
installation from starving the rest, priorities age so a long-waiting fix is never starved by
strict class order, retries follow per-kind budgets with backoff (``webhook_retry``), and a job
whose transient-failure budget is spent goes to a visible ``dead`` state instead of a silent
``failed``.

Claims serialize on a Postgres advisory lock (``codna_webhook:claim``): the per-tenant cap is then
exact rather than approximate, and at ~10 claims/s even at 50x the modelled peak the ~2 ms of
serialization is irrelevant. Every timestamp is Postgres-side (``now()``), so clock skew between
machines cannot expire or extend a lease; tests inject a clock through ``clock=``.

Import-light: ``psycopg`` is imported when the first queue is opened, never at module import, so
the CLI package loads without the ``webhook`` extra.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from . import webhook_retry
from .webhook import WebhookJob, webhook_marker
from .webhook_control import SUPERSEDABLE_KINDS
from .webhook_pg_schema import DEFAULT_SCHEMA, KINDS, connect, lock_key, migrate, validate_schema_name
from .webhook_queue import QueuedJob

# A job is brought back automatically after ONE interruption (a lost lease, a restart, a drain
# that outran its bound); the second interruption of the same job is not re-run: the job is
# dead-lettered with the re-trigger hint. Same rule as webhook_resume._MAX_RESUMES.
MAX_RESUMES = 1
_NOW = "COALESCE(%(now)s::timestamptz, now())"
_JOB_COLUMNS = ("id, delivery_id, kind, repo, ref, installation_id, issue_number, pr_number, context, reason, "
                "status, priority, attempts, max_attempts, resumes, not_before, owner, lease_expires_at, "
                "check_run_id, check_started_at, posted, cancel_requested, last_error, result, created_at, "
                "claimed_at, finished_at, src_sqlite_id")
_JOB_COLUMNS_J = ", ".join("j." + c.strip() for c in _JOB_COLUMNS.split(","))  # qualified, for UPDATE ... FROM
# The kinds a newer head of the same pull request retires (webhook_queue uses the same set).
_SUPERSEDABLE_SQL = ",".join(f"'{k}'" for k in sorted(SUPERSEDABLE_KINDS))


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook-queue", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


# SQLite statuses that mean the file RAN the job to its end. Only a shadow twin whose original is in
# one of these is retired by reconcile_shadow_twins; `migrated` / `exported` are hand-offs, not runs.
SHADOW_FINISHED = frozenset({"done", "failed", "superseded", "cancelled", "dead"})


def default_owner() -> str:
    """``<machine>:<pid>``: the Fly machine id when there is one (a restart of the same machine
    then finds its predecessor's rows by prefix), else the hostname."""
    machine = os.environ.get("FLY_MACHINE_ID") or socket.gethostname() or "local"
    return f"{machine}:{os.getpid()}"


def idempotency_key(job: WebhookJob) -> str:
    """The trigger identity (repo / kind / head-or-issue), NOT the delivery id: what
    :func:`codna.webhook.webhook_marker` embeds in fix PRs, hashed."""
    return hashlib.sha256(webhook_marker(job).encode("utf-8")).hexdigest()


def _job_from_row(row: Mapping[str, Any]) -> WebhookJob:
    context = row.get("context")
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except ValueError:
            context = None
    return WebhookJob(
        kind=row["kind"], repo_full_name=row["repo"], ref=row["ref"], installation_id=row["installation_id"],
        issue_number=row["issue_number"], pr_number=row["pr_number"],
        context=context if isinstance(context, dict) else None, reason=row.get("reason") or "",
    )


@dataclass
class _Lease:
    row_id: int
    owner: str
    stop: threading.Event
    thread: threading.Thread | None = None
    lost: str | None = None


class PostgresQueue:
    """See the module docstring. One instance per process; ``owner`` identifies the machine."""

    def __init__(self, url: str, *, schema: str = DEFAULT_SCHEMA, owner: str | None = None,
                 lease_s: float = 90.0, heartbeat_s: float = 30.0,
                 retry_policy: Mapping[str, webhook_retry.RetryPolicy] | None = None,
                 clock: Callable[[], datetime] | None = None, tenant_default_cap: int = 3,
                 aging_step_s: float = 600.0, pool_min: int = 1, pool_max: int = 8,
                 on_lease_lost: Callable[[int, str], None] | None = None,
                 ensure_schema: bool = False, statement_timeout_ms: int = 10_000) -> None:
        import psycopg_pool
        from psycopg.rows import dict_row

        self._url = url
        self._schema = validate_schema_name(schema)
        self._owner = owner or default_owner()
        self._lease_s = float(lease_s)
        self._heartbeat_s = float(heartbeat_s)
        self._policy = dict(retry_policy) if retry_policy is not None else dict(webhook_retry.RETRY_POLICY)
        self._clock = clock
        self._default_cap = int(tenant_default_cap)
        self._aging_step_s = float(aging_step_s)
        self._on_lease_lost = on_lease_lost
        self._leases: dict[int, _Lease] = {}
        self._leases_lock = threading.Lock()
        self.draining = False  # set by the worker registration when the scaler asks; claim() returns None
        # statement_timeout and application_name are applied with SET on every new connection, not
        # as the libpq `options` startup parameter: Fly Managed Postgres hands out its PgBouncer
        # endpoint, and PgBouncer refuses that parameter outright ("FATAL: unsupported startup
        # parameter in options: statement_timeout", measured 2026-09-20 from the ingress machine),
        # which would have refused every pooled connection at boot. In PgBouncer's session mode
        # (Fly's default, and the mode the leader locks need) a SET holds for the connection's life.
        timeout_ms = int(statement_timeout_ms)

        def _configure(conn: Any) -> None:
            conn.execute(f"SET statement_timeout = {timeout_ms}")
            conn.execute("SET application_name = 'codna-webhook'")

        self._pool = psycopg_pool.ConnectionPool(
            url, min_size=max(1, pool_min), max_size=max(pool_min, pool_max), open=True, timeout=10.0,
            kwargs={"autocommit": True, "row_factory": dict_row, "connect_timeout": 5}, configure=_configure,
        )
        if ensure_schema:
            with self._pool.connection() as conn:
                migrate(conn, schema=self._schema, log=lambda _m: None)

    # --- plumbing ----------------------------------------------------------------------------
    @property
    def schema(self) -> str:
        return self._schema

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def path(self) -> str:
        """Kept for callers that expect the SQLite file's attribute (the registry dir derives from
        it): the connection URL with any password removed."""
        return _redact_url(self._url)

    def _now(self) -> datetime | None:
        return self._clock() if self._clock is not None else None

    def now(self) -> datetime | None:
        """The injected clock's reading, or None for Postgres-side ``now()``. Shared with the ops
        read models so every query in a process agrees on what "now" is (tests inject a clock)."""
        return self._now()

    def active_leases(self) -> list[tuple[int, str]]:
        """``(row_id, owner)`` for every lease this process currently holds -- the truth about how
        many slots are busy (worker heartbeats) and what a drain has to wait for."""
        with self._leases_lock:
            return [(lease.row_id, lease.owner) for lease in self._leases.values()]

    def _t(self, table: str) -> str:
        return f"{self._schema}.{table}"

    def connection(self):
        """A pooled autocommit connection (context manager)."""
        return self._pool.connection()

    def dedicated_connection(self):
        """A connection OUTSIDE the pool for the leader election's transaction-held advisory locks
        (webhook_pg_ops.LeaderLock): a pooled connection could be handed to another caller while
        the lock is meant to be held."""
        from psycopg.rows import dict_row

        return connect(self._url, row_factory=dict_row)

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001 -- the answer IS the exception
            return False

    def close(self) -> None:
        with self._leases_lock:
            leases = list(self._leases.values())
            self._leases.clear()
        for lease in leases:
            lease.stop.set()
        try:
            self._pool.close()
        except Exception:  # noqa: BLE001
            pass

    def event(self, job_id: int | None, event: str, *, actor: str | None = None,
              detail: Mapping[str, Any] | None = None, conn: Any = None) -> None:
        """Append to the audit trail. Every operator mutation and every state transition lands here."""
        params = (job_id, event, actor or self._owner, json.dumps(dict(detail or {}), default=str))
        sql = f"INSERT INTO {self._t('job_events')} (job_id, event, actor, detail) VALUES (%s, %s, %s, %s::jsonb)"
        if conn is not None:
            conn.execute(sql, params)
            return
        with self._pool.connection() as c:
            c.execute(sql, params)

    def policy_for(self, kind: str) -> webhook_retry.RetryPolicy:
        return self._policy.get(kind) or webhook_retry.policy_for(kind)

    # --- enqueue -------------------------------------------------------------------------------
    def enqueue(self, job: WebhookJob, *, delivery_id: str | None, priority: int = 0,
                unless_pending: bool = False, created_at: datetime | None = None,
                src_sqlite_id: int | None = None, attempts: int = 0, resumes: int = 0,
                not_before: datetime | None = None, check_run_id: int | None = None) -> bool:
        """Insert a job. False when it collapses onto an existing row: the same ``delivery_id``
        (a redelivery), a queued/running review or fix for the SAME head (``jobs_one_live_head``:
        a new GUID for a head already covered), a CI-failure fix for a (repo, ref) that already
        had one in the last 24 h (the rule ported line for line from the SQLite queue), or -- with
        ``unless_pending`` (the worker re-queueing a head that moved under a review, codna#569, and
        the reaper queueing the current head of a reaped job) -- any job of this kind for exactly
        this head that is waiting, running OR already finished: a head that has its review does not
        get a second one because an older head's job noticed, as it ended, that the head had moved
        (the older job is normally cancelled first; this closes the window in which it is not).

        ``check_run_id`` is the Check Run the ingress already opened ``queued`` for this job (see
        ``webhook_queued_check``); the row is born owning it, ``check_started_at`` is stamped now
        (the SLO clock: the tenant can see the check from this moment), and the worker that claims
        the row updates that run instead of creating one."""
        if job.kind not in KINDS:
            raise ValueError(f"unknown job kind {job.kind!r}")
        if delivery_id is None:
            material = f"{job.kind}|{job.repo_full_name}|{job.ref}|{job.pr_number}|{job.issue_number}|{job.reason}"
            delivery_id = "synth-" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        now = self._now()
        with self._pool.connection() as conn:
            if job.reason == "check_suite_failure" and job.ref:
                dup = conn.execute(
                    f"SELECT 1 FROM {self._t('jobs')} WHERE kind='fix' AND repo=%(repo)s AND ref=%(ref)s "
                    f"AND reason=%(reason)s AND (status IN ('queued','running') OR created_at >= {_NOW} - interval '24 hours') LIMIT 1",
                    {"repo": job.repo_full_name, "ref": job.ref, "reason": job.reason, "now": now},
                ).fetchone()
                if dup is not None:
                    return False
            with conn.transaction():
                if unless_pending and job.ref:
                    # Two requeuers can race for one head (two reapers servicing two reaped jobs of
                    # the same pull request; a mover and a reaper). For queued/running rows the
                    # jobs_one_live_head index already decides the race; a `done` row has no index,
                    # so the check and the insert are serialized per head: same-head requeues take
                    # this lock in turn, everything else is untouched.
                    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                                 (lock_key(self._schema, f"head:{job.repo_full_name}|{job.pr_number}|{job.ref}|{job.kind}"),))
                    if self._covered_within(conn, repo=job.repo_full_name, pr_number=job.pr_number, ref=job.ref,
                                            kind=job.kind, now=now):
                        return False
                row = conn.execute(
                    f"INSERT INTO {self._t('jobs')} (delivery_id, idempotency_key, kind, repo, ref, installation_id, "
                    f"issue_number, pr_number, context, reason, priority, attempts, max_attempts, resumes, not_before, "
                    f"check_run_id, check_started_at, created_at, src_sqlite_id) "
                    f"VALUES (%(delivery_id)s, %(key)s, %(kind)s, %(repo)s, %(ref)s, %(installation_id)s, %(issue_number)s, "
                    f"%(pr_number)s, %(context)s::jsonb, %(reason)s, %(priority)s, %(attempts)s, %(max_attempts)s, "
                    f"%(resumes)s, %(not_before)s, %(check_run_id)s, "
                    f"CASE WHEN %(check_run_id)s::bigint IS NULL THEN NULL ELSE {_NOW} END, "
                    f"COALESCE(%(created_at)s::timestamptz, {_NOW}), %(src)s) "
                    f"ON CONFLICT DO NOTHING RETURNING id",
                    {
                        "delivery_id": delivery_id, "key": idempotency_key(job), "kind": job.kind,
                        "repo": job.repo_full_name, "ref": job.ref, "installation_id": job.installation_id,
                        "issue_number": job.issue_number, "pr_number": job.pr_number,
                        "context": json.dumps(job.context) if job.context is not None else None,
                        "reason": job.reason or "", "priority": int(priority), "attempts": int(attempts),
                        "max_attempts": self.policy_for(job.kind).budget, "resumes": int(resumes),
                        "not_before": not_before, "check_run_id": check_run_id, "created_at": created_at,
                        "now": now, "src": src_sqlite_id,
                    },
                ).fetchone()
                if row is None:
                    return False
                new_id = int(row["id"])
                self.event(new_id, "enqueued", conn=conn,
                           detail={"delivery_id": delivery_id, "kind": job.kind, "reason": job.reason,
                                   "priority": int(priority), "imported_from_sqlite": src_sqlite_id is not None,
                                   "check_run_id": check_run_id})
                if job.kind in SUPERSEDABLE_KINDS and job.pr_number is not None and job.ref:
                    # A new head of this pull request retires the WAITING older heads in the same
                    # transaction as its own insert -- the SQLite queue's rule (webhook_queue
                    # ``_supersede_within``), which this backend lacked: with the queue saturated,
                    # push A then push B had A's job run later, read the moved head, review B, and
                    # B's own job review B again (two `codna review` runs on one commit, measured
                    # 2026-09-20). The RUNNING older heads are flagged ``cancel_requested`` so the
                    # worker holding them -- on ANY machine, not only the ingress's own process --
                    # stops at its next heartbeat (webhook_service wires the lost lease into the
                    # same cancellation the ingress applies in-process). The row just written is
                    # excluded: a redelivery of the SAME head must not retire its predecessor.
                    self._supersede_within(conn, repo=job.repo_full_name, pr_number=job.pr_number,
                                           new_ref=job.ref, exclude_row_id=new_id, now=now)
                    self._request_cancel_within(conn, repo=job.repo_full_name, pr_number=job.pr_number,
                                                new_ref=job.ref, exclude_row_id=new_id)
        return True

    def _pending_within(self, conn: Any, *, repo: str, pr_number: int | None, ref: str | None, kind: str) -> bool:
        row = conn.execute(
            f"SELECT 1 FROM {self._t('jobs')} WHERE repo = %s AND pr_number IS NOT DISTINCT FROM %s AND ref = %s "
            f"AND kind = %s AND status IN ('queued', 'running') LIMIT 1", (repo, pr_number, ref, kind)).fetchone()
        return row is not None

    def _covered_within(self, conn: Any, *, repo: str, pr_number: int | None, ref: str | None, kind: str,
                        now: datetime | None) -> bool:
        """``_pending_within`` plus rows that ran to a verdict (``done``) within the last 24 hours --
        what ``enqueue``'s ``unless_pending`` asks. The window this guards is minutes wide (an older
        head's job outliving the new head's review by at most the job timeout plus a lease), and
        the bound keeps an old verdict from silencing a requeue of a head force-pushed back months
        later; 24 h is the check_suite_failure dedup rule's window. A ``failed``/``dead``/
        ``cancelled`` row does not cover the head."""
        row = conn.execute(
            f"SELECT 1 FROM {self._t('jobs')} WHERE repo = %(repo)s AND pr_number IS NOT DISTINCT FROM %(pr)s AND ref = %(ref)s "
            f"AND kind = %(kind)s AND (status IN ('queued', 'running') "
            f"OR (status = 'done' AND finished_at >= {_NOW} - interval '24 hours')) LIMIT 1",
            {"repo": repo, "pr": pr_number, "ref": ref, "kind": kind, "now": now}).fetchone()
        return row is not None

    def has_pending(self, *, repo: str, pr_number: int | None, ref: str | None, kind: str) -> bool:
        """Whether a job of ``kind`` for exactly this pull request head is already waiting or
        running -- the SQLite queue's question, answered from the shared table. ``ref`` None never
        matches (a `@codna review` comment has no head)."""
        if not ref:
            return False
        with self._pool.connection() as conn:
            return self._pending_within(conn, repo=repo, pr_number=pr_number, ref=ref, kind=kind)

    # --- the Check Run the ingress opens at enqueue (webhook_queued_check) -----------------------
    def has_delivery(self, delivery_id: str | None) -> bool:
        """Whether a row for this ``delivery_id`` already exists, whatever its state. The ingress
        asks before opening a queued Check Run for a delivery: a redelivery of an already-processed
        GUID collapses in ``enqueue`` and must not leave a second run on the commit."""
        if not delivery_id:
            return False
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT 1 FROM {self._t('jobs')} WHERE delivery_id = %s LIMIT 1", (delivery_id,)).fetchone()
        return row is not None

    def check_run_bound(self, delivery_id: str | None) -> int | None:
        """The Check Run id the row for this delivery carries, or None (no row, or none bound). The
        ingress reads it right after ``enqueue`` to learn whether the run it opened is owned by a
        row -- anything else is an orphan it must close at once."""
        if not delivery_id:
            return None
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT check_run_id FROM {self._t('jobs')} WHERE delivery_id = %s", (delivery_id,)).fetchone()
        return int(row["check_run_id"]) if row and row.get("check_run_id") else None

    def adopt_check_run(self, job: WebhookJob, check_run_id: int, *, delivery_id: str | None = None) -> bool:
        """Hand an already-open Check Run to the WAITING row for exactly this head (or, with
        ``delivery_id``, to this delivery's own waiting row) when that row carries none yet. True when
        a row took it: the worker that claims it will update this run instead of creating one.

        Two callers: the ingress whose insert collapsed onto a live row for the same head (a
        second delivery for one commit) -- if that row was written without a run (by a worker's or
        the reaper's requeue, or an older ingress) it adopts this one, and nothing is orphaned; and
        the late path of a pre-create that outran the ingress's time budget, which adopts onto the
        delivery's own row if no worker has claimed it. Conditional on ``status = 'queued'`` and
        ``check_run_id IS NULL``, under the row lock the UPDATE takes, so a claim that lands first
        wins and the caller closes the run instead."""
        if not check_run_id:
            return False
        if delivery_id:
            where, params = "delivery_id = %(d)s", {"d": delivery_id}
        else:
            if not job.ref:
                return False
            where = "repo = %(repo)s AND kind = %(kind)s AND pr_number IS NOT DISTINCT FROM %(pr)s AND ref = %(ref)s"
            params = {"repo": job.repo_full_name, "kind": job.kind, "pr": job.pr_number, "ref": job.ref}
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"UPDATE {self._t('jobs')} SET check_run_id = %(cid)s, check_started_at = COALESCE(check_started_at, {_NOW}) "
                f"WHERE {where} AND status = 'queued' AND check_run_id IS NULL RETURNING id",
                {**params, "cid": int(check_run_id), "now": self._now()},
            ).fetchone()
            if row is None:
                return False
            self.event(int(row["id"]), "check_run_adopted", conn=conn, detail={"check_run_id": int(check_run_id)})
        return True

    # --- claim + lease -------------------------------------------------------------------------
    def claim(self, *, reserve_priority: bool = False, classes: Iterable[int] | None = None,
              lease_s: float | None = None) -> QueuedJob | None:
        """Atomically claim the next runnable job for this owner, or None.

        Runnable: ``queued``, due (``not_before``), under its attempt budget, tenant not paused
        and under its concurrency cap. Order: class after AGING (a job loses one class per
        ``aging_step_s`` waited, floored at 0: a fix waiting 30 min competes with fresh reviews),
        then ``priority`` DESC, then the least-loaded tenant per unit of ``weight``, then arrival.
        ``reserve_priority`` (the pool's last-free-thread rule) tries the merge-gating classes
        [0, 1] first and falls back to everything only when none is waiting. Both selects and the
        UPDATE run under one advisory lock, so the tenant cap is exact across machines."""
        if self.draining:
            return None
        thread_owner = f"{self._owner}:{threading.get_ident()}"
        lease = float(lease_s if lease_s is not None else self._lease_s)
        wanted = list(classes) if classes is not None else None
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key(self._schema, "claim"),))
            row = None
            if reserve_priority and wanted is None:
                row = self._claim_one(conn, classes=[0, 1], owner=thread_owner, lease_s=lease)
            if row is None:
                row = self._claim_one(conn, classes=wanted, owner=thread_owner, lease_s=lease)
            if row is None:
                return None
            self.event(int(row["id"]), "claimed", conn=conn,
                       detail={"owner": thread_owner, "attempt": int(row["attempts"]), "lease_s": lease})
        qjob = QueuedJob(row_id=int(row["id"]), delivery_id=row["delivery_id"], attempts=int(row["attempts"]),
                         job=_job_from_row(row),
                         check_run_id=int(row["check_run_id"]) if row.get("check_run_id") else None)
        self._start_lease(qjob.row_id, thread_owner)
        return qjob

    def _claim_one(self, conn: Any, *, classes: list[int] | None, owner: str, lease_s: float) -> Mapping[str, Any] | None:
        jobs, tenants = self._t("jobs"), self._t("tenants")
        return conn.execute(
            f"""
            WITH running AS (
                SELECT installation_id, count(*) AS n FROM {jobs} WHERE status = 'running' GROUP BY installation_id
            ), cand AS (
                SELECT j.id FROM {jobs} j
                LEFT JOIN {tenants} t ON t.installation_id = j.installation_id
                LEFT JOIN running r ON r.installation_id IS NOT DISTINCT FROM j.installation_id
                WHERE j.status = 'queued'
                  AND (j.not_before IS NULL OR j.not_before <= {_NOW})
                  AND j.attempts < j.max_attempts
                  AND NOT COALESCE(t.paused, false)
                  AND COALESCE(r.n, 0) < COALESCE(t.max_concurrency, %(default_cap)s)
                  AND (%(classes)s::smallint[] IS NULL OR j.class = ANY(%(classes)s::smallint[]))
                ORDER BY GREATEST(j.class - floor(EXTRACT(EPOCH FROM ({_NOW} - j.created_at)) / %(aging)s), 0),
                         j.priority DESC,
                         COALESCE(r.n, 0)::float / GREATEST(COALESCE(t.weight, 1), 1),
                         j.id
                LIMIT 1
                FOR UPDATE OF j SKIP LOCKED
            )
            UPDATE {jobs} j
               SET status = 'running', attempts = j.attempts + 1, claimed_at = {_NOW}, owner = %(owner)s,
                   lease_expires_at = {_NOW} + make_interval(secs => %(lease)s), cancel_requested = NULL,
                   finished_at = NULL
              FROM cand
             WHERE j.id = cand.id
            RETURNING {_JOB_COLUMNS_J}
            """,
            {"now": self._now(), "default_cap": self._default_cap, "classes": classes,
             "aging": self._aging_step_s, "owner": owner, "lease": lease_s},
        ).fetchone()

    def _start_lease(self, row_id: int, owner: str) -> None:
        lease = _Lease(row_id=row_id, owner=owner, stop=threading.Event())
        with self._leases_lock:
            self._leases[row_id] = lease
        if self._heartbeat_s <= 0:
            return  # tests drive heartbeat() by hand
        lease.thread = threading.Thread(target=self._keep_lease, args=(lease,),
                                        name=f"codna-webhook-lease-{row_id}", daemon=True)
        lease.thread.start()

    def _keep_lease(self, lease: _Lease) -> None:
        while not lease.stop.wait(self._heartbeat_s):
            try:
                alive = self.heartbeat(lease.row_id, owner=lease.owner)
            except Exception as exc:  # noqa: BLE001 -- a DB blip is not a lost lease; the next beat re-asserts it
                _log("lease_heartbeat_error", row_id=lease.row_id, error=type(exc).__name__)
                continue
            if alive:
                continue
            reason = self._cancel_reason(lease.row_id) or "lease_lost"
            lease.lost = reason
            _log("lease_lost", row_id=lease.row_id, owner=lease.owner, reason=reason)
            if self._on_lease_lost is not None:
                try:
                    self._on_lease_lost(lease.row_id, reason)
                except Exception as exc:  # noqa: BLE001 -- the callback must never kill the keeper
                    _log("lease_lost_callback_error", row_id=lease.row_id, error=type(exc).__name__)
            return

    def heartbeat(self, row_id: int, *, owner: str | None = None) -> bool:
        """Renew this owner's lease. False when the row is no longer ours to renew: reaped and
        re-queued (``lease_lost``), cancelled by an operator or a superseding head
        (``cancel_requested`` set), or already terminal."""
        if owner is None:
            with self._leases_lock:
                lease = self._leases.get(row_id)
            owner = lease.owner if lease is not None else f"{self._owner}:{threading.get_ident()}"
        with self._pool.connection() as conn:
            row = conn.execute(
                f"UPDATE {self._t('jobs')} SET lease_expires_at = {_NOW} + make_interval(secs => %(lease)s) "
                f"WHERE id = %(id)s AND owner = %(owner)s AND status = 'running' AND cancel_requested IS NULL RETURNING 1",
                {"now": self._now(), "lease": self._lease_s, "id": row_id, "owner": owner},
            ).fetchone()
        return row is not None

    def _cancel_reason(self, row_id: int) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT cancel_requested FROM {self._t('jobs')} WHERE id = %s", (row_id,)).fetchone()
        return (row or {}).get("cancel_requested") if row else None

    def set_lease_lost_handler(self, callback: Callable[[int, str], None] | None) -> None:
        """Install (or clear) what the lease keeper calls when a lease can no longer be renewed:
        the row was cancelled from outside (a newer head, an operator) or reaped. The service wires
        the worker's own cancellation here once the queue exists (``webhook_service.build_service``)."""
        self._on_lease_lost = callback

    def lease_lost_reason(self, row_id: int) -> str | None:
        """What the heartbeat recorded for a lease this process no longer holds, else None."""
        with self._leases_lock:
            lease = self._leases.get(row_id)
        return lease.lost if lease is not None else None

    def abandon_leases(self) -> None:
        """Stop renewing every lease WITHOUT completing the rows -- what a crash does. Tests use it
        to make a live worker look dead; the service never calls it."""
        with self._leases_lock:
            leases = list(self._leases.values())
            self._leases.clear()
        for lease in leases:
            lease.stop.set()

    # --- complete ------------------------------------------------------------------------------
    def complete(self, row_id: int, *, status: str, result: Mapping[str, object] | None = None,
                 retry: bool = True, retry_after_s: float | None = None) -> None:
        """Mark a claimed job terminal, or hand it back for another attempt.

        ``done`` ends it. ``failed`` with ``retry=False`` (the worker knows the failure is a
        property of the inputs) ends it as ``failed``. ``failed`` with ``retry=True`` becomes
        ``queued`` again with ``not_before`` = now + ``retry_after_s`` (when the worker named the
        wait: the account bridge) or the kind's backoff -- until the budget is spent, when the
        job is ``dead`` (transient) or ``failed`` (the text names a deterministic cause). A row
        this process no longer owns (its lease was reaped, or it was cancelled) is left exactly as
        the reaper or operator put it: the outcome of a zombie attempt must not overwrite theirs."""
        with self._leases_lock:
            lease = self._leases.pop(row_id, None)
        if lease is not None:
            lease.stop.set()
        owner = lease.owner if lease is not None else None
        payload = dict(result or {})
        summary = str(payload.get("summary") or payload.get("message") or payload.get("error") or "")
        code = str(payload.get("error") or "") or None
        now = self._now()
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"SELECT id, kind, attempts, max_attempts, owner, status FROM {self._t('jobs')} WHERE id = %s FOR UPDATE",
                (row_id,),
            ).fetchone()
            if row is None:
                return
            # Ours to complete only while it is still running under THIS process: the exact lease
            # owner when this instance still holds the lease record, else any thread of this
            # process (owner prefix) -- never another machine's row, whatever the caller believes.
            ours = (row["owner"] == owner) if owner is not None else str(row["owner"] or "").startswith(self._owner + ":")
            if row["status"] != "running" or not ours:
                self.event(row_id, "complete_ignored", conn=conn,
                           detail={"status": status, "row_status": row["status"], "row_owner": row["owner"],
                                   "caller_owner": owner, "summary": summary[:300]})
                _log("complete_ignored", row_id=row_id, row_status=row["status"], caller_owner=owner)
                return
            final, hold_s, event = status, None, "completed"
            if status == "failed":
                deterministic = (not retry) or webhook_retry.classify(summary, error_code=code) == "deterministic"
                if deterministic:
                    final, event = "failed", "failed"
                elif int(row["attempts"]) < int(row["max_attempts"]):
                    final, event = "queued", "retry"
                    hold_s = (float(retry_after_s) if retry_after_s is not None
                              else _backoff_from(self.policy_for(row["kind"]), int(row["attempts"])))
                else:
                    final, event = "dead", "dead"
            conn.execute(
                f"UPDATE {self._t('jobs')} SET status = %(final)s, finished_at = CASE WHEN %(final)s = 'queued' THEN NULL ELSE {_NOW} END, "
                f"result = %(result)s::jsonb, last_error = %(last_error)s, owner = NULL, lease_expires_at = NULL, "
                f"not_before = CASE WHEN %(hold)s::float IS NULL THEN NULL ELSE {_NOW} + make_interval(secs => %(hold)s) END "
                f"WHERE id = %(id)s",
                {"final": final, "now": now, "result": json.dumps(payload, default=str),
                 "last_error": summary[:2000] if final != "done" else None, "hold": hold_s, "id": row_id},
            )
            self.event(row_id, event, conn=conn, detail={"attempt": int(row["attempts"]), "max_attempts": int(row["max_attempts"]),
                                                          "hold_s": hold_s, "summary": summary[:300]})

    # --- recovery ------------------------------------------------------------------------------
    def recover_stale(self) -> dict[str, int]:
        """On boot: resolve the rows THIS machine's previous process left ``running`` (owner prefix
        match; other machines' rows are alive and belong to the reaper). Under the attempt cap and
        never interrupted before -> ``queued`` with ``resumes + 1``; interrupted once already ->
        ``dead`` (``interrupted_after_resume``); out of attempts -> ``failed``
        (``orphaned_after_max_attempts``). Returns {"requeued": N, "failed": M} like SQLite."""
        prefix = self._owner.split(":", 1)[0]
        now = self._now()
        with self._pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                f"SELECT id, attempts, max_attempts, resumes FROM {self._t('jobs')} "
                f"WHERE status = 'running' AND owner LIKE %s FOR UPDATE SKIP LOCKED",
                (prefix + ":%",),
            ).fetchall()
            requeued = failed = 0
            for row in rows:
                rid = int(row["id"])
                if int(row["attempts"]) >= int(row["max_attempts"]):
                    conn.execute(
                        f"UPDATE {self._t('jobs')} SET status='failed', finished_at={_NOW}, owner=NULL, lease_expires_at=NULL, "
                        f"last_error='orphaned_after_max_attempts', result=%(r)s::jsonb WHERE id=%(id)s",
                        {"now": now, "r": json.dumps({"error": "orphaned_after_max_attempts"}), "id": rid})
                    self.event(rid, "failed", conn=conn, detail={"error": "orphaned_after_max_attempts", "by": "recover_stale"})
                    failed += 1
                elif int(row["resumes"]) >= MAX_RESUMES:
                    conn.execute(
                        f"UPDATE {self._t('jobs')} SET status='dead', finished_at={_NOW}, owner=NULL, lease_expires_at=NULL, "
                        f"last_error='interrupted_after_resume', result=%(r)s::jsonb WHERE id=%(id)s",
                        {"now": now, "r": json.dumps({"error": "interrupted_after_resume", "attempt": int(row["attempts"])}), "id": rid})
                    self.event(rid, "dead", conn=conn, detail={"error": "interrupted_after_resume", "by": "recover_stale"})
                    failed += 1
                else:
                    conn.execute(
                        f"UPDATE {self._t('jobs')} SET status='queued', owner=NULL, lease_expires_at=NULL, resumes=resumes+1, "
                        f"last_error='interrupted:restart' WHERE id=%s", (rid,))
                    self.event(rid, "interrupted", conn=conn, detail={"cause": "restart", "by": "recover_stale"})
                    requeued += 1
        return {"requeued": requeued, "failed": failed}

    # --- reads -----------------------------------------------------------------------------------
    def row_state(self, row_id: int) -> tuple[str, int] | None:
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT status, attempts FROM {self._t('jobs')} WHERE id = %s", (row_id,)).fetchone()
        if row is None:
            return None
        status = "retry" if (row["status"] == "queued" and int(row["attempts"]) > 0) else str(row["status"])
        return status, int(row["attempts"])

    def counts(self) -> dict[str, int]:
        """By status, in the SQLite vocabulary: a queued row that already ran once reports as
        ``retry`` so dashboards and the parity tests read the same on both backends."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT CASE WHEN status='queued' AND attempts > 0 THEN 'retry' ELSE status END AS s, count(*) AS c "
                f"FROM {self._t('jobs')} GROUP BY 1").fetchall()
        return {str(r["s"]): int(r["c"]) for r in rows}

    def depth_by_kind(self) -> dict[str, int]:
        """How many jobs of each kind are runnable right now (queued, due, under budget)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT kind, count(*) AS c FROM {self._t('jobs')} WHERE status='queued' "
                f"AND (not_before IS NULL OR not_before <= {_NOW}) AND attempts < max_attempts GROUP BY kind",
                {"now": self._now()}).fetchall()
        return {str(r["kind"]): int(r["c"]) for r in rows}

    def recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 50))
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT id, delivery_id, kind, repo, ref, installation_id, reason, "
                f"CASE WHEN status='queued' AND attempts > 0 THEN 'retry' ELSE status END AS status, attempts, max_attempts, "
                f"resumes, owner, priority, created_at, claimed_at, finished_at, result, last_error "
                f"FROM {self._t('jobs')} ORDER BY id DESC LIMIT %s", (safe_limit,)).fetchall()
        return [_public_row(r) for r in rows]

    def job(self, row_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT {_JOB_COLUMNS} FROM {self._t('jobs')} WHERE id = %s", (row_id,)).fetchone()
        return _public_row(row) if row else None

    def events(self, row_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT id, at, event, actor, detail FROM {self._t('job_events')} WHERE job_id = %s ORDER BY id LIMIT %s",
                (row_id, max(1, min(int(limit), 1000)))).fetchall()
        return [{"id": int(r["id"]), "at": _iso(r["at"]), "event": r["event"], "actor": r["actor"],
                 "detail": r["detail"] if isinstance(r["detail"], dict) else {}} for r in rows]

    # --- per-row bookkeeping the worker and the reaper share -------------------------------------
    def record_check_run(self, row_id: int, check_run_id: int | None) -> None:
        """The job's Check Run exists: keep its id on the row so a reaper on ANOTHER machine can
        complete it honestly, and stamp when it appeared (the SLO clock stops here)."""
        if not check_run_id:
            return
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {self._t('jobs')} SET check_run_id = %(cid)s, check_started_at = COALESCE(check_started_at, {_NOW}) "
                f"WHERE id = %(id)s", {"cid": int(check_run_id), "now": self._now(), "id": row_id})

    def mark_posted(self, row_id: int, key: str) -> bool:
        """Set ``posted.<key>`` exactly once. True when THIS call set it (the caller may post);
        False when it was already set (a duplicate attempt must not post again)."""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"UPDATE {self._t('jobs')} SET posted = posted || jsonb_build_object(%(k)s, {_NOW}) "
                f"WHERE id = %(id)s AND NOT (posted ? %(k)s) RETURNING 1",
                {"k": key, "now": self._now(), "id": row_id}).fetchone()
        return row is not None

    def supersede_queued(self, *, repo: str, pr_number: int | None, new_ref: str | None,
                         exclude_row_id: int | None = None) -> list[int]:
        """Retire WAITING review/fix rows for this pull request whose head is no longer current
        (the queue keeps the newest head only). Same contract as the SQLite method."""
        if not repo or pr_number is None or not new_ref:
            return []
        with self._pool.connection() as conn, conn.transaction():
            return self._supersede_within(conn, repo=repo, pr_number=pr_number, new_ref=new_ref,
                                          exclude_row_id=exclude_row_id, now=self._now())

    def _supersede_within(self, conn: Any, *, repo: str, pr_number: int, new_ref: str,
                          exclude_row_id: int | None, now: datetime | None) -> list[int]:
        """The retirement itself, inside a transaction the CALLER owns -- so ``enqueue`` makes the
        insert and the retirement of the heads it replaces one atomic step."""
        rows = conn.execute(
            f"UPDATE {self._t('jobs')} SET status='superseded', finished_at={_NOW}, "
            f"result=%(r)s::jsonb, last_error='superseded' WHERE repo=%(repo)s AND pr_number=%(pr)s "
            f"AND kind IN ({_SUPERSEDABLE_SQL}) AND status='queued' AND ref IS NOT NULL AND ref <> %(ref)s "
            f"AND id <> %(ex)s RETURNING id",
            {"now": now, "r": json.dumps({"summary": f"superseded by {new_ref[:8]}"}),
             "repo": repo, "pr": pr_number, "ref": new_ref, "ex": -1 if exclude_row_id is None else exclude_row_id},
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        for rid in ids:
            self.event(rid, "superseded", conn=conn, detail={"new_ref": new_ref})
        return ids

    def _request_cancel_within(self, conn: Any, *, repo: str, pr_number: int, new_ref: str,
                               exclude_row_id: int) -> list[int]:
        """Ask the worker running an older head of this pull request to stop: ``cancel_requested``
        fails its next heartbeat (``heartbeat``), and the lease keeper's callback cancels the job
        where it runs. Rows already asked keep their first reason; the reaper finishes whichever
        row its worker does not (``webhook_lease.Reaper._reap_one``)."""
        rows = conn.execute(
            f"UPDATE {self._t('jobs')} SET cancel_requested = %(why)s WHERE repo=%(repo)s AND pr_number=%(pr)s "
            f"AND kind IN ({_SUPERSEDABLE_SQL}) AND status='running' AND ref IS NOT NULL AND ref <> %(ref)s "
            f"AND cancel_requested IS NULL AND id <> %(ex)s RETURNING id",
            {"why": f"superseded by {new_ref[:8]}", "repo": repo, "pr": pr_number, "ref": new_ref, "ex": exclude_row_id},
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        for rid in ids:
            self.event(rid, "cancel_requested", conn=conn, detail={"reason": "superseded", "new_ref": new_ref})
        return ids

    # --- migration to and from the SQLite file --------------------------------------------------
    def reconcile_shadow_twins(self, sqlite_queue: Any) -> dict[str, int]:
        """Retire the ``queued`` rows the shadow phase wrote for deliveries the SQLite file has
        already finished. On ``shadow`` every accepted delivery gets a Postgres twin that nobody
        claims (SQLite runs the job), so at the boot that makes Postgres authoritative those twins
        would be claimed and run AGAIN: a second review on an already-reviewed head, a second fix
        PR (measured 2026-09-20: the first shadow review's twin sat ``queued`` in the cluster with
        its age past 800 s). Runs before :meth:`import_sqlite_backlog`, on every postgres boot.

        Only a row whose SQLite original the file FINISHED ITSELF (``SHADOW_FINISHED``) is
        retired. Everything else is kept: an original still in flight (queued / retry / running)
        is the backlog; ``migrated`` means a previous postgres boot handed the row over -- it is
        queued in Postgres because no worker has claimed it yet, not because SQLite ran it (the
        second boot must not drop it); ``exported`` is a rollback's hand-off; a row the file never
        saw was never a twin. One transaction retires them all. Idempotent."""
        with self._pool.connection() as conn:
            rows = conn.execute(f"SELECT id, delivery_id FROM {self._t('jobs')} WHERE status = 'queued' ORDER BY id").fetchall()
        out = {"reconciled": 0, "kept_in_flight": 0, "kept_handed_off": 0, "no_counterpart": 0}
        if not rows:
            return out
        statuses = sqlite_queue.statuses_for([str(r["delivery_id"]) for r in rows])
        to_retire: list[tuple[int, str]] = []
        for r in rows:
            st = statuses.get(str(r["delivery_id"]))
            if st is None:
                out["no_counterpart"] += 1
            elif st in SHADOW_FINISHED:
                to_retire.append((int(r["id"]), st))
            elif st in ("queued", "retry", "running"):
                out["kept_in_flight"] += 1
            else:
                out["kept_handed_off"] += 1
        if not to_retire:
            return out
        with self._pool.connection() as conn, conn.transaction():
            for row_id, st in to_retire:
                done = conn.execute(
                    f"UPDATE {self._t('jobs')} SET status='cancelled', finished_at={_NOW}, last_error=%(why)s, "
                    f"result=%(r)s::jsonb WHERE id=%(id)s AND status='queued'",
                    {"now": self._now(), "why": f"shadow_twin:{st}", "id": row_id,
                     "r": json.dumps({"summary": f"shadow twin retired: the SQLite file already finished this delivery ({st})"})},
                ).rowcount
                if done:
                    self.event(row_id, "cancelled", actor="backend:shadow-reconcile",
                               detail={"reason": "shadow_twin", "sqlite_status": st}, conn=conn)
                    out["reconciled"] += 1
        return out

    def import_sqlite_backlog(self, sqlite_queue: Any) -> dict[str, int]:
        """Move every SQLite row still in flight (queued / retry / running) into Postgres, once.

        Idempotent (``ON CONFLICT (delivery_id) DO NOTHING``, and the SQLite row is marked
        ``migrated`` only after its twin exists). A ``running`` row is imported as ``queued`` with
        ``resumes + 1`` -- the process that was running it is gone -- and its Check Run id (from the
        JSON registry beside the file, when present) travels with it so the reaper can close the
        stale run. ``attempts`` and ``not_before`` carry over."""
        moved = skipped = 0
        for row in sqlite_queue.export_rows():
            job = row["job"]
            not_before = None
            if row.get("not_before"):
                try:
                    not_before = datetime.fromisoformat(str(row["not_before"]))
                except ValueError:
                    not_before = None
            created = None
            if row.get("created_at"):
                try:
                    created = datetime.fromisoformat(str(row["created_at"]))
                except ValueError:
                    created = None
            interrupted = row["status"] == "running"
            inserted = self.enqueue(
                job, delivery_id=row["delivery_id"], created_at=created, src_sqlite_id=int(row["id"]),
                attempts=int(row["attempts"]), resumes=1 if interrupted else 0, not_before=not_before,
                check_run_id=row.get("check_run_id"),
            )
            if inserted:
                moved += 1
            else:
                skipped += 1
            sqlite_queue.mark_rows([int(row["id"])], status="migrated",
                                   result={"migrated_to": "postgres", "imported": inserted})
        return {"imported": moved, "already_present": skipped}

    def export_backlog(self, sqlite_queue: Any) -> dict[str, int]:
        """The inverse, for a rollback to ``backend=sqlite``: every Postgres row still in flight
        becomes a SQLite ``retry`` row (claimable at once, attempts carried) and is marked
        ``exported`` here. Rows this process is running are included: the caller is shutting the
        Postgres path down. Idempotent through the SQLite delivery_id UNIQUE."""
        now = self._now()
        with self._pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM {self._t('jobs')} WHERE status IN ('queued','running') ORDER BY id FOR UPDATE",
            ).fetchall()
            exported = 0
            for row in rows:
                job = _job_from_row(row)
                ok = sqlite_queue.import_row(job, delivery_id=row["delivery_id"], attempts=int(row["attempts"]),
                                             not_before=_iso(row["not_before"]), created_at=_iso(row["created_at"]))
                conn.execute(
                    f"UPDATE {self._t('jobs')} SET status='exported', finished_at={_NOW}, owner=NULL, lease_expires_at=NULL, "
                    f"result=%(r)s::jsonb WHERE id=%(id)s",
                    {"now": now, "r": json.dumps({"exported_to": "sqlite", "inserted": ok}), "id": int(row["id"])})
                self.event(int(row["id"]), "exported", conn=conn, detail={"to": "sqlite", "inserted": ok})
                exported += 1
        return {"exported": exported}


def _backoff_from(policy: webhook_retry.RetryPolicy, attempts: int) -> float:
    """Full-jitter draw from the policy's ceiling for this many failed attempts."""
    return round(policy.wait_s(attempts) * random.random(), 3)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("context",):
            continue  # job content: never in operator listings
        out[key] = _iso(value) if isinstance(value, datetime) else value
    result = out.get("result")
    if result is None:
        out["result"] = {}
    elif not isinstance(result, dict):
        out["result"] = {"value": str(result)}
    return out


def _redact_url(url: str) -> str:
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        pass
    return url
