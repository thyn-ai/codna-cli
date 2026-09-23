"""Operator queries and mutations over the Postgres job queue -- the ONE implementation behind the
authenticated ``/ops/*`` HTTP surface, the ``codna webhook ops …`` CLI and the ``webhook-ops.yml``
workflow. Every mutation writes ``job_events`` with its actor, so "who cancelled #1234" is always
answerable.

Also here: the read models the metrics endpoint (``webhook_metrics``), the SLO endpoint and the
autoscaler consume (:func:`metrics_snapshot`, :func:`slo_snapshot`, :func:`scaler_inputs`), the
worker registry (:func:`register_worker` / :func:`worker_heartbeat` / :func:`workers`), and the
leader election the reaper and the scaler use (:class:`LeaderLock`: an advisory lock held inside
one open transaction on a dedicated connection, so it is exclusive through a transaction-pooling
PgBouncer as well as on a direct connection).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .webhook_pg_queue import PostgresQueue, _iso, _public_row

_NOW = "COALESCE(%(now)s::timestamptz, now())"
PRIORITY_BOUNDS = (-100, 100)
# Histogram buckets (seconds) for job wait and duration; the SLO is 60 s, so the boundaries bracket it.
WAIT_BUCKETS = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 300.0, 600.0, 1800.0)
DURATION_BUCKETS = (15.0, 30.0, 60.0, 90.0, 120.0, 180.0, 300.0, 600.0, 900.0, 1800.0, 3600.0)


def _q(queue: PostgresQueue) -> str:
    return queue.schema


def _now_param(queue: PostgresQueue) -> dict[str, Any]:
    return {"now": queue.now()}


# --- jobs ------------------------------------------------------------------------------------------
def list_jobs(queue: PostgresQueue, *, status: str | None = None, kind: str | None = None,
              installation_id: int | None = None, repo: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    clauses, params = [], {"limit": max(1, min(int(limit), 500))}
    if status:
        clauses.append("status = %(status)s")
        params["status"] = status
    if kind:
        clauses.append("kind = %(kind)s")
        params["kind"] = kind
    if installation_id is not None:
        clauses.append("installation_id = %(inst)s")
        params["inst"] = int(installation_id)
    if repo:
        clauses.append("repo = %(repo)s")
        params["repo"] = repo
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT id, delivery_id, kind, repo, ref, installation_id, pr_number, issue_number, reason, status, "
            f"priority, attempts, max_attempts, resumes, owner, lease_expires_at, check_run_id, not_before, "
            f"cancel_requested, last_error, created_at, claimed_at, finished_at "
            f"FROM {_q(queue)}.jobs {where} ORDER BY id DESC LIMIT %(limit)s", params).fetchall()
    return [_public_row(r) for r in rows]


def job_detail(queue: PostgresQueue, row_id: int) -> dict[str, Any] | None:
    row = queue.job(row_id)
    if row is None:
        return None
    row["events"] = queue.events(row_id)
    return row


def cancel(queue: PostgresQueue, row_id: int, *, actor: str, reason: str = "operator") -> dict[str, Any]:
    """A queued row ends now as ``cancelled``. A running row gets ``cancel_requested``: its
    worker's next heartbeat fails, the worker stops the job (JobHandle.cancel where the pool has
    one; the lease keeper's callback otherwise), and the reaper finishes the row. Terminal rows
    are left alone."""
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(f"SELECT status FROM {_q(queue)}.jobs WHERE id = %s FOR UPDATE", (row_id,)).fetchone()
        if row is None:
            return {"id": row_id, "outcome": "not_found"}
        if row["status"] == "queued":
            conn.execute(
                f"UPDATE {_q(queue)}.jobs SET status='cancelled', finished_at={_NOW}, last_error=%(why)s, "
                f"result=%(r)s::jsonb WHERE id=%(id)s",
                {**_now_param(queue), "why": f"cancelled:{reason}", "r": json.dumps({"summary": f"cancelled by {actor}: {reason}"}), "id": row_id})
            outcome = "cancelled"
        elif row["status"] == "running":
            conn.execute(f"UPDATE {_q(queue)}.jobs SET cancel_requested = %s WHERE id = %s", (f"{reason} (by {actor})", row_id))
            outcome = "cancel_requested"
        else:
            return {"id": row_id, "outcome": "already_terminal", "status": row["status"]}
        queue.event(row_id, outcome, actor=actor, detail={"reason": reason}, conn=conn)
    return {"id": row_id, "outcome": outcome}


def retry_now(queue: PostgresQueue, row_id: int, *, actor: str) -> dict[str, Any]:
    """Make a dead / failed / cancelled row runnable again at once, with a fresh budget. The
    interruption count is kept: an operator retry is not a second automatic resume."""
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(f"SELECT status, kind FROM {_q(queue)}.jobs WHERE id = %s FOR UPDATE", (row_id,)).fetchone()
        if row is None:
            return {"id": row_id, "outcome": "not_found"}
        if row["status"] not in ("dead", "failed", "cancelled", "queued"):
            return {"id": row_id, "outcome": "not_retryable", "status": row["status"]}
        # The row keeps its Check Run id: the worker that claims it asks GitHub whether that run is
        # still open (webhook_queued_check.open_check_run) -- it continues one the sweep has not
        # reached yet, and opens a fresh one when the verdict, the dead-letter notice or the sweep
        # already completed it. Forgetting the id here would leave a still-open run with nothing
        # pointing at it. `check_closed` is cleared so a later retirement of THIS row sweeps whatever
        # run it then carries.
        conn.execute(
            f"UPDATE {_q(queue)}.jobs SET status='queued', attempts=0, max_attempts=%(budget)s, not_before=NULL, "
            f"finished_at=NULL, owner=NULL, lease_expires_at=NULL, cancel_requested=NULL, "
            f"posted = posted - 'dead_letter' - 'check_closed' WHERE id=%(id)s",
            {"budget": queue.policy_for(row["kind"]).budget, "id": row_id})
        queue.event(row_id, "retry_now", actor=actor, detail={"from_status": row["status"]}, conn=conn)
    return {"id": row_id, "outcome": "queued"}


def reprioritize(queue: PostgresQueue, row_id: int, priority: int, *, actor: str) -> dict[str, Any]:
    lo, hi = PRIORITY_BOUNDS
    value = max(lo, min(hi, int(priority)))
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(
            f"UPDATE {_q(queue)}.jobs SET priority = %s WHERE id = %s AND status = 'queued' RETURNING id", (value, row_id)
        ).fetchone()
        if row is None:
            return {"id": row_id, "outcome": "not_queued"}
        queue.event(row_id, "reprioritized", actor=actor, detail={"priority": value}, conn=conn)
    return {"id": row_id, "outcome": "reprioritized", "priority": value}


# --- tenants ------------------------------------------------------------------------------------------
_TENANT_FIELDS = {"paused": bool, "max_concurrency": int, "weight": int, "dead_letter_conclusion": str}


def tenant_set(queue: PostgresQueue, installation_id: int, *, actor: str, **fields: Any) -> dict[str, Any]:
    """Upsert one tenant's knobs: ``paused``, ``max_concurrency`` (1..64), ``weight`` (1..16),
    ``dead_letter_conclusion`` (``neutral`` | ``action_required`` | None = the global default)."""
    updates: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in _TENANT_FIELDS:
            raise ValueError(f"unknown tenant field {key!r}")
        if key == "max_concurrency":
            value = max(1, min(64, int(value)))
        elif key == "weight":
            value = max(1, min(16, int(value)))
        elif key == "paused":
            value = bool(value)
        elif key == "dead_letter_conclusion" and value not in (None, "neutral", "action_required"):
            raise ValueError("dead_letter_conclusion must be neutral, action_required or empty")
        updates[key] = value
    if not updates:
        raise ValueError("nothing to set")
    sets = ", ".join(f"{k} = %({k})s" for k in updates)
    with queue.connection() as conn, conn.transaction():
        conn.execute(
            f"INSERT INTO {_q(queue)}.tenants (installation_id, updated_at, updated_by) VALUES (%(id)s, {_NOW}, %(actor)s) "
            f"ON CONFLICT (installation_id) DO NOTHING", {"id": int(installation_id), "actor": actor, **_now_param(queue)})
        conn.execute(
            f"UPDATE {_q(queue)}.tenants SET {sets}, updated_at = {_NOW}, updated_by = %(actor)s WHERE installation_id = %(id)s",
            {**updates, "id": int(installation_id), "actor": actor, **_now_param(queue)})
        queue.event(None, "tenant_updated", actor=actor, detail={"installation_id": int(installation_id), **updates}, conn=conn)
    return tenant(queue, installation_id) or {}


def tenant(queue: PostgresQueue, installation_id: int) -> dict[str, Any] | None:
    with queue.connection() as conn:
        row = conn.execute(f"SELECT * FROM {_q(queue)}.tenants WHERE installation_id = %s", (int(installation_id),)).fetchone()
    return _public_row(row) if row else None


def tenants(queue: PostgresQueue) -> list[dict[str, Any]]:
    with queue.connection() as conn:
        rows = conn.execute(f"SELECT * FROM {_q(queue)}.tenants ORDER BY installation_id").fetchall()
    return [_public_row(r) for r in rows]


# --- workers ---------------------------------------------------------------------------------------------
def register_worker(queue: PostgresQueue, *, owner: str | None = None, machine_id: str | None, region: str | None,
                    image: str | None, role: str, slots: int) -> str:
    who = owner or queue.owner
    with queue.connection() as conn:
        conn.execute(
            f"INSERT INTO {_q(queue)}.workers (owner, machine_id, region, image, role, slots, busy, started_at, heartbeat_at) "
            f"VALUES (%(o)s, %(m)s, %(r)s, %(i)s, %(role)s, %(s)s, 0, {_NOW}, {_NOW}) "
            f"ON CONFLICT (owner) DO UPDATE SET machine_id=EXCLUDED.machine_id, region=EXCLUDED.region, image=EXCLUDED.image, "
            f"role=EXCLUDED.role, slots=EXCLUDED.slots, busy=0, draining=false, drain_acked_at=NULL, "
            f"started_at={_NOW}, heartbeat_at={_NOW}",
            {"o": who, "m": machine_id, "r": region, "i": image, "role": role, "s": int(slots), **_now_param(queue)})
    return who


def worker_heartbeat(queue: PostgresQueue, *, owner: str | None = None, busy: int) -> dict[str, Any] | None:
    """Refresh liveness + load; returns the row (so the caller learns ``draining``). A worker asked
    to drain acknowledges by stamping ``drain_acked_at`` -- the scaler waits for that stamp AND
    ``busy = 0`` before it stops the machine."""
    who = owner or queue.owner
    with queue.connection() as conn:
        row = conn.execute(
            f"UPDATE {_q(queue)}.workers SET heartbeat_at = {_NOW}, busy = %(b)s, "
            f"drain_acked_at = CASE WHEN draining AND drain_acked_at IS NULL THEN {_NOW} ELSE drain_acked_at END "
            f"WHERE owner = %(o)s RETURNING *", {"b": int(busy), "o": who, **_now_param(queue)}).fetchone()
    return _public_row(row) if row else None


def deregister_worker(queue: PostgresQueue, *, owner: str | None = None) -> None:
    with queue.connection() as conn:
        conn.execute(f"DELETE FROM {_q(queue)}.workers WHERE owner = %s", (owner or queue.owner,))


def workers(queue: PostgresQueue, *, stale_after_s: float = 30.0) -> list[dict[str, Any]]:
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT *, (heartbeat_at < {_NOW} - make_interval(secs => %(stale)s)) AS stale "
            f"FROM {_q(queue)}.workers ORDER BY started_at", {"stale": float(stale_after_s), **_now_param(queue)}).fetchall()
    return [_public_row(r) for r in rows]


def set_draining(queue: PostgresQueue, owner: str, flag: bool, *, actor: str) -> dict[str, Any] | None:
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(
            f"UPDATE {_q(queue)}.workers SET draining = %s, drain_acked_at = CASE WHEN %s THEN drain_acked_at ELSE NULL END "
            f"WHERE owner = %s RETURNING *", (bool(flag), bool(flag), owner)).fetchone()
        queue.event(None, "worker_drain" if flag else "worker_undrain", actor=actor, detail={"owner": owner}, conn=conn)
    return _public_row(row) if row else None


def prune_dead_workers(queue: PostgresQueue, *, older_than_s: float = 600.0) -> int:
    """Forget workers whose heartbeat is long gone (a machine that was destroyed); their jobs are
    the reaper's business, not this row's."""
    with queue.connection() as conn:
        rows = conn.execute(
            f"DELETE FROM {_q(queue)}.workers WHERE heartbeat_at < {_NOW} - make_interval(secs => %(age)s) RETURNING owner",
            {"age": float(older_than_s), **_now_param(queue)}).fetchall()
    return len(rows)


# --- leases + dead letters (what the reaper reads and writes) ------------------------------------------------
def expired_leases(queue: PostgresQueue, *, limit: int = 100) -> list[dict[str, Any]]:
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT id, kind, repo, ref, pr_number, installation_id, issue_number, reason, attempts, max_attempts, "
            f"resumes, owner, check_run_id, cancel_requested, context FROM {_q(queue)}.jobs "
            f"WHERE status='running' AND lease_expires_at < {_NOW} ORDER BY lease_expires_at LIMIT %(limit)s",
            {"limit": int(limit), **_now_param(queue)}).fetchall()
    return [dict(r) for r in rows]


def requeue_interrupted(queue: PostgresQueue, row_id: int, *, cause: str, hold_s: float, actor: str) -> bool:
    """A running row whose lease lapsed: back to ``queued`` with ``resumes + 1`` after ``hold_s``.
    Only while it is still the row the reaper saw (status running) -- the worker may have finished
    it in the meantime, in which case the completion wins."""
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(
            f"UPDATE {_q(queue)}.jobs SET status='queued', owner=NULL, lease_expires_at=NULL, resumes=resumes+1, "
            f"check_run_id=NULL, last_error=%(err)s, not_before={_NOW} + make_interval(secs => %(hold)s) "
            f"WHERE id=%(id)s AND status='running' AND lease_expires_at < {_NOW} RETURNING id",
            {"err": f"interrupted:{cause}", "hold": float(hold_s), "id": row_id, **_now_param(queue)}).fetchone()
        if row is None:
            return False
        queue.event(row_id, "interrupted", actor=actor, detail={"cause": cause, "hold_s": hold_s}, conn=conn)
    return True


def finish_row(queue: PostgresQueue, row_id: int, *, status: str, error: str, actor: str,
               summary: str, require_running: bool = True) -> bool:
    """Terminal transition by the reaper (``dead`` / ``cancelled`` / ``failed``)."""
    guard = "AND status='running'" if require_running else "AND status IN ('running','queued')"
    with queue.connection() as conn, conn.transaction():
        row = conn.execute(
            f"UPDATE {_q(queue)}.jobs SET status=%(s)s, finished_at={_NOW}, owner=NULL, lease_expires_at=NULL, "
            f"last_error=%(err)s, result=%(r)s::jsonb WHERE id=%(id)s {guard} RETURNING id",
            {"s": status, "err": error, "r": json.dumps({"summary": summary, "error": error}), "id": row_id, **_now_param(queue)}).fetchone()
        if row is None:
            return False
        queue.event(row_id, status, actor=actor, detail={"error": error, "summary": summary[:300]}, conn=conn)
    return True


def unposted_dead(queue: PostgresQueue, *, limit: int = 50) -> list[dict[str, Any]]:
    """Dead-lettered rows whose tenant has not been told yet (``posted.dead_letter`` unset)."""
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT j.id, j.kind, j.repo, j.ref, j.pr_number, j.installation_id, j.issue_number, j.attempts, "
            f"j.check_run_id, j.last_error, j.context, t.dead_letter_conclusion "
            f"FROM {_q(queue)}.jobs j LEFT JOIN {_q(queue)}.tenants t ON t.installation_id = j.installation_id "
            f"WHERE j.status='dead' AND NOT (j.posted ? 'dead_letter') ORDER BY j.finished_at LIMIT %s", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


# Terminal states a row can reach WITHOUT a worker ever closing the Check Run it owns: retired while
# waiting (a newer head, an operator's cancel, a rollback export) or failed before the worker could
# report (a token that would not mint, orphaned by a restart on its last attempt).
RETIRED_STATUSES = ("superseded", "cancelled", "exported", "failed")


def retired_check_runs(queue: PostgresQueue, *, repo: str | None = None, pr_number: int | None = None,
                       since_s: float = 7 * 24 * 3600, limit: int = 100) -> list[dict[str, Any]]:
    """Rows in a ``RETIRED_STATUSES`` state that still carry a Check Run id and have not had that run
    confirmed closed (``posted.check_closed`` unset), oldest first. The ingress asks for one pull
    request right after its enqueue retired the older heads; the reaper asks for everything as the
    backstop. The window bounds the one-time backfill over rows that predate the sweep."""
    clauses = ["status = ANY(%(statuses)s)", "check_run_id IS NOT NULL", "NOT (posted ? 'check_closed')",
               f"finished_at >= {_NOW} - make_interval(secs => %(since)s)"]
    params: dict[str, Any] = {"statuses": list(RETIRED_STATUSES), "since": float(since_s), "limit": int(limit),
                              **_now_param(queue)}
    if repo:
        clauses.append("repo = %(repo)s")
        params["repo"] = repo
    if pr_number is not None:
        clauses.append("pr_number = %(pr)s")
        params["pr"] = int(pr_number)
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT id, kind, repo, ref, pr_number, installation_id, status, check_run_id, last_error, result, finished_at "
            f"FROM {_q(queue)}.jobs WHERE {' AND '.join(clauses)} ORDER BY finished_at LIMIT %(limit)s", params).fetchall()
    return [dict(r) for r in rows]


def dead_after_outage(queue: PostgresQueue, *, since_s: float = 6 * 3600, limit: int = 200) -> list[dict[str, Any]]:
    """Dead rows from the last ``since_s`` whose failure text names an outage -- the candidates the
    post-outage sweep runs again (never those already retried by an operator)."""
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT id, kind, last_error, result FROM {_q(queue)}.jobs WHERE status='dead' "
            f"AND finished_at >= {_NOW} - make_interval(secs => %(since)s) AND NOT (posted ? 'outage_retry') "
            f"ORDER BY finished_at LIMIT %(limit)s", {"since": float(since_s), "limit": int(limit), **_now_param(queue)}).fetchall()
    return [dict(r) for r in rows]


# --- leader election ---------------------------------------------------------------------------------------
class LeaderLock:
    """An advisory lock held INSIDE ONE OPEN TRANSACTION on a DEDICATED connection for as long as
    this object is alive. ``try_acquire`` is non-blocking; whoever holds it runs the reaper / the
    scaler. Losing the connection ends the transaction and with it the lock, so a dead leader is
    replaced within one poll.

    Transaction-level (``pg_try_advisory_xact_lock``) rather than session-level on purpose. The
    pooled ``DATABASE_URL`` Fly Managed Postgres hands out is PgBouncer in TRANSACTION mode: the
    server connection goes back to the pool after every autocommit statement, so a session-level
    lock taken through it stays with the server backend, not with this client, and the next client
    to land on that backend "acquires" the same lock. Observed 2026-09-21, the first time two
    ingress machines ran: both published ``codna_webhook_scaler_leader 1`` and both scalers ticked
    (a probe from inside a machine took one key on two connections, six times out of six). An open
    transaction pins the server connection to this client for the transaction's whole life under
    either pool mode, and a transaction-level lock lives exactly as long as that transaction. A
    loser rolls its probing transaction back at once so it holds no server connection between polls.
    The server's ``idle_in_transaction_session_timeout`` is 0 on that cluster; the holder's ticks
    (2 s scaler, 15 s reaper) keep the transaction from ever being idle for long anyway.
    """

    def __init__(self, queue: PostgresQueue, purpose: str) -> None:
        from .webhook_pg_schema import lock_key

        self._queue = queue
        self._key = lock_key(queue.schema, purpose)
        self._conn: Any = None
        self.held = False

    def try_acquire(self) -> bool:
        try:
            if self._conn is None or self._conn.closed:
                self._conn = self._queue.dedicated_connection()
                self._conn.autocommit = False  # every statement below runs in ONE transaction that stays open while the lock is held
                self.held = False
            # Asked again on every poll, not only once: inside the open transaction that holds the
            # lock this is idempotent (true again), and if that transaction ended behind our back
            # (a pooler or the server closed it) it is the truthful answer -- false when another
            # candidate has taken the lock meanwhile, and leadership is dropped right here rather
            # than believed in on the strength of a `SELECT 1`.
            row = self._conn.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS got", (self._key,)).fetchone()
            self.held = bool(row and row["got"])
            if not self.held:
                self._conn.rollback()  # end the probing transaction: a pooled server connection returns to the pool until the next poll
            return self.held
        except Exception:  # noqa: BLE001 -- not the leader while the database is unreachable
            self.held = False
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
            return False

    def release(self) -> None:
        try:
            if self._conn is not None and self.held:
                self._conn.rollback()  # ending the transaction releases the lock
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.held = False
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


# --- read models: metrics, SLO, scaler inputs --------------------------------------------------------------
def _hist(values: list[float], buckets: tuple[float, ...]) -> dict[str, Any]:
    counts = []
    for edge in buckets:
        counts.append(sum(1 for v in values if v <= edge))
    return {"buckets": list(buckets), "counts": counts, "count": len(values), "sum": float(sum(values))}


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return float(ordered[idx])


def metrics_snapshot(queue: PostgresQueue, *, window_s: float = 3600.0, top_installations: int = 25) -> dict[str, Any]:
    """Everything ``/metrics`` renders, from one round of queries. Bounded label cardinality: at
    most ``top_installations`` installations appear by name; the rest fold into ``other``."""
    s = _q(queue)
    p = {"win": float(window_s), **_now_param(queue)}
    out: dict[str, Any] = {"pg_up": 1}
    with queue.connection() as conn:
        depth = conn.execute(
            f"SELECT kind, installation_id, count(*) AS c FROM {s}.jobs WHERE status='queued' "
            f"AND (not_before IS NULL OR not_before <= {_NOW}) AND attempts < max_attempts GROUP BY 1, 2", p).fetchall()
        oldest = conn.execute(
            f"SELECT kind, EXTRACT(EPOCH FROM ({_NOW} - min(created_at))) AS age FROM {s}.jobs WHERE status='queued' "
            f"AND (not_before IS NULL OR not_before <= {_NOW}) AND attempts < max_attempts GROUP BY kind", p).fetchall()
        running = conn.execute(f"SELECT kind, count(*) AS c FROM {s}.jobs WHERE status='running' GROUP BY kind").fetchall()
        totals = conn.execute(
            f"SELECT kind, installation_id, status, count(*) AS c FROM {s}.jobs "
            f"WHERE finished_at >= {_NOW} - make_interval(secs => %(win)s) OR status IN ('queued','running') GROUP BY 1,2,3", p).fetchall()
        waits = conn.execute(
            f"SELECT kind, EXTRACT(EPOCH FROM (claimed_at - created_at)) AS w FROM {s}.jobs "
            f"WHERE claimed_at IS NOT NULL AND claimed_at >= {_NOW} - make_interval(secs => %(win)s)", p).fetchall()
        durations = conn.execute(
            f"SELECT kind, EXTRACT(EPOCH FROM (finished_at - claimed_at)) AS d FROM {s}.jobs "
            f"WHERE finished_at IS NOT NULL AND claimed_at IS NOT NULL AND status IN ('done','failed','dead') "
            f"AND finished_at >= {_NOW} - make_interval(secs => %(win)s)", p).fetchall()
        events = conn.execute(
            f"SELECT event, count(*) AS c FROM {s}.job_events WHERE event IN ('retry','dead','interrupted','cancelled') "
            f"AND at >= {_NOW} - make_interval(secs => %(win)s) GROUP BY event", p).fetchall()
        dead_total = conn.execute(f"SELECT count(*) AS c FROM {s}.jobs WHERE status='dead'").fetchone()
        wk = conn.execute(
            f"SELECT count(*) FILTER (WHERE heartbeat_at >= {_NOW} - interval '30 seconds' AND NOT draining) AS live, "
            f"count(*) FILTER (WHERE draining) AS draining, count(*) FILTER (WHERE heartbeat_at < {_NOW} - interval '30 seconds') AS stale, "
            f"COALESCE(sum(slots) FILTER (WHERE heartbeat_at >= {_NOW} - interval '30 seconds' AND NOT draining), 0) AS slots, "
            f"COALESCE(sum(busy) FILTER (WHERE heartbeat_at >= {_NOW} - interval '30 seconds'), 0) AS busy FROM {s}.workers", p).fetchone()
    keep = {r["installation_id"] for r in sorted(depth, key=lambda r: -int(r["c"]))[:top_installations]}

    def _inst(value: Any) -> str:
        return str(value) if value in keep and value is not None else ("none" if value is None else "other")

    out["queue_depth"] = {}
    for r in depth:
        key = (r["kind"], _inst(r["installation_id"]))
        out["queue_depth"][key] = out["queue_depth"].get(key, 0) + int(r["c"])
    out["oldest_runnable_age_s"] = {r["kind"]: float(r["age"] or 0.0) for r in oldest}
    out["running"] = {r["kind"]: int(r["c"]) for r in running}
    out["jobs_total"] = {}
    for r in totals:
        key = (r["kind"], _inst(r["installation_id"]), r["status"])
        out["jobs_total"][key] = out["jobs_total"].get(key, 0) + int(r["c"])
    by_kind_w: dict[str, list[float]] = {}
    for r in waits:
        by_kind_w.setdefault(r["kind"], []).append(float(r["w"] or 0.0))
    out["job_wait_s"] = {k: _hist(v, WAIT_BUCKETS) for k, v in by_kind_w.items()}
    by_kind_d: dict[str, list[float]] = {}
    for r in durations:
        by_kind_d.setdefault(r["kind"], []).append(float(r["d"] or 0.0))
    out["job_duration_s"] = {k: _hist(v, DURATION_BUCKETS) for k, v in by_kind_d.items()}
    out["events_window"] = {r["event"]: int(r["c"]) for r in events}
    out["dead_total"] = int(dead_total["c"]) if dead_total else 0
    out["workers"] = {k: int(wk[k] or 0) for k in ("live", "draining", "stale", "slots", "busy")} if wk else {}
    return out


def slo_snapshot(queue: PostgresQueue, *, window_s: float = 3600.0, target_s: float = 60.0, min_samples: int = 20) -> dict[str, Any]:
    """The documented SLO, measured: over the rolling window, the p95 of (Check Run created -
    delivery queued) for ``review`` jobs -- the moment the tenant SEES the check exist -- against
    ``target_s``. ``claimed_at`` stands in for rows that never reached a Check Run (a failure
    before creating one still waited in the queue). Reports ``insufficient_samples`` below
    ``min_samples`` instead of a verdict on noise."""
    s = _q(queue)
    p = {"win": float(window_s), **_now_param(queue)}
    with queue.connection() as conn:
        rows = conn.execute(
            f"SELECT EXTRACT(EPOCH FROM (COALESCE(check_started_at, claimed_at) - created_at)) AS w FROM {s}.jobs "
            f"WHERE kind='review' AND claimed_at IS NOT NULL AND claimed_at >= {_NOW} - make_interval(secs => %(win)s)", p).fetchall()
        dead = conn.execute(
            f"SELECT count(*) AS c FROM {s}.jobs WHERE kind='review' AND status='dead' "
            f"AND finished_at >= {_NOW} - make_interval(secs => %(win)s)", p).fetchone()
    waits = [max(0.0, float(r["w"] or 0.0)) for r in rows]
    p95 = _pct(waits, 0.95)
    enough = len(waits) >= int(min_samples)
    return {
        "slo": f"95% of codna review checks created within {target_s:.0f}s of the queued delivery, rolling {window_s / 3600:.0f}h, >= {min_samples} reviews",
        "window_s": window_s, "target_s": target_s, "samples": len(waits), "min_samples": min_samples,
        "p50_s": _pct(waits, 0.50), "p95_s": p95, "p99_s": _pct(waits, 0.99), "max_s": max(waits) if waits else None,
        "reviews_dead_in_window": int(dead["c"]) if dead else 0,
        "met": (p95 is not None and p95 <= target_s) if enough else None,
        "verdict": ("met" if (p95 is not None and p95 <= target_s) else "breached") if enough else "insufficient_samples",
    }


def scaler_inputs(queue: PostgresQueue, *, worker_stale_s: float = 30.0) -> dict[str, Any]:
    """What the autoscaler decides on: runnable rows now, the oldest one's age, and the live
    workers' slots/busy (fresh heartbeats only)."""
    s = _q(queue)
    p = {"stale": float(worker_stale_s), **_now_param(queue)}
    with queue.connection() as conn:
        q = conn.execute(
            f"SELECT count(*) AS runnable, COALESCE(EXTRACT(EPOCH FROM ({_NOW} - min(created_at))), 0) AS oldest_age "
            f"FROM {s}.jobs WHERE status='queued' AND (not_before IS NULL OR not_before <= {_NOW}) AND attempts < max_attempts", p).fetchone()
        w = conn.execute(
            f"SELECT owner, machine_id, region, slots, busy, draining, drain_acked_at, "
            f"(heartbeat_at < {_NOW} - make_interval(secs => %(stale)s)) AS stale FROM {s}.workers WHERE role IN ('worker','all')", p).fetchall()
    live = [dict(r) for r in w if not r["stale"]]
    return {
        "runnable": int(q["runnable"]) if q else 0,
        "oldest_runnable_age_s": float(q["oldest_age"] or 0.0) if q else 0.0,
        "workers": [{**r, "drain_acked_at": _iso(r.get("drain_acked_at"))} for r in live],
        "free_slots": sum(max(0, int(r["slots"]) - int(r["busy"])) for r in live if not r["draining"]),
        "busy_slots": sum(int(r["busy"]) for r in live),
    }


def as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
