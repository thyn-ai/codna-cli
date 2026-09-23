"""Durable, local job queue for Codna's GitHub App webhook.

Codna is always local; the GitHub App is the only hosted surface — so this queue is a
single-file SQLite database on the app's own volume, not a distributed broker. It gives the
thin ingress three Cursor/Codex-grade properties without any external service:

  * durability — jobs survive a restart (WAL-mode SQLite on disk);
  * idempotency — one row per ``X-GitHub-Delivery`` (redelivered webhooks don't double-run);
  * safe claim — worker threads claim atomically (``BEGIN IMMEDIATE``), so a job runs once.

Thread-safe by opening a short-lived connection per operation (no shared cursor across the
ingress + worker threads).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .webhook import WebhookJob
from .webhook_control import PRIORITY_KINDS, SUPERSEDABLE_KINDS

_MAX_ATTEMPTS = 3
# SQL fragment: 0 for the kinds that gate a merge (review + the merge group inheriting its verdict),
# 1 for everything else. Built from PRIORITY_KINDS so the ordering and the pool's reserved-thread
# rule can never drift apart -- they are the same policy seen from two places.
_PRIORITY_SQL = (
    "CASE WHEN kind IN (" + ",".join(f"'{k}'" for k in sorted(PRIORITY_KINDS)) + ") THEN 0 ELSE 1 END"
)
_SUPERSEDABLE_SQL = ",".join(f"'{k}'" for k in sorted(SUPERSEDABLE_KINDS))
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id     TEXT UNIQUE,
    kind            TEXT NOT NULL,
    repo            TEXT NOT NULL,
    ref             TEXT,
    installation_id INTEGER,
    issue_number    INTEGER,
    pr_number       INTEGER,
    context         TEXT,
    reason          TEXT,
    status          TEXT NOT NULL DEFAULT 'queued',
    attempts        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    claimed_at      TEXT,
    finished_at     TEXT,
    result          TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status, id);
"""

# Columns added after the table first shipped. CREATE TABLE IF NOT EXISTS never alters an existing
# table, so a queue.db already on the Fly volume would be missing these — add them idempotently on
# open. (pr_number was silently dropped before this; review + comment-fix jobs depend on it.)
# not_before: a retry the worker asked to hold back (the account bridge gave no answer) is
# invisible to claim() until this UTC ISO timestamp; NULL = claimable now.
# priority: a nudge WITHIN a kind's class, not across it (the kind ordering below still wins).
# Only the boot reconciler sets it, for a head that lost its check to a restart: that head is
# already waiting on a check the ruleset requires, so it goes ahead of rows queued while it waited.
_ADDED_COLUMNS = {"pr_number": "INTEGER", "context": "TEXT", "not_before": "TEXT",
                  "priority": "INTEGER NOT NULL DEFAULT 0"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class QueuedJob:
    row_id: int
    job: WebhookJob
    delivery_id: str | None
    attempts: int
    # The Check Run this row already owns, when the ingress opened one at enqueue (Postgres backend:
    # a review's `codna review` run is created ``queued`` the moment the delivery is accepted, so it
    # is visible while the job waits for a slot). The worker UPDATES that run instead of creating a
    # second one. None on the SQLite queue and for rows an older ingress wrote: the worker then
    # creates the run itself when it claims the job, as it always has.
    check_run_id: int | None = None


class WebhookQueue:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    @property
    def path(self) -> str:
        """The database file; the restart registry (webhook_resume) lives in a sibling directory."""
        return self._path

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add any columns introduced after the table first shipped (idempotent)."""
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for col, coltype in _ADDED_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {coltype}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def supersede_queued(self, *, repo: str, pr_number: int | None, new_ref: str | None,
                         exclude_row_id: int | None = None) -> list[int]:
        """Retire every WAITING review/fix row for this pull request whose head is no longer current.

        The queue keeps the newest head only. Before this, a burst of pushes queued one review per
        head and the pool worked through them in order, so the checks that appeared were for commits
        that no longer existed while the current head waited behind them -- and a `codna fix` queued
        against a stale head would have opened a pull request rebuilt on a base that had moved.

        Rows with no ``ref`` are left alone (a `@codna review` comment resolves the head when it
        runs, so it is never stale), and so are merge-group rows, whose ``ref`` is the queue's own
        group commit rather than a PR head. Terminal state is ``superseded``: claim() only ever
        looks at 'queued'/'retry', so those rows can never run, and ``recent()``/``counts()`` keep
        saying what became of them.
        """
        if not repo or pr_number is None or not new_ref:
            return []
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ids = self._supersede_within(conn, repo=repo, pr_number=pr_number, new_ref=new_ref,
                                             exclude_row_id=exclude_row_id)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return ids

    @staticmethod
    def _supersede_within(conn: sqlite3.Connection, *, repo: str, pr_number: int,
                          new_ref: str, exclude_row_id: int | None) -> list[int]:
        """The supersede itself, inside a transaction the CALLER owns -- so ``enqueue`` can make
        the insert and the retirement of the heads it replaces one atomic step."""
        rows = conn.execute(
            f"SELECT id FROM jobs WHERE repo=? AND pr_number=? AND kind IN ({_SUPERSEDABLE_SQL}) "
            "AND status IN ('queued','retry') AND ref IS NOT NULL AND ref <> ? AND id <> ?",
            (repo, pr_number, new_ref, -1 if exclude_row_id is None else exclude_row_id),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if ids:
            conn.executemany(
                "UPDATE jobs SET status='superseded', finished_at=?, result=? WHERE id=?",
                [(_utc_now(), json.dumps({"summary": f"superseded by {new_ref[:8]}"}), i) for i in ids],
            )
        return ids

    def enqueue(self, job: WebhookJob, *, delivery_id: str | None, priority: int = 0,
                unless_pending: bool = False) -> bool:
        """Insert a job. Returns False if this delivery_id was already enqueued (dedup), or if it
        is a CI-failure fix for a (repo, ref) that already has one queued, running, or finished in
        the last 24 h -- every red check_suite on a PR head fires its own delivery, and one fix
        attempt per head is all that can ever be useful.

        ``unless_pending`` also returns False when a job of this kind for exactly this pull request
        head is already queued, retrying or running -- decided INSIDE the insert's own transaction,
        so a delivery for that head landing between the check and the insert cannot slip a second
        row in (the worker's requeue of a head that moved under a review, thyn-ai/codna#569)."""
        if job.reason == "check_suite_failure" and job.ref:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            with self._conn() as conn:
                dup = conn.execute(
                    "SELECT 1 FROM jobs WHERE kind='fix' AND repo=? AND ref=? AND reason=? "
                    "AND (status IN ('queued','retry','running') OR created_at >= ?) LIMIT 1",
                    (job.repo_full_name, job.ref, job.reason, cutoff),
                ).fetchone()
            if dup is not None:
                return False
        with self._conn() as conn:
            # ONE transaction for the insert and the retirement of the heads it replaces: a crash
            # (or an exception) between the two would otherwise leave the older heads queued
            # forever, and the pool would work through commits that no longer exist.
            conn.execute("BEGIN IMMEDIATE")
            try:
                if unless_pending and self._pending_within(conn, repo=job.repo_full_name, pr_number=job.pr_number,
                                                           ref=job.ref, kind=job.kind):
                    conn.execute("COMMIT")
                    return False
                cur = conn.execute(
                    "INSERT OR IGNORE INTO jobs "
                    "(delivery_id, kind, repo, ref, installation_id, issue_number, pr_number, context, "
                    "reason, created_at, priority) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        delivery_id,
                        job.kind,
                        job.repo_full_name,
                        job.ref,
                        job.installation_id,
                        job.issue_number,
                        job.pr_number,
                        json.dumps(job.context) if job.context is not None else None,
                        job.reason,
                        _utc_now(),
                        int(priority),
                    ),
                )
                inserted = cur.rowcount == 1
                if inserted and job.kind in SUPERSEDABLE_KINDS and job.pr_number is not None and job.ref:
                    # Excludes the row just written, so a re-delivery of the SAME head (same ref,
                    # different delivery id) cannot retire its own predecessor's work.
                    self._supersede_within(conn, repo=job.repo_full_name, pr_number=job.pr_number,
                                           new_ref=job.ref, exclude_row_id=int(cur.lastrowid))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return inserted

    @staticmethod
    def _pending_within(conn: sqlite3.Connection, *, repo: str, pr_number: int | None, ref: str | None,
                        kind: str) -> bool:
        """The ``has_pending`` predicate inside a transaction the CALLER owns (see ``enqueue``)."""
        if not repo or pr_number is None or not ref:
            return False
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE repo=? AND pr_number=? AND kind=? AND ref=? "
            "AND status IN ('queued','retry','running') LIMIT 1",
            (repo, pr_number, kind, ref),
        ).fetchone()
        return row is not None

    def has_pending(self, *, repo: str, pr_number: int | None, ref: str | None, kind: str) -> bool:
        """Whether a job of ``kind`` for exactly this pull request head is already waiting or
        running (thyn-ai/codna#569: one review of a head is all that is ever useful). A read for
        diagnostics and tests; a caller about to INSERT uses ``enqueue(unless_pending=True)``, which
        asks the same question inside the insert's transaction."""
        with self._conn() as conn:
            return self._pending_within(conn, repo=repo, pr_number=pr_number, ref=ref, kind=kind)

    def claim(self, *, reserve_priority: bool = False) -> QueuedJob | None:
        """Atomically claim the next runnable job (retryable rows included). None if empty.

        Ordering is by CLASS first: ``review`` and the merge-group ``queue`` job gate merges (the
        org rulesets make `codna review` a required check), while ``fix`` / ``secure`` / triage
        work does not -- a burst of CI-failure fixes must never starve them. ``priority`` then
        orders within a class, and ``id`` (arrival) breaks the tie.

        ``reserve_priority`` is the pool's last-free-thread rule: try the merge-gating classes
        first and only fall back to the rest when nothing there is waiting, so a fix can never take
        the last thread out from under a review that is already queued. Both selects live in the
        one ``BEGIN IMMEDIATE`` transaction, so the answer cannot change between them.
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                runnable = ("SELECT * FROM jobs WHERE status IN ('queued','retry') "
                            "AND (not_before IS NULL OR not_before <= ?) AND attempts < ? ")
                order = f"ORDER BY {_PRIORITY_SQL}, priority DESC, id LIMIT 1"
                args = (_utc_now(), _MAX_ATTEMPTS)
                row = None
                if reserve_priority:
                    row = conn.execute(
                        runnable + f"AND {_PRIORITY_SQL} = 0 " + order, args
                    ).fetchone()
                if row is None:
                    # Nothing gating a merge is waiting, so the reserved thread is free to do other
                    # work rather than idle: holding it empty would only trade one kind of starved
                    # queue for another.
                    row = conn.execute(runnable + order, args).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE jobs SET status='running', attempts=attempts+1, claimed_at=? WHERE id=?",
                    (_utc_now(), row["id"]),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return QueuedJob(
            row_id=row["id"],
            delivery_id=row["delivery_id"],
            attempts=row["attempts"] + 1,
            job=WebhookJob(
                kind=row["kind"],
                repo_full_name=row["repo"],
                ref=row["ref"],
                installation_id=row["installation_id"],
                issue_number=row["issue_number"],
                pr_number=row["pr_number"],
                context=json.loads(row["context"]) if row["context"] else None,
                reason=row["reason"] or "",
            ),
        )

    def complete(self, row_id: int, *, status: str, result: Mapping[str, object] | None = None,
                 retry: bool = True, retry_after_s: float | None = None) -> None:
        """Mark a claimed job done/failed. A failed job under the attempt cap becomes 'retry' --
        unless the caller says the failure is deterministic (retry=False), e.g. the CLI rejected
        the inputs: re-running it three times just repeats the same error. ``retry_after_s`` holds
        the retry back from claim() for that long (the account bridge gave no answer: re-asking
        immediately would burn every attempt inside the same outage) -- in the QUEUE, so no
        worker thread sleeps through the wait."""
        final = status
        if status == "failed" and retry:
            with self._conn() as conn:
                row = conn.execute("SELECT attempts FROM jobs WHERE id=?", (row_id,)).fetchone()
            if row is not None and row["attempts"] < _MAX_ATTEMPTS:
                final = "retry"
        not_before = None
        if final == "retry" and retry_after_s is not None:
            not_before = (datetime.now(timezone.utc) + timedelta(seconds=retry_after_s)).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, result=?, not_before=? WHERE id=?",
                (final, _utc_now(), json.dumps(dict(result or {})), not_before, row_id),
            )

    def recover_stale(self) -> dict[str, int]:
        """On startup, resolve jobs left 'running' by a crash.

        Rows still under the attempt cap are requeued ('retry'); rows that already used their
        last attempt are marked terminal 'failed' (never revived into an unclaimable 'retry').
        Returns {"requeued": N, "failed": M}.
        """
        with self._conn() as conn:
            failed = conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, result=? "
                "WHERE status='running' AND attempts >= ?",
                (_utc_now(), json.dumps({"error": "orphaned_after_max_attempts"}), _MAX_ATTEMPTS),
            ).rowcount
            requeued = conn.execute(
                "UPDATE jobs SET status='retry' WHERE status='running' AND attempts < ?",
                (_MAX_ATTEMPTS,),
            ).rowcount
        return {"requeued": requeued, "failed": failed}

    def row_state(self, row_id: int) -> tuple[str, int] | None:
        """``(status, attempts)`` of one row, or None if it does not exist. The restart reconciler
        reads this right after recover_stale to decide whether a registered job runs again."""
        with self._conn() as conn:
            row = conn.execute("SELECT status, attempts FROM jobs WHERE id=?", (row_id,)).fetchone()
        return None if row is None else (str(row["status"]), int(row["attempts"]))

    def counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    def depth_by_kind(self) -> dict[str, int]:
        """How many jobs of each kind are WAITING to run ('queued' + a retry that is due).

        Counts only, keyed by kind -- no repo, no ref, no job content -- because /healthz is
        public. Depth by STATUS (what /ready already reported) could not answer the question
        tonight's saturation actually posed: whether the backlog was reviews (a required check, so
        every one of them was blocking a merge) or fixes (which block nothing).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) c FROM jobs WHERE status IN ('queued','retry') "
                "AND (not_before IS NULL OR not_before <= ?) AND attempts < ? GROUP BY kind",
                (_utc_now(), _MAX_ATTEMPTS),
            ).fetchall()
        return {str(r["kind"]): int(r["c"]) for r in rows}

    def recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return newest queue rows for authenticated operator diagnostics only."""
        safe_limit = max(1, min(int(limit), 50))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, delivery_id, kind, repo, ref, installation_id, reason, status, attempts, "
                "created_at, claimed_at, finished_at, result FROM jobs ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            result: dict[str, Any] = {}
            if row["result"]:
                try:
                    loaded = json.loads(row["result"])
                    result = loaded if isinstance(loaded, dict) else {"value": str(loaded)}
                except ValueError:
                    result = {"value": str(row["result"])}
            out.append(
                {
                    "id": row["id"],
                    "delivery_id": row["delivery_id"],
                    "kind": row["kind"],
                    "repo": row["repo"],
                    "ref": row["ref"],
                    "installation_id": row["installation_id"],
                    "reason": row["reason"],
                    "status": row["status"],
                    "attempts": row["attempts"],
                    "created_at": row["created_at"],
                    "claimed_at": row["claimed_at"],
                    "finished_at": row["finished_at"],
                    "result": result,
                }
            )
        return out

    # --- backend hand-off (webhook_backend): this file as the write-ahead SPOOL and the rollback target --
    # Appended below the historical API on purpose: nothing above changes when the Postgres backend
    # is selected, and these are the only operations the migration needs from the SQLite side.

    def export_rows(self) -> list[dict[str, Any]]:
        """Every row still in flight (queued / retry / running), oldest first, with its job -- what
        ``PostgresQueue.import_sqlite_backlog`` moves across. The Check Run id of a running row is
        read from the JSON registry beside this file when one exists (webhook_resume), so the
        importer can hand it to the reaper."""
        from .webhook_resume import RunningJobRegistry, registry_dir_for

        registry = RunningJobRegistry(registry_dir_for(self._path))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','retry','running') ORDER BY id"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            entry = registry.get(int(row["id"])) if row["status"] == "running" else None
            out.append({
                "id": int(row["id"]), "delivery_id": row["delivery_id"], "status": str(row["status"]),
                "attempts": int(row["attempts"]), "not_before": row["not_before"], "created_at": row["created_at"],
                "check_run_id": entry.check_run_id if entry is not None else None,
                "job": WebhookJob(
                    kind=row["kind"], repo_full_name=row["repo"], ref=row["ref"],
                    installation_id=row["installation_id"], issue_number=row["issue_number"],
                    pr_number=row["pr_number"],
                    context=json.loads(row["context"]) if row["context"] else None, reason=row["reason"] or "",
                ),
            })
        return out

    def mark_rows(self, row_ids: list[int], *, status: str, result: Mapping[str, object] | None = None) -> int:
        """Set a terminal hand-off status (``migrated`` / ``exported``) on rows that now live in
        the other backend. ``claim()`` only ever looks at queued/retry, so a marked row can never
        run from here again, while ``counts()``/``recent()`` keep saying what became of it."""
        if not row_ids:
            return 0
        with self._conn() as conn:
            cur = conn.executemany(
                "UPDATE jobs SET status=?, finished_at=?, result=? WHERE id=? AND status IN ('queued','retry','running')",
                [(status, _utc_now(), json.dumps(dict(result or {})), int(rid)) for rid in row_ids],
            )
            return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(row_ids)

    def import_row(self, job: WebhookJob, *, delivery_id: str | None, attempts: int = 0,
                   not_before: str | None = None, created_at: str | None = None) -> bool:
        """A row coming BACK from the other backend (rollback): inserted directly as a claimable
        ``retry`` (or ``queued`` when it never ran), attempts carried, dedup'd on delivery_id.
        Bypasses ``enqueue``'s 24 h fix rule on purpose: the row already passed it once."""
        status = "retry" if int(attempts) > 0 else "queued"
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO jobs (delivery_id, kind, repo, ref, installation_id, issue_number, pr_number, "
                "context, reason, status, attempts, not_before, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (delivery_id, job.kind, job.repo_full_name, job.ref, job.installation_id, job.issue_number,
                 job.pr_number, json.dumps(job.context) if job.context is not None else None, job.reason,
                 status, min(int(attempts), _MAX_ATTEMPTS - 1), not_before, created_at or _utc_now()),
            )
            return cur.rowcount == 1

    def find_in_flight(self, delivery_id: str) -> int | None:
        """The row id of a WAITING (queued / retry) row for this delivery, or None. The spool uses
        it to retire a copy whose Postgres write turned out to have succeeded after the timeout."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE delivery_id=? AND status IN ('queued','retry') LIMIT 1", (delivery_id,)
            ).fetchone()
        return None if row is None else int(row["id"])

    def statuses_for(self, delivery_ids: list[str]) -> dict[str, str]:
        """``delivery_id -> status`` for the rows this file holds among ``delivery_ids`` (ids the
        file never saw are absent). ``PostgresQueue.reconcile_shadow_twins`` asks this at the boot
        that makes Postgres authoritative."""
        out: dict[str, str] = {}
        ids = [d for d in delivery_ids if d]
        if not ids:
            return out
        with self._conn() as conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(f"SELECT delivery_id, status FROM jobs WHERE delivery_id IN ({marks})", chunk):
                    out[str(row["delivery_id"])] = str(row["status"])
        return out

    def claim_kinds(self, kinds: set[str] | frozenset[str] | list[str] | tuple[str, ...]) -> QueuedJob | None:
        """``claim()`` restricted to the given kinds -- the ingress's degraded lane, which runs
        reviews from its own spool while the shared queue is unreachable and must never run a
        fix there. Same transaction shape and ordering as ``claim()``."""
        wanted = sorted({str(k) for k in kinds})
        if not wanted:
            return None
        marks = ",".join("?" for _ in wanted)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"SELECT * FROM jobs WHERE status IN ('queued','retry') AND kind IN ({marks}) "
                    "AND (not_before IS NULL OR not_before <= ?) AND attempts < ? "
                    "ORDER BY CASE WHEN kind='review' THEN 0 ELSE 1 END, id LIMIT 1",
                    (*wanted, _utc_now(), _MAX_ATTEMPTS),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "UPDATE jobs SET status='running', attempts=attempts+1, claimed_at=? WHERE id=?",
                    (_utc_now(), row["id"]),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return QueuedJob(
            row_id=row["id"], delivery_id=row["delivery_id"], attempts=row["attempts"] + 1,
            job=WebhookJob(
                kind=row["kind"], repo_full_name=row["repo"], ref=row["ref"],
                installation_id=row["installation_id"], issue_number=row["issue_number"],
                pr_number=row["pr_number"],
                context=json.loads(row["context"]) if row["context"] else None, reason=row["reason"] or "",
            ),
        )
