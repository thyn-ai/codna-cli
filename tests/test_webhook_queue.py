"""Durable job queue for the Codna webhook -- dedup, atomic claim, retry, recovery -- on BOTH
backends. Every test below that takes ``h`` runs once against the SQLite file and once against
Postgres (skipped with a message when no CODNA_TEST_DATABASE_URL is set; CI's pg job sets one):
the same assertions are the parity proof for ``CODNA_WEBHOOK_QUEUE_BACKEND=postgres``. The
Postgres harness runs with a flat 3-attempt, zero-backoff policy so the two vocabularies line up
(``retry`` = a queued row that already ran once; a spent budget is terminal either way -- see
``_terminal_failures``); the per-kind budgets and backoff are tested in test_webhook_pg_queue.py.
The tests at the bottom are about the SQLite FILE itself (legacy schema migration) and stay
SQLite-only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from codna import webhook_queue
from codna.webhook import WebhookJob
from codna.webhook_queue import WebhookQueue


class _Harness:
    def __init__(self, name, queue, advance):
        self.name = name
        self.queue = queue
        self.advance = advance  # move the queue's notion of "now" forward by N seconds


@pytest.fixture(params=["sqlite", "postgres"])
def h(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        offset = [0.0]
        monkeypatch.setattr(webhook_queue, "_utc_now",
                            lambda: (datetime.now(timezone.utc) + timedelta(seconds=offset[0])).isoformat())

        def advance(seconds):
            offset[0] += seconds

        return _Harness("sqlite", WebhookQueue(tmp_path / "queue.db"), advance)
    make = request.getfixturevalue("make_pg_queue")
    from codna.webhook_pg_schema import KINDS
    from codna.webhook_retry import RetryPolicy

    now = [datetime.now(timezone.utc)]
    queue = make(clock=lambda: now[0], retry_policy={k: RetryPolicy(budget=3, base_s=0.0, factor=1.0, cap_s=0.0) for k in KINDS})

    def advance(seconds):
        now[0] = now[0] + timedelta(seconds=seconds)

    return _Harness("postgres", queue, advance)


def _q(tmp_path):
    """The SQLite file alone -- for the tests about the file itself (schema, spool, main's own)."""
    return WebhookQueue(tmp_path / "queue.db")


def _job(kind="fix", repo="acme/app", ref="sha1"):
    return WebhookJob(kind, repo, ref=ref, installation_id=42, reason="test")


def _terminal_failures(queue) -> int:
    """SQLite says ``failed`` for a spent budget; Postgres says ``dead`` (transient) or ``failed``
    (deterministic). Both are terminal and unclaimable, which is what these tests assert."""
    counts = queue.counts()
    return counts.get("failed", 0) + counts.get("dead", 0)


def test_enqueue_dedups_on_delivery_id(h):
    q = h.queue
    assert q.enqueue(_job(), delivery_id="d1") is True
    assert q.enqueue(_job(), delivery_id="d1") is False  # redelivered webhook -> not re-queued
    assert q.enqueue(_job(), delivery_id="d2") is True
    assert q.counts().get("queued") == 2


def test_claim_is_one_shot_and_preserves_the_job(h):
    q = h.queue
    q.enqueue(_job(kind="secure", ref="deadbeef"), delivery_id="d1")
    claimed = q.claim()
    assert claimed is not None
    assert claimed.job.kind == "secure"
    assert claimed.job.repo_full_name == "acme/app"
    assert claimed.job.ref == "deadbeef"
    assert claimed.attempts == 1
    assert q.claim() is None  # nothing left to claim


def test_complete_done_removes_from_the_runnable_set(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    claimed = q.claim()
    q.complete(claimed.row_id, status="done", result={"summary": "ok"})
    assert q.counts().get("done") == 1
    assert q.claim() is None


def test_failed_under_cap_becomes_retry_and_is_reclaimable(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    c1 = q.claim()
    q.complete(c1.row_id, status="failed", result={"error": "boom"})
    assert q.counts().get("retry") == 1
    c2 = q.claim()  # retryable
    assert c2 is not None
    assert c2.row_id == c1.row_id
    assert c2.attempts == 2


def test_failed_past_cap_stays_terminal(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    for _ in range(3):
        c = q.claim()
        assert c is not None
        q.complete(c.row_id, status="failed")
    assert q.claim() is None  # attempts exhausted
    assert _terminal_failures(q) == 1


def test_recover_stale_requeues_running_jobs_after_a_crash(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    q.claim()  # now 'running' (simulate crash before complete)
    assert q.claim() is None
    if h.name == "postgres":
        q.abandon_leases()  # the process died: nothing renews the lease
    recovered = q.recover_stale()
    assert recovered == {"requeued": 1, "failed": 0}
    assert q.claim() is not None  # reclaimable again


def test_recover_stale_fails_a_job_that_crashed_on_its_last_attempt(h):
    """A crash on the final attempt must go terminal, never stranded in an unclaimable 'retry'."""
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    # burn the first two attempts (fail -> retry), then leave the third 'running' (crash)
    for _ in range(2):
        c = q.claim()
        q.complete(c.row_id, status="failed")
    c3 = q.claim()  # attempts == 3, status 'running'
    assert c3.attempts == 3
    if h.name == "postgres":
        q.abandon_leases()
    recovered = q.recover_stale()
    assert recovered == {"requeued": 0, "failed": 1}  # exhausted -> terminal, not revived
    assert q.claim() is None
    assert _terminal_failures(q) == 1


# --- pr_number + context survive the queue round-trip (regression: they were dropped) ----------

def test_pr_number_survives_enqueue_claim_roundtrip(h):
    """Regression: a review job's pr_number was silently dropped (no column), so codna_command later
    raised 'review_requires_pr'. It must round-trip through the durable queue."""
    q = h.queue
    q.enqueue(WebhookJob("review", "acme/app", pr_number=17, installation_id=42, reason="pr"), delivery_id="d1")
    claimed = q.claim()
    assert claimed.job.pr_number == 17


def test_context_dict_survives_roundtrip(h):
    q = h.queue
    ctx = {"in_reply_to_id": 555, "path": "src/a.py", "line": 10, "head_ref": "feature", "is_fork": False}
    q.enqueue(WebhookJob("fix", "acme/app", ref="sha1", pr_number=7, context=ctx,
                         reason="review_comment_codna_fix"), delivery_id="d1")
    claimed = q.claim()
    assert claimed.job.context == ctx
    assert claimed.job.pr_number == 7


def test_null_context_roundtrips_as_none(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")  # no context
    assert q.claim().job.context is None


def test_reviews_are_claimed_before_older_fix_jobs(h):
    q = h.queue
    assert q.enqueue(_job(kind="fix", ref="f1"), delivery_id="d-fix") is True
    assert q.enqueue(_job(kind="review", ref="r1"), delivery_id="d-review") is True
    first = q.claim()
    assert first is not None and first.job.kind == "review"
    second = q.claim()
    assert second is not None and second.job.kind == "fix"


def test_ci_failure_fixes_collapse_to_one_per_repo_and_head(h):
    q = h.queue
    ci = WebhookJob("fix", "acme/app", ref="head1", reason="check_suite_failure")
    assert q.enqueue(ci, delivery_id="suite-1") is True
    assert q.enqueue(ci, delivery_id="suite-2") is False          # second red suite, same head
    assert q.enqueue(WebhookJob("fix", "acme/app", ref="head2", reason="check_suite_failure"),
                     delivery_id="suite-3") is True               # a new head gets its one attempt
    assert q.enqueue(WebhookJob("fix", "acme/app", ref="head1", reason="comment_codna_fix"),
                     delivery_id="human-1") is True               # a human asking is never collapsed
    claimed = q.claim()
    q.complete(claimed.row_id, status="failed", retry=False)
    assert q.enqueue(ci, delivery_id="suite-4") is False          # still collapsed after it finished


def test_non_retryable_failure_is_terminal_on_the_first_attempt(h):
    q = h.queue
    q.enqueue(_job(), delivery_id="d1")
    row = q.claim()
    q.complete(row.row_id, status="failed", result={"summary": "cli_error"}, retry=False)
    assert q.claim() is None
    assert q.counts().get("failed") == 1


def test_retry_after_holds_the_row_in_the_queue_until_due(h):
    """The account bridge gave no answer: the retry must not be re-claimed inside the same outage,
    and the wait belongs to the QUEUE (a worker thread sleeping through it blocks the pool)."""
    q = h.queue
    assert q.enqueue(_job(kind="review"), delivery_id="d1")
    first = q.claim()
    q.complete(first.row_id, status="failed", result={"summary": "bridge unavailable"}, retry=True,
               retry_after_s=60)
    assert q.claim() is None                                   # held: not claimable yet
    assert q.counts().get("retry") == 1
    h.advance(61)
    second = q.claim()                                         # due: claimable again, attempt 2
    assert second is not None and second.row_id == first.row_id and second.attempts == 2


def test_retry_without_a_hold_is_claimable_immediately_and_clears_an_old_hold(h):
    q = h.queue
    assert q.enqueue(_job(kind="fix"), delivery_id="d1")
    first = q.claim()
    q.complete(first.row_id, status="failed", retry=True)      # plain retry: no not_before
    second = q.claim()
    assert second is not None and second.attempts == 2


def test_row_state_and_recent_report_the_same_vocabulary(h):
    q = h.queue
    q.enqueue(_job(kind="review"), delivery_id="d1")
    row = q.claim()
    assert q.row_state(row.row_id) == ("running", 1)
    q.complete(row.row_id, status="failed")
    assert q.row_state(row.row_id) == ("retry", 1)
    assert q.row_state(10**9) is None
    [recent] = [r for r in q.recent(limit=5) if r["delivery_id"] == "d1"]
    assert recent["kind"] == "review" and recent["status"] == "retry" and recent["attempts"] == 1
    assert "context" not in recent  # job content never appears in operator listings


# --- the SQLite FILE: legacy schemas on the volume -----------------------------------------------

def test_migration_adds_columns_to_a_preexisting_db(tmp_path):
    """An old queue.db on the Fly volume lacks pr_number/context; opening it must ALTER them in
    (CREATE TABLE IF NOT EXISTS never alters), else claim() KeyErrors on the missing columns."""
    import sqlite3

    db = tmp_path / "queue.db"
    # Build a legacy schema WITHOUT pr_number / context.
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, delivery_id TEXT UNIQUE, "
        "kind TEXT NOT NULL, repo TEXT NOT NULL, ref TEXT, installation_id INTEGER, issue_number INTEGER, "
        "reason TEXT, status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0, "
        "created_at TEXT NOT NULL, claimed_at TEXT, finished_at TEXT, result TEXT);"
    )
    con.commit()
    con.close()

    q = WebhookQueue(db)  # opening must migrate the columns in
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(jobs)")}  # r[1] = column name
    assert {"pr_number", "context"}.issubset(cols)
    # and it works end-to-end on the migrated db
    q.enqueue(WebhookJob("review", "acme/app", pr_number=3, reason="x"), delivery_id="d1")
    assert q.claim().job.pr_number == 3


def test_not_before_column_is_added_to_a_queue_db_that_predates_it(tmp_path):
    """The Fly volume carries a queue.db created before this column existed."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, delivery_id TEXT UNIQUE, kind TEXT NOT NULL,"
        " repo TEXT NOT NULL, ref TEXT, installation_id INTEGER, issue_number INTEGER, reason TEXT,"
        " status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,"
        " claimed_at TEXT, finished_at TEXT, result TEXT);"
    )
    conn.close()
    q = WebhookQueue(path)
    assert q.enqueue(_job(), delivery_id="d1") and q.claim() is not None   # claim() reads not_before


def test_has_pending_sees_exactly_the_waiting_and_running_rows_for_that_head(tmp_path):
    """The worker asks before queueing a head a review found had moved under it (codna#569)."""
    q = _q(tmp_path)
    review = WebhookJob("review", "acme/app", ref="head2", pr_number=7, reason="pull_request_synchronize")
    ask = dict(repo="acme/app", pr_number=7, ref="head2", kind="review")
    assert q.has_pending(**ask) is False
    assert q.enqueue(review, delivery_id="d1")
    assert q.has_pending(**ask) is True                                   # queued
    assert q.has_pending(**{**ask, "ref": "head1"}) is False               # another head
    assert q.has_pending(**{**ask, "kind": "fix"}) is False                # another kind
    assert q.has_pending(**{**ask, "pr_number": 8}) is False               # another pull request
    assert q.has_pending(**{**ask, "ref": None}) is False                  # no head, no answer
    claimed = q.claim()
    assert q.has_pending(**ask) is True                                   # running
    q.complete(claimed.row_id, status="failed")
    assert q.has_pending(**ask) is True                                   # retry (still under the cap)
    q.complete(claimed.row_id, status="done")
    assert q.has_pending(**ask) is False                                  # terminal


def test_enqueue_unless_pending_decides_inside_the_inserts_own_transaction(tmp_path, monkeypatch):
    """codna review of #573: a has_pending() read followed by enqueue() is two transactions, and a
    delivery for the same head landing in between would make it two rows. The check now runs on the
    insert's connection, inside its BEGIN IMMEDIATE, so nothing can land between the two."""
    q = _q(tmp_path)
    review = WebhookJob("review", "acme/app", ref="head2", pr_number=7, reason="head_moved_during_review")
    seen = []
    real = type(q)._pending_within

    def spy(conn, **kw):
        seen.append(conn.in_transaction)
        return real(conn, **kw)

    monkeypatch.setattr(type(q), "_pending_within", staticmethod(spy))
    assert q.enqueue(review, delivery_id="moved-1", priority=1, unless_pending=True) is True
    assert seen == [True]                                                  # asked inside the transaction
    assert q.enqueue(review, delivery_id="moved-2", priority=1, unless_pending=True) is False   # queued
    assert q.enqueue(review, delivery_id="plain", priority=0) is True      # a plain delivery still lands
    assert q.counts() == {"queued": 2}
    claimed = q.claim()
    assert q.enqueue(review, delivery_id="moved-3", unless_pending=True) is False             # running counts
    q.complete(claimed.row_id, status="done")
    second = q.claim()
    assert second is not None and q.claim() is None
    q.complete(second.row_id, status="done")
    assert q.enqueue(review, delivery_id="moved-4", unless_pending=True) is True              # all terminal: lands


# --- the SQLite file as spool / rollback target (webhook_backend) ----------------------------------

def test_export_rows_lists_only_in_flight_rows_with_their_jobs(tmp_path):
    q = WebhookQueue(tmp_path / "queue.db")
    q.enqueue(_job(kind="review", ref="a"), delivery_id="done-1")
    q.enqueue(_job(kind="fix", ref="b"), delivery_id="running-1")
    q.enqueue(_job(kind="fix", ref="c"), delivery_id="retry-1")
    q.enqueue(_job(kind="secure", ref="d"), delivery_id="queued-1")
    done = q.claim()                                   # the review goes first
    q.complete(done.row_id, status="done")
    running = q.claim()                                # then the oldest of the rest: fix b
    assert running.delivery_id == "running-1"
    retry = q.claim()                                  # fix c, failed once -> retry
    q.complete(retry.row_id, status="failed")
    exported = {r["delivery_id"]: r for r in q.export_rows()}
    assert set(exported) == {"running-1", "retry-1", "queued-1"}          # done rows are not in flight
    assert {d: r["status"] for d, r in exported.items()} == {"running-1": "running", "retry-1": "retry",
                                                             "queued-1": "queued"}
    assert exported["retry-1"]["attempts"] == 1 and exported["running-1"]["job"].ref == "b"
    assert exported["running-1"]["check_run_id"] is None                  # no registry entry recorded
    assert all(r["job"].repo_full_name == "acme/app" for r in exported.values())


def test_mark_rows_and_import_row_round_trip(tmp_path):
    q = WebhookQueue(tmp_path / "queue.db")
    q.enqueue(_job(kind="review", ref="a"), delivery_id="d1")
    [row] = q.export_rows()
    assert q.mark_rows([row["id"]], status="migrated", result={"to": "postgres"}) == 1
    assert q.counts() == {"migrated": 1}
    assert q.claim() is None                                   # a migrated row never runs from here
    assert q.mark_rows([row["id"]], status="migrated") == 0    # idempotent: already terminal
    assert q.import_row(_job(kind="fix", ref="b"), delivery_id="d2", attempts=1) is True
    assert q.import_row(_job(kind="fix", ref="b"), delivery_id="d2", attempts=1) is False  # dedup on delivery id
    assert q.counts().get("retry") == 1
    claimed = q.claim()
    assert claimed.delivery_id == "d2" and claimed.attempts == 2  # attempts carried, then the claim adds one


def test_find_in_flight_sees_only_waiting_rows(tmp_path):
    q = WebhookQueue(tmp_path / "queue.db")
    q.enqueue(_job(kind="review", ref="a"), delivery_id="d1")
    rid = q.find_in_flight("d1")
    assert rid is not None and q.find_in_flight("nope") is None
    claimed = q.claim()
    assert claimed.row_id == rid and q.find_in_flight("d1") is None          # running is not waiting
    q.complete(claimed.row_id, status="failed")                              # retry: waiting again
    assert q.find_in_flight("d1") == rid


def test_claim_kinds_never_hands_out_another_kind(tmp_path):
    q = WebhookQueue(tmp_path / "queue.db")
    q.enqueue(_job(kind="fix", ref="f"), delivery_id="fix-1")
    q.enqueue(_job(kind="review", ref="r"), delivery_id="review-1")
    assert q.claim_kinds({"secure"}) is None
    claimed = q.claim_kinds({"review"})
    assert claimed is not None and claimed.job.kind == "review"
    assert q.claim_kinds({"review"}) is None                   # the fix stays where it is
    assert q.claim_kinds(set()) is None
    assert q.counts() == {"running": 1, "queued": 1}
