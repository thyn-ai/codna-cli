"""Queue backend selection (codna.webhook_backend): the flag, the shadow dual-write, the SQLite
file as a write-ahead spool in front of Postgres (spool on failure, forward when it answers, run
reviews from the spool while it is down, never a fix), the disjoint row ids, and the factory's
zero-loss hand-off in both directions. The Postgres parts need CODNA_TEST_DATABASE_URL."""
from __future__ import annotations

import threading
import time

import pytest

from codna import webhook_backend as wb
from codna.webhook import WebhookJob
from codna.webhook_queue import WebhookQueue


def _job(kind="review", ref="sha1", pr=7, installation=42):
    return WebhookJob(kind, "acme/app", ref=ref, pr_number=pr, installation_id=installation, reason="test")


# --- the flag ----------------------------------------------------------------------------------------

def test_backend_defaults_to_sqlite_and_refuses_typos():
    assert wb.backend_name({}) == "sqlite"
    assert wb.backend_name({"CODNA_WEBHOOK_QUEUE_BACKEND": " Postgres "}) == "postgres"
    assert wb.backend_name({"CODNA_WEBHOOK_QUEUE_BACKEND": "shadow"}) == "shadow"
    with pytest.raises(ValueError, match="expected one of sqlite, shadow, postgres"):
        wb.backend_name({"CODNA_WEBHOOK_QUEUE_BACKEND": "postgress"})


def test_degraded_lane_defaults_to_reviews_and_never_admits_fixes():
    assert wb.degraded_lane_kinds({}) == frozenset({"review"})
    assert wb.degraded_lane_kinds({"CODNA_WEBHOOK_DEGRADED_LANE": "review, fix ,queue"}) == frozenset({"review", "queue"})
    assert wb.degraded_lane_kinds({"CODNA_WEBHOOK_DEGRADED_LANE": ""}) == frozenset()


def test_sqlite_backend_is_the_plain_queue(tmp_path):
    queue = wb.open_queue(str(tmp_path / "queue.db"), environ={})
    assert isinstance(queue, WebhookQueue)


def test_postgres_backend_refuses_to_boot_without_a_database_url(tmp_path):
    with pytest.raises(RuntimeError, match="needs DATABASE_URL"):
        wb.open_queue(str(tmp_path / "queue.db"), environ={"CODNA_WEBHOOK_QUEUE_BACKEND": "postgres"})
    with pytest.raises(RuntimeError, match="needs DATABASE_URL"):
        wb.open_queue(str(tmp_path / "queue.db"), environ={"CODNA_WEBHOOK_QUEUE_BACKEND": "shadow"})


def test_current_job_is_thread_local():
    seen = {}

    def other():
        seen["other"] = wb.current_job_id()

    wb.set_current_job(17)
    try:
        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert wb.current_job_id() == 17 and seen["other"] is None
    finally:
        wb.set_current_job(None)
    assert wb.current_job_id() is None


# --- a Postgres that fails on demand --------------------------------------------------------------------

class _Flaky:
    """Wraps a real PostgresQueue; ``down`` makes every call raise (the database is unreachable)."""

    def __init__(self, real):
        self._real = real
        self.down = False
        self.enqueue_calls = 0

    def enqueue(self, *a, **k):
        self.enqueue_calls += 1
        if self.down:
            raise ConnectionError("no route to postgres")
        return self._real.enqueue(*a, **k)

    def claim(self, *a, **k):
        if self.down:
            raise ConnectionError("no route to postgres")
        return self._real.claim(*a, **k)

    def ping(self):
        return not self.down

    def __getattr__(self, name):
        if self.down and name in ("counts", "recent", "complete", "row_state", "recover_stale"):
            raise ConnectionError("no route to postgres")
        return getattr(self._real, name)


@pytest.fixture
def spool(tmp_path, make_pg_queue):
    now = [0.0]
    flaky = _Flaky(make_pg_queue())
    sqlite = WebhookQueue(tmp_path / "queue.db")
    queue = wb.SpoolingQueue(flaky, sqlite, enqueue_timeout_s=2.0, degraded_after_s=60.0,
                             degraded_kinds=("review",), clock=lambda: now[0])
    queue.tick = lambda seconds: now.__setitem__(0, now[0] + seconds)
    queue.flaky = flaky
    yield queue
    queue.stop()


def test_a_delivery_lands_in_the_spool_when_postgres_cannot_answer_and_is_forwarded_later(spool):
    flaky = spool.flaky
    assert spool.enqueue(_job(ref="a"), delivery_id="d-a") is True   # Postgres up: straight through
    flaky.down = True
    assert spool.enqueue(_job(ref="b", pr=8), delivery_id="d-b") is True   # down: the file takes it, GitHub still gets 202
    assert spool.enqueue(_job(ref="b", pr=8), delivery_id="d-b") is False  # redelivery: dedup'd in the file too
    assert spool.spool_rows() == 1 and spool.spooled == 1
    assert spool.postgres_down_for_s() == 0.0
    assert spool.forward_once() == 0                                    # still down: nothing moves
    flaky.down = False
    assert spool.forward_once() == 1                                    # answers again: the row crosses over
    assert spool.spool_rows() == 0 and spool.forwarded == 1
    assert spool.spool.counts() == {"migrated": 1}
    kinds = sorted(c.delivery_id for c in (spool.claim(), spool.claim()))
    assert kinds == ["d-a", "d-b"]                                      # both run from Postgres, once each
    assert flaky.counts()["running"] == 2


def test_the_enqueue_timeout_spools_a_hung_write(tmp_path, make_pg_queue):
    class _Hang(_Flaky):
        def enqueue(self, *a, **k):
            time.sleep(5)
            return True

    hang = _Hang(make_pg_queue())
    queue = wb.SpoolingQueue(hang, WebhookQueue(tmp_path / "q.db"), enqueue_timeout_s=0.2)
    started = time.monotonic()
    assert queue.enqueue(_job(), delivery_id="d1") is True
    assert time.monotonic() - started < 2.0                             # GitHub's 10 s window is never at risk
    assert queue.spool_rows() == 1


def test_a_write_that_answers_after_the_timeout_retires_its_spool_copy(tmp_path, make_pg_queue):
    """The timeout spooled the delivery; the Postgres write then succeeded anyway. There must never
    be two claimable copies: the late result retires the spool row, and Postgres holds the one job."""
    class _Slow(_Flaky):
        def enqueue(self, *a, **k):
            time.sleep(0.6)                                                  # answers, but after the bound
            return self._real.enqueue(*a, **k)

    slow = _Slow(make_pg_queue())
    queue = wb.SpoolingQueue(slow, WebhookQueue(tmp_path / "q.db"), enqueue_timeout_s=0.2)
    assert queue.enqueue(_job(), delivery_id="d1") is True                    # 202 to GitHub at once
    assert queue.spool_rows() == 1 and queue.postgres_down_for_s() >= 0.0
    deadline = time.monotonic() + 5
    while queue.spool_rows() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert queue.spool_rows() == 0                                           # retired by the late result
    assert queue.spool.counts() == {"migrated": 1}
    assert slow._real.counts() == {"queued": 1}                              # exactly one job, in Postgres
    assert queue.forwarded == 1 and queue.postgres_down_for_s() == 0.0        # and the false outage window closed
    claimed = slow._real.claim()
    assert claimed is not None and claimed.delivery_id == "d1" and slow._real.claim() is None


def test_degraded_lane_runs_reviews_from_the_spool_after_the_threshold_never_fixes(spool):
    flaky = spool.flaky
    flaky.down = True
    spool.enqueue(_job(kind="fix", ref="f", pr=7), delivery_id="d-fix")
    spool.enqueue(_job(kind="review", ref="r", pr=8), delivery_id="d-review")   # a different PR: nothing supersedes anything
    with pytest.raises(ConnectionError):
        spool.claim()                                                    # under the threshold: the error propagates
    assert not spool.degraded()
    spool.tick(61)
    assert spool.degraded()
    claimed = spool.claim()
    assert claimed is not None and claimed.job.kind == "review" and claimed.delivery_id == "d-review"
    assert claimed.row_id >= wb.SPOOL_ROW_OFFSET                        # a spool row id: complete() routes it back
    assert spool.row_state(claimed.row_id) == ("running", 1)
    assert spool.claim() is None                                         # the fix stays in the file
    spool.complete(claimed.row_id, status="done", result={"summary": "reviewed in degraded mode"})
    assert spool.spool.counts() == {"done": 1, "queued": 1}
    flaky.down = False
    assert spool.forward_once() == 1                                     # the fix crosses over when Postgres is back
    assert spool.claim().job.kind == "fix"
    assert spool.degraded_claims == 1


def test_recover_stale_covers_both_stores(spool):
    flaky = spool.flaky
    flaky.down = True
    spool.enqueue(_job(kind="review", ref="r"), delivery_id="d-review")
    spool.tick(61)
    claimed = spool.claim()                                              # running in the file (degraded lane)
    assert claimed.row_id >= wb.SPOOL_ROW_OFFSET
    flaky.down = False
    spool.enqueue(_job(kind="review", ref="r2", pr=9), delivery_id="d-pg")
    pg_claimed = spool.claim()
    flaky._real.abandon_leases()
    assert spool.recover_stale() == {"requeued": 2, "failed": 0}
    assert spool.counts()["spool_rows"] == 1 and pg_claimed.row_id < wb.SPOOL_ROW_OFFSET


def test_recover_stale_at_boot_survives_postgres_being_down(spool):
    """Booting while Postgres is unreachable is the outage the spool exists for: the spool's own
    rows are recovered and the service comes up; the Postgres side is left to the reaper."""
    flaky = spool.flaky
    flaky.down = True
    spool.enqueue(_job(kind="review", ref="r"), delivery_id="d-review")
    spool.tick(61)
    claimed = spool.claim()                                              # a degraded-lane row, running in the file
    assert claimed.row_id >= wb.SPOOL_ROW_OFFSET
    assert spool.recover_stale() == {"requeued": 1, "failed": 0}         # no exception; the spool row is runnable again
    assert spool.spool.counts() == {"retry": 1}
    assert spool.postgres_down_for_s() >= 0.0 and spool.degraded()


def test_the_late_path_waits_for_the_spool_write_before_retiring(tmp_path):
    """The Postgres write answers after the timeout while the caller is still writing the spool:
    the late path must not look for the spool row before it exists (it would find nothing and
    leave two claimable copies until the forwarder's next pass)."""
    import threading

    order = []
    gate = threading.Event()

    def slow():
        time.sleep(0.3)
        return True

    def late(value):
        order.append(("late", value, gate.is_set()))

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        wb._call_with_timeout(slow, 0.05, on_late=late, late_gate=gate)
    time.sleep(0.5)                                                      # the helper has finished; the gate is not set
    assert order == []                                                   # ...so the late path has not run
    order.append(("spool-written",))
    gate.set()
    deadline = time.monotonic() + 2
    while len(order) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert order == [("spool-written",), ("late", True, True)]           # strictly after the caller's spool write
    assert time.monotonic() - started < 3


def test_counts_and_recent_still_answer_while_postgres_is_down(spool):
    spool.flaky.down = True
    spool.enqueue(_job(), delivery_id="d1")
    assert spool.counts() == {"spool_rows": 1}
    assert [r["delivery_id"] for r in spool.recent(limit=5)] == ["d1"]


# --- shadow --------------------------------------------------------------------------------------------

def test_shadow_dual_writes_and_counts_but_never_fails_the_delivery(tmp_path, make_pg_queue):
    flaky = _Flaky(make_pg_queue())
    shadow = wb.ShadowQueue(WebhookQueue(tmp_path / "q.db"), flaky)
    assert shadow.enqueue(_job(ref="a"), delivery_id="d1") is True
    assert shadow.enqueue(_job(ref="a"), delivery_id="d1") is False        # SQLite dedup: the shadow is not asked again
    flaky.down = True
    assert shadow.enqueue(_job(ref="b", pr=8), delivery_id="d2") is True   # Postgres down: SQLite still accepted it
    assert (shadow.shadow_writes, shadow.shadow_errors, shadow.shadow_duplicates) == (1, 1, 0)
    flaky.down = False
    assert shadow.claim().delivery_id == "d1"                            # claims come from SQLite, untouched
    assert shadow.shadow_drift() == 1                                    # d2 has no twin: the Phase-2 gate says so
    calls = []
    real_connection = flaky._real.connection

    class _Counting:
        def __enter__(self):
            self._cm = real_connection()
            conn = self._cm.__enter__()
            calls.append(conn)
            return conn

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    flaky._real.connection = lambda: _Counting()
    assert shadow.shadow_drift() == 1 and len(calls) == 1                # one connection, one round trip for 50 rows
    assert shadow.counts()["running"] == 1                              # everything else is SQLite's
    assert flaky.counts() == {"queued": 1}                              # the twin was never claimed


# --- the factory: import once, roll back once, never lose a row -------------------------------------------

def test_open_queue_postgres_imports_the_file_once_and_sqlite_rollback_exports_it_back(tmp_path, pg_url, pg_schema):
    path = str(tmp_path / "queue.db")
    sqlite = WebhookQueue(path)
    for i in range(3):
        assert sqlite.enqueue(_job(ref=f"s{i}", pr=i), delivery_id=f"d{i}")
    running = sqlite.claim()
    environ = {"CODNA_WEBHOOK_QUEUE_BACKEND": "postgres", "DATABASE_URL": pg_url, "CODNA_WEBHOOK_DEGRADED_LANE": "review"}
    queue = wb.open_queue(path, environ=environ, schema=pg_schema, heartbeat_s=0, pool_max=2)
    try:
        assert isinstance(queue, wb.SpoolingQueue) and queue.forwarder_alive()
        assert sqlite.counts() == {"migrated": 3}
        ran = []
        while (claimed := queue.claim()) is not None:
            ran.append(claimed.delivery_id)
            queue.complete(claimed.row_id, status="done")
        assert sorted(ran) == ["d0", "d1", "d2"]                          # every job exactly once, the running one included
        assert queue.job(next(r["id"] for r in queue.recent(limit=10) if r["delivery_id"] == running.delivery_id))["resumes"] == 1
        # a second postgres boot imports nothing (idempotent)
        again = wb.open_queue(path, environ=environ, schema=pg_schema, heartbeat_s=0, pool_max=2)
        try:
            assert again.counts().get("done") == 3
        finally:
            again.stop()
            again.primary.close()
        # new work arrives on Postgres, then the flag is flipped back: the boot exports it into the file
        for i in range(3, 5):
            assert queue.enqueue(_job(ref=f"s{i}", pr=i), delivery_id=f"d{i}")
        queue.claim()                                                    # one of them is running at the moment of rollback
    finally:
        queue.stop()
        queue.primary.close()
    rolled = wb.open_queue(path, environ={"CODNA_WEBHOOK_QUEUE_BACKEND": "sqlite", "DATABASE_URL": pg_url},
                           rollback_from_postgres=True, schema=pg_schema, heartbeat_s=0, pool_max=2)
    assert isinstance(rolled, WebhookQueue)
    ran = []
    while (claimed := rolled.claim()) is not None:
        ran.append(claimed.delivery_id)
        rolled.complete(claimed.row_id, status="done")
    assert sorted(ran) == ["d3", "d4"]                                  # both come back, once each; the file never re-runs d0..d2
    assert rolled.counts() == {"migrated": 3, "done": 2}


def test_the_postgres_boot_retires_the_shadow_twins_the_file_already_finished(tmp_path, pg_url, pg_schema, make_pg_queue):
    """Phase 1 -> Phase 2. On shadow every accepted delivery has a Postgres twin nobody claims; the
    boot that makes Postgres authoritative must not run those again (measured 2026-09-20: the first
    shadow review's twin sat `queued` in the cluster with its age climbing past 800 s)."""
    path = str(tmp_path / "queue.db")
    sqlite = WebhookQueue(path)
    shadow = wb.ShadowQueue(sqlite, make_pg_queue())
    for i in range(3):
        assert shadow.enqueue(_job(ref=f"s{i}", pr=i), delivery_id=f"d{i}")
    first = shadow.claim()                                                # SQLite runs d0 to completion...
    shadow.complete(first.row_id, status="done")
    second = shadow.claim()                                               # ...d1 to a terminal (deterministic) failure...
    shadow.complete(second.row_id, status="failed", retry=False)
    assert sqlite.counts() == {"done": 1, "failed": 1, "queued": 1}      # ...and d2 is still waiting at the flip
    environ = {"CODNA_WEBHOOK_QUEUE_BACKEND": "postgres", "DATABASE_URL": pg_url}
    first_boot = wb.open_queue(path, environ=environ, schema=pg_schema, heartbeat_s=0, pool_max=2)
    first_boot.stop()
    first_boot.primary.close()
    assert sqlite.counts() == {"done": 1, "failed": 1, "migrated": 1}      # only the in-flight row was handed over
    # A restart BEFORE any worker claimed d2: its SQLite original now reads `migrated` (a hand-off,
    # not a run) while it is still `queued` in Postgres. The second boot must keep it.
    queue = wb.open_queue(path, environ=environ, schema=pg_schema, heartbeat_s=0, pool_max=2)
    try:
        ran = []
        while (claimed := queue.claim()) is not None:
            ran.append(claimed.delivery_id)
            queue.complete(claimed.row_id, status="done")
        assert ran == ["d2"]                                              # once; the finished deliveries never run again
        rows = {r["delivery_id"]: r for r in queue.recent(limit=10)}
        assert rows["d0"]["status"] == rows["d1"]["status"] == "cancelled"
        assert rows["d2"]["status"] == "done"
        actors = {e["event"]: e["actor"] for e in queue.events(rows["d0"]["id"])}
        assert actors["cancelled"] == "backend:shadow-reconcile"
        assert queue.primary.reconcile_shadow_twins(sqlite) == {"reconciled": 0, "kept_in_flight": 0,
                                                                "kept_handed_off": 0, "no_counterpart": 0}  # idempotent
    finally:
        queue.stop()
        queue.primary.close()
