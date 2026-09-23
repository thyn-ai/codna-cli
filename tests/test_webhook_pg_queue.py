"""The Postgres queue backend (codna.webhook_pg_queue) beyond parity: per-kind budgets with
backoff and the dead-letter state, per-tenant caps and fairness, priority aging, one live row per
head, leases (heartbeat, loss, zombie completion ignored), crash recovery through the reaper with
the Check Run completed exactly once, many claimers across threads AND processes never
double-claiming, and the zero-loss hand-off from the SQLite file and back.

Every test needs CODNA_TEST_DATABASE_URL (see conftest.py); CI's pg job provides it."""
from __future__ import annotations

import multiprocessing
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from codna import webhook_pg_ops, webhook_retry
from codna.webhook import WebhookJob
from codna.webhook_lease import Reaper
from codna.webhook_pg_queue import MAX_RESUMES, PostgresQueue, idempotency_key
from codna.webhook_pg_schema import KINDS, REQUIRED_VERSION, check_compatible, connect, current_version, migrate
from codna.webhook_queue import WebhookQueue

pytestmark = pytest.mark.usefixtures("pg_url")


class _Clock:
    def __init__(self):
        self.now = datetime.now(timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


def _job(kind="review", repo="acme/app", ref="sha1", pr=7, installation=42, reason="pull_request_opened", **kw):
    return WebhookJob(kind, repo, ref=ref, pr_number=pr, installation_id=installation, reason=reason, **kw)


class _GitHub:
    """The GitHub calls the reaper and the dead-letter notifier make, recorded in order."""

    def __init__(self, *, head=None, open_runs=()):
        self.calls = []
        self.head = head
        self.open_runs = set(open_runs)

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", installation_id, kind))
        return "scoped-token"

    def pull_request_head_sha(self, repo, token, number):
        self.calls.append(("head", number))
        return self.head

    def complete_check_run_if_open(self, repo, token, check_run_id, *, conclusion, summary, name):
        self.calls.append(("complete", check_run_id, conclusion, summary))
        if check_run_id in self.open_runs:
            self.open_runs.discard(check_run_id)
            return True
        return False

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name):
        self.calls.append(("update", check_run_id, conclusion, summary))

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.calls.append(("create", head_sha, summary))
        return 5150

    def post_review_comment_reply(self, repo, token, pr_number, in_reply_to, body):
        self.calls.append(("reply", pr_number, in_reply_to, body))

    def post_issue_comment(self, repo, token, issue_number, body):
        self.calls.append(("comment", issue_number, body))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


# --- schema ----------------------------------------------------------------------------------------

def test_migrate_is_idempotent_and_compatible(pg_url, pg_schema):
    with connect(pg_url) as conn:
        assert current_version(conn, schema=pg_schema) == REQUIRED_VERSION
        assert migrate(conn, schema=pg_schema, log=lambda _m: None) == REQUIRED_VERSION  # nothing to apply
        assert check_compatible(conn, schema=pg_schema) == REQUIRED_VERSION
        # every table the code names exists
        tables = {r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (pg_schema,)).fetchall()}
        assert {"jobs", "tenants", "workers", "job_events", "schema_version"} <= tables


def test_boot_refuses_a_schema_older_than_the_code(pg_url):
    from codna.webhook_pg_schema import drop_schema

    name = "t_older_" + os.urandom(3).hex()
    with connect(pg_url) as conn:
        try:
            conn.execute(f"CREATE SCHEMA {name}")
            with pytest.raises(RuntimeError, match="codna webhook migrate"):
                check_compatible(conn, schema=name)
        finally:
            drop_schema(conn, schema=name)


# --- budgets, backoff, dead letter -------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_max_attempts_is_stamped_from_the_kinds_policy(make_pg_queue, kind):
    q = make_pg_queue()
    assert q.enqueue(_job(kind=kind), delivery_id=f"d-{kind}")
    [row] = q.recent(limit=1)
    assert row["max_attempts"] == webhook_retry.budget_for(kind)


def test_transient_failures_back_off_then_dead_letter_at_the_budget(make_pg_queue):
    clock = _Clock()
    q = make_pg_queue(clock=clock)
    policy = webhook_retry.policy_for("review")
    q.enqueue(_job(kind="review"), delivery_id="d1")
    for attempt in range(1, policy.budget + 1):
        claimed = q.claim()
        assert claimed is not None and claimed.attempts == attempt
        q.complete(claimed.row_id, status="failed", result={"summary": "provider 503 Service Unavailable"})
        if attempt < policy.budget:
            assert q.claim() is None, "held back by the backoff"
            assert q.counts().get("retry") == 1
            clock.advance(policy.wait_s(attempt) + 1)  # the jittered hold is at most the ceiling
    assert q.counts() == {"dead": 1}
    assert q.claim() is None
    [row] = q.recent(limit=1)
    assert row["status"] == "dead" and row["attempts"] == policy.budget
    assert "503" in row["last_error"]
    events = [e["event"] for e in q.events(row["id"])]
    assert events.count("retry") == policy.budget - 1 and events[-1] == "dead"


def test_deterministic_failure_text_is_terminal_failed_even_when_the_worker_says_retry(make_pg_queue):
    q = make_pg_queue()
    q.enqueue(_job(kind="fix"), delivery_id="d1")
    claimed = q.claim()
    q.complete(claimed.row_id, status="failed", result={"summary": "codna fix failed (cli_error): bad input"}, retry=True)
    assert q.counts() == {"failed": 1}


def test_retry_after_from_the_worker_wins_over_the_policy_backoff(make_pg_queue):
    clock = _Clock()
    q = make_pg_queue(clock=clock)
    q.enqueue(_job(kind="fix"), delivery_id="d1")  # fix: base 120 s
    claimed = q.claim()
    q.complete(claimed.row_id, status="failed", result={"summary": "account bridge unavailable"}, retry_after_s=30)
    clock.advance(31)
    assert q.claim() is not None


# --- tenants: caps, pause, fairness -----------------------------------------------------------------------

def test_per_tenant_concurrency_cap_is_exact_and_others_keep_flowing(make_pg_queue):
    q = make_pg_queue(tenant_default_cap=2)
    for i in range(4):
        q.enqueue(_job(kind="review", ref=f"a{i}", pr=i, installation=1), delivery_id=f"noisy-{i}")
    q.enqueue(_job(kind="review", ref="q0", pr=100, installation=2), delivery_id="quiet-0")
    claimed = [q.claim() for _ in range(5)]
    got = [c for c in claimed if c is not None]
    by_tenant = {}
    for c in got:
        by_tenant[c.job.installation_id] = by_tenant.get(c.job.installation_id, 0) + 1
    assert by_tenant == {1: 2, 2: 1}                     # the noisy tenant is capped at 2, the quiet one runs
    assert q.claim() is None
    q.complete(got[0].row_id, status="done")
    nxt = q.claim()
    assert nxt is not None and nxt.job.installation_id == 1  # a freed slot goes back to the backlog


def test_tenant_override_cap_and_pause(make_pg_queue):
    q = make_pg_queue(tenant_default_cap=3)
    webhook_pg_ops.tenant_set(q, 1, actor="test", max_concurrency=1)
    for i in range(2):
        q.enqueue(_job(kind="review", ref=f"a{i}", pr=i, installation=1), delivery_id=f"t1-{i}")
    assert q.claim() is not None and q.claim() is None       # cap 1
    webhook_pg_ops.tenant_set(q, 2, actor="test", paused=True)
    q.enqueue(_job(kind="review", ref="b", pr=9, installation=2), delivery_id="t2-0")
    assert q.claim() is None                                  # paused tenants are invisible to claim
    webhook_pg_ops.tenant_set(q, 2, actor="test", paused=False)
    assert q.claim().job.installation_id == 2
    assert [t["installation_id"] for t in webhook_pg_ops.tenants(q)] == [1, 2]


def test_least_loaded_tenant_goes_first_within_a_class(make_pg_queue):
    q = make_pg_queue(tenant_default_cap=10)
    # tenant 1 already has two running; tenant 2 has none; both have a fresh review queued (tenant 1's older)
    for i in range(2):
        q.enqueue(_job(kind="review", ref=f"r{i}", pr=i, installation=1), delivery_id=f"run-{i}")
        assert q.claim() is not None
    q.enqueue(_job(kind="review", ref="older", pr=50, installation=1), delivery_id="t1-older")
    q.enqueue(_job(kind="review", ref="newer", pr=60, installation=2), delivery_id="t2-newer")
    assert q.claim().job.installation_id == 2  # fairness beats arrival order inside one class


# --- ordering: class, aging, priority, one live head -----------------------------------------------------------

def test_class_order_review_queue_secure_fix(make_pg_queue):
    q = make_pg_queue(tenant_default_cap=10)  # one tenant: the cap must not mask the order
    order = ["fix", "secure", "queue", "review"]
    for i, kind in enumerate(order):
        q.enqueue(_job(kind=kind, ref=f"s{i}", pr=i), delivery_id=f"d-{kind}")
    assert [q.claim().job.kind for _ in order] == ["review", "queue", "secure", "fix"]


def test_a_fix_that_waited_long_enough_competes_with_a_fresh_review(make_pg_queue):
    clock = _Clock()
    q = make_pg_queue(clock=clock, aging_step_s=600)
    q.enqueue(_job(kind="fix", ref="old", pr=1), delivery_id="old-fix")
    clock.advance(31 * 60)                                    # 31 minutes: class 3 - 3 = 0
    q.enqueue(_job(kind="review", ref="new", pr=2), delivery_id="new-review")
    first = q.claim()
    assert first.job.kind == "fix"                           # aging promoted it to the review class; older id wins the tie
    assert q.claim().job.kind == "review"


def test_priority_orders_within_a_class_only(make_pg_queue):
    q = make_pg_queue()
    q.enqueue(_job(kind="review", ref="a", pr=1), delivery_id="r-a")
    q.enqueue(_job(kind="review", ref="b", pr=2), delivery_id="r-b", priority=5)
    q.enqueue(_job(kind="fix", ref="c", pr=3), delivery_id="f-c", priority=100)
    assert [q.claim().delivery_id for _ in range(3)] == ["r-b", "r-a", "f-c"]


def test_reserve_priority_serves_the_merge_gating_classes_first_and_falls_back(make_pg_queue):
    q = make_pg_queue()
    q.enqueue(_job(kind="fix", ref="f", pr=1), delivery_id="fix")
    assert q.claim(reserve_priority=True).job.kind == "fix"  # nothing gating a merge is waiting
    q.enqueue(_job(kind="fix", ref="g", pr=2), delivery_id="fix2")
    q.enqueue(_job(kind="queue", ref="m", pr=3), delivery_id="mg")
    assert q.claim(reserve_priority=True).job.kind == "queue"


def test_one_live_row_per_head_collapses_a_new_delivery_for_the_same_head(make_pg_queue):
    q = make_pg_queue()
    assert q.enqueue(_job(kind="review", ref="head1", pr=7), delivery_id="d1") is True
    assert q.enqueue(_job(kind="review", ref="head1", pr=7), delivery_id="d2-new-guid") is False
    claimed = q.claim()
    assert q.enqueue(_job(kind="review", ref="head1", pr=7), delivery_id="d3-while-running") is False
    q.complete(claimed.row_id, status="done")
    assert q.enqueue(_job(kind="review", ref="head1", pr=7), delivery_id="d4-after-done") is True  # a re-review is allowed
    assert q.enqueue(_job(kind="review", ref=None, pr=7), delivery_id="comment-1") is True      # `@codna review` has no ref
    assert q.enqueue(_job(kind="review", ref=None, pr=7), delivery_id="comment-2") is True


def test_idempotency_key_is_the_trigger_not_the_delivery(make_pg_queue):
    job = _job(kind="fix", ref="h", pr=3)
    assert idempotency_key(job) == idempotency_key(WebhookJob("fix", "acme/app", ref="h", pr_number=3, installation_id=42, reason="other"))
    assert idempotency_key(job) != idempotency_key(_job(kind="fix", ref="h2", pr=3))


def test_supersede_queued_retires_stale_heads_only(make_pg_queue):
    q = make_pg_queue()
    q.enqueue(_job(kind="review", ref="old", pr=7), delivery_id="r-old")
    q.enqueue(_job(kind="fix", ref="old", pr=7), delivery_id="f-old")
    q.enqueue(_job(kind="review", ref=None, pr=7), delivery_id="comment")
    q.enqueue(_job(kind="review", ref="other", pr=8), delivery_id="other-pr")
    retired = q.supersede_queued(repo="acme/app", pr_number=7, new_ref="new")
    assert len(retired) == 2
    assert q.counts() == {"superseded": 2, "queued": 2}


# --- leases: heartbeat, loss, zombie completion --------------------------------------------------------------

def test_heartbeat_renews_only_the_owners_live_lease(make_pg_queue):
    clock = _Clock()
    q = make_pg_queue(clock=clock, lease_s=10)
    q.enqueue(_job(), delivery_id="d1")
    claimed = q.claim()
    assert q.heartbeat(claimed.row_id) is True
    clock.advance(11)
    assert webhook_pg_ops.expired_leases(q) and webhook_pg_ops.expired_leases(q)[0]["id"] == claimed.row_id
    assert q.heartbeat(claimed.row_id) is True                # still ours: renewing revives it
    assert webhook_pg_ops.expired_leases(q) == []
    other = make_pg_queue(owner="someone-else:1", clock=clock)
    assert other.heartbeat(claimed.row_id, owner="someone-else:1:t") is False


def test_a_new_head_on_enqueue_retires_waiting_older_heads_and_flags_the_running_one(make_pg_queue):
    """The SQLite queue's rule, on this backend too: push A then push B on one pull request must
    leave ONE live job for it. Measured without this (2026-09-20, load proof): A's job ran later,
    read the moved head, reviewed B -- and B's own job reviewed B again, two `codna review` runs
    on one commit. The waiting older heads are retired in the insert's own transaction; the
    running one is asked to stop (``cancel_requested``), which its worker learns at its next beat."""
    q = make_pg_queue()
    other = make_pg_queue(tenant_default_cap=3)
    assert q.enqueue(_job(ref="A", pr=7), delivery_id="dA")
    running = q.claim()
    assert running is not None and running.job.ref == "A"
    assert q.enqueue(_job(ref="B", pr=7), delivery_id="dB")                                           # waits behind A
    assert q.enqueue(_job(kind="fix", ref="B", pr=7, reason="review_comment_codna_fix"), delivery_id="dBfix")
    assert q.enqueue(_job(ref="B", pr=7), delivery_id="dB-redelivered") is False                     # same head, new GUID: nothing retired
    assert q.enqueue(_job(kind="queue", ref="G", pr=7, reason="merge_group_checks_requested"), delivery_id="dG")  # a group commit is never stale
    assert q.enqueue(_job(ref=None, pr=7, reason="comment_codna_review"), delivery_id="dNoRef")     # resolves its head when it runs
    assert q.enqueue(_job(ref="A", pr=8, installation=43), delivery_id="dA8")                       # another pull request
    assert q.enqueue(_job(ref="C", pr=7), delivery_id="dC")
    by = {r["delivery_id"]: r for r in q.recent(limit=20)}
    assert by["dB"]["status"] == "superseded" and by["dBfix"]["status"] == "superseded"
    assert by["dB"]["result"] == {"summary": "superseded by C"} and by["dBfix"]["last_error"] == "superseded"
    assert {by[d]["status"] for d in ("dG", "dNoRef", "dA8", "dC")} == {"queued"}
    # The running row was asked to stop when B arrived and keeps that first reason (a later head
    # does not overwrite it); its worker learns at its next beat.
    assert by["dA"]["status"] == "running" and q.job(running.row_id)["cancel_requested"] == "superseded by B"
    assert q.heartbeat(running.row_id) is False
    assert [e["event"] for e in q.events(running.row_id)][-1] == "cancel_requested"
    assert [e["event"] for e in q.events(by["dB"]["id"])][-1] == "superseded"
    claimed = [c.delivery_id for c in (other.claim() for _ in range(5)) if c is not None]
    assert set(claimed) == {"dA8", "dNoRef", "dC"}                                                    # tenant 42: A + two more = its cap of 3
    assert not {"dB", "dBfix"} & set(claimed)


def test_the_worker_holding_a_superseded_head_learns_it_through_its_lease(make_pg_queue):
    """Cross-machine: the push lands at an ingress, the review runs on a worker elsewhere. The
    worker's lease keeper reports the row's reason, which webhook_service turns into the same
    cancellation the ingress applies to its own process."""
    ingress = make_pg_queue(owner="ingress:1")
    lost = []
    worker = make_pg_queue(owner="worker-b:1", heartbeat_s=0.05, lease_s=5,
                           on_lease_lost=lambda rid, why: lost.append((rid, why)))
    assert ingress.enqueue(_job(ref="A", pr=7), delivery_id="dA")
    claimed = worker.claim()
    assert ingress.enqueue(_job(ref="B", pr=7), delivery_id="dB")
    pause = threading.Event()
    for _ in range(200):
        if lost:
            break
        pause.wait(0.05)
    assert lost == [(claimed.row_id, "superseded by B")]
    assert worker.lease_lost_reason(claimed.row_id) == "superseded by B"


def test_unless_pending_also_skips_a_head_that_already_has_its_verdict(make_pg_queue):
    """The window the supersede leaves open: an older head's job ends normally before its worker
    learns it was superseded, sees the head moved, and re-queues the new head -- which was already
    reviewed. ``unless_pending`` treats a finished (``done``) row as covering the head; a plain
    enqueue (a `@codna review` re-request) still queues it."""
    clock = _Clock()
    q = make_pg_queue(clock=clock)
    assert q.enqueue(_job(ref="B", pr=7), delivery_id="dB")
    claimed = q.claim()
    q.complete(claimed.row_id, status="done", result={"summary": "reviewed"})
    assert q.enqueue(_job(ref="B", pr=7, reason="head_moved_during_review"), delivery_id="head-moved-1-B", unless_pending=True) is False
    clock.advance(23 * 3600)
    assert q.enqueue(_job(ref="B", pr=7, reason="head_moved_during_review"), delivery_id="head-moved-1-B-later", unless_pending=True) is False
    clock.advance(2 * 3600)                                                                    # a verdict older than 24 h covers nothing
    assert q.enqueue(_job(ref="B", pr=7, reason="head_moved_during_review"), delivery_id="head-moved-1-B-old", unless_pending=True) is True
    q.complete(q.claim().row_id, status="done", result={"summary": "reviewed again"})
    assert q.enqueue(_job(ref="B", pr=7, reason="head_moved_during_review"), delivery_id="head-moved-1-B-again", unless_pending=True) is False
    assert q.enqueue(_job(ref="F", pr=9), delivery_id="dF")                                    # a head whose only row FAILED
    failed = q.claim()
    assert failed is not None and failed.delivery_id == "dF"
    q.complete(failed.row_id, status="failed", result={"error": "cli_error"}, retry=False)
    assert q.job(failed.row_id)["status"] == "failed"
    assert q.enqueue(_job(ref="F", pr=9), delivery_id="head-moved-2-F", unless_pending=True) is True   # not covered
    assert q.enqueue(_job(ref="B", pr=7, reason="comment_codna_review"), delivery_id="dB-again") is True
    assert q.counts() == {"done": 2, "failed": 1, "queued": 2}


def test_racing_requeues_for_one_reviewed_head_insert_at_most_one_row(make_pg_queue):
    """Two requeuers (two reapers, a mover and a reaper) asking ``unless_pending`` for the same
    head at the same time: the check runs inside the insert's transaction under a per-head
    advisory lock, so only one can insert -- and none when the head already has its verdict."""
    q = make_pg_queue(pool_max=12)
    assert q.enqueue(_job(ref="B", pr=7), delivery_id="dB")
    q.complete(q.claim().row_id, status="done", result={"summary": "reviewed"})
    results, barrier = [], threading.Barrier(8)

    def _requeue(i):
        barrier.wait()
        results.append(q.enqueue(_job(ref="B", pr=7, reason="head_moved_during_review"), delivery_id=f"requeue-{i}", unless_pending=True))

    threads = [threading.Thread(target=_requeue, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results == [False] * 8 and q.counts() == {"done": 1}
    # and for a head with NO verdict yet, exactly one of the racers inserts
    results.clear()
    barrier = threading.Barrier(8)
    threads = [threading.Thread(target=lambda i=i: (barrier.wait(), results.append(
        q.enqueue(_job(ref="C", pr=7, reason="head_moved_during_review"), delivery_id=f"requeue-C-{i}", unless_pending=True))))
        for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert results.count(True) == 1 and q.counts() == {"done": 1, "queued": 1}


def test_the_reaper_does_not_queue_a_current_head_that_was_already_reviewed(make_pg_queue):
    """Same window, reaper side: a worker dies while finishing an older head after the current head
    was already reviewed. The stale row is cancelled and its Check Run closed; the current head is
    NOT queued a second time."""
    clock = _Clock()
    ingress = make_pg_queue(owner="ingress:1", clock=clock, lease_s=10)
    worker = make_pg_queue(owner="worker-a:1", clock=clock, lease_s=10)
    ingress.enqueue(_job(ref="newhead", pr=7), delivery_id="dNew")
    reviewed = worker.claim()
    worker.complete(reviewed.row_id, status="done", result={"summary": "reviewed newhead"})
    ingress.enqueue(_job(ref="oldhead", pr=7, reason="late_delivery"), delivery_id="dOld")   # a late event for the older head
    stale = worker.claim()
    worker.record_check_run(stale.row_id, 77)
    worker.abandon_leases()
    clock.advance(11)
    github = _GitHub(head="newhead", open_runs={77})
    reaper = Reaper(ingress, app_id="APP", private_key="pem", github=github, environ={}, leader=_Held())
    assert reaper.reap_expired()["superseded"] == 1
    assert ingress.job(stale.row_id)["status"] == "cancelled"
    assert [c[1:3] for c in github.of("complete")] == [(77, "cancelled")]
    assert worker.claim() is None                                   # newhead already has its verdict: nothing queued
    assert ingress.counts() == {"done": 1, "cancelled": 1}


def test_cancel_requested_fails_the_heartbeat_and_the_lost_lease_is_reported(make_pg_queue):
    lost = []
    q = make_pg_queue(heartbeat_s=0.05, lease_s=5, on_lease_lost=lambda rid, why: lost.append((rid, why)))
    q.enqueue(_job(), delivery_id="d1")
    claimed = q.claim()
    out = webhook_pg_ops.cancel(q, claimed.row_id, actor="ops", reason="stale head")
    assert out["outcome"] == "cancel_requested"
    deadline = threading.Event()
    for _ in range(100):
        if lost:
            break
        deadline.wait(0.05)
    assert lost and lost[0][0] == claimed.row_id and "stale head" in lost[0][1]
    assert q.lease_lost_reason(claimed.row_id) is not None


def test_a_zombie_completion_after_the_lease_was_reaped_is_ignored(make_pg_queue):
    clock = _Clock()
    a = make_pg_queue(owner="machine-a:1", clock=clock, lease_s=5)
    b = make_pg_queue(owner="machine-b:1", clock=clock, lease_s=5)
    a.enqueue(_job(), delivery_id="d1")
    first = a.claim()
    a.abandon_leases()                                        # machine A "dies"
    clock.advance(6)
    assert webhook_pg_ops.requeue_interrupted(a, first.row_id, cause="lease_expired", hold_s=0, actor="reaper") is True
    second = b.claim()
    assert second is not None and second.row_id == first.row_id and second.attempts == 2
    a.complete(first.row_id, status="done", result={"summary": "zombie says done"})   # A comes back from the dead
    assert b.row_state(first.row_id) == ("running", 2)      # B still owns it, untouched
    b.complete(first.row_id, status="done", result={"summary": "real"})
    assert b.counts() == {"done": 1}
    events = [e["event"] for e in b.events(first.row_id)]
    assert "complete_ignored" in events and events.count("completed") == 1


# --- crash recovery through the reaper: the check run is completed exactly once ------------------------------------

def test_worker_killed_mid_job_lease_expires_another_worker_finishes_check_completed_once(make_pg_queue):
    clock = _Clock()
    ingress = make_pg_queue(owner="ingress:1", clock=clock, lease_s=30)
    a = make_pg_queue(owner="worker-a:1", clock=clock, lease_s=30)
    b = make_pg_queue(owner="worker-b:1", clock=clock, lease_s=30)
    github = _GitHub(head="sha1", open_runs={9001})
    reaper = Reaper(ingress, app_id="APP", private_key="pem", github=github, environ={}, leader=_Held())

    ingress.enqueue(_job(kind="review", ref="sha1", pr=7), delivery_id="d1")
    first = a.claim()
    a.record_check_run(first.row_id, 9001)                    # the job opened its Check Run
    a.abandon_leases()                                        # SIGKILL: the heartbeat thread is gone
    assert reaper.reap_expired()["resumed"] == 0              # lease still valid: nothing to do
    clock.advance(31)
    out = reaper.reap_expired()
    assert out["resumed"] == 1
    assert github.of("complete") == [("complete", 9001, "cancelled", github.of("complete")[0][3])]
    assert "re-queued automatically as attempt 2" in github.of("complete")[0][3]
    assert b.claim() is None, "held back by the review backoff"
    clock.advance(webhook_retry.policy_for("review").wait_s(1) + 1)
    second = b.claim()
    assert second is not None and second.row_id == first.row_id and second.attempts == 2
    b.record_check_run(second.row_id, 9002)
    b.complete(second.row_id, status="done", result={"summary": "ok"})
    row = ingress.job(first.row_id)
    assert row["status"] == "done" and row["resumes"] == 1 and row["check_run_id"] == 9002
    assert github.of("complete") == [github.of("complete")[0]]  # exactly one stale run completed
    assert reaper.reap_expired()["resumed"] == 0               # nothing left to reap
    a.complete(first.row_id, status="failed", result={"summary": "zombie"})  # A's late outcome changes nothing
    assert ingress.job(first.row_id)["status"] == "done"


def test_second_interruption_dead_letters_and_notifies_exactly_once(make_pg_queue):
    clock = _Clock()
    ingress = make_pg_queue(owner="ingress:1", clock=clock, lease_s=10)
    worker = make_pg_queue(owner="worker-a:1", clock=clock, lease_s=10)
    github = _GitHub(head="sha1", open_runs={1, 2})
    reaper = Reaper(ingress, app_id="APP", private_key="pem", github=github, environ={}, leader=_Held())
    ingress.enqueue(_job(kind="review", ref="sha1", pr=7), delivery_id="d1")
    for run_id in (1, 2):
        claimed = worker.claim()
        assert claimed is not None
        worker.record_check_run(claimed.row_id, run_id)
        worker.abandon_leases()
        clock.advance(11)
        reaper.reap_expired()
        clock.advance(webhook_retry.policy_for("review").wait_s(1) + 1)
    row = ingress.job(claimed.row_id)
    assert row["status"] == "dead" and row["resumes"] == MAX_RESUMES and row["last_error"] == "interrupted_after_resume"
    assert [c[1] for c in github.of("complete")] == [1, 2]      # each stale run closed once
    assert reaper.notify_dead_letters() == 1
    assert reaper.notify_dead_letters() == 0                    # told once
    [(kind, cid, conclusion, text)] = github.of("update")
    assert cid == 2 and conclusion == "neutral" and "not a verdict on your code" in text
    assert ingress.job(claimed.row_id)["posted"].keys() == {"dead_letter"}


def test_superseded_head_is_cancelled_and_the_current_head_queued(make_pg_queue):
    clock = _Clock()
    ingress = make_pg_queue(owner="ingress:1", clock=clock, lease_s=10)
    worker = make_pg_queue(owner="worker-a:1", clock=clock, lease_s=10)
    github = _GitHub(head="newhead", open_runs={77})
    reaper = Reaper(ingress, app_id="APP", private_key="pem", github=github, environ={}, leader=_Held())
    ingress.enqueue(_job(kind="review", ref="oldhead", pr=7), delivery_id="d1")
    claimed = worker.claim()
    worker.record_check_run(claimed.row_id, 77)
    worker.abandon_leases()
    clock.advance(11)
    assert reaper.reap_expired()["superseded"] == 1
    assert ingress.job(claimed.row_id)["status"] == "cancelled"
    assert [c[1:3] for c in github.of("complete")] == [(77, "cancelled")]
    nxt = worker.claim()
    assert nxt is not None and nxt.job.ref == "newhead" and nxt.job.pr_number == 7
    assert nxt.delivery_id == f"reaper-{claimed.row_id}-newhead"


def test_recover_stale_touches_only_this_machines_rows(make_pg_queue):
    a = make_pg_queue(owner="machine-a:100")
    b = make_pg_queue(owner="machine-b:200")
    a.enqueue(_job(ref="a", pr=1), delivery_id="a1")
    b.enqueue(_job(ref="b", pr=2), delivery_id="b1")
    ra, rb = a.claim(), b.claim()
    a.abandon_leases()
    a_restarted = make_pg_queue(owner="machine-a:101")        # same machine, new pid
    assert a_restarted.recover_stale() == {"requeued": 1, "failed": 0}
    assert a_restarted.row_state(ra.row_id) == ("retry", 1)   # queued again (attempts unchanged, resumes+1)
    assert a_restarted.row_state(rb.row_id) == ("running", 1)  # machine B's job is alive and untouched
    assert a_restarted.job(ra.row_id)["resumes"] == 1


class _Held:
    held = True

    def try_acquire(self):
        return True

    def release(self):
        return None


# --- many claimers, never a double claim ---------------------------------------------------------------------

def _claim_worker(url, schema, owner, n_threads, sink):
    """Runs in a CHILD PROCESS: n_threads claimers drain the queue, each claim recorded with the
    per-tenant running count observed right after it (under the advisory lock the count is exact)."""
    queue = PostgresQueue(url, schema=schema, owner=owner, heartbeat_s=0, pool_max=n_threads + 1, tenant_default_cap=3)
    seen = []
    lock = threading.Lock()

    def loop():
        while True:
            claimed = queue.claim()
            if claimed is None:
                return
            with queue.connection() as conn:
                running = conn.execute(
                    f"SELECT count(*) AS c FROM {schema}.jobs WHERE status='running' AND installation_id=%s",
                    (claimed.job.installation_id,)).fetchone()["c"]
            with lock:
                seen.append((claimed.row_id, claimed.job.installation_id, int(running)))
            queue.complete(claimed.row_id, status="done")

    threads = [threading.Thread(target=loop) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    queue.close()
    sink.put(seen)


def test_claimers_across_threads_and_processes_never_double_claim_and_respect_caps(pg_url, pg_schema, make_pg_queue):
    q = make_pg_queue(tenant_default_cap=3)
    tenants = [1, 2, 3, 4, 5]
    rows = 0
    for i in range(300):
        inst = tenants[i % len(tenants)]
        if q.enqueue(_job(kind=["review", "fix", "secure"][i % 3], ref=f"s{i}", pr=i, installation=inst), delivery_id=f"d{i}"):
            rows += 1
    assert rows == 300
    ctx = multiprocessing.get_context("spawn")
    sink = ctx.Queue()
    procs = [ctx.Process(target=_claim_worker, args=(pg_url, pg_schema, f"proc-{n}:1", 6, sink)) for n in range(2)]
    for p in procs:
        p.start()
    results = []
    _claim_worker(pg_url, pg_schema, "main:1", 4, _ListSink(results))   # this process claims too
    for _ in procs:
        results.extend(sink.get(timeout=120))
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    claimed_ids = [r[0] for r in results]
    assert len(claimed_ids) == rows and len(set(claimed_ids)) == rows   # every row exactly once, none lost
    assert max(r[2] for r in results) <= 3                              # the per-tenant cap held at every claim
    assert q.counts() == {"done": rows}


class _ListSink:
    def __init__(self, target):
        self._target = target

    def put(self, value):
        self._target.extend(value)


# --- zero job loss across a backend switch, both directions ---------------------------------------------------------

def test_sqlite_backlog_is_imported_once_and_every_job_runs_exactly_once(tmp_path, make_pg_queue):
    sqlite = WebhookQueue(tmp_path / "queue.db")
    # Built in the order the SQLite queue claims (reviews first, then arrival), so each step's
    # claim() returns exactly the row the step wants: running, done, retry, then three queued.
    steps = [("review", "r2", 4, "running"), ("fix", "f2", 5, "done"), ("secure", "s1", 3, "retry"),
             ("review", "r1", 1, "queued"), ("fix", "f1", 2, "queued"), ("queue", "m1", 6, "queued")]
    for kind, ref, pr, wanted in steps:
        assert sqlite.enqueue(_job(kind=kind, ref=ref, pr=pr, reason="test"), delivery_id=f"d-{ref}")
        if wanted == "queued":
            continue
        claimed = sqlite.claim()
        assert claimed.delivery_id == f"d-{ref}"
        if wanted == "done":
            sqlite.complete(claimed.row_id, status="done")
        elif wanted == "retry":
            sqlite.complete(claimed.row_id, status="failed")
    assert {r["delivery_id"]: r["status"] for r in sqlite.export_rows()} == {
        "d-r2": "running", "d-s1": "retry", "d-r1": "queued", "d-f1": "queued", "d-m1": "queued"}
    in_flight = {f"d-{ref}" for _, ref, _, wanted in steps if wanted != "done"}
    pg = make_pg_queue()
    assert pg.import_sqlite_backlog(sqlite) == {"imported": len(in_flight), "already_present": 0}
    assert pg.import_sqlite_backlog(sqlite) == {"imported": 0, "already_present": 0}   # idempotent: nothing left in flight
    assert sqlite.counts() == {"migrated": len(in_flight), "done": 1}
    assert sqlite.claim() is None                                                      # the file runs nothing any more
    ran = []
    while (claimed := pg.claim()) is not None:
        ran.append(claimed.delivery_id)
        pg.complete(claimed.row_id, status="done")
    assert sorted(ran) == sorted(in_flight)                                            # each in-flight job exactly once
    rows = {r["delivery_id"]: r for r in pg.recent(limit=50)}
    assert rows["d-r2"]["resumes"] == 1 and rows["d-r2"]["attempts"] == 2      # was running: one interruption, attempt carried
    assert rows["d-s1"]["attempts"] == 2                                       # failed once in the file, ran once here
    assert all(r["src_sqlite_id"] is not None for r in (pg.job(rows[d]["id"]) for d in in_flight))


def test_rollback_exports_postgres_backlog_into_the_file_and_every_job_runs_exactly_once(tmp_path, make_pg_queue):
    pg = make_pg_queue()
    for i in range(4):
        assert pg.enqueue(_job(kind=["review", "fix"][i % 2], ref=f"h{i}", pr=i), delivery_id=f"d{i}")
    running = pg.claim()
    done = pg.claim()
    pg.complete(done.row_id, status="done")
    sqlite = WebhookQueue(tmp_path / "queue.db")
    assert pg.export_backlog(sqlite) == {"exported": 3}
    assert pg.counts() == {"exported": 3, "done": 1}
    assert pg.claim() is None
    ran = []
    while (claimed := sqlite.claim()) is not None:
        ran.append(claimed.delivery_id)
        sqlite.complete(claimed.row_id, status="done")
    assert sorted(ran) == sorted(f"d{i}" for i in range(4) if f"d{i}" != done.delivery_id)
    assert running.delivery_id in ran
    assert pg.export_backlog(sqlite) == {"exported": 0}                                # idempotent


def _claim_specific(sqlite, row_id):
    """Claim rows until the wanted one comes out (SQLite claims in class order); the others are
    completed as done so the fixture ends in exactly the shape the test asked for."""
    for _ in range(20):
        claimed = sqlite.claim()
        if claimed is None:
            raise AssertionError(f"row {row_id} never came out of claim()")
        if claimed.row_id == row_id:
            return row_id
        sqlite.complete(claimed.row_id, status="done")
    raise AssertionError("too many claims")


def test_python_sees_the_same_kinds_the_schema_checks(pg_url, pg_schema):
    with connect(pg_url) as conn:
        checks = [r[0] for r in conn.execute(
            "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace WHERE n.nspname = %s AND t.relname = 'jobs' "
            "AND c.contype = 'c'", (pg_schema,)).fetchall()]
    [kind_check] = [c for c in checks if c.startswith("CHECK ((kind")]
    for kind in KINDS:
        assert f"'{kind}'" in kind_check


# --- connection settings reach the server without a startup `options` parameter --------------------

def test_pooled_connections_carry_the_timeout_and_name_via_set_not_startup_options(make_pg_queue):
    """Fly Managed Postgres serves its PgBouncer endpoint, which refuses libpq's `options` startup
    parameter ("unsupported startup parameter in options: statement_timeout", 2026-09-20). The
    settings must therefore arrive as SET on each new connection -- and still be in force."""
    q = make_pg_queue(statement_timeout_ms=7_000)
    with q.connection() as conn:
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "7s"
        assert conn.execute("SHOW application_name").fetchone()["application_name"] == "codna-webhook"
        # the pool's connect kwargs never carry a startup `options` string
        assert "options" not in q._pool.kwargs
