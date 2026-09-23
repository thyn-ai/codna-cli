"""Worker liveness + the reaper (codna.webhook_lease): registration heartbeats and the drain
handshake, an operator cancel finishing through the reaper, dead letters told once with the
configured conclusion (check run, comment-fix thread, issue), the post-outage retry, and the
leader lock. Postgres-backed (CODNA_TEST_DATABASE_URL); the reaper's lease-expiry verdicts
themselves are in test_webhook_pg_queue.py."""
from __future__ import annotations

import pytest

from codna import webhook_pg_ops
from codna.webhook import WebhookJob
from codna.webhook_lease import Reaper, WorkerRegistration, dead_letter_conclusion
from codna.webhook_pg_ops import LeaderLock

pytestmark = pytest.mark.usefixtures("pg_url")


class _GitHub:
    def __init__(self):
        self.calls = []

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        return "tok"

    def pull_request_head_sha(self, repo, token, number):
        return None

    def complete_check_run_if_open(self, repo, token, cid, *, conclusion, summary, name):
        self.calls.append(("complete", cid, conclusion))
        return True

    def update_check_run(self, repo, token, cid, *, conclusion, summary, name):
        self.calls.append(("update", cid, conclusion, summary))

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.calls.append(("create", head_sha))
        return 4242

    def post_review_comment_reply(self, repo, token, pr, in_reply_to, body):
        self.calls.append(("reply", pr, in_reply_to, body))

    def post_issue_comment(self, repo, token, issue, body):
        self.calls.append(("comment", issue, body))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


class _Held:
    held = True

    def try_acquire(self):
        return True

    def release(self):
        self.held = False


def _job(kind="review", ref="sha1", pr=7, **kw):
    return WebhookJob(kind, "acme/app", ref=ref, pr_number=pr, installation_id=42, reason="test", **kw)


def _reaper(queue, github, environ=None, **kw):
    return Reaper(queue, app_id="APP", private_key="pem", github=github, environ=environ or {}, leader=_Held(), **kw)


# --- registration + drain handshake -------------------------------------------------------------------

def test_registration_heartbeats_and_acknowledges_a_drain(make_pg_queue):
    q = make_pg_queue(owner="m1:1")
    drained = []
    reg = WorkerRegistration(q, role="worker", slots=3, machine_id="m1", region="iad", image="img:1",
                             on_drained=lambda: drained.append(True))
    reg.register()
    [w] = webhook_pg_ops.workers(q)
    assert (w["owner"], w["machine_id"], w["region"], w["slots"], w["busy"], w["draining"], w["stale"]) == \
        ("m1:1", "m1", "iad", 3, 0, False, False)
    q.enqueue(_job(), delivery_id="d1")
    claimed = q.claim()
    row = reg.beat()
    assert row["busy"] == 1 and not q.draining
    webhook_pg_ops.set_draining(q, "m1:1", True, actor="scaler")
    row = reg.beat()
    assert row["draining"] and row["drain_acked_at"] is not None       # acknowledged on the next beat
    assert q.draining and q.claim() is None                             # stops claiming at once
    assert drained == []                                                # ...but is not drained while busy
    q.complete(claimed.row_id, status="done")
    reg.beat()
    assert drained == [True] and reg.drained                            # idle + draining = done, exit is the caller's
    reg.stop()
    assert webhook_pg_ops.workers(q) == []


def test_a_pruned_worker_re_registers_on_its_next_beat(make_pg_queue):
    q = make_pg_queue(owner="m1:1")
    reg = WorkerRegistration(q, role="worker", slots=2)
    reg.register()
    webhook_pg_ops.deregister_worker(q)
    assert reg.beat()["owner"] == "m1:1"


def test_scaler_inputs_count_only_fresh_non_draining_slots(make_pg_queue):
    from datetime import datetime, timedelta, timezone

    now = [datetime.now(timezone.utc)]
    q = make_pg_queue(owner="m1:1", clock=lambda: now[0])
    other = make_pg_queue(owner="m2:1", clock=lambda: now[0])
    WorkerRegistration(q, role="worker", slots=3).register()
    WorkerRegistration(other, role="worker", slots=3).register()
    q.enqueue(_job(), delivery_id="d1")
    q.claim()
    webhook_pg_ops.worker_heartbeat(q, busy=1)
    inputs = webhook_pg_ops.scaler_inputs(q)
    assert (inputs["runnable"], inputs["free_slots"], inputs["busy_slots"]) == (0, 5, 1)
    now[0] += timedelta(seconds=45)                                      # both heartbeats go stale
    inputs = webhook_pg_ops.scaler_inputs(q)
    assert inputs["free_slots"] == 0 and inputs["workers"] == []
    webhook_pg_ops.worker_heartbeat(q, busy=1)
    webhook_pg_ops.set_draining(q, "m1:1", True, actor="t")
    assert webhook_pg_ops.scaler_inputs(q)["free_slots"] == 0            # a draining worker offers no slots


# --- cancel through the reaper -------------------------------------------------------------------------------

def test_operator_cancel_of_a_running_job_is_finished_by_the_reaper_and_the_check_closed(make_pg_queue):
    from datetime import datetime, timedelta, timezone

    now = [datetime.now(timezone.utc)]
    q = make_pg_queue(owner="m1:1", clock=lambda: now[0], lease_s=10)
    github = _GitHub()
    reaper = _reaper(q, github)
    q.enqueue(_job(), delivery_id="d1")
    claimed = q.claim()
    q.record_check_run(claimed.row_id, 77)
    assert webhook_pg_ops.cancel(q, claimed.row_id, actor="ops:angel", reason="stale head")["outcome"] == "cancel_requested"
    assert q.heartbeat(claimed.row_id) is False                          # the worker learns on its next beat
    now[0] += timedelta(seconds=11)                                      # ...and once the lease lapses
    assert reaper.reap_expired()["cancelled"] == 1
    row = q.job(claimed.row_id)
    assert row["status"] == "cancelled" and "stale head" in row["last_error"]
    assert github.of("complete") == [("complete", 77, "cancelled")]
    actors = [(e["event"], e["actor"]) for e in q.events(claimed.row_id)]
    assert ("cancel_requested", "ops:angel") in actors and any(e == "cancelled" for e, _ in actors)


# --- dead letters -------------------------------------------------------------------------------------------

def test_dead_letter_conclusion_policy():
    assert dead_letter_conclusion(None, {}) == "neutral"
    assert dead_letter_conclusion(None, {"CODNA_WEBHOOK_DEAD_LETTER_CONCLUSION": "action_required"}) == "action_required"
    assert dead_letter_conclusion("action_required", {}) == "action_required"          # tenant override wins
    assert dead_letter_conclusion("nonsense", {"CODNA_WEBHOOK_DEAD_LETTER_CONCLUSION": "bogus"}) == "neutral"


def _dead(q, job, delivery_id):
    """Spend the kind's whole transient budget so the row is dead."""
    q.enqueue(job, delivery_id=delivery_id)
    while (claimed := q.claim()) is not None:
        q.complete(claimed.row_id, status="failed", result={"summary": "provider 503"})
    [row] = [r for r in q.recent(limit=20) if r["delivery_id"] == delivery_id]
    assert row["status"] == "dead"
    return row["id"]


def _zero_backoff():
    from codna.webhook_pg_schema import KINDS
    from codna.webhook_retry import RetryPolicy, policy_for

    return {k: RetryPolicy(budget=policy_for(k).budget, base_s=0.0, factor=1.0, cap_s=0.0) for k in KINDS}


def test_dead_review_completes_its_check_with_the_env_conclusion_and_a_tenant_override(make_pg_queue):
    q = make_pg_queue(retry_policy=_zero_backoff())
    github = _GitHub()
    reaper = _reaper(q, github, environ={"CODNA_WEBHOOK_DEAD_LETTER_CONCLUSION": "action_required"})
    rid = _dead(q, _job(kind="review"), "d1")
    q.record_check_run(rid, 900)
    assert reaper.notify_dead_letters() == 1
    [(_, cid, conclusion, text)] = github.of("update")
    assert cid == 900 and conclusion == "action_required" and f"job #{rid}" in text and "@codna review" in text
    webhook_pg_ops.tenant_set(q, 42, actor="t", dead_letter_conclusion="neutral")
    rid2 = _dead(q, _job(kind="review", ref="sha2", pr=8), "d2")
    assert reaper.notify_dead_letters() == 1
    assert github.of("create") == [("create", "sha2")]                  # no run recorded: one is created, then completed
    assert github.of("update")[-1][2] == "neutral" and q.job(rid2)["check_run_id"] == 4242


def test_dead_comment_fix_replies_in_thread_and_dead_issue_fix_comments_once(make_pg_queue):
    q = make_pg_queue(retry_policy=_zero_backoff())
    github = _GitHub()
    reaper = _reaper(q, github)
    rid = _dead(q, _job(kind="fix", context={"in_reply_to_id": 555, "head_ref": "feature"}), "d-thread")
    rid2 = _dead(q, WebhookJob("fix", "acme/app", installation_id=42, issue_number=12, reason="labeled_codna_fix"), "d-issue")
    assert reaper.notify_dead_letters() == 2
    assert [c[:3] for c in github.of("reply")] == [("reply", 7, 555)] and "@codna fix" in github.of("reply")[0][3]
    assert [c[:2] for c in github.of("comment")] == [("comment", 12)]
    assert reaper.notify_dead_letters() == 0                            # told once
    assert set(q.job(rid)["posted"]) == set(q.job(rid2)["posted"]) == {"dead_letter"}


def test_dead_rows_from_an_outage_run_again_once_postgres_has_been_healthy_long_enough(make_pg_queue):
    q = make_pg_queue(retry_policy=_zero_backoff())
    clock = [0.0]
    reaper = _reaper(q, _GitHub(), outage_retry_after_s=300, clock=lambda: clock[0])
    outage = _dead(q, _job(kind="review", ref="a", pr=1), "d-outage")
    q.enqueue(_job(kind="review", ref="b", pr=2), delivery_id="d-code")
    while (c := q.claim()) is not None:
        q.complete(c.row_id, status="failed", result={"summary": "the agent produced no findings JSON"})  # dead, not an outage
    assert q.counts() == {"dead": 2}
    assert reaper.retry_after_outage() == 0                             # healthy since just now: wait
    clock[0] += 301
    assert reaper.retry_after_outage() == 1                             # only the outage-shaped one
    assert q.job(outage)["status"] == "queued" and q.job(outage)["attempts"] == 0
    assert reaper.retry_after_outage() == 0                             # exactly once
    assert "outage_retry" in q.job(outage)["posted"]


def test_tick_runs_every_duty_and_reports_counts(make_pg_queue):
    q = make_pg_queue(retry_policy=_zero_backoff())
    reaper = _reaper(q, _GitHub())
    _dead(q, _job(), "d1")
    out = reaper.tick()
    assert out["dead_letters_notified"] == 1 and out["resumed"] == 0 and reaper.ticks == 1
    assert set(out) >= {"resumed", "superseded", "dead", "cancelled", "skipped", "dead_letters_notified",
                        "outage_retries", "workers_pruned"}


# --- leader election -----------------------------------------------------------------------------------------

def test_leader_lock_is_exclusive_per_purpose_and_released_on_close(make_pg_queue):
    a, b = make_pg_queue(owner="a:1"), make_pg_queue(owner="b:1")
    la, lb, other = LeaderLock(a, "reaper"), LeaderLock(b, "reaper"), LeaderLock(b, "scaler")
    try:
        assert la.try_acquire() is True and la.held
        assert lb.try_acquire() is False and not lb.held               # one leader per purpose
        assert la.try_acquire() is True                                 # re-asserting is cheap and stays true
        assert other.try_acquire() is True                              # a different purpose is a different lock
        la.release()
        assert lb.try_acquire() is True                                 # the lock moves as soon as it is free
    finally:
        for lock in (la, lb, other):
            lock.release()


def test_leader_lock_lives_inside_the_holders_open_transaction(make_pg_queue):
    """Fly Managed Postgres's pooled URL is PgBouncer in TRANSACTION mode: the server connection is
    handed to whichever client runs the next statement, so a session-level lock taken through it
    belongs to the backend, not to the client, and a second client landing on that backend gets it
    too (two ingress machines both held the scaler lock, 2026-09-21). The lock must therefore be
    transaction-level and the holder's transaction must stay OPEN (the pooler pins the server
    connection to a client for exactly that long); a loser must leave no transaction open."""
    from psycopg.pq import TransactionStatus

    a, b = make_pg_queue(owner="a:1"), make_pg_queue(owner="b:1")
    la, lb = LeaderLock(a, "scaler"), LeaderLock(b, "scaler")
    try:
        assert la.try_acquire() is True
        assert la._conn.info.transaction_status == TransactionStatus.INTRANS   # the lock's transaction is open ...
        granted = la._conn.execute(
            "SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND granted AND pid = pg_backend_pid()").fetchone()["n"]
        assert granted == 1                                                    # ... and the lock is on this backend
        assert lb.try_acquire() is False
        assert lb._conn.info.transaction_status == TransactionStatus.IDLE      # a loser holds no transaction (no pinned server connection)
        assert la.try_acquire() is True                                        # re-asserting stays inside the same transaction
        assert la._conn.info.transaction_status == TransactionStatus.INTRANS
        # Transaction-level means exactly that: when the transaction ends -- a pooler or the server
        # closing it, not only release() -- the lock is gone and the next poll elects someone else.
        la._conn.rollback()
        assert lb.try_acquire() is True
        assert la.try_acquire() is False                                       # the old holder learns it lost on its very next poll
        assert not la.held
    finally:
        for lock in (la, lb):
            lock.release()
