"""Scheduling: supersede on a new head, merge-gating priority with a reserved thread, the
worker-enforced job deadline, and what /healthz says about saturation.

The night this closes (2026-09-19/20): `codna fix` on thyn-ai/security-toolchain#14 head 92730476
started 01:46:13Z and held one of two threads for ~30 minutes while the other reviewed serially;
five PR heads pushed 02:00-02:06 across four repositories got no `codna review` check for 15+
minutes (a REQUIRED check on the public rulesets, so every one of them was blocking a merge); and
reviews of superseded heads ran to completion anyway -- thyn-ai/mojo-kernels#32 reviewed ba252b20
at 01:58:30 and 04e3c8a5 at 02:01:57 while the head had already moved to 21e3610.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

import codna.webhook_control as control_module
import codna.webhook_worker as worker_module
from codna.webhook import WebhookJob
from codna.webhook_control import (
    JobControl,
    PRIORITY_KINDS,
    SUPERSEDABLE_KINDS,
    cancel_jobs,
    default_concurrency,
    health_payload,
)
from codna.webhook_pool import WorkerPool
from codna.webhook_queue import QueuedJob, WebhookQueue
from codna.webhook_worker import JobResult


def _q(tmp_path):
    return WebhookQueue(tmp_path / "queue.db")


def _review(repo="acme/app", pr=7, ref="head1"):
    return WebhookJob("review", repo, ref=ref, pr_number=pr, installation_id=42,
                      reason="pull_request_synchronize")


def _fix(repo="acme/app", pr=7, ref="head1"):
    return WebhookJob("fix", repo, ref=ref, pr_number=pr, installation_id=42,
                      reason="check_suite_failure")


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- 1. the queue keeps the newest head only -------------------------------------------------

def test_a_new_head_supersedes_the_queued_review_of_the_older_one(tmp_path):
    q = _q(tmp_path)
    assert q.enqueue(_review(ref="ba252b20"), delivery_id="d1") is True
    assert q.enqueue(_review(ref="04e3c8a5"), delivery_id="d2") is True
    assert q.enqueue(_review(ref="21e3610a"), delivery_id="d3") is True

    claimed = q.claim()
    assert claimed is not None and claimed.job.ref == "21e3610a"   # only the current head runs
    assert q.claim() is None                                       # the older two are retired
    # Derived from what was enqueued, not hardcoded: every head but the last.
    assert q.counts().get("superseded") == 2


def test_a_moving_head_also_retires_a_queued_fix_for_the_stale_head(tmp_path):
    """A fix computed against a stale head must not open a pull request on a base that has moved."""
    q = _q(tmp_path)
    assert q.enqueue(_fix(ref="92730476"), delivery_id="suite-1") is True
    assert q.enqueue(_review(ref="deadbeef"), delivery_id="push-1") is True

    kinds = []
    while (claimed := q.claim()) is not None:
        kinds.append((claimed.job.kind, claimed.job.ref))
    assert kinds == [("review", "deadbeef")]
    assert q.counts().get("superseded") == 1


def test_re_delivery_of_the_same_head_never_retires_its_own_predecessor(tmp_path):
    """Same head, two delivery ids (a redelivery, a replay): the queued work must survive."""
    q = _q(tmp_path)
    assert q.enqueue(_review(ref="samehead"), delivery_id="d1") is True
    assert q.enqueue(_review(ref="samehead"), delivery_id="d2") is True
    assert q.counts().get("superseded") is None
    assert q.counts().get("queued") == 2


def test_a_comment_triggered_review_without_a_head_is_never_superseded(tmp_path):
    """`@codna review` carries no ref -- it resolves the head when it runs, so it is never stale."""
    q = _q(tmp_path)
    comment = WebhookJob("review", "acme/app", pr_number=7, installation_id=42,
                         reason="comment_codna_review")
    assert q.enqueue(comment, delivery_id="c1") is True
    assert q.enqueue(_review(ref="newhead"), delivery_id="d1") is True
    assert q.counts().get("superseded") is None
    assert q.counts().get("queued") == 2


def test_a_merge_group_job_survives_a_pr_head_move(tmp_path):
    """A merge group's ref is the queue's own group commit. Superseding it would strip a required
    check off the group and get the pull request kicked out of the queue."""
    q = _q(tmp_path)
    group = WebhookJob("queue", "acme/app", ref="groupsha", pr_number=7, installation_id=42,
                       reason="merge_group_checks_requested")
    assert q.enqueue(group, delivery_id="g1") is True
    assert q.enqueue(_review(ref="newhead"), delivery_id="d1") is True
    assert q.counts().get("superseded") is None


def test_another_pull_request_is_untouched_by_a_head_move(tmp_path):
    q = _q(tmp_path)
    assert q.enqueue(_review(pr=7, ref="a1"), delivery_id="d1") is True
    assert q.enqueue(_review(pr=8, ref="b1"), delivery_id="d2") is True
    assert q.enqueue(_review(pr=7, ref="a2"), delivery_id="d3") is True
    remaining = sorted((c.job.pr_number, c.job.ref) for c in iter(q.claim, None))
    assert remaining == [(7, "a2"), (8, "b1")]


def test_the_insert_and_the_supersede_are_one_transaction(tmp_path, monkeypatch):
    """A failure while retiring the old heads must not leave the new row queued beside them --
    the pool would then work through commits that no longer exist."""
    q = _q(tmp_path)
    q.enqueue(_review(ref="old"), delivery_id="d1")

    def _boom(*_a, **_kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(type(q), "_supersede_within", staticmethod(_boom))
    with pytest.raises(RuntimeError):
        q.enqueue(_review(ref="new"), delivery_id="d2")

    monkeypatch.undo()
    claimed = [c.job.ref for c in iter(q.claim, None)]
    assert claimed == ["old"]      # the new row rolled back with the supersede; nothing half-applied


def test_the_retry_supersede_semantics_still_hold(tmp_path):
    """Pre-existing behaviour: a failed job under the cap is re-claimable as the SAME row."""
    q = _q(tmp_path)
    q.enqueue(_review(ref="head1"), delivery_id="d1")
    first = q.claim()
    q.complete(first.row_id, status="failed", result={"error": "boom"})
    second = q.claim()
    assert second is not None and second.row_id == first.row_id and second.attempts == 2


def test_a_new_head_retires_a_held_retry_of_the_old_head(tmp_path):
    q = _q(tmp_path)
    q.enqueue(_review(ref="old"), delivery_id="d1")
    first = q.claim()
    q.complete(first.row_id, status="failed", result={"error": "boom"})   # -> 'retry'
    q.enqueue(_review(ref="new"), delivery_id="d2")
    claimed = q.claim()
    assert claimed is not None and claimed.job.ref == "new"
    assert q.claim() is None


# --- 2. priority + the reserved thread --------------------------------------------------------

def test_merge_gating_kinds_are_claimed_before_everything_else(tmp_path):
    q = _q(tmp_path)
    order = [("fix", "f1"), ("secure", "s1"), ("review", "r1"), ("queue", "g1")]
    for kind, ref in order:
        q.enqueue(WebhookJob(kind, "acme/app", ref=ref, pr_number=1, installation_id=42, reason="t"),
                  delivery_id=f"d-{ref}")
    claimed = [c.job.kind for c in iter(q.claim, None)]
    # Every merge-gating kind comes out before any other, and arrival order breaks ties within a class.
    gating = [k for k in claimed if k in PRIORITY_KINDS]
    assert claimed[:len(gating)] == gating
    assert set(gating) == {k for k, _ in order} & set(PRIORITY_KINDS)


def test_reserve_priority_holds_the_last_thread_for_a_waiting_review(tmp_path):
    q = _q(tmp_path)
    q.enqueue(_fix(ref="f1"), delivery_id="d-fix")
    q.enqueue(_review(pr=9, ref="r1"), delivery_id="d-review")
    claimed = q.claim(reserve_priority=True)
    assert claimed is not None and claimed.job.kind == "review"


def test_reserve_priority_still_runs_a_fix_when_nothing_gates_a_merge(tmp_path):
    """Reserving a thread must not idle it: with no review waiting, the fix runs."""
    q = _q(tmp_path)
    q.enqueue(_fix(ref="f1"), delivery_id="d-fix")
    claimed = q.claim(reserve_priority=True)
    assert claimed is not None and claimed.job.kind == "fix"


def test_priority_column_orders_within_a_class(tmp_path):
    """The boot reconciler's re-queued head goes ahead of reviews queued while it waited."""
    q = _q(tmp_path)
    q.enqueue(_review(pr=1, ref="older"), delivery_id="d1")
    q.enqueue(_review(pr=2, ref="restart"), delivery_id="d2", priority=1)
    claimed = q.claim()
    assert claimed is not None and claimed.job.ref == "restart"


def test_the_pool_reserves_its_last_thread_once_fixes_hold_the_rest():
    pool = WorkerPool(_RecordingQueue(), concurrency=3)
    assert pool._reserve_priority_thread() is False
    with pool._busy_lock:
        pool._busy_kinds["w0"] = "fix"
    assert pool._reserve_priority_thread() is False      # one thread still free besides the last
    with pool._busy_lock:
        pool._busy_kinds["w1"] = "fix"
    assert pool._reserve_priority_thread() is True       # taking another fix would leave none
    with pool._busy_lock:
        pool._busy_kinds["w1"] = "review"
    assert pool._reserve_priority_thread() is False      # a review does not consume the reservation


def test_a_single_thread_pool_prefers_reviews_without_starving_fixes(tmp_path):
    """With one thread the reservation is always engaged -- it degrades to "review first" -- and
    the queue's fallback is what keeps it from idling on a fix-only backlog."""
    pool = WorkerPool(_RecordingQueue(), concurrency=1)
    assert pool._reserve_priority_thread() is True

    q = _q(tmp_path)
    q.enqueue(_fix(ref="f1"), delivery_id="d-fix")
    assert q.claim(reserve_priority=True).job.kind == "fix"   # nothing gating: the fix still runs


class _RecordingQueue:
    """Matches the fakes in test_webhook_worker.py: only what the pool actually calls."""

    def __init__(self, qjob=None):
        self.qjob = qjob
        self.completed = []

    def recover_stale(self):
        return {"requeued": 0, "failed": 0}

    def claim(self):
        qjob, self.qjob = self.qjob, None
        return qjob

    def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
        self.completed.append((row_id, status, result))


class _ReserveRecordingQueue(_RecordingQueue):
    """A queue that DOES take the reserved-thread flag, so the pool passes it."""

    def __init__(self, qjob=None):
        super().__init__(qjob)
        self.reserve_calls = []

    def claim(self, *, reserve_priority=False):
        self.reserve_calls.append(reserve_priority)
        qjob, self.qjob = self.qjob, None
        return qjob


def test_the_pool_passes_the_reservation_to_a_queue_that_supports_it():
    queue = _ReserveRecordingQueue()
    pool = WorkerPool(queue, concurrency=3, poll_interval=0.01, process=lambda _q: JobResult(True, "ok"))
    assert pool._claim_takes_reserve is True
    pool.start()
    assert _wait_until(lambda: queue.reserve_calls)
    pool.stop(timeout=1)
    # An idle 3-thread pool holds nothing back; the flag is computed per claim, not pinned.
    assert all(call is False for call in queue.reserve_calls)


def test_a_queue_without_the_flag_keeps_the_plain_claim():
    """An embedder's own queue (or an older fake) must not break on the new keyword."""
    queue = _RecordingQueue()
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, process=lambda _q: JobResult(True, "ok"))
    assert pool._claim_takes_reserve is False
    pool.start()
    time.sleep(0.05)
    pool.stop(timeout=1)  # no TypeError reached the loop


# --- 3. cancelling a running job --------------------------------------------------------------

class _CheckRunGitHub:
    """Stands in for webhook_github. `ok=False` is a run that could not be closed (already
    completed by someone else, or the call failed) -- the helper cannot tell those apart."""

    def __init__(self, ok=True, raises=False):
        self.updates = []
        self.ok = ok
        self.raises = raises

    def complete_check_run_if_open(self, repo, token, check_run_id, *, conclusion, summary, name):
        if self.raises:
            raise RuntimeError("github is down")
        self.updates.append({"repo": repo, "token": token, "id": check_run_id,
                             "conclusion": conclusion, "summary": summary, "name": name})
        return self.ok

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name):
        self.updates.append({"repo": repo, "token": token, "id": check_run_id,
                             "conclusion": conclusion, "summary": summary, "name": name})
        return True



# process_job resolves metering through keyword defaults bound at import time, so a linked org is
# injected rather than monkeypatched.
_LINKED_ORG = {
    "resolve_engine_key": lambda _installation: "org-key",
    "resolve_fix_enabled": lambda _installation: True,
    "resolve_provider_credentials": lambda _installation: (None, None),
}


class _ProcessGitHub(_CheckRunGitHub):
    """_CheckRunGitHub plus the two calls process_job makes before the runner."""

    def installation_token(self, *_a, **_k):
        return "scoped-token"

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        return 4242


def _handle(control, *, row_id=1, kind="review", ref="old", pr=7, deadline_s=1800.0):
    handle = control.register(row_id=row_id, kind=kind, repo="acme/app", pr_number=pr, ref=ref,
                              deadline_s=deadline_s)
    handle.attach_check_run("scoped-token", 4242)
    return handle


def test_superseding_completes_the_check_run_neutral_and_names_the_new_head():
    control = JobControl()
    handle = _handle(control)
    gh = _CheckRunGitHub()
    queue = _RecordingQueue()

    cancelled = control_module.cancel_jobs(
        control.superseded_by(repo="acme/app", pr_number=7, new_ref="21e3610a"),
        reason="superseded", summary="superseded by 21e3610", github=gh, queue=queue)

    assert cancelled == 1 and handle.cancelled is True
    assert len(gh.updates) == 1
    update = gh.updates[0]
    assert update["conclusion"] == "neutral"          # NEVER failure: it examined nothing
    assert "21e3610" in update["summary"]
    assert update["name"] == "codna review"
    assert queue.completed == [(1, "done", {"summary": "superseded by 21e3610", "cancelled": "superseded"})]


def test_supersede_matches_only_the_same_pull_request_with_a_different_head():
    control = JobControl()
    same_head = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7,
                                 ref="current", deadline_s=1800.0)
    other_pr = control.register(row_id=2, kind="review", repo="acme/app", pr_number=8,
                                ref="old", deadline_s=1800.0)
    other_repo = control.register(row_id=3, kind="review", repo="other/app", pr_number=7,
                                  ref="old", deadline_s=1800.0)
    headless = control.register(row_id=4, kind="review", repo="acme/app", pr_number=7,
                                ref=None, deadline_s=1800.0)
    stale_fix = control.register(row_id=5, kind="fix", repo="acme/app", pr_number=7,
                                 ref="old", deadline_s=1800.0)
    group = control.register(row_id=6, kind="queue", repo="acme/app", pr_number=7,
                             ref="groupsha", deadline_s=1800.0)

    matched = control.superseded_by(repo="acme/app", pr_number=7, new_ref="current")
    assert [h.row_id for h in matched] == [stale_fix.row_id]
    for survivor in (same_head, other_pr, other_repo, headless, group):
        assert survivor.cancelled is False


def test_supersede_running_is_driven_by_the_job_that_arrived():
    control = control_module.get_control()
    handle = control.register(row_id=99, kind="review", repo="acme/app", pr_number=7,
                              ref="stale", deadline_s=1800.0)
    handle.attach_check_run("tok", 1)
    gh = _CheckRunGitHub()
    try:
        cancelled = control_module.supersede_running(_review(ref="fresh"), github=gh, background=False)
        assert cancelled == 1
        assert handle.cancelled is True
        assert gh.updates[0]["summary"] == "superseded by fresh"
    finally:
        control.release(99)


def test_the_ingress_path_stops_the_job_without_waiting_on_github():
    """The ingress must acknowledge GitHub inside its 10 s delivery timeout, or the delivery is
    retried. So the kill happens inline and the 30 s-timeout Check Run call does not."""
    control = control_module.get_control()
    handle = control.register(row_id=98, kind="review", repo="acme/app", pr_number=7,
                              ref="stale", deadline_s=1800.0)
    handle.attach_check_run("tok", 1)
    entered = threading.Event()
    release = threading.Event()

    class _SlowGitHub:
        def __init__(self):
            self.updates = []

        def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name):
            entered.set()
            release.wait(10)          # stands in for a GitHub call that runs to its 30 s timeout
            self.updates.append(summary)
            return True

    gh = _SlowGitHub()
    try:
        started = time.monotonic()
        cancelled = control_module.supersede_running(_review(ref="fresh"), github=gh)
        elapsed = time.monotonic() - started

        assert cancelled == 1
        assert elapsed < 1.0, f"the ingress call blocked for {elapsed:.1f}s"
        assert handle.cancelled is True          # the stale job is stopped, synchronously
        assert _wait_until(entered.is_set)       # and the slow settling is happening off-thread
        assert gh.updates == []                  # ... still in flight, not on the request path
    finally:
        release.set()
        control.release(98)


def test_a_cancel_never_blocks_on_the_runtime_teardown(monkeypatch):
    """`cancel` is the fast half by contract: teardown waits on processes, so it belongs to the
    settle phase, which the ingress runs on its own thread."""
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    handle.attach_runtime("/tmp/codna-job-scratch-that-does-not-exist")
    torn = []
    import codna.webhook_procs as procs_module

    monkeypatch.setattr(procs_module, "_teardown_job_runtime", lambda tmp: torn.append(tmp) or 0)

    handle.cancel(reason="superseded")
    assert torn == []                    # not during the signal
    handle.teardown_runtime()
    assert torn == ["/tmp/codna-job-scratch-that-does-not-exist"]   # during the settle


def test_a_failed_close_leaves_the_row_for_another_path_to_resolve():
    """A terminal row tells the next boot there is nothing to resolve. Writing one after a close
    that did NOT confirm would strand the Check Run `in_progress` with no path left to close it --
    permanently unmergeable where `codna review` is required."""
    control = JobControl()
    handle = _handle(control)
    queue = _RecordingQueue()
    gh = _CheckRunGitHub(raises=True)

    cancel_jobs([handle], reason="superseded", summary="superseded by 21e3610", github=gh, queue=queue)

    assert handle.check_closed is False
    assert queue.completed == []          # NOT terminal: the worker (or the next boot) still owns it
    assert handle.finalize() is True      # ... and the row is still there to be written


def test_a_confirmed_close_marks_the_handle_and_writes_the_row():
    control = JobControl()
    handle = _handle(control)
    queue = _RecordingQueue()
    cancel_jobs([handle], reason="superseded", summary="superseded by 21e3610",
                github=_CheckRunGitHub(ok=True), queue=queue)
    assert handle.check_closed is True
    assert [(row, status) for row, status, _ in queue.completed] == [(1, "done")]


def test_the_close_never_clobbers_findings_a_review_already_posted():
    """`complete_check_run_if_open` reads the run first and leaves a completed one alone; the
    cancel path must prefer it over a blind update."""
    control = JobControl()
    handle = _handle(control)
    calls = []

    class _GitHub(_CheckRunGitHub):
        def update_check_run(self, *a, **kw):
            calls.append("update")
            return True

        def complete_check_run_if_open(self, *a, **kw):
            calls.append("if_open")
            return True

    control_module.close_check_run(handle, summary="superseded by x", github=_GitHub())
    assert calls == ["if_open"]


def test_the_worker_closes_the_run_when_no_canceller_confirmed_it():
    """The settle thread may not have run yet, or GitHub may have refused it. The worker thread
    must not assume someone else closed the run just because the job was cancelled."""
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)

    def _runner(job, **kw):
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        return worker_module._cancelled_result(job, handle)

    gh = _ProcessGitHub()
    qjob = QueuedJob(row_id=1, delivery_id="d1", attempts=1, job=_review())
    result = worker_module.process_job(qjob, app_id="APP", private_key="pem", github=gh,
                                       runner=_runner, control=handle, **_LINKED_ORG)

    assert result.cancelled is True
    assert handle.check_closed is True
    [closed] = [u for u in gh.updates if u["id"] == 4242]
    assert closed["conclusion"] == "neutral" and "superseded by 21e3610" in closed["summary"]


def test_the_worker_does_not_close_a_run_a_canceller_already_closed():
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)

    gh = _ProcessGitHub()

    def _runner(job, **kw):
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        handle.mark_check_closed()            # the settle thread got there first
        return worker_module._cancelled_result(job, handle)

    worker_module.process_job(QueuedJob(row_id=1, delivery_id="d1", attempts=1, job=_review()),
                              app_id="APP", private_key="pem", github=gh, runner=_runner,
                              control=handle, **_LINKED_ORG)
    assert gh.updates == []                   # no second call for the same run


def test_a_cancelled_job_without_a_check_run_completes_quietly():
    """Cancelled before it opened a run: there is nothing to close, and nothing may raise."""
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    gh = _CheckRunGitHub()
    assert cancel_jobs([handle], reason="superseded", summary="superseded by x", github=gh) == 1
    assert gh.updates == []


def test_cancel_kills_the_registered_process_group():
    import subprocess

    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    proc = subprocess.Popen(["sh", "-c", "sleep 30 & exec sleep 30"], start_new_session=True)
    handle.attach_process(proc)
    handle.cancel(reason="superseded")
    assert _wait_until(lambda: proc.poll() is not None, timeout=10), "the process group survived the cancel"


def test_a_process_attached_after_the_cancel_is_killed_immediately():
    """The cancel can land between Popen and attach_process; the kill must not be lost."""
    import subprocess

    control = JobControl()
    handle = control.register(row_id=1, kind="fix", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    handle.cancel(reason="job_deadline")
    proc = subprocess.Popen(["sh", "-c", "exec sleep 30"], start_new_session=True)
    handle.attach_process(proc)
    assert _wait_until(lambda: proc.poll() is not None, timeout=10)


def test_only_one_writer_finalizes_a_cancelled_row():
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    assert handle.finalize() is True
    assert handle.finalize() is False        # the worker thread must not overwrite the cancellation


def test_the_runner_reports_the_cancellation_instead_of_the_killed_run(monkeypatch):
    """Whatever the killed subprocess printed describes a run that no longer matters."""
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)

    def _fake_run(argv, **kwargs):
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        return worker_module.subprocess.CompletedProcess(argv, 1, stdout="half a review", stderr="")

    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    monkeypatch.setattr(worker_module, "_teardown_job_runtime", lambda tmp: 0)
    result = worker_module.run_codna_job(_review(), token="t", engine_key="k", control=handle)

    assert result.cancelled is True
    assert result.ok is True and result.retryable is False
    assert result.conclusion == "neutral"
    assert "superseded by 21e3610" in result.summary
    assert result.check_completed is True   # the canceller already closed the run


def test_a_cancelled_timeout_is_reported_as_the_cancellation_not_as_a_timeout(monkeypatch):
    control = JobControl()
    handle = control.register(row_id=1, kind="fix", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)

    def _fake_run(argv, **kwargs):
        handle.cancel(reason="job_deadline", detail="stopped after 1845s")
        raise worker_module.subprocess.TimeoutExpired(argv, 1800)

    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    monkeypatch.setattr(worker_module, "_teardown_job_runtime", lambda tmp: 0)
    result = worker_module.run_codna_job(_fix(), token="t", engine_key="k", control=handle)
    assert result.cancelled is True and "stopped after 1845s" in result.summary


def test_the_runner_registers_its_subprocess_for_cancellation(monkeypatch):
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    seen = {}

    def _fake_run(argv, **kwargs):
        seen["on_process"] = kwargs.get("on_process")
        return worker_module.subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    monkeypatch.setattr(worker_module, "_teardown_job_runtime", lambda tmp: 0)
    worker_module.run_codna_job(_review(), token="t", engine_key="k", control=handle)
    assert seen["on_process"] == handle.attach_process


def test_the_worker_does_not_write_a_row_the_canceller_already_finalized():
    queue = _RecordingQueue()
    control = JobControl()
    handle = control.register(row_id=5, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    assert handle.finalize() is True            # the canceller got there first
    pool = WorkerPool(queue, process=lambda _q: JobResult(True, "cancelled", cancelled=True))
    qjob = QueuedJob(row_id=5, delivery_id="d1", attempts=1, job=_review())
    pool._run_claimed_inner(qjob, handle)
    assert queue.completed == []


def test_an_unconfirmed_close_leaves_the_row_and_the_registry_entry_for_the_next_boot(tmp_path):
    """Same invariant as settle_cancelled, on the worker's own path: a terminal row would tell the
    next boot there is nothing to resolve, stranding the Check Run `in_progress` for good."""
    from codna.webhook_resume import RunningJobRegistry

    queue = _RecordingQueue()
    registry = RunningJobRegistry(tmp_path / "running")
    control = JobControl()
    qjob = QueuedJob(row_id=5, delivery_id="d1", attempts=1, job=_review())

    def _process(_qjob):
        handle = control.get(5)
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        return JobResult(True, "superseded by 21e3610", retryable=False, cancelled=True,
                         check_completed=True)                 # ... but check_closed stays False

    pool = WorkerPool(queue, process=_process, registry=registry)
    pool._control = control
    pool._run_claimed(qjob)

    assert queue.completed == []                                # no terminal row
    assert registry.get(5) is not None                          # ... and the run id is still on disk


def test_no_exception_path_can_write_the_row_an_unconfirmed_close_forbids(tmp_path, monkeypatch):
    """The early return runs inside the try; if anything after it raises (a broken stderr pipe in
    the phase log is enough), the handler must not be able to write the forbidden terminal row."""
    queue = _RecordingQueue()
    control = JobControl()

    def _process(_qjob):
        handle = control.get(5)
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        return JobResult(True, "superseded by 21e3610", retryable=False, cancelled=True,
                         check_completed=True)

    calls = []

    def _exploding_phase(qjob, phase):
        calls.append(phase)
        if phase == "cancel_close_unconfirmed":
            raise OSError("stderr pipe is gone")

    monkeypatch.setattr(worker_module, "_log_worker_phase", _exploding_phase)
    pool = WorkerPool(queue, process=_process)
    pool._control = control
    pool._run_claimed(QueuedJob(row_id=5, delivery_id="d1", attempts=1, job=_review()))

    assert "cancel_close_unconfirmed" in calls    # the guarded path really was taken
    assert queue.completed == []                  # ... and the crash handler wrote nothing


def test_a_confirmed_close_clears_the_registry_and_writes_the_row(tmp_path):
    from codna.webhook_resume import RunningJobRegistry

    queue = _RecordingQueue()
    registry = RunningJobRegistry(tmp_path / "running")
    control = JobControl()

    def _process(_qjob):
        handle = control.get(5)
        handle.cancel(reason="superseded", detail="superseded by 21e3610")
        handle.mark_check_closed()
        return JobResult(True, "superseded by 21e3610", retryable=False, cancelled=True,
                         check_completed=True)

    pool = WorkerPool(queue, process=_process, registry=registry)
    pool._control = control
    pool._run_claimed(QueuedJob(row_id=5, delivery_id="d1", attempts=1, job=_review()))

    assert [(row, status) for row, status, _ in queue.completed] == [(5, "done")]
    assert registry.get(5) is None


def test_a_normal_job_still_writes_its_own_terminal_row():
    queue = _RecordingQueue()
    control = JobControl()
    handle = control.register(row_id=5, kind="review", repo="acme/app", pr_number=7, ref="old",
                              deadline_s=1800.0)
    pool = WorkerPool(queue, process=lambda _q: JobResult(True, "ok"))
    pool._run_claimed_inner(QueuedJob(row_id=5, delivery_id="d1", attempts=1, job=_review()), handle)
    assert [(row, status) for row, status, _ in queue.completed] == [(5, "done")]


# --- 4. the worker-enforced job deadline ------------------------------------------------------

def test_the_watchdog_cancels_a_job_that_outlived_the_whole_job_deadline(monkeypatch):
    """_run_job_process bounds only the CLI child; the watchdog bounds the JOB.

    Drives the real pool: a job that never returns, a deadline set short, and the real GitHub
    client swapped for a recorder so the cancellation's Check Run is inspectable offline.
    """
    queue = _RecordingQueue()
    release = threading.Event()
    gh = _CheckRunGitHub()
    real_cancel = control_module.cancel_jobs

    def _cancel_with_fake_github(handles, *, reason, summary, github=None, queue=None):
        return real_cancel(handles, reason=reason, summary=summary, github=gh, queue=queue)

    monkeypatch.setattr(control_module, "cancel_jobs", _cancel_with_fake_github)

    def _slow(qjob):
        handle = control_module.get_control().get(qjob.row_id)
        handle.attach_check_run("scoped-token", 4242)
        release.wait(10)                       # a job that would hold its thread indefinitely
        return JobResult(True, "ok")

    queue.qjob = QueuedJob(row_id=3, delivery_id="d1", attempts=1, job=_review())
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, process=_slow)
    pool._job_deadline_s = 0.05
    pool.start()
    try:
        assert _wait_until(lambda: gh.updates, timeout=5), "the watchdog never fired"
        update = gh.updates[0]
        assert update["conclusion"] == "neutral"       # never `failure`: it concluded nothing
        assert str(int(pool._job_deadline_s)) in update["summary"]
        assert update["name"] == "codna review"
        # The row is terminal, written by the canceller rather than left 'running' forever.
        assert [(row, status) for row, status, _ in queue.completed] == [(3, "done")]
    finally:
        release.set()
        pool.stop(timeout=2)


def test_overdue_lists_only_jobs_past_their_own_deadline():
    control = JobControl()
    fresh = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="a",
                             deadline_s=3600.0)
    stale = control.register(row_id=2, kind="fix", repo="acme/app", pr_number=8, ref="b",
                             deadline_s=0.0)
    time.sleep(0.01)
    assert [h.row_id for h in control.overdue()] == [stale.row_id]
    stale.cancel(reason="job_deadline")
    assert control.overdue() == []          # already cancelled: never cancelled twice
    assert fresh.cancelled is False


def test_the_pool_deadline_covers_more_than_the_subprocess_bound(monkeypatch):
    monkeypatch.setattr(worker_module, "_JOB_TIMEOUT_S", 1800)
    pool = WorkerPool(_RecordingQueue(), concurrency=1)
    assert pool._job_deadline_s > worker_module._JOB_TIMEOUT_S + worker_module._KILL_DRAIN_S


# --- 5. /healthz says why the queue is not draining -------------------------------------------

class _Server:
    def __init__(self, queue=None, pool=None):
        self.queue = queue
        self.worker_pool = pool


def test_healthz_reports_queue_depth_by_kind_and_running_jobs(tmp_path, monkeypatch):
    q = _q(tmp_path)
    waiting = [("review", "r1", 1), ("review", "r2", 2), ("fix", "f1", 3), ("secure", "s1", 4)]
    for kind, ref, pr in waiting:
        q.enqueue(WebhookJob(kind, "acme/app", ref=ref, pr_number=pr, installation_id=42, reason="t"),
                  delivery_id=f"d-{ref}")
    control = JobControl()
    control.register(row_id=1, kind="fix", repo="acme/app", pr_number=14,
                     ref="92730476aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", deadline_s=1800.0)
    monkeypatch.setattr(control_module, "get_control", lambda: control)

    payload = health_payload(_Server(queue=q), ok=True)

    assert payload["ok"] is True and payload["service"] == "codna-webhook"
    expected = {}
    for kind, _ref, _pr in waiting:
        expected[kind] = expected.get(kind, 0) + 1
    assert payload["queue_depth"] == expected            # derived from what was enqueued
    assert payload["running"] == [{"kind": "fix", "repo": "acme/app", "pr": 14,
                                   "head": "92730476", "age_s": payload["running"][0]["age_s"],
                                   "cancelled": False}]


def test_healthz_never_leaks_a_token_a_summary_or_a_delivery_id(tmp_path, monkeypatch):
    q = _q(tmp_path)
    q.enqueue(_review(ref="head1"), delivery_id="delivery-secret-1")
    control = JobControl()
    handle = control.register(row_id=1, kind="review", repo="acme/app", pr_number=7, ref="head1",
                              deadline_s=1800.0)
    handle.attach_check_run("ghs_supersecrettoken", 4242)
    monkeypatch.setattr(control_module, "get_control", lambda: control)

    blob = json.dumps(health_payload(_Server(queue=q), ok=True))
    for secret in ("ghs_supersecrettoken", "delivery-secret-1", "4242", "installation"):
        assert secret not in blob


def test_healthz_survives_a_queue_that_cannot_answer(monkeypatch):
    class _BrokenQueue:
        def depth_by_kind(self):
            raise RuntimeError("database is locked")

    payload = health_payload(_Server(queue=_BrokenQueue()), ok=True)
    assert payload["ok"] is True and payload["queue_depth"] is None


def test_healthz_reports_worker_occupancy(tmp_path):
    q = _q(tmp_path)
    pool = WorkerPool(q, concurrency=3)
    with pool._busy_lock:
        pool._busy_kinds["w0"] = "fix"
        pool._busy_kinds["w1"] = "fix"
    payload = health_payload(_Server(queue=q, pool=pool), ok=True)
    assert payload["workers"]["configured_threads"] == 3
    assert payload["workers"]["priority_thread_reserved"] is True


def test_depth_by_kind_counts_only_what_can_still_run(tmp_path):
    q = _q(tmp_path)
    q.enqueue(_review(pr=1, ref="r1"), delivery_id="d1")
    q.enqueue(_fix(pr=2, ref="f1"), delivery_id="d2")
    claimed = q.claim()                       # now 'running', not waiting
    q.complete(claimed.row_id, status="done")
    assert q.depth_by_kind() == {"fix": 1}


# --- 6. memory-derived pool size ---------------------------------------------------------------

def test_concurrency_is_derived_from_the_machine_memory(monkeypatch):
    import codna.webhook_control as webhook_module

    monkeypatch.setattr(webhook_module, "_machine_memory_mb", lambda: 8192)
    derived = default_concurrency({})
    budget = webhook_module._JOB_MEMORY_MB
    reserved = webhook_module._RESERVED_MEMORY_MB
    assert derived == min(webhook_module._MAX_DERIVED_CONCURRENCY, (8192 - reserved) // budget)
    assert derived * budget < 8192 - reserved      # never sized past what the machine holds


@pytest.mark.parametrize("total_mb", [2048, 4096, 8192, 16384])
def test_a_derived_pool_always_fits_in_its_machine(monkeypatch, total_mb):
    import codna.webhook_control as webhook_module

    monkeypatch.setattr(webhook_module, "_machine_memory_mb", lambda: total_mb)
    derived = default_concurrency({})
    assert derived >= 1
    assert derived * webhook_module._JOB_MEMORY_MB <= total_mb - webhook_module._RESERVED_MEMORY_MB


def test_the_env_override_still_wins(monkeypatch):
    import codna.webhook_control as webhook_module

    monkeypatch.setattr(webhook_module, "_machine_memory_mb", lambda: 8192)
    assert default_concurrency({"CODNA_WEBHOOK_CONCURRENCY": "2"}) == 2


def test_an_unreadable_machine_keeps_the_long_standing_default(monkeypatch):
    import codna.webhook_control as webhook_module

    monkeypatch.setattr(webhook_module, "_machine_memory_mb", lambda: None)
    assert default_concurrency({}) == 2


def test_a_cgroup_limit_beats_the_hosts_total(tmp_path, monkeypatch):
    """A container on a big host must size the pool from ITS limit, not the host's RAM."""
    import codna.webhook_control as webhook_module

    limit = tmp_path / "memory.max"
    limit.write_text(str(4 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(webhook_module, "_CGROUP_MEMORY_LIMIT_PATHS", (str(limit),))
    assert webhook_module._machine_memory_mb() == 4096


def test_an_unlimited_cgroup_falls_through_to_the_machine(tmp_path, monkeypatch):
    """cgroup v2 writes 'max' for "no limit", and v1 a sentinel near 2^63: neither is a size."""
    import codna.webhook_control as webhook_module

    unlimited = tmp_path / "memory.max"
    unlimited.write_text("max", encoding="utf-8")
    sentinel = tmp_path / "memory.limit_in_bytes"
    sentinel.write_text(str(2 ** 63 - 1), encoding="utf-8")
    monkeypatch.setattr(webhook_module, "_CGROUP_MEMORY_LIMIT_PATHS", (str(unlimited), str(sentinel)))
    total = webhook_module._machine_memory_mb()
    assert total is None or total > 0      # whatever sysconf says on this host, never the sentinel
    assert total != (2 ** 63 - 1) // (1024 * 1024)


# --- 7. the kinds themselves --------------------------------------------------------------------

def test_the_scheduling_classes_agree_with_the_check_run_names():
    from codna.webhook import CHECK_RUN_NAMES

    # Every supersedable kind names a real Check Run, so a cancellation can always close one.
    assert SUPERSEDABLE_KINDS <= set(CHECK_RUN_NAMES)
    # `queue` gates a merge but reports under the review check's name (it inherits that verdict).
    assert "review" in PRIORITY_KINDS and "queue" in PRIORITY_KINDS
    assert PRIORITY_KINDS.isdisjoint({"fix", "secure"})


def test_a_retargeted_handle_is_superseded_by_a_different_head_but_not_by_its_own():
    """The worker resolves a review's head at start and anchors its check there (codna#569); the
    handle has to follow, or the late event for that very head would cancel the job reviewing it."""
    control = JobControl()
    handle = _handle(control, ref="old")
    handle.retarget("current")
    assert handle.ref == "current" and handle.snapshot()["head"] == "current"[:8]
    assert control.superseded_by(repo="acme/app", pr_number=7, new_ref="current") == []
    assert control.superseded_by(repo="acme/app", pr_number=7, new_ref="newer") == [handle]
