"""Restart survival for webhook jobs (codna.webhook_resume): the running-job registry beside the
queue, the startup reconciliation (resume once / superseded on head move / cancelled at the cap),
the SIGTERM path, and the worker hooks that feed them.

Live case this closes: thyn-ai/algenta#1056 head 13b07d1b -- `codna review` check run
105937057683 started 2026-09-19T17:27:51Z, the deploy replaced the machine 17:28:49Z-17:29:27Z,
the retry burned its attempts before creating a run, and the required check sat `in_progress`.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import types
from dataclasses import asdict

from codna import webhook_resume
from codna.webhook import WebhookError, WebhookJob
from codna.webhook_queue import QueuedJob, WebhookQueue
from codna.webhook_resume import RunningJobRegistry, reconcile_interrupted, registry_dir_for
from codna.webhook_worker import JobResult, WorkerPool, process_job


class _GitHub:
    """The three GitHub calls reconciliation makes, recorded in order."""

    def __init__(self, *, head=None, open_runs=(), token_ok=True):
        self.calls = []
        self.head = head
        self.open_runs = set(open_runs)
        self.token_ok = token_ok

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", installation_id, kind))
        if not self.token_ok:
            raise WebhookError("installation_token_failed", "installation token: 401")
        return "scoped-token"

    def pull_request_head_sha(self, repo, token, number):
        self.calls.append(("head", number))
        if isinstance(self.head, Exception):
            raise self.head
        return self.head

    def complete_check_run_if_open(self, repo, token, check_run_id, *, conclusion, summary, name):
        self.calls.append(("complete", check_run_id, conclusion, summary))
        if check_run_id in self.open_runs:
            self.open_runs.discard(check_run_id)
            return True
        return False

    def completed(self):
        return [c for c in self.calls if c[0] == "complete"]


def _job(kind="review", ref="sha1", pr_number=7):
    return WebhookJob(kind, "acme/app", ref=ref, pr_number=pr_number, installation_id=42,
                      reason="pull_request_opened")


def _qjob(row_id=5, attempts=1, **job_kw):
    return QueuedJob(row_id=row_id, delivery_id="d1", attempts=attempts, job=_job(**job_kw))


def _interrupted(tmp_path, *, kind="review", ref="sha1", pr_number=7, check_run_id=9001,
                 resumes=0, failed_before=0):
    """A queue + registry exactly as the previous process left them: the row claimed (status
    'running'), its registry entry written, the Check Run id recorded -- and then the kill."""
    queue = WebhookQueue(tmp_path / "queue.db")
    registry = RunningJobRegistry(registry_dir_for(queue.path))
    queue.enqueue(_job(kind, ref, pr_number), delivery_id="d1")
    for _ in range(failed_before):
        prior = queue.claim()
        queue.complete(prior.row_id, status="failed")
    qjob = queue.claim()
    if resumes:  # an earlier restart already brought this job back once
        registry.start(qjob)
        for _ in range(resumes):
            registry.mark_resumed(qjob.row_id)
    registry.start(qjob)
    if check_run_id:
        registry.record_check_run(qjob.row_id, check_run_id)
    return queue, registry, qjob


def _reconcile(queue, registry, github):
    """What WorkerPool.start does on boot: recover_stale, then reconcile, before any claim."""
    queue.recover_stale()
    return reconcile_interrupted(queue, registry, app_id="APP", private_key="pem", github=github,
                                 environ={})


def _wait_until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# --- the registry: written on claim, updated with the run id, cleared on finish, survives death --

def test_registry_dir_sits_beside_the_queue():
    from pathlib import Path

    assert registry_dir_for("/data/queue.db") == Path("/data/running")


def test_registry_writes_on_start_records_the_check_run_and_clears_on_finish(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    entry = registry.start(_qjob(row_id=5))
    assert (tmp_path / "running" / "5.json").exists()
    assert (entry.row_id, entry.attempt, entry.check_run_id, entry.resumes) == (5, 1, None, 0)
    registry.record_check_run(5, 9001)
    assert registry.get(5).check_run_id == 9001
    registry.finish(5)
    assert registry.get(5) is None and registry.entries() == []
    registry.finish(5)  # idempotent: a double clear is not an error


def test_registry_survives_a_simulated_crash(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    registry.start(_qjob(row_id=5, kind="fix", ref="abc", pr_number=3))
    registry.record_check_run(5, 77)
    del registry  # the process dies here: finish() never runs
    fresh = RunningJobRegistry(tmp_path / "running")  # the next boot
    [entry] = fresh.entries()
    assert (entry.row_id, entry.kind, entry.repo, entry.ref, entry.pr_number) == (5, "fix", "acme/app", "abc", 3)
    assert entry.check_run_id == 77 and entry.installation_id == 42 and entry.delivery_id == "d1"
    assert entry.interrupted_by is None  # a crash, not a drained SIGTERM


def test_registry_carries_the_resume_count_into_the_attempt_that_replaces_it(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    registry.start(_qjob(row_id=5, attempts=1))
    registry.record_check_run(5, 9001)
    registry.mark_resumed(5)
    resumed = registry.get(5)
    assert resumed.resumes == 1 and resumed.check_run_id is None
    again = registry.start(_qjob(row_id=5, attempts=2))  # the retry claims the same row
    assert (again.resumes, again.attempt, again.check_run_id) == (1, 2, None)


def test_registry_mark_interrupted_stamps_every_live_entry(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    registry.start(_qjob(row_id=1))
    registry.start(_qjob(row_id=2))
    assert registry.mark_interrupted("sigterm") == 2
    assert all(e.interrupted_by == "sigterm" and e.interrupted_at for e in registry.entries())


def test_registry_skips_a_corrupt_or_foreign_file(tmp_path):
    root = tmp_path / "running"
    registry = RunningJobRegistry(root)
    registry.start(_qjob(row_id=1))
    (root / "2.json").write_text("{not json", encoding="utf-8")
    (root / "3.json").write_text(json.dumps({"row_id": 3, "unknown_field": 1}), encoding="utf-8")
    # a staging file a crash left mid-write (mkstemp name: `<row>.<random>.tmp`) is not an entry
    (root / "4.k3j9x2.tmp").write_text(json.dumps(asdict(registry.get(1)) | {"row_id": 4}), encoding="utf-8")
    assert [e.row_id for e in registry.entries()] == [1]


def test_registry_never_raises_when_its_directory_is_unusable(tmp_path):
    blocker = tmp_path / "running"
    blocker.write_text("a file where the directory should be")
    registry = RunningJobRegistry(blocker)  # mkdir fails: log, carry on
    registry.start(_qjob(row_id=1))
    registry.record_check_run(1, 9)
    registry.mark_interrupted("sigterm")
    registry.finish(1)
    assert registry.entries() == []


def test_registry_record_check_run_and_mark_interrupted_never_lose_each_other(tmp_path):
    """The two writers a redeploy interleaves: worker threads persisting the run id they just
    opened (`record_check_run`, right after `create_check_run`) and the main thread stamping every
    live entry once the drain grace ran out (`mark_interrupted`, from WorkerPool.stop). Without
    one lock around each read-modify-write, a stamp that snapshotted the entries before the run
    id landed wrote last and the file lost `check_run_id` -- and the next boot skipped the close,
    leaving the stale in_progress run this module exists to complete. Whatever the interleaving,
    the surviving file must carry BOTH fields, and no staging file may be left behind."""
    root = tmp_path / "running"
    registry = RunningJobRegistry(root)
    rows = list(range(1, 9))
    for trial in range(120):
        for row in rows:
            registry.start(_qjob(row_id=row))
        gate = threading.Barrier(2)

        def record():
            gate.wait()
            for row in rows:
                registry.record_check_run(row, 9000 + row)

        def stamp():
            gate.wait()
            registry.mark_interrupted("sigterm")

        threads = [threading.Thread(target=record), threading.Thread(target=stamp)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for row in rows:
            entry = registry.get(row)
            assert entry is not None, (trial, row)
            assert (entry.check_run_id, entry.interrupted_by) == (9000 + row, "sigterm"), (trial, row, entry)
        for row in rows:
            registry.finish(row)
    assert list(root.iterdir()) == []  # every unique staging file was renamed or removed


def test_registry_finish_racing_mark_interrupted_never_resurrects_a_finished_job(tmp_path):
    """A job that ends while stop() is stamping must stay gone: the pre-lock code could read the
    entry, lose to finish(), then write it back -- a phantom the next boot would 'reconcile'."""
    registry = RunningJobRegistry(tmp_path / "running")
    rows = list(range(1, 9))
    for trial in range(120):
        for row in rows:
            registry.start(_qjob(row_id=row))
        gate = threading.Barrier(2)

        def finish_all():
            gate.wait()
            for row in rows:
                registry.finish(row)

        def stamp():
            gate.wait()
            registry.mark_interrupted("sigterm")

        threads = [threading.Thread(target=finish_all), threading.Thread(target=stamp)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert registry.entries() == [], trial


# --- startup reconciliation -----------------------------------------------------------------

def test_reconcile_resumes_once_and_closes_the_stale_check_run(tmp_path):
    queue, registry, qjob = _interrupted(tmp_path)
    gh = _GitHub(head="sha1", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["resumed"] == 1
    assert gh.calls[0] == ("token", 42, "review")  # the job's own scope: it may complete its run
    [(_, cid, conclusion, summary)] = gh.completed()
    assert cid == 9001 and conclusion == "cancelled"
    assert "re-queued automatically as attempt 2" in summary and "@codna review" in summary
    entry = registry.get(qjob.row_id)
    assert entry is not None and entry.resumes == 1 and entry.check_run_id is None
    again = queue.claim()  # the same job runs again, as attempt 2
    assert again is not None and again.row_id == qjob.row_id and again.attempts == 2


def test_reconcile_supersedes_a_job_whose_pr_head_moved_and_queues_the_current_head(tmp_path):
    """The stale head is not re-run -- and the CURRENT head is queued for its own review.

    "which gets its own run" used to be an assumption, and on 2026-09-19/20 it was wrong: the push
    that moved the head was delivered while the machine was being replaced, so no job for the new
    head existed anywhere and GitHub does not redeliver. Reviews for heads pushed 02:00-02:06 only
    appeared after 02:17.
    """
    queue, registry, qjob = _interrupted(tmp_path, ref="old-sha")
    gh = _GitHub(head="new-sha", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["superseded"] == 1
    [(_, cid, conclusion, summary)] = gh.completed()
    assert cid == 9001 and conclusion == "cancelled" and "moved to new-sha" in summary
    assert queue.row_state(qjob.row_id)[0] == "done"
    assert registry.get(qjob.row_id) is None

    queued = queue.claim()
    assert queued is not None and queued.row_id != qjob.row_id     # NOT the stale row re-run
    assert (queued.job.ref, queued.job.kind, queued.job.pr_number) == ("new-sha", "review", 7)
    assert queue.claim() is None                                   # exactly one, for the new head


def test_reconcile_queues_the_current_head_only_once_across_repeated_restarts(tmp_path):
    queue, registry, _ = _interrupted(tmp_path, ref="old-sha")
    gh = _GitHub(head="new-sha", open_runs={9001})
    _reconcile(queue, registry, gh)
    before = queue.counts().get("queued")
    _reconcile(queue, registry, _GitHub(head="new-sha", open_runs={9001}))
    assert queue.counts().get("queued") == before   # the delivery id is derived from row + head


def test_reconcile_does_not_queue_a_fix_against_a_head_it_never_examined(tmp_path):
    """A fix spends a metered run and opens a pull request: a restart must not start one."""
    queue, registry, qjob = _interrupted(tmp_path, kind="fix", ref="old-sha")
    gh = _GitHub(head="new-sha", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["superseded"] == 1
    assert queue.claim() is None                    # nothing re-queued for either head
    assert queue.row_state(qjob.row_id)[0] == "done"


def test_reconcile_cancels_at_the_resume_cap_with_the_retrigger_comment(tmp_path):
    queue, registry, qjob = _interrupted(tmp_path, resumes=1)  # already brought back once
    gh = _GitHub(head="sha1", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["cancelled"] == 1
    [(_, _cid, conclusion, summary)] = gh.completed()
    assert conclusion == "cancelled" and "after one automatic resume" in summary and "@codna review" in summary
    assert ("head", 7) not in gh.calls  # decided before any PR lookup
    assert queue.claim() is None and queue.row_state(qjob.row_id) == ("failed", 1)
    assert registry.get(qjob.row_id) is None


def test_reconcile_cancels_a_job_that_died_on_its_last_attempt(tmp_path):
    queue, registry, _ = _interrupted(tmp_path, failed_before=2)  # attempt 3 of 3 was running
    gh = _GitHub(head="sha1", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)  # recover_stale marks it terminal; nothing re-runs
    assert outcome["cancelled"] == 1
    [(_, _cid, conclusion, summary)] = gh.completed()
    assert conclusion == "cancelled" and "no attempts left" in summary and "@codna review" in summary
    assert queue.claim() is None and registry.entries() == []


def test_reconcile_names_the_redeploy_when_sigterm_was_recorded(tmp_path):
    queue, registry, _ = _interrupted(tmp_path)
    registry.mark_interrupted("sigterm")  # what WorkerPool.stop does after the drain grace
    gh = _GitHub(head="sha1", open_runs={9001})
    _reconcile(queue, registry, gh)
    assert "interrupted by a redeploy" in gh.completed()[0][3]


def test_reconcile_without_a_recorded_run_resumes_and_touches_no_check_run(tmp_path):
    queue, registry, _ = _interrupted(tmp_path, check_run_id=None)  # killed before create_check_run
    gh = _GitHub(head="sha1")
    outcome = _reconcile(queue, registry, gh)
    assert outcome["resumed"] == 1 and gh.completed() == []
    assert queue.claim() is not None


def test_reconcile_keeps_everything_for_the_next_boot_when_no_token_can_be_minted(tmp_path):
    queue, registry, qjob = _interrupted(tmp_path)
    gh = _GitHub(head="sha1", open_runs={9001}, token_ok=False)
    outcome = _reconcile(queue, registry, gh)
    assert outcome["skipped"] == 1 and gh.completed() == []
    assert registry.get(qjob.row_id).resumes == 0  # not counted as a resume
    assert queue.claim() is not None  # recover_stale's requeue stands; the retry's own hygiene applies


def test_reconcile_uses_github_token_env_when_the_app_is_not_configured(tmp_path):
    queue, registry, _ = _interrupted(tmp_path)
    gh = _GitHub(head="sha1", open_runs={9001})
    queue.recover_stale()
    outcome = reconcile_interrupted(queue, registry, app_id=None, private_key=None, github=gh,
                                    environ={"GITHUB_TOKEN": "ghs_self_hosted"})
    assert outcome["resumed"] == 1 and gh.completed()
    assert not [c for c in gh.calls if c[0] == "token"]


def test_reconcile_drops_an_entry_whose_row_already_finished(tmp_path):
    queue, registry, qjob = _interrupted(tmp_path)
    queue.complete(qjob.row_id, status="done", result={"summary": "ok"})  # done, killed before finish()
    gh = _GitHub(head="sha1", open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["finished"] == 1 and gh.calls == [] and registry.entries() == []


def test_reconcile_resumes_a_job_without_a_pull_request_and_skips_the_head_lookup(tmp_path):
    queue, registry, _ = _interrupted(tmp_path, kind="fix", pr_number=None)  # a check_suite fix
    gh = _GitHub(open_runs={9001})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["resumed"] == 1 and not [c for c in gh.calls if c[0] == "head"]
    assert "@codna fix" in gh.completed()[0][3]


def test_reconcile_one_failing_entry_does_not_strand_the_others(tmp_path):
    queue = WebhookQueue(tmp_path / "queue.db")
    registry = RunningJobRegistry(registry_dir_for(queue.path))
    queue.enqueue(_job(pr_number=1), delivery_id="d1")
    queue.enqueue(_job(pr_number=2), delivery_id="d2")
    first, second = queue.claim(), queue.claim()
    for qjob, cid in ((first, 11), (second, 12)):
        registry.start(qjob)
        registry.record_check_run(qjob.row_id, cid)

    class _Flaky(_GitHub):
        def pull_request_head_sha(self, repo, token, number):
            if number == 1:
                raise RuntimeError("transport")
            return super().pull_request_head_sha(repo, token, number)

    gh = _Flaky(head="sha1", open_runs={11, 12})
    outcome = _reconcile(queue, registry, gh)
    assert outcome["skipped"] == 1 and outcome["resumed"] == 1
    assert [c[1] for c in gh.completed()] == [12]
    assert registry.get(first.row_id) is not None  # left for the next boot


def test_reconcile_logs_one_line_per_decision(tmp_path, capsys):
    queue, registry, qjob = _interrupted(tmp_path)
    _reconcile(queue, registry, _GitHub(head="sha1", open_runs={9001}))
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    [decision] = [line for line in lines if line.get("event") == "job_resume"]
    assert decision["service"] == "codna-webhook-worker"
    assert (decision["row_id"], decision["check_run_id"], decision["verdict"]) == (qjob.row_id, 9001, "resumed")
    # mark_resumed has cleared the id from the entry by now; the log still names the run closed
    assert decision["closed_check_run_id"] == 9001 and registry.get(qjob.row_id).check_run_id is None
    assert "scoped-token" not in json.dumps(lines)


def test_reconcile_log_distinguishes_a_run_the_job_had_already_completed(tmp_path, capsys):
    queue, registry, qjob = _interrupted(tmp_path)
    _reconcile(queue, registry, _GitHub(head="sha1", open_runs=set()))  # `codna review --post` finished it
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    [decision] = [line for line in lines if line.get("event") == "job_resume"]
    assert decision["verdict"] == "resumed"
    assert (decision["check_run_id"], decision["closed_check_run_id"]) == (9001, None)


def test_reconcile_log_names_the_closed_run_for_a_cancelled_job(tmp_path, capsys):
    queue, registry, _ = _interrupted(tmp_path, resumes=1)
    _reconcile(queue, registry, _GitHub(head="sha1", open_runs={9001}))
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    [decision] = [line for line in lines if line.get("event") == "job_resume"]
    assert (decision["verdict"], decision["closed_check_run_id"]) == ("cancelled", 9001)


# --- worker hooks: process_job reports its run, the pool registers/clears, stop() stamps -------

class _ProcessGitHub:
    def installation_token(self, *a, **k):
        return "tok"

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        return 4242

    def update_check_run(self, *a, **k):
        return None

    def find_open_pr_by_marker(self, *a, **k):
        return None


def test_process_job_hands_the_check_run_it_opened_to_the_registry_hook():
    seen = []
    result = process_job(_qjob(row_id=1), app_id="APP", private_key="pem", github=_ProcessGitHub(),
                         runner=lambda job, **kw: JobResult(True, "ok"),
                         resolve_engine_key=lambda iid: "org-key",
                         resolve_provider_credentials=lambda iid: (None, None),
                         resolve_fix_enabled=lambda iid: True,
                         on_check_run=seen.append)
    assert result.ok and seen == [4242]


class _RecordingQueue:
    def __init__(self, qjob=None):
        self.qjob = qjob
        self.completed = []

    def recover_stale(self):
        return {"requeued": 0, "failed": 0}

    def claim(self):
        qjob, self.qjob = self.qjob, None
        return qjob

    def complete(self, row_id, *, status, result=None, retry=True, retry_after_s=None):
        self.completed.append((row_id, status))


def test_pool_registers_a_claimed_job_and_clears_it_when_it_ends(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    seen = {}

    def process(qjob):
        seen["entry"] = registry.get(qjob.row_id)
        return JobResult(True, "ok")

    pool = WorkerPool(_RecordingQueue(), process=process, registry=registry)
    pool._run_claimed(_qjob(row_id=3))
    assert seen["entry"] is not None and seen["entry"].row_id == 3  # visible while running
    assert registry.get(3) is None  # cleared on finish


def test_pool_clears_the_registry_even_when_the_job_raises(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    pool = WorkerPool(_RecordingQueue(), registry=registry,
                      process=lambda _q: (_ for _ in ()).throw(RuntimeError("boom")))
    pool._run_claimed(_qjob(row_id=3))
    assert registry.entries() == []


def test_pool_without_a_registry_behaves_as_before(tmp_path):
    pool = WorkerPool(_RecordingQueue(), process=lambda _q: JobResult(True, "ok"))
    pool._run_claimed(_qjob(row_id=3))
    pool.start()
    assert pool._reconciled.is_set()  # nothing to reconcile: claims are not held
    pool.stop(timeout=1)


def test_pool_stop_stamps_a_job_that_outlives_the_drain_as_interrupted_by_sigterm(tmp_path):
    registry = RunningJobRegistry(tmp_path / "running")
    release = threading.Event()

    def slow(qjob):
        release.wait(5)
        return JobResult(True, "ok")

    queue = _RecordingQueue(_qjob(row_id=3))
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, process=slow, registry=registry)
    pool.start()
    assert _wait_until(lambda: registry.get(3) is not None)
    pool.stop(timeout=0.05)  # the grace runs out with the job still running: a redeploy's SIGTERM
    entry = registry.get(3)
    assert entry is not None and entry.interrupted_by == "sigterm" and entry.interrupted_at
    release.set()
    assert _wait_until(lambda: queue.completed)  # let the thread finish; it clears its own entry
    assert _wait_until(lambda: registry.get(3) is None)


def test_pool_start_resumes_an_interrupted_job_before_the_first_claim(tmp_path, monkeypatch):
    queue, registry, qjob = _interrupted(tmp_path)
    gh = _GitHub(head="sha1", open_runs={9001})
    monkeypatch.setattr(webhook_resume, "webhook_github", gh)
    ran = []
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, app_id="APP", private_key="pem",
                      registry=registry, process=lambda q: ran.append(q.attempts) or JobResult(True, "ok"))
    pool.start()
    assert _wait_until(lambda: queue.counts().get("done") == 1)
    pool.stop(timeout=2)
    assert ran == [2]  # the same job, attempt 2, exactly once
    assert gh.completed()[0][1:3] == (9001, "cancelled")  # its stale run was closed first
    assert registry.entries() == []


def test_pool_start_reruns_the_current_head_not_the_stale_one(tmp_path, monkeypatch):
    """End to end through the pool: the stale head's row is never claimed, and the head that
    replaced it is picked up on this same boot instead of waiting for a delivery that will not come."""
    queue, registry, stale = _interrupted(tmp_path, ref="old-sha")
    gh = _GitHub(head="new-sha", open_runs={9001})
    monkeypatch.setattr(webhook_resume, "webhook_github", gh)
    ran = []
    pool = WorkerPool(queue, concurrency=1, poll_interval=0.01, app_id="APP", private_key="pem",
                      registry=registry,
                      process=lambda q: ran.append((q.row_id, q.job.ref)) or JobResult(True, "ok"))
    pool.start()
    assert _wait_until(lambda: pool._reconciled.is_set())
    assert _wait_until(lambda: ran)
    pool.stop(timeout=2)
    assert [ref for _row, ref in ran] == ["new-sha"]       # the current head, exactly once
    assert all(row != stale.row_id for row, _ref in ran)   # never the stale row
    assert gh.completed()


# --- webhook_github.complete_check_run_if_open: completes a known run only while it is open ----

def _fake_httpx(*, status="in_progress", title="codna review", summary="codna review started (x)",
                get_status=200, patch_status=200, after_patch=None, readback_error=False):
    """httpx as complete_check_run_if_open uses it: ONE Client, GET and PATCH on it. `run` is the
    Check Run as GitHub holds it; a 2xx PATCH applies to it (GitHub answers 200 to the owning App
    even on a completed run), then `after_patch` -- what someone else writes right after ours."""
    run = {"id": 9001, "status": status, "conclusion": None, "head_sha": "sha1",
           "check_suite": {"id": 555}, "output": {"title": title, "summary": summary}}
    seen = {"gets": [], "patched": [], "clients": 0, "closed": 0}

    class _Resp:
        def __init__(self, code, payload):
            self.status_code, self._payload, self.text = code, payload, ""

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *, headers, timeout, follow_redirects):
            seen["clients"] += 1
            seen["headers"] = headers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            seen["closed"] += 1
            return False

        def get(self, url):
            seen["gets"].append(url)
            if readback_error and seen["patched"]:
                raise RuntimeError("transport")
            return _Resp(get_status, json.loads(json.dumps(run)))

        def patch(self, url, *, json: dict):
            seen["patched"].append((url, json["conclusion"], json["output"]["summary"]))
            if patch_status < 300:
                run.update(status="completed", conclusion=json["conclusion"],
                           output={"title": json["output"]["title"], "summary": json["output"]["summary"]})
                if after_patch:
                    run.update(after_patch)
            return _Resp(patch_status, {})

    return types.SimpleNamespace(Client=_Client), seen


def _warnings(capsys):
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()
            if line.strip() and json.loads(line).get("event") == "check_run_clobbered"]


def test_complete_check_run_if_open_completes_an_open_run_on_one_connection(monkeypatch, capsys):
    from codna.webhook_github import complete_check_run_if_open

    fake, seen = _fake_httpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="restart",
                                      name="codna review") is True
    url = seen["gets"][0]
    assert url.endswith("/repos/acme/app/check-runs/9001")
    assert seen["patched"] == [(url, "cancelled", "restart")]
    # read -> write -> read-back, all on the same client (no second connection between the read
    # and the write: that handshake WAS the window); the read-back matched, so nothing is warned
    assert seen["gets"] == [url, url] and (seen["clients"], seen["closed"]) == (1, 1)
    assert seen["headers"]["Authorization"] == "Bearer t"
    assert _warnings(capsys) == []


def test_complete_check_run_if_open_leaves_a_completed_run_and_its_findings_alone(monkeypatch):
    from codna.webhook_github import complete_check_run_if_open

    # `codna review --post` completed it with its findings before this boot read it
    fake, seen = _fake_httpx(status="completed", summary="3 findings, 1 blocking")
    monkeypatch.setitem(sys.modules, "httpx", fake)
    assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="restart",
                                      name="codna review") is False
    assert seen["patched"] == [] and len(seen["gets"]) == 1  # decided on the read right before the write


def test_complete_check_run_if_open_warns_when_someone_completed_it_right_after(monkeypatch, capsys):
    from codna.webhook_github import complete_check_run_if_open

    # the interrupted job's own CLI, still alive, wrote its findings right after this PATCH:
    # its content stands (good), the cancellation note is gone, and the race is now on record
    fake, seen = _fake_httpx(after_patch={"conclusion": "failure",
                                          "output": {"title": "codna review", "summary": "3 findings"}})
    monkeypatch.setitem(sys.modules, "httpx", fake)
    assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="restart",
                                      name="codna review") is True
    assert len(seen["patched"]) == 1
    [warning] = _warnings(capsys)
    assert (warning["service"], warning["level"]) == ("codna-webhook-worker", "warning")
    assert (warning["repo"], warning["check_run_id"], warning["check_suite_id"]) == ("acme/app", 9001, 555)
    assert (warning["written_summary"], warning["observed_summary"]) == ("restart", "3 findings")
    assert (warning["observed_conclusion"], warning["head_sha"]) == ("failure", "sha1")


def test_complete_check_run_if_open_read_back_is_diagnostics_only(monkeypatch, capsys):
    from codna.webhook_github import complete_check_run_if_open

    # a read-back that fails, or that is not yet consistent (still says in_progress), warns nothing
    for kwargs in ({"readback_error": True}, {"after_patch": {"status": "in_progress"}}):
        fake, seen = _fake_httpx(**kwargs)
        monkeypatch.setitem(sys.modules, "httpx", fake)
        assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="restart",
                                          name="codna review") is True
        assert len(seen["patched"]) == 1
    assert _warnings(capsys) == []


def test_complete_check_run_if_open_is_best_effort_and_never_raises(monkeypatch):
    from codna.webhook_github import complete_check_run_if_open

    for kwargs in ({"get_status": 404}, {"patch_status": 422}):
        fake, seen = _fake_httpx(**kwargs)
        monkeypatch.setitem(sys.modules, "httpx", fake)
        assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="s",
                                          name="codna review") is False
        assert len(seen["gets"]) == 1  # no read-back of a write that did not happen
    monkeypatch.setitem(sys.modules, "httpx", None)  # httpx not installed
    assert complete_check_run_if_open("acme/app", "t", 9001, conclusion="cancelled", summary="s",
                                      name="codna review") is False
