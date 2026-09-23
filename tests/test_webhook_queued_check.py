"""The review Check Run opened `queued` by the ingress (codna.webhook_queued_check), on both queue
backends: the pure lifecycle with fakes (pre-create, bind-or-close, retired verdicts), the Postgres
ingress end to end (a signed delivery -> a queued run bound to the row -> the claim carries it), the
supersede of a WAITING row completing its run neutral, the reaper's backstop sweep, the duplicate
and late paths, and the SQLite backend keeping create-at-claim."""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from codna import webhook_pg_ops, webhook_queued_check as qc
from codna.webhook import WebhookJob, _Handler
from codna.webhook_queue import WebhookQueue
from codna.webhook_service import Service

SHA_A = "a" * 40
SHA_B = "b" * 40


def _job(kind="review", ref=SHA_A, pr=7, reason="pull_request_opened", **kw):
    return WebhookJob(kind, "acme/app", ref=ref, pr_number=pr, installation_id=42, reason=reason, **kw)


class _GitHub:
    """Every call the ingress, the worker and the reaper make on a Check Run, recorded."""

    def __init__(self, *, create_delay_s=0.0, fail_create=False, token="scoped-review-token"):
        self.calls: list[tuple] = []
        self.created: list[dict] = []
        self.updates: list[tuple[int, str, str]] = []
        self.open_runs: set[int] = set()
        self.completed: dict[int, tuple[str, str]] = {}
        self._next = 9000
        self._delay = create_delay_s
        self._fail_create = fail_create
        self._token = token
        self.lock = threading.Lock()

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", installation_id, kind))
        return self._token

    def create_check_run(self, repo, token, *, name, head_sha, summary, status="in_progress"):
        if self._delay:
            time.sleep(self._delay)
        if self._fail_create:
            raise RuntimeError("422 Validation Failed")
        with self.lock:
            self._next += 1
            cid = self._next
            self.created.append({"id": cid, "head_sha": head_sha, "status": status, "summary": summary, "name": name})
            self.open_runs.add(cid)
        return cid

    def start_check_run(self, repo, token, check_run_id, *, name, summary):
        self.calls.append(("start", check_run_id))

    def update_check_run(self, repo, token, check_run_id, *, conclusion, summary, name="codna"):
        with self.lock:
            self.updates.append((check_run_id, conclusion, summary))
            self.open_runs.discard(check_run_id)
            self.completed[check_run_id] = (conclusion, summary)

    def complete_check_run_if_open(self, repo, token, check_run_id, *, conclusion, summary, name):
        with self.lock:
            if check_run_id not in self.open_runs:
                return False
        self.update_check_run(repo, token, check_run_id, conclusion=conclusion, summary=summary, name=name)
        return True

    def check_run_status(self, repo, token, check_run_id):
        with self.lock:
            if check_run_id in self.open_runs:
                return "queued"
            return "completed" if check_run_id in self.completed else None

    def pull_request_head_sha(self, repo, token, number):
        return None

    def by_id(self, cid):
        return next(c for c in self.created if c["id"] == cid)


class _FakeQueue:
    """Only the questions precreate/bind_or_close ask, with scripted answers."""

    def __init__(self, *, seen=False, pending=False, bound=None, adopt=False, raising=False):
        self.seen, self.pending, self.bound, self.adopt, self.raising = seen, pending, bound, adopt, raising
        self.adopted: list[tuple] = []

    def has_delivery(self, delivery_id):
        if self.raising:
            raise ConnectionError("no route to postgres")
        return self.seen

    def has_pending(self, *, repo, pr_number, ref, kind):
        return self.pending

    def check_run_bound(self, delivery_id):
        if self.raising:
            raise ConnectionError("no route to postgres")
        return self.bound

    def adopt_check_run(self, job, check_run_id, *, delivery_id=None):
        self.adopted.append((check_run_id, delivery_id))
        return self.adopt


def _pre(job=None, **kw):
    gh = kw.pop("github", None) or _GitHub()
    queue = kw.pop("queue", None) or _FakeQueue()
    made = qc.precreate(job or _job(), queue=queue, delivery_id=kw.pop("delivery_id", "d1"), github=gh,
                        app_id="APP", private_key="pem", **kw)
    return made, gh, queue


# --- the pure lifecycle ---------------------------------------------------------------------------------

def test_only_a_review_of_a_known_head_is_pre_created():
    assert qc.wants_queued_check(_job())
    assert not qc.wants_queued_check(_job(ref=None, reason="comment_codna_review"))   # resolves its head when it runs
    assert not qc.wants_queued_check(_job(kind="fix", reason="check_suite_failure"))  # one of several deliveries per head
    assert not qc.wants_queued_check(_job(kind="queue", reason="merge_group_checks_requested"))
    assert not qc.wants_queued_check(_job(pr=None))


def test_precreate_opens_the_run_queued_on_the_event_head_with_the_jobs_own_token():
    made, gh, _ = _pre()
    assert made is not None and made.check_run_id == 9001 and made.head_sha == SHA_A and made.token == "scoped-review-token"
    [run] = gh.created
    assert run["status"] == "queued" and run["head_sha"] == SHA_A and run["name"] == "codna review"
    assert run["summary"] == "codna review queued (pull_request_opened); waiting for a worker"
    assert ("token", 42, "review") in gh.calls                                 # least privilege: the review scope


def test_precreate_skips_what_enqueue_would_collapse_and_when_the_queue_cannot_answer(capsys):
    """A run opened for a delivery that then collapses would be a second run on the commit: the
    dedup questions are asked first, and a queue that cannot answer (Postgres unreachable, the row is
    about to be spooled) means today's path."""
    for queue, reason in ((_FakeQueue(seen=True), "delivery_seen"), (_FakeQueue(pending=True), "head_already_pending"),
                          (_FakeQueue(raising=True), "queue_unavailable")):
        made, gh, _ = _pre(queue=queue)
        assert made is None and gh.created == []
        line = [json.loads(ln) for ln in capsys.readouterr().err.splitlines() if "check_run_precreate_skipped" in ln][-1]
        assert line["reason"] == reason
    assert _pre(job=_job(kind="fix"))[0] is None
    made, gh, _ = _pre(github=_GitHub(token=None))                            # nothing to mint with
    assert made is None and gh.created == []
    made, gh, _ = _pre(github=_GitHub(fail_create=True))                      # GitHub refused: create at claim
    assert made is None


def test_a_create_that_outruns_the_budget_adopts_onto_the_rows_own_row_or_is_closed():
    """GitHub answering after the budget must not leave an unowned `queued` run: once the caller's
    enqueue has returned (the gate), the late run is handed to this delivery's waiting row, or -- a
    worker already claimed it and opened its own -- completed at once."""
    gate = threading.Event()
    queue = _FakeQueue(adopt=True)
    made, gh, _ = _pre(github=_GitHub(create_delay_s=0.4), queue=queue, budget_s=0.05, late_gate=gate)
    assert made is None                                                         # the ingress moved on
    gate.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not queue.adopted:
        time.sleep(0.02)
    assert queue.adopted == [(9001, "d1")] and gh.updates == []                 # the row owns it now

    gate = threading.Event()
    queue = _FakeQueue(adopt=False)
    made, gh, _ = _pre(github=_GitHub(create_delay_s=0.4), queue=queue, budget_s=0.05, late_gate=gate)
    gate.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not gh.updates:
        time.sleep(0.02)
    [(cid, conclusion, summary)] = gh.updates
    assert (cid, conclusion) == (9001, "cancelled") and "another run" in summary


def test_bind_or_close_bound_adopted_closed():
    pre = qc.PreCreated(check_run_id=77, token="t", head_sha=SHA_A)
    gh = _GitHub()
    assert qc.bind_or_close(pre, job=_job(), queue=_FakeQueue(bound=77), delivery_id="d1", fresh=True, github=gh) == "bound"
    assert gh.updates == []
    # the insert collapsed onto a live row for the same head that had no run: it owns this one now
    q = _FakeQueue(bound=None, adopt=True)
    assert qc.bind_or_close(pre, job=_job(), queue=q, delivery_id="d1", fresh=False, github=gh) == "adopted"
    assert q.adopted == [(77, None)] and gh.updates == []
    # ... or that already has its own run: this one is closed `cancelled`, never a passing conclusion
    # (it is the NEWEST run of the name on the current head).
    assert qc.bind_or_close(pre, job=_job(), queue=_FakeQueue(bound=55, adopt=False), delivery_id="d1", fresh=False, github=gh) == "closed"
    [(cid, conclusion, summary)] = gh.updates
    assert (cid, conclusion) == (77, "cancelled") and "second delivery" in summary and "@codna review" in summary
    # Postgres failed at the insert (the row went to the spool): the worker opens a fresh run there
    gh = _GitHub()
    spooled = _FakeQueue(raising=True)
    spooled.spool = type("Spool", (), {"find_in_flight": staticmethod(lambda delivery_id: 41)})()
    assert qc.bind_or_close(pre, job=_job(), queue=spooled, delivery_id="d1", fresh=True, github=gh) == "closed"
    assert gh.updates[0][1] == "cancelled" and "spooled" in gh.updates[0][2]
    # ... but a Postgres blip on the READ-BACK after a successful insert is not a spooled delivery:
    # the row was born with this id, the run stays open for the worker (codna review of #596).
    gh = _GitHub()
    blip = _FakeQueue(raising=True)
    blip.spool = type("Spool", (), {"find_in_flight": staticmethod(lambda delivery_id: None)})()
    assert qc.bind_or_close(pre, job=_job(), queue=blip, delivery_id="d1", fresh=True, github=gh) == "bound"
    assert qc.bind_or_close(pre, job=_job(), queue=_FakeQueue(raising=True), delivery_id="d1", fresh=True, github=gh) == "bound"
    assert gh.updates == []
    # "cannot tell" is never read as "not spooled": a spool that lacks the finder, or whose file
    # raises, means the run is closed (the worker repairs a wrongly closed bound run by reading
    # it before reuse; a wrongly kept spooled run would have no owner at all).
    for cannot_tell in (type("Spool", (), {})(), type("Spool", (), {"find_in_flight": staticmethod(lambda d: (_ for _ in ()).throw(OSError("disk")))})()):
        gh = _GitHub()
        unknown = _FakeQueue(raising=True)
        unknown.spool = cannot_tell
        assert qc._in_spool(unknown, "d1") is None
        assert qc.bind_or_close(pre, job=_job(), queue=unknown, delivery_id="d1", fresh=True, github=gh) == "closed"
        assert gh.updates[0][1] == "cancelled"
    assert qc._in_spool(_FakeQueue(), "d1") is False and qc._in_spool(spooled, "d1") is True


def test_retired_verdicts():
    neutral, text = qc.retired_verdict({"kind": "review", "status": "superseded", "result": {"summary": "superseded by bbbbbbbb"}})
    assert neutral == "neutral" and text.startswith("codna review: superseded by bbbbbbbb") and "no longer the pull request head" in text
    cancelled, text = qc.retired_verdict({"kind": "fix", "status": "cancelled", "last_error": "cancelled:stale head"},
                                         retrigger={"fix": "Reply `@codna fix`."})
    assert cancelled == "cancelled" and "(stale head)" in text and text.endswith("Reply `@codna fix`.")
    assert qc.retired_verdict({"kind": "review", "status": "exported"})[0] == "cancelled"
    failure, text = qc.retired_verdict({"kind": "review", "status": "failed", "last_error": "installation token: 401"})
    assert failure == "failure" and "failed before it could report (installation token: 401)" in text


def test_the_service_switches_pre_creation_on_for_postgres_only(tmp_path):
    """SQLite keeps create-at-claim: no reaper sweeps a stranded run there, and the pool runs in the
    same process one poll interval after the enqueue."""
    queue = WebhookQueue(tmp_path / "q.db")
    assert Service(queue=queue, backend="sqlite", role="all").queued_check_runs is False
    assert Service(queue=queue, backend="shadow", role="all").queued_check_runs is False
    assert Service(queue=queue, backend="postgres", role="ingress").queued_check_runs is True
    assert Service(queue=queue, backend="postgres", role="all").queued_check_runs is True
    assert Service(queue=queue, backend="postgres", role="worker").queued_check_runs is False


def test_settle_retired_is_a_no_op_without_postgres(tmp_path):
    assert qc.settle_retired(WebhookQueue(tmp_path / "q.db"), github=_GitHub(), token_for=lambda row: "t") == 0


# --- the ingress end to end, on Postgres ----------------------------------------------------------------

def _serve(queue, github, *, secret="hook-secret", monkeypatch):
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)
    monkeypatch.setenv("GITHUB_APP_ID", "APP")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "pem")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.queue = queue
    httpd.github = github
    httpd.queued_check_runs = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def post(delivery_id, *, sha=SHA_A, action="opened", number=7):
        payload = json.dumps({
            "action": action, "repository": {"full_name": "acme/app"}, "installation": {"id": 42},
            "pull_request": {"number": number, "draft": False, "head": {"sha": sha}},
        }).encode()
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhooks/github", data=payload, method="POST",
            headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": sig,
                     "X-GitHub-Delivery": delivery_id, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    return httpd, post


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.usefixtures("pg_url")
def test_ingress_opens_the_run_queued_and_the_claim_carries_it(make_pg_queue, monkeypatch):
    """A signed pull_request delivery: 202 names the run, the row is born owning it with
    check_started_at stamped (the SLO clock), the worker's claim carries it -- and neither a
    redelivery of the same GUID nor a new GUID for the same head opens a second run."""
    q = make_pg_queue()
    gh = _GitHub()
    httpd, post = _serve(q, gh, monkeypatch=monkeypatch)
    try:
        t0 = time.monotonic()
        status, body = post("d1")
        assert status == 202 and body["check_run_id"] == 9001 and body["check_run"] == "bound"
        assert time.monotonic() - t0 < 5
        [run] = gh.created
        assert run["status"] == "queued" and run["head_sha"] == SHA_A
        row = q.job(int(q.recent(limit=1)[0]["id"]))
        assert row["check_run_id"] == 9001 and row["check_started_at"] is not None and row["status"] == "queued"
        assert [e["event"] for e in q.events(row["id"])] == ["enqueued"]
        assert q.events(row["id"])[0]["detail"]["check_run_id"] == 9001
        assert post("d1") == (200, {"status": "duplicate", "kind": "review", "repo": "acme/app",
                                    "reason": "pull_request_opened", "superseded_running": 0})
        status, body = post("d1-new-guid", action="synchronize")
        assert status == 200 and "check_run_id" not in body                # the head already has its live job
        assert len(gh.created) == 1
        claimed = q.claim()
        assert claimed is not None and claimed.check_run_id == 9001         # (e) the worker updates THIS run
        assert gh.updates == []
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.usefixtures("pg_url")
def test_a_newer_head_completes_the_waiting_rows_queued_run_neutral(make_pg_queue, monkeypatch):
    """(a) Push A then push B on one pull request while no worker is free: A's row is retired in B's
    insert transaction, and A's `queued` run ends neutral "superseded by <sha>" exactly as a RUNNING
    row's does today -- off the request thread, with B's own token. Exactly once (posted.check_closed)."""
    q = make_pg_queue()
    gh = _GitHub()
    httpd, post = _serve(q, gh, monkeypatch=monkeypatch)
    try:
        assert post("dA")[0] == 202
        status, body = post("dB", sha=SHA_B, action="synchronize")
        assert status == 202 and body["check_run_id"] == 9002 and body["check_run"] == "bound"
        assert _wait(lambda: len(gh.updates) == 1)
        [(cid, conclusion, summary)] = gh.updates
        assert (cid, conclusion) == (9001, "neutral")
        assert summary.startswith(f"codna review: superseded by {SHA_B[:8]}") and "no longer the pull request head" in summary
        by = {r["delivery_id"]: r for r in q.recent(limit=5)}
        assert by["dA"]["status"] == "superseded" and by["dB"]["status"] == "queued"
        assert q.job(by["dA"]["id"])["posted"].keys() == {"check_closed"}
        assert webhook_pg_ops.retired_check_runs(q) == []                    # nothing left for the reaper
        assert qc.settle_retired(q, github=gh, token_for=lambda row: "t") == 0 and len(gh.updates) == 1
        claimed = q.claim()
        assert claimed.job.ref == SHA_B and claimed.check_run_id == 9002
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.usefixtures("pg_url")
def test_the_reaper_sweeps_the_runs_of_rows_retired_without_a_worker(make_pg_queue):
    """(a)(c)(d) The backstop: an operator's cancel of a queued row, a rollback export, a row that
    failed before reporting, and a supersede the ingress did not settle. Each run is completed once,
    with the conclusion its ending warrants; a run someone else already completed is left alone."""
    from codna.webhook_lease import Reaper

    class _Held:
        held = True

        def try_acquire(self):
            return True

        def release(self):
            self.held = False

    q = make_pg_queue()
    gh = _GitHub()
    for i, ref in enumerate(("1" * 40, "2" * 40, "3" * 40, "4" * 40, "5" * 40)):
        cid = gh.create_check_run("acme/app", "t", name="codna review", head_sha=ref, summary="queued", status="queued")
        assert q.enqueue(_job(ref=ref, pr=10 + i), delivery_id=f"d{i}", check_run_id=cid)
    by = {r["delivery_id"]: r["id"] for r in q.recent(limit=10)}
    assert webhook_pg_ops.cancel(q, by["d0"], actor="ops:angel", reason="duplicate")["outcome"] == "cancelled"
    with q.connection() as conn:                                                  # a rollback's export
        conn.execute(f"UPDATE {q.schema}.jobs SET status='exported', finished_at=now() WHERE id=%s", (by["d1"],))
    claimed = q.claim()                                                           # d2 (oldest queued) fails before reporting
    q.complete(claimed.row_id, status="failed", retry=False, result={"error": "installation_token_failed", "message": "401"})
    q.enqueue(_job(ref="9" * 40, pr=13), delivery_id="d3-newer")                 # supersedes d3 (nobody settled it)
    gh.update_check_run("acme/app", "t", 9005, conclusion="success", summary="the CLI got there", name="codna review")
    with q.connection() as conn:                                                  # d4: retired, but its run is already complete
        conn.execute(f"UPDATE {q.schema}.jobs SET status='cancelled', finished_at=now(), last_error='cancelled:x' WHERE id=%s", (by["d4"],))
    reaper = Reaper(q, app_id="APP", private_key="pem", github=gh, environ={}, leader=_Held())
    out = reaper.tick()
    assert out["retired_checks_closed"] == 5
    verdicts = {cid: gh.completed[cid] for cid in (9001, 9002, 9003, 9004)}
    assert verdicts[9001][0] == "cancelled" and "(duplicate)" in verdicts[9001][1] and "@codna review" in verdicts[9001][1]
    assert verdicts[9002][0] == "cancelled" and "SQLite file" in verdicts[9002][1]
    assert verdicts[9003][0] == "failure" and "installation_token_failed" in verdicts[9003][1]
    assert verdicts[9004][0] == "neutral" and "superseded by 99999999" in verdicts[9004][1]
    assert gh.completed[9005] == ("success", "the CLI got there")               # left alone, marked settled
    assert reaper.tick()["retired_checks_closed"] == 0                           # exactly once
    assert all(q.job(by[d])["posted"].get("check_closed") for d in ("d0", "d1", "d2", "d3", "d4"))
    # retry-now keeps the row's run id (the worker asks GitHub whether it is still open before
    # reusing it) and clears `check_closed`, so a later retirement of the re-run sweeps its run.
    assert webhook_pg_ops.retry_now(q, by["d0"], actor="ops:angel")["outcome"] == "queued"
    row = q.job(by["d0"])
    assert row["check_run_id"] == 9001 and row["posted"] == {}
    assert q.claim().check_run_id == 9001                                        # the claim carries it; GitHub decides


@pytest.mark.usefixtures("pg_url")
def test_retry_now_before_the_sweep_leaves_the_still_open_run_for_the_worker(make_pg_queue):
    """An operator retries a failed row in the window before the reaper's sweep closed its run: the
    id stays on the row, the re-run's claim carries it, and the worker continues the open run instead
    of anything being stranded (codna review of #596)."""
    q = make_pg_queue()
    gh = _GitHub()
    cid = gh.create_check_run("acme/app", "t", name="codna review", head_sha=SHA_A, summary="queued", status="queued")
    assert q.enqueue(_job(), delivery_id="d1", check_run_id=cid)
    claimed = q.claim()
    q.complete(claimed.row_id, status="failed", retry=False, result={"error": "installation_token_failed", "message": "401"})
    assert webhook_pg_ops.retired_check_runs(q)[0]["id"] == claimed.row_id       # the sweep WOULD close it...
    assert webhook_pg_ops.retry_now(q, claimed.row_id, actor="ops:angel")["outcome"] == "queued"
    assert webhook_pg_ops.retired_check_runs(q) == []                            # ...but the row is live again
    again = q.claim()
    assert again.check_run_id == cid and gh.check_run_status("acme/app", "t", cid) == "queued"   # still open, reusable


@pytest.mark.usefixtures("pg_url")
def test_a_dead_rows_ingress_opened_run_is_completed_by_the_dead_letter_notice(make_pg_queue):
    """(d) A review that exhausts its transient budget goes `dead`; the reaper's dead-letter notice
    completes the run the row carries (update_check_run by id) -- the ingress-opened one included --
    so the pool's own close deliberately leaves `dead` rows to it (codna review of #596, finding 3)."""
    from codna.webhook_lease import Reaper

    class _Held:
        held = True

        def try_acquire(self):
            return True

        def release(self):
            self.held = False

    q = make_pg_queue()                                                          # not_before is cleared by hand below
    gh = _GitHub()
    cid = gh.create_check_run("acme/app", "t", name="codna review", head_sha=SHA_A, summary="queued", status="queued")
    assert q.enqueue(_job(), delivery_id="d1", check_run_id=cid)
    budget = q.policy_for("review").budget
    for _ in range(budget):
        with q.connection() as conn:
            conn.execute(f"UPDATE {q.schema}.jobs SET not_before = NULL")
        claimed = q.claim()
        assert claimed is not None and claimed.check_run_id == cid
        q.complete(claimed.row_id, status="failed", retry=True, result={"error": "job_crashed", "message": "connection reset by peer"})
    assert q.row_state(claimed.row_id)[0] == "dead"
    reaper = Reaper(q, app_id="APP", private_key="pem", github=gh, environ={}, leader=_Held())
    out = reaper.tick()
    assert out["dead_letters_notified"] == 1 and out["retired_checks_closed"] == 0
    conclusion, text = gh.completed[cid]
    assert conclusion == "neutral" and "could not complete" in text


@pytest.mark.usefixtures("pg_url")
def test_a_duplicate_delivery_racing_the_insert_never_strands_a_run(make_pg_queue):
    """The window the dedup questions leave open: two deliveries for one head in the same instant.
    The second insert collapses; its run is handed to the live row when that row has none (a
    requeue's), else closed at once."""
    q = make_pg_queue()
    gh = _GitHub()
    # a row a worker's requeue wrote (no run) -- the ingress's pre-created run for the same head is adopted
    assert q.enqueue(WebhookJob("review", "acme/app", ref=SHA_A, pr_number=7, installation_id=42,
                                reason="head_moved_during_review"), delivery_id="requeue-1", priority=1)
    pre = qc.PreCreated(check_run_id=gh.create_check_run("acme/app", "t", name="codna review", head_sha=SHA_A,
                                                         summary="queued", status="queued"), token="t", head_sha=SHA_A)
    assert q.enqueue(_job(), delivery_id="d-late", check_run_id=pre.check_run_id) is False
    assert qc.bind_or_close(pre, job=_job(), queue=q, delivery_id="d-late", fresh=False, github=gh) == "adopted"
    row = q.job(int(q.recent(limit=1)[0]["id"]))
    assert row["check_run_id"] == 9001 and row["check_started_at"] is not None
    assert [e["event"] for e in q.events(row["id"])][-1] == "check_run_adopted"
    assert q.claim().check_run_id == 9001
    # the live row already has its run: the second one is closed cancelled
    pre2 = qc.PreCreated(check_run_id=gh.create_check_run("acme/app", "t", name="codna review", head_sha=SHA_A,
                                                          summary="queued", status="queued"), token="t", head_sha=SHA_A)
    assert q.enqueue(_job(), delivery_id="d-later", check_run_id=pre2.check_run_id) is False
    assert qc.bind_or_close(pre2, job=_job(), queue=q, delivery_id="d-later", fresh=False, github=gh) == "closed"
    assert gh.completed[9002][0] == "cancelled"
    assert q.check_run_bound("d-late") is None and q.check_run_bound("requeue-1") == 9001
    assert q.has_delivery("requeue-1") and not q.has_delivery("d-late")
    # adoption by delivery id (the late path) is conditional on the row still waiting without a run
    assert q.adopt_check_run(_job(), 4242, delivery_id="requeue-1") is False    # running now


@pytest.mark.usefixtures("pg_url")
def test_the_spool_drops_the_pre_created_run_and_the_ingress_closes_it(tmp_path, make_pg_queue, monkeypatch):
    """Postgres fails between the dedup question and the insert: the delivery lands in the SQLite
    spool WITHOUT the run (the file has no owner for it), the ingress finds it bound to no row and
    completes it, and the forwarded row gets a fresh run from its worker -- never a stranded one."""
    from codna import webhook_backend as wb

    class _Flaky:
        def __init__(self, real):
            self._real, self.down = real, False

        def enqueue(self, *a, **k):
            if self.down:
                raise ConnectionError("no route to postgres")
            return self._real.enqueue(*a, **k)

        def check_run_bound(self, delivery_id):
            if self.down:
                raise ConnectionError("no route to postgres")
            return self._real.check_run_bound(delivery_id)

        def __getattr__(self, name):
            return getattr(self._real, name)

    flaky = _Flaky(make_pg_queue())
    spool = wb.SpoolingQueue(flaky, WebhookQueue(tmp_path / "queue.db"), enqueue_timeout_s=2.0)
    gh = _GitHub()
    httpd, post = _serve(spool, gh, monkeypatch=monkeypatch)
    try:
        flaky.down = True
        status, body = post("d-spooled")
        assert status == 202 and body["check_run"] == "closed"
        assert gh.completed[9001][0] == "cancelled" and "spooled" in gh.completed[9001][1]
        assert spool.spool_rows() == 1
        flaky.down = False
        assert spool.forward_once() == 1
        claimed = spool.claim()
        assert claimed is not None and claimed.check_run_id is None            # the worker creates at claim
    finally:
        httpd.shutdown()
        spool.stop()
        httpd.server_close()


def test_sqlite_ingress_keeps_create_at_claim(tmp_path, monkeypatch):
    """The SQLite backend never pre-creates (the service leaves the switch off); a handler wired that
    way opens no run and the claim carries none, so the worker creates it -- today's behaviour."""
    q = WebhookQueue(tmp_path / "q.db")
    gh = _GitHub()
    httpd, post = _serve(q, gh, monkeypatch=monkeypatch)
    httpd.queued_check_runs = False
    try:
        status, body = post("d1")
        assert status == 202 and "check_run_id" not in body and gh.created == []
        assert q.claim().check_run_id is None
    finally:
        httpd.shutdown()
        httpd.server_close()
