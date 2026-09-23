"""The operator surface (codna.webhook_ops + webhook_pg_ops + webhook_metrics): bearer auth in
constant time with its OWN secret, every verb through the one implementation the HTTP route, the
CLI and the ops workflow share, every mutation leaving a job_events row with its actor, the
Prometheus rendering, and the SLO computation. Postgres-backed (CODNA_TEST_DATABASE_URL)."""
from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone

import pytest

from codna import webhook_metrics, webhook_ops, webhook_pg_ops
from codna.webhook import WebhookJob
from codna.webhook_ops import VERBS, handle_ops_get, ops_authorized, run_verb

pytestmark = pytest.mark.usefixtures("pg_url")
TOKEN = "ops-secret-token-value"


def _job(kind="review", ref="sha1", pr=7, installation=42):
    return WebhookJob(kind, "acme/app", ref=ref, pr_number=pr, installation_id=installation, reason="pull_request_opened")


# --- auth ---------------------------------------------------------------------------------------------

def test_ops_auth_requires_its_own_bearer_token_and_is_off_without_one(monkeypatch):
    assert ops_authorized({"Authorization": f"Bearer {TOKEN}"}, {}) is False            # no token configured: OFF
    env = {"CODNA_WEBHOOK_OPS_TOKEN": TOKEN, "CODNA_GITHUB_WEBHOOK_SECRET": "hmac-secret"}
    assert ops_authorized({"Authorization": f"Bearer {TOKEN}"}, env) is True
    assert ops_authorized({"authorization": f"Bearer {TOKEN}"}, env) is True               # header case
    assert ops_authorized({"Authorization": "Bearer hmac-secret"}, env) is False          # the webhook HMAC secret is NOT accepted
    assert ops_authorized({"Authorization": f"Bearer {TOKEN}x"}, env) is False
    assert ops_authorized({"Authorization": TOKEN}, env) is False                          # must be a Bearer
    assert ops_authorized({}, env) is False
    compared = []
    real = hmac.compare_digest
    monkeypatch.setattr(webhook_ops.hmac, "compare_digest", lambda a, b: compared.append((a, b)) or real(a, b))
    ops_authorized({"Authorization": "Bearer nope"}, env)
    assert compared == [(TOKEN, "nope")]                                                  # constant-time compare, always


def test_http_front_answers_401_then_404_then_400(make_pg_queue, monkeypatch):
    q = make_pg_queue()
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", TOKEN)
    assert handle_ops_get("/ops/jobs", {}, queue=q)[0] == 401
    ok = {"Authorization": f"Bearer {TOKEN}"}
    status, payload = handle_ops_get("/ops/nonsense", ok, queue=q)
    assert status == 404 and payload["verbs"] == list(VERBS)
    status, payload = handle_ops_get("/ops/cancel", ok, queue=q)                          # id missing
    assert status == 400 and "id is required" in payload["message"]
    status, payload = handle_ops_get("/ops/cancel?id=abc", ok, queue=q)
    assert status == 400 and "integer" in payload["message"]


# --- verbs --------------------------------------------------------------------------------------------

def test_every_mutation_writes_job_events_with_the_actor(make_pg_queue, monkeypatch):
    q = make_pg_queue()
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", TOKEN)
    ok = {"Authorization": f"Bearer {TOKEN}"}
    for i in range(3):
        q.enqueue(_job(ref=f"s{i}", pr=i), delivery_id=f"d{i}")
    rows = {r["delivery_id"]: r["id"] for r in q.recent(limit=10)}
    status, out = handle_ops_get(f"/ops/reprioritize?id={rows['d0']}&priority=7&actor=angel", ok, queue=q)
    assert (status, out["outcome"], out["priority"]) == (200, "reprioritized", 7)
    status, out = handle_ops_get(f"/ops/cancel?id={rows['d1']}&reason=duplicate&actor=angel", ok, queue=q)
    assert (status, out["outcome"]) == (200, "cancelled")
    status, out = handle_ops_get(f"/ops/retry-now?id={rows['d1']}&actor=angel", ok, queue=q)
    assert (status, out["outcome"]) == (200, "queued")
    running = q.claim()
    status, out = handle_ops_get(f"/ops/cancel?id={running.row_id}&actor=angel", ok, queue=q)
    assert out["outcome"] == "cancel_requested" and q.heartbeat(running.row_id) is False
    status, out = handle_ops_get(f"/ops/cancel?id={running.row_id}&actor=angel", ok, queue=q)
    assert out["outcome"] == "cancel_requested"                                             # idempotent
    events = {rid: [(e["event"], e["actor"]) for e in q.events(rid)] for rid in rows.values()}
    assert ("reprioritized", "ops-http:angel") in events[rows["d0"]]
    assert ("cancelled", "ops-http:angel") in events[rows["d1"]] and ("retry_now", "ops-http:angel") in events[rows["d1"]]
    assert ("cancel_requested", "ops-http:angel") in events[running.row_id]
    status, out = handle_ops_get(f"/ops/job?id={rows['d1']}", ok, queue=q)
    assert status == 200 and out["status"] == "queued" and [e["event"] for e in out["events"]][0] == "enqueued"
    assert "context" not in out                                                             # never job content


def test_jobs_listing_filters_and_dead_listing(make_pg_queue):
    q = make_pg_queue()
    q.enqueue(_job(kind="fix", ref="a", pr=1, installation=1), delivery_id="d1")
    q.enqueue(_job(kind="review", ref="b", pr=2, installation=2), delivery_id="d2")
    out = run_verb(q, "jobs", {"kind": "fix"}, actor="t")
    assert [j["delivery_id"] for j in out["jobs"]] == ["d1"]
    out = run_verb(q, "jobs", {"installation": "2", "status": "queued"}, actor="t")
    assert [j["delivery_id"] for j in out["jobs"]] == ["d2"]
    assert run_verb(q, "dead", {}, actor="t")["dead"] == []


def test_tenant_verbs(make_pg_queue):
    q = make_pg_queue()
    assert run_verb(q, "tenant", {"installation": "5"}, actor="t")["tenant"]["defaults"] is True
    out = run_verb(q, "tenant", {"installation": "5", "action": "pause"}, actor="ops-cli:angel")
    assert out["tenant"]["paused"] is True and out["tenant"]["updated_by"] == "ops-cli:angel"
    assert run_verb(q, "tenant", {"installation": "5", "action": "cap", "value": "2"}, actor="t")["tenant"]["max_concurrency"] == 2
    assert run_verb(q, "tenant", {"installation": "5", "action": "weight", "value": "99"}, actor="t")["tenant"]["weight"] == 16  # clamped
    assert run_verb(q, "tenant", {"installation": "5", "action": "dead-letter", "value": "action_required"}, actor="t")["tenant"]["dead_letter_conclusion"] == "action_required"
    assert run_verb(q, "tenant", {"installation": "5", "action": "dead-letter", "value": "default"}, actor="t")["tenant"]["dead_letter_conclusion"] is None
    assert run_verb(q, "tenant", {"installation": "5", "action": "resume"}, actor="t")["tenant"]["paused"] is False
    with pytest.raises(ValueError):
        run_verb(q, "tenant", {"installation": "5", "action": "explode"}, actor="t")
    assert [t["installation_id"] for t in run_verb(q, "tenants", {}, actor="t")["tenants"]] == [5]


def test_worker_verbs_drain_and_undrain(make_pg_queue):
    q = make_pg_queue(owner="m1:1")
    webhook_pg_ops.register_worker(q, machine_id="m1", region="iad", image="i", role="worker", slots=3)
    out = run_verb(q, "drain", {"owner": "m1:1"}, actor="t")
    assert out["worker"]["draining"] is True
    assert run_verb(q, "undrain", {"owner": "m1:1"}, actor="t")["worker"]["draining"] is False
    with pytest.raises(ValueError, match="not found"):
        run_verb(q, "drain", {"owner": "ghost"}, actor="t")
    assert run_verb(q, "workers", {}, actor="t")["workers"][0]["owner"] == "m1:1"


def test_verbs_that_need_postgres_say_so_on_sqlite(tmp_path):
    from codna.webhook_queue import WebhookQueue

    sqlite = WebhookQueue(tmp_path / "q.db")
    with pytest.raises(ValueError, match="needs the postgres"):
        run_verb(sqlite, "jobs", {}, actor="t")
    assert run_verb(sqlite, "slo", {}, actor="t")["verdict"] == "not_measured"
    text = run_verb(sqlite, "metrics", {}, actor="t", local_gauges=lambda: {"backend": "sqlite", "role": "all"})["text"]
    assert "codna_webhook_pg_up 0" in text and 'backend="sqlite"' in text
    assert run_verb(sqlite, "scaler", {}, actor="t") == {"configured": False}


# --- metrics + slo ------------------------------------------------------------------------------------------

def test_metrics_snapshot_and_rendering_cover_the_documented_series(make_pg_queue):
    now = [datetime.now(timezone.utc)]
    q = make_pg_queue(clock=lambda: now[0])
    for i in range(3):
        q.enqueue(_job(ref=f"s{i}", pr=i, installation=1 + i % 2), delivery_id=f"d{i}")
    claimed = q.claim()
    now[0] += timedelta(seconds=12)
    q.complete(claimed.row_id, status="done")
    q.enqueue(_job(kind="fix", ref="f", pr=9), delivery_id="fix")
    fix = q.claim(classes=[3])                                                              # the fix, not the next review
    assert fix.job.kind == "fix"
    q.complete(fix.row_id, status="failed", result={"summary": "provider 503"})
    running = q.claim()                                                                     # one review in flight at snapshot time
    assert running.job.kind == "review"
    snap = webhook_pg_ops.metrics_snapshot(q)
    assert snap["pg_up"] == 1 and snap["dead_total"] == 0
    assert sum(snap["queue_depth"].values()) == 1 and snap["running"] == {"review": 1}     # one review still runnable, one running
    assert snap["events_window"].get("retry") == 1
    assert snap["job_wait_s"]["review"]["count"] == 2 and snap["job_duration_s"]["review"]["count"] == 1
    text = webhook_metrics.render(snap, local={"backend": "postgres", "role": "ingress", "scaler_desired_machines": 2,
                                               "spool_rows": 0})
    for series in ("codna_webhook_queue_depth{", "codna_webhook_oldest_runnable_age_seconds{", "codna_webhook_running{",
                   "codna_webhook_jobs_total{", "codna_webhook_job_wait_seconds_bucket{", "codna_webhook_job_duration_seconds_bucket{",
                   "codna_webhook_retries_total 1", "codna_webhook_dead_total 0", "codna_webhook_workers{state=\"live\"}",
                   "codna_webhook_pg_up 1", "codna_webhook_scaler_desired_machines 2", "codna_webhook_spool_rows 0",
                   'codna_webhook_backend_info{backend="postgres",role="ingress"} 1'):
        assert series in text, series
    assert 'le="+Inf"' in text and 'le="60"' in text
    assert text.endswith("\n")


def test_metrics_rendering_bounds_installation_labels(make_pg_queue):
    q = make_pg_queue(tenant_default_cap=100)
    for i in range(40):
        q.enqueue(_job(ref=f"s{i}", pr=i, installation=1000 + i), delivery_id=f"d{i}")
    snap = webhook_pg_ops.metrics_snapshot(q, top_installations=5)
    labels = {inst for (_, inst) in snap["queue_depth"]}
    assert len(labels) <= 6 and "other" in labels                                           # top-N by name, the rest folded


def test_slo_snapshot_measures_the_check_creation_wait(make_pg_queue):
    now = [datetime.now(timezone.utc)]
    q = make_pg_queue(clock=lambda: now[0], tenant_default_cap=100)
    waits = [5, 10, 20, 30, 40, 50, 55, 58, 59, 61, 70, 90, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for i, wait in enumerate(waits):
        q.enqueue(_job(ref=f"s{i}", pr=i), delivery_id=f"d{i}")
        now[0] += timedelta(seconds=wait)
        claimed = q.claim()
        q.record_check_run(claimed.row_id, 100 + i)                                        # the check exists: the SLO clock stops
        q.complete(claimed.row_id, status="done")
    slo = webhook_pg_ops.slo_snapshot(q)
    ordered = sorted(waits)
    assert slo["samples"] == len(waits)
    assert slo["p95_s"] == pytest.approx(ordered[round(0.95 * (len(waits) - 1))], abs=0.5)
    assert slo["verdict"] == ("met" if slo["p95_s"] <= 60 else "breached") and slo["met"] is (slo["p95_s"] <= 60)
    thin = webhook_pg_ops.slo_snapshot(q, min_samples=len(waits) + 1)
    assert thin["verdict"] == "insufficient_samples" and thin["met"] is None
