"""``codna webhook serve`` wiring (codna.webhook_service): the default flag keeps today's shape
exactly (SQLite queue, JSON registry, in-process pool, no reaper/scaler), the role/backend
validation, the operator routes on the handler (/metrics, /slo, /ready, /ops, the moved
/debug/queue), the worker role's liveness, and the review run id reaching the job env."""
from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from codna import webhook_backend
from codna.webhook_queue import WebhookQueue
from codna.webhook_resume import RunningJobRegistry
from codna.webhook_service import Service, _ServiceHandler, build_service, role_name


def test_role_defaults_to_all_and_refuses_typos():
    assert role_name({}) == "all"
    assert role_name({"CODNA_WEBHOOK_ROLE": " Worker "}) == "worker"
    with pytest.raises(ValueError, match="expected one of all, ingress, worker"):
        role_name({"CODNA_WEBHOOK_ROLE": "workers"})


class _Pool:
    def __init__(self, queue, **kw):
        self.queue = queue
        self.kw = kw
        self.started = False
        self.stopped = False

    def _default_process(self, qjob):
        return ("processed", qjob.row_id, webhook_backend.current_job_id())

    _process = None

    def start(self):
        self.started = True

    def stop(self, timeout=None):
        self.stopped = True

    def diagnostics(self):
        return {"ready": True, "alive_threads": 2, "busy_threads": 0, "longest_running_s": 0.0, "wedged": False}


def test_default_flag_builds_todays_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE", str(tmp_path / "queue.db"))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    for var in ("CODNA_WEBHOOK_QUEUE_BACKEND", "CODNA_WEBHOOK_ROLE", "DATABASE_URL", "CODNA_WEBHOOK_FLY_TOKEN", "FLY_API_TOKEN",
                "CODNA_WEBHOOK_REDELIVER"):
        monkeypatch.delenv(var, raising=False)
    svc = build_service(pool_factory=_Pool)
    assert (svc.backend, svc.role) == ("sqlite", "all")
    assert isinstance(svc.queue, WebhookQueue)
    assert isinstance(svc.pool.kw["registry"], RunningJobRegistry)          # the JSON registry beside the file
    from codna import webhook_control

    assert svc.pool.kw["concurrency"] == webhook_control.default_concurrency()   # main's own rule, unchanged
    assert svc.reaper is None and svc.scaler is None and svc.sweeper is None and svc.registration is None
    gauges = svc.local_gauges()
    assert gauges == {"backend": "sqlite", "role": "all", "pool_busy_threads": 0, "pool_alive_threads": 2}


def test_pool_process_wrapper_exposes_the_row_id_to_the_job_env(tmp_path, monkeypatch):
    from codna.webhook import WebhookJob
    from codna.webhook_queue import QueuedJob
    from codna.webhook_worker import _job_env

    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE", str(tmp_path / "queue.db"))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    for var in ("CODNA_WEBHOOK_QUEUE_BACKEND", "CODNA_WEBHOOK_ROLE", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    svc = build_service(pool_factory=_Pool)
    qjob = QueuedJob(row_id=31, delivery_id="d", attempts=1, job=WebhookJob("review", "o/r", ref="s", pr_number=1))
    assert svc.pool._process(qjob) == ("processed", 31, 31)                  # the id is visible WHILE the job runs
    assert webhook_backend.current_job_id() is None                            # ...and cleared afterwards
    webhook_backend.set_current_job(31)
    try:
        assert _job_env("tok", "ekey")["CODNA_REVIEW_RUN_ID"] == "31"          # what the CLI subprocess sees
    finally:
        webhook_backend.set_current_job(None)
    assert "CODNA_REVIEW_RUN_ID" not in _job_env("tok", "ekey")                # never outside a job


def test_postgres_backend_without_a_url_is_refused_at_boot(tmp_path, monkeypatch):
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE", str(tmp_path / "queue.db"))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CODNA_WEBHOOK_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="needs DATABASE_URL"):
        build_service(pool_factory=_Pool)


# --- the handler's operator routes -----------------------------------------------------------------------------

@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_token")
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", "ops-token")
    queue = WebhookQueue(tmp_path / "queue.db")
    svc = Service(queue=queue, backend="sqlite", role="all", pool=_Pool(queue))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ServiceHandler)
    httpd.queue = queue
    httpd.worker_pool = svc.pool
    httpd.service = svc
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path, headers=None):
        req = Request(base + path, headers=headers or {})
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
        except HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", ""), exc.read().decode()

    yield get, svc
    httpd.shutdown()
    httpd.server_close()


def test_metrics_slo_and_ready_are_served(served):
    get, svc = served
    status, ctype, body = get("/metrics")
    assert status == 200 and ctype.startswith("text/plain") and "codna_webhook_pg_up 0" in body
    assert 'codna_webhook_backend_info{backend="sqlite",role="all"} 1' in body
    status, _, body = get("/slo")
    assert status == 200 and json.loads(body)["verdict"] == "not_measured"
    status, _, body = get("/ready")
    payload = json.loads(body)
    assert status == 200 and payload["ok"] is True and payload["backend"] == "sqlite"
    assert payload["checks"] == {"webhook_secret": True, "github_app_auth": True, "worker_pool": True}
    assert payload["queue"] == {}                                               # counts, never content


def test_ops_routes_are_bearer_gated_and_debug_queue_moved_under_the_ops_token(served):
    get, _ = served
    assert get("/ops/jobs")[0] == 401
    assert get("/ops/jobs", {"Authorization": "Bearer s3cret"})[0] == 401          # the HMAC secret is not a key
    status, _, body = get("/ops/scaler", {"Authorization": "Bearer ops-token"})
    assert status == 200 and json.loads(body) == {"configured": False}
    status, _, body = get("/ops/jobs", {"Authorization": "Bearer ops-token"})
    assert status == 400 and "needs the postgres" in json.loads(body)["message"]
    assert get("/debug/queue", {"X-Codna-Debug-Token": "s3cret"})[0] == 401     # the old header no longer works
    status, _, body = get("/debug/queue", {"Authorization": "Bearer ops-token"})
    assert status == 200 and json.loads(body)["queue"]["configured"] is True
    assert get("/health")[0] == 200 and get("/nope")[0] == 404                   # everything else is the old handler


def test_worker_role_refuses_deliveries_and_reports_its_own_liveness(tmp_path, monkeypatch):
    monkeypatch.delenv("CODNA_GITHUB_WEBHOOK_SECRET", raising=False)
    queue = WebhookQueue(tmp_path / "queue.db")

    class _Reg:
        interval_s = 10.0
        last = {"heartbeat_at": "2000-01-01T00:00:00+00:00", "slots": 3, "busy": 1, "draining": False}

    svc = Service(queue=queue, backend="postgres", role="worker", pool=_Pool(queue), registration=_Reg())
    ok, detail = svc.worker_liveness()
    assert ok is False and detail["registered"] is True and detail["busy"] == 1       # a stale heartbeat is not alive
    from datetime import datetime, timezone

    _Reg.last = {**_Reg.last, "heartbeat_at": datetime.now(timezone.utc).isoformat()}
    ok, detail = svc.worker_liveness()
    assert ok is True and detail["slots"] == 3
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ServiceHandler)
    httpd.queue, httpd.worker_pool, httpd.service = queue, svc.pool, svc
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urlopen(base + "/healthz", timeout=5) as resp:
            assert resp.status == 200 and json.loads(resp.read())["service"] == "codna-webhook-worker"
        req = Request(base + "/webhooks/github", data=b"{}", method="POST")
        with pytest.raises(HTTPError) as exc:
            urlopen(req, timeout=5)
        assert exc.value.code == 404                                                     # deliveries go to the ingress
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_service_stop_drains_the_pool_and_closes_the_parts(tmp_path):
    class _Part:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    queue = WebhookQueue(tmp_path / "queue.db")
    pool, reaper, scaler = _Pool(queue), _Part(), _Part()
    svc = Service(queue=queue, backend="sqlite", role="all", pool=pool, reaper=reaper, scaler=scaler)
    svc.stop()
    assert pool.stopped and reaper.stopped and scaler.stopped


def test_cli_dispatch_routes_ops_verbs(monkeypatch):
    from codna import webhook_cli

    calls = []
    monkeypatch.setattr("codna.webhook_ops.cli_ops", lambda args: calls.append(("ops", args.verb)) or 0)
    monkeypatch.setattr("codna.webhook_ops.cli_migrate", lambda args: calls.append(("migrate",)) or 0)
    monkeypatch.setattr("codna.webhook_ops.cli_queue", lambda args: calls.append(("queue", args.direction)) or 0)

    class _Args:
        pass

    a = _Args()
    a.action_name, a.verb = "ops", "jobs"
    assert webhook_cli.cmd_webhook(a) == 0
    b = _Args()
    b.action_name = "migrate"
    assert webhook_cli.cmd_webhook(b) == 0
    c = _Args()
    c.action_name, c.direction = "queue", "import"
    assert webhook_cli.cmd_webhook(c) == 0
    assert calls == [("ops", "jobs"), ("migrate",), ("queue", "import")]


def test_argparse_exposes_the_new_subcommands():
    from codna.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["webhook", "ops", "jobs", "--kind", "review"])
    assert ns.action_name == "ops" and ns.verb == "jobs" and ns.kind == "review"
    ns = parser.parse_args(["webhook", "serve", "--role", "worker"])
    assert ns.action == "serve" and ns.role == "worker"
    ns = parser.parse_args(["webhook", "queue", "export"])
    assert ns.action_name == "queue" and ns.direction == "export"
    assert parser.parse_args(["webhook", "migrate"]).action_name == "migrate"
    assert parser.parse_args(["webhook"]).action == "serve"                    # bare `codna webhook` still serves


def test_a_lost_lease_cancels_the_job_this_process_runs(make_pg_queue, tmp_path, monkeypatch):
    """The Postgres backend's cross-machine cancellation, wired end to end: a row cancelled from
    outside (here a newer head enqueued as another ingress would) fails the worker's next heartbeat,
    and build_service's lease-lost handler stops the local job exactly as an in-process supersede
    does -- handle cancelled, row written once with the reason, the next claim is the new head."""
    from codna import webhook_control
    from codna.webhook import WebhookJob

    pg = make_pg_queue(heartbeat_s=0.05, lease_s=5)
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE", str(tmp_path / "queue.db"))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE_BACKEND", "postgres")
    monkeypatch.setenv("CODNA_WEBHOOK_ROLE", "worker")
    monkeypatch.delenv("CODNA_WEBHOOK_FLY_TOKEN", raising=False)
    monkeypatch.delenv("FLY_API_TOKEN", raising=False)
    build_service(queue=pg, pool_factory=_Pool)                     # the injected queue gets the handler too
    assert pg.enqueue(WebhookJob("review", "acme/app", ref="old", pr_number=7, installation_id=42, reason="t"), delivery_id="d1")
    claimed = pg.claim()
    handle = webhook_control.get_control().register(row_id=claimed.row_id, kind="review", repo="acme/app",
                                                    pr_number=7, ref="old", deadline_s=600)
    try:
        assert pg.enqueue(WebhookJob("review", "acme/app", ref="new", pr_number=7, installation_id=42, reason="t"), delivery_id="d2")
        for _ in range(200):
            if handle.cancelled and pg.row_state(claimed.row_id)[0] != "running":
                break
            time.sleep(0.05)
        assert handle.cancelled and handle.cancel_detail == "superseded by new"
        assert pg.row_state(claimed.row_id)[0] == "done"
        assert pg.job(claimed.row_id)["result"] == {"summary": "superseded by new", "cancelled": "cancel_requested"}
        nxt = pg.claim()
        assert nxt is not None and nxt.job.ref == "new"
    finally:
        webhook_control.get_control().release(claimed.row_id)
        pg.set_lease_lost_handler(None)
