"""``codna webhook serve``: the ingress + queue + worker pool, with the queue backend and the
process ROLE chosen by environment.

This supersedes :func:`codna.webhook.serve` as the CLI's entrypoint and, with the defaults
(``CODNA_WEBHOOK_QUEUE_BACKEND=sqlite``, ``CODNA_WEBHOOK_ROLE=all``), does exactly what it did:
one SQLite queue on the volume, one in-process pool, the same handler. The additions only switch
on when the flag says so:

* backend ``shadow`` / ``postgres`` (``webhook_backend.open_queue``): the shared queue, leases,
  the reaper, per-tenant caps, retries with budgets, dead letters;
* role ``ingress``: verify + enqueue only, no worker pool; hosts the reaper and scaler leaders,
  the redelivery sweeper and the operator surface. Role ``worker``: no listener beyond
  ``/healthz`` + ``/metrics`` on a non-routed port; registers in ``workers`` and claims. Role
  ``all``: everything in one process (today's single machine, and Phase 2).
* ``/metrics`` (Prometheus text), ``/slo`` and the bearer-authenticated ``/ops/*`` on every role;
  ``/ready`` folds the backend's truth in (Postgres reachable or spooling with the forwarder
  alive; on the ingress, registered live workers >= floor) -- and stays a check for the canary
  and the deploy verify, NEVER the Fly routing check (``test_fly_toml.py`` guards that).

The routing health check remains ``/health`` (liveness only): the 2026-09-17 incident, when a
``/ready`` routing check unrouted the only machine and dropped every delivery, must not recur.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from typing import Any, Callable

from . import webhook_backend
from .webhook import _WEBHOOK_SECRET_ENV, _Handler, default_queue_path, require_webhook_secret

ROLE_ENV = "CODNA_WEBHOOK_ROLE"
ROLES = ("all", "ingress", "worker")


def role_name(environ: Any = None) -> str:
    source = os.environ if environ is None else environ
    role = (source.get(ROLE_ENV) or "all").strip().lower()
    if role not in ROLES:
        raise ValueError(f"{ROLE_ENV}={role!r}: expected one of {', '.join(ROLES)}")
    return role


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


class Service:
    """Everything one process runs, so ``/ready``, ``/metrics`` and the ops verbs can see it."""

    def __init__(self, *, queue: Any, backend: str, role: str, pool: Any = None, reaper: Any = None,
                 scaler: Any = None, sweeper: Any = None, registration: Any = None,
                 floor_workers: int = 0) -> None:
        self.queue = queue
        self.backend = backend
        self.role = role
        self.pool = pool
        self.reaper = reaper
        self.scaler = scaler
        self.sweeper = sweeper
        self.registration = registration
        self.floor_workers = int(floor_workers)
        self.started_at = time.time()
        # The ingress opens a review's Check Run `queued` at enqueue (webhook_queued_check) only on
        # the Postgres backend: the shared table carries the run's id to whichever worker claims
        # the row, and the reaper sweeps the runs of rows retired before a worker saw them. The
        # SQLite queue keeps create-at-claim (same process, one poll interval away, no reaper).
        self.queued_check_runs = backend == "postgres" and role in ("all", "ingress")

    def pg(self) -> Any:
        from .webhook_ops import _pg

        return _pg(self.queue)

    def local_gauges(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.backend, "role": self.role}
        if self.scaler is not None:
            out.update(self.scaler.gauges())
        if self.reaper is not None:
            out["reaper_leader"] = 1 if self.reaper.is_leader() else 0
        if isinstance(self.queue, webhook_backend.SpoolingQueue):
            try:
                out["spool_rows"] = self.queue.spool_rows()
            except Exception:  # noqa: BLE001
                out["spool_rows"] = -1
            out["degraded_lane"] = 1 if self.queue.degraded() else 0
        if isinstance(self.queue, webhook_backend.ShadowQueue):
            out["shadow_errors_total"] = self.queue.shadow_errors
            out["shadow_writes_total"] = self.queue.shadow_writes
        if self.pool is not None:
            try:
                diag = self.pool.diagnostics()
                out["pool_busy_threads"] = diag.get("busy_threads", 0)
                out["pool_alive_threads"] = diag.get("alive_threads", 0)
            except Exception:  # noqa: BLE001
                pass
        return out

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        """The truthful answer: config present, App auth signable, pool healthy (when this role
        has one), and the backend able to take a delivery. On ``postgres``: Postgres answers OR
        the spool is taking deliveries with its forwarder alive; on the ingress role, live
        registered workers >= the floor."""
        secret_ready = bool(os.environ.get(_WEBHOOK_SECRET_ENV)) or self.role == "worker"
        if os.environ.get("GITHUB_TOKEN"):
            app_ready = True
        else:
            from .webhook_github import app_auth_config_ready

            app_ready = app_auth_config_ready(os.environ.get("GITHUB_APP_ID"), os.environ.get("GITHUB_APP_PRIVATE_KEY"))
        pool_diag = self.pool.diagnostics() if self.pool is not None else None
        worker_ready = True if pool_diag is None else bool(pool_diag.get("ready"))
        checks: dict[str, Any] = {"webhook_secret": secret_ready, "github_app_auth": app_ready, "worker_pool": worker_ready}
        backend_ok = True
        pg = self.pg()
        if pg is not None:
            pg_up = pg.ping()
            checks["postgres"] = pg_up
            if isinstance(self.queue, webhook_backend.SpoolingQueue):
                spooling = self.queue.forwarder_alive()
                checks["spool_forwarder"] = spooling
                checks["spool_rows"] = self.queue.spool_rows()
                checks["degraded_lane"] = self.queue.degraded()
                backend_ok = pg_up or spooling
            else:
                backend_ok = pg_up or self.backend == "shadow"
            if self.role == "ingress" and pg_up:
                from . import webhook_pg_ops

                try:
                    live = [w for w in webhook_pg_ops.workers(pg) if not w.get("stale") and w.get("role") in ("worker", "all")]
                except Exception:  # noqa: BLE001
                    live = []
                checks["live_workers"] = len(live)
                checks["floor_workers"] = self.floor_workers
                backend_ok = backend_ok and len(live) >= self.floor_workers
        try:
            queue_counts = self.queue.counts()
        except Exception:  # noqa: BLE001 -- readiness must never fail because a diagnostic did
            queue_counts = None
        ready = secret_ready and app_ready and worker_ready and backend_ok
        return ready, {
            "service": "codna-webhook", "ok": ready, "backend": self.backend, "role": self.role, "checks": checks,
            "worker": None if pool_diag is None else {
                k: pool_diag.get(k) for k in ("alive_threads", "busy_threads", "longest_running_s", "wedged")},
            "queue": queue_counts,
        }

    def worker_liveness(self) -> tuple[bool, dict[str, Any]]:
        """Fresh registration heartbeat (within three intervals) and a live pool."""
        reg = self.registration
        if reg is None:
            return False, {"registered": False}
        last = reg.last or {}
        beat = last.get("heartbeat_at")
        fresh = False
        if beat:
            from .webhook_pg_ops import as_datetime

            when = as_datetime(beat)
            fresh = when is not None and (time.time() - when.timestamp()) <= 3 * reg.interval_s
        pool_ok = bool(self.pool.diagnostics().get("ready")) if self.pool is not None else True
        booting = reg.last is None and (time.time() - self.started_at) < 3 * reg.interval_s  # first beat pending
        return (fresh or booting) and pool_ok, {
            "registered": bool(last), "heartbeat_at": beat, "slots": last.get("slots"), "busy": last.get("busy"),
            "draining": bool(last.get("draining")), "pool_ready": pool_ok}

    def stop(self) -> None:
        for part in (self.scaler, self.reaper, self.sweeper):
            if part is not None:
                try:
                    part.stop()
                except Exception:  # noqa: BLE001
                    pass
        if self.pool is not None:
            self.pool.stop()  # drain in-flight jobs up to the grace window (no abandon on redeploy)
        if self.registration is not None:
            try:
                self.registration.stop()
            except Exception:  # noqa: BLE001
                pass
        if isinstance(self.queue, webhook_backend.SpoolingQueue):
            self.queue.stop()
        pg = self.pg()
        if pg is not None:
            try:
                pg.close()
            except Exception:  # noqa: BLE001
                pass


class _ServiceHandler(_Handler):
    """The webhook handler plus the operator routes. Everything ``_Handler`` does is unchanged;
    only paths it would 404 are taken here first."""

    server_version = "codna-webhook/3"

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0].rstrip("/")
        service: Service | None = getattr(self.server, "service", None)
        if service is None:
            super().do_GET()
            return
        if path in ("/health", "/healthz", "") and service.role == "worker":
            # A worker holds no webhook secret (deliveries arrive at the ingress), so the ingress's
            # "secret configured" liveness would be a permanent 503 here. Liveness for a worker is
            # "its heartbeat to the queue is fresh" -- the same thing the scaler trusts.
            ok, detail = service.worker_liveness()
            self._reply(200 if ok else 503, {"service": "codna-webhook-worker", "ok": ok, **detail})
            return
        if path == "/metrics":
            from .webhook_ops import run_verb

            text = run_verb(service.queue, "metrics", {}, actor="metrics", local_gauges=service.local_gauges)["text"]
            body = str(text).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/slo":
            from .webhook_ops import run_verb

            self._reply(200, run_verb(service.queue, "slo", {}, actor="slo"))
            return
        if path == "/ready":
            ready, payload = service.readiness()
            self._reply(200 if ready else 503, payload)
            return
        if path.startswith("/ops/"):
            from .webhook_ops import handle_ops_get

            status, payload = handle_ops_get(self.path, dict(self.headers.items()), queue=service.queue,
                                             scaler=service.scaler, sweeper=service.sweeper,
                                             local_gauges=service.local_gauges)
            if isinstance(payload, str):
                body = payload.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._reply(status, payload)
            return
        if path == "/debug/queue":
            # Moved under the ops token: the webhook HMAC secret is for GitHub's signatures only.
            from .webhook import _queue_debug_payload
            from .webhook_ops import ops_authorized

            if not ops_authorized(dict(self.headers.items())):
                self._reply(401, {"error": "unauthorized", "hint": "Authorization: Bearer $CODNA_WEBHOOK_OPS_TOKEN"})
                return
            self._reply(200, _queue_debug_payload(self.server))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        service: Service | None = getattr(self.server, "service", None)
        if service is not None and service.role == "worker":
            self._reply(404, {"error": "not_found", "hint": "worker role: deliveries go to the ingress"})
            return
        super().do_POST()


def build_service(*, environ: Any = None, queue: Any = None, pool_factory: Callable[..., Any] | None = None,
                  machines: Any = None, github: Any = None) -> Service:
    """Assemble the parts for the configured backend + role (pure wiring; tests pass fakes)."""
    e = os.environ if environ is None else environ
    backend = webhook_backend.backend_name(e)
    role = role_name(e)
    app_id, private_key = e.get("GITHUB_APP_ID"), e.get("GITHUB_APP_PRIVATE_KEY")
    from . import webhook_control
    from .webhook_pool import WorkerPool
    from .webhook_procs import prepare_scratch_root
    from .webhook_resume import RunningJobRegistry, registry_dir_for

    prepare_scratch_root()  # before anything touches tempfile: job scratch lives on the volume
    queue_path = default_queue_path()
    if queue is None:
        rollback = backend == "sqlite" and (e.get("CODNA_WEBHOOK_ROLLBACK_FROM_POSTGRES") or "0") in ("1", "true")
        queue = webhook_backend.open_queue(
            queue_path, backend=backend, environ=e, rollback_from_postgres=rollback,
            lease_s=float(e.get("CODNA_WEBHOOK_LEASE_S", "90")), heartbeat_s=float(e.get("CODNA_WEBHOOK_HEARTBEAT_S", "30")),
            tenant_default_cap=int(e.get("CODNA_WEBHOOK_TENANT_DEFAULT_CAP", "3")),
        )
    from .webhook_ops import _pg

    pg = _pg(queue)
    if pg is not None and backend == "postgres" and role in ("all", "worker"):
        # A lease this process can no longer renew means the row was taken from it: cancelled from
        # outside (a newer head arriving at ANY ingress, an operator's `cancel`) or reaped after a
        # missed heartbeat. The job still running here is stopped exactly as an in-process
        # supersede stops it -- JobHandle.cancel + settle: runtime torn down, Check Run closed
        # neutral, the row written once -- so a superseded review does not run on to post on a
        # commit nobody is merging, and a cancelled fix never opens its pull request. Without this
        # the ingress's ``supersede_running`` reached only its own process (measured 2026-09-20 on
        # the ingress + worker-pool shape: the old head's review ran to completion).
        pg.set_lease_lost_handler(_cancel_local_job(queue, github or _default_github()))
    concurrency = webhook_control.default_concurrency(e)  # memory-derived, CODNA_WEBHOOK_CONCURRENCY wins when set
    pool = None
    registration = None
    if role in ("all", "worker"):
        if pg is not None and backend == "postgres":
            # Rows carry the lease and the Check Run id; the JSON registry is the SQLite path's.
            registry = _RowRegistry(pg)
        else:
            registry = RunningJobRegistry(registry_dir_for(queue_path))
        make_pool = pool_factory or WorkerPool
        pool = make_pool(queue, concurrency=concurrency, app_id=app_id, private_key=private_key, registry=registry)
        default = getattr(pool, "_default_process", None)
        if callable(default) and hasattr(pool, "_process"):
            # Wrap the pool's own per-job entry (kept, not replaced: whatever bookkeeping it does
            # still happens) so the queue row id is visible to _job_env while the job runs. The
            # pool exposes no hook for this; it is the one place this module reaches into it.
            pool._process = _process_with_job_context(pool_default=default)
        if pg is not None and backend == "postgres":
            from .webhook_lease import WorkerRegistration

            registration = WorkerRegistration(pg, role=role, slots=concurrency,
                                              interval_s=float(e.get("CODNA_WEBHOOK_WORKER_HEARTBEAT_S", "10")),
                                              on_drained=_exit_when_drained if role == "worker" else None)
    reaper = scaler = sweeper = None
    if pg is not None and backend == "postgres" and role in ("all", "ingress"):
        from .webhook_lease import Reaper
        from .webhook_scaler import Scaler, ScalerConfig, machines_client_from_env

        reaper = Reaper(pg, app_id=app_id, private_key=private_key, github=github or _default_github(),
                        interval_s=float(e.get("CODNA_WEBHOOK_REAPER_INTERVAL_S", "15")))
        client = machines if machines is not None else machines_client_from_env(e)
        if client is not None and role == "ingress":
            scaler = Scaler(pg, client, ScalerConfig.from_env(e))
    if role in ("all", "ingress"):
        from . import webhook_redeliver

        if webhook_redeliver.enabled(e):
            sweeper = webhook_redeliver.Sweeper(app_id=app_id, private_key=private_key)
    floor = int(e.get("CODNA_WEBHOOK_FLOOR_MACHINES", "2")) if role == "ingress" else 0
    return Service(queue=queue, backend=backend, role=role, pool=pool, reaper=reaper, scaler=scaler,
                   sweeper=sweeper, registration=registration, floor_workers=floor)


def _default_github() -> Any:
    from . import webhook_github

    return webhook_github


def _cancel_local_job(queue: Any, github: Any) -> Callable[[int, str], None]:
    """The lease keeper's callback: cancel the job this process runs for a row it no longer holds.
    ``reason`` is the row's ``cancel_requested`` text (``superseded by <sha>``, an operator's
    reason) or ``lease_lost`` (reaped: another worker owns it now). A handle already cancelled --
    the ingress's own in-process supersede got there first -- is left to that path."""
    from . import webhook_control

    def _on_lease_lost(row_id: int, reason: str) -> None:
        handle = webhook_control.get_control().get(row_id)
        if handle is None or handle.cancelled:
            return
        _log("job_cancelled_by_queue", row_id=row_id, kind=handle.kind, repo=handle.repo, reason=reason)
        webhook_control.cancel_jobs([handle], reason="cancel_requested", summary=reason, github=github, queue=queue)

    return _on_lease_lost


def _process_with_job_context(*, pool_default: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the pool's per-job entry so the queue row id is visible to ``_job_env`` (thread-local)
    while the job runs: the CLI's review body then carries ``codna:review:run=<id>``."""

    def _process(qjob: Any) -> Any:
        webhook_backend.set_current_job(getattr(qjob, "row_id", None))
        try:
            return pool_default(qjob)
        finally:
            webhook_backend.set_current_job(None)

    return _process


class _RowRegistry:
    """``RunningJobRegistry``'s duck-typed surface backed by the Postgres row. ``start``/``finish``
    are no-ops (the claim and completion already are the record); ``record_check_run`` writes the
    id to the row so a reaper anywhere can complete the run; ``entries`` is empty because the
    reaper, not the boot reconciler, resolves interrupted rows."""

    def __init__(self, pg: Any) -> None:
        self._pg = pg

    @property
    def root(self) -> str:
        return f"postgres:{self._pg.schema}.jobs"

    def start(self, qjob: Any) -> None:
        return None

    def record_check_run(self, row_id: int, check_run_id: int | None) -> None:
        try:
            self._pg.record_check_run(row_id, check_run_id)
        except Exception as exc:  # noqa: BLE001 -- bookkeeping must never take a live job down
            _log("record_check_run_failed", row_id=row_id, error=type(exc).__name__)

    def finish(self, row_id: int) -> None:
        return None

    def mark_resumed(self, row_id: int) -> None:
        return None

    def mark_interrupted(self, reason: str) -> int:
        leases = self._pg.active_leases()
        for row_id, _owner in leases:
            try:
                self._pg.event(row_id, "interrupted_signal", detail={"reason": reason})
            except Exception:  # noqa: BLE001 -- a stamp that fails must not stop the shutdown
                pass
        return len(leases)

    def entries(self) -> list[Any]:
        return []

    def get(self, row_id: int) -> Any:
        return None


def _exit_when_drained() -> None:
    """A drained worker exits 0 so the machine can stop (the worker toml's restart policy is
    on-failure); the scaler stops it via the API in any case."""
    _log("worker_exit_drained")
    threading.Thread(target=lambda: (time.sleep(1.0), os.kill(os.getpid(), signal.SIGTERM)), daemon=True).start()


def serve(host: str = "0.0.0.0", port: int = 8080, *, service: Service | None = None) -> None:
    """Run the webhook (blocking). See the module docstring for what each backend/role adds."""
    svc = service or build_service()
    if svc.role != "worker":
        require_webhook_secret()
    if svc.pool is not None:
        svc.pool.start()
    if svc.registration is not None:
        svc.registration.start()
    for part in (svc.reaper, svc.scaler, svc.sweeper):
        if part is not None:
            part.start()
    httpd = ThreadingHTTPServer((host, port), _ServiceHandler)
    httpd.queue = svc.queue  # type: ignore[attr-defined]
    httpd.worker_pool = svc.pool  # type: ignore[attr-defined]
    httpd.service = svc  # type: ignore[attr-defined]
    httpd.queued_check_runs = svc.queued_check_runs  # type: ignore[attr-defined]
    _log("listening", host=host, port=port, backend=svc.backend, role=svc.role,
         path="/webhooks/github" if svc.role != "worker" else "/healthz",
         queued_check_runs=svc.queued_check_runs)

    def _graceful(_sig: int, _frame: Any) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _graceful)
        except (ValueError, OSError):
            pass  # not the main thread (e.g. under a test harness) — skip handler install
    try:
        httpd.serve_forever()
    finally:
        svc.stop()
        httpd.server_close()
