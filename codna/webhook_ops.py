"""The operator surface of the webhook: authenticated ``/ops/*`` over HTTP, ``/metrics`` and
``/slo``, and the ``codna webhook ops …`` / ``migrate`` / ``queue export|import`` subcommands.

One implementation of every verb (:func:`run_verb`) behind three fronts:

* HTTP on the ingress, ``Authorization: Bearer $CODNA_WEBHOOK_OPS_TOKEN`` (its own secret,
  compared in constant time; the webhook HMAC secret is NOT reused -- ``/debug/queue`` used to,
  and stops here);
* the CLI, run inside a machine with ``DATABASE_URL`` (``fly ssh console -C "codna webhook ops …"``).
  A verb whose answer is the SERVING process's state rather than a table (:data:`LIVE_VERBS`:
  ``scaler``) is asked of the running server over its own loopback ``/ops/<verb>``, with the
  machine's ``CODNA_WEBHOOK_OPS_TOKEN`` -- a separate CLI process builds no Scaler, so its own
  view could only ever say "not configured" (it did, 2026-09-20, while the token was set). The
  answer carries ``"view": "live"``; when that call cannot be made the CLI's own view is printed
  with ``"view": "cli-rebuild"`` and the reason;
* ``webhook-ops.yml`` (``workflow_dispatch``), which wraps the CLI so nobody needs a Fly login.

Every mutation records its ``actor`` in ``job_events``. Nothing here is reachable on the
``sqlite`` backend beyond the read-only ``/metrics`` (process-local gauges) and ``/slo``
(``backend: sqlite`` -- no measurement): the SQLite file has no leases, tenants or workers.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlsplit

from . import webhook_pg_ops
from .webhook_pg_queue import PostgresQueue

OPS_TOKEN_ENV = "CODNA_WEBHOOK_OPS_TOKEN"
VERBS = ("jobs", "job", "cancel", "retry-now", "reprioritize", "tenant", "tenants", "workers", "drain", "undrain",
         "scaler", "slo", "dead", "metrics", "redeliver-sweep", "events")
# Verbs whose answer is the state of the serving process (its Scaler thread), not a row in the
# shared table: a separate CLI process has none of it, so the CLI asks the live server instead.
LIVE_VERBS = ("scaler",)
LIVE_PORT_ENV = "CODNA_WEBHOOK_PORT"


def ops_authorized(headers: Mapping[str, str], environ: Mapping[str, str] | None = None) -> bool:
    """Bearer token, constant-time. No token configured = the surface is OFF (401 for everyone)."""
    source = os.environ if environ is None else environ
    expected = (source.get(OPS_TOKEN_ENV) or "").strip()
    if not expected:
        return False
    header = headers.get("Authorization") or headers.get("authorization") or ""
    provided = header[len("Bearer "):].strip() if header.startswith("Bearer ") else ""
    return bool(provided) and hmac.compare_digest(expected, provided)


def _pg(queue: Any) -> PostgresQueue | None:
    """The PostgresQueue under whatever wrapper the service runs (SpoolingQueue / ShadowQueue), or
    None on the sqlite backend."""
    if isinstance(queue, PostgresQueue):
        return queue
    for attr in ("primary", "shadow"):
        inner = getattr(queue, attr, None)
        if isinstance(inner, PostgresQueue):
            return inner
    return None


def run_verb(queue: Any, verb: str, args: Mapping[str, Any], *, actor: str,
             scaler: Any = None, sweeper: Any = None, local_gauges: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Execute one operator verb. ``args`` are already-parsed strings/ints (from the query string
    or argparse). Raises ``ValueError`` for a bad request (the HTTP front answers 400)."""
    pg = _pg(queue)
    if verb == "metrics":
        from . import webhook_metrics

        snap = None
        if pg is not None:
            try:
                snap = webhook_pg_ops.metrics_snapshot(pg)
            except Exception as exc:  # noqa: BLE001 -- pg_up 0 says it; the log says why
                print(json.dumps({"service": "codna-webhook", "event": "metrics_snapshot_error", "error": type(exc).__name__}),
                      file=sys.stderr, flush=True)
                snap = None
        return {"text": webhook_metrics.render(snap, local=local_gauges() if local_gauges else {})}
    if verb == "slo":
        if pg is None:
            return {"backend": "sqlite", "verdict": "not_measured",
                    "slo": "95% of codna review checks created within 60s of the queued delivery (Postgres backend only)"}
        return webhook_pg_ops.slo_snapshot(pg, window_s=float(args.get("window_s") or 3600),
                                           target_s=float(args.get("target_s") or 60), min_samples=int(args.get("min_samples") or 20))
    if verb == "scaler":
        if scaler is None:
            return {"configured": False}
        action = str(args.get("action") or "status")
        if action == "pause":
            scaler.pause(True)
        elif action == "resume":
            scaler.pause(False)
        elif action != "status":
            raise ValueError("scaler action must be status | pause | resume")
        return {"configured": True, **scaler.status()}
    if verb == "redeliver-sweep":
        if sweeper is None:
            return {"configured": False}
        return {"configured": True, "redelivered": sweeper.sweep(), "errors_total": sweeper.errors_total}
    if pg is None:
        raise ValueError(f"`{verb}` needs the postgres (or shadow) backend; this process runs on sqlite")
    if verb == "jobs":
        inst = args.get("installation")
        return {"jobs": webhook_pg_ops.list_jobs(pg, status=args.get("status") or None, kind=args.get("kind") or None,
                                                 installation_id=int(inst) if inst not in (None, "") else None,
                                                 repo=args.get("repo") or None, limit=int(args.get("limit") or 50))}
    if verb == "dead":
        since_h = float(args.get("since_h") or 24)
        rows = webhook_pg_ops.list_jobs(pg, status="dead", limit=int(args.get("limit") or 200))
        return {"dead": rows, "since_h": since_h}
    if verb in ("job", "events"):
        rid = _int_arg(args, "id")
        detail = webhook_pg_ops.job_detail(pg, rid)
        if detail is None:
            raise ValueError(f"job {rid} not found")
        return detail if verb == "job" else {"id": rid, "events": detail["events"]}
    if verb == "cancel":
        return webhook_pg_ops.cancel(pg, _int_arg(args, "id"), actor=actor, reason=str(args.get("reason") or "operator"))
    if verb == "retry-now":
        return webhook_pg_ops.retry_now(pg, _int_arg(args, "id"), actor=actor)
    if verb == "reprioritize":
        return webhook_pg_ops.reprioritize(pg, _int_arg(args, "id"), _int_arg(args, "priority"), actor=actor)
    if verb == "tenants":
        return {"tenants": webhook_pg_ops.tenants(pg)}
    if verb == "tenant":
        inst = _int_arg(args, "installation")
        action = str(args.get("action") or "show")
        if action == "show":
            return {"tenant": webhook_pg_ops.tenant(pg, inst) or {"installation_id": inst, "defaults": True}}
        if action == "pause":
            return {"tenant": webhook_pg_ops.tenant_set(pg, inst, actor=actor, paused=True)}
        if action == "resume":
            return {"tenant": webhook_pg_ops.tenant_set(pg, inst, actor=actor, paused=False)}
        if action == "cap":
            return {"tenant": webhook_pg_ops.tenant_set(pg, inst, actor=actor, max_concurrency=_int_arg(args, "value"))}
        if action == "weight":
            return {"tenant": webhook_pg_ops.tenant_set(pg, inst, actor=actor, weight=_int_arg(args, "value"))}
        if action == "dead-letter":
            value = args.get("value")
            return {"tenant": webhook_pg_ops.tenant_set(pg, inst, actor=actor,
                                                        dead_letter_conclusion=(None if value in (None, "", "default") else str(value)))}
        raise ValueError("tenant action must be show | pause | resume | cap | weight | dead-letter")
    if verb == "workers":
        return {"workers": webhook_pg_ops.workers(pg)}
    if verb in ("drain", "undrain"):
        owner = str(args.get("owner") or "")
        if not owner:
            raise ValueError("owner is required")
        row = webhook_pg_ops.set_draining(pg, owner, verb == "drain", actor=actor)
        if row is None:
            raise ValueError(f"worker {owner!r} not found")
        return {"worker": row}
    raise ValueError(f"unknown ops verb {verb!r}")


def _int_arg(args: Mapping[str, Any], key: str) -> int:
    value = args.get(key)
    if value in (None, ""):
        raise ValueError(f"{key} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


# --- the HTTP front (used by webhook_service._ServiceHandler) --------------------------------------
def handle_ops_get(path: str, headers: Mapping[str, str], *, queue: Any, actor_env: Mapping[str, str] | None = None,
                   scaler: Any = None, sweeper: Any = None,
                   local_gauges: Callable[[], dict[str, Any]] | None = None) -> tuple[int, dict[str, Any] | str]:
    """Route ``/ops/<verb>?k=v``. Returns (status, payload). Mutations are GET-with-query on purpose:
    the surface is bearer-authenticated, called by ``curl`` from the ops workflow, and every
    mutation is idempotent (cancelling a cancelled row is a no-op that says so)."""
    parts = urlsplit(path)
    verb = parts.path[len("/ops/"):].strip("/")
    if not ops_authorized(headers, actor_env):
        return 401, {"error": "unauthorized"}
    if verb not in VERBS:
        return 404, {"error": "unknown_verb", "verbs": list(VERBS)}
    args = {k: v[-1] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
    actor = f"ops-http:{args.pop('actor', '') or 'operator'}"
    try:
        out = run_verb(queue, verb, args, actor=actor, scaler=scaler, sweeper=sweeper, local_gauges=local_gauges)
    except ValueError as exc:
        return 400, {"error": "bad_request", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- never leak a traceback; the class name is safe
        return 503, {"error": "ops_failed", "type": type(exc).__name__}
    if verb == "metrics":
        return 200, str(out["text"])
    return 200, out


# --- the CLI front (webhook_cli.cmd_webhook dispatches here) ----------------------------------------
def _live_client() -> Any:
    import httpx

    return httpx.Client(timeout=5.0)


def live_verb(verb: str, args: Mapping[str, Any], *, actor: str,
              environ: Mapping[str, str] | None = None) -> tuple[dict[str, Any] | None, str]:
    """The RUNNING server's answer to ``verb``: its own loopback ``/ops/<verb>``, authenticated with
    the ``CODNA_WEBHOOK_OPS_TOKEN`` of this environment (the ingress machine holds it for its HTTP
    front). Returns ``(payload + {"view": "live"}, "")``, or ``(None, reason)`` when the call cannot
    be made -- no ops token here, nothing listening on the port, a non-200 -- where the reason names
    an exception class or a status code and never the token."""
    import httpx

    e = os.environ if environ is None else environ
    token = (e.get(OPS_TOKEN_ENV) or "").strip()
    if not token:
        return None, f"{OPS_TOKEN_ENV} is not set in this process's environment"
    port = (e.get(LIVE_PORT_ENV) or "8080").strip()
    params = {k: str(v) for k, v in args.items() if v not in (None, "")}
    params["actor"] = actor
    try:
        with _live_client() as client:
            resp = client.get(f"http://127.0.0.1:{port}/ops/{verb}", params=params,
                              headers={"Authorization": f"Bearer {token}"})
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        # InvalidURL is not an HTTPError: a CODNA_WEBHOOK_PORT that is not a port lands here too.
        return None, f"loopback /ops/{verb} unreachable: {type(exc).__name__}"
    if resp.status_code != 200:
        return None, f"loopback /ops/{verb} answered HTTP {resp.status_code}"
    try:
        payload = resp.json()
    except ValueError:
        return None, f"loopback /ops/{verb} answered a non-JSON body"
    if not isinstance(payload, dict):
        return None, f"loopback /ops/{verb} answered a non-object body"
    return {**payload, "view": "live"}, ""


def add_ops_parser(pw_sub: argparse._SubParsersAction) -> None:
    """``codna webhook ops <verb> [options]`` + ``migrate`` + ``queue export|import``."""
    po = pw_sub.add_parser("ops", help="Operate the hosted webhook queue (Postgres backend; needs DATABASE_URL).")
    po.add_argument("verb", choices=VERBS)
    po.add_argument("--id", type=int, help="job id (job, events, cancel, retry-now, reprioritize)")
    po.add_argument("--priority", type=int, help="new priority (reprioritize)")
    po.add_argument("--status", help="filter (jobs)")
    po.add_argument("--kind", help="filter (jobs)")
    po.add_argument("--repo", help="filter (jobs)")
    po.add_argument("--installation", type=int, help="installation id (jobs filter, tenant)")
    po.add_argument("--action", help="tenant: show|pause|resume|cap|weight|dead-letter; scaler: status|pause|resume")
    po.add_argument("--value", help="tenant cap/weight/dead-letter value")
    po.add_argument("--owner", help="worker owner (drain, undrain)")
    po.add_argument("--reason", default="operator", help="cancel reason")
    po.add_argument("--limit", type=int, default=50)
    po.add_argument("--actor", default=None, help="who is acting (defaults to $USER)")
    po.set_defaults(action_name="ops")
    pm = pw_sub.add_parser("migrate", help="Apply the Postgres queue schema (idempotent; the Fly release command).")
    pm.add_argument("--database-url", dest="database_url", default=None)
    pm.set_defaults(action_name="migrate")
    pq = pw_sub.add_parser("queue", help="Move in-flight jobs between the SQLite file and Postgres.")
    pq.add_argument("direction", choices=["export", "import"],
                    help="export: Postgres -> SQLite (rollback); import: SQLite -> Postgres (idempotent)")
    pq.add_argument("--database-url", dest="database_url", default=None)
    pq.set_defaults(action_name="queue")


def cli_ops(args: argparse.Namespace) -> int:
    from .webhook_pg_schema import database_url

    params = {k: getattr(args, k) for k in ("id", "priority", "status", "kind", "repo", "installation", "action",
                                              "value", "owner", "reason", "limit") if getattr(args, k, None) is not None}
    who = args.actor or os.environ.get("USER") or "operator"
    live_unavailable = ""
    if args.verb in LIVE_VERBS:
        live, live_unavailable = live_verb(args.verb, params, actor=who)
        if live is not None:
            print(json.dumps(live, indent=2, sort_keys=True, default=str))
            return 0
    url = database_url()
    if not url:
        print("codna: `codna webhook ops` needs DATABASE_URL (or CODNA_WEBHOOK_DATABASE_URL)", file=sys.stderr)
        return 2
    queue = PostgresQueue(url, heartbeat_s=0, pool_max=2)
    try:
        actor = f"ops-cli:{who}"
        out = run_verb(queue, args.verb, params, actor=actor)
        if args.verb in LIVE_VERBS:
            # This process built no Scaler, so its own answer says nothing about the serving one.
            out = {**out, "view": "cli-rebuild", "live_unavailable": live_unavailable,
                   "hint": "the serving process answers GET /ops/scaler on the ingress; this CLI process has no Scaler of its own"}
        if args.verb == "metrics":
            sys.stdout.write(str(out["text"]))
        else:
            print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0
    except ValueError as exc:
        print(f"codna: {exc}", file=sys.stderr)
        return 2
    finally:
        queue.close()


def cli_migrate(args: argparse.Namespace) -> int:
    from .webhook_pg_schema import DEFAULT_SCHEMA, connect, database_url, migrate

    url = getattr(args, "database_url", None) or database_url()
    if not url:
        print("codna: `codna webhook migrate` needs DATABASE_URL", file=sys.stderr)
        return 2
    with connect(url) as conn:
        version = migrate(conn, schema=os.environ.get("CODNA_WEBHOOK_PG_SCHEMA", DEFAULT_SCHEMA))
    print(json.dumps({"schema_version": version}))
    return 0


def cli_queue(args: argparse.Namespace) -> int:
    from .webhook import default_queue_path
    from .webhook_pg_schema import database_url
    from .webhook_queue import WebhookQueue

    url = getattr(args, "database_url", None) or database_url()
    if not url:
        print("codna: `codna webhook queue` needs DATABASE_URL", file=sys.stderr)
        return 2
    sqlite = WebhookQueue(default_queue_path())
    pg = PostgresQueue(url, heartbeat_s=0, pool_max=2)
    try:
        out = pg.import_sqlite_backlog(sqlite) if args.direction == "import" else pg.export_backlog(sqlite)
        print(json.dumps(out, sort_keys=True))
        return 0
    finally:
        pg.close()
