"""Prometheus text exposition for the webhook (``/metrics``), stdlib only.

Fly's Prometheus scrapes the path named in the app's ``[metrics]`` block every 15 s; nothing here
depends on a client library. Series (all prefixed ``codna_webhook_``):

  queue_depth{kind,installation}       runnable rows right now
  oldest_runnable_age_seconds{kind}    how long the oldest runnable row has waited
  running{kind}                        rows a worker holds a lease on
  jobs_total{kind,installation,status} rows finished in the window, plus everything in flight
  job_wait_seconds_bucket{kind,le}     created -> claimed (the queue wait), rolling window
  job_duration_seconds_bucket{kind,le} claimed -> finished, rolling window
  retries_total{}, interrupted_total{}, cancelled_total{}   events in the window
  dead_total                           rows dead-lettered, all time (an increase is the alert)
  workers{state}                       live | draining | stale; workers_slots, workers_busy
  pg_up                                1 when the last snapshot succeeded
  scaler_desired_machines, scaler_actual_machines, scaler_at_ceiling, scaler_errors_total
  spool_rows, degraded_lane            the ingress's fallback state
  backend_info{backend,role}           1

Labels are bounded: installation is one of the top-N by queue depth, ``other`` or ``none``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

PREFIX = "codna_webhook_"


def _esc(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**labels: Any) -> str:
    inner = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items() if v is not None)
    return f"{{{inner}}}" if inner else ""


def _line(name: str, value: Any, **labels: Any) -> str:
    v = value
    if isinstance(v, bool):
        v = 1 if v else 0
    return f"{PREFIX}{name}{_labels(**labels)} {v}"


def _histogram(name: str, kind: str, hist: Mapping[str, Any]) -> Iterable[str]:
    for edge, count in zip(hist["buckets"], hist["counts"]):
        yield _line(f"{name}_bucket", count, kind=kind, le=f"{edge:g}")
    yield _line(f"{name}_bucket", hist["count"], kind=kind, le="+Inf")
    yield _line(f"{name}_count", hist["count"], kind=kind)
    yield _line(f"{name}_sum", round(float(hist["sum"]), 3), kind=kind)


def render(snapshot: Mapping[str, Any] | None, *, local: Mapping[str, Any] | None = None) -> str:
    """``snapshot`` is :func:`webhook_pg_ops.metrics_snapshot` (None when Postgres is unreachable:
    ``pg_up 0`` and the local gauges are still emitted). ``local`` carries process-local gauges:
    backend, role, scaler_*, spool_rows, degraded_lane, shadow_*."""
    out: list[str] = []
    loc = dict(local or {})
    out.append(f"# TYPE {PREFIX}pg_up gauge")
    out.append(_line("pg_up", 1 if snapshot else 0))
    if snapshot:
        out.append(f"# TYPE {PREFIX}queue_depth gauge")
        for (kind, inst), n in sorted(snapshot.get("queue_depth", {}).items()):
            out.append(_line("queue_depth", n, kind=kind, installation=inst))
        out.append(f"# TYPE {PREFIX}oldest_runnable_age_seconds gauge")
        for kind, age in sorted(snapshot.get("oldest_runnable_age_s", {}).items()):
            out.append(_line("oldest_runnable_age_seconds", round(float(age), 1), kind=kind))
        out.append(f"# TYPE {PREFIX}running gauge")
        for kind, n in sorted(snapshot.get("running", {}).items()):
            out.append(_line("running", n, kind=kind))
        out.append(f"# TYPE {PREFIX}jobs_total gauge")
        for (kind, inst, status), n in sorted(snapshot.get("jobs_total", {}).items()):
            out.append(_line("jobs_total", n, kind=kind, installation=inst, status=status))
        out.append(f"# TYPE {PREFIX}job_wait_seconds histogram")
        for kind, hist in sorted(snapshot.get("job_wait_s", {}).items()):
            out.extend(_histogram("job_wait_seconds", kind, hist))
        out.append(f"# TYPE {PREFIX}job_duration_seconds histogram")
        for kind, hist in sorted(snapshot.get("job_duration_s", {}).items()):
            out.extend(_histogram("job_duration_seconds", kind, hist))
        events = snapshot.get("events_window", {})
        out.append(f"# TYPE {PREFIX}retries_total gauge")
        out.append(_line("retries_total", events.get("retry", 0)))
        out.append(f"# TYPE {PREFIX}interrupted_total gauge")
        out.append(_line("interrupted_total", events.get("interrupted", 0)))
        out.append(f"# TYPE {PREFIX}cancelled_total gauge")
        out.append(_line("cancelled_total", events.get("cancelled", 0)))
        out.append(f"# TYPE {PREFIX}dead_total gauge")
        out.append(_line("dead_total", snapshot.get("dead_total", 0)))
        workers = snapshot.get("workers", {})
        out.append(f"# TYPE {PREFIX}workers gauge")
        for state in ("live", "draining", "stale"):
            out.append(_line("workers", workers.get(state, 0), state=state))
        out.append(f"# TYPE {PREFIX}workers_slots gauge")
        out.append(_line("workers_slots", workers.get("slots", 0)))
        out.append(f"# TYPE {PREFIX}workers_busy gauge")
        out.append(_line("workers_busy", workers.get("busy", 0)))
    for name in ("scaler_desired_machines", "scaler_actual_machines", "scaler_at_ceiling", "scaler_errors_total",
                 "scaler_leader", "reaper_leader", "spool_rows", "degraded_lane", "shadow_errors_total",
                 "shadow_writes_total", "pool_busy_threads", "pool_alive_threads"):
        if name in loc:
            out.append(f"# TYPE {PREFIX}{name} gauge")
            out.append(_line(name, loc[name]))
    out.append(f"# TYPE {PREFIX}backend_info gauge")
    out.append(_line("backend_info", 1, backend=loc.get("backend", "sqlite"), role=loc.get("role", "all")))
    return "\n".join(out) + "\n"
