"""Guards on the Fly configs under infra/fly/: the invariants that keep the live merge gate up.

The routing health check of the ingress is ``/health`` (liveness) and nothing else -- on
2026-09-17 a ``/ready`` routing check unrouted the only volume-pinned machine the moment its
worker pool looked unhealthy and every GitHub delivery was refused at the edge. The worker pool
config has NO ``[http_service]`` at all (nothing to unroute), bounds its kill timeout at Fly's
maximum, stays stopped after a clean drain, and documents the cost ceiling the capacity model
implies. Expectations are read from the files, never hardcoded counts. The ingress ``[[vm]]`` is
sized for the ingress role plus ONE degraded-lane review thread, and that arithmetic is checked
against the pool's own constants (``webhook_control``), not against numbers copied into this file."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INGRESS = ROOT / "infra" / "fly" / "codna-webhook.fly.toml"
WORKER = ROOT / "infra" / "fly" / "codna-webhook-worker.fly.toml"
DEPLOY_WORKFLOW = ROOT / ".github" / "workflows" / "deploy-codna-webhook.yml"


@pytest.fixture(scope="module")
def ingress():
    return tomllib.loads(INGRESS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def worker():
    return tomllib.loads(WORKER.read_text(encoding="utf-8"))


def _seconds(value: str) -> int:
    m = re.fullmatch(r"(\d+)(s|m)", value.strip())
    assert m, f"unparseable duration {value!r}"
    return int(m.group(1)) * (60 if m.group(2) == "m" else 1)


def _memory_mb(value: str) -> int:
    m = re.fullmatch(r"(\d+)(mb|gb)", value.strip().lower())
    assert m, f"unparseable memory {value!r}"
    return int(m.group(1)) * (1024 if m.group(2) == "gb" else 1)


def test_both_configs_parse_and_name_distinct_apps(ingress, worker):
    assert ingress["app"] == "codna-webhook"
    assert worker["app"] == "codna-webhook-worker"
    assert ingress["build"]["dockerfile"] == worker["build"]["dockerfile"] == "../docker/Dockerfile.webhook"


def test_ingress_routing_health_check_is_health_and_nothing_else(ingress):
    checks = ingress["http_service"]["checks"]
    assert [c["path"] for c in checks] == ["/health"], (
        "the ONLY routing check on the ingress is /health (liveness). /ready is for the canary and the "
        "deploy verify; as a routing check it unrouted the single machine on 2026-09-17 and dropped deliveries"
    )
    assert all(c["method"] == "GET" for c in checks)
    # the incident is recorded next to the check so nobody re-adds /ready in good faith
    text = INGRESS.read_text(encoding="utf-8")
    assert "/ready" in text and "unroute" in text


def test_ingress_never_stops_its_last_machine(ingress):
    svc = ingress["http_service"]
    assert svc["auto_stop_machines"] == "off"
    assert svc["min_machines_running"] >= 1
    assert ingress["mounts"]["destination"] == "/data"
    assert ingress["env"]["CODNA_WEBHOOK_QUEUE"].startswith("/data/")  # the SQLite file lives on the volume
    # the queue backend is NOT flipped in the config: production stays on sqlite until the owner does
    assert "CODNA_WEBHOOK_QUEUE_BACKEND" not in ingress["env"]


def test_ingress_is_sized_for_its_role_plus_one_degraded_lane_thread(ingress):
    """Role ``ingress`` runs no worker pool (webhook_service.build_service builds one for ``all`` and
    ``worker`` only) and was measured at 130 MB resident; what sets its memory is the degraded lane
    on the ``role=all`` rollback -- one ``review`` thread, sized by the pool's own constants."""
    from codna import webhook_control

    [vm] = ingress["vm"]
    assert vm["size"] == "shared-cpu-1x", "the smallest shared shape; the service is one Python process"
    memory_mb = _memory_mb(vm["memory"])
    assert memory_mb == 2048, "shared-cpu-1x carries at most 2048 MB (fly platform vm-sizes)"
    reserved, per_job = webhook_control._RESERVED_MEMORY_MB, webhook_control._JOB_MEMORY_MB
    # one degraded-lane review thread fits without overcommit ...
    assert reserved + per_job <= memory_mb, f"{reserved} + {per_job} MB does not fit in {memory_mb} MB"
    # ... it IS one thread by the derivation the pool would run on this machine ...
    assert min(webhook_control._MAX_DERIVED_CONCURRENCY, (memory_mb - reserved) // per_job) == 1
    # ... and the next shape down could not hold it: the reason the machine is not smaller
    assert reserved + per_job > memory_mb // 2
    text = INGRESS.read_text(encoding="utf-8")
    for needle in ("DEGRADED LANE", "default_concurrency", "NO worker pool", "shared-cpu-4x / 8 GB",
                   "ingress-scale", "$10.69/mo", "Rollback:"):
        assert needle in text, f"the ingress toml must document {needle!r}"


def test_worker_pool_has_no_routing_surface(worker):
    assert "http_service" not in worker, "workers take no proxy traffic: there must be nothing to unroute"
    assert "services" not in worker
    checks = worker["checks"]
    assert [c["path"] for c in checks.values()] == ["/healthz"]
    assert worker["metrics"]["path"] == "/metrics" and worker["metrics"]["port"] == 8080


def test_worker_pool_runs_the_postgres_backend_as_a_worker(worker):
    env = worker["env"]
    assert env["CODNA_WEBHOOK_QUEUE_BACKEND"] == "postgres"
    assert env["CODNA_WEBHOOK_ROLE"] == "worker"
    assert env["TMPDIR"].startswith("/data/") and worker["mounts"]["destination"] == "/data"
    assert int(env["CODNA_WEBHOOK_CONCURRENCY"]) >= 1
    assert int(env["CODNA_WEBHOOK_LEASE_S"]) > int(env["CODNA_WEBHOOK_HEARTBEAT_S"]) * 2  # two missed beats never lose a lease


def test_worker_kill_timeout_is_at_flys_maximum_and_a_drained_worker_stays_stopped(worker):
    assert _seconds(worker["kill_timeout"]) == 300  # Fly's maximum: a longer fix must be DRAINED, never just stopped
    # The pool's own drain grace must fill that window (its code default, 25 s, is the INGRESS
    # machine's 30 s kill_timeout): measured 2026-09-20, a `fly machine stop` with the default
    # exited 25 s into a 120 s review and handed it to the reaper.
    drain_s = int(worker["env"]["CODNA_WEBHOOK_DRAIN_S"])
    assert 240 <= drain_s <= _seconds(worker["kill_timeout"]) - 5
    [restart] = worker["restart"]
    assert restart["policy"] == "on-failure"  # exit 0 after a drain = the machine stays stopped (the pool shrinks)


def test_worker_config_documents_the_scaling_signal_and_the_cost_ceiling():
    text = WORKER.read_text(encoding="utf-8")
    for needle in ("SIGNAL", "FORMULA", "desired = clamp(", "FLOOR = 2", "CAP", "NEVER creates machines",
                   "HARD CEILING", "CODNA_WEBHOOK_MAX_MACHINES", "scaler_at_ceiling",
                   "1x   (cap 4)", "10x  (cap 10)", "50x  (cap 11", "/mo"):
        assert needle in text, f"the worker toml must document {needle!r}"


def test_deploy_workflow_still_deploys_only_the_ingress_config():
    """Nothing deploys the worker pool until the owner flips it (Phase 3); the ingress deploy is unchanged."""
    text = DEPLOY_WORKFLOW.read_text(encoding="utf-8")
    assert "infra/fly/codna-webhook.fly.toml" in text
    assert "codna-webhook-worker" not in text
