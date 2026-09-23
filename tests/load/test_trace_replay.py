"""Load proof for the SLO claim: replay the real 2026-09-17..20 arrival trace, superposed
``CODNA_TRACE_REPLAY_SCALE`` times (default 3; the pg CI job and the nightly run it at 10), through
the REAL PostgresQueue (claim ordering, tenant caps, leases) and the REAL Scaler driving a fake
Machines API with start latency ``CODNA_TRACE_REPLAY_L`` seconds (default 10, the Phase-0 spike's
budget), as a discrete-event simulation on an injected clock -- so 73 hours of traffic replay in
well under a minute and nothing sleeps.

Asserted, with every expectation derived from the trace and the config rather than typed in:
every job is claimed exactly once and finishes; the review queue wait's p95 is under the 60 s SLO
(the check-creation phase is not simulated: the SLO clock stops at the claim here); a fix waits
less than 600 s at p95; the scaler never exceeds the cap and returns to the floor once the trace
ends; and the database's own measurement of the waits (what /metrics and /slo read) agrees with
the simulation's.

The fixture ``cli/tests/fixtures/webhook_trace_7d.tsv`` was derived from the App's completed check
runs by model3.py's rules: arrival = the check suite's creation (the push) when it precedes the run
by < 6 h and is the first run for that (sha, kind), else the run's start; service = the run's wall
time. Kinds: review, fix, secure. Tenants are the repository hashed into seven installations;
superposed copies become new installations (the "10 tenants" premise of the model)."""
from __future__ import annotations

import heapq
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from codna import webhook_pg_ops
from codna.webhook import WebhookJob
from codna.webhook_scaler import STARTED_STATES, FakeMachines, Machine, Scaler, ScalerConfig

pytestmark = pytest.mark.usefixtures("pg_url")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "webhook_trace_7d.tsv"
SCALE = int(os.environ.get("CODNA_TRACE_REPLAY_SCALE", "3"))
MODE = os.environ.get("CODNA_TRACE_REPLAY_MODE", "bizhours")
START_LATENCY_S = float(os.environ.get("CODNA_TRACE_REPLAY_L", "10"))
SLOTS = 3
FLOOR = 2
CAP = 4 if SCALE <= 5 else 10           # model3.py: 10x bizhours holds the SLO at cap 4, 10x sync needs cap 10
IDLE_S = 300.0
SCALER_TICK_S = 10.0                    # the real scaler ticks every 2 s; coarser here only to bound the DB round trips
EPOCH = datetime(2026, 9, 17, tzinfo=timezone.utc)
SLO_S = 60.0
FIX_P95_S = 600.0


def _trace() -> list[tuple[str, float, float, int]]:
    rows = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        kind, t, s, tenant = line.split("\t")
        rows.append((kind, float(t), float(s), int(tenant)))
    rows.sort(key=lambda r: r[1])
    return rows


def _superposed(base, scale: int, mode: str, rng: random.Random) -> list[tuple[str, float, float, int]]:
    """model3.py's ``scaled``: copy k is the trace shifted (business-hours: +-4 h; synchronized: +-2 min)
    and wrapped over the span, with service times re-sampled from the kind's own distribution."""
    span = base[-1][1] - base[0][1]
    by_kind: dict[str, list[float]] = {}
    for kind, _t, s, _ in base:
        by_kind.setdefault(kind, []).append(s)
    out = []
    for k in range(scale):
        shift = 0.0 if k == 0 else (rng.uniform(-120, 120) if mode == "sync" else rng.uniform(-4 * 3600, 4 * 3600))
        for kind, t, s, tenant in base:
            out.append((kind, (t + shift) % span, s if k == 0 else rng.choice(by_kind[kind]), tenant + 10 * k))
    out.sort(key=lambda r: r[1])
    return out


def _pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))] if ordered else 0.0


class _Sim:
    def __init__(self, make_pg_queue, pg_url, pg_schema):
        self.now = 0.0
        self.make = make_pg_queue
        self.ingress = make_pg_queue(owner="ingress:1", clock=self.wall, tenant_default_cap=3)
        pool = [Machine(id=f"m{i}", state="started" if i < FLOOR else "stopped", region=("iad", "ord")[i % 2])
                for i in range(CAP)]
        self.api = FakeMachines(pool, start_latency_s=START_LATENCY_S, clock=lambda: self.now)
        self.scaler = Scaler(self.ingress, self.api, ScalerConfig(floor=FLOOR, cap=CAP, slots_per_machine=SLOTS, idle_s=IDLE_S,
                                                                   scale_up_age_s=5.0, tick_s=SCALER_TICK_S),
                             leader=_Held(), clock=lambda: self.now, rng=random.Random(3))
        self.machines: dict[str, dict] = {}       # id -> {"queue", "busy": {row_id: done_at}, "draining"}
        self.completions: list[tuple[float, str, int]] = []
        self.waits: dict[str, list[float]] = {}
        self.claimed: dict[int, str] = {}
        self.arrivals: dict[str, float] = {}
        self.max_started = 0
        self.waiting = 0

    def wall(self):
        return EPOCH + timedelta(seconds=self.now)

    def _ensure_registered(self, machine_id: str):
        if machine_id in self.machines:
            return
        q = self.make(owner=f"{machine_id}:1", clock=self.wall, tenant_default_cap=3)
        webhook_pg_ops.register_worker(q, machine_id=machine_id, region=self.api.machines[machine_id].region,
                                       image="img", role="worker", slots=SLOTS)
        self.machines[machine_id] = {"queue": q, "busy": {}, "draining": False}

    def heartbeat(self):
        started = {m.id for m in self.api.list() if m.state in STARTED_STATES}
        self.max_started = max(self.max_started, len(started))
        for mid in list(self.machines):
            if mid not in started:
                webhook_pg_ops.deregister_worker(self.machines[mid]["queue"])
                self.machines[mid]["queue"].close()
                del self.machines[mid]
        for mid in started:
            if self.api.machines[mid].state != "started":
                continue
            self._ensure_registered(mid)
            row = webhook_pg_ops.worker_heartbeat(self.machines[mid]["queue"], busy=len(self.machines[mid]["busy"]))
            self.machines[mid]["draining"] = bool(row and row.get("draining"))

    def claim_where_possible(self):
        for mid, m in self.machines.items():
            if m["draining"] or self.api.machines[mid].state != "started":
                continue
            while len(m["busy"]) < SLOTS and self.waiting > 0:
                claimed = m["queue"].claim()
                if claimed is None:
                    return  # tenant caps or nothing due: stop trying this round
                assert claimed.row_id not in self.claimed, "double claim"
                self.claimed[claimed.row_id] = claimed.delivery_id
                self.waiting -= 1
                self.waits.setdefault(claimed.job.kind, []).append(self.now - self.arrivals[claimed.delivery_id])
                done_at = self.now + self.service[claimed.delivery_id]
                m["busy"][claimed.row_id] = done_at
                heapq.heappush(self.completions, (done_at, mid, claimed.row_id))

    def run(self, jobs):
        self.service = {f"t{i}": s for i, (_, _, s, _) in enumerate(jobs)}
        pending = [(t, i) for i, (_, t, _, _) in enumerate(jobs)]
        heapq.heapify(pending)
        next_tick = 0.0
        while pending or self.completions or self.waiting or any(m["busy"] for m in self.machines.values()):
            candidates = [next_tick]
            if pending:
                candidates.append(pending[0][0])
            if self.completions:
                candidates.append(self.completions[0][0])
            self.now = max(self.now, min(candidates))
            while pending and pending[0][0] <= self.now:
                _, i = heapq.heappop(pending)
                kind, t, s, tenant = jobs[i]
                delivery = f"t{i}"
                job = WebhookJob(kind, f"trace/repo{tenant}", ref=f"sha{i}", pr_number=i, installation_id=tenant, reason="trace")
                assert self.ingress.enqueue(job, delivery_id=delivery)
                self.arrivals[delivery] = t
                self.waiting += 1
            while self.completions and self.completions[0][0] <= self.now:
                _, mid, row_id = heapq.heappop(self.completions)
                m = self.machines.get(mid)
                if m is not None:
                    m["queue"].complete(row_id, status="done", result={"summary": "trace"})
                    m["busy"].pop(row_id, None)
            if self.now >= next_tick:
                self.heartbeat()
                self.scaler.tick()
                next_tick = self.now + SCALER_TICK_S
            self.claim_where_possible()
            if not pending and not self.completions and not self.waiting:
                # let the scaler wind the burst machines down (idle_s, then drain, then stop)
                self.now += SCALER_TICK_S
                self.heartbeat()
                self.scaler.tick()
                if self.scaler.state.actual <= FLOOR and self.now - max(self.arrivals.values(), default=0) > IDLE_S * 3:
                    break
        for m in self.machines.values():
            m["queue"].close()


class _Held:
    held = True

    def try_acquire(self):
        return True

    def release(self):
        return None


def test_trace_replay_holds_the_review_slo_with_the_autoscaler(make_pg_queue, pg_url, pg_schema):
    base = _trace()
    jobs = _superposed(base, SCALE, MODE, random.Random(11))
    sim = _Sim(make_pg_queue, pg_url, pg_schema)
    sim.run(jobs)

    # 1. exactly once, nothing lost
    assert len(sim.claimed) == len(jobs)
    assert sim.ingress.counts() == {"done": len(jobs)}

    # 2. the SLO and the fix bound (expectations from the config, not typed numbers)
    review_p95 = _pct(sim.waits["review"], 0.95)
    fix_p95 = _pct(sim.waits.get("fix", []), 0.95)
    print(f"\ntrace replay x{SCALE} {MODE} L={START_LATENCY_S}s cap={CAP}: jobs={len(jobs)} review p50={_pct(sim.waits['review'], .5):.1f}s "
          f"p95={review_p95:.1f}s p99={_pct(sim.waits['review'], .99):.1f}s max={max(sim.waits['review']):.1f}s | fix p95={fix_p95:.1f}s "
          f"| max machines={sim.max_started}")
    assert review_p95 <= SLO_S, f"review p95 wait {review_p95:.1f}s breaches the {SLO_S:.0f}s SLO"
    assert fix_p95 < FIX_P95_S

    # 3. the scaler stayed inside the pool and came back down
    assert sim.max_started <= CAP
    assert sim.scaler.state.actual <= FLOOR
    assert sim.scaler.state.errors_total == 0

    # 4. the database measures the same waits the simulation saw (what /metrics + /slo read)
    with sim.ingress.connection() as conn:
        row = conn.execute(
            f"SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (claimed_at - created_at))) AS p95, "
            f"count(*) AS n FROM {pg_schema}.jobs WHERE kind = 'review'").fetchone()
    assert row["n"] == len(sim.waits["review"])
    assert abs(float(row["p95"]) - review_p95) <= 1.0
    assert math.isfinite(review_p95)
