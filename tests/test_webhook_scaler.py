"""The queue-driven autoscaler (codna.webhook_scaler): the desired-machines formula, scale-up from
a stopped pool in region order and in parallel, the ceiling alert, drain-then-stop scale-down
that never touches the floor or a busy machine, backoff on API failure, and leadership. The
Machines API is a fake with a start latency; the queue is real Postgres (CODNA_TEST_DATABASE_URL)."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from codna import webhook_pg_ops
from codna.webhook import WebhookJob
from codna.webhook_lease import WorkerRegistration
from codna.webhook_scaler import (
    FakeMachines,
    Machine,
    Scaler,
    ScalerConfig,
    desired_machines,
    machines_client_from_env,
)

pytestmark = pytest.mark.usefixtures("pg_url")


@pytest.mark.parametrize("started,runnable,free,expected", [
    (2, 0, 6, 2),      # idle: the floor
    (2, 6, 6, 2),      # backlog fits the free slots
    (2, 7, 6, 3),      # one more than fits: +1 machine
    (2, 12, 6, 4),     # 6 over, 3 slots each: +2
    (2, 100, 0, 4),    # capped
    (0, 0, 0, 2),      # never below the floor even if nothing is started
    (3, 0, 9, 2),      # three started, all idle: the floor is what the work needs (idle timers decide WHEN)
    (3, 1, 7, 2),      # two slots busy + one runnable = one machine's worth: still the floor
])
def test_desired_machines_formula(started, runnable, free, expected):
    assert desired_machines(started=started, runnable=runnable, free_slots=free, slots_per_machine=3, floor=2, cap=4) == expected


def test_config_from_env_clamps_cap_to_floor_and_parses_regions():
    cfg = ScalerConfig.from_env({"CODNA_WEBHOOK_FLOOR_MACHINES": "3", "CODNA_WEBHOOK_MAX_MACHINES": "2",
                                 "CODNA_WEBHOOK_REGIONS": "ord, iad"})
    assert (cfg.floor, cfg.cap, cfg.regions_order) == (3, 3, ("ord", "iad"))
    assert ScalerConfig.from_env({}).cap == 4 and ScalerConfig.from_env({}).floor == 2


def test_no_fly_token_means_no_machines_client():
    assert machines_client_from_env({}) is None
    client = machines_client_from_env({"FLY_API_TOKEN": "t", "CODNA_WEBHOOK_WORKER_APP": "w"})
    assert client is not None and client._app == "w"
    ours = machines_client_from_env({"CODNA_WEBHOOK_FLY_TOKEN": "ours", "FLY_API_TOKEN": "t"})
    assert ours is not None and ours._token == "ours"           # the service's own name wins


class _Held:
    held = True

    def try_acquire(self):
        return True

    def release(self):
        self.held = False


class _World:
    """A pool of machines, a sim clock shared by the queue, the fake API and the scaler, and the
    registered workers that make the queue's view of free slots match the started machines."""

    def __init__(self, make_pg_queue, *, cap=4, floor=2, latency=5.0, idle_s=300.0):
        self.now = 0.0
        self.wall = datetime.now(timezone.utc)
        self.queue = make_pg_queue(owner="ingress:1", clock=lambda: self.wall, tenant_default_cap=100)
        self.make = make_pg_queue
        regions = ["iad", "ord"] * (cap // 2 + 1)
        pool = [Machine(id=f"m{i}", state="started" if i < floor else "stopped", region=regions[i]) for i in range(cap)]
        self.api = FakeMachines(pool, start_latency_s=latency, clock=lambda: self.now)
        self.cfg = ScalerConfig(floor=floor, cap=cap, slots_per_machine=3, idle_s=idle_s, scale_up_age_s=5.0)
        self.scaler = Scaler(self.queue, self.api, self.cfg, leader=_Held(), clock=lambda: self.now, rng=random.Random(1))
        self.workers = {}
        for m in pool[:floor]:
            self.register(m.id)

    def register(self, machine_id, busy=0):
        q = self.make(owner=f"{machine_id}:1", clock=lambda: self.wall, tenant_default_cap=100)
        reg = WorkerRegistration(q, role="worker", slots=3, machine_id=machine_id, region="iad")
        reg.register()
        webhook_pg_ops.worker_heartbeat(q, busy=busy)
        self.workers[machine_id] = (q, reg)
        return q

    def beat_all(self, busy=None):
        for mid, (q, _) in self.workers.items():
            webhook_pg_ops.worker_heartbeat(q, busy=busy.get(mid, 0) if busy else 0)

    def tick(self, seconds=2.0):
        self.now += seconds
        self.wall += timedelta(seconds=seconds)
        self.beat_all()
        return self.scaler.tick()

    def enqueue(self, n, kind="review"):
        for i in range(n):
            base = int(self.now * 1000) + i
            self.queue.enqueue(WebhookJob(kind, "acme/app", ref=f"s{base}", pr_number=base, installation_id=1, reason="t"),
                               delivery_id=f"d{base}")

    def states(self):
        return {m.id: m.state for m in self.api.list()}


def test_scale_up_starts_stopped_machines_in_region_order_and_stops_at_the_cap(make_pg_queue):
    w = _World(make_pg_queue, cap=4, floor=2, latency=5.0)
    assert w.tick()["desired"] == 2                                    # idle: nothing happens
    w.enqueue(10)                                                       # 10 runnable, 6 free slots -> +2 machines
    out = w.tick()
    assert out["desired"] == 4 and out["started_ids"] == ["m2", "m3"]    # iad (m2) before ord (m3); parallel start
    assert w.states()["m2"] == "starting" and w.api.calls == [("start", "m2"), ("start", "m3")]
    w.tick(6.0)                                                          # start latency elapsed
    assert w.states()["m2"] == w.states()["m3"] == "started"
    w.enqueue(30)                                                        # far more than the pool can absorb
    out = w.tick()
    assert out["desired"] == 4 and out["started_ids"] == []              # nothing left to start: at the ceiling
    assert w.scaler.state.at_ceiling_since is not None and not w.scaler.at_ceiling()
    w.tick(601.0)
    assert w.scaler.at_ceiling() and w.scaler.gauges()["scaler_at_ceiling"] == 1
    assert w.scaler.gauges()["scaler_actual_machines"] == 4


def test_an_old_runnable_row_alone_triggers_one_extra_machine(make_pg_queue):
    w = _World(make_pg_queue, cap=4, floor=2)
    w.enqueue(1)
    assert w.tick()["started_ids"] == []                                  # fits in the free slots
    out = w.tick(6.0)                                                     # nobody claimed it for 6 s > scale_up_age_s
    assert out["started_ids"] == ["m2"], "a row waiting past the age threshold is a scale-up signal on its own"


def test_scale_down_drains_first_never_below_the_floor_never_a_busy_machine(make_pg_queue):
    w = _World(make_pg_queue, cap=4, floor=2, idle_s=300.0)
    for mid in ("m2", "m3"):
        w.api.machines[mid].state = "started"
        w.register(mid)
    assert w.tick()["desired"] == 2                                        # 4 started, nothing to do: 2 in excess
    w.tick(200.0)
    assert not [c for c in w.api.calls if c[0] == "stop"]                  # idle < idle_s: kept
    # m3 becomes busy; m2 stays idle past the threshold
    w.beat_all({"m3": 1})
    w.now += 101.0
    w.wall += timedelta(seconds=101.0)
    w.beat_all({"m3": 1})
    out = w.scaler.tick()
    assert out["stopped_ids"] == []                                        # first the DRAIN request, not a stop
    [m2] = [x for x in webhook_pg_ops.workers(w.queue) if x["machine_id"] == "m2"]
    assert m2["draining"] is True
    m3 = next(x for x in webhook_pg_ops.workers(w.queue) if x["machine_id"] == "m3")
    assert m3["draining"] is False                                         # busy: never asked
    w.beat_all({"m3": 1})                                                  # m2's heartbeat acknowledges the drain
    out = w.scaler.tick()
    assert out["stopped_ids"] == ["m2"] and w.states()["m2"] == "stopped"
    assert ("stop", "m2") in w.api.calls and ("stop", "m3") not in w.api.calls
    # floor machines are never candidates, whatever their idleness
    w.beat_all({"m3": 0})
    for _ in range(3):
        w.tick(400.0)
    assert w.states()["m0"] == w.states()["m1"] == "started"


def test_api_failures_back_off_and_count_but_never_raise(make_pg_queue):
    w = _World(make_pg_queue, cap=4, floor=2)
    w.enqueue(10)
    w.api.fail_next = 2
    out = w.tick()
    assert out["started_ids"] == [] and w.scaler.state.errors_total == 2
    assert w.tick()["backoff"] is True                                    # jittered pause after an error
    out = w.tick(11.0)                                                    # the pause is over: this tick recovers
    assert out["started_ids"] == ["m2", "m3"]
    assert w.scaler.gauges()["scaler_errors_total"] == 2


def test_pause_and_status(make_pg_queue):
    w = _World(make_pg_queue)
    w.scaler.pause(True)
    assert w.tick() == {"paused": True}
    status = w.scaler.status()
    assert status["paused"] is True and status["cap"] == 4 and status["floor"] == 2 and status["scaler_leader"] == 1
    w.scaler.pause(False)
    assert "desired" in w.tick()


def test_only_the_leader_ticks(make_pg_queue):
    import threading
    import time

    class _NotHeld:
        held = False

        def try_acquire(self):
            return False

        def release(self):
            return None

    w = _World(make_pg_queue)
    fast = Scaler(w.queue, w.api, ScalerConfig(floor=2, cap=4, tick_s=0.01), leader=_NotHeld(), clock=lambda: w.now)
    ticks = []
    fast.tick = lambda: ticks.append(1)
    thread = threading.Thread(target=fast._loop, daemon=True)
    thread.start()
    time.sleep(0.15)
    assert ticks == [] and fast.gauges()["scaler_leader"] == 0           # not the leader: it never decides
    fast._leader = _Held()
    time.sleep(0.15)
    fast.stop()
    thread.join(timeout=2)
    assert ticks, "the moment the lock is held, the loop ticks"
