"""Queue-driven autoscaler for the worker pool (``codna-webhook-worker`` on Fly).

The scaling SIGNAL is the queue itself, not CPU: ``runnable`` rows versus the live workers' free
slots, and the oldest runnable row's age. Both come from one SQL round
(:func:`webhook_pg_ops.scaler_inputs`). Every 2 s the leader (whoever holds the
``codna_webhook:scaler`` advisory lock -- an ingress machine) computes::

    desired = clamp(floor, started + ceil(max(0, runnable - free_slots) / slots_per_machine), cap)

and STARTS stopped machines from a pre-created pool when ``desired > started`` (or when the oldest
runnable row has waited more than ``scale_up_age_s``), several in parallel -- Fly's Machines API
rate limit is per machine for Start, so starting N distinct machines is not throttled by one shared
bucket. It never CREATES machines: the pool IS the hard cost ceiling (``CODNA_WEBHOOK_MAX_MACHINES``);
a leaked token can start what exists but never grow the fleet, and ``scaler_at_ceiling`` is the
alert when the pool is exhausted for 10 minutes.

Scale-down never kills work: a non-floor machine that has had ``busy = 0`` for ``idle_s`` is asked
to DRAIN (``workers.draining``); its heartbeat acknowledges, it stops claiming, and only once
``drain_acked_at`` is set and ``busy = 0`` still holds is the machine stopped (SIGTERM, the
worker's own ``kill_timeout`` governs). Floor machines are never stopped. Every Machines API call
has a 10 s timeout and jittered backoff; failures count in ``scaler_errors_total`` and the floor
keeps serving through any scaler failure. Split leadership is idempotent: starting a started
machine is a no-op.

``FlyMachinesClient`` is the only thing that talks to Fly; tests pass :class:`FakeMachines`.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from . import webhook_pg_ops
from .webhook_pg_queue import PostgresQueue

MACHINES_API = "https://api.machines.dev/v1"
STARTED_STATES = frozenset({"started", "starting", "replacing"})
STOPPED_STATES = frozenset({"stopped", "suspended", "created"})


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook-scaler", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ScalerConfig:
    app: str = "codna-webhook-worker"
    floor: int = 2
    cap: int = 4
    slots_per_machine: int = 3
    idle_s: float = 300.0
    scale_up_age_s: float = 5.0
    tick_s: float = 2.0
    at_ceiling_alert_s: float = 600.0
    regions_order: tuple[str, ...] = ("iad", "ord")

    @classmethod
    def from_env(cls, environ: Any = None) -> "ScalerConfig":
        e = os.environ if environ is None else environ
        regions = tuple(r.strip() for r in (e.get("CODNA_WEBHOOK_REGIONS") or "iad,ord").split(",") if r.strip())
        floor = max(1, int(e.get("CODNA_WEBHOOK_FLOOR_MACHINES", "2")))
        cap = max(floor, int(e.get("CODNA_WEBHOOK_MAX_MACHINES", "4")))
        return cls(
            app=e.get("CODNA_WEBHOOK_WORKER_APP", "codna-webhook-worker"), floor=floor, cap=cap,
            slots_per_machine=max(1, int(e.get("CODNA_WEBHOOK_SLOTS_PER_MACHINE", "3"))),
            idle_s=float(e.get("CODNA_WEBHOOK_SCALE_IDLE_S", "300")),
            scale_up_age_s=float(e.get("CODNA_WEBHOOK_SCALE_UP_AGE_S", "5")),
            tick_s=float(e.get("CODNA_WEBHOOK_SCALER_TICK_S", "2")),
            regions_order=regions or ("iad",),
        )


@dataclass
class Machine:
    id: str
    state: str
    region: str
    name: str = ""


class MachinesAPI(Protocol):
    def list(self) -> list[Machine]: ...
    def start(self, machine_id: str) -> None: ...
    def stop(self, machine_id: str, *, timeout_s: int = 30) -> None: ...


class FlyMachinesClient:
    """The Machines REST API with an app-scoped deploy token (``fly tokens create deploy -a <app>``)."""

    def __init__(self, app: str, token: str, *, base_url: str = MACHINES_API, timeout_s: float = 10.0) -> None:
        self._app = app
        self._token = token
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout_s)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        import httpx

        resp = httpx.request(method, f"{self._base}/apps/{self._app}{path}", headers=self._headers(),
                             json=body, timeout=self._timeout)
        if resp.status_code >= 300:
            raise RuntimeError(f"machines api {method} {path}: {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError:
            return None

    def list(self) -> list[Machine]:
        rows = self._request("GET", "/machines") or []
        return [Machine(id=str(r.get("id")), state=str(r.get("state") or ""), region=str(r.get("region") or ""),
                        name=str(r.get("name") or "")) for r in rows if isinstance(r, dict)]

    def start(self, machine_id: str) -> None:
        self._request("POST", f"/machines/{machine_id}/start")

    def stop(self, machine_id: str, *, timeout_s: int = 30) -> None:
        self._request("POST", f"/machines/{machine_id}/stop", {"signal": "SIGTERM", "timeout": f"{int(timeout_s)}s"})


class FakeMachines:
    """A pool of machines with a start latency, for tests and the trace replay."""

    def __init__(self, machines: list[Machine], *, start_latency_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.machines = {m.id: m for m in machines}
        self._latency = float(start_latency_s)
        self._clock = clock
        self._pending: dict[str, float] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_next: int = 0

    def _settle(self) -> None:
        now = self._clock()
        for mid, ready_at in list(self._pending.items()):
            if now >= ready_at:
                self.machines[mid].state = "started"
                del self._pending[mid]

    def list(self) -> list[Machine]:
        self._settle()
        return [Machine(m.id, m.state, m.region, m.name) for m in self.machines.values()]

    def start(self, machine_id: str) -> None:
        self.calls.append(("start", machine_id))
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("machines api: 503")
        m = self.machines[machine_id]
        if m.state in STARTED_STATES:
            return
        m.state = "starting"
        self._pending[machine_id] = self._clock() + self._latency

    def stop(self, machine_id: str, *, timeout_s: int = 30) -> None:
        self.calls.append(("stop", machine_id))
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("machines api: 503")
        self.machines[machine_id].state = "stopped"
        self._pending.pop(machine_id, None)


def desired_machines(*, started: int, runnable: int, free_slots: int, slots_per_machine: int, floor: int, cap: int) -> int:
    """How many machines the CURRENT work needs: the slots in use plus the runnable backlog, in
    machines, clamped to [floor, cap]. Above ``started`` it is the scale-up formula from the module
    docstring (``started + ceil((runnable - free_slots) / slots)``, the two agree whenever there is
    a backlog); below ``started`` it is how far the pool may shrink -- the idle timers in
    :meth:`Scaler.tick` decide WHEN."""
    slots = max(1, int(slots_per_machine))
    busy = max(0, int(started) * slots - int(free_slots))
    want = int(math.ceil((busy + max(0, int(runnable))) / slots))
    return max(int(floor), min(int(cap), want))


@dataclass
class ScalerState:
    desired: int = 0
    actual: int = 0
    at_ceiling_since: float | None = None
    idle_since: dict[str, float] = field(default_factory=dict)
    errors_total: int = 0
    last_tick_at: float | None = None
    paused: bool = False


class Scaler:
    """See the module docstring. ``leader`` is duck-typed (``try_acquire()``/``release()``)."""

    def __init__(self, queue: PostgresQueue, machines: MachinesAPI, cfg: ScalerConfig, *,
                 leader: Any = None, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self._queue = queue
        self._machines = machines
        self.cfg = cfg
        self._leader = leader if leader is not None else webhook_pg_ops.LeaderLock(queue, "scaler")
        self._clock = clock
        self._rng = rng or random.Random()
        self.state = ScalerState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backoff_until = 0.0

    # -- one decision ----------------------------------------------------------------------------------
    def tick(self) -> dict[str, Any]:
        now = self._clock()
        st = self.state
        st.last_tick_at = now
        if st.paused:
            return {"paused": True}
        if now < self._backoff_until:
            return {"backoff": True}
        try:
            inputs = webhook_pg_ops.scaler_inputs(self._queue)
            pool = self._machines.list()
        except Exception as exc:  # noqa: BLE001 -- the floor keeps serving; try again after a jittered pause
            return self._fail("inputs", exc)
        started = [m for m in pool if m.state in STARTED_STATES]
        stopped = [m for m in pool if m.state in STOPPED_STATES]
        st.actual = len(started)
        runnable, free = inputs["runnable"], inputs["free_slots"]
        desired = desired_machines(started=len(started), runnable=runnable, free_slots=free,
                                   slots_per_machine=self.cfg.slots_per_machine, floor=self.cfg.floor, cap=self.cfg.cap)
        if runnable > 0 and inputs["oldest_runnable_age_s"] > self.cfg.scale_up_age_s:
            # Something runnable has waited past the threshold although slots may look free (a
            # worker not claiming, a boot in progress): one more machine, whatever the arithmetic says.
            desired = max(desired, min(self.cfg.cap, len(started) + 1))
        st.desired = desired
        out: dict[str, Any] = {"runnable": runnable, "free_slots": free, "started": len(started),
                               "desired": desired, "started_ids": [], "stopped_ids": []}
        short = False
        if desired > len(started):
            need = desired - len(started)
            candidates = self._ordered(stopped)[:need]
            short = len(candidates) < need
            out["started_ids"] = self._start_parallel(candidates)
        # At the ceiling: work is waiting beyond the free slots and the pool has nothing more to start.
        if short or (runnable > free and len(started) >= self.cfg.cap):
            if st.at_ceiling_since is None:
                st.at_ceiling_since = now
        else:
            st.at_ceiling_since = None
        if desired < len(started):
            out["stopped_ids"] = self._scale_down(started, inputs, now, len(started) - desired)
        else:
            self._forget_idle(started, inputs, now)
        return out

    def _ordered(self, machines: list[Machine]) -> list[Machine]:
        rank = {r: i for i, r in enumerate(self.cfg.regions_order)}
        return sorted(machines, key=lambda m: (rank.get(m.region, len(rank)), m.id))

    def _floor_set(self, started: list[Machine]) -> set[str]:
        """The ``floor`` machines that are never stopped: one per region in region order, round
        robin (iad, ord, iad, ...) -- so the floor is spread across regions, which is the whole
        point of having one (a single machine or host loss never stops reviews)."""
        by_region: dict[str, list[Machine]] = {}
        for m in self._ordered(started):
            by_region.setdefault(m.region, []).append(m)
        regions = [r for r in self.cfg.regions_order if r in by_region] + sorted(r for r in by_region if r not in self.cfg.regions_order)
        chosen: set[str] = set()
        while len(chosen) < self.cfg.floor and any(by_region.values()):
            for region in regions:
                if by_region[region] and len(chosen) < self.cfg.floor:
                    chosen.add(by_region[region].pop(0).id)
        return chosen

    def _start_parallel(self, machines: list[Machine]) -> list[str]:
        if not machines:
            return []
        started: list[str] = []
        with ThreadPoolExecutor(max_workers=min(8, len(machines))) as pool:
            futures = {pool.submit(self._machines.start, m.id): m.id for m in machines}
            for fut, mid in futures.items():
                try:
                    fut.result()
                    started.append(mid)
                    _log("machine_start", machine_id=mid)
                except Exception as exc:  # noqa: BLE001
                    self._fail("start", exc, machine_id=mid)
        return started

    def _worker_rows(self, inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(w.get("machine_id")): w for w in inputs["workers"] if w.get("machine_id")}

    def _scale_down(self, started: list[Machine], inputs: dict[str, Any], now: float, excess: int) -> list[str]:
        """Drain-then-stop, one machine per tick at most ``excess``; never below the floor, never a
        machine with running jobs, never one whose worker has not acknowledged the drain."""
        stopped: list[str] = []
        by_machine = self._worker_rows(inputs)
        protected = self._floor_set(started)
        for m in reversed(self._ordered(started)):
            if len(stopped) >= excess or m.id in protected:
                continue
            row = by_machine.get(m.id)
            busy = int(row.get("busy", 0)) if row else 0
            if busy > 0:
                self.state.idle_since.pop(m.id, None)
                continue
            since = self.state.idle_since.setdefault(m.id, now)
            if now - since < self.cfg.idle_s:
                continue
            if row is None:
                # No worker ever registered from this machine (boot failed?): stopping it loses nothing.
                if self._stop_machine(m.id):
                    stopped.append(m.id)
                continue
            if not row.get("draining"):
                try:
                    webhook_pg_ops.set_draining(self._queue, str(row["owner"]), True, actor="scaler")
                    _log("machine_drain_requested", machine_id=m.id, owner=row["owner"])
                except Exception as exc:  # noqa: BLE001
                    self._fail("drain", exc, machine_id=m.id)
                continue
            if row.get("drain_acked_at") and busy == 0:
                if self._stop_machine(m.id):
                    stopped.append(m.id)
                    self.state.idle_since.pop(m.id, None)
        return stopped

    def _forget_idle(self, started: list[Machine], inputs: dict[str, Any], now: float) -> None:
        by_machine = self._worker_rows(inputs)
        for m in started:
            row = by_machine.get(m.id)
            if row and int(row.get("busy", 0)) > 0:
                self.state.idle_since.pop(m.id, None)

    def _stop_machine(self, machine_id: str) -> bool:
        try:
            self._machines.stop(machine_id, timeout_s=30)
            _log("machine_stop", machine_id=machine_id)
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail("stop", exc, machine_id=machine_id)
            return False

    def _fail(self, what: str, exc: BaseException, **fields: Any) -> dict[str, Any]:
        self.state.errors_total += 1
        self._backoff_until = self._clock() + self._rng.uniform(2.0, 10.0)
        _log("scaler_error", op=what, error=f"{type(exc).__name__}: {str(exc)[:200]}", **fields)
        return {"error": what}

    # -- gauges + loop ----------------------------------------------------------------------------------
    def at_ceiling(self) -> bool:
        since = self.state.at_ceiling_since
        return since is not None and (self._clock() - since) >= self.cfg.at_ceiling_alert_s

    def gauges(self) -> dict[str, Any]:
        return {"scaler_desired_machines": self.state.desired, "scaler_actual_machines": self.state.actual,
                "scaler_at_ceiling": 1 if self.at_ceiling() else 0, "scaler_errors_total": self.state.errors_total,
                "scaler_leader": 1 if getattr(self._leader, "held", False) else 0}

    def status(self) -> dict[str, Any]:
        return {**self.gauges(), "paused": self.state.paused, "cap": self.cfg.cap, "floor": self.cfg.floor,
                "slots_per_machine": self.cfg.slots_per_machine, "idle_s": self.cfg.idle_s,
                "at_ceiling_since": self.state.at_ceiling_since, "last_tick_at": self.state.last_tick_at}

    def pause(self, flag: bool) -> None:
        self.state.paused = bool(flag)
        _log("scaler_paused" if flag else "scaler_resumed")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="codna-webhook-scaler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.cfg.tick_s):
            try:
                if not self._leader.try_acquire():
                    continue
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self._fail("tick", exc)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._leader.release()
        except Exception:  # noqa: BLE001
            pass


# The autoscaler's Machines API credential: a deploy token scoped to the WORKER app (`fly tokens
# create deploy -a codna-webhook-worker`), set as a secret on the ingress. The service's own name
# is read first; FLY_API_TOKEN -- flyctl's login variable and the name Fly's Machines API docs use
# -- stays a fallback so an ingress configured under that name keeps its scaler.
FLY_TOKEN_ENVS = ("CODNA_WEBHOOK_FLY_TOKEN", "FLY_API_TOKEN")


def fly_token_from_env(environ: Any = None) -> str | None:
    """The first non-blank value among :data:`FLY_TOKEN_ENVS`, or None when neither is set."""
    e = os.environ if environ is None else environ
    for name in FLY_TOKEN_ENVS:
        value = (e.get(name) or "").strip()
        if value:
            return value
    return None


def machines_client_from_env(environ: Any = None) -> FlyMachinesClient | None:
    """None when no Fly token is configured under either name in :data:`FLY_TOKEN_ENVS` -- the
    scaler then does not run at all, and the floor machines serve alone (exactly today's shape)."""
    e = os.environ if environ is None else environ
    token = fly_token_from_env(e)
    if not token:
        return None
    return FlyMachinesClient(e.get("CODNA_WEBHOOK_WORKER_APP", "codna-webhook-worker"), token,
                             base_url=e.get("FLY_MACHINES_API", MACHINES_API))
