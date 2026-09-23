"""MC-predictive admission control for parallel Codna fixes.

A fix's wall-clock is dominated by the agent + model calls; running too many at once makes every
in-flight fix share the model's throughput, so the slow tail blows past the SLA (the rust/clap
>900s timeouts we measured). A static concurrency cap can't see that — it admits a fix that will
miss the SLA, or it idles capacity that's actually free.

This admitter governs concurrency *predictively*:

  1. Deterministic floor: admit freely up to ``det_cap = min(ceiling, model_rate_budget)`` — clearly
     safe, no prediction needed.
  2. Hard ceiling: never exceed ``ceiling`` (the sidecar's AGENT_CORE_MAX_CONCURRENCY).
  3. MC zone (between det_cap and ceiling): run a Monte-Carlo over the observed fix-duration
     distribution under the contention the new fix would create, and admit only while
     ``P(new fix breaches the SLA) < breach_threshold``.

This is Algenta's own Monte-Carlo discipline applied to its own concurrency. The breach predictor is
**pluggable** (``mc_fn``), with two strategies — both fully LOCAL, ZERO network round-trip:

  • **mojo kernel (default)** — Algenta's compiled ``monte_carlo`` kernel makes the decision, reached
    through the LOCAL SDK exactly like every fix step (``POST /v1/simulate`` → the loopback daemon, a
    kernel memory copy on 127.0.0.1, no NIC, no egress). It fits a lognormal to the observed durations
    (shifted by the contention factor) with objective ``sla - duration`` and reads back
    ``metrics.probability_of_loss`` ( == P(duration > sla) ) — verified against the live kernel. This
    is the "mojo decides, in real time" design: the same native engine that governs each fix also
    governs how many fixes run at once. (The predictor only runs in the contended zone — det_cap < n ≤
    ceiling; single / low concurrency is admitted at the deterministic floor outright.)
  • **numpy in-process (fallback)** — an empirical bootstrap over the same observed durations, used
    automatically if the kernel isn't reachable yet (daemon still warming up) and as the explicit
    choice via ``CODNA_ADMISSION_MC=numpy``. So the gate can NEVER stall: a kernel hiccup degrades to
    in-process math, it does not block admission.

Thread-safe: a single lock guards the in-flight counter + the duration window, so it can front a
thread-pool of fixes.
"""

from __future__ import annotations

import contextlib
import sys as _sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

try:
    import numpy as _np
except Exception:  # noqa: BLE001 — numpy optional; pure-Python fallback below
    _np = None


# A breach predictor: P(a fix admitted under this contention exceeds the SLA), given the observed
# duration window, the contention factor, the SLA, and the draw count. Strategies below.
BreachFn = Callable[[list[float], float, float, int], float]


def _numpy_breach_probability(durations: list[float], contention: float, sla: float, n: int) -> float:
    """Default in-process predictor: empirical bootstrap. Draw base durations from the observed
    distribution and inflate by the contention factor (fixes share the model rate budget, so each
    one's wall-clock stretches by ``concurrency / det_cap``). Sub-millisecond, zero network."""
    if not durations:
        return 1.0
    if _np is not None:
        base = _np.asarray(durations, dtype="float64")
        draws = _np.random.choice(base, size=n, replace=True) * contention
        return float((draws > sla).mean())
    import random

    breaches = sum(1 for _ in range(n) if random.choice(durations) * contention > sla)
    return breaches / n


def mojo_breach_probability(
    durations: list[float], contention: float, sla: float, n: int, *, client: object | None = None
) -> float:
    """Opt-in predictor backed by the engine's compiled ``monte_carlo`` mojo kernel.

    Fits a lognormal to the observed durations (μ = mean(ln d) + ln(contention), σ = stdev(ln d) — a
    constant scale on a lognormal is an additive shift on μ, matching the numpy path's
    ``× contention``) and drives ``POST /v1/simulate`` (expert mode) with objective ``sla - duration``.
    The kernel returns ``metrics.probability_of_loss`` = P(sla - duration < 0) = P(duration > sla) =
    the breach probability — verified against the live kernel.

    Reached over the LOCAL SDK (loopback daemon, zero network) — the same channel every fix step uses.
    Raises on any transport/shape error so the caller can fall back to in-process numpy."""
    import math
    import statistics

    logs = [math.log(d) for d in durations if d > 0]
    if not logs:
        return 1.0
    mu = statistics.fmean(logs) + math.log(max(contention, 1e-9))
    sigma = max(statistics.pstdev(logs) if len(logs) > 1 else 0.25, 1e-3)
    body = {
        "mode": "expert",
        "simulation_model": "monte_carlo",
        "simulation": {
            "variables": [{"name": "duration", "distribution": "lognormal", "params": {"mean": mu, "std": sigma}}],
            "objective_function": f"{float(sla)} - duration",
        },
        "runs": int(min(max(n, 100), 1_000_000)),
        "seed": 7,  # deterministic: the window already moves, so a fixed seed keeps the estimate reproducible
    }
    if client is not None:                      # injected (tests / an explicit HTTP client)
        resp = client._request("POST", "/v1/simulate", json=body)
        return float(resp["metrics"]["probability_of_loss"])
    return _encapsulated_simulate(body)


def _encapsulated_simulate(body: dict) -> float:
    """Run the simulation IN-PROCESS through the encapsulated engine. No HTTP, no daemon, no cloud.

    This replaces a `client._request("POST", "/v1/simulate", ...)` call that could never work on the
    local path: with no remote engine configured, ``codna.cli._client()`` returns
    ``LocalCodnaRuntimeClient``, which has no ``_request`` method at all — so every local call raised
    AttributeError and make_mojo_breach_fn silently swallowed it into the numpy fallback. The kernel
    "decision" only ever happened by pointing at a REMOTE HTTP engine, which is exactly the shape
    Codna does not ship: Mojo is encapsulated, reached through the local SDK, never a sidecar or a
    cloud round-trip.

    Same in-process import route ``local_client`` already uses for repository intelligence
    (``_ensure_decision_engine_imports`` → ``apps.api_server.services.*``), and it sets
    ``RUNTIME_FALLBACK_MODE=deny`` so the compiled kernel — not a silent Python reimplementation —
    computes the answer. Raises on anything unexpected so the caller's fallback is a DELIBERATE,
    logged degradation rather than an invisible default."""
    import asyncio
    import os
    import uuid

    try:                                        # a running loop means asyncio.run() would explode
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("encapsulated simulate called from an async context")

    from .local_client import _ensure_decision_engine_imports
    from .runtime.config import resolve_runtime_config

    os.environ.setdefault("RUNTIME_FALLBACK_MODE", "deny")   # compiled kernel, never a Python stand-in
    _ensure_decision_engine_imports(resolve_runtime_config())

    from pydantic import TypeAdapter

    from apps.api_server.schemas.simulate import SimulateRequest        # type: ignore[import-not-found]
    from apps.api_server.services.simulation_service import run_simulation_fast  # type: ignore[import-not-found]

    request = TypeAdapter(SimulateRequest).validate_python(body)        # SimulateRequest is a union
    envelope, _meta = asyncio.run(run_simulation_fast(request, uuid.uuid4(), None))
    payload = envelope.model_dump() if hasattr(envelope, "model_dump") else envelope
    return float((payload.get("metrics") or {})["probability_of_loss"])


def make_mojo_breach_fn(client: object | None = None, *, fallback: BreachFn = _numpy_breach_probability) -> BreachFn:
    """Build a ``BreachFn`` that routes to the encapsulated kernel, falling back to ``fallback``
    (numpy by default) on any error so a predictor/engine outage never stalls admission.

    The fallback is logged ONCE per process. It used to be silent, which made the headline claim
    ("the compiled kernel decides") unfalsifiable: the kernel path was broken on every local call and
    nothing said so — a green test suite proved only that *some* number came back.

    Which branch answered is OBSERVABLE, not only logged. The returned callable carries
    ``probe(durations, contention, sla, n) -> (probability, "mojo" | "numpy")`` — the branch name
    travels with the value, so a caller sees per call what answered without a race; ``fallback``, the
    predictor the fallback branch runs, so a caller that saw one fallback can stay on it instead of
    paying for the kernel to fail again (review_budget's bisection makes the fallback sticky for the
    rest of its decision); and ``last_predictor``, the branch the most recent plain call took
    (informational: one attribute shared by every thread). This matters where the fallback is the
    norm — the webhook image ships no ``apps`` package, so there EVERY call answers from numpy, and a
    log line that said ``mojo`` because that was the CONFIGURED predictor described nothing real."""
    warned = [False]

    def probe(durations: list[float], contention: float, sla: float, n: int) -> tuple[float, str]:
        try:
            return mojo_breach_probability(durations, contention, sla, n, client=client), "mojo"
        except Exception as exc:  # noqa: BLE001 — never let the predictor block the gate
            if not warned[0]:
                warned[0] = True
                print(
                    "codna: admission predictor fell back to in-process numpy "
                    f"({type(exc).__name__}: {exc}); concurrency is still governed, but not by the "
                    "compiled kernel",
                    file=_sys.stderr,
                )
            return fallback(durations, contention, sla, n), "numpy"

    def _fn(durations: list[float], contention: float, sla: float, n: int) -> float:
        probability, _fn.last_predictor = probe(durations, contention, sla, n)
        return probability

    _fn.probe = probe
    _fn.fallback = fallback
    _fn.last_predictor = None
    return _fn


# The sidecar's own capacity gate: run-server.ts reads AGENT_CORE_MAX_CONCURRENCY (same 8 fallback)
# and sheds anything beyond it with 429 `agent_core_saturated`. Read it from ONE place so the admitter
# tracks the real limit instead of a constant that merely happens to match — nothing sets this variable
# when the sidecar is spawned (runtime/local_stack.py), so the two 8s were previously equal only by
# coincidence. Admitting above it cannot buy throughput; it just converts admissions into 429 retries.
_SIDECAR_CEILING_FALLBACK = 8


def sidecar_ceiling() -> int:
    """Hard concurrency the agent-core sidecar will actually accept."""
    import os

    try:
        return max(1, int(os.environ.get("AGENT_CORE_MAX_CONCURRENCY", _SIDECAR_CEILING_FALLBACK)))
    except ValueError:
        return _SIDECAR_CEILING_FALLBACK


@dataclass
class AdmissionConfig:
    ceiling: int = _SIDECAR_CEILING_FALLBACK  # hard max; clamped to sidecar_ceiling() in get_admitter()
    # Concurrent model calls the provider rate-limit comfortably allows. 4, NOT 8: this is the
    # deterministic floor (det_cap = min(ceiling, model_rate_budget)), and get_admitter() has always
    # constructed it with 4 while this default said 8 — so det_cap silently doubled depending on which
    # entry point built the config. One value, one meaning.
    model_rate_budget: int = 4
    sla_s: float = 900.0             # per-fix wall-clock budget (the timeout)
    breach_threshold: float = 0.10   # admit in the MC zone only while P(SLA breach) < this
    trials: int = 4000               # MC draws
    window: int = 200                # observed-duration ring size


@dataclass
class AdmissionVerdict:
    admit: bool
    reason: str
    in_flight: int
    det_cap: int
    p_breach: float | None = None


class MCAdmissionDecider:
    def __init__(self, config: AdmissionConfig | None = None, mc_fn: BreachFn | None = None) -> None:
        self.cfg = config or AdmissionConfig()
        # The breach predictor. None → in-process numpy bootstrap; pass make_mojo_breach_fn() to let
        # Algenta's compiled kernel decide (it already falls back to numpy if the daemon isn't up).
        self._mc_fn: BreachFn = mc_fn or _numpy_breach_probability
        self._lock = threading.Lock()
        self._in_flight = 0
        self._durations: deque[float] = deque(maxlen=self.cfg.window)
        self._arrivals: deque[float] = deque(maxlen=self.cfg.window)

    # ── lifecycle the orchestrator calls around each fix ────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._arrivals.append(time.monotonic())

    def finish(self, duration_s: float) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if duration_s > 0:
                self._durations.append(float(duration_s))

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    # ── the decision ────────────────────────────────────────────────────────────────────────────
    def _det_cap(self, model_rate_budget: int | None) -> int:
        cfg = self.cfg
        return max(1, min(cfg.ceiling, model_rate_budget if model_rate_budget is not None else cfg.model_rate_budget))

    def _decide(self, inflight: int, durations: list[float], det_cap: int) -> AdmissionVerdict:
        cfg = self.cfg
        prospective = inflight + 1
        if prospective <= det_cap:
            return AdmissionVerdict(True, "within deterministic capacity", inflight, det_cap)
        if prospective > cfg.ceiling:
            return AdmissionVerdict(False, "at hard ceiling", inflight, det_cap)
        if not durations:
            # No history yet in the MC zone — be conservative: hold at the deterministic floor.
            return AdmissionVerdict(False, "no duration history; holding at det_cap", inflight, det_cap)
        p = self._mc_breach_probability(durations, concurrency=prospective, det_cap=det_cap)
        if p < cfg.breach_threshold:
            return AdmissionVerdict(True, f"MC admit (P breach {p:.2f} < {cfg.breach_threshold})", inflight, det_cap, p)
        return AdmissionVerdict(False, f"MC throttle (P breach {p:.2f} >= {cfg.breach_threshold})", inflight, det_cap, p)

    def can_admit(self, model_rate_budget: int | None = None) -> AdmissionVerdict:
        """Non-reserving PEEK at the decision (inspection/metrics). To actually take a slot use
        try_admit()/gate() — they check-and-reserve atomically, avoiding the admit/start race."""
        det_cap = self._det_cap(model_rate_budget)
        with self._lock:
            return self._decide(self._in_flight, list(self._durations), det_cap)

    def decide_for(
        self,
        inflight: int,
        durations: list[float],
        model_rate_budget: int | None = None,
    ) -> AdmissionVerdict:
        """Evaluate the SAME decision against occupancy this process does not own.

        `can_admit`/`try_admit` read `self._in_flight`, which is per-process. CI admission
        (`ci_admission`) counts jobs across every runner on the machine via a lease directory, and
        reads durations from that job class's on-disk history — so it must supply both. Exposed rather
        than reaching into `_decide` so there is exactly one copy of the floor/MC-zone/ceiling policy;
        a second copy would be free to drift from the documented behaviour.

        Reserving is the caller's job here: the fleet's slot is the lease file, not this counter.
        """
        return self._decide(inflight, list(durations), self._det_cap(model_rate_budget))

    def try_admit(self, model_rate_budget: int | None = None) -> AdmissionVerdict:
        """Atomic check-and-reserve: the in-flight slot is taken under the SAME lock as the capacity
        check, so concurrent callers can't all pass the check before any increments. Pair with
        finish(duration) when the fix completes (gate() does both)."""
        det_cap = self._det_cap(model_rate_budget)
        with self._lock:
            verdict = self._decide(self._in_flight, list(self._durations), det_cap)
            if verdict.admit:
                self._in_flight += 1
                self._arrivals.append(time.monotonic())
        return verdict

    def _mc_breach_probability(self, durations: list[float], concurrency: int, det_cap: int) -> float:
        """P(a fix admitted at this concurrency exceeds the SLA). Contention model: when concurrency
        exceeds det_cap the shared model throughput stretches each fix's wall-clock by
        ``concurrency / det_cap`` (fixes share the provider rate budget). The configured predictor
        (mojo kernel by default, numpy fallback) turns that into a breach probability."""
        contention = max(1.0, concurrency / det_cap)
        return self._mc_fn(durations, contention, self.cfg.sla_s, self.cfg.trials)

    def snapshot(self) -> dict:
        """Lightweight state for metrics/observability."""
        with self._lock:
            durs = list(self._durations)
        return {"in_flight": self._in_flight, "samples": len(durs),
                "p50_s": (sorted(durs)[len(durs) // 2] if durs else None)}

    @contextlib.contextmanager
    def gate(self, model_rate_budget: int | None = None, poll_s: float = 1.5, max_wait_s: float = 1800.0):
        """Wrap a fix: block until admitted (polling the predictive decision), then track the run.

            with decider.gate():
                run_one_fix()

        Any parallel-fix orchestrator (a batch CLI, the hosted dispatcher, the soak) drops this around
        each fix; concurrency self-governs to predicted capacity. Raises TimeoutError if never admitted
        within max_wait_s."""
        deadline = time.monotonic() + max_wait_s
        while True:
            verdict = self.try_admit(model_rate_budget)   # atomic check-and-reserve
            if verdict.admit:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"admission gate timed out: {verdict.reason}")
            time.sleep(poll_s)
        started = time.monotonic()
        try:
            yield verdict
        finally:
            self.finish(time.monotonic() - started)


def duration_to_learn(kind: str | None, duration_s: float) -> float:
    """What the FIX admitter's duration window should learn from a finished webhook job.

    ``MCAdmissionDecider.finish`` records every positive duration it is handed, and the webhook worker
    hands it each claimed job's wall-clock — reviews included, whose ten-minute turns (review_budget)
    would stretch the distribution the gate forecasts FIX admission against. Only a fix's wall-clock
    is a sample of the series this gate models (its SLA is ``CODNA_FIX_SLA_S``); for every other kind
    this returns 0.0, which ``finish`` treats as "release the slot, record nothing". Reviews already
    have their own series: ``review_budget.observe_review_turn`` records the review TURN inside the
    job, which is the duration the review budget actually learns from."""
    return float(duration_s) if kind == "fix" and duration_s > 0 else 0.0


_ADMITTER: MCAdmissionDecider | None = None
_ADMITTER_LOCK = threading.Lock()


def get_admitter() -> MCAdmissionDecider:
    """Process-wide admitter shared by all concurrent fixes in this process. Tunable via env:
    CODNA_FIX_CEILING (hard max), CODNA_MODEL_RATE_BUDGET (deterministic floor), CODNA_FIX_SLA_S.
    Predictor: ``CODNA_ADMISSION_MC`` = ``mojo`` (default — Algenta's compiled kernel via the local
    SDK, numpy fallback if the daemon isn't up) or ``numpy`` (pure in-process)."""
    global _ADMITTER
    if _ADMITTER is None:
        import os
        with _ADMITTER_LOCK:
            if _ADMITTER is None:
                mc = os.environ.get("CODNA_ADMISSION_MC", "mojo").strip().lower()
                mc_fn = make_mojo_breach_fn() if mc == "mojo" else _numpy_breach_probability
                # CLAMPED to what the sidecar will accept: admitting past its MAX_CONCURRENCY only
                # turns admissions into 429 `agent_core_saturated` retries, so a larger
                # CODNA_FIX_CEILING would buy queueing, not throughput.
                hard = sidecar_ceiling()
                _ADMITTER = MCAdmissionDecider(
                    AdmissionConfig(
                        ceiling=min(int(os.environ.get("CODNA_FIX_CEILING", str(hard))), hard),
                        model_rate_budget=int(
                            os.environ.get("CODNA_MODEL_RATE_BUDGET", str(AdmissionConfig.model_rate_budget))
                        ),
                        sla_s=float(os.environ.get("CODNA_FIX_SLA_S", "900")),
                    ),
                    mc_fn=mc_fn,
                )
    return _ADMITTER
