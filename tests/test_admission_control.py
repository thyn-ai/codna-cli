"""MC-predictive admission control (`codna.admission_control`).

This module shipped complete but wired to NOTHING — no callers, no tests — while a static
`CODNA_WEBHOOK_CONCURRENCY=2` actually governed how many fixes ran at once. These tests pin the
contract now that `WorkerPool` gates on it, and they cover two self-inconsistencies found while
wiring it up (see `test_model_rate_budget_is_one_value` and `test_ceiling_is_clamped_to_the_sidecar`).

The predictor is always INJECTED here (`mc_fn`) so no test depends on the daemon, the mojo kernel, or
numpy's RNG. The kernel path is verified separately against a live runtime — a green run of this file
proves the *policy*, never that the kernel decided anything.
"""
from __future__ import annotations

import importlib
import threading

import pytest

import codna.admission_control as ac


def _cfg(**kw) -> ac.AdmissionConfig:
    base = {"ceiling": 8, "model_rate_budget": 2, "sla_s": 900.0, "breach_threshold": 0.10}
    base.update(kw)
    return ac.AdmissionConfig(**base)


def _decider(*, mc_fn=None, **kw) -> ac.MCAdmissionDecider:
    return ac.MCAdmissionDecider(_cfg(**kw), mc_fn=mc_fn)


# ── the deterministic floor: no prediction at all ──────────────────────────────────────────────
def test_deterministic_floor_never_consults_the_predictor():
    """Below det_cap the answer is "clearly safe" by construction, so the MC predictor must not run.
    If it did, every trivially-safe fix would pay a simulation (and, on the mojo path, a round trip
    to the engine) for a decision that cannot come out any other way."""
    calls = []

    def _spy(durations, contention, sla, n):
        calls.append(contention)
        return 0.0

    d = _decider(mc_fn=_spy, model_rate_budget=2, ceiling=8)
    # Seed history so the predictor COULD run — proving the floor, not an empty-window shortcut.
    d.finish(10.0)
    d.finish(12.0)

    assert d.try_admit().admit is True     # prospective 1 <= det_cap 2
    assert d.try_admit().admit is True     # prospective 2 <= det_cap 2
    assert calls == [], f"predictor ran inside the deterministic floor: {calls}"


def test_hard_ceiling_is_absolute_even_when_the_predictor_says_zero_risk():
    """A predictor that always returns 0.0 must still not push past the ceiling: beyond it the
    sidecar itself sheds load with 429 agent_core_saturated, so admitting is worse than waiting."""
    d = _decider(mc_fn=lambda *a: 0.0, model_rate_budget=2, ceiling=3)
    d.finish(10.0)
    for _ in range(3):
        assert d.try_admit().admit is True
    verdict = d.try_admit()
    assert verdict.admit is False
    assert "ceiling" in verdict.reason
    assert d.in_flight == 3


# ── the MC zone ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "p_breach, expect_admit",
    [(0.0, True), (0.09, True), (0.10, False), (0.5, False)],
    ids=["no-risk", "just-under", "at-threshold", "way-over"],
)
def test_mc_zone_admits_strictly_below_the_breach_threshold(p_breach, expect_admit):
    """Between det_cap and ceiling the verdict is P(SLA breach) < breach_threshold. At-threshold must
    THROTTLE (the comparison is strict `<`), which is the boundary a refactor is most likely to flip."""
    d = _decider(mc_fn=lambda *a: p_breach, model_rate_budget=1, ceiling=4, breach_threshold=0.10)
    d.finish(10.0)
    assert d.try_admit().admit is True          # fills the deterministic floor
    verdict = d.try_admit()                     # now in the MC zone
    assert verdict.admit is expect_admit
    assert verdict.p_breach == pytest.approx(p_breach)


def test_mc_zone_with_no_history_holds_at_the_floor():
    """With an empty duration window there is nothing to simulate. Holding at det_cap is the
    conservative branch; guessing would mean admitting on a fabricated distribution."""
    d = _decider(mc_fn=lambda *a: 0.0, model_rate_budget=1, ceiling=4)
    assert d.try_admit().admit is True
    verdict = d.try_admit()
    assert verdict.admit is False
    assert "no duration history" in verdict.reason


def test_contention_grows_with_concurrency():
    """The contention model is concurrency/det_cap — the reason a fix's wall-clock stretches as more
    fixes share the provider's rate budget. Pin it: without it the MC zone would simulate the
    UNCONTENDED distribution and cheerfully admit into a breach."""
    seen = []
    d = _decider(mc_fn=lambda dur, contention, sla, n: seen.append(contention) or 0.0,
                 model_rate_budget=2, ceiling=6)
    d.finish(10.0)
    for _ in range(2):
        d.try_admit()          # floor: no predictor call
    d.try_admit()              # prospective 3, det_cap 2 -> 1.5
    d.try_admit()              # prospective 4            -> 2.0
    assert seen == [pytest.approx(1.5), pytest.approx(2.0)]


# ── never stall ────────────────────────────────────────────────────────────────────────────────
def test_a_raising_predictor_degrades_instead_of_blocking_admission():
    """A predictor/engine outage must never wedge the gate. make_mojo_breach_fn wraps the kernel with
    exactly this fallback, so verify the wrapper, not just the docstring."""
    def _boom(*_a, **_k):
        raise RuntimeError("engine down")

    fn = ac.make_mojo_breach_fn(fallback=lambda *a: 0.0)
    # mojo_breach_probability will fail (no client/daemon here) -> falls back to the given predictor.
    assert fn([10.0, 11.0], 1.5, 900.0, 128) == 0.0

    # And an explicitly exploding fallback surfaces rather than hanging.
    fn_bad = ac.make_mojo_breach_fn(fallback=_boom)
    with pytest.raises(RuntimeError):
        fn_bad([10.0], 1.5, 900.0, 8)


# ── atomicity ──────────────────────────────────────────────────────────────────────────────────
def test_try_admit_is_atomic_under_concurrent_callers():
    """check-and-reserve happens under ONE lock. With a separate can_admit()+start() every thread
    could pass the check before any incremented, overshooting the ceiling."""
    ceiling = 4
    d = _decider(mc_fn=lambda *a: 0.0, model_rate_budget=ceiling, ceiling=ceiling)
    admitted, peak, lock = [], [0], threading.Lock()
    start = threading.Barrier(24)

    def _worker():
        start.wait()
        if d.try_admit().admit:
            with lock:
                admitted.append(1)
                peak[0] = max(peak[0], d.in_flight)

    threads = [threading.Thread(target=_worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(admitted) == ceiling, f"admitted {len(admitted)}, ceiling {ceiling}"
    assert peak[0] <= ceiling
    assert d.in_flight == ceiling


def test_finish_releases_the_slot_and_only_records_positive_durations():
    """WorkerPool calls finish(0.0) when a reserved slot found an empty queue: that must free capacity
    WITHOUT polluting the window the predictor learns from (an idle poll is not a fix duration)."""
    d = _decider(mc_fn=lambda *a: 0.0)
    d.try_admit()
    assert d.in_flight == 1
    d.finish(0.0)
    assert d.in_flight == 0
    assert d.snapshot()["samples"] == 0

    d.try_admit()
    d.finish(12.5)
    assert d.in_flight == 0
    assert d.snapshot()["samples"] == 1
    assert d.snapshot()["p50_s"] == pytest.approx(12.5)


# ── the two self-inconsistencies found while wiring this up ────────────────────────────────────
def test_model_rate_budget_is_one_conservative_value():
    """REGRESSION: AdmissionConfig's dataclass default said 8 while get_admitter() constructed 4.
    det_cap = min(ceiling, model_rate_budget), so the deterministic floor silently DOUBLED depending
    on which entry point built the config — 8 admits four extra fixes with NO breach prediction.

    Asserts the VALUE, not merely that the two agree: get_admitter() now derives its env default from
    the dataclass, so an agreement-only check is a tautology that passes with the bug reinstated
    (verified by mutation). 4 is the deliberate conservative floor; changing it should be a conscious
    edit here too."""
    assert ac.AdmissionConfig().model_rate_budget == 4
    assert ac.get_admitter().cfg.model_rate_budget == 4


def test_ceiling_is_clamped_to_the_sidecar(monkeypatch):
    """REGRESSION: `ceiling`'s comment claimed it equalled the sidecar's AGENT_CORE_MAX_CONCURRENCY,
    but nothing linked them — the two 8s matched by coincidence. Admitting past what run-server.ts
    accepts only converts admissions into 429 agent_core_saturated retries."""
    monkeypatch.setenv("AGENT_CORE_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("CODNA_FIX_CEILING", "99")
    monkeypatch.setenv("CODNA_ADMISSION_MC", "numpy")
    reloaded = importlib.reload(ac)
    try:
        assert reloaded.sidecar_ceiling() == 3
        assert reloaded.get_admitter().cfg.ceiling == 3
    finally:
        # Restore a clean module + singleton for the rest of the session.
        monkeypatch.undo()
        importlib.reload(ac)


def test_sidecar_ceiling_survives_a_garbage_env_value(monkeypatch):
    monkeypatch.setenv("AGENT_CORE_MAX_CONCURRENCY", "not-a-number")
    assert ac.sidecar_ceiling() == ac._SIDECAR_CEILING_FALLBACK


# ── the predictor must reach the engine IN-PROCESS, never over HTTP ────────────────────────────
def test_default_predictor_does_not_use_an_http_client():
    """REGRESSION, and the reason the kernel claim was never true locally: this used to do
    `codna.cli._client()._request("POST", "/v1/simulate", ...)`. With no remote engine configured
    that returns LocalCodnaRuntimeClient, which has NO `_request` attribute — so every local call
    raised AttributeError and was swallowed into the numpy fallback. The kernel could only ever be
    reached by pointing at a REMOTE HTTP engine, which is not what Codna ships: Mojo is encapsulated
    and reached in-process through the local SDK.

    Pinned structurally: the module must not route the default path through cli._client()."""
    import inspect

    src = inspect.getsource(ac.mojo_breach_probability)
    assert "_encapsulated_simulate" in src, "default path must go in-process"
    # cli._client() may only be consulted when a client was explicitly injected.
    assert "from codna.cli import" not in src, (
        "the default predictor must not build an HTTP/CLI client — that is the bug this replaced"
    )
    assert "LocalCodnaRuntimeClient" not in src

    enc = inspect.getsource(ac._encapsulated_simulate)
    assert "run_simulation_fast" in enc, "must call the engine's in-process simulation entry point"
    assert 'RUNTIME_FALLBACK_MODE' in enc, "must forbid a silent Python stand-in for the kernel"


def test_injected_client_is_still_honoured():
    """An explicitly injected client (tests, or a deliberate remote engine) still routes through it —
    the in-process path is the DEFAULT, not the only option."""
    seen = {}

    class _FakeClient:
        def _request(self, method, path, json=None):
            seen["call"] = (method, path)
            return {"metrics": {"probability_of_loss": 0.33}}

    p = ac.mojo_breach_probability([300.0, 400.0], 1.5, 900.0, 256, client=_FakeClient())
    assert p == pytest.approx(0.33)
    assert seen["call"] == ("POST", "/v1/simulate")


def test_fallback_is_announced_not_silent(capsys):
    """A silent fallback is what made "the compiled kernel decides" unfalsifiable — the path was
    broken on every local call and nothing said so."""
    fn = ac.make_mojo_breach_fn(fallback=lambda *a: 0.5)

    class _Boom:
        def _request(self, *a, **k):
            raise RuntimeError("engine unavailable")

    assert fn.__name__ == "_fn"
    # Force the failure path through an injected client that raises.
    boom = ac.make_mojo_breach_fn(client=_Boom(), fallback=lambda *a: 0.5)
    assert boom([300.0], 1.5, 900.0, 100) == pytest.approx(0.5)
    err = capsys.readouterr().err
    assert "fell back to in-process numpy" in err
    assert "not by the compiled kernel" in err

    # ...and only ONCE per built predictor, so a throttled fleet does not spam the log.
    boom([300.0], 1.5, 900.0, 100)
    assert capsys.readouterr().err == ""


# ── what the fix admitter learns from ──────────────────────────────────────────────────────────
def test_duration_to_learn_keeps_only_fix_wall_clocks():
    """The webhook loop hands the admitter every claimed job's duration; only a fix's belongs in the
    window the FIX gate forecasts against. 0.0 is `finish`'s "release, record nothing"."""
    assert ac.duration_to_learn("fix", 612.5) == 612.5
    assert ac.duration_to_learn("review", 612.5) == 0.0
    assert ac.duration_to_learn("secure", 90.0) == 0.0
    assert ac.duration_to_learn("queue", 3.0) == 0.0
    assert ac.duration_to_learn(None, 12.0) == 0.0
    assert ac.duration_to_learn("fix", 0.0) == 0.0 and ac.duration_to_learn("fix", -1.0) == 0.0
    d = ac.MCAdmissionDecider(ac.AdmissionConfig(ceiling=2, model_rate_budget=2), mc_fn=lambda *a: 0.0)
    d.start()
    d.finish(ac.duration_to_learn("review", 600.0))
    d.start()
    d.finish(ac.duration_to_learn("fix", 120.0))
    assert d.in_flight == 0 and d.snapshot()["samples"] == 1 and d.snapshot()["p50_s"] == 120.0


def test_the_kernel_predictor_reports_which_branch_answered(capsys):
    """`make_mojo_breach_fn` used to hide whether the kernel or numpy answered; now every call can
    say so -- `probe` returns the branch with the value, `fallback` is the predictor the fallback
    branch runs, `last_predictor` is the most recent plain call's branch."""
    class _Boom:
        def _request(self, *a, **k):
            raise RuntimeError("engine unavailable")

    class _Kernel:
        def _request(self, *a, **k):
            return {"metrics": {"probability_of_loss": 0.25}}

    def fallback(*_a):          # a named stand-in the assertions can identify
        return 0.5

    down = ac.make_mojo_breach_fn(client=_Boom(), fallback=fallback)
    assert down.last_predictor is None and down.fallback is fallback
    assert down.probe([300.0], 1.5, 900.0, 100) == (pytest.approx(0.5), "numpy")
    assert down([300.0], 1.5, 900.0, 100) == pytest.approx(0.5) and down.last_predictor == "numpy"
    up = ac.make_mojo_breach_fn(client=_Kernel(), fallback=fallback)
    assert up.probe([300.0], 1.5, 900.0, 100) == (pytest.approx(0.25), "mojo")
    assert up([300.0], 1.5, 900.0, 100) == pytest.approx(0.25) and up.last_predictor == "mojo"
    assert "fell back to in-process numpy" in capsys.readouterr().err

