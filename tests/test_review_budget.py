"""The adaptive review turn budget (`codna.review_budget`) and its wiring.

Live failure this closes: `codna review` check run 105948327905 on thyn-ai/algenta-sdk#71
(+1008/-15 across 5 files, one 865-line fast-check test file) failed 2026-09-19 18:47-18:52Z with

    cause_code: agent_core_run_failed / terminal_state: runtime_invalid /
    error: turn exceeded timeout budget of 240000ms

because a review turn ran under ONE fixed number (`review_timeout_s = 240`) while a fix turn got
`_dynamic_timeout_ms` and the admission gate got a Monte-Carlo forecast. These tests pin the review
budget as the same two mechanisms reused: the size formula, and the gate's breach predictor over
observed review turns. The predictor is always INJECTED (a deterministic analytic stand-in for the
kernel, or the seeded pure-Python fallback), exactly as tests/test_admission_control.py does, so no
test depends on the daemon, the mojo kernel, or numpy's RNG.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from pathlib import Path

import pytest

import codna.admission_control as ac
from codna import packaged_agent_runner as par
from codna import review_budget as rb
from codna.packaged_repository_advanced import PackagedAgentRunRequest, PackagedRepositoryAdvancedError

REPO = pathlib.Path(__file__).resolve().parents[2]

# The shape that failed live, and a trivial pull request for contrast.
SDK71 = dict(changed_lines=1023, changed_files=5, prompt_tokens=12_000)
TINY = dict(changed_lines=10, changed_files=1, prompt_tokens=400)


class _AnalyticKernel:
    """Stands in for the engine's `monte_carlo` kernel behind `mojo_breach_probability`: reads the
    lognormal the predictor fitted (mean/std of ln duration, already shifted by the size factor) and
    the `<sla> - duration` objective, and returns P(duration > sla) in closed form. Deterministic,
    and it also asserts the request has the shape the real kernel is sent."""

    def __init__(self):
        self.calls: list[dict] = []

    def _request(self, method, path, json=None):
        assert (method, path) == ("POST", "/v1/simulate")
        assert json["simulation_model"] == "monte_carlo" and json["mode"] == "expert"
        var = json["simulation"]["variables"][0]
        assert var["name"] == "duration" and var["distribution"] == "lognormal"
        sla = float(json["simulation"]["objective_function"].split(" - ")[0])
        self.calls.append({"sla": sla, **var["params"]})
        z = (math.log(sla) - var["params"]["mean"]) / var["params"]["std"]
        return {"metrics": {"probability_of_loss": 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))}}


def _kernel_fn(kernel: _AnalyticKernel | None = None) -> ac.BreachFn:
    return ac.make_mojo_breach_fn(client=kernel or _AnalyticKernel())


def _samples(duration_s: float, n: int = 5, **shape) -> list[rb.ReviewSample]:
    shape = {**TINY, **shape}
    return [rb.ReviewSample(duration_s=duration_s, **shape) for _ in range(n)]


def _budget(samples=(), breach_fn=None, **shape) -> rb.ReviewBudget:
    return rb.review_turn_budget(**{**TINY, **shape}, samples=list(samples), breach_fn=breach_fn or _kernel_fn())


# ── source 1: the size formula ──────────────────────────────────────────────────────────────────
def test_the_size_formula_is_the_fix_paths_formula():
    """`_dynamic_timeout_ms` (fixes) must still compute exactly what it did -- 300 s base + 2 ms per
    token (cap 900 s) + 1 s per file (cap 600 s), cap 1800 s -- now through the shared helper."""
    assert par._dynamic_timeout_ms({}) == 300_000
    assert par._dynamic_timeout_ms({"raw_repo_token_estimate": 100_000, "snapshot_file_count": 500}) == 1_000_000
    assert par._dynamic_timeout_ms({"raw_repo_token_estimate": 10_000_000, "file_count": 10_000_000}) == 1_800_000
    # and the review path calls the same function with the review coefficients
    assert rb.review_size_ms(0, 0, 0) == rb.size_budget_ms(base_ms=rb.REVIEW_FLOOR_MS, cap_ms=rb.REVIEW_CAP_MS)


def test_budget_is_monotone_in_lines_files_and_tokens():
    zero = dict(changed_lines=0, changed_files=0, prompt_tokens=0)
    for key in zero:
        last = _budget(**zero).budget_ms
        for value in (5, 50, 500, 5_000, 50_000):
            now = _budget(**{**zero, key: value}).budget_ms
            assert now >= last, f"{key}={value} lowered the budget ({last} -> {now})"
            last = now
        assert last > _budget(**zero).budget_ms, f"{key} never moved the budget"


def test_a_ten_line_pull_request_stays_near_todays_budget():
    d = _budget(**TINY)
    assert d.source == "size"
    assert 240_000 <= d.budget_ms <= 260_000, d


def test_the_algenta_sdk_71_shape_lands_comfortably_above_240s():
    """+1008/-15 across 5 files: the live failure. 'Comfortably' = at least 2.5x the old fixed budget,
    and well under the cap, so a bigger PR still has room to grow."""
    d = _budget(**SDK71)
    assert d.source == "size"
    assert d.budget_ms >= 600_000, d
    assert d.budget_ms < rb.REVIEW_CAP_MS


def test_floor_is_todays_240s_and_is_never_undercut():
    assert rb.REVIEW_FLOOR_MS == 240_000
    assert _budget(changed_lines=0, changed_files=0, prompt_tokens=0).budget_ms == 240_000
    # very FAST history must not pull the budget below the size formula, let alone the floor
    d = _budget(samples=_samples(5.0), **TINY)
    assert d.budget_ms == d.size_ms >= rb.REVIEW_FLOOR_MS and d.source == "size"


def test_cap_is_respected_and_sits_strictly_below_the_webhook_job_bound():
    """The cap must leave the job bound room for the +30 s sidecar-wait slack
    (packaged_agent_runner._run_sidecar) plus the clone and the post around the turn."""
    d = _budget(changed_lines=100_000, changed_files=1_000, prompt_tokens=1_000_000)
    assert d.budget_ms == rb.REVIEW_CAP_MS and d.source == "cap"

    fly = (REPO / "infra" / "fly" / "codna-webhook.fly.toml").read_text(encoding="utf-8")
    fly_job_s = int(next(ln for ln in fly.splitlines() if ln.strip().startswith("CODNA_WEBHOOK_JOB_TIMEOUT_S")).split('"')[1])
    assert fly_job_s == 1800
    assert rb.REVIEW_CAP_MS / 1000 + 30 < fly_job_s
    from codna import webhook_worker
    assert rb.REVIEW_CAP_MS / 1000 + 30 < webhook_worker._JOB_TIMEOUT_S
    # and an MC verdict past the cap is clamped too
    d = _budget(samples=_samples(3_000.0), **SDK71)
    assert d.budget_ms == rb.REVIEW_CAP_MS and d.source == "cap"


def test_env_override_wins_as_it_does_for_fixes(monkeypatch):
    monkeypatch.setenv("ALGENTA_AGENT_TURN_TIMEOUT_MS", "90000")
    d = _budget(samples=_samples(3_000.0), **SDK71)
    assert d.budget_ms == 90_000 and d.source == "env"
    assert par._dynamic_timeout_ms({"raw_repo_token_estimate": 1_000_000}) == 90_000
    monkeypatch.setenv("ALGENTA_AGENT_TURN_TIMEOUT_MS", "garbage")
    assert _budget().budget_ms == 30_000                     # the same clamp _dynamic_timeout_ms applies


# ── source 2: the gate's Monte-Carlo model over observed review turns ───────────────────────────
def test_size_factor_is_the_size_formula_over_its_base():
    assert rb.size_factor(0, 0, 0) == 1.0
    assert rb.size_factor(**SDK71) == pytest.approx(1 + (1023 * 350 + 5 * 5_000 + 12_000 * 2) / 240_000)


def test_slow_observed_turns_raise_the_budget_above_the_size_formula():
    """Five small reviews that each took 400 s say the rate is ~385 s per unit of size; a review 2.15
    units big is then forecast at ~830 s, so the budget must move to the first grid step above that
    -- above the 516 s the size formula alone would grant."""
    kernel = _AnalyticKernel()
    d = _budget(samples=_samples(400.0), breach_fn=_kernel_fn(kernel), changed_lines=700, changed_files=3, prompt_tokens=8_000)
    assert d.source == "mc" and d.mc_ms == d.budget_ms
    assert d.size_ms < d.budget_ms <= rb.REVIEW_CAP_MS
    expected = 400.0 / rb.size_factor(**TINY) * d.size_factor
    assert expected * 1000 < d.budget_ms <= expected * 1000 + 2 * rb._GRID_MS
    # the kernel was probed with THIS review's size factor in the contention slot: ln(rate) + ln(factor)
    assert all(c["mean"] == pytest.approx(math.log(400.0 / rb.size_factor(**TINY)) + math.log(d.size_factor)) for c in kernel.calls)
    assert 1 < len(kernel.calls) <= 10, "bisection, not a full grid walk"


def test_the_pure_python_fallback_predictor_reaches_the_same_verdict(monkeypatch):
    """The numpy/pure-Python fallback (`_numpy_breach_probability`) is what decides when the kernel
    is unreachable; driven deterministically (no numpy, seeded RNG) it must agree with the kernel."""
    monkeypatch.setattr(ac, "_np", None)
    random.seed(7)
    d = _budget(samples=_samples(400.0), breach_fn=ac._numpy_breach_probability,
                changed_lines=700, changed_files=3, prompt_tokens=8_000)
    assert d.source == "mc" and d.size_ms < d.budget_ms
    expected = 400.0 / rb.size_factor(**TINY) * d.size_factor
    assert expected * 1000 < d.budget_ms <= expected * 1000 + 2 * rb._GRID_MS


def test_the_mc_branch_never_lowers_the_budget_below_the_size_formula():
    d = _budget(samples=_samples(60.0), **SDK71)
    assert d.mc_ms is not None and d.mc_ms < d.size_ms
    assert d.budget_ms == d.size_ms and d.source == "size"


def test_a_wider_observed_distribution_buys_more_margin():
    tight = _budget(samples=_samples(300.0), **SDK71)
    spread = _budget(samples=[rb.ReviewSample(s, **TINY) for s in (150.0, 200.0, 300.0, 450.0, 600.0)], **SDK71)
    assert spread.budget_ms > tight.budget_ms


def test_no_history_means_the_size_formula_and_no_kernel_probe():
    kernel = _AnalyticKernel()
    d = _budget(samples=[], breach_fn=_kernel_fn(kernel), **SDK71)
    assert d.source == "size" and d.mc_ms is None and kernel.calls == []


def test_an_exhausted_budget_is_a_censored_sample_that_grows_the_retry():
    """The turn that timed out is a lower bound, not a duration: kept OUT of the rate distribution
    (normalised, it would pull the tail down) and turned into a floor for its size class -- the
    retry of the same pull request gets at least the exhausted budget plus a quarter."""
    exhausted = rb.ReviewSample(duration_s=647.0, timed_out=True, budget_ms=647_000, **SDK71)
    d = _budget(samples=[exhausted], **SDK71)
    assert d.source == "censored" and d.censored_samples == 1 and d.mc_ms is None
    assert d.budget_ms == pytest.approx(647.0 * rb.CENSORED_GROWTH * 1000, abs=1)
    assert d.budget_ms > d.size_ms
    # pro-rated to size: the same evidence gives a tiny PR a proportionally smaller floor
    tiny = _budget(samples=[exhausted], **TINY)
    assert tiny.censored_ms == pytest.approx(647.0 * rb.CENSORED_GROWTH * 1000 * rb.size_factor(**TINY) / rb.size_factor(**SDK71), abs=1)


def test_epsilon_is_configurable_and_tighter_means_more_budget(monkeypatch):
    samples = [rb.ReviewSample(s, **TINY) for s in (150.0, 200.0, 300.0, 450.0, 600.0)]
    loose = rb.review_turn_budget(**TINY, samples=samples, breach_fn=_kernel_fn(), epsilon=0.20)
    tight = rb.review_turn_budget(**TINY, samples=samples, breach_fn=_kernel_fn(), epsilon=0.01)
    assert loose.source == tight.source == "mc"
    assert rb.REVIEW_FLOOR_MS < loose.budget_ms < tight.budget_ms < rb.REVIEW_CAP_MS
    monkeypatch.setenv("CODNA_REVIEW_BUDGET_EPSILON", "0.5")
    assert _budget().epsilon == 0.5
    monkeypatch.setenv("CODNA_REVIEW_BUDGET_EPSILON", "7")
    assert _budget().epsilon == rb.DEFAULT_EPSILON


# ── the persisted window ────────────────────────────────────────────────────────────────────────
def test_history_roundtrip_is_tolerant_and_trimmed(tmp_path):
    path = tmp_path / "w" / "review-turns.jsonl"
    assert rb.load_samples(path) == []                       # no file, no dir: empty window
    rb.record_sample(rb.ReviewSample(0.0, **TINY), path)     # non-positive: never recorded
    assert not path.exists()
    for i in range(1, 8):
        rb.record_sample(rb.ReviewSample(float(i), **TINY, timed_out=(i == 7), budget_ms=240_000), path, window=3)
    kept = rb.load_samples(path, window=3)
    assert [s.duration_s for s in kept] == [5.0, 6.0, 7.0]
    assert kept[-1].timed_out is True and kept[-1].budget_ms == 240_000
    assert len(path.read_text().splitlines()) <= 6              # trimmed at 2 x window
    path.write_text(path.read_text() + "not json\n" + json.dumps({"duration_s": "x"}) + "\n")
    assert [s.duration_s for s in rb.load_samples(path, window=3)] == [5.0, 6.0, 7.0]   # garbage never costs a sample


def test_history_path_prefers_the_configured_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_REVIEW_HISTORY_DIR", str(tmp_path / "h"))
    assert rb.history_path() == tmp_path / "h" / rb.HISTORY_FILE
    monkeypatch.delenv("CODNA_REVIEW_HISTORY_DIR")
    assert rb.history_path() == Path.home() / ".codna" / "review-budget" / rb.HISTORY_FILE


# ── wiring: the request the review builds, the payload the sidecar gets ─────────────────────────
def _review_request(**kw) -> PackagedAgentRunRequest:
    base = dict(repository_id="review_r", snapshot_id="review_s", repo_root=Path("/no/such/repo"),
                issue_text="Review the following pull-request diff ...", model="m",
                signals={rb.SIGNAL_CHANGED_LINES: 1023, rb.SIGNAL_CHANGED_FILES: 5}, evidence_bundle={},
                snapshot={"snapshot_id": "review_s", "snapshot_file_count": 0, "raw_repo_token_estimate": 12_000},
                task_kind="review")
    base.update(kw)
    return PackagedAgentRunRequest(**base)


def test_review_payload_carries_the_adaptive_budget_and_logs_the_decision(capsys):
    payload = par._sidecar_payload(_review_request(), Path("/tmp"))
    assert payload["limits"]["timeoutMs"] > 240_000
    assert payload["limits"]["timeoutMs"] == rb.review_size_ms(1023, 5, 12_000)
    line = next(ln for ln in capsys.readouterr().err.splitlines() if '"review_turn_budget"' in ln)
    logged = json.loads(line)
    assert logged["changed_lines"] == 1023 and logged["changed_files"] == 5 and logged["prompt_tokens"] == 12_000
    assert logged["samples"] == 0 and logged["budget_ms"] == payload["limits"]["timeoutMs"] and logged["source"] == "size"
    assert logged["floor_ms"] == 240_000 and logged["cap_ms"] == rb.REVIEW_CAP_MS
    assert logged["predictor"] is None and logged["predictor_configured"] == rb.predictor_name()  # no forecast ran
    assert rb.decision_for("review_s").budget_ms == payload["limits"]["timeoutMs"]   # the hand-off the review reads


def test_review_payload_learns_from_the_persisted_window(monkeypatch):
    """End to end through the file: recorded slow turns -> a larger sidecar budget for the next review."""
    monkeypatch.setenv("CODNA_ADMISSION_MC", "numpy")
    monkeypatch.setattr(rb, "_BREACH_FN", None)
    monkeypatch.setattr(ac, "_np", None)
    random.seed(7)
    cold = par._sidecar_payload(_review_request(), Path("/tmp"))["limits"]["timeoutMs"]
    for s in _samples(400.0):
        rb.record_sample(s)
    warm = par._sidecar_payload(_review_request(), Path("/tmp"))["limits"]["timeoutMs"]
    assert warm > cold
    monkeypatch.setattr(rb, "_BREACH_FN", None)


def test_an_explicit_review_timeout_signal_is_still_a_callers_pin():
    payload = par._sidecar_payload(_review_request(signals={"review_timeout_s": 45, "review_max_iterations": 3}), Path("/tmp"))
    assert payload["limits"] == {"maxIterations": 3, "timeoutMs": 45_000}


def test_the_fix_payload_does_not_go_through_the_review_budget():
    payload = par._sidecar_payload(_review_request(task_kind="fix", signals={}), Path("/tmp"))
    assert payload["limits"]["timeoutMs"] == par._dynamic_timeout_ms({"raw_repo_token_estimate": 12_000})


def test_the_python_subprocess_wait_tracks_the_adaptive_budget(monkeypatch):
    """`_run_sidecar` waits limits.timeoutMs/1000 + 30 s -- so the wait must follow the budget or the
    CLI would give up on the sidecar before the sidecar gave up on the turn."""
    seen = {}

    class _Resp:
        status_code = 200
        text = json.dumps({"type": "final", "status": "succeeded", "terminal_state": "succeeded", "text": '{"findings": []}'})

    class _Endpoint:
        url = "http://127.0.0.1:1"

    monkeypatch.setattr(par, "ensure_agent_core_running", lambda keys=None: _Endpoint())
    monkeypatch.setattr(par.httpx, "post", lambda url, json, timeout: seen.update(payload=json, timeout=timeout) or _Resp())
    runner = par.SidecarPackagedAgentRunner(config=object(), keys=None)
    runner._run_sidecar(_review_request(), Path("/tmp"))
    assert seen["payload"]["limits"]["timeoutMs"] > 240_000
    assert seen["timeout"] == pytest.approx(seen["payload"]["limits"]["timeoutMs"] / 1000 + 30)


# ── a turn that still outruns its budget ────────────────────────────────────────────────────────
_SIDECAR_TIMEOUT_DETAILS = {
    "sidecar_url": "http://127.0.0.1:34048", "status": "failed", "terminal_state": "runtime_invalid",
    "task_kind": "review", "accepted_terminal_states": ["apply_blocked", "succeeded"],
    "error": "turn exceeded timeout budget of 647050ms",
}


def test_the_sidecars_timeout_text_is_the_one_the_classifier_matches():
    """The classifier keys on agent-core's own message; pin the two sides together so a reworded
    sidecar cannot silently turn every timeout back into a generic review_error."""
    src = (REPO / "agent-core" / "vendor" / "cline" / "algenta" / "server" / "execution.ts").read_text(encoding="utf-8")
    assert "turn exceeded timeout budget of ${timeoutMs}ms" in src
    assert rb.is_turn_timeout(_SIDECAR_TIMEOUT_DETAILS, cause_code="agent_core_run_failed", elapsed_s=650.0, budget_ms=None)
    assert rb.budget_from_error(_SIDECAR_TIMEOUT_DETAILS) == 647_050
    # the other shape: the sidecar never answered and codna's own wait (budget + 30 s) gave up
    assert rb.is_turn_timeout({"error": "ReadTimeout"}, cause_code="agent_core_transport_failed", elapsed_s=680.0, budget_ms=647_050)
    assert not rb.is_turn_timeout({"error": "connection refused"}, cause_code="agent_core_transport_failed", elapsed_s=2.0, budget_ms=647_050)
    assert not rb.is_turn_timeout({"error": "provider request rejected", "terminal_state": "failed"}, cause_code="agent_core_run_failed", elapsed_s=650.0, budget_ms=647_050)


def _stub_sidecar(monkeypatch, run):
    import codna.cli as cli_mod
    import codna.runtime.config as rc

    monkeypatch.setattr(cli_mod, "_runtime_keys", lambda include_keychain=True: {})
    monkeypatch.setattr(rc, "resolve_runtime_config", lambda keys=None: object())
    seen = []

    class _Runner:
        def __init__(self, *, config, keys):
            pass

        def run(self, request):
            seen.append(request)
            return run(request)

    monkeypatch.setattr(par, "SidecarPackagedAgentRunner", _Runner)
    return seen


def test_a_turn_that_outruns_its_budget_is_classified_as_review_timeout(monkeypatch, tmp_path):
    from codna import review_findings as rf
    from codna.cline_agent import ClineAgentError, ReviewTurnTimeout

    def _timeout(request):
        # the runner computed and remembered the budget, as _sidecar_payload does for real
        rb.remember_decision(request.snapshot_id, rb.review_turn_budget(changed_lines=1023, changed_files=5, prompt_tokens=request.snapshot["raw_repo_token_estimate"], samples=[], breach_fn=_kernel_fn()))
        raise PackagedRepositoryAdvancedError("agent_core_run_failed", "Local agent-core did not complete the packaged fix run successfully.", dict(_SIDECAR_TIMEOUT_DETAILS))

    seen = _stub_sidecar(monkeypatch, _timeout)
    with pytest.raises(ReviewTurnTimeout) as exc:
        rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=1023, changed_files=5)
    assert len(seen) == 1, "a timeout is not an output-contract miss: no repair turn"
    assert seen[0].signals[rb.SIGNAL_CHANGED_LINES] == 1023 and seen[0].signals[rb.SIGNAL_CHANGED_FILES] == 5
    assert "review_timeout_s" not in seen[0].signals              # adaptive by default: no pin
    err = exc.value
    assert isinstance(err, ClineAgentError) and err.code == "review_timeout"
    msg = str(err)
    assert "1023 changed line(s) across 5 file(s)" in msg and "granted 647 s" in msg
    assert "`@codna review`" in msg and "split the pull request" in msg
    assert err.details["cause_code"] == "agent_core_run_failed" and err.details["terminal_state"] == "runtime_invalid"
    assert err.details["budget_ms"] == 647_050 and err.details["changed_lines"] == 1023
    # the exhausted turn was recorded as a censored sample, so the retry's budget grows
    recorded = rb.load_samples()
    assert len(recorded) == 1 and recorded[0].timed_out is True and recorded[0].budget_ms == 647_050
    assert recorded[0].changed_lines == 1023 and recorded[0].changed_files == 5


def test_a_completed_turn_is_recorded_for_the_next_decision(monkeypatch, tmp_path):
    from codna import review_findings as rf

    class _Result:
        status = terminal_state = "succeeded"
        agent_run_id, session_id, text = "run_1", "sess_1", '{"findings": []}'
        files_changed, patch_diff, telemetry, artifacts, runtime = [], "", {"model": "m"}, {}, {}

    _stub_sidecar(monkeypatch, lambda request: _Result())
    raw, _agent = rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=12, changed_files=2)
    assert raw == []
    recorded = rb.load_samples()
    assert len(recorded) == 1 and recorded[0].timed_out is False
    assert recorded[0].changed_lines == 12 and recorded[0].changed_files == 2 and recorded[0].duration_s > 0


def test_a_crash_that_is_not_a_timeout_stays_a_generic_agent_failure(monkeypatch, tmp_path):
    from codna import review_findings as rf

    def _boom(request):
        raise PackagedRepositoryAdvancedError("agent_core_run_failed", "nope", {"terminal_state": "failed", "error": "provider request rejected"})

    _stub_sidecar(monkeypatch, _boom)
    with pytest.raises(PackagedRepositoryAdvancedError):
        rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=12, changed_files=2)
    assert rb.load_samples() == [], "a crash is not a review duration"


def test_run_findings_review_reports_review_timeout_and_never_posts(monkeypatch, tmp_path):
    """The CLI-level contract the check run is built from: outer code `review_timeout`, the message
    is the how-to-proceed text, and `post_review` (the only path that can APPROVE) is never reached."""
    from types import SimpleNamespace

    from codna import review, review_findings as rf, review_github as rg
    from codna.cline_agent import ReviewTurnTimeout

    def _timeout(local, **kw):
        raise ReviewTurnTimeout(rf._review_timeout_message(1023, 5, 12_000, 647_050, 650.0),
                                {"cause_code": "agent_core_run_failed", "terminal_state": "runtime_invalid", "budget_ms": 647_050})

    posted = []
    monkeypatch.setattr(rf, "run_diff_review", _timeout)
    monkeypatch.setattr(rg, "post_review", lambda *a, **k: posted.append(1))
    monkeypatch.setattr(rg, "parse_pr_arg", lambda value, repo_dir: ("acme/app", 7))
    args = SimpleNamespace(repo=str(tmp_path), pr="7", post=True, github_token="t", diff=None, base=None, full=False)
    with pytest.raises(review.ReviewTimeout) as exc:
        review.run_findings_review(args)
    assert posted == []
    assert exc.value.code == "review_timeout" and isinstance(exc.value, review.ReviewError)
    assert str(exc.value).startswith("codna review ran out of time on this diff: 1023 changed line(s)")
    assert exc.value.details["cause_code"] == "agent_core_run_failed" and exc.value.details["budget_ms"] == 647_050
    # and the CLI serialises it under that code (cli.main -> _structured_error reads .code/.details)
    from codna.cli import _structured_error
    assert _structured_error(exc.value)["error"]["code"] == "review_timeout"


def test_the_check_run_summary_for_a_timeout_names_the_way_forward_and_is_retried():
    """The webhook's check-run text comes from the CLI's structured error; a timeout must read as one
    sentence with the size, the budget and the two options -- and stay retryable, so the queue's
    retry runs with the grown budget. Conclusion is `failure` like every other review failure
    (process_job: non-ok -> failure), never a neutral that a merge queue would let through."""
    from codna import review_findings as rf
    from codna.webhook_summaries import _error_json_summary, _summary_is_retryable

    err = {"error": {"code": "review_timeout",
                     "message": rf._review_timeout_message(1023, 5, 12_000, 647_050, 650.0),
                     "details": {"cause_code": "agent_core_run_failed", "terminal_state": "runtime_invalid",
                                 "error": "turn exceeded timeout budget of 647050ms"}}}
    summary = _error_json_summary("review", None, json.dumps(err, indent=2))
    assert summary.startswith("codna review failed (review_timeout): codna review ran out of time on this diff: 1023 changed line(s) across 5 file(s)")
    assert "granted 647 s" in summary and "`@codna review`" in summary and "split the pull request" in summary
    assert "terminal_state: runtime_invalid" in summary and "approved" not in summary.lower()
    assert _summary_is_retryable(summary) is True


def test_diff_changed_lines_counts_body_lines_only():
    from codna.review_findings import diff_changed_lines
    diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,4 @@\n context\n-old\n+new\n+added\n"
            "diff --git a/b.py b/b.py\n--- /dev/null\n+++ b/b.py\n@@ -0,0 +1,2 @@\n+x\n+y\n")
    assert diff_changed_lines(diff) == 5
    assert diff_changed_lines("") == 0


def test_run_diff_review_hands_the_diffs_size_to_the_agent(tmp_path, monkeypatch):
    import subprocess

    from codna import review_findings as rf

    def _git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    _git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "."), _git("-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "base")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("w = 4\n", encoding="utf-8")
    _git("add", ".")
    seen = {}

    def fake_agent(repo_dir, prompt, **kw):
        seen.update(kw)
        return [], {"model": "m"}

    monkeypatch.setattr(rf, "run_review_agent", fake_agent)
    rf.run_diff_review(str(tmp_path), repository="local", base="HEAD", config=rf.ReviewConfig())
    assert seen["changed_lines"] == 3 and seen["changed_files"] == 2
    assert seen["timeout_s"] is None                                  # adaptive unless a caller pins it


def test_webhook_job_env_forwards_the_history_dir(monkeypatch, tmp_path):
    """The webhook runs each review in a scrubbed subprocess with a per-job CODNA_RUNTIME_ROOT; the
    window must reach it through the allowlist or every job would start cold."""
    from codna import webhook_worker
    monkeypatch.setenv("CODNA_REVIEW_HISTORY_DIR", "/data/review-budget")
    env = webhook_worker._job_env("tok", "ekey", tmp=str(tmp_path))
    assert env["CODNA_REVIEW_HISTORY_DIR"] == "/data/review-budget"
    assert not env["CODNA_REVIEW_HISTORY_DIR"].startswith(env["CODNA_RUNTIME_ROOT"])


# ── follow-up: ONE deadline per review -- the repair turn shares it ────────────────────────────
def test_the_deadline_is_the_cap_and_two_turns_plus_slack_fit_the_job_bound():
    """Two independent adaptive budgets could total 2 x cap = 2400 s > the 1800 s job bound, whose
    expiry kills the process group and reports a generic job timeout. The review turn and the repair
    turn now share ONE deadline, so the worst case is deadline + two sidecar waits (+30 s each,
    packaged_agent_runner._run_sidecar) -- and that must fit the job bound with room for the clone and
    the post."""
    assert rb.REVIEW_DEADLINE_MS == rb.REVIEW_CAP_MS
    assert rb.REPAIR_MIN_MS >= 30_000                                     # never below the pin path's floor
    fly = (REPO / "infra" / "fly" / "codna-webhook.fly.toml").read_text(encoding="utf-8")
    fly_job_s = int(next(ln for ln in fly.splitlines() if ln.strip().startswith("CODNA_WEBHOOK_JOB_TIMEOUT_S")).split('"')[1])
    assert rb.REVIEW_DEADLINE_MS / 1000 + 2 * 30 < fly_job_s
    from codna import webhook_worker
    assert rb.REVIEW_DEADLINE_MS / 1000 + 2 * 30 < webhook_worker._JOB_TIMEOUT_S


@pytest.mark.parametrize("first_budget_ms, elapsed_s, expected_s", [
    (647_050, 20.0, 647),          # a tiny first turn: the repair gets the first turn's own budget
    (1_200_000, 1_100.0, 100),     # the first turn consumed most of the cap: the repair gets the remainder
    (1_200_000, 1_139.0, 61),      # just above the minimum worth starting
    (None, 300.0, 900),            # no remembered decision and no pin (a stubbed runner): the remainder
    (45_000, 10.0, 45),            # a caller's pin: the repair repeats it
])
def test_the_repair_turn_is_pinned_to_the_deadlines_remainder(first_budget_ms, elapsed_s, expected_s, capsys):
    got = rb.repair_turn_budget_s(first_budget_ms=first_budget_ms, elapsed_s=elapsed_s)
    assert got == expected_s
    assert elapsed_s * 1000 + got * 1000 <= rb.REVIEW_DEADLINE_MS          # the two turns fit the deadline
    logged = json.loads(next(ln for ln in capsys.readouterr().err.splitlines() if '"review_repair_turn_budget"' in ln))
    assert logged["repair_budget_ms"] == got * 1000 and logged["repair_skipped"] is False
    assert logged["deadline_ms"] == rb.REVIEW_DEADLINE_MS and logged["first_budget_ms"] == first_budget_ms


@pytest.mark.parametrize("elapsed_s", [1_141.0, 1_170.0, 1_200.0, 1_500.0])
def test_no_headroom_means_no_repair(elapsed_s, capsys):
    assert rb.repair_turn_budget_s(first_budget_ms=1_200_000, elapsed_s=elapsed_s) is None
    logged = json.loads(next(ln for ln in capsys.readouterr().err.splitlines() if '"review_repair_turn_budget"' in ln))
    assert logged["repair_skipped"] is True and logged["repair_budget_ms"] is None


class _Reply:
    status = terminal_state = "succeeded"
    files_changed, patch_diff, telemetry, artifacts, runtime = [], "", {"model": "m"}, {}, {}

    def __init__(self, text: str, run_id: str = "run_1"):
        self.text, self.agent_run_id, self.session_id = text, run_id, "sess"


def _scripted_turns(monkeypatch, script, *, decision=None):
    """A sidecar whose turns take scripted wall-clock (review_findings measures each with
    time.monotonic(); the clock here advances without sleeping) and return scripted replies. Every
    adaptive (unpinned) turn remembers ``decision`` the way _sidecar_payload does for real."""
    import types

    from codna import review_findings as rf

    now = [1_000.0]
    monkeypatch.setattr(rf, "time", types.SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0]))
    replies = iter(script)

    def _turn(request):
        if decision is not None and "review_timeout_s" not in request.signals:
            rb.remember_decision(request.snapshot_id, decision)
        text, seconds = next(replies)
        now[0] += seconds
        return _Reply(text)

    return _stub_sidecar(monkeypatch, _turn)


def test_a_long_first_turn_pins_the_repair_to_what_is_left_of_the_deadline(monkeypatch, tmp_path):
    from codna import review_findings as rf

    at_cap = _budget(samples=_samples(3_000.0), **SDK71)                         # budget_ms == cap
    assert at_cap.budget_ms == rb.REVIEW_CAP_MS
    seen = _scripted_turns(monkeypatch, [("prose, not JSON", 1_100.0), ('{"findings": []}', 5.0)], decision=at_cap)
    raw, agent = rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=1023, changed_files=5)
    assert raw == [] and agent["repaired"] is True and len(seen) == 2
    assert "review_timeout_s" not in seen[0].signals                              # the first turn is adaptive
    assert seen[1].signals["review_timeout_s"] == 100                              # 1200 s deadline - 1100 s used
    # the repair's pin reaches the sidecar payload through the caller-pin path, exactly as a caller's would
    assert par._sidecar_payload(seen[1], Path("/tmp"))["limits"]["timeoutMs"] == 100_000
    recorded = rb.load_samples()
    assert [s.timed_out for s in recorded] == [False, False] and recorded[1].budget_ms == 100_000


def test_a_tiny_first_turn_gives_the_repair_the_first_turns_budget(monkeypatch, tmp_path):
    from codna import review_findings as rf

    sized = _budget(**SDK71)                                                       # 647 050 ms, source size
    seen = _scripted_turns(monkeypatch, [("prose", 20.0), ('{"findings": []}', 5.0)], decision=sized)
    rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=1023, changed_files=5)
    assert len(seen) == 2 and seen[1].signals["review_timeout_s"] == 647           # min(first budget, remainder)


def test_a_callers_pin_is_repeated_by_the_repair(monkeypatch, tmp_path):
    from codna import review_findings as rf

    seen = _scripted_turns(monkeypatch, [("prose", 3.0), ('{"findings": []}', 2.0)])
    rf.run_review_agent(str(tmp_path), "Review ...", timeout_s=45, changed_lines=10, changed_files=1)
    assert [r.signals["review_timeout_s"] for r in seen] == [45, 45]


def test_no_headroom_skips_the_repair_and_fails_closed_as_review_timeout(monkeypatch, tmp_path, capsys):
    from codna import review_findings as rf
    from codna.cline_agent import ReviewTurnTimeout

    at_cap = _budget(samples=_samples(3_000.0), **SDK71)
    seen = _scripted_turns(monkeypatch, [("prose, not JSON", 1_170.0), ('{"findings": []}', 5.0)], decision=at_cap)
    with pytest.raises(ReviewTurnTimeout) as exc:
        rf.run_review_agent(str(tmp_path), "Review ...", changed_lines=1023, changed_files=5)
    assert len(seen) == 1, "30 s left of the deadline: no repair turn is started"
    err = exc.value
    assert err.code == "review_timeout"
    assert err.details["repair_skipped"] is True and err.details["cause_code"] == "review_unparseable_output"
    assert err.details["budget_ms"] == rb.REVIEW_CAP_MS and err.details["deadline_ms"] == rb.REVIEW_DEADLINE_MS
    assert err.details["elapsed_s"] == 1170.0 and err.details["first_attempt"]["raw_head"].startswith("prose")
    msg = str(err)
    assert "1023 changed line(s) across 5 file(s)" in msg and "granted 1200 s" in msg
    assert "prose instead of findings JSON" in msg and "1200 s deadline" in msg and "No review was posted" in msg
    assert "`@codna review`" in msg and "split the pull request" in msg
    # the first turn COMPLETED: it is a real duration sample, not a censored one
    recorded = rb.load_samples()
    assert len(recorded) == 1 and recorded[0].timed_out is False and recorded[0].duration_s == 1170.0
    logged = json.loads(next(ln for ln in capsys.readouterr().err.splitlines() if '"review_repair_turn_budget"' in ln))
    assert logged["repair_skipped"] is True and logged["remaining_ms"] == 30_000


def test_the_skipped_repair_reaches_the_check_run_as_review_timeout(monkeypatch, tmp_path):
    """review.py wraps ReviewTurnTimeout as ReviewTimeout (code review_timeout) regardless of which
    path raised it, so the check run reads the same classification and the queue retries it."""
    from codna import review
    from codna.cline_agent import ReviewTurnTimeout
    from codna.webhook_summaries import _NON_RETRYABLE_CODES

    exc = ReviewTurnTimeout("codna review ran out of time on this diff: ...",
                            {"cause_code": "review_unparseable_output", "repair_skipped": True})
    wrapped = review._wrapped_details(exc)
    assert wrapped["cause_code"] == "review_unparseable_output" and wrapped["repair_skipped"] is True
    assert review.ReviewTimeout.code == "review_timeout" and "review_timeout" not in _NON_RETRYABLE_CODES


# ── follow-up: the predictor that ANSWERED, not the one configured ─────────────────────────────
class _FlakyKernel(_AnalyticKernel):
    """Raises on its first probe -- as the kernel does in the webhook image, where the `apps` package
    is absent, or while the daemon warms up -- and would answer afterwards. A sticky decision never
    comes back to ask."""

    def __init__(self, fail_first: int = 1):
        super().__init__()
        self.failures_left = fail_first
        self.raised = 0

    def _request(self, method, path, json=None):
        if self.failures_left > 0:
            self.failures_left -= 1
            self.raised += 1
            raise ModuleNotFoundError("No module named 'apps'")
        return super()._request(method, path, json)


def _pure_python_fallback(monkeypatch):
    """The fallback branch drives `_numpy_breach_probability`; run it on the pure-Python path, seeded.
    Same reason the file's earlier fallback tests do: in the full suite an earlier test has already
    put the engine's own site-packages on sys.path (`_ensure_decision_engine_imports`), after which
    numpy's lazy attribute loading re-imports a SECOND numpy and fails with "cannot load module more
    than once per process". Identical sample durations make the bootstrap deterministic anyway."""
    monkeypatch.setattr(ac, "_np", None)
    random.seed(7)


def test_the_log_names_the_predictor_that_answered_and_the_one_configured(monkeypatch):
    _pure_python_fallback(monkeypatch)
    monkeypatch.delenv("CODNA_ADMISSION_MC", raising=False)
    kernel = _AnalyticKernel()
    d = _budget(samples=_samples(400.0), breach_fn=_kernel_fn(kernel), **SDK71)
    assert d.predictor == "mojo" and d.predictor_configured == "mojo" and len(kernel.calls) >= 2
    d = _budget(samples=_samples(400.0), breach_fn=_kernel_fn(_FlakyKernel()), **SDK71)
    assert d.predictor == "numpy" and d.predictor_configured == "mojo"      # the hosted truth, finally visible
    assert d.to_log()["predictor"] == "numpy" and d.to_log()["predictor_configured"] == "mojo"
    d = _budget(samples=_samples(400.0), breach_fn=ac._numpy_breach_probability, **SDK71)
    assert d.predictor == "numpy" and d.source == "mc"
    assert _budget(samples=[], **SDK71).predictor is None                   # no forecast ran
    monkeypatch.setenv("ALGENTA_AGENT_TURN_TIMEOUT_MS", "300000")
    d = _budget(samples=_samples(400.0), **SDK71)
    assert d.source == "env" and d.predictor is None and d.predictor_configured == "mojo"
    monkeypatch.delenv("ALGENTA_AGENT_TURN_TIMEOUT_MS")
    monkeypatch.setenv("CODNA_ADMISSION_MC", "numpy")
    d = _budget(samples=_samples(400.0), breach_fn=_kernel_fn(kernel), **SDK71)
    assert d.predictor == "mojo" and d.predictor_configured == "numpy"      # an injected kernel answered anyway


def test_a_fallback_is_sticky_for_the_rest_of_the_decision(monkeypatch):
    """A kernel that failed once is not retried by the remaining bisection probes: one that TIMES OUT
    instead of failing fast would otherwise cost one timeout per probe, up to ~8 per decision."""
    _pure_python_fallback(monkeypatch)
    kernel = _FlakyKernel(fail_first=1)
    d = _budget(samples=_samples(400.0), breach_fn=_kernel_fn(kernel), **SDK71)
    assert d.predictor == "numpy" and d.source == "mc"
    assert kernel.raised == 1 and kernel.calls == [], "the kernel was probed exactly once, then left alone"
    # ...and the fallback's verdict is the pure predictor's own (deterministic here: no numpy, seeded)
    d_numpy = _budget(samples=_samples(400.0), breach_fn=ac._numpy_breach_probability, **SDK71)
    assert d_numpy.source == "mc" and abs(d.budget_ms - d_numpy.budget_ms) <= 2 * rb._GRID_MS
    # a healthy kernel is never abandoned: every probe answers from it
    healthy = _AnalyticKernel()
    assert _budget(samples=_samples(400.0), breach_fn=_kernel_fn(healthy), **SDK71).predictor == "mojo"
    assert len(healthy.calls) >= 2


# ── follow-up: the size signal counts every changed line, headers-lookalikes included ──────────
def test_diff_changed_lines_keeps_body_lines_that_look_like_file_headers():
    """A removed SQL/Lua/Haskell comment `-- x` renders as `--- x` and an added `++ x` as `+++ x`; both
    are changed lines the prefix test used to drop. Counted from the hunk ranges instead."""
    from codna.review_findings import diff_changed_lines

    diff = (
        "diff --git a/q.sql b/q.sql\nindex 1..2 100644\n--- a/q.sql\n+++ b/q.sql\n"
        "@@ -1,4 +1,4 @@\n"
        " SELECT 1;\n"
        "--- old comment\n"            # removed: "-- old comment"
        "--- another\n"                # removed: "-- another"
        "+-- new comment\n"            # added
        "+++ weird\n"                  # added: "++ weird"
        " FROM t;\n"
        "diff --git a/h.hs b/h.hs\n--- a/h.hs\n+++ b/h.hs\n"
        "@@ -1 +1 @@\n"                 # no counts: one line each side
        "-a\n"
        "+b\n"
        "\\ No newline at end of file\n"
    )
    assert diff_changed_lines(diff) == 6
    # a plain `diff -u` concatenation (no `diff --git` lines): the second file's headers follow the
    # first hunk's body directly and are still not counted
    plain = ("--- a.txt\n+++ b.txt\n@@ -1,2 +1,2 @@\n-x\n+y\n z\n"
             "--- c.txt\n+++ d.txt\n@@ -1 +1 @@\n-- dash\n+++ plus\n")
    assert diff_changed_lines(plain) == 4
    assert diff_changed_lines("diff --git a/x b/x\n--- a/x\n+++ b/x\n") == 0      # headers only, no hunk


def test_diff_changed_lines_agrees_with_git_numstat_on_header_lookalikes(tmp_path):
    """Against real `git diff` output: the count is added + deleted from `git diff --numstat`."""
    import subprocess

    from codna import review_findings as rf

    def _git(*a, capture=False):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True).stdout

    _git("init", "-q")
    (tmp_path / "q.sql").write_text("SELECT 1;\n-- old comment\n-- another\nFROM t;\n", encoding="utf-8")
    (tmp_path / "h.hs").write_text("a = [1]\n", encoding="utf-8")
    _git("add", "."), _git("-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "base")
    (tmp_path / "q.sql").write_text("SELECT 1;\n-- new comment\n++ weird\nFROM t;\n", encoding="utf-8")
    (tmp_path / "h.hs").write_text("a = [1]\n  ++ [2]", encoding="utf-8")            # no trailing newline
    _git("add", ".")
    diff_text, changed = rf.compute_diff(str(tmp_path), base="HEAD")
    assert sorted(changed) == ["h.hs", "q.sql"]
    numstat = sum(int(a) + int(d) for a, d, _ in (ln.split("\t") for ln in _git("diff", "--numstat", "HEAD").splitlines()))
    assert rf.diff_changed_lines(diff_text) == numstat == 5                   # q.sql 2+2, h.hs 1+0
    assert "\n--- old comment\n" in diff_text and "\n+++ weird\n" in diff_text  # the shapes the prefix test dropped


# ── follow-up: concurrent recorders share one history safely ───────────────────────────────────
def test_concurrent_recorders_never_tear_or_lose_the_window(tmp_path):
    """On the webhook every job is its own process appending to one shared history; two unlocked
    trims (read the whole file, rewrite it) could drop each other's fresh sample or leave a torn
    file. Append and trim run under one exclusive lock and the trim replaces the file atomically."""
    import threading

    path = tmp_path / "review-turns.jsonl"
    window, writers, each = 10, 8, 40

    def _record(i: int) -> None:
        for k in range(each):
            rb.record_sample(rb.ReviewSample(duration_s=1.0 + i + k / 1000, changed_lines=k, changed_files=1,
                                             prompt_tokens=10), path, window=window)

    threads = [threading.Thread(target=_record, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert window <= len(lines) <= 2 * window, len(lines)
    for line in lines:
        json.loads(line)                                                     # never torn
    assert len(rb.load_samples(path, window=window)) == window
    assert not list(tmp_path.glob("*.tmp")), "the trim's temp file is replaced, never left behind"
    if rb.fcntl is not None:
        assert (tmp_path / "review-turns.jsonl.lock").exists()

