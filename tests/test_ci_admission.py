"""Fleet-wide CI admission (`codna.ci_admission`).

The gate's job is to keep a runner fleet inside a share of the machine its owner chose. Two
properties matter more than any individual rule, and both are pinned here because violating either
makes the gate worse than having no gate:

  * it FAILS OPEN — a broken gate admits rather than wedging CI;
  * it NEVER ADMITS 0 — no combination of settings can deadlock the fleet.

Every test injects its own machine probe and state dir, so nothing here depends on the real load of
the box running the suite (which is exactly the value the gate reads in production).
"""
from __future__ import annotations

import itertools
import json
import os
import threading

import pytest

import codna.ci_admission as ci


def _machine(cores: int = 16, load1: float | None = 0.0, free_mem_gb: float | None = 64.0):
    return ci.MachineProbe(cores=cores, load1=load1, free_mem_gb=free_mem_gb)


def _budget(tmp_path, **kw) -> ci.CIBudget:
    base = {
        "state_dir": tmp_path / "ci-admission",
        "load_headroom": 99.0,      # off unless a test is about load
        "min_free_mem_gb": 0.0,     # off unless a test is about memory
        "max_wait_s": 0.0,
        "poll_s": 0.5,
    }
    base.update(kw)
    return ci.CIBudget(**base)


# ── the capacity budget: the whole point of the module ─────────────────────────────────────────────
def test_ceiling_scales_with_the_user_chosen_share():
    m = _machine(cores=16)
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.75), m) == 12
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.5), m) == 8
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.25), m) == 4


def test_share_and_reserve_are_both_honoured_and_the_stricter_wins():
    """They are two ways to say "leave me some machine". Each is an explicit request for headroom, so
    honouring only the looser one would silently ignore something the operator asked for."""
    m = _machine(cores=16)
    # reserve is stricter (16-12=4) than share (12) -> 4
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.75, reserve_cores=12), m) == 4
    # share is stricter (8) than reserve (16-2=14) -> 8
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.5, reserve_cores=2), m) == 8


def test_explicit_max_jobs_overrides_the_derivation():
    """An operator who measured their own fleet beats a heuristic derived from core count."""
    m = _machine(cores=16)
    assert ci.derive_ceiling(ci.CIBudget(cpu_share=0.9, max_jobs=3), m) == 3


@pytest.mark.parametrize(
    "kw",
    [{"cpu_share": 0.01}, {"reserve_cores": 999}, {"max_jobs": 0}, {"max_jobs": -5}],
    ids=["tiny-share", "reserve-everything", "zero-jobs", "negative-jobs"],
)
def test_ceiling_never_reaches_zero(kw):
    """NEVER ADMIT 0. A settings mistake must degrade to "one job at a time", never to a deadlocked
    fleet where nothing is admitted and every job burns its full wait budget."""
    assert ci.derive_ceiling(ci.CIBudget(**kw), _machine(cores=2)) >= 1


def test_a_nonsensical_share_falls_back_instead_of_deadlocking(monkeypatch):
    for bad in ("0", "-1", "1.5", "not-a-number", ""):
        monkeypatch.setenv(ci.ENV_CPU_SHARE, bad)
        assert ci.CIBudget.from_env().cpu_share == ci.DEFAULT_CPU_SHARE


# ── live pressure is measured against the budget, not a fixed per-core constant ─────────────────────
def test_load_limit_is_relative_to_the_chosen_share_not_a_fixed_per_core_value(tmp_path):
    """REGRESSION. The first cut blocked when loadavg/cores > 1.0, which ignored `cpu_share`
    entirely: the operator's knob had no effect on the live-pressure check, and a machine in ordinary
    use (load 17 of 16 cores) was treated the same whether the fleet was allowed 25% or 100% of it.

    Load 9 on 16 cores is the discriminating case: under the old fixed rule per-core is 0.56 and BOTH
    admit; under the budget-relative rule a 0.5 share (limit 8) refuses while a 1.0 share (limit 16)
    admits. Asserting both directions is what makes this fail if the share is dropped again.
    """
    m = _machine(cores=16, load1=9.0)
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()

    # Evaluated with one job already running, since over-budget load caps the ceiling at the live
    # count rather than vetoing outright (an idle fleet is always allowed one job).
    tight = ci.CIBudget(cpu_share=0.5, load_headroom=1.0, min_free_mem_gb=0.0)
    admit, reason = ci._capacity_check(tight, m, ci.derive_ceiling(tight, m), store, "j")(1)
    assert admit is False and "over its CI budget" in reason

    loose = ci.CIBudget(cpu_share=1.0, load_headroom=1.0, min_free_mem_gb=0.0)
    admit, _ = ci._capacity_check(loose, m, ci.derive_ceiling(loose, m), store, "j")(1)
    assert admit is True


def test_unknown_load_or_memory_never_blocks(tmp_path):
    """A probe that cannot read the machine must not be read as "machine is full"."""
    m = _machine(cores=8, load1=None, free_mem_gb=None)
    b = ci.CIBudget(cpu_share=0.5, load_headroom=0.0001, min_free_mem_gb=10_000.0)
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    admit, _ = ci._capacity_check(b, m, ci.derive_ceiling(b, m), store, "j")(0)
    assert admit is True


def test_low_free_memory_stops_the_fleet_growing(tmp_path):
    m = _machine(cores=16, load1=0.0, free_mem_gb=2.0)
    b = ci.CIBudget(load_headroom=99.0, min_free_mem_gb=8.0)
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    admit, reason = ci._capacity_check(b, m, ci.derive_ceiling(b, m), store, "j")(1)
    assert admit is False and "free memory" in reason


def test_an_idle_fleet_is_admitted_even_when_the_machine_is_overloaded(tmp_path):
    """REGRESSION. Live pressure used to VETO, which read well and behaved badly: a workstation sits
    above its CI budget most of the day, so every job refused, burned its entire wait budget, and was
    admitted anyway by the fail-open — pure latency, no limiting. Pressure now caps the ceiling at the
    live count with a floor of 1, so an idle fleet always makes immediate progress."""
    m = _machine(cores=16, load1=500.0, free_mem_gb=0.1)     # absurdly overloaded
    b = ci.CIBudget(cpu_share=0.5, load_headroom=1.0, min_free_mem_gb=8.0)
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    admit, _ = ci._capacity_check(b, m, ci.derive_ceiling(b, m), store, "j")(0)
    assert admit is True, "an empty fleet must never be blocked by machine pressure alone"


# ── the fleet's occupancy: leases ──────────────────────────────────────────────────────────────────
def _each_call_a_new_runner(monkeypatch) -> None:
    """Model a fleet: successive acquires come from DIFFERENT runners.

    Leases are one-per-owner, so without this a loop in one test process would keep replacing its own
    lease and the fleet would never appear to fill up.
    """
    counter = itertools.count()
    monkeypatch.setattr(ci, "owner_identity", lambda: (f"runner-{next(counter)}", "runner"))


def _each_thread_a_new_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        ci, "owner_identity",
        lambda: (f"runner-{threading.current_thread().name}", "runner"),
    )


def _no_breach_risk(monkeypatch) -> None:
    """Deterministic predictor: keeps the MC zone open so a test can reach the hard ceiling. Without
    it these tests would depend on the mojo kernel / numpy RNG and on this machine's real load."""
    monkeypatch.setattr(ci, "_MC_FN", lambda durations, contention, sla, n: 0.0)


def _seed_history(budget: ci.CIBudget, job_class: str = "seed", n: int = 5) -> None:
    store = ci.LeaseStore(budget.state_dir)
    store.ensure_dirs()
    for i in range(n):
        store.record_duration(job_class, 60.0 + i)


def test_ceiling_is_enforced_across_the_fleet(tmp_path, monkeypatch):
    """Occupancy is the lease directory, not an in-process counter — that is what makes the limit
    fleet-wide instead of per-repo. Four leases against a ceiling of 4 must refuse the fifth."""
    monkeypatch.setattr(ci, "probe_machine", lambda: _machine(cores=16))
    _each_call_a_new_runner(monkeypatch)
    _no_breach_risk(monkeypatch)
    b = _budget(tmp_path, max_jobs=4)
    _seed_history(b)
    leases = []
    for i in range(4):
        lease, verdict = ci.acquire(f"job-{i}", {}, b)
        assert lease is not None, verdict.reason
        leases.append(lease)

    lease, verdict = ci.acquire("job-overflow", {}, b)
    assert lease is None
    assert verdict.admit is True                       # bounded wait expired -> fail open
    assert "capacity ceiling" in verdict.reason
    assert "wait budget expired" in verdict.reason

    store = ci.LeaseStore(b.state_dir)
    assert len(store.active()) == 4
    for lease in leases:
        store.release(lease)
    assert store.active() == []


def test_acquire_is_atomic_under_concurrent_runners(tmp_path, monkeypatch):
    """The decision and the lease write happen under ONE flock. A peek followed by an unlocked write
    is the classic overshoot where every runner sees the same last free slot."""
    monkeypatch.setattr(ci, "probe_machine", lambda: _machine(cores=16))
    _no_breach_risk(monkeypatch)
    _each_thread_a_new_runner(monkeypatch)
    ceiling = 3
    b = _budget(tmp_path, max_jobs=ceiling)
    _seed_history(b)
    granted, lock = [], threading.Lock()
    start = threading.Barrier(16)

    def _worker(i: int) -> None:
        start.wait()
        lease, _ = ci.acquire(f"class-{i}", {}, b)
        if lease is not None:
            with lock:
                granted.append(lease)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == ceiling, f"granted {len(granted)} leases against ceiling {ceiling}"
    assert len(ci.LeaseStore(b.state_dir).active()) == ceiling


def test_a_runner_lease_outlives_the_process_that_acquired_it(tmp_path, monkeypatch):
    """REGRESSION, and the bug that made the whole gate count nothing.

    Leases were owned by the acquiring PID. But `codna ci admit` is a short-lived CLI call that exits
    immediately, long before the job it admitted runs — so the very next `reap()` saw a dead PID and
    freed the slot. `codna ci status` reported 0 in flight while a job held a lease, and the ceiling
    could never bind.

    On a runner the owner is the RUNNER, and liveness is by replacement plus the age cap, never PID.
    Simulated here by writing the lease then reaping from a state where the recorded pid is long gone.
    """
    monkeypatch.setenv("RUNNER_NAME", "thyn-mac-arm64-2")
    owner, kind = ci.owner_identity()
    assert (owner, kind) == ("thyn-mac-arm64-2", "runner")

    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    lease, _, _ = store.acquire("ci/unit-tests", {}, lambda live: (True, "ok"))
    # Rewrite the recorded pid to one that is certainly not running — i.e. the admit process exited.
    rec = json.loads(lease.path.read_text())
    rec["pid"] = 2**22
    lease.path.write_text(json.dumps(rec))

    assert store.reap() == 1, "a runner-owned lease must survive its acquirer exiting"
    assert lease.path.exists()


def test_a_new_job_on_the_same_runner_replaces_its_stale_lease(tmp_path, monkeypatch):
    """A runner runs one job at a time, so a second acquire from it proves the first job ended without
    releasing (cancelled workflow, killed step). Counting both would let one crashed job hold a slot
    until the age cap."""
    monkeypatch.setenv("RUNNER_NAME", "runner-a")
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    first, _, _ = store.acquire("job-1", {}, lambda live: (True, "ok"))
    second, _, live_before = store.acquire("job-2", {}, lambda live: (True, "ok"))

    assert live_before == 0, "the runner's own previous lease must not count against it"
    assert len(store.active()) == 1
    assert not (first.path != second.path and first.path.exists())


def test_a_lease_whose_owner_died_is_reaped(tmp_path):
    """A cancelled workflow or rebooted host cannot release its own lease. Without PID reaping the
    fleet's capacity would leak away permanently, one abandoned job at a time."""
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    dead = store.leases_dir / "dead.json"
    # PID 2**22 is above every plausible pid_max, so it is reliably not running.
    dead.write_text(json.dumps({"owner": "pid-4194304", "owner_kind": "pid", "pid": 2**22,
                                "job_class": "j", "started": ci.time.time()}))
    live = store.leases_dir / "live.json"
    live.write_text(json.dumps({"owner": f"pid-{os.getpid()}", "owner_kind": "pid",
                                "pid": os.getpid(), "job_class": "j", "started": ci.time.time()}))

    assert store.reap() == 1
    assert not dead.exists()
    assert live.exists()


def test_an_expired_lease_is_reaped_even_if_its_pid_was_recycled(tmp_path):
    """The age cap is the backstop for a PID that an unrelated process later reused."""
    store = ci.LeaseStore(tmp_path / "s", lease_max_age_s=1.0)
    store.ensure_dirs()
    stale = store.leases_dir / "stale.json"
    stale.write_text(json.dumps({"owner": "runner-x", "owner_kind": "runner", "pid": os.getpid(),
                                 "job_class": "j", "started": ci.time.time() - 3600}))
    assert store.reap() == 0
    assert not stale.exists()


def test_an_unreadable_lease_is_cleared_not_counted(tmp_path):
    """Garbage is not evidence of a running job; counting it would shrink capacity forever."""
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    junk = store.leases_dir / "junk.json"
    junk.write_text("{not json")
    assert store.reap() == 0
    assert not junk.exists()


# ── durations: what the MC zone learns from ────────────────────────────────────────────────────────
def test_a_cold_fleet_holds_at_the_deterministic_floor(tmp_path, monkeypatch):
    """With no history anywhere, the MC zone has nothing to simulate, so the inherited policy holds at
    det_cap (= ceiling // 2). Pinned deliberately: it is the conservative branch, and the alternative
    would be admitting on a fabricated distribution."""
    monkeypatch.setattr(ci, "probe_machine", lambda: _machine(cores=16))
    _each_call_a_new_runner(monkeypatch)
    b = _budget(tmp_path, max_jobs=4)          # det_cap = 2
    admitted = [ci.acquire(f"cold-{i}", {}, b)[0] for i in range(4)]
    assert sum(1 for lease in admitted if lease is not None) == 2


def test_an_unseen_job_class_borrows_the_fleets_pooled_history(tmp_path, monkeypatch):
    """REGRESSION. Durations are per job class, and every workflow job is its own class — so keying
    ONLY on the class meant a never-before-seen job always hit the cold-start floor, holding the fleet
    at half its ceiling as the normal case rather than an edge case. An unseen class must fall back to
    the pooled history; its own history still wins once it has any."""
    monkeypatch.setattr(ci, "probe_machine", lambda: _machine(cores=16))
    _each_call_a_new_runner(monkeypatch)
    _no_breach_risk(monkeypatch)
    b = _budget(tmp_path, max_jobs=4)          # det_cap = 2
    _seed_history(b, job_class="some-other-job")

    store = ci.LeaseStore(b.state_dir)
    assert store.durations("brand-new-job") == [], "precondition: this class has no history"
    assert store.pooled_durations(), "precondition: the fleet does have history"

    admitted = [ci.acquire("brand-new-job", {}, b)[0] for _ in range(4)]
    assert sum(1 for lease in admitted if lease is not None) == 4, (
        "an unseen class must reach the ceiling via pooled history, not stall at det_cap"
    )


def test_release_records_the_duration_for_that_job_class(tmp_path):
    """Per-CLASS history is the point: a 30-second lint and a 20-minute build must not share one
    distribution, or the forecast is fitted to a bimodal blur of both."""
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    lease, _, _ = store.acquire("lint", {}, lambda live: (True, "ok"))
    store.release(lease)
    assert len(store.durations("lint")) == 1
    assert store.durations("build") == []


def test_nonpositive_durations_are_not_recorded(tmp_path):
    """A zero-length observation is not a job duration; it would drag the fitted distribution down."""
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    store.record_duration("j", 0.0)
    store.record_duration("j", -3.0)
    assert store.durations("j") == []


def test_duration_history_is_trimmed(tmp_path):
    """Append-only history on a long-lived runner would grow without bound."""
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    for i in range(ci.DURATION_WINDOW * 2 + 25):
        store.record_duration("j", float(i + 1))
    assert len(store.durations("j")) <= ci.DURATION_WINDOW


def test_job_class_cannot_escape_the_state_directory():
    """Job classes come from workflow/job names, which are attacker-influencable in a fork PR. A
    class of "../../x" must not write outside the state dir."""
    assert "/" not in ci._safe_name("../../etc/passwd")
    assert ".." not in ci._safe_name("../../etc/passwd")
    assert ci._safe_name("") == "job"


# ── platform agnostic: the fleet must be adoptable on any runner ───────────────────────────────────
def test_the_gate_imports_without_posix_only_modules():
    """REGRESSION. The lock was `fcntl.flock`, and `import fcntl` is POSIX-only — so this module
    could not even be IMPORTED on Windows and a Windows runner could not run the gate at all. The
    gate is meant to work on any platform, so it must not depend on an OS-specific module.

    Checked structurally because the suite itself runs on POSIX, where a regression would pass
    silently: importing fcntl here succeeds, so only the absence of the import proves portability.

    Parsed with `ast` rather than grepped: a substring check matches the module's own docstring
    (which names `fcntl.flock` to explain why it is gone) and so fails while the code is correct —
    the same false-positive trap as asserting on a comment.
    """
    import ast
    import inspect

    imported: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(ci))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    posix_only = imported & {"fcntl", "msvcrt", "termios", "pwd", "grp"}
    assert not posix_only, f"platform-specific dependency reintroduced: {sorted(posix_only)}"


def test_locking_survives_a_holder_that_died_mid_section(tmp_path, monkeypatch):
    """A crashed job must not wedge the fleet. An O_EXCL lock file has no OS-level owner, so nothing
    releases it automatically — an abandoned lock is stolen once it is older than the stale window.

    Asserts it happens PROMPTLY, not merely that it happens: the fail-open path would eventually
    admit anyway after `_LOCK_WAIT_S`, so a lease-only assertion passes with stealing removed
    entirely (verified by mutation). The elapsed time is what distinguishes stolen from waited-out.
    """
    monkeypatch.setattr(ci, "_LOCK_WAIT_S", 5.0)
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    store.lock_path.write_text("")                      # an abandoned lock, no owner
    old = ci.time.time() - (ci._LOCK_STALE_S + 60)
    os.utime(store.lock_path, (old, old))

    started = ci.time.monotonic()
    lease, _, _ = store.acquire("j", {}, lambda live: (True, "ok"))
    elapsed = ci.time.monotonic() - started

    assert lease is not None, "a stale lock must be stolen, not waited on forever"
    assert elapsed < 1.0, f"stale lock was waited out ({elapsed:.1f}s), not stolen"


def test_locking_never_blocks_a_build_indefinitely(tmp_path, monkeypatch):
    """If the lock cannot be taken at all, the section runs UNLOCKED rather than blocking. Briefly
    risking an over-admit is strictly better than failing a build on a lock file."""
    monkeypatch.setattr(ci, "_LOCK_WAIT_S", 0.05)
    monkeypatch.setattr(ci, "_LOCK_STALE_S", 10_000.0)   # fresh enough that it is never stolen
    store = ci.LeaseStore(tmp_path / "s")
    store.ensure_dirs()
    store.lock_path.write_text("")                       # held, and not stale

    lease, _, _ = store.acquire("j", {}, lambda live: (True, "ok"))
    assert lease is not None, "an unavailable lock must not block admission"


# ── fail open, always ──────────────────────────────────────────────────────────────────────────────
def test_admission_failure_admits_rather_than_blocking_ci(tmp_path, monkeypatch):
    """The gate is infrastructure protecting throughput; it must never become the reason a build
    cannot run. Any unexpected error admits, with the cause in the reason string."""
    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(ci, "probe_machine", _boom)
    lease, verdict = ci.acquire("j", {}, _budget(tmp_path))
    assert lease is None
    assert verdict.admit is True
    assert "admission gate unavailable" in verdict.reason
    assert "probe exploded" in verdict.reason


def test_disabling_admission_skips_the_gate_entirely(tmp_path, monkeypatch):
    monkeypatch.setenv(ci.ENV_ENABLED, "off")
    lease, verdict = ci.acquire("j", {}, _budget(tmp_path))
    assert lease is None and verdict.admit is True
    assert "admission disabled" in verdict.reason
    assert not (tmp_path / "ci-admission" / "leases").exists(), "disabled gate must not touch state"


def test_admission_is_on_by_default(monkeypatch):
    """Opt-OUT: a fleet nobody configured is still protected."""
    monkeypatch.delenv(ci.ENV_ENABLED, raising=False)
    assert ci.admission_enabled() is True


# ── one policy, not two ────────────────────────────────────────────────────────────────────────────
def test_the_mc_policy_is_reused_not_reimplemented():
    """The floor/MC-zone/ceiling policy must have exactly ONE implementation. A second copy here
    would be free to drift from the documented behaviour and from `codna fix`'s own gate."""
    import inspect

    src = inspect.getsource(ci)
    assert "decide_for" in src, "must delegate the decision to MCAdmissionDecider"
    for reimplemented in ("breach_threshold", "_mc_breach_probability", "contention ="):
        assert reimplemented not in src, f"policy appears reimplemented here: {reimplemented}"


def test_state_dir_follows_the_runtime_root(monkeypatch, tmp_path):
    """Runners on one machine share $HOME, so the state dir must move with CODNA_RUNTIME_ROOT —
    otherwise isolating a runner's runtime would silently leave its admission state shared."""
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "rt"))
    monkeypatch.delenv(ci.ENV_STATE_DIR, raising=False)
    assert ci.CIBudget.from_env().state_dir == tmp_path / "rt" / "ci-admission"
