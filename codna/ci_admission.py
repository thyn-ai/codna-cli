"""Fleet-wide CI job admission: keep a runner fleet inside a share of the machine its owner chose.

Self-hosted runners have no fleet-wide limit. GitHub's `concurrency:` is *repo*-scoped, so N repos
sharing one machine can each sit under their own cap and still oversubscribe the host together. And
the host is usually somebody's workstation: it has to stay usable while CI runs, which no per-repo
setting can express.

This gate is that missing limit, and it is deliberately **capacity**-shaped rather than repo-shaped —
nothing here knows a repo name, a language, or a workflow's contents, so it behaves the same for any
project at any company.

It reuses the MC discipline that governs `codna fix` (`admission_control.MCAdmissionDecider`) via
`decide_for()`, with a CI-shaped config: the SLA is the job's own timeout, and the durations are that
job class's observed history. What differs is where occupancy comes from — not an in-process counter,
but a directory of leases on the shared filesystem, so every runner on the box sees the same
occupancy no matter which repo it is serving.

Two hard rules, because a broken gate must never be worse than no gate at all:

* **Fail open.** Any unexpected error admits. CI never blocks because the gate itself is unhealthy.
* **Never admit 0.** The derived ceiling floors at 1, so no combination of settings can deadlock a
  fleet into admitting nothing.

Waiting here is correct, unlike waiting for a *rate limit*: a job that waits for CPU is waiting for a
resource that its own start would otherwise oversubscribe. But a waiting job still holds its runner
slot, so the wait is bounded (`CODNA_CI_MAX_WAIT_S`) and expiry admits rather than fails.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .admission_control import AdmissionConfig, MCAdmissionDecider, make_mojo_breach_fn

# ── user-adjustable budget ─────────────────────────────────────────────────────────────────────────
# The point of this module: the fleet's share of the machine is a setting, not a constant.
ENV_ENABLED = "CODNA_CI_ADMISSION"
ENV_CPU_SHARE = "CODNA_CI_CPU_SHARE"
ENV_RESERVE_CORES = "CODNA_CI_RESERVE_CORES"
ENV_MAX_JOBS = "CODNA_CI_MAX_JOBS"
ENV_MIN_FREE_MEM_GB = "CODNA_CI_MIN_FREE_MEMORY_GB"
ENV_LOAD_HEADROOM = "CODNA_CI_LOAD_HEADROOM"
ENV_JOB_SLA_S = "CODNA_CI_JOB_SLA_S"
ENV_MAX_WAIT_S = "CODNA_CI_MAX_WAIT_S"
ENV_POLL_S = "CODNA_CI_POLL_S"
ENV_STATE_DIR = "CODNA_CI_STATE_DIR"

DEFAULT_CPU_SHARE = 0.75          # leave a quarter of the box to whoever is sitting at it
DEFAULT_MIN_FREE_MEM_GB = 8.0
DEFAULT_LOAD_HEADROOM = 1.0       # per-core loadavg above this means the machine is already busy
DEFAULT_JOB_SLA_S = 3600.0
DEFAULT_MAX_WAIT_S = 120.0        # bounded: a waiting job is still holding a runner slot
DEFAULT_POLL_S = 5.0
DEFAULT_LEASE_MAX_AGE_S = 6 * 3600.0
DURATION_WINDOW = 200
_LOCK_WAIT_S = 10.0               # then run unlocked -- a lock file must never block a build
_LOCK_STALE_S = 60.0              # older than this, its holder died mid-section
_LOCK_POLL_S = 0.05


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return default if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int | None) -> int | None:
    try:
        raw = os.environ.get(name)
        return default if raw in (None, "") else int(raw)
    except (TypeError, ValueError):
        return default


def admission_enabled() -> bool:
    """Opt-out, not opt-in: the fleet is protected by default, and one variable disables it."""
    return (os.environ.get(ENV_ENABLED) or "on").strip().lower() not in {"off", "0", "false", "no"}


@dataclass
class CIBudget:
    cpu_share: float = DEFAULT_CPU_SHARE
    reserve_cores: int = 0
    max_jobs: int | None = None
    min_free_mem_gb: float = DEFAULT_MIN_FREE_MEM_GB
    load_headroom: float = DEFAULT_LOAD_HEADROOM
    sla_s: float = DEFAULT_JOB_SLA_S
    max_wait_s: float = DEFAULT_MAX_WAIT_S
    poll_s: float = DEFAULT_POLL_S
    state_dir: Path = field(default_factory=lambda: _default_state_dir())

    @classmethod
    def from_env(cls) -> CIBudget:
        share = _env_float(ENV_CPU_SHARE, DEFAULT_CPU_SHARE)
        # A share outside (0, 1] is a typo, not an intent — 0 would mean "no CI ever".
        if not 0.0 < share <= 1.0:
            share = DEFAULT_CPU_SHARE
        state = os.environ.get(ENV_STATE_DIR)
        return cls(
            cpu_share=share,
            reserve_cores=max(0, _env_int(ENV_RESERVE_CORES, 0) or 0),
            max_jobs=_env_int(ENV_MAX_JOBS, None),
            min_free_mem_gb=max(0.0, _env_float(ENV_MIN_FREE_MEM_GB, DEFAULT_MIN_FREE_MEM_GB)),
            load_headroom=max(0.1, _env_float(ENV_LOAD_HEADROOM, DEFAULT_LOAD_HEADROOM)),
            sla_s=max(1.0, _env_float(ENV_JOB_SLA_S, DEFAULT_JOB_SLA_S)),
            max_wait_s=max(0.0, _env_float(ENV_MAX_WAIT_S, DEFAULT_MAX_WAIT_S)),
            poll_s=max(0.5, _env_float(ENV_POLL_S, DEFAULT_POLL_S)),
            state_dir=Path(state).expanduser() if state else _default_state_dir(),
        )


def _default_state_dir() -> Path:
    root = os.environ.get("CODNA_RUNTIME_ROOT")
    base = Path(root).expanduser() if root else Path.home() / ".codna"
    return base / "ci-admission"


# ── machine capacity ───────────────────────────────────────────────────────────────────────────────
@dataclass
class MachineProbe:
    cores: int
    load1: float | None          # absolute 1-minute loadavg, NOT normalised per core
    free_mem_gb: float | None

    @property
    def load_per_core(self) -> float | None:
        if self.load1 is None:
            return None
        return self.load1 / max(1, self.cores)


def _free_memory_gb() -> float | None:
    """Best-effort free memory. Returns None when unknown, and an unknown never blocks a job."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page_size, free_pages = 4096, 0.0
            for line in out.splitlines():
                if "page size of" in line:
                    page_size = int(line.rsplit("of", 1)[1].strip().split()[0])
                for label in ("Pages free:", "Pages inactive:", "Pages speculative:"):
                    if line.startswith(label):
                        free_pages += float(line.split(":", 1)[1].strip().rstrip("."))
            return (free_pages * page_size) / (1024**3)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024**2)
    except Exception:
        return None
    return None


def probe_machine() -> MachineProbe:
    cores = os.cpu_count() or 1
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        load1 = None
    return MachineProbe(cores=cores, load1=load1, free_mem_gb=_free_memory_gb())


def core_budget(budget: CIBudget, machine: MachineProbe) -> float:
    """Cores the fleet is allowed to keep busy — the machine-capacity share, in absolute cores."""
    by_share = machine.cores * budget.cpu_share
    by_reserve = float(machine.cores - budget.reserve_cores) if budget.reserve_cores else by_share
    return max(1.0, min(by_share, by_reserve))


def derive_ceiling(budget: CIBudget, machine: MachineProbe) -> int:
    """How many concurrent CI jobs this machine is allowed to run.

    `cpu_share` and `reserve_cores` are two ways to say the same thing ("don't take the whole box");
    when both are set the more conservative wins, because each was an explicit request for headroom.
    An explicit `max_jobs` overrides the derivation entirely — that is what an operator who has
    measured their own fleet wants. Floors at 1: a share of a small machine must still run CI.
    """
    if budget.max_jobs is not None:
        return max(1, budget.max_jobs)
    by_share = int(machine.cores * budget.cpu_share)
    by_reserve = machine.cores - budget.reserve_cores if budget.reserve_cores else by_share
    return max(1, min(by_share, by_reserve))


# ── the fleet's occupancy: leases on the shared filesystem ─────────────────────────────────────────
@dataclass
class Lease:
    path: Path
    job_class: str
    started: float


def owner_identity() -> tuple[str, str]:
    """Who holds a lease, and how its liveness can be checked.

    This cannot be the acquiring PID. `codna ci admit` is a short-lived CLI call that exits before the
    job it admitted does any work, so PID liveness would reap every lease microseconds after it was
    written and the gate would count nothing at all.

    On a self-hosted runner the natural owner is the RUNNER: a runner executes one job at a time, so
    there is at most one live lease per runner, and a new job appearing on that runner is proof the
    previous one ended. Liveness there is by replacement plus the age cap, not by PID.

    Off a runner (a developer invoking this directly) the caller's own process is the owner and PID
    liveness is exactly right.
    """
    runner = os.environ.get("RUNNER_NAME") or os.environ.get("CODNA_CI_OWNER")
    if runner:
        return runner, "runner"
    return f"pid-{os.getpid()}", "pid"


class LeaseStore:
    """Cross-process, cross-repo occupancy for one machine.

    A lease is a file; the fleet's in-flight count is the number of live lease files. This is the
    right substrate here precisely because the contention is local: every runner shares one host, so
    no network hop, no daemon, and no service to operate is involved in the decision.

    Liveness is by PID, not by trust: a job killed mid-run (cancelled workflow, rebooted host) cannot
    release its own lease, so a lease whose PID is gone is reaped. The age cap is the backstop for a
    PID that got recycled by an unrelated process.
    """

    def __init__(self, state_dir: Path, lease_max_age_s: float = DEFAULT_LEASE_MAX_AGE_S) -> None:
        self.state_dir = state_dir
        self.leases_dir = state_dir / "leases"
        self.durations_dir = state_dir / "durations"
        self.lock_path = state_dir / "admission.lock"
        self.lease_max_age_s = lease_max_age_s

    def ensure_dirs(self) -> None:
        self.leases_dir.mkdir(parents=True, exist_ok=True)
        self.durations_dir.mkdir(parents=True, exist_ok=True)

    # ── locking ────────────────────────────────────────────────────────────────────────────────
    @contextlib.contextmanager
    def _locked(self):
        """Exclusive lock around check-and-reserve, so two runners cannot both take the last slot.

        Uses an O_EXCL lock FILE rather than `fcntl.flock`. flock is the obvious choice and is
        POSIX-only: importing it made this whole module fail to import on Windows, so a Windows
        runner could not run the gate at all. `os.open(..., O_CREAT | O_EXCL)` is atomic on every
        platform Python supports, and nothing here may assume an OS — the fleet is meant to be
        adoptable on any runner.

        Two safety valves, because a lock must never be able to wedge CI:
          * a lock older than `_LOCK_STALE_S` is stolen — its holder died mid-section;
          * if it still cannot be taken within `_LOCK_WAIT_S`, the section runs UNLOCKED. Briefly
            risking an over-admit is strictly better than blocking a build on a lock file.
        """
        self.ensure_dirs()
        deadline = time.monotonic() + _LOCK_WAIT_S
        fd = None
        while fd is None:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock_path) > _LOCK_STALE_S:
                        os.unlink(self.lock_path)
                        continue
                except OSError:
                    pass            # vanished under us — just retry
                if time.monotonic() >= deadline:
                    break           # proceed unlocked rather than block CI
                time.sleep(_LOCK_POLL_S)
            except OSError:
                break               # unwritable state dir — proceed unlocked
        try:
            yield fd
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                finally:
                    try:
                        os.unlink(self.lock_path)
                    except OSError:
                        pass

    # ── occupancy ──────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True          # exists but owned by another user
        except OSError:
            return True          # unknown: assume alive rather than free a live slot
        return True

    def reap(self, exclude_owner: str | None = None) -> int:
        """Drop dead leases and return the live count.

        `exclude_owner` drops this owner's own previous lease first. A runner that starts a new job
        while still holding a lease is proof the previous job ended without releasing (cancelled
        workflow, killed step) — so that lease is stale by definition, and counting it would let one
        crashed job permanently consume a slot until the age cap expired.
        """
        live = 0
        now = time.time()
        for path in sorted(self.leases_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text())
                started = float(rec.get("started", 0.0))
            except Exception:
                # An unreadable lease is not evidence of a running job; clear it.
                path.unlink(missing_ok=True)
                continue
            if exclude_owner is not None and rec.get("owner") == exclude_owner:
                path.unlink(missing_ok=True)
                continue
            if (now - started) > self.lease_max_age_s:
                path.unlink(missing_ok=True)
                continue
            # PID liveness only for owners whose PID actually outlives the acquiring call.
            if rec.get("owner_kind") == "pid" and not self._pid_alive(int(rec.get("pid", -1))):
                path.unlink(missing_ok=True)
                continue
            live += 1
        return live

    def active(self) -> list[dict]:
        out = []
        for path in sorted(self.leases_dir.glob("*.json")):
            try:
                out.append(json.loads(path.read_text()))
            except Exception:
                continue
        return out

    # ── durations, per job class ───────────────────────────────────────────────────────────────
    def _durations_path(self, job_class: str) -> Path:
        return self.durations_dir / f"{_safe_name(job_class)}.jsonl"

    def durations(self, job_class: str) -> list[float]:
        path = self._durations_path(job_class)
        if not path.exists():
            return []
        try:
            lines = path.read_text().splitlines()[-DURATION_WINDOW:]
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                value = float(json.loads(line)["duration_s"])
            except Exception:
                continue
            if value > 0:
                out.append(value)
        return out

    def pooled_durations(self) -> list[float]:
        """Every class's history combined — the prior for a class this fleet has not seen before.

        Without this, a cold class holds at the deterministic floor, and since *every workflow job is
        its own class* that is the normal case, not an edge case: the fleet would sit at half its
        configured ceiling more or less permanently. Pooling is defensible because the resource being
        forecast is the shared machine, so any class's observed stretch under contention is evidence
        about this one. A class's OWN history always wins once it has any.
        """
        out: list[float] = []
        for path in sorted(self.durations_dir.glob("*.jsonl")):
            out.extend(self.durations(path.stem))
        return out[-DURATION_WINDOW:]

    def record_duration(self, job_class: str, duration_s: float) -> None:
        """Append an observed job duration so the predictor starts warm on the next run.

        Kept append-only and trimmed on write: this is the history the MC zone learns from, and a job
        that never records one leaves its class permanently at the deterministic floor.
        """
        if duration_s <= 0:
            return
        self.ensure_dirs()
        path = self._durations_path(job_class)
        try:
            # 6dp, not 3: rounding a sub-millisecond duration to 3 writes 0.0, which `durations()`
            # then filters out as non-positive — silently dropping a sample the store had accepted.
            with open(path, "a") as fh:
                fh.write(json.dumps({"duration_s": round(float(duration_s), 6), "at": time.time()}) + "\n")
            lines = path.read_text().splitlines()
            if len(lines) > DURATION_WINDOW * 2:
                path.write_text("\n".join(lines[-DURATION_WINDOW:]) + "\n")
        except OSError:
            pass

    # ── acquire / release ──────────────────────────────────────────────────────────────────────
    def acquire(self, job_class: str, meta: dict, ceiling_check) -> tuple[Lease | None, str, int]:
        """Atomically reserve a slot if `ceiling_check(live_count)` allows it.

        The decision and the write happen under one flock, which is the whole point: a peek followed
        by an unlocked write is the classic overshoot where every runner sees the last free slot.
        """
        owner, owner_kind = owner_identity()
        with self._locked():
            live = self.reap(exclude_owner=owner)
            admit, reason = ceiling_check(live)
            if not admit:
                return None, reason, live
            started = time.time()
            # One lease file per OWNER, not per call: a runner holds at most one, so a re-acquire
            # overwrites rather than double-counting the same runner.
            path = self.leases_dir / f"{_safe_name(owner)}.json"
            record = {"owner": owner, "owner_kind": owner_kind, "pid": os.getpid(),
                      "job_class": job_class, "started": started, **meta}
            path.write_text(json.dumps(record))
            return Lease(path=path, job_class=job_class, started=started), reason, live

    def release(self, lease: Lease, record_duration: bool = True) -> float:
        duration = max(0.0, time.time() - lease.started)
        lease.path.unlink(missing_ok=True)
        if record_duration:
            self.record_duration(lease.job_class, duration)
        return duration


def _safe_name(raw: str) -> str:
    """Job classes derive from workflow/job names, which a fork PR can influence, so this is a
    filename sanitiser and not merely a tidier. Dots collapse too: a bare `..` would name the parent
    directory, and while every current caller appends a suffix, relying on that is a trap for the next
    one to add a path built from a class name.
    """
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (raw or "job")]
    return "".join(keep).strip("-")[:120] or "job"


# ── the verdict ────────────────────────────────────────────────────────────────────────────────────
@dataclass
class CIVerdict:
    admit: bool
    reason: str
    in_flight: int
    ceiling: int
    waited_s: float = 0.0


_MC_FN = None


def _breach_fn():
    """One predictor per process, built lazily.

    Two reasons it is not per-call: `make_mojo_breach_fn` announces a numpy fallback exactly once per
    predictor, so a fresh one on every poll would re-announce every few seconds; and the mojo path
    imports the engine, which is far too expensive to repeat inside a polling loop.
    """
    global _MC_FN
    if _MC_FN is None:
        _MC_FN = make_mojo_breach_fn()
    return _MC_FN


def _decider(budget: CIBudget, ceiling: int) -> MCAdmissionDecider:
    """A CI-shaped AdmissionConfig over the SAME decision policy `codna fix` uses.

    `model_rate_budget` is the deterministic floor. For CI the analogous "clearly safe" region is
    half the capacity budget: below that the machine is not contended enough for a forecast to change
    the answer, so no simulation is run at all.
    """
    cfg = AdmissionConfig(
        ceiling=ceiling,
        model_rate_budget=max(1, ceiling // 2),
        sla_s=budget.sla_s,
    )
    return MCAdmissionDecider(cfg, mc_fn=_breach_fn())


def evaluate(job_class: str, budget: CIBudget | None = None) -> CIVerdict:
    """Non-reserving peek — what would happen if this job asked for a slot right now."""
    budget = budget or CIBudget.from_env()
    machine = probe_machine()
    ceiling = derive_ceiling(budget, machine)
    store = LeaseStore(budget.state_dir)
    store.ensure_dirs()
    live = store.reap()
    admit, reason = _capacity_check(budget, machine, ceiling, store, job_class)(live)
    return CIVerdict(admit=admit, reason=reason, in_flight=live, ceiling=ceiling)


def _capacity_check(budget: CIBudget, machine: MachineProbe, ceiling: int,
                    store: LeaseStore, job_class: str):
    """Compose the checks into one predicate over the fleet's live count.

    Order matters: the cheap absolute limits answer first, and the MC forecast — the only one that
    can consult the engine — runs only when the cheap ones did not already decide.
    """
    def _check(live: int) -> tuple[bool, str]:
        # Live pressure from work this gate does not own — the owner's own builds, editors, agents —
        # measured against the fleet's CORE BUDGET rather than a fixed per-core 1.0, so the operator's
        # share actually governs it.
        #
        # It SHRINKS the ceiling instead of vetoing. A hard veto looks right and behaves badly: a
        # developer machine sits above its CI budget most of the day, so every job would refuse, burn
        # its whole wait budget, and then be admitted anyway by the fail-open — pure added latency,
        # zero limiting. Capping at "what is already running, but never below one" stops the fleet
        # GROWING while the machine is loaded, yet always lets an idle fleet make progress at once.
        effective, pressure = ceiling, ""
        load_limit = core_budget(budget, machine) * budget.load_headroom
        if machine.load1 is not None and machine.load1 > load_limit:
            effective = max(1, min(effective, live))
            pressure = (f"machine over its CI budget (load {machine.load1:.1f} > "
                        f"{load_limit:.1f} of {machine.cores} cores)")
        if machine.free_mem_gb is not None and machine.free_mem_gb < budget.min_free_mem_gb:
            effective = max(1, min(effective, live))
            pressure = (f"low free memory ({machine.free_mem_gb:.1f}GB < "
                        f"{budget.min_free_mem_gb:.1f}GB)")

        if live + 1 > effective:
            detail = f" — {pressure}" if pressure else ""
            return False, (f"fleet at capacity ceiling ({live}/{effective} jobs on this "
                           f"machine){detail}")
        history = store.durations(job_class) or store.pooled_durations()
        verdict = _decider(budget, effective).decide_for(live, history)
        return verdict.admit, verdict.reason
    return _check


def acquire(job_class: str, meta: dict | None = None,
            budget: CIBudget | None = None) -> tuple[Lease | None, CIVerdict]:
    """Take a fleet slot, waiting up to `max_wait_s`. Fails OPEN on any internal error.

    Expiry admits rather than fails: the job is already holding a runner slot, so refusing it would
    burn the slot AND lose the work.
    """
    budget = budget or CIBudget.from_env()
    if not admission_enabled():
        return None, CIVerdict(True, f"admission disabled ({ENV_ENABLED}=off)", 0, 0)

    deadline = time.monotonic() + budget.max_wait_s
    started = time.monotonic()
    last = "no decision recorded"
    try:
        store = LeaseStore(budget.state_dir)
        while True:
            machine = probe_machine()
            ceiling = derive_ceiling(budget, machine)
            lease, reason, live = store.acquire(
                job_class,
                {"host": platform.node(), **(meta or {})},
                _capacity_check(budget, machine, ceiling, store, job_class),
            )
            waited = time.monotonic() - started
            if lease is not None:
                return lease, CIVerdict(True, reason, live + 1, ceiling, waited)
            last = reason
            if time.monotonic() >= deadline:
                # Bounded wait expired. Admit, and say why — a silently-throttled fleet is worse
                # than a briefly oversubscribed one.
                return None, CIVerdict(True, f"wait budget expired after {waited:.0f}s; admitting "
                                             f"anyway (last: {last})", live, ceiling, waited)
            time.sleep(min(budget.poll_s, max(0.0, deadline - time.monotonic())))
    except Exception as exc:  # fail OPEN — never block CI on this gate's own failure
        return None, CIVerdict(True, f"admission gate unavailable ({exc.__class__.__name__}: {exc}); "
                                     f"admitting (last: {last})", 0, 0,
                               time.monotonic() - started)


# ── entry point ────────────────────────────────────────────────────────────────────────────────────
# Runnable as `python -m codna.ci_admission` so a CI job can gate itself BEFORE installing anything:
# this module's decision path is stdlib-only, while `codna.cli` pulls in httpx. `codna ci ...`
# delegates here so both surfaces share one implementation.
def main(argv: list[str] | None = None) -> int:
    """ALWAYS returns 0 — this gate must never be the reason a build fails."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="codna ci",
        description="Keep a self-hosted runner fleet inside a chosen share of one machine.",
    )
    sub = parser.add_subparsers(dest="action")
    p_admit = sub.add_parser("admit", help="Wait for fleet capacity before doing real work.")
    p_admit.add_argument("--job-class", required=True)
    p_admit.add_argument("--meta", action="append", default=[], metavar="K=V")
    p_release = sub.add_parser("release", help="Release a slot and record the job's duration.")
    p_release.add_argument("--lease")
    sub.add_parser("status", help="Show the capacity budget and what is running.")
    args = parser.parse_args(argv)

    budget = CIBudget.from_env()
    action = args.action or "status"

    if action == "status":
        machine = probe_machine()
        ceiling = derive_ceiling(budget, machine)
        store = LeaseStore(budget.state_dir)
        store.ensure_dirs()
        live = store.reap()
        load = "unknown" if machine.load1 is None else f"{machine.load1:.1f}"
        free = "unknown" if machine.free_mem_gb is None else f"{machine.free_mem_gb:.1f}GB"
        print(f"fleet   : {live}/{ceiling} jobs in flight")
        print(f"machine : {machine.cores} cores · load {load} · free {free}")
        print(f"budget  : share {budget.cpu_share:.2f}"
              + (f" · reserve {budget.reserve_cores} cores" if budget.reserve_cores else "")
              + (f" · max_jobs {budget.max_jobs}" if budget.max_jobs is not None else "")
              + f" · load limit {core_budget(budget, machine) * budget.load_headroom:.1f}")
        print(f"state   : {budget.state_dir}")
        for rec in store.active():
            print(f"  - {rec.get('job_class', '?')} @ {rec.get('owner', '?')}")
        return 0

    if action == "release":
        raw = getattr(args, "lease", None) or os.environ.get("CODNA_CI_LEASE")
        if not raw:
            print("release: no lease given (--lease or $CODNA_CI_LEASE); nothing to do")
            return 0
        path = Path(raw)
        if not path.exists():
            print(f"release: lease already gone ({path.name}); nothing to do")
            return 0
        try:
            rec = json.loads(path.read_text())
            lease = Lease(path=path, job_class=rec.get("job_class", "job"),
                          started=float(rec.get("started", time.time())))
            duration = LeaseStore(budget.state_dir).release(lease)
            print(f"released {lease.job_class} after {duration:.1f}s (duration recorded)")
        except Exception as exc:
            # A lease that cannot be parsed is still a slot to free. The reaper would eventually get
            # it, but leaving it would hold capacity until the age cap.
            path.unlink(missing_ok=True)
            print(f"release: freed unparseable lease ({exc.__class__.__name__})")
        return 0

    meta = {}
    for item in args.meta or []:
        key, _, value = item.partition("=")
        if key:
            meta[key] = value
    lease, verdict = acquire(args.job_class, meta, budget)
    waited = f" (waited {verdict.waited_s:.0f}s)" if verdict.waited_s >= 1 else ""
    print(f"{'admit' if verdict.admit else 'hold'}: {verdict.reason}{waited}")
    if lease is not None:
        print(f"lease: {lease.path}")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            try:
                with open(out, "a") as fh:
                    fh.write(f"lease={lease.path}\n")
            except OSError:
                pass
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
