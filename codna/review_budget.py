"""The adaptive turn budget for ``codna review`` -- the fix path's machinery, reused for reviews.

A review turn used to run under one fixed number: ``review_timeout_s = 240`` (cline_agent.py), which
``packaged_agent_runner._timeout_ms_for_request`` forwarded to the sidecar as ``limits.timeoutMs``
while a FIX got ``_dynamic_timeout_ms`` (base + tokens + files) and the admission gate got a Monte
Carlo forecast over observed durations (admission_control). The fixed number failed on the first
large pull request that reached it: thyn-ai/algenta-sdk#71 (+1008/-15 across 5 files, one 865-line
property-test file) ended with ``turn exceeded timeout budget of 240000ms`` (check run 105948327905,
2026-09-19 18:47-18:52Z). Not flaky -- a 240 s ceiling on a ~1000-line diff is simply too small,
and no single bigger constant is right for both a 10-line PR and a 10-file rewrite.

So the review budget is now decided the way a fix's is, from two sources, and the larger wins:

1. **Size** -- the same formula as ``_dynamic_timeout_ms`` (base + capped per-token + per-file
   terms, see :func:`size_budget_ms`, which that function now calls too), fed with the REVIEW's own
   size signals: the diff's changed lines and files, plus the prompt-token estimate.
2. **Observation** -- the admission gate's Monte-Carlo duration model. Every review turn's observed
   wall-clock is recorded (like ``MCAdmissionDecider.finish``), normalised by that review's size
   factor so small pull requests inform large ones, and the budget is the smallest value whose breach
   probability for THIS pull request's size is <= epsilon -- computed by the SAME predictor the gate
   uses: ``admission_control.make_mojo_breach_fn`` -> ``mojo_breach_probability`` (the Algenta engine's
   compiled ``monte_carlo`` kernel through the local SDK, ``_encapsulated_simulate``), falling back to
   ``_numpy_breach_probability``. The size factor plays the role the contention factor plays for
   admission: a multiplicative stretch on the observed distribution.

Bounds: never below today's 240 s; never above :data:`REVIEW_CAP_MS`, which sits strictly under the
webhook's per-job bound (``CODNA_WEBHOOK_JOB_TIMEOUT_S=1800`` in infra/fly/codna-webhook.fly.toml,
code default 3600 in webhook_worker.py) with room for the sidecar wait's +30 s slack
(packaged_agent_runner._run_sidecar) and the clone/post work around the turn. ``ALGENTA_AGENT_TURN_TIMEOUT_MS``
still overrides everything, exactly as it does for fixes.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .admission_control import AdmissionConfig, BreachFn, _numpy_breach_probability, make_mojo_breach_fn

try:
    import fcntl
except ImportError:  # not POSIX: no advisory locks, so the history is appended and trimmed unlocked
    fcntl = None

# ── bounds ──────────────────────────────────────────────────────────────────────────────────────
REVIEW_FLOOR_MS = 240_000       # today's fixed budget: the adaptive one never goes below it
# Strictly below the webhook job bound (fly.toml CODNA_WEBHOOK_JOB_TIMEOUT_S=1800; code default 3600,
# webhook_worker._JOB_TIMEOUT_S), leaving 600 s for the two turns' +30 s sidecar-wait slacks, the PR
# clone and posting -- so a review that uses its whole budget still reports itself instead of being
# killed by the job bound and reported as "timed out after 1800s". The repair turn does not need room
# of its own: it shares the cap with the review turn (REVIEW_DEADLINE_MS below).
REVIEW_CAP_MS = 1_200_000
# ONE deadline per review, shared by the review turn and the single repair turn run_review_agent adds
# when the reply was not findings JSON (see :func:`repair_turn_budget_s`). Each turn used to draw its
# own budget from the same cap, so two turns could total 2 x REVIEW_CAP_MS = 2400 s -- past the 1800 s
# job bound, whose expiry kills the process group and reports a generic job timeout in place of the
# review's own classification. The cap test pins REVIEW_CAP_MS + two sidecar-wait slacks < job bound.
REVIEW_DEADLINE_MS = REVIEW_CAP_MS
# The least a repair turn is worth starting with. Below it the review fails closed as review_timeout:
# a turn that cannot finish only delays the same failure while eating into the job bound.
REPAIR_MIN_MS = 60_000

# ── the size formula (the fix path's shape, review coefficients) ───────────────────────────────
# Tuned so the algenta-sdk#71 shape (+1008/-15 lines, 5 files) lands at ~10 min and a 10-line PR
# stays within seconds of today's 240 s: lines are the review's dominant size signal (a review reads
# every changed line); files add the per-file open/orient cost; tokens use the fix path's own
# coefficient (2 ms/token, cap 900 s) for the prompt the model actually receives.
REVIEW_LINE_MS = 350
REVIEW_LINE_CAP_MS = 900_000
REVIEW_FILE_MS = 5_000
REVIEW_FILE_CAP_MS = 300_000
REVIEW_TOKEN_MS = 2
REVIEW_TOKEN_CAP_MS = 900_000

# ── the observation model ───────────────────────────────────────────────────────────────────────
DEFAULT_EPSILON = 0.02          # accept a 2% forecast risk of the turn outrunning its budget
# A turn that hit its budget is a CENSORED sample: its true duration is unknown but larger. It is
# kept out of the rate distribution (it would pull the tail DOWN once normalised) and instead sets a
# floor for its size class: the next budget for that size is at least the exhausted one plus a
# quarter -- this is what makes "comment `@codna review` to retry" grow the budget.
CENSORED_GROWTH = 1.25
_GRID_MS = 15_000               # budget resolution of the search (the kernel is sampled per probe)

ENV_TURN_TIMEOUT = "ALGENTA_AGENT_TURN_TIMEOUT_MS"
ENV_HISTORY_DIR = "CODNA_REVIEW_HISTORY_DIR"
ENV_EPSILON = "CODNA_REVIEW_BUDGET_EPSILON"
ENV_PREDICTOR = "CODNA_ADMISSION_MC"     # shared with the admission gate: "mojo" (default) | "numpy"
HISTORY_FILE = "review-turns.jsonl"

# The exact text the sidecar raises when a turn outruns ``limits.timeoutMs``
# (agent-core/vendor/cline/algenta/server/execution.ts, sendTurnWithTimeout).
_TURN_TIMEOUT_RE = re.compile(r"turn exceeded timeout budget of (\d+)ms")

SIGNAL_CHANGED_LINES = "review_changed_lines"
SIGNAL_CHANGED_FILES = "review_changed_files"


def size_budget_ms(
    *,
    base_ms: int,
    cap_ms: int,
    tokens: int = 0,
    token_ms: int = 0,
    token_cap_ms: int = 0,
    files: int = 0,
    file_ms: int = 0,
    file_cap_ms: int = 0,
    lines: int = 0,
    line_ms: int = 0,
    line_cap_ms: int = 0,
) -> int:
    """``max(base, min(cap, base + sum of capped per-unit terms))`` -- the one size formula.

    ``packaged_agent_runner._dynamic_timeout_ms`` (fixes) and :func:`review_turn_budget` (reviews)
    both call this; they differ only in which signals they feed it and with which coefficients."""
    terms = _terms_ms(tokens=tokens, token_ms=token_ms, token_cap_ms=token_cap_ms,
                      files=files, file_ms=file_ms, file_cap_ms=file_cap_ms,
                      lines=lines, line_ms=line_ms, line_cap_ms=line_cap_ms)
    return max(base_ms, min(cap_ms, base_ms + terms))


def _terms_ms(*, tokens, token_ms, token_cap_ms, files, file_ms, file_cap_ms, lines, line_ms, line_cap_ms) -> int:
    return (
        min(token_cap_ms, max(0, tokens) * token_ms)
        + min(file_cap_ms, max(0, files) * file_ms)
        + min(line_cap_ms, max(0, lines) * line_ms)
    )


def _review_terms_ms(changed_lines: int, changed_files: int, prompt_tokens: int) -> int:
    return _terms_ms(
        tokens=prompt_tokens, token_ms=REVIEW_TOKEN_MS, token_cap_ms=REVIEW_TOKEN_CAP_MS,
        files=changed_files, file_ms=REVIEW_FILE_MS, file_cap_ms=REVIEW_FILE_CAP_MS,
        lines=changed_lines, line_ms=REVIEW_LINE_MS, line_cap_ms=REVIEW_LINE_CAP_MS,
    )


def review_size_ms(changed_lines: int, changed_files: int, prompt_tokens: int) -> int:
    """Source 1: the size-derived budget, in [REVIEW_FLOOR_MS, REVIEW_CAP_MS]."""
    return size_budget_ms(
        base_ms=REVIEW_FLOOR_MS, cap_ms=REVIEW_CAP_MS,
        tokens=prompt_tokens, token_ms=REVIEW_TOKEN_MS, token_cap_ms=REVIEW_TOKEN_CAP_MS,
        files=changed_files, file_ms=REVIEW_FILE_MS, file_cap_ms=REVIEW_FILE_CAP_MS,
        lines=changed_lines, line_ms=REVIEW_LINE_MS, line_cap_ms=REVIEW_LINE_CAP_MS,
    )


def size_factor(changed_lines: int, changed_files: int, prompt_tokens: int) -> float:
    """How much longer than a trivial review this one should take: the size formula's terms over
    its base, uncapped overall (>= 1.0). The observation model divides each recorded duration by
    ITS review's factor (seconds per unit of size) and multiplies by this review's -- the same
    multiplicative stretch ``MCAdmissionDecider._mc_breach_probability`` applies for contention."""
    return 1.0 + _review_terms_ms(changed_lines, changed_files, prompt_tokens) / REVIEW_FLOOR_MS


# ── observed review turns (the duration window, persisted) ─────────────────────────────────────
@dataclass(frozen=True)
class ReviewSample:
    duration_s: float
    changed_lines: int
    changed_files: int
    prompt_tokens: int
    timed_out: bool = False
    budget_ms: int | None = None

    @property
    def factor(self) -> float:
        return size_factor(self.changed_lines, self.changed_files, self.prompt_tokens)


def history_path() -> Path:
    """Where observed review turns live. ``CODNA_REVIEW_HISTORY_DIR`` when set (the webhook points
    it at its volume, so the window survives redeploys and is shared by every job on the machine --
    a job's own ``CODNA_RUNTIME_ROOT`` is per-job scratch and would forget everything), else
    ``~/.codna/review-budget`` like ci_admission's state dir."""
    raw = os.environ.get(ENV_HISTORY_DIR, "").strip()
    base = Path(raw).expanduser() if raw else Path.home() / ".codna" / "review-budget"
    return base / HISTORY_FILE


def load_samples(path: Path | None = None, *, window: int | None = None) -> list[ReviewSample]:
    """The last ``window`` recorded turns (default: the admission window size). Tolerant: a missing
    file is an empty window and a garbage line is skipped, never raised -- this runs on the review's
    critical path and a broken history must not break the review. The file is trimmed on write, so
    parsing it whole and keeping the last ``window`` SAMPLES costs nothing and a stray line never
    costs a sample."""
    path = path or history_path()
    limit = window or AdmissionConfig().window
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[ReviewSample] = []
    for line in lines:
        try:
            rec = json.loads(line)
            sample = ReviewSample(
                duration_s=float(rec["duration_s"]),
                changed_lines=max(0, int(rec.get("changed_lines", 0))),
                changed_files=max(0, int(rec.get("changed_files", 0))),
                prompt_tokens=max(0, int(rec.get("prompt_tokens", 0))),
                timed_out=bool(rec.get("timed_out", False)),
                budget_ms=int(rec["budget_ms"]) if rec.get("budget_ms") is not None else None,
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if sample.duration_s > 0:
            out.append(sample)
    return out[-limit:]


def record_sample(sample: ReviewSample, path: Path | None = None, *, window: int | None = None) -> None:
    """Append one observed turn, exactly like ``MCAdmissionDecider.finish`` records a fix (positive
    durations only) and like ci_admission's on-disk history: append-only, trimmed to the window on
    write, and never raising -- recording is bookkeeping, not part of the review's result.

    Append and trim run under one exclusive lock (:func:`_history_lock`): on the webhook every job is
    its own process writing the same shared history, and two unlocked trims -- each a read of the
    whole file followed by a rewrite -- would let one overwrite the other's freshly appended sample or
    leave a torn file behind. The trim writes a temp file and ``os.replace``s it, so a concurrent
    reader (``load_samples`` takes no lock) always sees a complete file."""
    if sample.duration_s <= 0:
        return
    path = path or history_path()
    limit = window or AdmissionConfig().window
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {**asdict(sample), "duration_s": round(float(sample.duration_s), 6), "at": time.time()}
        with _history_lock(path):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) > limit * 2:
                tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                tmp.write_text("\n".join(lines[-limit:]) + "\n", encoding="utf-8")
                os.replace(tmp, path)
    except OSError:
        pass


@contextlib.contextmanager
def _history_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock for the history's append+trim. A SIDECAR lock file, not the history
    itself: the trim replaces the history's inode, and a lock on the old inode would not cover the
    writer that opens the new one. Without ``fcntl`` (not POSIX) the section runs unlocked."""
    if fcntl is None:
        yield
        return
    with open(path.with_name(path.name + ".lock"), "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# ── the decision ────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ReviewBudget:
    budget_ms: int
    source: str                 # env | size | mc | censored | cap
    size_ms: int
    size_factor: float
    mc_ms: int | None
    censored_ms: int | None
    changed_lines: int
    changed_files: int
    prompt_tokens: int
    samples: int
    censored_samples: int
    epsilon: float
    # The predictor that ANSWERED the forecast's probes: "mojo" only if every probe came from the
    # kernel, "numpy" once any probe fell back (admission_control.make_mojo_breach_fn), "custom" for an
    # injected plain callable, None when no forecast ran (no recorded turns yet, or the env override).
    # ``predictor_configured`` is the CODNA_ADMISSION_MC setting; the two differed silently before --
    # the webhook image ships no ``apps`` package, so every hosted decision was numpy under a "mojo" log.
    predictor: str | None
    predictor_configured: str

    def to_log(self) -> dict[str, Any]:
        return {
            "service": "codna-review",
            "event": "review_turn_budget",
            **asdict(self),
            "size_factor": round(self.size_factor, 3),
            "floor_ms": REVIEW_FLOOR_MS,
            "cap_ms": REVIEW_CAP_MS,
        }


_BREACH_FN: BreachFn | None = None
_BREACH_LOCK = threading.Lock()


def predictor_name() -> str:
    return "numpy" if os.environ.get(ENV_PREDICTOR, "mojo").strip().lower() == "numpy" else "mojo"


def _default_breach_fn() -> BreachFn:
    """The gate's own predictor, built once per process: ``make_mojo_breach_fn()`` (the compiled
    kernel via the local SDK, numpy fallback announced once) unless ``CODNA_ADMISSION_MC=numpy``."""
    global _BREACH_FN
    if _BREACH_FN is None:
        with _BREACH_LOCK:
            if _BREACH_FN is None:
                _BREACH_FN = _numpy_breach_probability if predictor_name() == "numpy" else make_mojo_breach_fn()
    return _BREACH_FN


def _epsilon() -> float:
    try:
        value = float(os.environ.get(ENV_EPSILON, "") or DEFAULT_EPSILON)
    except ValueError:
        return DEFAULT_EPSILON
    return value if 0.0 < value < 1.0 else DEFAULT_EPSILON


def _env_override_ms() -> int | None:
    raw = os.environ.get(ENV_TURN_TIMEOUT, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 30_000
    return max(30_000, min(3_600_000, value))     # the same clamp _dynamic_timeout_ms applies


def _grid(ms: float) -> int:
    return int(round(ms / _GRID_MS)) * _GRID_MS


def _is_numpy_predictor(fn: BreachFn) -> bool:
    """``fn is _numpy_breach_probability``, robust to ``admission_control`` having been reloaded
    (tests reload it to reset the process-wide admitter; a reload makes a new function object while
    this module still holds the one it imported)."""
    return (getattr(fn, "__module__", None) == _numpy_breach_probability.__module__
            and getattr(fn, "__qualname__", None) == _numpy_breach_probability.__qualname__)


class _DecisionPredictor:
    """The predictor as ONE budget decision sees it. Callable as a ``BreachFn``; ``answered`` names
    the branch that produced this decision's probes: "mojo" only if every probe came from the kernel,
    "numpy" once any probe fell back -- and from then on the decision STAYS on the fallback, because a
    kernel that times out rather than failing fast would otherwise cost one timeout per bisection step
    (up to ~8 per decision) -- or "custom" for an injected plain callable (tests). Per decision, so
    two threads deciding at once cannot read each other's branch."""

    def __init__(self, fn: BreachFn) -> None:
        self._fn = fn
        self._probe = getattr(fn, "probe", None)          # make_mojo_breach_fn's (probability, branch)
        self._fallback = getattr(fn, "fallback", None)    # ...and the predictor its fallback branch runs
        self._plain = "numpy" if _is_numpy_predictor(fn) else "custom"
        self.answered: str | None = None

    def __call__(self, durations: list[float], contention: float, sla: float, n: int) -> float:
        if self.answered == "numpy" and self._fallback is not None:
            return self._fallback(durations, contention, sla, n)
        if self._probe is None:
            self.answered = self._plain
            return self._fn(durations, contention, sla, n)
        probability, branch = self._probe(durations, contention, sla, n)
        if self.answered != "numpy":
            self.answered = branch
        return probability


def _smallest_budget_within_epsilon(rates: list[float], factor: float, fn: BreachFn, epsilon: float, trials: int) -> int:
    """Smallest grid budget in [floor, cap] with ``P(rate x factor > budget) <= epsilon``, or one grid
    step PAST the cap when even the cap breaches (the caller clamps and reports the cap as the source).

    ``fn`` is a ``BreachFn`` -- ``(durations, contention, sla, n) -> P(duration x contention > sla)``
    -- so the size factor travels in the contention slot and each candidate budget in the SLA slot.
    P is non-increasing in the budget, so a bisection over the grid needs ~6 kernel probes."""
    def breaches(budget_ms: int) -> bool:
        return fn(rates, factor, budget_ms / 1000.0, trials) > epsilon

    if breaches(REVIEW_CAP_MS):
        return REVIEW_CAP_MS + _GRID_MS
    if not breaches(REVIEW_FLOOR_MS):
        return REVIEW_FLOOR_MS
    lo, hi = REVIEW_FLOOR_MS, REVIEW_CAP_MS        # lo breaches, hi does not
    while hi - lo > _GRID_MS:
        mid = _grid((lo + hi) / 2)
        if mid <= lo or mid >= hi:
            break
        if breaches(mid):
            lo = mid
        else:
            hi = mid
    return hi


def review_turn_budget(
    *,
    changed_lines: int,
    changed_files: int,
    prompt_tokens: int,
    samples: list[ReviewSample],
    breach_fn: BreachFn | None = None,
    epsilon: float | None = None,
    trials: int | None = None,
) -> ReviewBudget:
    """Decide one review turn's budget. Pure given its inputs (the env override aside); the predictor
    is injectable exactly as ``MCAdmissionDecider(mc_fn=...)`` is, so tests never touch the kernel."""
    lines, files, tokens = max(0, changed_lines), max(0, changed_files), max(0, prompt_tokens)
    eps = epsilon if epsilon is not None else _epsilon()
    n_trials = trials or AdmissionConfig().trials
    size_raw = REVIEW_FLOOR_MS + _review_terms_ms(lines, files, tokens)   # before the cap
    size_ms = review_size_ms(lines, files, tokens)
    factor = size_factor(lines, files, tokens)
    complete = [s for s in samples if not s.timed_out and s.duration_s > 0]
    censored = [s for s in samples if s.timed_out and s.duration_s > 0]

    override = _env_override_ms()
    if override is not None:
        return ReviewBudget(override, "env", size_ms, factor, None, None, lines, files, tokens,
                            len(samples), len(censored), eps, None, predictor_name())

    mc_ms: int | None = None
    answered: str | None = None
    if complete:
        rates = [s.duration_s / s.factor for s in complete]        # seconds per unit of size
        predictor = _DecisionPredictor(breach_fn or _default_breach_fn())
        mc_ms = _smallest_budget_within_epsilon(rates, factor, predictor, eps, n_trials)
        answered = predictor.answered

    censored_ms: int | None = None
    if censored:
        # pro-rate each exhausted budget's observed duration to this review's size, then grow it
        censored_ms = int(max(s.duration_s * (factor / s.factor) * CENSORED_GROWTH for s in censored) * 1000)

    source, raw = "size", size_raw
    for name, value in (("mc", mc_ms), ("censored", censored_ms)):
        if value is not None and value > raw:
            source, raw = name, value
    budget = max(REVIEW_FLOOR_MS, min(REVIEW_CAP_MS, raw))
    if raw > REVIEW_CAP_MS:
        source = "cap"                 # the winning term asked for more than the cap allows
    return ReviewBudget(budget, source, size_ms, factor, mc_ms, censored_ms, lines, files, tokens,
                        len(samples), len(censored), eps, answered, predictor_name())


def repair_turn_budget_s(*, first_budget_ms: int | None, elapsed_s: float) -> int | None:
    """The budget the ONE repair turn may still be granted (whole seconds, for the caller-pin path
    ``review_timeout_s``), or None: too little of the review's deadline is left and the review must
    fail closed as review_timeout instead of starting a turn that cannot finish.

    Both turns share :data:`REVIEW_DEADLINE_MS`. The first turn's wall-clock is gone; the repair gets
    the remainder, never more than the first turn's own budget (the repair prompt is the first plus a
    short preamble, so it needs no more), never less than the pin path's 30 s floor. Logged as one
    structured line (``review_repair_turn_budget``) like every other budget decision."""
    remaining_ms = REVIEW_DEADLINE_MS - int(elapsed_s * 1000)
    budget_s: int | None = None
    if remaining_ms >= REPAIR_MIN_MS:
        share_ms = remaining_ms if first_budget_ms is None else min(first_budget_ms, remaining_ms)
        budget_s = max(30, share_ms // 1000)
    _log({"service": "codna-review", "event": "review_repair_turn_budget",
          "first_budget_ms": first_budget_ms, "first_elapsed_s": round(float(elapsed_s), 3),
          "deadline_ms": REVIEW_DEADLINE_MS, "remaining_ms": max(0, remaining_ms),
          "repair_budget_ms": None if budget_s is None else budget_s * 1000, "repair_skipped": budget_s is None})
    return budget_s


# ── the runner's entry point + the hand-off back to the review ─────────────────────────────────
_DECISIONS: dict[str, ReviewBudget] = {}
_DECISIONS_LOCK = threading.Lock()
_DECISIONS_KEEP = 16


def review_turn_budget_ms(signals: Mapping[str, Any], snapshot: Mapping[str, Any]) -> int:
    """What ``packaged_agent_runner._timeout_ms_for_request`` returns for ``task_kind == "review"``
    when no caller pinned ``review_timeout_s``: the decision above over the persisted window, logged
    as one structured stderr line so an operator can read every input behind the number."""
    lines = _int(signals.get(SIGNAL_CHANGED_LINES))
    files = _int(signals.get(SIGNAL_CHANGED_FILES))
    tokens = _int(snapshot.get("raw_repo_token_estimate"))
    decision = review_turn_budget(changed_lines=lines, changed_files=files, prompt_tokens=tokens,
                                  samples=load_samples())
    remember_decision(str(snapshot.get("snapshot_id") or ""), decision)
    _log(decision.to_log())
    return decision.budget_ms


def remember_decision(snapshot_id: str, decision: ReviewBudget) -> None:
    with _DECISIONS_LOCK:
        _DECISIONS[snapshot_id] = decision
        while len(_DECISIONS) > _DECISIONS_KEEP:
            del _DECISIONS[next(iter(_DECISIONS))]


def decision_for(snapshot_id: str) -> ReviewBudget | None:
    """The budget granted to the run with this snapshot id (the sidecar payload does not come back
    with the result), so the review can name it when the turn outruns it and record it alongside
    the observed duration."""
    with _DECISIONS_LOCK:
        return _DECISIONS.get(snapshot_id)


def observe_review_turn(
    *,
    duration_s: float,
    changed_lines: int,
    changed_files: int,
    prompt_tokens: int,
    budget_ms: int | None,
    timed_out: bool,
) -> None:
    """Record one finished (or exhausted) review turn -- ``MCAdmissionDecider.finish`` for reviews --
    and say so on stderr."""
    sample = ReviewSample(duration_s=float(duration_s), changed_lines=max(0, changed_lines),
                          changed_files=max(0, changed_files), prompt_tokens=max(0, prompt_tokens),
                          timed_out=timed_out, budget_ms=budget_ms)
    record_sample(sample)
    _log({"service": "codna-review", "event": "review_turn_observed", **asdict(sample),
          "duration_s": round(float(duration_s), 3), "size_factor": round(sample.factor, 3)})


def is_turn_timeout(details: Mapping[str, Any] | None, *, cause_code: str | None,
                    elapsed_s: float, budget_ms: int | None) -> bool:
    """Did this agent-core failure mean "the turn outran its budget"?

    Two shapes: the sidecar enforced ``limits.timeoutMs`` itself and said so in the final frame
    (``terminal_state: runtime_invalid``, ``error: turn exceeded timeout budget of <n>ms`` --
    agent_core_run_failed), or the sidecar never answered and codna's own HTTP wait (budget + 30 s)
    gave up (agent_core_transport_failed after at least the budget elapsed)."""
    details = details or {}
    error = str(details.get("error") or "")
    if _TURN_TIMEOUT_RE.search(error):
        return True
    if cause_code == "agent_core_transport_failed" and budget_ms is not None:
        return elapsed_s * 1000.0 >= budget_ms
    return False


def budget_from_error(details: Mapping[str, Any] | None) -> int | None:
    """The budget the sidecar itself reported exhausting, when its message carries one."""
    m = _TURN_TIMEOUT_RE.search(str((details or {}).get("error") or ""))
    return int(m.group(1)) if m else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _log(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)
