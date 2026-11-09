"""A fake sidecar and a harness, so the durable suite drives the REAL activities off a real server.

The `/v1` wire is already covered against a real socket by `test_sidecar_client.py`, so what the
workflow tests need is the layer above it: a sidecar-shaped object whose answers let the real
`run_cline_turn` / `plan_task` bodies run, so the things under test are the activities' own
behaviour — the `turn_key` they send, the `?after=` they resume from, the ledger they append as
events arrive, the abort they issue on cancellation — against the real workflow, the real graph and
a real git repository.

Why not stub the activities: a stub proves the workflow calls *something* in the right order, which
these tests can already see in the ledger. Using the real bodies proves the two halves agree — the
state a turn returns is the state the checkpoint commits is the state the reviewer reads. That is the
seam where a durable run actually breaks.

Everything substituted is named in `Harness`: the sidecar (no network), the test runner (no
subprocess), the reviewer (no model) and the push (no GitHub). Git is NOT substituted: the
checkpoint's property — one commit and one ref per ``turn_seq``, idempotent under retry — is a
property of git, and a mocked git would test the mock.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

from codna.durable.activities import ActivityRuntime, configure_default_runtime
from codna.durable.config import DurablePaths
from codna.durable.ledger import RunLedger
from codna.durable.sidecar_client import (
    OMIT_WRITE_SCOPE,
    SidecarClientError,
    SidecarSession,
    SseEvent,
    TurnStart,
)

TERMINAL_OK = "session.turn_completed"
TERMINAL_FAILED = "session.turn_failed"
TERMINAL_ABORTED = "session.turn_aborted"

# The origin every harness repository carries by default, so the publish step finds a GitHub remote
# and the scripted push is what runs. `make_repo(..., origin=None)` builds the local-only repository
# instead — the case the product actually meets and the one `publish_skip` exists for.
GITHUB_ORIGIN = "https://github.com/thyn-ai/codna.git"


@dataclass
class FakeTurn:
    turn_id: str
    turn_key: str
    prompt: str
    session_id: str
    read_only: bool = False


@dataclass
class FakeSidecar:
    """One session, many turns — enough of the engine's behaviour for the real activities.

    ``event_counter`` is per SESSION, which is the real wire's contract and what makes ``?after=``
    meaningful: a later turn's events continue where the earlier turn's stopped. A per-turn counter
    would make every turn after the first look like a replay of it.
    """

    cost_per_turn: float = 0.25
    tokens_per_turn: int = 100
    terminal: str = TERMINAL_OK
    # A real turn EDITS the checkout. Off, every checkpoint would anchor at the same commit and the
    # "one commit per turn" property would be untestable — which is why this is on by default and a
    # test that needs "a turn that changed nothing" turns it off.
    writes_files: bool = True
    # Set by a test to make the NEXT stream observe a cancellation after N events.
    cancel_after_events: int | None = None
    cancel_probe: Callable[[], bool] | None = None
    # D4: when set, every turn's terminal event reports `finish_reason: "ask"` and carries this ask —
    # the exact shape `execution.ts` now emits when a tool call was parked for a human. Set as a dict
    # so a test can assert the durable side parks on THAT ask rather than on one it invented.
    ask_on_terminal: dict[str, Any] | None = None

    turns: list[FakeTurn] = field(default_factory=list)
    # D7/OVERLAP: one `(task_id, started_at, finished_at)` entry per turn, on the contract CLOCK.
    # Populated only when `overlap_clock` is set, so an ordinary test pays nothing for it.
    #
    # The point of this is the multitask assertion: "two tasks both finished" is satisfied by a
    # strictly sequential run, so overlap has to be measured as INTERVALS. The fake is the right place
    # for it because it is the boundary where a paid turn actually occupies a worker slot.
    overlap_clock: Callable[[], float] | None = None
    intervals: list[tuple[str, float, float]] = field(default_factory=list)
    #: turn_id -> monotonic start, filled at `start_turn` and closed at the terminal event.
    overlap_started: dict[str, float] = field(default_factory=dict)
    #: How long a turn OCCUPIES the fake before it yields its terminal event, in real seconds.
    #: `0` is the historical behaviour (yield immediately). It exists so a test can make a turn long
    #: enough that two of them MUST overlap to be visible at all: with an instantaneous turn, a
    #: strictly sequential run and a concurrent one are indistinguishable from timing alone.
    turn_hold_s: float = 0.0
    # Sessions the engine has FINALIZED — the SDK's `shutdownSession` removes the id from its active map at
    # the end of a non-interactive session's first turn, so any later `send()` on it throws
    # `SessionNotFoundError`. Modelled here because the fake previously accepted every id forever, which is
    # precisely why the whole durable suite passed while a real soak died on the second turn.
    finalized: set[str] = field(default_factory=set)
    # Sessions this PROCESS never heard of (a sidecar restart). A `resume_from` naming one is answered the
    # way the real engine does it: recover the transcript, mint a NEW id, report `resumed: true`.
    restarted: set[str] = field(default_factory=set)
    # PER SESSION, exactly like the engine. `server/state.ts` declares
    # `turnKeyIndex: Map<sessionId, Map<key, record>>` and `turn-key.ts::lookupTurnKey` reads
    # `state.turnKeyIndex.get(sessionId)?.get(key)` — so a key is only idempotent inside the session
    # that saw it first. A single global dict here would silently hide the real defect this suite
    # exists to catch: a re-created session gives the same key a fresh, empty index, and the retry
    # it was supposed to replay becomes a second paid turn (`turn_key_rescoped`).
    turn_keys: dict[str, dict[str, str]] = field(default_factory=dict)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    #: One entry per `start_turn`: the TURN's write scope as it arrived on the wire (`None` = the turn
    #: named none and keeps the session's). Recorded separately from `sessions` because the scope is a
    #: per-turn property — a durable task's read-only plan turn and writable implement turn share one
    #: session, so "was this session created read-only?" cannot answer "may THIS turn write?".
    turn_write_scopes: list[dict[str, Any]] = field(default_factory=list)
    aborts: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    closed: int = 0
    events_sent: int = 0
    live_sessions: set[str] = field(default_factory=set)
    event_counters: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- SidecarClient's surface ----------------------------------------------------

    def create_session(
        self,
        *,
        working_dir: str,
        prompt: str = "",
        resume_from: str | None = None,
        spec: dict[str, Any] | None = None,
    ) -> SidecarSession:
        """Create a session, or resume one this fake already issued.

        A NEW session id per creation, like the real engine: two tasks are two conversations, and a
        fake that reused one id would make their usage cumulative and their `turn_key` sequences
        indistinguishable.
        """
        with self.lock:
            honoured = resume_from is not None and resume_from in self.live_sessions
            recovered = resume_from is not None and resume_from in self.restarted
            if honoured:
                session_id = str(resume_from)
            else:
                session_id = f"session-durable-{len(self.sessions) + 1}"
            # Recorded with the id it produced, so `_spec_for` can answer "was THIS session declared
            # multi-turn?" — the flag travels on the create that actually minted the id.
            self.sessions.append(
                {
                    "session_id": session_id,
                    "working_dir": working_dir,
                    "resume_from": resume_from,
                    "spec": dict(spec or {}),
                }
            )
            if honoured:
                return SidecarSession(
                    session_id=session_id,
                    resumed=True,
                    resume_from_ignored=False,
                )
            self.live_sessions.add(session_id)
            if recovered:
                # The engine's own recovery: the transcript was restored, so continuity IS preserved — the
                # reply says `resumed: true` even though the id is new and nothing was ignored.
                return SidecarSession(
                    session_id=session_id,
                    resumed=True,
                    resume_from_ignored=False,
                )
            return SidecarSession(
                session_id=session_id,
                resumed=False,
                resume_from_ignored=resume_from is not None,
            )

    def start_turn(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_key: str | None = None,
        allowed_write_paths: Any = OMIT_WRITE_SCOPE,
    ) -> TurnStart:
        key = turn_key or ""
        with self.lock:
            self._require_live(session_id)
            self.prompts.append(prompt)
            # The TURN's scope, exactly as it arrived: `OMIT_WRITE_SCOPE` = the caller named none,
            # `None` = explicit "no additional restriction" (JSON null), a sequence = that scope
            # (`[]` is read-only). Recorded per turn because a durable task's plan and implement turns
            # share one session, so the SESSION's scope cannot answer "may THIS turn write?".
            self.turn_write_scopes.append(
                {
                    "session_id": session_id,
                    "allowed_write_paths": (
                        "omitted"
                        if allowed_write_paths is OMIT_WRITE_SCOPE
                        else allowed_write_paths
                    ),
                }
            )
            session_keys = self.turn_keys.setdefault(session_id, {})
            if key and key in session_keys:
                # `turn_key` idempotency, exactly as the wire specifies: the SAME turn id, no new
                # turn. The activity must treat this as a replay and not bill twice. Scoped to
                # THIS session, because that is the engine's scope: the same key sent to a session
                # this index has never seen is a first call and starts a real turn.
                return TurnStart(session_id=session_id, turn_id=session_keys[key], replayed=True)
            turn_id = f"turn-{len(self.turns) + 1}"
            read_only = self.read_only_turn(
                session_id, allowed_write_paths=allowed_write_paths
            )
            self.turns.append(
                FakeTurn(
                    turn_id=turn_id,
                    turn_key=key,
                    prompt=prompt,
                    session_id=session_id,
                    read_only=read_only,
                )
            )
            if key:
                session_keys[key] = turn_id
            if self.writes_files and not read_only:
                self._edit_checkout(session_id, turn_id)
            if self.overlap_clock is not None:
                self.overlap_started[turn_id] = self.overlap_clock()
            return TurnStart(session_id=session_id, turn_id=turn_id, replayed=False)

    def _require_live(self, session_id: str) -> None:
        """Refuse a turn on a session the SDK tore down — the real engine's `SessionNotFoundError`.

        Raised as the client's own error type with the engine's code, so an activity that swallows
        ``SidecarClientError`` would still be visible, and so a test can assert the exact failure the
        soak reported rather than "some exception".
        """
        if session_id not in self.finalized:
            return
        raise SidecarClientError(
            f"start_turn failed: 404 session_not_found: session not found: {session_id}",
            code="session_not_found",
            status=404,
            details={"session_id": session_id},
        )

    def _finalize_if_one_shot(self, session_id: str) -> None:
        """Model the SDK's end-of-turn teardown for a session that was NOT declared multi-turn.

        The real failure, measured: `runTurn` → `finalizeSingleRun` → `shutdownSession` →
        `this.sessions.delete(sessionId)` for a non-interactive session, so the NEXT turn on that id
        answers `SessionNotFoundError: session not found`. A session created with `multi_turn: true`
        stays alive (that is what `completeInteractiveTurn` does instead).

        Kept as its own method so the fake's rule is readable next to the engine's, and so a test that
        needs a session to survive can assert exactly this flag rather than a magic `live_sessions.clear()`.
        """
        spec = self._spec_for(session_id)
        if spec.get("multi_turn") is True:
            return
        # Deliberately NOT removed from `live_sessions`. The two layers are different and conflating them
        # hides the defect: the ENGINE still holds its own SessionRecord (so `resume_from` is honoured and
        # answers `resumed: true` with the SAME id), while the SDK underneath has dropped the active
        # session (so the turn POST fails). That split is exactly what the real soak reported: an
        # acknowledged `session.resumed` followed by `SessionNotFoundError`.
        self.finalized.add(session_id)

    def forget_session(self, session_id: str) -> None:
        """Simulate the SIDECAR RESTARTING: this process no longer knows the id at all.

        Distinct from `finalized`, which is the ENGINE still holding its record while the SDK underneath
        dropped the active session (a completed one-shot turn). Here the whole process is gone, so a
        `resume_from` cannot be honoured in place. The real engine answers that by recovering the
        persisted transcript and seeding a NEW session from it — `resumed: true`, a DIFFERENT id,
        `resume_from_ignored: false` — so the fake reproduces exactly that instead of pretending the id
        is still live. Without this the durable side's "the conversation moved" branch is unreachable.
        """
        self.restarted.add(session_id)
        self.live_sessions.discard(session_id)
        self.finalized.discard(session_id)

    def _spec_for(self, session_id: str) -> dict[str, Any]:
        """The create spec that produced ``session_id`` (the LAST one, which is what a resume re-states)."""
        for session in reversed(self.sessions):
            if session.get("session_id") == session_id:
                return dict(session.get("spec") or {})
        return {}

    def read_only_turn(self, session_id: str, *, allowed_write_paths: Any = OMIT_WRITE_SCOPE) -> bool:
        """Whether THIS turn writes nothing — the engine's allow-list, not the prompt's ask.

        The turn's own scope wins whenever the caller named one:
        ``[]`` = read-only, a list = restricted but writable, ``None`` (JSON null) = explicitly
        unrestricted, which LIFTS a read-only session. Only a turn that named nothing inherits the
        session's create spec — and that inheritance is exactly why a durable run's writable
        ``implement`` turn must send ``None`` rather than omit the field.
        """
        if allowed_write_paths is not OMIT_WRITE_SCOPE:
            return allowed_write_paths is not None and len(allowed_write_paths) == 0
        for session in reversed(self.sessions):
            spec = session.get("spec") or {}
            if spec.get("allowed_write_paths") == []:
                return True
            break
        return False

    def _edit_checkout(self, session_id: str, turn_id: str) -> None:
        """A turn changes the tree. Without this every checkpoint would anchor at the same commit."""
        working_dir = next(
            (session["working_dir"] for session in reversed(self.sessions) if session.get("working_dir")),
            None,
        )
        if not working_dir:
            return
        path = Path(working_dir) / "app.py"
        path.write_text(
            f"def hello():\n    return 'hello from {turn_id}'\n", encoding="utf-8"
        )

    def stream_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        after: int = 0,
        should_reconnect: Callable[[int], bool] | None = None,
    ) -> Iterable[SseEvent]:
        """Three progress events, then a terminal. Indices continue the SESSION's sequence."""
        turn = next(turn for turn in self.turns if turn.turn_id == turn_id)
        start = max(after, self.event_counters.get(session_id, 0))
        # The hold models a turn's DURATION: a real paid turn occupies its worker slot for seconds.
        # It happens before the first event so the whole stream belongs to one held interval, which is
        # what makes `intervals` a truthful record of "when did this slot open and close".
        if self.turn_hold_s:
            time.sleep(self.turn_hold_s)
        for number in range(1, 4):
            index = start + number
            yield self._event(turn, index, "agent.text", {"n": number})
            if self.cancel_after_events is not None and number >= self.cancel_after_events:
                # The stream stops WITHOUT a terminal, which is what makes the activity's
                # cancellation check reachable: it sees the next event was never delivered.
                if self.cancel_probe is not None and self.cancel_probe():
                    return
        # A duplicated TERMINAL is impossible on the real wire; a test that wants a repeat should not
        # get it by accident.
        yield self._event(
            turn,
            start + 4,
            self.terminal,
            {
                "usage": self.cumulative_usage(),
                # D4: the engine reports WHY the turn stopped, and carries the ask when it parked one.
                # Both are on the terminal event (see `execution.ts`), which is what lets the activity
                # decide synchronously with the turn's end rather than by inspecting later state.
                **({"finish_reason": "ask", "ask": dict(self.ask_on_terminal)} if self.ask_on_terminal else {}),
            },
        )
        if self.overlap_clock is not None:
            # The interval CLOSES at the terminal event, which is when this worker slot is released.
            # Recorded before the fake's stream ends so an overlapping turn's start is always ordered
            # against a real finish rather than against the generator being garbage collected.
            with self.lock:
                opened = self.overlap_started.pop(turn.turn_id, None)
                if opened is not None:
                    self.intervals.append(
                        (turn.turn_key or turn.turn_id, opened, self.overlap_clock())
                    )
        # The teardown happens at the END of the turn, exactly as the SDK does it: the terminal event is
        # already on the wire, so the turn that just finished still reports success and only the NEXT
        # `start_turn` observes the missing session. That ordering is what the real soak showed.
        if self.terminal == TERMINAL_OK:
            with self.lock:
                self._finalize_if_one_shot(session_id)

    def _event(
        self, turn: FakeTurn, index: int, event_type: str, payload: dict[str, Any]
    ) -> SseEvent:
        with self.lock:
            self.events_sent = max(self.events_sent, index)
            self.event_counters[turn.session_id] = max(
                self.event_counters.get(turn.session_id, 0), index
            )
        return SseEvent(
            event_id=f"evt-{turn.turn_id}-{index}",
            event_name="session_event",
            event_type=event_type,
            event_index=index,
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            payload=dict(payload),
        )

    def cumulative_usage(self) -> dict[str, Any]:
        """CUMULATIVE session usage, as the engine reports it on the terminal event.

        Cumulative for the SESSION, which is what makes the per-turn DELTA the activity reports
        distinguishable from the total — and what a run-level cost sum depends on.
        """
        started = max(1, len(self.turns))
        return {
            "inputTokens": self.tokens_per_turn * started,
            "outputTokens": 0,
            "totalCost": self.cost_per_turn * started,
        }

    def session_for(self, turn_id: str) -> str:
        return next(turn.session_id for turn in self.turns if turn.turn_id == turn_id)

    def abort(self, session_id: str) -> None:
        with self.lock:
            self.aborts.append(session_id)

    def close(self) -> None:
        with self.lock:
            self.closed += 1


@dataclass
class FakeVerify:
    """A verification outcome shaped like `SandboxResult`."""

    passed: bool = True
    returncode: int = 0
    log_path: str | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def accepted(self, accepted_exit_codes: tuple[int, ...] = (0,)) -> bool:
        return (not self.timed_out) and self.returncode in accepted_exit_codes


@dataclass
class FakeReviewReport:
    """A review report shaped like `ReviewResult`, with a scripted conclusion."""

    conclusion: str = "success"
    severities: tuple[str, ...] = ()

    @property
    def findings(self) -> list[Any]:
        from types import SimpleNamespace

        return [SimpleNamespace(severity=severity) for severity in self.severities]


@dataclass
class Harness:
    """Everything one durable test needs: a repository, a run directory, and the seams wired.

    ``configure_default_runtime`` is called here because the activities are registered FUNCTIONS: by
    the time a worker runs one there is no place to pass an argument, and a process-wide runtime is
    exactly how the worker will inject the keystore-resolved token in D2.4.
    """

    repo: Path
    runs_root: Path
    sidecar: FakeSidecar
    runtime: ActivityRuntime
    verify_passes: list[bool] = field(default_factory=lambda: [True])
    review_conclusions: list[str] = field(default_factory=lambda: ["success"])
    pr_number: int = 1
    verify_calls: int = 0
    review_calls: int = 0
    push_calls: int = 0
    notifications: list[dict[str, Any]] = field(default_factory=list)
    # D5 goal mode: scripted CI answers, one per `watch_ci` call, defaulting to "green".
    ci_states: list[str] = field(default_factory=list)
    ci_calls: list[Any] = field(default_factory=list)

    # -- the seams ------------------------------------------------------------------

    def run_tests(self, context: Any, commands: tuple[str, ...]) -> FakeVerify:
        """Scripted verification: pop a result per call, defaulting to "passes"."""
        self.verify_calls += 1
        passed = self.verify_passes.pop(0) if self.verify_passes else True
        return FakeVerify(passed=passed, returncode=0 if passed else 1)

    def run_review(self, request: Any, sha: str) -> FakeReviewReport:
        """Scripted review: pop a conclusion per call, defaulting to a clean review."""
        self.review_calls += 1
        conclusion = self.review_conclusions.pop(0) if self.review_conclusions else "success"
        return FakeReviewReport(conclusion=conclusion)

    def push_and_open(self, request: Any, sha: str) -> dict[str, Any]:
        self.push_calls += 1
        return {
            "number": self.pr_number,
            "url": f"https://github.com/thyn-ai/codna/pull/{self.pr_number}",
            "branch": request.context.branch,
        }

    def watch_ci(self, request: Any) -> Any:
        """Scripted CI: pop a state per call, defaulting to green.

        The FAKE CI boundary is the GitHub API — the one thing genuinely external — so it is the one
        thing scripted. Everything above it (the loop, the bounds, the ledger) is the real code.
        """
        from codna.durable.contracts import WatchCiOutcome

        self.ci_calls.append(request)
        state = self.ci_states.pop(0) if self.ci_states else "passed"
        if state == "failed":
            return WatchCiOutcome(
                state="failed",
                checks=(("ci / test", "FAILURE"),),
                head_sha="b" * 40,
            )
        return WatchCiOutcome(state=state, checks=(), head_sha="b" * 40)

    # -- reading the result ---------------------------------------------------------

    def ledger(self, run_id: str = "run-1") -> RunLedger:
        return self.runtime.ledger_factory(run_id)

    def route(self, run_id: str = "run-1", task_id: str = "task-1") -> list[str]:
        """The task's route, as the LEDGER recorded it — the same for both engines.

        This is the audit-visible sequence a human reads to answer "what did this run actually do",
        which makes it the honest thing to compare engines on: it cannot be produced by a stub, only
        by the real activities doing their work.
        """
        route: list[str] = []
        for record in self.ledger(run_id).records():
            if record.task_id != task_id:
                continue
            if record.kind == "turn.started":
                route.append("plan" if record.payload.get("phase") == "plan" else "implement")
            elif record.kind in ("checkpoint", "verify", "review"):
                route.append(record.kind)
            elif record.kind == "note" and record.payload.get("event") == "pull_request.opened":
                route.append("open_pr")
            elif record.kind == "note" and record.payload.get("event") == "publish.skipped":
                # The publish step RAN and recorded why it published nothing. Kept as its own route
                # entry rather than dropped, so "the route ends at review because `open_pr` never
                # ran" stays distinguishable from "it ran and skipped".
                route.append("open_pr_skipped")
        return route

    def checkpoints(self, run_id: str = "run-1") -> list[dict[str, Any]]:
        return [
            record.payload
            for record in self.ledger(run_id).records()
            if record.kind == "checkpoint"
        ]

    def turn_keys(self) -> list[str]:
        return [turn.turn_key for turn in self.sidecar.turns]

    def session_ids(self) -> list[str]:
        return sorted(self.sidecar.live_sessions, key=lambda value: int(value.rsplit("-", 1)[-1]))


def build_harness(
    repo: Path,
    runs_root: Path,
    *,
    cost_per_turn: float = 0.01,
    tokens_per_turn: int = 10,
    measure_overlap: bool = False,
    turn_hold_s: float = 0.0,
) -> Harness:
    """Wire a runtime to the fakes and install it as the process default."""
    sidecar = FakeSidecar(cost_per_turn=cost_per_turn, tokens_per_turn=tokens_per_turn)
    if measure_overlap:
        # `time.monotonic`, not the injected `clock`: the injected clock is FROZEN at 1_700_000_000 for
        # determinism, and a frozen clock cannot measure an interval. This is the harness observing the
        # wall clock, which is legitimate here precisely because the assertion is about ELAPSED overlap
        # rather than about anything the product decides.
        sidecar.overlap_clock = time.monotonic
        sidecar.turn_hold_s = turn_hold_s
    paths = _paths(runs_root)
    harness = Harness(repo=repo, runs_root=runs_root, sidecar=sidecar, runtime=None)  # type: ignore[arg-type]
    runtime = ActivityRuntime(
        runs_root=runs_root,
        sidecar_factory=lambda _owner: sidecar,
        ledger_factory=lambda run_id: RunLedger.for_run(
            paths, run_id, clock=lambda: 1_700_000_000.0
        ),
        clock=lambda: 1_700_000_000.0,
        heartbeat=lambda details: None,
        log=harness.notifications.append,
        github_token=lambda: "test-token",
        run_tests=harness.run_tests,
        run_review=harness.run_review,
        push_and_open=harness.push_and_open,
        # The CI boundary is genuinely external (GitHub's checks API), so it is the one seam a test
        # scripts. The activity's own body still runs: it validates, classifies and ledgers.
        watch_ci=harness.watch_ci,
        git=None,  # real git: the checkpoint's property is a property of git
    )
    harness.runtime = runtime
    configure_default_runtime(runtime)
    return harness


def _paths(runs_root: Path) -> DurablePaths:
    """A `DurablePaths` whose runs root is the test's temp directory."""
    root = runs_root.parent
    return DurablePaths(
        root=root,
        temporal_dir=root / "temporal",
        dev_db_path=root / "temporal" / "dev.db",
        dev_server_bin_dir=root / "temporal" / "bin",
        runs_root=runs_root,
        logs_dir=root / "logs",
    )


# -- a real git repository ---------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr}"
    return completed.stdout


def make_repo(path: Path, *, origin: str | None = GITHUB_ORIGIN) -> Path:
    """A real repository on ``main`` with one commit, ready for the checkpoint activity.

    ``origin`` is a REAL remote (`git remote add`), not a stubbed helper, because it is what the
    publish step probes: with one, a task publishes (through the harness's scripted push); with
    ``None`` the repository is local-only and publishing is SKIPPED. A fixture that faked the origin
    would test the fake, and the "no remote" case is exactly what this suite must be able to build.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@codna.ai")
    git(path, "config", "user.name", "test")
    if origin:
        git(path, "remote", "add", "origin", origin)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")
