"""The worker must not outlive the app that forked it.

MEASURED on this host with a hand-started worker: kill the app, and `ps -o pid,ppid,command` shows the
worker reparented to PPID 1, still polling the task queue and still holding the sidecar. A parent
cannot run code when it is SIGKILLed, so the only mechanism that covers every parent death is one
that runs IN THE CHILD and observes the parent's absence.

The guard is tested as LOGIC (an injected ppid reader, an injected scheduler, an injected stdin
watcher), because the properties that matter are "which signal fires", "does it fire exactly once" and
"does it stay silent in a process that was never supervised". The real-signal half is proven by
`test_parent_guard_real.py`, which forks a real child and SIGKILLs its parent.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Callable

from codna.durable.parent_guard import (
    DEFAULT_POLL_MS,
    REASON_REPARENTED,
    REASON_STDIN_EOF,
    ParentGuard,
)


class ManualClock:
    """A scheduler a test drives by hand: `tick()` runs every armed callback, like a timer firing."""

    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []
        self.delays: list[int] = []
        self.cancelled = 0

    def schedule(self, callback: Callable[[], None], delay_ms: int) -> Callable[[], None]:
        self.pending.append(callback)
        self.delays.append(delay_ms)

        def cancel() -> None:
            self.cancelled += 1
            if callback in self.pending:
                self.pending.remove(callback)

        return cancel

    def tick(self) -> None:
        for callback in list(self.pending):
            if callback in self.pending:
                self.pending.remove(callback)
                callback()

    def live(self) -> int:
        """Timers still scheduled — i.e. the wakeups that can still fire. 0 means fully disarmed."""
        return len(self.pending)


class FakeParent:
    """A ppid reader whose value a test changes, exactly as the OS does on a reparent."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def __call__(self) -> int:
        return self.pid


def make_guard(
    *,
    ppid: FakeParent,
    clock: ManualClock,
    poll_ms: int = DEFAULT_POLL_MS,
) -> tuple[ParentGuard, list[str], dict[str, Callable[[], None]]]:
    losses: list[str] = []
    stdin_callbacks: dict[str, Callable[[], None]] = {}

    def watch_stdin(callback: Callable[[], None]) -> None:
        stdin_callbacks["cb"] = callback

    guard = ParentGuard(
        losses.append,
        read_ppid=ppid,
        schedule=clock.schedule,
        watch_stdin=watch_stdin,
        poll_ms=poll_ms,
    )
    return guard, losses, stdin_callbacks


# ── activation ────────────────────────────────────────────────────────────────────────────────


def test_the_guard_reports_the_parent_it_was_installed_under() -> None:
    ppid = FakeParent(4242)
    guard, _losses, _stdin = make_guard(ppid=ppid, clock=ManualClock())
    assert guard.started_ppid == 4242


def test_a_live_parent_is_never_a_reason_to_stop() -> None:
    """The poll re-arms itself while the parent is alive. A guard that stopped an alive worker would
    turn every long run into a coin flip on a timer."""
    ppid = FakeParent(4242)
    clock = ManualClock()
    guard, losses, _stdin = make_guard(ppid=ppid, clock=clock)
    guard.start()
    for _ in range(5):
        clock.tick()
    assert losses == []
    # One initial arm plus one re-arm per tick that found the parent alive.
    assert len(clock.delays) == 6
    assert all(delay == DEFAULT_POLL_MS for delay in clock.delays)


# ── the two signals ──────────────────────────────────────────────────────────────────────────


def test_a_reparented_child_stops_and_names_the_reason() -> None:
    ppid = FakeParent(4242)
    clock = ManualClock()
    guard, losses, _stdin = make_guard(ppid=ppid, clock=clock)
    guard.start()
    # The OS renumbers the child's parent when the original dies (1 on macOS/Linux, a subreaper else).
    ppid.pid = 1
    clock.tick()
    assert losses == [REASON_REPARENTED]


def test_stdin_eof_stops_the_worker_without_waiting_for_a_poll() -> None:
    """The immediate signal: the parent's write end closing is observable at once, while the reparent
    may not be visible until the next poll. Measured on this host, EOF arrives first."""
    ppid = FakeParent(4242)
    clock = ManualClock()
    guard, losses, stdin = make_guard(ppid=ppid, clock=clock)
    guard.start()
    assert "cb" in stdin
    stdin["cb"]()
    assert losses == [REASON_STDIN_EOF]


def test_both_signals_fire_the_stop_exactly_once() -> None:
    """A reparent AND an EOF are the same fact. Two stop requests would mean two drains, and the second
    one arrives after the worker has already shut its activities down."""
    ppid = FakeParent(4242)
    clock = ManualClock()
    guard, losses, stdin = make_guard(ppid=ppid, clock=clock)
    guard.start()
    ppid.pid = 1
    clock.tick()
    stdin["cb"]()
    clock.tick()
    assert losses == [REASON_REPARENTED]


def test_a_stop_disarms_the_poll() -> None:
    """The graceful path stops the guard so a clean shutdown is never attributed to a lost parent."""
    ppid = FakeParent(4242)
    clock = ManualClock()
    guard, losses, _stdin = make_guard(ppid=ppid, clock=clock)
    guard.start()
    guard.stop()
    assert clock.cancelled == 1
    ppid.pid = 1
    clock.tick()
    assert losses == []


# ── the disarm is FINAL, against a concurrent stop() ─────────────────────────────────────────
#
# Each test places `stop()` (or a second signal) at the ONE instant the old code lost the race: after a
# poll had read the ppid and decided to re-arm, but before it stored the new timer's canceller. The seam
# is the injected `read_ppid`, which the poll calls BETWEEN those two steps — so the interleaving is
# exact and deterministic, with no sleeps and no thread timing. On the old code (`_cancel_poll = ...`
# outside any lock, `_fired` read unsynchronised, `stop()` not final) each of these leaves a live timer
# that the guard no longer holds a canceller for, and the next tick signals a worker told to stop.


class _StopDuringPollRead(FakeParent):
    """An alive parent whose ppid READ runs `stop()` on this thread, inside `_poll_once`.

    That is exactly the window the finding names: the poll has passed the `_fired` check and read the
    (unchanged) ppid, so on the old code it proceeds to arm a successor with the handle `stop()` just
    cleared. Production reaches the same state when the main thread calls `stop()` on a signal while a
    timer callback is running on the timer thread.
    """

    def __init__(self, pid: int, stop: Callable[[], None]) -> None:
        super().__init__(pid)
        self._stop = stop
        # `ParentGuard.__init__` reads the ppid once too; the hook must only fire on the POLL read.
        self.armed = False

    def __call__(self) -> int:
        if self.armed:
            self.armed = False
            self._stop()
        return self.pid


def test_a_stop_during_the_poll_leaves_the_guard_disarmed() -> None:
    """REAL — `_poll_once` re-armed after `stop()` because the arm and the disarm were unsynchronised.

    After the fix, `stop()` is final: the poll's re-arm sees `_stopped` and installs nothing.
    """
    clock = ManualClock()
    guard_ref: list[ParentGuard] = []
    ppid = _StopDuringPollRead(4242, lambda: guard_ref[0].stop())
    guard, losses, _stdin = make_guard(ppid=ppid, clock=clock)
    guard_ref.append(guard)
    guard.start()
    assert clock.live() == 1

    ppid.armed = True
    clock.tick()  # the poll runs; `stop()` lands inside its ppid read

    assert clock.live() == 0, "stop() must be final: the poll left a timer armed after the shutdown"
    # And the timer that would have been armed cannot signal: nothing is armed to fire.
    ppid.pid = 1
    clock.tick()
    assert losses == [], "a stopped guard fired a parent-loss signal"


class _SignalDuringPollRead(FakeParent):
    """An alive parent whose ppid READ fires the OTHER signal (stdin EOF) on this thread.

    This is the `_fire` race: the fire sets `_fired`, then on the old code RELEASES the lock before
    `stop()`, and the poll that already read the ppid re-arms in between. The orphaned timer wakes a
    guard that already reported the parent gone.
    """

    def __init__(self, pid: int, fire: Callable[[], None]) -> None:
        super().__init__(pid)
        self._fire = fire
        # `ParentGuard.__init__` reads the ppid once too; the hook must only fire on the POLL read.
        self.armed = False

    def __call__(self) -> int:
        if self.armed:
            self.armed = False
            self._fire()
        return self.pid


def test_a_signal_while_the_poll_re_arms_cancels_the_new_timer() -> None:
    """REAL — `_fire` called `stop()` only after releasing the lock, so a poll could arm in between.

    After the fix, `_fire` sets `_fired`/`_stopped` and takes the live canceller in ONE locked step, and
    the re-arm re-checks those flags after arming and cancels what it just created.
    """
    clock = ManualClock()
    signals: dict[str, Callable[[], None]] = {"stdin": lambda: None}

    def watch_stdin(callback: Callable[[], None]) -> None:
        signals["stdin"] = callback

    losses: list[str] = []
    ppid = _SignalDuringPollRead(4242, lambda: signals["stdin"]())
    guard = ParentGuard(
        losses.append,
        read_ppid=ppid,
        schedule=clock.schedule,
        watch_stdin=watch_stdin,
    )
    guard.start()

    ppid.armed = True
    clock.tick()  # the poll reads a live ppid; EOF fires inside that read

    assert losses == [REASON_STDIN_EOF], "the lost parent must be reported exactly once"
    assert clock.live() == 0, "a timer armed around the fire survived it"
    ppid.pid = 1
    clock.tick()
    assert losses == [REASON_STDIN_EOF], "the fire must not repeat"


# ── the validated pid, not a re-read (the TOCTOU the flag validation left open) ─────────────


def test_the_guard_watches_the_supplied_parent_not_a_re_read() -> None:
    """The guard must observe the pid the CALLER VALIDATED, not whatever the OS answers next.

    `--guard-parent-pid` is checked against `os.getppid()` by the CLI and then the guard used to read
    the ppid AGAIN. Between those two reads the answer can change (the parent dies and the child is
    reparented, or a pid is reused), so the guard would watch a process nobody validated. Passing the
    validated value closes that window; this asserts the guard uses it verbatim.
    """
    read_calls: list[int] = []

    def read_ppid() -> int:
        read_calls.append(1)
        return 999_999  # a "current" parent that must NOT be the one watched

    losses: list[str] = []
    clock = ManualClock()
    guard = ParentGuard(
        losses.append,
        read_ppid=read_ppid,
        schedule=clock.schedule,
        watch_stdin=None,
        started_ppid=4242,
    )
    guard.start()
    assert guard.started_ppid == 4242, "the guard watched a pid other than the validated one"
    assert read_calls == [], "the guard re-read the ppid it was given"

    # And the CONSEQUENCE, not just the field: the OS says the parent is 999_999, which is not the
    # validated 4242, so this guard reports a lost parent on its first poll.
    clock.tick()
    assert losses == [REASON_REPARENTED], "a guard watching the validated pid must notice it is gone"

# ── the stdin channel must actually BE a supervision channel ────────────────────────────────
#
# MEASURED while writing `test_parent_guard_real.py`: a child spawned with `stdin=DEVNULL` arms an EOF
# watcher that fires on its FIRST read, so the worker stops the instant it starts. `/dev/null` is not
# a pipe with a remote write end — nothing can ever "close" it — and treating it as one is the guard
# reaping a healthy process.


def test_devnull_stdin_never_arms_an_eof_stop() -> None:
    from codna.durable.parent_guard import pipe_stdin_watcher

    with open(os.devnull, "r", encoding="utf-8") as devnull:
        calls: list[None] = []
        pipe_stdin_watcher(devnull)(lambda: calls.append(None))
        time.sleep(0.05)
    assert calls == []


def test_a_real_pipe_reports_eof_when_its_write_end_closes() -> None:
    from codna.durable.parent_guard import pipe_stdin_watcher

    read_fd, write_fd = os.pipe()
    stopped = threading.Event()
    try:
        with os.fdopen(read_fd, "r", encoding="utf-8") as read_end:
            pipe_stdin_watcher(read_end)(stopped.set)
            # The write end is still open: no EOF, so no stop. This is the "parent is alive" state the
            # watcher must not confuse with the failure it is looking for.
            assert not stopped.wait(0.05)
            os.close(write_fd)
            assert stopped.wait(5.0), "closing the write end must report the parent's absence"
    finally:
        with contextlib.suppress(OSError):
            os.close(write_fd)
