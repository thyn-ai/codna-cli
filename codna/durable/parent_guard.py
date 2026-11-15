"""THE WORKER MUST NOT OUTLIVE THE APP THAT STARTED IT.

── THE MEASURED DEFECT ──────────────────────────────────────────────────────────────────────────

The packaged IDE spawns `codna worker` as a child of its backend. A parent cannot run code when it is
SIGKILLed — that is the whole point of SIGKILL — so a signal-handler-only cleanup covers the graceful
quit and NOTHING ELSE. Measured on this host against an earlier, hand-started worker: kill the app,
and `ps -o pid,ppid,command -p <worker>` shows the worker reparented to PPID 1, still polling the task
queue, still holding the sidecar connection, with the next launch competing with it for the same
queue and the same ~/.codna state. The same shape's precedent in this repository is the TypeScript
`parentGuard.ts` (which reaps the Theia backend when the Electron main dies); this is its Python twin
for the one process the APP now forks itself.

── WHY THE CHILD DECIDES ────────────────────────────────────────────────────────────────────────

Two independent signals, both measured to fire on a SIGKILLed parent, and either one sufficient:

  * a PPID POLL — the child re-reads `os.getppid()`; when the parent dies the OS reparents the child,
    so the value changes to the reaper's pid (1 on macOS and Linux, a subreaper elsewhere). This is
    the signal that needs nothing from the parent at all, which is why it is the only one that
    covers a fork that was never wired to a pipe.
  * STDIN EOF — when the app spawns the worker with `stdin: "pipe"` (which it does, so the child has a
    lifecycle independent of a terminal), the read end reports EOF the moment the parent's write end
    goes away. This fires before the reparent poll on most platforms, so a stop is usually immediate
    rather than up to `poll_s` late.

── WHAT THIS MUST NOT DO ───────────────────────────────────────────────────────────────────────

It must not fire in a process that was NOT started as a supervised child (`codna worker` typed by a
human in a terminal). The activation is explicit — `--guard-parent-pid <pid>` — so a hand-run worker
behaves exactly as it always did and the guard cannot be installed by accident. It must not fire on a
transient value (the poll only reacts to a CHANGE from the pid it started under), and it must never
throw: a guard that crashes the worker is worse than the orphan it prevents.

The decision is pure and injected (`ParentGuard` takes the ppid reader, the scheduler and the exit
callback separately), so it is tested without spawning anything.
"""
from __future__ import annotations

import os
import stat
import sys
import threading
from typing import Any, Callable, Final

__all__ = [
    "DEFAULT_POLL_MS",
    "ParentGuard",
    "ParentLossReason",
    "install_parent_guard",
    "pipe_stdin_watcher",
]

# A reparented child is reaped within a second of the change being observable; polling faster than
# that buys nothing but wakeups, and polling much slower lets a dead app hold a task queue for a
# noticeable time. One second is the same value `parentGuard.ts` uses.
DEFAULT_POLL_MS: Final = 1000

# Why the guard decided to stop. Named so the log line is greppable and a test can assert on it.
ParentLossReason = str
REASON_REPARENTED: Final = "parent-reparented"
REASON_STDIN_EOF: Final = "parent-stdin-closed"


class ParentGuard:
    """Observe one parent's liveness and call `on_lost` exactly once when it goes away.

    Everything external is injected so the decision is testable with no process, no timer and no
    tty:

      * `read_ppid` — `os.getppid` in production.
      * `schedule` — `(callback, delay_ms) -> canceller`, wrapping `threading.Timer` in production.
      * `watch_stdin` — `(callback) -> None`, installing the EOF observer in production.
      * `on_lost` — the stop action, called ONCE with the reason.
      * `started_ppid` — the ALREADY-VALIDATED parent. Supplied by a caller that checked the value
        against the process table (see `worker_cli._validate_guard_parent_pid`); omitted, this guard
        reads it. Reading it a second time is a TOCTOU window: the caller validated one pid and the
        guard would watch whichever pid the OS reports a moment later.

    ── THE LOCK, AND WHY `stop()` IS THE HARD DISARM ────────────────────────────────────────────

    A timer callback and `stop()` run on different threads, so "check the flags, then arm" is not one
    step unless it is taken under `_lock`. Without that, a `_poll_once` already past its flag read
    re-arms AFTER `stop()` cleared the handle, and the new timer is live and uncancellable: a guard
    that keeps observing — and can still signal — a worker that is shutting down. That matters more
    here than in ordinary code: this callback TERMINATES A RUNNING WORKER, so the conservative direction
    is to miss a parent death (the app's own supervision reaps the worker) rather than to fire on a
    worker that was told to stop. `_stopped` is therefore set under the same lock and checked by every
    arm and by `_fire`, which makes `stop()` final.
    """

    def __init__(
        self,
        on_lost: Callable[[ParentLossReason], None],
        *,
        read_ppid: Callable[[], int] = os.getppid,
        schedule: Callable[[Callable[[], None], int], Callable[[], None]] | None = None,
        watch_stdin: Callable[[Callable[[], None]], None] | None = None,
        poll_ms: int = DEFAULT_POLL_MS,
        started_ppid: int | None = None,
    ) -> None:
        self._on_lost = on_lost
        self._read_ppid = read_ppid
        self._schedule = schedule or _timer_scheduler
        self._watch_stdin = watch_stdin or _stdin_eof_watcher
        self._poll_ms = poll_ms
        self._started_ppid = read_ppid() if started_ppid is None else started_ppid
        self._fired = False
        self._stopped = False
        self._lock = threading.Lock()
        self._cancel_poll: Callable[[], None] | None = None

    @property
    def started_ppid(self) -> int:
        """The parent this guard was installed under. Read so the banner can report it."""
        return self._started_ppid

    def start(self) -> None:
        """Arm both signals. Idempotent; safe to call once from the worker's startup path."""
        self._watch_stdin(lambda: self._fire(REASON_STDIN_EOF))
        with self._lock:
            if self._stopped or self._fired or self._cancel_poll is not None:
                return
            self._cancel_poll = self._schedule(self._poll_once, self._poll_ms)

    def stop(self) -> None:
        """Disarm, permanently. Called on the graceful path so a clean shutdown is not a lost parent.

        FINAL, by construction: `_stopped` is set under `_lock`, and `_rearm` refuses once it is set,
        so a poll already in flight cannot install a timer after this returns.
        """
        with self._lock:
            self._stopped = True
            cancel = self._cancel_poll
            self._cancel_poll = None
        if cancel is not None:
            cancel()

    def _poll_once(self) -> None:
        # The ppid read is outside the lock on purpose: it is the ONE fact this callback exists to
        # learn, and holding a lock across it would serialise the stop path behind a syscall.
        current = self._read_ppid()
        if current != self._started_ppid:
            self._fire(REASON_REPARENTED)
            return
        self._rearm()

    def _rearm(self) -> None:
        # Re-arm. A chain of one-shot timers (rather than an interval) means a slow callback cannot
        # queue up behind itself, and the canceller always names the timer that is actually live.
        #
        # THE DISARM IS CHECKED AGAIN AFTER ARMING, and that second check is what makes `stop()` final.
        # Arming can RUN code on this thread (`threading.Timer.start` above all), so a concurrent
        # `stop()` can land inside the arming step and clear `_cancel_poll` before this assignment
        # happens: the timer would then be live with nobody holding its canceller. So the flags are
        # read again under the lock, and a timer armed after the flag was set is cancelled here by the
        # holder that just created it.
        with self._lock:
            if self._stopped or self._fired:
                self._cancel_poll = None
                return
            cancel = self._schedule(self._poll_once, self._poll_ms)
            if self._stopped or self._fired:
                self._cancel_poll = None
                cancel()
                return
            self._cancel_poll = cancel

    def _fire(self, reason: ParentLossReason) -> None:
        # ONE transition for both signals, under the lock: the flags are set and the live timer is
        # taken and cleared together, so neither a second signal nor an in-flight poll can arm or fire
        # again. `_on_lost` runs OUTSIDE the lock because it is arbitrary caller code (it sends a
        # signal) and must not be able to deadlock a concurrent `stop()`.
        with self._lock:
            if self._fired or self._stopped:
                return
            self._fired = True
            self._stopped = True
            cancel = self._cancel_poll
            self._cancel_poll = None
        if cancel is not None:
            cancel()
        self._on_lost(reason)


def _timer_scheduler(callback: Callable[[], None], delay_ms: int) -> Callable[[], None]:
    timer = threading.Timer(delay_ms / 1000.0, callback)
    # A watchdog must never hold the process open: the worker has real work keeping it alive, and a
    # non-daemon timer would turn an idle worker into one that cannot exit.
    timer.daemon = True
    timer.start()
    return timer.cancel


def _stdin_is_a_pipe(stream: Any) -> bool:
    """Whether this stdin can carry the one fact the EOF signal is about.

    MEASURED, and it is a correctness bug rather than a detail: a supervisor that spawns the worker
    with `stdin: "ignore"` gets `/dev/null`, whose very first read returns EOF. Arming the watcher
    there stopped the worker the instant it started — the guard reaping a perfectly healthy process.
    Only a FIFO or a socket has a WRITE END that can be closed by somebody else, so only those two can
    report "the parent went away". A tty never reports EOF while its session lives, and a regular file
    reports it at once; neither is a supervision channel.
    """
    try:
        mode = os.fstat(stream.fileno()).st_mode
    except (OSError, ValueError, AttributeError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def pipe_stdin_watcher(stream: Any) -> Callable[[Callable[[], None]], None]:
    """A stdin watcher bound to a GIVEN stream — the production shape, made injectable for tests.

    Tests drive this with a real `os.pipe()` and with `/dev/null`, because the difference between the
    two is the defect this function was corrected for (see `_stdin_is_a_pipe`).
    """

    def watch(callback: Callable[[], None]) -> None:
        if stream is None or getattr(stream, "closed", True) or not _stdin_is_a_pipe(stream):
            return

        def read_until_eof() -> None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
            except (OSError, ValueError):
                # A closed or invalid descriptor IS the parent being gone; a genuine read error is the
                # same fact from this process's point of view.
                pass
            callback()

        threading.Thread(target=read_until_eof, name="codna-parent-stdin", daemon=True).start()

    return watch


def _stdin_eof_watcher(callback: Callable[[], None]) -> None:
    """Call `callback` once when stdin (a supervised pipe) reaches EOF.

    The worker reads nothing from stdin, so the only event on a supervised pipe is the parent's write
    end closing. A daemon thread rather than a selector so this adds no dependency and no import cost
    to a worker started by a human (where stdin is a tty that never reaches EOF).
    """
    pipe_stdin_watcher(sys.stdin)(callback)


def install_parent_guard(
    on_lost: Callable[[ParentLossReason], None],
    *,
    poll_ms: int = DEFAULT_POLL_MS,
    parent_pid: int | None = None,
) -> ParentGuard:
    """Install the guard under the CURRENT parent and arm it. Returns the guard so it can be stopped.

    `parent_pid` is the ALREADY-VALIDATED parent (`--guard-parent-pid`). Passing it is what closes the
    TOCTOU window: this guard watches exactly the pid the caller checked against the process table,
    rather than re-reading `os.getppid()` and possibly watching a different process. Omitted, the guard
    reads the current parent, which is right for a caller that has not validated one.
    """
    guard = ParentGuard(on_lost, poll_ms=poll_ms, started_ppid=parent_pid)
    guard.start()
    return guard
