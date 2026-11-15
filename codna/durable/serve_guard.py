"""`--guard-parent-pid` for `codna serve`: the same supervision contract the worker already has.

── WHY THE RUN API NEEDS THIS AT ALL ─────────────────────────────────────────────────────────────

The desktop app now STARTS the loopback Run API (`codna serve`) the same way it starts the worker, so
the same question applies to it: what happens when the app dies without running a line of cleanup?

A parent cannot run code when it is SIGKILLed. Measured against the worker before `--guard-parent-pid`
existed: `kill -9 <app>` left the child reparented to PPID 1, still holding its port and its state, and
the next launch competed with it. The Run API is worse than the worker in one respect — it OWNS the
listen socket — so an orphan would make the next launch refuse with `run_api_bind_failed` while a
perfectly good server it cannot authenticate against keeps the port.

── WHY THIS IS A SHARED MODULE AND NOT A SECOND COPY ────────────────────────────────────────────

The worker's contract lives in `parent_guard.py` and its flag handling in `worker_cli.py`. The Run API
needs BOTH halves, identically: validate that the pid names this process's parent (a bare flag is not a
contract), and move onto the ordinary SIGTERM path when it goes away (never `os._exit`, so the server
drains its streams and reports its own stop). Re-implementing either half here would be a second
definition of one contract, and the second one is the one that drifts.

So the worker's own functions are REUSED verbatim:

  * the validation is `worker_cli._validate_guard_parent_pid` — the same refusal codes
    (`worker_parent_pid_invalid`, `worker_parent_pid_mismatch`) on both surfaces, so a client branching
    on why a guarded process refused does not have to know which command it was; and
  * the armed-then-disarmed lifecycle is a context manager so a command cannot install a guard and
    forget to stop it on the way out (a live timer in a process that is exiting is a thread nobody joins).

**Nothing here imports `temporalio`**, so the base CLI keeps answering `codna serve --help` without the
durable extra installed.
"""
from __future__ import annotations

import contextlib
import os
import signal
from typing import Any, Final, Iterator

__all__ = [
    "GUARD_EVENTS",
    "GUARD_FLAG",
    "arm_parent_guard",
    "guard_parent_pid",
]

#: The flag, spelled once. Both `codna worker` and `codna serve` register THIS string.
GUARD_FLAG: Final = "--guard-parent-pid"

#: The events a guarded process logs, so an operator can tell an app-owned one from a hand-run one.
#: `worker.parent_lost` already exists and is asserted by `test_worker_parent_guard_flag.py`; the run
#: API adds the second so a listener table can name WHICH process the app owns.
GUARD_EVENTS: Final = ("parent.lost", "run_api.parent_lost")


def guard_parent_pid(raw: Any) -> int | None:
    """The validated parent pid, or `None` for a hand-run process.

    Delegates to the worker's validator rather than restating it: the contract is "the value must name
    THIS process's parent", and a second implementation is a second chance to get that wrong. The import
    is local because `worker_cli` imports the durable package's CLI module at module scope.
    """
    from ..durable.worker_cli import _validate_guard_parent_pid

    return _validate_guard_parent_pid(raw)


def _log_parent_lost(reason: str) -> None:
    """Say which process the app owned and why it is stopping, on the structured channel."""
    from ..durable.worker import _log_json

    _log_json("run_api.parent_lost", reason=reason, signal="SIGTERM")


def _signal_self_for_parent_loss(reason: str) -> None:
    """Move a guarded server onto its standard SIGTERM path when its desktop parent dies."""
    _log_parent_lost(reason)
    os.kill(os.getpid(), signal.SIGTERM)


@contextlib.contextmanager
def arm_parent_guard(parent_pid: int | None) -> Iterator[None]:
    """Arm the guard for the duration of the caller's block, and disarm it afterwards.

    DISARMED ON THE WAY OUT, always: a clean stop is not a lost parent, and a timer left armed in a
    process that is shutting down is a thread that could fire a SIGTERM into the shutdown itself.
    `None` means this process was not started by a supervisor (a human typed `codna serve`), and a
    terminal losing its shell must not abandon an API a teammate's window is driving — so no guard.

    Armed BEFORE the config, the dev server and the socket, for the worker's reason: a crash during
    startup leaves an orphan just as effectively as one during a run.
    """
    if parent_pid is None:
        yield
        return
    from ..durable.parent_guard import install_parent_guard

    guard = install_parent_guard(_signal_self_for_parent_loss, parent_pid=parent_pid)
    try:
        yield
    finally:
        guard.stop()
