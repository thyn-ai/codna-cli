"""THE REAL-SIGNAL HALF: a SIGKILLed parent must not leave a listening worker behind.

`test_parent_guard.py` proves the decision with an injected ppid reader and an injected scheduler.
This file proves the mechanism itself against the kernel, because the failure this guard exists to
prevent is precisely the one a fake cannot reproduce: a parent that dies without running any code.

Shape, and why it is three processes:

    pytest  ──spawns──▶  parent (a `sleep`-alike)  ──spawns──▶  child (the guard)

The child is the process under test and its parent is the one we SIGKILL. Killing the test's OWN
parent would kill the test runner, so the middle process exists only to be a real, killable parent.

TWO tests, because there are two independent signals and each must be proven on its own:

  * `test_a_sigkilled_parent_stops_the_guarded_child` — the child's stdin is `/dev/null`, so the
    stdin-EOF signal is NOT available to it. The PPID POLL has to be what fires, which is the signal
    that must work in a process nobody wired a pipe to.
  * `test_a_closed_pipe_stops_the_guarded_child` — the child's stdin is a real pipe whose write end
    lives in the parent. SIGKILL closes that end, so the child stops on a READ rather than on a poll.

── WHY THE EVIDENCE IS A FILE AND NOT `kill(pid, 0)` ─────────────────────────────────────────────

MEASURED on this host while writing these tests: `os.kill(pid, 0)` reports "the process exists" for a
zombie — a child reaped by the kernel-adjacent reaper on macOS keeps its pid reserved until the
reaper retires it, so a pid check kept saying "alive" for a child that had ALREADY stopped and exited.
Using it as the pass/fail signal made a working guard look broken. So each child writes one JSON line
to a path the test named, immediately before exiting, and the assertions are:

  1. the child reported a stop, and the reason is the signal the test isolated; and
  2. the process is no longer RUNNING (`ps -o stat=` says absent, or `Z`, or `E`).

Both facts together are what "no orphan" means: exited, and not a live process.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

CLI_ROOT = Path(__file__).resolve().parents[2]

# How long a reparented child may take to notice. The default poll is 1000 ms; this is generous so a
# loaded CI runner reports the real behaviour rather than a timing flake.
EXIT_DEADLINE_S = 15.0

CHILD_SCRIPT = textwrap.dedent(
    """
    import json, os, sys, time, threading
    from codna.durable.parent_guard import install_parent_guard

    evidence_path = sys.argv[1]
    stopped = threading.Event()

    def on_lost(reason):
        # A stop EVENT, not sys.exit: the guard runs on a timer thread, and sys.exit there ends only
        # that thread while the main loop keeps going. This mirrors the worker's real contract, where
        # the guard sets an asyncio stop event on the loop that owns the worker.
        with open(evidence_path + ".stop", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"reason": reason, "ppid": os.getppid()}))
        stopped.set()

    install_parent_guard(on_lost, poll_ms=100)
    # Announce readiness only AFTER the guard is armed, so the test never kills the parent during the
    # window between spawn and `install_parent_guard` (which would make this a test of nothing).
    print(json.dumps({"armed": True, "started_ppid": os.getppid()}), flush=True)
    while not stopped.is_set():
        time.sleep(0.05)
    with open(evidence_path + ".exited", "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    sys.exit(0)
    """
)

# `stdin_mode` is "devnull" or "pipe". In "pipe" mode the parent keeps the write end, so SIGKILLing it
# closes the only other end of the child's stdin.
PARENT_SCRIPT = textwrap.dedent(
    """
    import subprocess, sys, time
    stdin_mode, evidence_path = sys.argv[1], sys.argv[2]
    stdin = subprocess.DEVNULL if stdin_mode == "devnull" else subprocess.PIPE
    child = subprocess.Popen(
        [sys.executable, "-c", {child_literal!r}, evidence_path],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Relay the child's readiness line so the test can wait for the guard to be armed.
    print(child.stdout.readline().strip(), flush=True)
    print(child.pid, flush=True)
    time.sleep(120)
    """
)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = [str(CLI_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    # Never write bytecode into the checkout under test.
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path.home() / ".codna" / "pycache"))
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _is_running(pid: int) -> bool:
    """Whether `pid` is a live (non-zombie, non-exiting) process. `kill(pid, 0)` is NOT this."""
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = result.stdout.strip()
    if not state:
        return False
    # `Z` is a reaped-but-unretired child, `E` is a process that is exiting: neither is running work
    # and neither can hold a task queue.
    return not state.startswith(("Z", "E"))


def _spawn(stdin_mode: str) -> tuple[subprocess.Popen[str], int, Path]:
    evidence = Path(tempfile.mkdtemp(prefix="codna-parent-guard-")) / "evidence"
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT_SCRIPT.format(child_literal=CHILD_SCRIPT), stdin_mode, str(evidence)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    armed_line = parent.stdout.readline().strip() if parent.stdout else ""
    child_pid_line = parent.stdout.readline().strip() if parent.stdout else ""
    assert armed_line, "the child never armed its guard"
    armed = json.loads(armed_line)
    assert armed.get("armed") is True, f"the child did not arm: {armed_line!r}"
    assert child_pid_line.isdigit(), f"the parent did not report the child pid: {child_pid_line!r}"
    return parent, int(child_pid_line), evidence


def _assert_child_goes_away(parent: subprocess.Popen[str], child_pid: int, evidence: Path, reason: str) -> None:
    try:
        # THE SIGNAL THE GUARD EXISTS FOR: the parent runs no code at all.
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)

        deadline = time.monotonic() + EXIT_DEADLINE_S
        while time.monotonic() < deadline and not evidence.with_suffix(".exited").exists():
            time.sleep(0.05)

        stop_file = evidence.with_suffix(".stop")
        assert stop_file.exists(), (
            f"the guarded child (pid {child_pid}) never noticed its parent was SIGKILLed within "
            f"{EXIT_DEADLINE_S:g}s — this is the orphan the guard exists to prevent"
        )
        stopped = json.loads(stop_file.read_text(encoding="utf-8"))
        assert stopped["reason"] == reason, f"the guard fired for the wrong signal: {stopped!r}"
        # The OTHER half: it must actually exit, not merely decide to. The guard sets a stop event; the
        # process still has to act on it, and a guard that only logged would leave the worker alive.
        assert evidence.with_suffix(".exited").exists(), (
            "the guard reported the lost parent but the process never exited — a stop event nobody "
            "awaits is the same orphan with a log line"
        )
        time.sleep(0.2)
        assert not _is_running(child_pid), f"pid {child_pid} is still running after its parent was SIGKILLed"
    finally:
        if parent.poll() is None:  # pragma: no cover - only on an assertion failure above
            parent.kill()
            parent.wait(timeout=10)
        # Never leave the process under test behind, even when an assertion above failed.
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL/parent-death semantics differ on Windows")
def test_a_sigkilled_parent_stops_the_guarded_child() -> None:
    """The PPID POLL alone: the child's stdin is `/dev/null`, so EOF cannot be the signal."""
    parent, child_pid, evidence = _spawn("devnull")
    _assert_child_goes_away(parent, child_pid, evidence, "parent-reparented")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL/parent-death semantics differ on Windows")
def test_a_closed_pipe_stops_the_guarded_child() -> None:
    """The STDIN signal alone: the parent's write end vanishes with it, so the read reports EOF."""
    parent, child_pid, evidence = _spawn("pipe")
    _assert_child_goes_away(parent, child_pid, evidence, "parent-stdin-closed")
