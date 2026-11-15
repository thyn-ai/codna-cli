"""Sandbox tests (A–Z test plan: PRIV-01/02/03 privilege enforcement, timeout, cwd).

The timeout tests are about the INVARIANT, not the happy path: a sandboxed command must not outlive
its ``timeout_seconds`` by more than a small fixed grace, no matter what it spawns. A bare
``subprocess.run(timeout=...)`` cannot promise that — it kills only the direct child and then waits
for the capture pipes to reach EOF — so these tests use a grandchild that keeps those pipes open.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time

import pytest

from codna.sandbox import (
    PrivilegeSeparationError,
    Sandbox,
    assert_no_write_credentials,
    scrub_env,
)

PATH_ONLY = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}

#: Wall-clock slack allowed on top of ``timeout_seconds``. The sandbox's own bound is
#: ``timeout + termination grace + drain grace`` (2 s with the production constants); this is the
#: outer limit a test asserts, chosen to stay far below the "minutes" a pipe-EOF hang would cost
#: while leaving room for a loaded CI host.
BOUND_SLACK_S = 5.0

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX-only")


def _sandbox(timeout_seconds: int = 2) -> Sandbox:
    return Sandbox(env=dict(PATH_ONLY), timeout_seconds=timeout_seconds)


def _pids_matching(marker: str) -> list[int]:
    """PIDs whose command line contains ``marker`` — how a surviving grandchild is detected."""
    out = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return [int(pid) for pid in out.stdout.split()]


def _absent(marker: str, wait_s: float = 3.0) -> bool:
    """Whether every process carrying ``marker`` is gone, allowing the kernel a moment to reap."""
    deadline = time.monotonic() + wait_s
    while _pids_matching(marker):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


@contextlib.contextmanager
def _no_survivors(marker: str, wait_s: float = 5.0):
    """Fail-safe cleanup: no sleeper started by a test outlives it, pass or fail.

    Each test below tags its background children with a unique marker in their argv. On the way out
    this SIGKILLs whatever is left and waits briefly for it to disappear, so a regression can never
    leave minutes-long processes behind on the machine running the suite. Signalling first (rather
    than waiting for the processes to leave of their own accord) is what keeps the suite fast.
    """
    try:
        yield marker
    finally:
        for pid in _pids_matching(marker):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)
        deadline = time.monotonic() + wait_s
        while _pids_matching(marker) and time.monotonic() < deadline:
            time.sleep(0.05)


@contextlib.contextmanager
def _fd0_is_socket(payload: bytes | None = None, keep_open: bool = True):
    """Make this process's fd 0 a UNIX socket, the shape a worker's stdin actually has.

    ``payload`` is served to whoever reads fd 0; with ``keep_open`` the peer is never closed, so a
    reader that waits for EOF blocks rather than returning. This is how the sandbox's stdin policy is
    observed without depending on how pytest itself was started.
    """
    ours, theirs = socket.socketpair()
    theirs.sendall(payload) if payload else None
    saved = os.dup(0)
    os.dup2(ours.fileno(), 0)
    try:
        yield theirs
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        for sock in (ours, theirs):
            sock.close()


def test_scrub_env_removes_write_tokens_by_name_and_value():
    env = {"PATH": "/bin", "GITHUB_TOKEN": "x", "CODNA_GITHUB_TOKEN": "y",
           "SOME_VAR": "ghp_" + "a" * 36, "SAFE": "value"}
    scrubbed = scrub_env(env)
    assert "GITHUB_TOKEN" not in scrubbed
    assert "CODNA_GITHUB_TOKEN" not in scrubbed
    assert "SOME_VAR" not in scrubbed  # removed by value pattern (looks like a PAT)
    assert scrubbed["SAFE"] == "value" and scrubbed["PATH"] == "/bin"


def test_assert_no_write_credentials():
    assert_no_write_credentials({"PATH": "/bin"})  # no raise
    with pytest.raises(PrivilegeSeparationError):
        assert_no_write_credentials({"GH_TOKEN": "abc"})
    with pytest.raises(PrivilegeSeparationError):
        assert_no_write_credentials({"WHATEVER": "github_pat_" + "a" * 30})


def test_sandbox_construction_rejects_write_token():
    with pytest.raises(PrivilegeSeparationError):
        Sandbox(env={"GITHUB_TOKEN": "ghp_" + "z" * 36})


def test_sandbox_runs_command_in_cwd(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY))
    res = sb.run(["python3", "-c", "import os;print(os.getcwd())"], cwd=str(tmp_path))
    assert res.returncode == 0
    assert res.cwd == str(tmp_path)
    assert res.network == "deny" and sb.denies_network is True
    assert os.path.realpath(res.stdout.strip()) == os.path.realpath(str(tmp_path))


def test_sandbox_accepted_exit_codes(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY))
    res = sb.run(["python3", "-c", "import sys;sys.exit(3)"], cwd=str(tmp_path))
    assert res.returncode == 3
    assert res.accepted((0,)) is False
    assert res.accepted((0, 3)) is True


def test_sandbox_timeout(tmp_path):
    sb = Sandbox(env=dict(PATH_ONLY), timeout_seconds=1)
    res = sb.run(["python3", "-c", "import time;time.sleep(5)"], cwd=str(tmp_path))
    assert res.timed_out is True
    assert res.accepted((0,)) is False


def test_sandbox_keeps_stdout_and_stderr_for_a_normal_and_a_failing_command(tmp_path):
    """The capture contract is unchanged by the group-termination rewrite: both streams, both exits."""
    sb = _sandbox()
    ok = sb.run(["/bin/sh", "-c", "echo out; echo err 1>&2"], cwd=str(tmp_path))
    assert (ok.returncode, ok.timed_out, ok.stdout.strip(), ok.stderr.strip()) == (0, False, "out", "err")

    bad = sb.run(["/bin/sh", "-c", "echo out; echo err 1>&2; exit 7"], cwd=str(tmp_path))
    assert (bad.returncode, bad.timed_out) == (7, False)
    assert bad.accepted((0,)) is False and bad.accepted((0, 7)) is True
    assert bad.stdout.strip() == "out" and bad.stderr.strip() == "err"


@POSIX_ONLY
def test_sandbox_timeout_is_bounded_when_a_grandchild_holds_the_pipes(tmp_path):
    """THE regression: a grandchild inheriting stdout/stderr must not extend the timeout.

    Before the fix this test hung for the grandchild's lifetime: `subprocess.run(timeout=...)` killed
    only the shell and then blocked in `communicate()` waiting for the pipes to reach EOF, which never
    happened while the backgrounded `sleep` held them. A 45-minute verify ceiling that a child can
    postpone indefinitely is not a ceiling at all.
    """
    marker = f"codna_sandbox_bound_probe_{os.getpid()}"
    timeout_seconds = 2
    with _no_survivors(marker):
        started = time.monotonic()
        res = _sandbox(timeout_seconds).run(
            [
                "/bin/sh",
                "-c",
                f'python3 -c "import time;time.sleep(120)" {marker} & echo spawned; sleep 120',
            ],
            cwd=str(tmp_path),
        )
        elapsed = time.monotonic() - started
        assert res.timed_out is True
        assert res.returncode == 124
        assert res.accepted((0,)) is False
        assert elapsed < timeout_seconds + BOUND_SLACK_S, (
            f"took {elapsed:.1f}s with a {timeout_seconds}s timeout"
        )
        # The command's own output before the deadline is kept, not discarded.
        assert res.stdout.strip() == "spawned"
        # Both the shell AND the pipe-holding grandchild are gone when `run` returns.
        assert _absent(marker), "a grandchild survived the timeout"


@POSIX_ONLY
def test_sandbox_terminates_the_whole_process_group_on_timeout(tmp_path):
    """No orphan outlives the sandbox: the GRANDCHILD is signalled, not just the direct child."""
    marker = f"codna_sandbox_group_probe_{os.getpid()}"
    with _no_survivors(marker):
        res = _sandbox(2).run(
            [
                "/bin/sh",
                "-c",
                # The marker rides in the grandchild's argv, which is what `pgrep -f` matches.
                f'python3 -c "import time;time.sleep(120)" {marker} & echo spawned; sleep 120',
            ],
            cwd=str(tmp_path),
        )
        assert res.timed_out is True and res.returncode == 124
        assert _absent(marker), "a grandchild survived the sandbox timeout"


@POSIX_ONLY
def test_sandbox_returns_promptly_when_the_child_exits_but_a_grandchild_holds_the_pipes(tmp_path):
    """The other half of the invariant: a finished command is not held hostage by a survivor.

    The direct child exits in milliseconds; the backgrounded grandchild keeps the capture pipes open
    for two minutes. The call must return on the child's exit, not on the eventual pipe EOF.

    The survivor here is real — the test asserts that a detached `exit 0` tree is NOT killed — so the
    cleanup helper is what removes it afterwards.
    """
    marker = f"codna_sandbox_detach_probe_{os.getpid()}"
    with _no_survivors(marker):
        started = time.monotonic()
        res = _sandbox(30).run(
            ["/bin/sh", "-c", f'python3 -c "import time;time.sleep(120)" {marker} & echo spawned; exit 0'],
            cwd=str(tmp_path),
        )
        elapsed = time.monotonic() - started
        assert (res.timed_out, res.returncode) == (False, 0)
        assert res.stdout.strip() == "spawned"
        assert elapsed < BOUND_SLACK_S, f"waited {elapsed:.1f}s for a command that exited at once"


@POSIX_ONLY
def test_sandbox_stdin_is_devnull_so_a_reader_gets_eof(tmp_path):
    """A command that reads stdin must not block on the worker's stdin.

    Observed in production: a bare `python3` sat in CPython's `pymain_run_stdin` reading fd 0, which
    was a UNIX socket, and had to be killed by hand. Here fd 0 is that same shape and never reaches
    EOF, so only `stdin=DEVNULL` can make this return.
    """
    with _fd0_is_socket():
        started = time.monotonic()
        res = _sandbox(10).run(["/bin/sh", "-c", "cat; echo done"], cwd=str(tmp_path))
        elapsed = time.monotonic() - started
    assert (res.timed_out, res.returncode) == (False, 0)
    assert res.stdout.strip() == "done"
    assert elapsed < BOUND_SLACK_S, f"command blocked {elapsed:.1f}s reading inherited stdin"


@POSIX_ONLY
def test_sandbox_stdin_cannot_be_executed_as_a_program(tmp_path):
    """Bytes on the worker's stdin must never be run as a program.

    CPython reads its SCRIPT from fd 0 when no file is given, so the inherited stdin was not merely a
    hang — it was an execution channel from the worker into the sandbox.
    """
    marker = tmp_path / "executed_from_stdin"
    program = f"open({str(marker)!r}, 'w').write('ran')\n"
    with _fd0_is_socket(program.encode()):
        res = _sandbox(10).run(["/bin/sh", "-c", "python3"], cwd=str(tmp_path))
    assert res.timed_out is False
    assert not marker.exists(), "the sandbox executed a program read from the worker's stdin"

