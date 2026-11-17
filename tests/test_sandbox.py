"""Sandbox tests (A–Z test plan: PRIV-01/02/03 privilege enforcement, timeout, cwd).

The timeout tests are about the INVARIANT, not the happy path: a sandboxed command must not outlive
its ``timeout_seconds`` by more than a small fixed grace, no matter what it spawns. A bare
``subprocess.run(timeout=...)`` cannot promise that — it kills only the direct child and then waits
for the capture pipes to reach EOF — so these tests use a grandchild that keeps those pipes open.
"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from codna.sandbox import (
    PrivilegeSeparationError,
    Sandbox,
    _TERMINATION_GRACE_S,
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
def _sacrificial_group_leader():
    """A real, live process group that is SAFE for a test to signal.

    ``Sandbox._terminate_group`` derives ``pgid = proc.pid`` and signals that group with SIGTERM and
    SIGKILL. Any stand-in for that leader therefore *really* signals whatever process group its ``pid``
    names, and ``_signal_group`` suppresses only ``ProcessLookupError``/``PermissionError`` — a
    successful delivery cannot be caught. A mock pid of ``os.getpid()`` (or ``0``, or the runner's
    current pgid) makes the test signal THE TEST RUNNER'S OWN GROUP: on CI, where pytest is usually
    the group leader, that kills the runner and its siblings and the whole suite dies mid-run. Keep
    this a genuine separate group, never the runner's own.

    ``start_new_session=True`` gives the child its own session, so it is the leader of a fresh process
    group and its pid can never be the runner's. It sleeps in Python rather than shelling out so it
    carries no other process, and it is SIGKILLed and reaped on the way out so nothing is left behind.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    # Belt and braces: the whole point is that this group is NOT the runner's. If a future edit
    # drops `start_new_session=True`, fail loudly here instead of signalling the runner's group.
    assert child.pid != os.getpgid(0), "the sacrificial leader must not be in the test runner's group"
    try:
        yield child
    finally:
        # The test has already signalled this group; this is the guarantee that the sacrificial
        # process is gone even if the test failed before reaching that point. PermissionError is
        # expected here for the same reason `_signal_group` ignores it: macOS reports EPERM for a
        # group whose only remaining member is the exited-but-unreaped leader.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            child.kill()
        # BOUNDED, for the same reason the production code is: a child in uninterruptible kernel I/O
        # cannot take SIGKILL until that I/O completes, and an unbounded reap here would hang the
        # runner. This helper exists to demonstrate that rule; it must follow it. The grace matches
        # the production termination grace rather than inventing a second number.
        try:
            child.wait(timeout=_TERMINATION_GRACE_S)
        except subprocess.TimeoutExpired:
            # D-state child: deliberately abandoned. `pytest` must not outlive the suite on its account.
            pass


@contextlib.contextmanager
def _fd0_is_socket(payload: bytes | None = None, keep_open: bool = True):
    """Make this process's fd 0 a UNIX socket, the shape a worker's stdin actually has.

    ``payload`` is served to whoever reads fd 0; with ``keep_open`` the peer is never closed, so a
    reader that waits for EOF blocks rather than returning. This is how the sandbox's stdin policy is
    observed without depending on how pytest itself was started.

    ``None`` and ``b""`` are distinct inputs and now distinct code paths: ``None`` states "serve
    nothing", while ``b""`` states "serve a zero-length payload" — which, under this helper's
    keep-open contract, is an empty send rather than a skipped one. Keying on truthiness (the previous
    ``theirs.sendall(payload) if payload else None``) collapsed the two, so a caller asking for an
    empty payload got no send at all and had to reason about whether that was intended.
    """
    ours, theirs = socket.socketpair()
    if payload is not None:
        theirs.sendall(payload)
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
def test_sandbox_final_reap_is_bounded_when_the_leader_never_dies(tmp_path):
    """THE A1 REGRESSION: the post-SIGKILL reap must be bounded, not a bare `proc.wait()`.

    A leader stuck in uninterruptible kernel I/O cannot be killed by SIGKILL and remains unreapable
    until that I/O finishes, so an unbounded final `proc.wait()` blocks `run()` for an arbitrary
    length of time — breaking the ceiling `_terminate_group` documents. A D-state process cannot be
    fabricated deterministically in a test, so the leader is modelled at the seam the code actually
    calls: `Popen.wait` reports `TimeoutExpired` TWICE (once for the bounded reap, once for the
    bounded second attempt) and a plain call never returns. If the implementation ever issues a bare
    `wait()`, this test hangs instead of passing — so the assertion is on the CALL SHAPE, which is
    what makes it deterministic rather than wall-clock dependent.

    The mock's ``pid`` is a real sacrificial process in ITS OWN process group, because
    ``_terminate_group`` turns ``proc.pid`` into a pgid and signals it for real. It must never be this
    process's pid: that would send SIGTERM then SIGKILL to the test runner's own group, and a
    successful delivery is not suppressible — the suite would die instead of failing. See
    `_sacrificial_group_leader`.
    """
    calls: list[float | None] = []
    # Real pipe read-ends so `_read_available` observes the same shape it does in production; nothing is
    # written to them, so it returns at once (the group has already been killed).
    out_fd, out_w = os.pipe()
    err_fd, err_w = os.pipe()

    class PipeEnd:
        def __init__(self, fd: int):
            self._fd = fd

        def fileno(self) -> int:
            return self._fd

    with _sacrificial_group_leader() as sacrificial:

        class UndyingLeader:
            """A leader that never terminates, whatever it is signalled with."""

            #: A SEPARATE process group (the sacrificial child is its own session leader). Never
            #: `os.getpid()`: `_terminate_group` would then SIGTERM/SIGKILL the test runner's group.
            pid = sacrificial.pid
            args = ["undying"]
            encoding = "utf-8"
            stdout = PipeEnd(out_fd)
            stderr = PipeEnd(err_fd)

            def wait(self, timeout=None):
                calls.append(timeout)
                if timeout is None:
                    raise AssertionError("the final reap was unbounded: proc.wait() with no timeout")
                raise subprocess.TimeoutExpired(self.args, timeout)

        started = time.monotonic()
        try:
            out, err = Sandbox._terminate_group(UndyingLeader(), b"captured-out", b"captured-err")
        finally:
            for fd in (out_fd, out_w, err_fd, err_w):
                os.close(fd)
        elapsed = time.monotonic() - started

    # Both reaps were bounded (never `None`), so the call cannot outlive 2 x the grace.
    assert calls, "the leader was never reaped"
    assert all(timeout is not None for timeout in calls), calls
    assert elapsed < 2 * _TERMINATION_GRACE_S + BOUND_SLACK_S, f"termination took {elapsed:.1f}s"
    # The captured output is still returned, and no error escapes: the command is abandoned, not failed.
    assert b"".join(out) == b"captured-out"
    assert b"".join(err) == b"captured-err"


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


@POSIX_ONLY
def test_fd0_helper_serves_an_empty_payload_and_skips_only_no_payload(monkeypatch):
    """THE A2 REGRESSION: `b""` is a payload, so the helper must SEND it; only `None` means "nothing".

    The old expression ``theirs.sendall(payload) if payload else None`` was both an expression
    statement (a bare side effect) and wrong for one input: an empty payload took the "no payload"
    branch. Bytes on a socket cannot make the difference visible (`sendall(b"")` moves nothing), so the
    semantics are pinned where they are actually observable — the call itself — by recording what the
    helper asks the peer to send. The branch is keyed on ``payload is not None``, which is the helper's
    documented contract ("serve ``payload`` to whoever reads fd 0").
    """
    sent: list[bytes] = []
    real_socketpair = socket.socketpair

    class RecordingPeer:
        """Forwards to the real socket, recording every payload the helper serves."""

        def __init__(self, sock):
            self._sock = sock

        def sendall(self, data: bytes) -> None:
            sent.append(data)
            self._sock.sendall(data)

        def __getattr__(self, name):
            return getattr(self._sock, name)

    def recording_socketpair():
        ours, theirs = real_socketpair()
        return ours, RecordingPeer(theirs)

    monkeypatch.setattr(socket, "socketpair", recording_socketpair)

    with _fd0_is_socket(b""):
        pass
    assert sent == [b""], "an empty payload was skipped instead of served"

    sent.clear()
    with _fd0_is_socket():
        pass
    assert sent == [], "no payload is not a payload: nothing should be sent"

    sent.clear()
    with _fd0_is_socket(b"ping"):
        pass
    assert sent == [b"ping"], "a non-empty payload must still be served verbatim"

