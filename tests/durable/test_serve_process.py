"""`codna serve` AS A PROCESS: it must still be listening, and be answering, seconds after it starts.

**The gap this file closes.** `test_run_api*.py` construct `RunApiServer` IN-PROCESS and call its
routes. That proves the surface is correct and proves nothing about the thing the IDE actually
depends on: that the `codna serve` PROCESS is still alive, bound and serving after the banner. A
server can pass every route test and still never be reachable, because the failure is in the
lifetime of the process, not in a handler.

That is exactly what shipped. `codna serve` printed a correct durable banner and then exited
silently: the process was gone seconds later and the IDE reported "The durable run API could not be
reached". The in-process suite was green throughout, because the defect was that nothing kept the
PROCESS serving — no route was called enough times to notice.

**What killed it was a signal, not its own code.** Measured on macOS: `serve_forever()` blocks
correctly and the server serves indefinitely while its controlling terminal lives. It died on
``SIGHUP`` — the hangup the kernel sends when the terminal or session that started it goes away. A
foreground server an operator started (or a harness started for them) therefore vanished the moment
the shell that started it finished. `RunApiServer.serve_forever` now ignores ``SIGHUP`` while still
stopping on ``SIGINT``/``SIGTERM``, and states the stop it performed so it never exits in silence.

**How these tests run it.** As a real subprocess in a new session, either on a REAL PTY as its
controlling terminal or on a PIPE. The pty matters for one thing only: it is what makes "the
terminal went away" an event this test can really deliver. ``nohup``, ``DEVNULL`` and ``PIPE``
never deliver that signal, which is why the original bug reproduced locally and not under the
detachment people tried.

* The liveness and auth tests run on a pty — the shape a developer's terminal gives the process.
* The regression test closes that pty and asserts the process is STILL answering. This is the
  defect that shipped, and it is the only assertion here that fails without the fix.
* The stop-record test runs on a PIPE, deliberately: when a pty's session leader exits, the kernel
  discards whatever is still queued on the terminal, so a message written on the way out is
  unobservable there. A pipe is also how the IDE spawns this process, so that is the honest place
  to assert what a caller can read.

**No long sleeps, no fixed ports.** Every wait is a bounded poll on an observable condition (the
port accepting, the banner appearing), and the port is ephemeral (``--port 0``) with the real one
read from the banner. `--no-temporal` is used throughout so the test needs neither a Temporal server
nor the optional `durable` extra: it proves the process's LIFETIME, and durable mode would only add
a dependency to that proof (`test_run_api_real.py` covers the durable backend, gated).
"""
from __future__ import annotations

import contextlib
import fcntl
import http.client
import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import termios
import time
from pathlib import Path
from typing import Any, Final, Iterator

import pytest

# The package root, so the child is started from THIS checkout rather than whatever `codna` happens
# to be on PATH — a test that ran the developer's installed CLI would prove the wrong bytes served.
CLI_DIR: Final = Path(__file__).resolve().parents[2]

# The child gets a private runtime root, so its token and SIEM files land in the test's tmp_path and
# nothing is read from (or written to) the developer's `~/.codna`.
RUNTIME_ROOT_ENV: Final = "CODNA_RUNTIME_ROOT"
SIEM_ENV: Final = "CODNA_SIEM_DIR"

# How long the process may take to print its banner and accept a connection. Generous because a cold
# import of the durable stack is slow, but BOUNDED: a fixed ceiling, not a sleep, and it fails with
# the process's own output rather than a bare timeout.
START_TIMEOUT_S: Final = 60.0


class ServingProcess:
    """A `codna serve` subprocess, with the facts a test needs read from its OWN banner."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        fd: int,
        *,
        is_pty: bool,
        owner: Any = None,
    ) -> None:
        self.process = process
        # The readable end: a pty master, or the child's stdout pipe. `-1` once it is closed.
        self.fd = fd
        self.is_pty = is_pty
        # WHO OWNS `fd`. For a pty master it is this object (nothing else holds it), so it is closed
        # with `os.close`. For a pipe it is `process.stdout`, whose buffer OWNS the descriptor: the
        # raw fd must be closed through THAT object, because closing the number directly leaves the
        # buffer to close the same number again later — and by then the number may name something
        # else.
        self._owner = owner
        self.buffer = b""
        self.banner: dict[str, Any] | None = None
        self.port: int | None = None
        self.token: str | None = None

    # -- reading ---------------------------------------------------------------------------

    def drain(self, *, timeout_s: float = 0.0) -> str:
        """Read whatever the child has written, waiting up to ``timeout_s`` for the first byte."""
        if self.fd < 0:
            return self.buffer.decode("utf-8", "replace")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                return self.buffer.decode("utf-8", "replace")
            try:
                chunk = os.read(self.fd, 65536)
            except OSError:  # EIO / EBADF: the other end is gone
                return self.buffer.decode("utf-8", "replace")
            if not chunk:  # EOF
                return self.buffer.decode("utf-8", "replace")
            self.buffer += chunk
            if self.banner is None:
                self._parse_banner()

    def _parse_banner(self) -> None:
        """The banner is one JSON document; parse it as soon as it is complete."""
        text = self.buffer.decode("utf-8", "replace")
        start = text.find("{")
        if start < 0:
            return
        try:
            payload, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return
        self.banner = payload
        run_api = payload.get("run_api") or {}
        port = run_api.get("port")
        self.port = int(port) if isinstance(port, int) else None
        token_path = run_api.get("token_path")
        if isinstance(token_path, str) and Path(token_path).is_file():
            self.token = Path(token_path).read_text(encoding="utf-8").strip()

    def text(self) -> str:
        return self.buffer.decode("utf-8", "replace")

    # -- liveness --------------------------------------------------------------------------

    def wait_until_listening(self, *, timeout_s: float = START_TIMEOUT_S) -> "ServingProcess":
        """Poll until the banner names a port that ACCEPTS, or fail with the process's own output."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.drain(timeout_s=0.1)
            if self.process.poll() is not None and self.port is None:
                pytest.fail(
                    f"`codna serve` exited with {_describe(self.process.returncode)} before it "
                    f"named a port.\n--- child output ---\n{self.text()}"
                )
            if self.port is not None and _accepts(self.port):
                return self
        pytest.fail(
            f"`codna serve` did not accept a connection on port {self.port} within {timeout_s:g}s.\n"
            f"--- child output ---\n{self.text()}"
        )

    def is_listening(self) -> bool:
        return self.port is not None and _accepts(self.port)

    def request(self, method: str, path: str, *, body: Any = None) -> tuple[int, Any]:
        """One authenticated request, the way the IDE's client makes it."""
        if self.port is None or self.token is None:
            pytest.fail(f"the server's address or token was not read: {self.text()}")
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"authorization": f"Bearer {self.token}", "accept": "application/json"}
        if payload is not None:
            headers["content-type"] = "application/json"
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10.0)
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
        finally:
            connection.close()

    # -- teardown --------------------------------------------------------------------------

    def _close_read_end(self) -> None:
        """Close the readable end through whoever OWNS it. Idempotent."""
        if self.fd < 0:
            return
        if self._owner is not None:
            # The pipe buffer owns this descriptor: close it, not the number.
            with contextlib.suppress(OSError):
                self._owner.close()
        else:
            with contextlib.suppress(OSError):
                os.close(self.fd)
        self.fd = -1

    def hangup(self) -> None:
        """Make the CONTROLLING TERMINAL go away, the way a finished shell does.

        Closing the pty master is the real event: the kernel sends ``SIGHUP`` to the session's
        foreground group. A test that instead called ``kill(SIGHUP)`` would be testing the handler
        directly, which is the weaker statement.
        """
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def stop(self) -> None:
        """Stop the child by exact PID, and never leave it running or leak its descriptor.

        EVERY STEP IS TOLERANT OF THE RACE IT CAN LOSE, because this runs from `finally` blocks: an
        exception raised here replaces the test's real failure with a teardown traceback. The child
        can exit between `poll()` and `send_signal()` (`ProcessLookupError`), `kill()` can hit the
        same already-dead process (`ProcessLookupError` again — `os.kill` re-raises it), and the
        `wait()` after either can time out.
        """
        if self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=10.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=10.0)
        self._close_read_end()


def _describe(returncode: int | None) -> str:
    if returncode is None:
        return "no exit status"
    if returncode < 0:
        return f"signal {signal.Signals(-returncode).name}"
    return f"exit status {returncode}"


def _accepts(port: int) -> bool:
    """True when something is accepting on ``port`` — the fact the IDE's first request depends on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        RUNTIME_ROOT_ENV: str(tmp_path / "runtime"),
        SIEM_ENV: str(tmp_path / "runtime" / "siem"),
        # The child is a test process: unbuffered, and pointed at a private root so it cannot reach
        # the developer's real durable state, token or Temporal profile.
        "PYTHONUNBUFFERED": "1",
    }


def _argv() -> list[str]:
    return [sys.executable, "-m", "codna.cli", "serve", "--port", "0", "--no-temporal"]


@contextlib.contextmanager
def serve_on_pty(tmp_path: Path) -> Iterator[ServingProcess]:
    """Start `codna serve --no-temporal` with a REAL PTY as its controlling terminal."""
    master_fd, slave_fd = pty.openpty()
    # No echo (it would duplicate the banner) and no canonical input: nothing here types at it.
    attributes = termios.tcgetattr(slave_fd)
    attributes[3] = attributes[3] & ~termios.ECHO
    termios.tcsetattr(slave_fd, termios.TCSANOW, attributes)

    def _claim_controlling_terminal() -> None:
        """Make the pty the CHILD's controlling terminal, in the child, after ``setsid``.

        THIS IS THE WHOLE REGRESSION TEST. A process only receives ``SIGHUP`` when its CONTROLLING
        terminal goes away, and a tty becomes controlling when the session leader claims it with
        ``TIOCSCTTY`` — merely inheriting it as fd 0/1/2 is not enough, because the descriptor was
        already open, so the kernel never treats it as "the session leader opened a terminal".
        Without this call the pty is just three inherited descriptors: closing the master delivers
        nothing, the server survives for the wrong reason, and the test passes without the fix.
        """
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    process = subprocess.Popen(
        _argv(),
        cwd=str(CLI_DIR),
        env=_environment(tmp_path),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        # `start_new_session` calls setsid() in the child, which makes it a session leader with NO
        # controlling terminal; the preexec hook then claims this pty as that terminal.
        start_new_session=True,
        preexec_fn=_claim_controlling_terminal,
        close_fds=True,
    )
    os.close(slave_fd)
    serving = ServingProcess(process, master_fd, is_pty=True)
    try:
        yield serving
    finally:
        serving.stop()


@contextlib.contextmanager
def serve_on_pipe(tmp_path: Path) -> Iterator[ServingProcess]:
    """Start `codna serve --no-temporal` with a PIPE for output — how the IDE spawns it."""
    process = subprocess.Popen(
        _argv(),
        cwd=str(CLI_DIR),
        env=_environment(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    assert process.stdout is not None  # narrowed by `stdout=PIPE`
    os.set_blocking(process.stdout.fileno(), False)
    serving = ServingProcess(
        process, process.stdout.fileno(), is_pty=False, owner=process.stdout
    )
    try:
        yield serving
    finally:
        # `stop()` closes the read end THROUGH `process.stdout`, which owns that descriptor: closing
        # the raw number would let the buffer close it a second time, possibly after the OS has
        # handed the number to something else.
        serving.stop()


@pytest.fixture
def serve(tmp_path: Path) -> Iterator[ServingProcess]:
    """The process on a pty — the shape a developer's own terminal gives it."""
    with serve_on_pty(tmp_path) as running:
        yield running


def test_the_process_is_still_listening_and_authenticates_after_the_banner(
    serve: ServingProcess,
) -> None:
    """THE CONTRACT THE IDE DEPENDS ON: alive, bound, and authorised — not merely started."""
    serve.wait_until_listening()
    banner = serve.banner or {}
    run_api = banner.get("run_api") or {}
    # The banner is the caller's only source for the address, so it must name a REAL one.
    assert run_api.get("url") == f"http://127.0.0.1:{serve.port}", banner
    assert serve.token is not None and len(serve.token) == 64, "the token must exist to be usable"

    status, body = serve.request("GET", "/capabilities")
    assert status == 200, body
    capabilities = body["capabilities"]
    # `--no-temporal` is stated as a MODE with the routes it still serves, not as a failure.
    assert capabilities["mode"] == "ledger-only"
    assert capabilities["durable_execution"] is False
    assert "runs.list" in capabilities["available_routes"]

    # A second request later proves the first did not consume the process — the assertion the
    # in-process suite structurally cannot make.
    _poll_until(serve.is_listening, timeout_s=5.0)
    status, body = serve.request("GET", "/capabilities")
    assert status == 200, body
    assert serve.process.poll() is None, (
        f"`codna serve` exited with {_describe(serve.process.returncode)} while serving"
    )


def test_an_unauthenticated_request_is_refused_by_the_real_process(serve: ServingProcess) -> None:
    """The subprocess enforces the token too: a liveness test that skipped auth could pass a
    server that authenticated nothing."""
    serve.wait_until_listening()
    connection = http.client.HTTPConnection("127.0.0.1", serve.port, timeout=10.0)
    try:
        connection.request("GET", "/capabilities")
        response = connection.getresponse()
        body = json.loads(response.read())
    finally:
        connection.close()
    assert response.status == 401, body
    assert body["error"]["code"] == "run_api_unauthorized", body


def test_the_process_survives_the_terminal_that_started_it_going_away(
    serve: ServingProcess,
) -> None:
    """THE REGRESSION THAT SHIPPED: a hangup must not kill a server that is serving callers.

    Closing the pty is the real event — the kernel sends ``SIGHUP`` to the session's foreground
    group, exactly as when the shell that started `codna serve` finishes. Before the fix the process
    died right here and the port went away, after a banner that said everything was fine.
    """
    serve.wait_until_listening()
    status, _ = serve.request("GET", "/capabilities")
    assert status == 200

    serve.hangup()

    # The port must still accept, and a NEW request must still be authorised: a process that
    # survived but stopped serving would be no better than one that died.
    _poll_until(serve.is_listening, timeout_s=10.0)
    status, body = serve.request("GET", "/capabilities")
    assert status == 200, body
    assert serve.process.poll() is None, (
        f"`codna serve` died from {_describe(serve.process.returncode)} when its terminal went away"
    )


def test_the_stop_after_a_hangup_is_a_clean_exit_not_a_failure(tmp_path: Path) -> None:
    """A requested stop is exit 0 even when the terminal that would report it is already gone.

    The trap: the stop record is written on the way out, and when the pty has been hung up there is
    no reader for it. CPython turns an unflushable ``sys.stdout`` at interpreter exit into status
    120, which would report a clean shutdown as a failure — the same class of wrong answer as the
    silent exit, pointing the other way. A supervisor reading the status must see success.
    """
    with serve_on_pty(tmp_path) as serving:
        serving.wait_until_listening()
        serving.hangup()  # the terminal is gone; the process must still stop on request
        _poll_until(serving.is_listening, timeout_s=10.0)
        serving.process.send_signal(signal.SIGTERM)
        serving.process.wait(timeout=20.0)
        assert serving.process.returncode == 0, (
            f"a requested stop exited with {_describe(serving.process.returncode)}; "
            "an unreportable stop is still a successful one"
        )


def test_a_stop_signal_is_stated_in_the_process_output(tmp_path: Path) -> None:
    """The other half of the silence: when it DOES stop, it says so.

    The original defect was undiagnosable partly because the log ended at the banner — no traceback,
    no "shutdown", nothing distinguishing "stopped" from "was killed". The process now reports the
    reason, so the next occurrence is a line of output rather than a mystery.

    On a PIPE rather than a pty: a pty's session leader exiting makes the kernel discard what is
    still queued on the terminal, so a message written on the way out is unobservable there. A pipe
    is also how the IDE spawns this process, so this asserts what a caller actually reads.
    """
    with serve_on_pipe(tmp_path) as serving:
        serving.wait_until_listening()
        serving.process.send_signal(signal.SIGTERM)
        serving.process.wait(timeout=20.0)
        _poll_until(
            lambda: '"stopped": true' in serving.drain(timeout_s=0.2), timeout_s=5.0
        )
        text = serving.text()
        assert serving.process.returncode == 0, _describe(serving.process.returncode)
        assert '"stopped": true' in text, text
        assert '"reason": "SIGTERM"' in text, text


def _poll_until(predicate: Any, *, timeout_s: float, interval_s: float = 0.05) -> None:
    """Wait for ``predicate()`` to hold, or fail. Bounded and observable, never a fixed sleep."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    pytest.fail(f"condition did not hold within {timeout_s:g}s")