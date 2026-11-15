"""Sandbox — the privilege-enforcement primitive the analysis worker runs untrusted code in.

The worker executes attacker-influenced code (scanners, build, tests, generated patches), so
it must NEVER hold a write-capable token, and its egress must be denied. This module enforces
what Python can enforce on any platform — scrubbing write credentials from the environment and
detecting their presence (PRIV-01/02) — and *records* the network policy + bounds every command
with a timeout and a fixed cwd.

Honest scope: true kernel-level egress denial (namespaces/seccomp/firejail) is platform-specific
and is verified by a gated Linux-runner test, not here (see the plan's residual-risks section).
A real deployment supplies that enforcement via `network_backend`; the default records the policy
so `--open-pr` can refuse a manifest that doesn't deny egress (checked upstream in manifest.py).
"""
from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass, field

# Env var names that carry write-capable credentials and must not reach the worker.
WRITE_TOKEN_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "CODNA_GITHUB_TOKEN",
    "GH_PAT",
    "GITHUB_PAT",
    "GITHUB_APP_PRIVATE_KEY",
)
# Value patterns for GitHub tokens (classic/oauth/user/server/refresh/fine-grained).
WRITE_TOKEN_VALUE = re.compile(r"\b(gh[posru]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})")

#: The shell's own report that it could not resolve a program. The three shapes are the three shells
#: `verify` actually runs under, and they are NOT interchangeable — this was measured by running each
#: one with a missing binary:
#:
#: * ``/bin/sh: node: command not found``      — macOS's `/bin/sh`, and bash in POSIX mode
#: * ``/bin/bash: node: command not found``     — bash
#: * ``sh: 1: node: not found``                 — dash, which is what ``/bin/sh`` IS on Debian and
#:   therefore on `ubuntu-latest`. Note the word order AND the shorter message: a detector written for
#:   "command not found" alone reports nothing on CI, which would make this whole coded error dead
#:   code in the environment the suite runs in.
#: * ``zsh:1: command not found: node``         — zsh, the other word order
#:
#: Only the SHELL'S MESSAGE is matched, never the command line: `sh` may be running a pipeline, an
#: `&&` chain or a subshell, so the first word of the command is not reliably the missing program. The
#: match is anchored to the START of a stderr line, because that is where a shell writes its own
#: diagnostic — a TAP line that merely CONTAINS the phrase (`not ok: command not found`) is a test's
#: output, not the shell's report about the toolchain.
_COMMAND_NOT_FOUND = re.compile(
    r"^(?:[^\n:]*/)?(?:sh|bash|zsh|dash)(?:: ?(?:line )?\d+)?: "
    r"(?:([A-Za-z0-9._+-]+): command not found"
    r"|command not found: ([A-Za-z0-9._+-]+)"
    r"|([A-Za-z0-9._+-]+): not found)$",
    re.MULTILINE,
)

_NETWORK_DENY = {"deny", "none"}
_FORBIDDEN_NAMES = {n.upper() for n in WRITE_TOKEN_NAMES}

#: Termination budget of a timed-out command, in seconds. Together these two bounds define the hard
#: ceiling the class promises: `timeout_seconds` + `_TERMINATION_GRACE_S` + `_DRAIN_GRACE_S`.
#: `_TERMINATION_GRACE_S` is how long SIGTERM is given before SIGKILL; `_DRAIN_GRACE_S` is the most
#: time spent collecting whatever bytes the capture pipes already hold.
_TERMINATION_GRACE_S = 1.0
_DRAIN_GRACE_S = 1.0
_READ_SIZE = 32768
#: How often the drain loop re-checks the child's liveness while waiting on the pipes.
_POLL_SLICE_S = 0.05


class PrivilegeSeparationError(Exception):
    pass


def _offending_keys(env: dict) -> list[str]:
    bad = [k for k in env if k.upper() in _FORBIDDEN_NAMES]
    for k, v in env.items():
        if isinstance(v, str) and WRITE_TOKEN_VALUE.search(v):
            bad.append(k)
    return sorted(set(bad))


def assert_no_write_credentials(env: dict) -> None:
    """Raise if the environment contains any write-capable credential (PRIV-02)."""
    bad = _offending_keys(env)
    if bad:
        raise PrivilegeSeparationError(f"worker environment contains write credential(s): {bad}")


def scrub_env(env: dict) -> dict:
    """Return a copy of `env` with write credentials (by name OR value pattern) removed."""
    return {
        k: v
        for k, v in env.items()
        if k.upper() not in _FORBIDDEN_NAMES and not (isinstance(v, str) and WRITE_TOKEN_VALUE.search(v))
    }


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    argv: list[str]
    cwd: str
    network: str
    #: How the run's PATH was decided (see `codna.verify_env`). Carried on the RESULT, not only on the
    #: Sandbox, because the thing that must be diagnosable is the finished attempt: the run that
    #: exited 127 has to say which PATH it exited 127 under.
    env_note: str = ""

    def accepted(self, accepted_exit_codes=(0,)) -> bool:
        return (not self.timed_out) and self.returncode in tuple(accepted_exit_codes)

    def command_not_found(self) -> str | None:
        """The name the shell could not find, when the failure is a shell's "command not found".

        Distinguishing this from an ordinary non-zero exit is what stops a missing TOOLCHAIN from
        being reported as a failing TEST. Measured (`run-20261113T225225Z`): three verifies returned
        127, the graph read each as "the tests are red", two paid implement turns were spent
        rewriting a correct app, and the task then parked on an exhausted attempt budget — so the
        user approved a run whose own verdict had never once been true. The command was not missing.

        The identity is read out of the SHELL'S OWN message rather than guessed from the command
        line: `sh` may be running a pipeline, an `&&` chain or a subshell, so the first word of the
        command is not reliably the missing program. ``None`` means "not this shape" — an ordinary
        failure, which the caller must keep treating as a red result and never as a pass.
        """
        if self.returncode != 127 or self.timed_out:
            return None
        match = _COMMAND_NOT_FOUND.search(self.stderr or "")
        if match is None:
            return None
        return match.group(1) or match.group(2) or match.group(3)


@dataclass
class Sandbox:
    network: str = "deny"
    timeout_seconds: int = 1800
    env: dict = field(default_factory=dict)
    network_backend: object = None  # optional egress-deny wrapper (Linux runner)
    #: Where `env["PATH"]` came from, when the caller resolved it (see `codna.verify_env`). Recorded
    #: verbatim on every result so a failed attempt can be diagnosed from its log alone.
    env_note: str = ""

    def __post_init__(self):
        # The sandbox is the enforcement boundary: it DEMANDS a clean env and raises on any
        # write credential (PRIV-02). Callers (the worker) scrub the ambient env with
        # scrub_env() before handing it over — CI environments routinely carry GITHUB_TOKEN.
        assert_no_write_credentials(self.env)

    @property
    def denies_network(self) -> bool:
        return self.network in _NETWORK_DENY

    def run(self, argv, *, cwd: str, accepted_exit_codes=(0,)) -> SandboxResult:
        """Run a command under the sandbox: scrubbed env, fixed cwd, bounded timeout, and the
        recorded network policy. Never raises on a non-zero exit — callers inspect the result.

        INVARIANT: a sandboxed command cannot outlive ``timeout_seconds`` by more than
        ``_TERMINATION_GRACE_S + _DRAIN_GRACE_S`` — regardless of what it spawns. That is why this
        does not use ``subprocess.run(..., timeout=...)``: that kills only the DIRECT child and then
        blocks in ``communicate()`` until the capture pipes reach EOF, so a grandchild that inherited
        stdout/stderr keeps the call blocked well past the deadline. Measured here: a shell that
        backgrounded ``sleep 20`` and exited at once made ``subprocess.run(timeout=3)`` return after
        20.0s — the timeout never fired at all, because the direct child had already exited.

        Three properties make the bound real:
          1. the command runs in its OWN process group (``start_new_session=True``), so the whole
             tree — children and grandchildren — is signalled at once;
          2. on timeout the GROUP is terminated (SIGTERM, a short grace, then SIGKILL), never a
             single pid;
          3. the pipes are drained on a deadline, so a survivor that still holds them open cannot
             block the return. Whatever has been read is kept as stdout/stderr.
        """
        argv = list(argv)
        wrapped = self.network_backend.wrap(argv, network=self.network) if self.network_backend else argv
        deadline = time.monotonic() + self.timeout_seconds
        # DEVNULL, not the worker's fd 0. Inherited stdin is both a hang and a correctness hazard:
        # a command like a bare `python3` READS ITS PROGRAM from fd 0, so it would execute bytes from
        # the worker's stdin (observed: a real verify sat in CPython's `pymain_run_stdin` on a socket
        # fd 0 and had to be killed by hand). The worker's stdin is not the command's input; a command
        # that reads stdin must get EOF immediately.
        proc = subprocess.Popen(
            wrapped,
            cwd=cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out_chunks, err_chunks = self._drain(proc, deadline)
            timed_out = False
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            # `exc.stdout`/`exc.stderr` hold what the drain had already read; the termination step
            # appends the bytes still buffered in the pipes, so a timed-out attempt still reports its
            # output instead of an empty string.
            out_chunks, err_chunks = self._terminate_group(proc, exc.stdout, exc.stderr)
            timed_out = True
            # 124 is the documented "the sandbox stopped it" code; the signal that produced it is an
            # implementation detail and must not leak into a caller's exit-code comparison.
            returncode = 124
        finally:
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
        return SandboxResult(
            returncode,
            _decode(out_chunks, proc.encoding),
            _decode(err_chunks, proc.encoding),
            timed_out,
            argv,
            cwd,
            self.network,
            self.env_note,
        )

    @staticmethod
    def _drain(proc: subprocess.Popen, deadline: float) -> tuple[list[bytes], list[bytes]]:
        """Read both capture pipes until EOF or the child's exit; raise on the deadline.

        Raises ``subprocess.TimeoutExpired`` — carrying the bytes read so far as ``output`` — when
        ``deadline`` arrives while the child is still running. Once the child has exited, no further
        time is budgeted: the loop keeps only what is already buffered and returns. That is what stops
        a detached survivor holding the write end open from turning this into an unbounded wait.
        """
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        streams = {proc.stdout.fileno(): out_chunks, proc.stderr.fileno(): err_chunks}
        post_exit_deadline: float | None = None
        with selectors.DefaultSelector() as selector:
            for fd in streams:
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ)
            while selector.get_map():
                exited = proc.poll() is not None
                if exited:
                    # After the exit the pipes are only drained, never waited on: one slice of quiet
                    # means every byte the command wrote is in hand, and a survivor that keeps
                    # writing cannot extend this past the drain grace.
                    if post_exit_deadline is None:
                        post_exit_deadline = time.monotonic() + _DRAIN_GRACE_S
                    remaining = post_exit_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    wait = min(remaining, _POLL_SLICE_S)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            proc.args, 0, output=b"".join(out_chunks), stderr=b"".join(err_chunks)
                        )
                    wait = min(remaining, _POLL_SLICE_S)
                ready = selector.select(wait)
                for key, _ in ready:
                    try:
                        chunk = os.read(key.fd, _READ_SIZE)
                    except BlockingIOError:  # pragma: no cover - selector said readable
                        continue
                    if not chunk:
                        selector.unregister(key.fd)
                    else:
                        streams[key.fd].append(chunk)
                if exited and not ready:
                    break
        # Both pipes are done (EOF or drained); the child may still be running with them closed, so
        # it gets the rest of its budget — and no more, which is what preserves the timeout bound.
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0))
        except subprocess.TimeoutExpired:
            # Still running past the deadline even though its pipes are closed: hand the already-read
            # output to the termination path instead of discarding it.
            raise subprocess.TimeoutExpired(
                proc.args, 0, output=b"".join(out_chunks), stderr=b"".join(err_chunks)
            ) from None
        return out_chunks, err_chunks

    @staticmethod
    def _terminate_group(
        proc: subprocess.Popen, already_read_out: bytes | None, already_read_err: bytes | None
    ) -> tuple[list[bytes], list[bytes]]:
        """Terminate the command's whole process group and collect what was already written.

        Returns the chunks read so far — the bytes the drain captured before the deadline plus
        whatever the pipes still hold. SIGTERM is given ``_TERMINATION_GRACE_S`` to take effect and
        SIGKILL follows; both target the group, so grandchildren are covered.

        ORDER IS LOAD-BEARING: the group is signalled to completion BEFORE the leader is reaped.
        Reaping frees the pgid for reuse, and a later `killpg` on that number can then hit an
        unrelated group — observed as `PermissionError: [Errno 1] Operation not permitted` when the
        second signal landed on a process this user does not own. An unreaped leader keeps the pgid
        allocated, so every signal below provably addresses only this command's tree.
        """
        pgid = proc.pid  # start_new_session=True makes the child the leader of its own group
        _signal_group(pgid, signal.SIGTERM)
        _await_group_exit(pgid, _TERMINATION_GRACE_S)
        # SIGKILL the group before the leader is reaped: a live member keeps the pgid allocated, so
        # this cannot address anything but this command's tree. `_signal_group` ignores EPERM, which
        # macOS reports for a group already emptied down to the unreaped leader.
        _signal_group(pgid, signal.SIGKILL)
        _await_group_exit(pgid, _POLL_SLICE_S)
        try:
            # Only now may the leader be reaped. Its tree has already received SIGKILL, so the only
            # remaining case is a leader stuck in uninterruptible I/O; bound the reap rather than
            # blocking the sandbox on it forever.
            proc.wait(timeout=_TERMINATION_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal_group(pgid, signal.SIGKILL)
            proc.wait()
        out_chunks, err_chunks = _read_available(proc)
        if already_read_out:
            out_chunks.insert(0, already_read_out)
        if already_read_err:
            err_chunks.insert(0, already_read_err)
        return out_chunks, err_chunks


def _await_group_exit(pgid: int, timeout: float) -> None:
    """Wait at most ``timeout`` for every process in ``pgid`` to exit, WITHOUT reaping the leader.

    Liveness is probed with signal 0, which reports whether the group still has members and changes
    nothing. The leader stays unreaped here so its pgid cannot be recycled while it is still being
    signalled — see `Sandbox._terminate_group`.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_has_live_members(pgid):
            return
        time.sleep(_POLL_SLICE_S)


def _group_has_live_members(pgid: int) -> bool:
    """Whether ``pgid`` still holds a live (signalable) process. Raises nothing."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        # ESRCH: the group is empty. EPERM: measured on macOS, `killpg` reports EPERM for a group
        # whose only remaining member is the exited-but-unreaped leader, while a group with ANY live
        # member accepts the signal. Both therefore mean "nothing left to escalate against", and
        # treating EPERM as "still alive" would only burn the whole grace on a group that is gone.
        return False
    return True


def _signal_group(pgid: int, sig: int) -> None:
    """Signal ``pgid``; an already-empty group is not an error (see `_group_has_live_members`)."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _read_available(proc: subprocess.Popen) -> tuple[list[bytes], list[bytes]]:
    """Read whatever the capture pipes already hold, bounded by ``_DRAIN_GRACE_S``.

    The group has been killed, so this normally reaches EOF at once. The deadline exists for a
    survivor that left the group and still holds the write end: it bounds the read, it never waits.
    """
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    streams = {proc.stdout.fileno(): out_chunks, proc.stderr.fileno(): err_chunks}
    deadline = time.monotonic() + _DRAIN_GRACE_S
    with selectors.DefaultSelector() as selector:
        for fd in streams:
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready = selector.select(min(remaining, _POLL_SLICE_S))
            if not ready:
                # Nothing buffered in this slice: the bytes the pipes had are already in hand.
                break
            for key, _ in ready:
                try:
                    chunk = os.read(key.fd, _READ_SIZE)
                except BlockingIOError:  # pragma: no cover - selector said readable
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                else:
                    streams[key.fd].append(chunk)
    return out_chunks, err_chunks


def _decode(chunks: list[bytes], encoding: str | None) -> str:
    """Decode captured bytes the way ``text=True`` would, tolerating a truncated multi-byte tail."""
    return b"".join(chunks).decode(encoding or "utf-8", "replace")
