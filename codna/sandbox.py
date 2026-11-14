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

import re
import subprocess
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
        recorded network policy. Never raises on a non-zero exit — callers inspect the result."""
        argv = list(argv)
        wrapped = self.network_backend.wrap(argv, network=self.network) if self.network_backend else argv
        try:
            proc = subprocess.run(
                wrapped,
                cwd=cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return SandboxResult(
                proc.returncode, proc.stdout, proc.stderr, False, argv, cwd, self.network, self.env_note
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                124, exc.stdout or "", exc.stderr or "", True, argv, cwd, self.network, self.env_note
            )
