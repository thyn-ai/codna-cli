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

    def accepted(self, accepted_exit_codes=(0,)) -> bool:
        return (not self.timed_out) and self.returncode in tuple(accepted_exit_codes)


@dataclass
class Sandbox:
    network: str = "deny"
    timeout_seconds: int = 1800
    env: dict = field(default_factory=dict)
    network_backend: object = None  # optional real egress-deny wrapper (Linux runner)

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
            return SandboxResult(proc.returncode, proc.stdout, proc.stderr, False, argv, cwd, self.network)
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(124, exc.stdout or "", exc.stderr or "", True, argv, cwd, self.network)
