"""Network-jail backends — real kernel-level egress denial for the sandbox.

The `Sandbox` records its network policy and delegates enforcement to a `network_backend` with
a `wrap(argv, network) -> argv` method. This module provides the real backends: a fresh network
namespace via `bwrap --unshare-net` or `unshare --net`. When the policy denies egress the command
is wrapped so it runs with NO network interfaces; otherwise it passes through unchanged.

`default_network_backend()` picks the best available enforcer — and returns `None` off Linux (or
when no tool is present), where the sandbox falls back to recording the policy only. That gap is
exactly the residual risk the plan flags: real egress denial is verified by a Linux-runner test
(see test_netjail's gated case), not assumed on every platform.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

_DENY = {"deny", "none"}


class NoopNetworkBackend:
    """Records-only: never actually blocks egress (used where no enforcer is available)."""

    name = "noop"

    def wrap(self, argv, network: str):
        return list(argv)


class UnshareBackend:
    name = "unshare"

    def wrap(self, argv, network: str):
        if network in _DENY:
            return ["unshare", "--net", "--", *argv]
        return list(argv)


class BubblewrapBackend:
    name = "bwrap"

    def wrap(self, argv, network: str):
        if network in _DENY:
            return ["bwrap", "--unshare-net", "--dev-bind", "/", "/", "--", *argv]
        return list(argv)


def _command_succeeds(argv: list[str]) -> bool:
    try:
        return subprocess.run(argv, capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def can_enforce_bubblewrap_netns() -> bool:
    """True only if bubblewrap can actually create a network namespace on this host."""
    return sys.platform == "linux" and bool(shutil.which("bwrap")) and _command_succeeds(
        ["bwrap", "--unshare-net", "--dev-bind", "/", "/", "--", "true"]
    )


def can_enforce_unshare_netns() -> bool:
    """True only if unshare can actually create a network namespace on this host."""
    return sys.platform == "linux" and bool(shutil.which("unshare")) and _command_succeeds(
        ["unshare", "--net", "true"]
    )


def default_network_backend():
    """Best real enforcer for this host, or None if egress can't be hard-denied here.

    Hosted Linux runners may have `unshare` installed but lack the privilege/userns capability to
    use it. Probe capability before selecting a backend so normal `codna fix --tests` discovery does
    not fail with `unshare: Operation not permitted`; fail-closed callers still reject None.
    """
    if sys.platform != "linux":
        return None
    if can_enforce_bubblewrap_netns():
        return BubblewrapBackend()
    if can_enforce_unshare_netns():
        return UnshareBackend()
    return None


def can_enforce_netns() -> bool:
    """True only if a supported backend can actually create a network namespace on this host."""
    return can_enforce_bubblewrap_netns() or can_enforce_unshare_netns()
