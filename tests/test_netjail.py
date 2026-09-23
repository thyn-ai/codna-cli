"""Network-jail tests: wrap composition (portable) + a GATED real egress-deny test (Linux +
namespace capability only). On macOS/CI-without-privilege the real test skips by design."""
from __future__ import annotations

import os
import sys

import pytest

from codna.netjail import (
    BubblewrapBackend,
    NoopNetworkBackend,
    UnshareBackend,
    can_enforce_netns,
    default_network_backend,
)
from codna.sandbox import Sandbox

ARGV = ["python3", "-c", "print(1)"]


def test_noop_passthrough():
    assert NoopNetworkBackend().wrap(ARGV, "deny") == ARGV


def test_unshare_wrap_only_when_denied():
    assert UnshareBackend().wrap(ARGV, "deny") == ["unshare", "--net", "--", *ARGV]
    assert UnshareBackend().wrap(ARGV, "allow") == ARGV  # passthrough when egress permitted


def test_bwrap_wrap_when_denied():
    wrapped = BubblewrapBackend().wrap(ARGV, "none")
    assert wrapped[:2] == ["bwrap", "--unshare-net"]
    assert wrapped[-len(ARGV):] == ARGV


def test_default_backend_is_none_off_linux():
    if sys.platform != "linux":
        assert default_network_backend() is None
    else:
        be = default_network_backend()
        assert be is None or hasattr(be, "wrap")


def test_default_backend_requires_working_namespace(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("codna.netjail.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("codna.netjail._command_succeeds", lambda argv: False)

    assert default_network_backend() is None


def test_sandbox_routes_commands_through_backend(tmp_path):
    # A spy backend proves the sandbox actually applies wrap() to the argv it runs.
    class SpyBackend:
        def wrap(self, argv, network):
            return ["python3", "-c", "print('WRAPPED')"]

    sb = Sandbox(env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, network_backend=SpyBackend())
    res = sb.run(["echo", "unwrapped"], cwd=str(tmp_path))
    assert res.returncode == 0
    assert res.stdout.strip() == "WRAPPED"


@pytest.mark.skipif(not can_enforce_netns(), reason="needs Linux + network-namespace capability")
def test_real_egress_is_denied(tmp_path):
    # In a fresh net namespace there are no interfaces (lo is down), so any connect fails.
    backend = default_network_backend()
    assert backend is not None
    sb = Sandbox(
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        network_backend=backend,
        timeout_seconds=20,
    )
    code = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=2); sys.exit(0)\n"
        "except OSError:\n"
        "    sys.exit(7)\n"
    )
    res = sb.run(["python3", "-c", code], cwd=str(tmp_path))
    assert res.returncode != 0  # egress refused inside the namespace
