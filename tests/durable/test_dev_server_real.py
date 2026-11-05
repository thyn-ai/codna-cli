"""The REAL Temporal dev server: start it, start it again, and prove only one is running.

Everything else in `test_dev_server.py` is a fake, and deliberately so — the state machine is where
the bugs are and a fake makes it testable. This file is the complement: the half a fake cannot prove.

Three claims, each of which a fake can only mimic:

* **The argv we build actually starts a server.** ``--dynamic-config-value
  matching.enableDeploymentVersions=true``, ``--db-filename``, ``--port`` and ``--namespace`` are
  flags of a binary we do not own; a typo is a startup failure and nothing else in the suite would
  notice.
* **Double-start is SAFE, not merely idempotent.** Two `ensure_started` calls must leave exactly ONE
  listener, one pid file, and two statuses that agree — and the second must report ``adopted=True``
  rather than claiming it started a server it did not.
* **The pid file describes the process that is actually listening.** Stop must then kill THAT pid and
  leave the port free, so the next start is not racing a half-written SQLite database.

Gated on ``CODNA_TEST_TEMPORAL=1`` because it needs the Temporal CLI (downloaded on first use) and
network access for that download. The gate is an explicit env var rather than "try it and skip on
failure": a test that silently skips when the network is down is a test that quietly stops covering
the thing it exists for.

Run it with::

    CODNA_TEST_TEMPORAL=1 python -m pytest tests/durable/test_dev_server_real.py

Skip it (the base suite does) and nothing is lost but this file's coverage.
"""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

from codna.durable.dev_server import (
    DEPLOYMENT_VERSIONS_CONFIG,
    DevServer,
    DevServerSettings,
)

GATED = os.environ.get("CODNA_TEST_TEMPORAL") == "1"

pytestmark = [
    pytest.mark.skipif(not GATED, reason="set CODNA_TEST_TEMPORAL=1 to run against a real dev server"),
]


def _free_port() -> int:
    """An ephemeral port we immediately release.

    Racy in principle (another process may take it in the gap), and the standard way to do this in a
    test: the alternative is a hard-coded port that collides with whatever the developer is running,
    which is worse — a fixed port turns "the suite is green" into "the suite is green unless someone
    has Temporal open".
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def settings(tmp_path: Path) -> DevServerSettings:
    return DevServerSettings(
        host="127.0.0.1",
        port=_free_port(),
        namespace="codna",
        db_path=tmp_path / "temporal" / "dev.db",
        bin_dir=tmp_path / "temporal" / "bin",
        log_path=tmp_path / "logs" / "temporal-dev.log",
        pid_path=tmp_path / "temporal" / "dev-server.json",
    )


@pytest.fixture
def server(settings: DevServerSettings):
    """A real `DevServer` that is always stopped, even when an assertion fails mid-test.

    An unstopped dev server holds a port and a SQLite file for the rest of the session; the teardown
    is what keeps a failing test from turning into a failing suite.
    """
    instance = DevServer(settings)
    yield instance
    try:
        instance.stop(timeout_s=20.0)
    except Exception:  # noqa: BLE001 - teardown must not mask the real failure
        pass


def _listening_pids(port: int) -> list[int]:
    completed = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line[1:]) for line in completed.stdout.splitlines() if line.startswith("p")]


def test_the_argv_start_a_server_that_really_answers(server: DevServer, settings: DevServerSettings):
    """What the fake asserts about the flags, the real binary proves about the server."""
    status = server.ensure_started(timeout_s=90.0, headless=True)
    assert status.running is True
    assert status.managed is True
    assert status.pid is not None
    # The pid we wrote is the pid that is listening — the whole premise of `stop`.
    assert status.pid in _listening_pids(settings.port)
    assert Path(settings.pid_path).is_file()
    assert Path(settings.db_path).is_file(), "the SQLite database was not created"


def test_starting_twice_leaves_exactly_one_server(server: DevServer, settings: DevServerSettings):
    """The property the lock exists for: two starts, one server, and the second says so."""
    first = server.ensure_started(timeout_s=90.0, headless=True)
    assert first.running and first.adopted is False

    pids_after_first = _listening_pids(settings.port)
    assert len(pids_after_first) == 1

    second = server.ensure_started(timeout_s=90.0, headless=True)
    assert second.running is True
    assert second.adopted is True, "a second start must report that it adopted the running server"
    # Still exactly one listener, and it is the SAME process: no second `start-dev` was spawned.
    assert _listening_pids(settings.port) == pids_after_first
    assert second.pid == first.pid


def test_stop_ends_the_process_and_frees_the_port(server: DevServer, settings: DevServerSettings):
    started = server.ensure_started(timeout_s=90.0, headless=True)
    assert started.pid is not None

    stopped = server.stop(timeout_s=30.0)
    assert stopped.running is False
    assert not Path(settings.pid_path).exists()
    assert _listening_pids(settings.port) == [], "the port is still bound after stop"
    # Starting again must work: a stop that left the SQLite file locked would fail here.
    again = server.ensure_started(timeout_s=90.0, headless=True)
    assert again.running is True


def test_deployment_versioning_is_on_for_the_server_we_start(
    server: DevServer, settings: DevServerSettings
):
    """Versioning is a SERVER setting; the worker's PINNED registration depends on it being enabled.

    Asserted against the running server rather than only the argv (which `test_dev_server.py` already
    checks) because "we passed the flag" and "the server honours it" are different claims, and only
    the second one makes the worker's versioning work.
    """
    import asyncio

    from temporalio.api.workflowservice.v1 import DescribeWorkerDeploymentRequest
    from temporalio.client import Client

    assert DEPLOYMENT_VERSIONS_CONFIG in server.argv()
    server.ensure_started(timeout_s=90.0, headless=True)

    async def connect_with_retries() -> Client:
        for attempt in range(100):
            try:
                return await Client.connect(f"127.0.0.1:{settings.port}", namespace="codna")
            except RuntimeError as exc:
                if "Frontend is not healthy yet" not in str(exc) or attempt == 99:
                    raise
                await asyncio.sleep(0.1)
        raise AssertionError("unreachable")

    async def probe() -> str:
        connected = await connect_with_retries()
        try:
            # A deployment that does not exist yet raises a NAMED error; a server with deployment
            # support disabled raises "not supported". Those are different answers, and the second
            # is the one that would break a PINNED worker.
            try:
                await connected.workflow_service.describe_worker_deployment(
                    DescribeWorkerDeploymentRequest(namespace="codna", deployment_name="codna")
                )
            except Exception as exc:  # noqa: BLE001 - the message is the assertion
                return f"{type(exc).__name__}: {exc}"
            return "found"
        finally:
            # temporalio 1.34's Client has no `close`/`shutdown`: disconnecting is dropping the
            # reference (the bridge connection is not refcounted), so there is nothing to release.
            pass

    answer = asyncio.run(probe())
    assert "not supported" not in answer.lower(), answer
    assert "no Worker Deployment found" in answer, answer
