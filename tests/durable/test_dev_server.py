"""The local Temporal dev-server lifecycle: discovery, start, readiness, stop.

Nothing here needs ``temporalio`` — `dev_server.py` deliberately does not import it — so these run in
the BASE suite, which is the point: the failure a user hits first ("there is no server") must be
tested without installing the extra it is complaining about.

The one rule under test throughout: a state we did not create is never adopted. A port that is
listening without our pid file is a FOREIGN server (reported, never killed), a live pid that is not
listening is UNRESPONSIVE (reported, never double-started), and a dead pid file is stale (cleared,
because it is proof a previous start failed rather than evidence of someone else's process).

No test binds a privileged port, spawns the real CLI, or sends a signal to a pid it did not create:
the process substituted for "the server" is a ``sleep`` this file starts and reaps itself.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from codna.durable.dev_server import (
    POLL_INTERVAL_S,
    DevServer,
    DevServerError,
    DevServerSettings,
)

# A pid above Linux's default pid_max and macOS's, so `kill(pid, 0)` is a ProcessLookupError rather
# than a signal delivered to some unrelated process. Never signal a pid you did not create.
DEAD_PID = 1 << 22


class FakeClock:
    """A monotonic clock that only advances when the code under test sleeps.

    A readiness loop that waits on ``time.monotonic`` is untestable without this: the alternative is
    a test that really sleeps for the start timeout.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeProcess:
    """The ``Popen`` surface `DevServer` uses: a pid and a poll() result."""

    def __init__(self, pid: int = 4242, exit_code: int | None = None) -> None:
        self.pid = pid
        self.exit_code = exit_code

    def poll(self) -> int | None:
        return self.exit_code


@pytest.fixture
def settings(tmp_path: Path) -> DevServerSettings:
    temporal = tmp_path / "temporal"
    return DevServerSettings(
        host="127.0.0.1",
        port=7233,
        namespace="codna",
        db_path=temporal / "dev.db",
        bin_dir=temporal / "bin",
        log_path=tmp_path / "logs" / "temporal-dev.log",
        pid_path=temporal / "dev-server.json",
    )


@pytest.fixture
def cli_binary(settings: DevServerSettings) -> Path:
    """An executable at the MANAGED location, standing in for the downloaded Temporal CLI.

    It is never executed by these tests: only `argv()` reads the path, and the ready/stop loops use
    a fake process or a real `sleep` this file owns.
    """
    path = settings.bin_dir / "temporal"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _server(settings: DevServerSettings, **overrides) -> DevServer:
    """A server whose spawn/probe/clock are all explicit; each test overrides what it cares about.

    ``allow_download=False`` by default: a unit test must never reach the network, and the pinned
    binary is supplied through the lock file each test writes.
    """
    kwargs = {
        "which": lambda _name: None,
        "spawn": lambda argv, log_path: FakeProcess(),
        "probe": lambda host, port: False,
        "clock": FakeClock(),
        "sleep": lambda seconds: None,
        "allow_download": False,
        "trust_binary": False,
    }
    kwargs.update(overrides)
    clock = kwargs.pop("clock")
    if "sleep" not in overrides:
        kwargs["sleep"] = clock.sleep
    return DevServer(settings, **kwargs)


# -- discovery: the PINNED binary ----------------------------------------------------------


def _pinned_lock(tmp_path: Path, **overrides: Any) -> Path:
    """A lock file for this platform whose digests are supplied by the test.

    Written per-test rather than using the shipped lock so every refusal (mismatch, unsupported
    platform, corrupt pin) can be exercised without touching the real pin or downloading anything.
    """
    from codna.durable.dev_server import DevServer

    key = DevServer.platform_key()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "cli_version": "1.9.1",
        "archive_url": "https://example.invalid/cli/archive/v1.9.1",
        "platforms": {key: {"platform": key.split("-")[0], "arch": key.split("-")[1],
                            "binary_sha256": "0" * 64, "archive_sha256": ""}},
    }
    payload["platforms"][key].update(overrides.pop("entry", {}))
    payload.update(overrides)
    path = tmp_path / "temporal_cli.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pinned_server(settings: DevServerSettings, tmp_path: Path, binary: Path, **kwargs: Any) -> Any:
    """A server whose lock pins ``binary``'s real digest, so verification passes for real."""
    lock = _pinned_lock(tmp_path, entry={"binary_sha256": _sha256(binary)})
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": lock})
    return _server(pinned, binary=None, allow_download=False, **kwargs)


def test_the_managed_copy_is_used_when_it_matches_the_pin(settings: DevServerSettings, cli_binary: Path, tmp_path: Path):
    """The managed copy is the pinned one; PATH is irrelevant, which is the point of a pin."""
    server = _pinned_server(settings, tmp_path, cli_binary)
    assert server.binary() == str(cli_binary)


def test_path_is_never_consulted(settings: DevServerSettings, tmp_path: Path):
    """A developer's own `temporal` is a different build; falling back to it defeats the pin."""
    lock = _pinned_lock(tmp_path)  # pins a digest nothing on disk has
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": lock})
    server = _server(pinned, which=lambda name: "/usr/local/bin/temporal", allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    # NOT a PATH fallback: the refusal names the managed path and the digest it expected.
    assert caught.value.code == "dev_server_binary_missing"
    assert caught.value.details["binary_sha256"] == "0" * 64


def test_a_downloaded_binary_that_does_not_match_the_pin_is_refused(
    settings: DevServerSettings, cli_binary: Path, tmp_path: Path
):
    """The pin is worthless if a mismatched file is used anyway — this is the supply-chain check."""
    lock = _pinned_lock(tmp_path, entry={"binary_sha256": "1" * 64})
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": lock})
    server = _server(pinned, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_binary_digest_mismatch"
    assert caught.value.details["actual_sha256"] == _sha256(cli_binary)
    assert caught.value.details["expected_sha256"] == "1" * 64
    assert "reinstall" in caught.value.hint.lower() or "download" in caught.value.hint.lower()


def test_an_explicit_binary_is_still_verified_against_the_pin(
    settings: DevServerSettings, tmp_path: Path
):
    """An override must not become a way to run an unverified executable."""
    explicit = tmp_path / "elsewhere" / "temporal"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("#!/bin/sh\n", encoding="utf-8")
    explicit.chmod(0o755)
    lock = _pinned_lock(tmp_path)  # digest of zeros: this file cannot match
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": lock})
    server = _server(pinned, binary=explicit, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_binary_digest_mismatch"


def test_an_explicit_binary_can_be_trusted_by_a_test_that_says_so(
    settings: DevServerSettings, tmp_path: Path, cli_binary: Path
):
    server = _server(settings, binary=cli_binary, trust_binary=True)
    assert server.binary() == str(cli_binary)


def test_a_missing_binary_fails_loudly_with_the_expected_digest(
    settings: DevServerSettings, tmp_path: Path
):
    """The whole point of the module: absence is an error naming the pin, never a fallback."""
    lock = _pinned_lock(tmp_path)
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": lock})
    server = _server(pinned, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_binary_missing"
    assert caught.value.hint
    assert caught.value.details["cli_version"] == "1.9.1"
    assert caught.value.details["path"] == str(pinned.bin_dir / "temporal")


def test_a_missing_pin_file_is_an_error_not_an_invitation_to_download(
    settings: DevServerSettings, tmp_path: Path
):
    """An unreadable pin must not degrade into "install whatever is at the URL"."""
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": tmp_path / "nope.json"})
    server = _server(pinned, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_pin_missing"


def test_a_corrupt_pin_is_an_error(settings: DevServerSettings, tmp_path: Path):
    path = tmp_path / "temporal_cli.lock.json"
    path.write_text("{not json", encoding="utf-8")
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": path})
    server = _server(pinned, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_pin_corrupt"


def test_a_platform_the_pin_does_not_cover_is_refused(settings: DevServerSettings, tmp_path: Path):
    """A platform we have not verified is a binary we cannot vouch for — never a fallback digest."""
    path = tmp_path / "temporal_cli.lock.json"
    path.write_text(json.dumps({"cli_version": "1.9.1", "archive_url": "u", "platforms": {}}),
                    encoding="utf-8")
    pinned = DevServerSettings(**{**settings.__dict__, "lock_path": path})
    server = _server(pinned, allow_download=False)
    with pytest.raises(DevServerError) as caught:
        server.binary()
    assert caught.value.code == "dev_server_platform_unsupported"
    # The detail is this RUNNING host's tag, so it must be derived, not literalled. A hardcoded
    # "darwin-arm64" passed on the development machine and failed the whole CLI job on Linux CI —
    # the same defect class as pinning a version number in the build-id test: a test may pin the
    # FORMAT of what it asserts, never the machine it happens to run on.
    assert caught.value.details["platform"] == DevServer.platform_key()


def test_the_shipped_pin_covers_this_platform_with_a_full_digest():
    """The lock file we ship must describe the machine we are running on, or every start fails."""
    from codna.durable.dev_server import DevServer, default_lock_path

    payload = json.loads(default_lock_path().read_text(encoding="utf-8"))
    server = DevServer(DevServerSettings(
        host="127.0.0.1", port=7233, namespace="codna", db_path=Path("/tmp/unused.db"),
        bin_dir=Path("/tmp/unused-bin"), log_path=Path("/tmp/unused.log"),
        pid_path=Path("/tmp/unused.json"), lock_path=default_lock_path(),
    ))
    entry = server.platform_entry(payload)
    assert len(entry["binary_sha256"]) == 64
    # Every platform the wheels ship for is pinned, not just the one this machine happens to be.
    assert {"darwin-arm64", "linux-amd64"} <= set(payload["platforms"])


# -- argv ----------------------------------------------------------------------------------


def test_argv_names_the_database_port_and_namespace(
    settings: DevServerSettings, cli_binary: Path, tmp_path: Path
):
    """The database is durable state: it must be ours, not the CLI's default per-run temp file."""
    server = _pinned_server(settings, tmp_path, cli_binary)
    argv = server.argv()
    assert argv[0] == str(cli_binary)
    assert argv[1:3] == ["server", "start-dev"]
    assert argv[argv.index("--db-filename") + 1] == str(settings.db_path)
    assert argv[argv.index("--port") + 1] == str(settings.port)
    assert argv[argv.index("--namespace") + 1] == settings.namespace


def test_headless_only_adds_the_flag(settings: DevServerSettings, cli_binary: Path, tmp_path: Path):
    server = _pinned_server(settings, tmp_path, cli_binary)
    assert "--headless" not in server.argv()
    assert "--headless" in server.argv(headless=True)


def test_argv_enables_deployment_versioning(
    settings: DevServerSettings, cli_binary: Path, tmp_path: Path
):
    """A dev server leaves deployment versioning OFF by default.

    With it off, a PINNED worker's runs never reach it (measured: they stay ``__unversioned__`` and
    the workflow times out), so the versioning the plan relies on would be dead code on every fresh
    install. Passing it here makes "a build bump cannot steal an in-flight run" true out of the box.
    """
    argv = _pinned_server(settings, tmp_path, cli_binary).argv()
    pairs = [argv[index + 1] for index, item in enumerate(argv) if item == "--dynamic-config-value"]
    assert "matching.enableDeploymentVersions=true" in pairs


def test_a_non_loopback_address_is_refused(settings: DevServerSettings):
    """`start-dev` binds loopback; a remote address means the user wants a real server."""
    remote = DevServerSettings(**{**settings.__dict__, "host": "temporal.internal"})
    server = _server(remote, which=lambda name: None)
    with pytest.raises(DevServerError) as caught:
        server.argv()
    assert caught.value.code == "dev_server_address_unsupported"
    assert caught.value.details["address"] == "temporal.internal:7233"


def test_an_out_of_range_port_is_refused_before_anything_runs(settings: DevServerSettings):
    with pytest.raises(DevServerError) as caught:
        _server(DevServerSettings(**{**settings.__dict__, "port": 70000}))
    assert caught.value.code == "dev_server_port_invalid"


# -- status --------------------------------------------------------------------------------


def test_status_reports_a_managed_server(settings: DevServerSettings, cli_binary: Path):
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    server = _server(settings, probe=lambda host, port: True)
    status = server.status()
    assert status.running and status.managed
    assert status.pid == os.getpid()
    assert not status.stale_pid_file


def test_status_reports_a_stale_pid_file(settings: DevServerSettings):
    """A pid file whose process is gone is proof a previous start died — not a foreign server."""
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": DEAD_PID}), encoding="utf-8")
    server = _server(settings, probe=lambda host, port: False)
    status = server.status()
    assert not status.running
    assert status.stale_pid_file
    assert status.pid is None


def test_status_reports_a_foreign_server_as_unmanaged(settings: DevServerSettings):
    server = _server(settings, probe=lambda host, port: True)
    status = server.status()
    assert status.running and not status.managed
    assert status.pid is None
    assert status.to_payload()["running"] is True


def test_a_corrupt_pid_file_is_an_error_not_an_absence(settings: DevServerSettings):
    """Treating unparseable state as "no server" would leave an orphan holding the database file."""
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text("{not json", encoding="utf-8")
    server = _server(settings)
    with pytest.raises(DevServerError) as caught:
        server.read_pid()
    assert caught.value.code == "dev_server_pid_corrupt"


def test_a_pid_file_without_a_usable_pid_is_an_error(settings: DevServerSettings):
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": "not-a-pid"}), encoding="utf-8")
    with pytest.raises(DevServerError) as caught:
        _server(settings).read_pid()
    assert caught.value.code == "dev_server_pid_corrupt"


# -- start ---------------------------------------------------------------------------------


def test_ensure_started_does_nothing_when_a_server_is_already_answering(settings: DevServerSettings):
    """Idempotent by design: `run`, `worker` and `status` all call it and must not race."""
    spawned: list[list[str]] = []
    server = _server(
        settings,
        probe=lambda host, port: True,
        spawn=lambda argv, log_path: spawned.append(argv) or FakeProcess(),
    )
    status = server.ensure_started()
    assert spawned == []
    assert status.running


def test_ensure_started_refuses_a_live_but_unresponsive_process(settings: DevServerSettings):
    """A second start would fight over one SQLite file; killing it might be the user's process."""
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    server = _server(settings, probe=lambda host, port: False)
    with pytest.raises(DevServerError) as caught:
        server.ensure_started()
    assert caught.value.code == "dev_server_unresponsive"
    assert caught.value.details["pid"] == os.getpid()


def test_ensure_started_clears_a_stale_pid_file_then_spawns(settings: DevServerSettings, cli_binary: Path, tmp_path: Path):
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": DEAD_PID}), encoding="utf-8")
    spawned: list[list[str]] = []
    clock = FakeClock()
    # first probe: nothing; in-lock re-probe: still nothing; readiness probe: accepting
    states = iter([False, False, True])
    server = _pinned_server(
        settings,
        tmp_path,
        cli_binary,
        probe=lambda host, port: next(states, True),
        spawn=lambda argv, log_path: spawned.append(argv) or FakeProcess(pid=os.getpid()),
        clock=clock,
        sleep=clock.sleep,
    )
    status = server.ensure_started(timeout_s=5.0)
    assert len(spawned) == 1
    assert status.managed and status.running
    assert json.loads(settings.pid_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_ensure_started_reports_a_server_that_died_with_its_log_tail(settings: DevServerSettings, cli_binary: Path, tmp_path: Path):
    """"timed out waiting" with no output is a support ticket; the exit code and log are the fix."""
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.write_text("level=ERROR msg=\"database is locked\"\n", encoding="utf-8")
    clock = FakeClock()
    server = _pinned_server(
        settings,
        tmp_path,
        cli_binary,
        spawn=lambda argv, log_path: FakeProcess(pid=DEAD_PID, exit_code=3),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(DevServerError) as caught:
        server.ensure_started(timeout_s=5.0)
    error = caught.value
    assert error.code == "dev_server_exited"
    assert error.details["exit_code"] == 3
    assert "database is locked" in " ".join(error.details["log_tail"])
    # The failed start must not leave a pid file behind for the next start to trip over.
    assert not settings.pid_path.exists()


def test_ensure_started_reports_a_timeout_with_the_log_tail(settings: DevServerSettings, cli_binary: Path, tmp_path: Path):
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_path.write_text("starting...\n", encoding="utf-8")
    clock = FakeClock()
    server = _pinned_server(
        settings,
        tmp_path,
        cli_binary,
        spawn=lambda argv, log_path: FakeProcess(pid=DEAD_PID, exit_code=None),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(DevServerError) as caught:
        server.ensure_started(timeout_s=0.5)
    assert caught.value.code == "dev_server_start_timeout"
    assert "starting..." in " ".join(caught.value.details["log_tail"])
    assert clock.now >= 0.5


def test_ensure_started_reports_that_it_adopted_a_running_server(settings: DevServerSettings):
    """The second of two concurrent commands must be able to tell it did not start the server."""
    server = _server(settings, probe=lambda host, port: True)
    status = server.ensure_started()
    assert status.adopted is True
    assert server.status().adopted is False


def test_ensure_started_re_probes_after_waiting_for_the_start_lock(
    settings: DevServerSettings, cli_binary: Path
):
    """The double-start guard: a server started while this caller waited must be adopted, not doubled.

    The first ``probe`` reports down (so this call takes the lock); by the time the lock is ours the
    "winner" has finished starting, which the in-lock re-probe observes. No second spawn.
    """
    spawned: list[list[str]] = []
    states = iter([False, True])  # outside the lock: down; inside: the winner is up
    server = _server(
        settings,
        probe=lambda host, port: next(states, True),
        spawn=lambda argv, log_path: spawned.append(argv) or FakeProcess(),
        which=lambda name: str(cli_binary),
    )
    status = server.ensure_started()
    assert spawned == [], "a server that came up while waiting must not be started twice"
    assert status.adopted is True
    assert status.running is True


def test_the_start_lock_is_released_after_a_successful_start(
    settings: DevServerSettings, cli_binary: Path, tmp_path: Path
):
    """A lock left held would make every later command wait out its full timeout."""
    states = iter([False, False, True])
    server = _pinned_server(
        settings,
        tmp_path,
        cli_binary,
        probe=lambda host, port: next(states, True),
        spawn=lambda argv, log_path: FakeProcess(pid=os.getpid()),
    )
    server.ensure_started(timeout_s=5.0)
    # Re-acquiring must be immediate: a second DevServer on the same settings is another process.
    other = _server(settings, probe=lambda host, port: True, which=lambda name: str(cli_binary))
    assert other.ensure_started().adopted is True


def test_ensure_started_rejects_a_non_positive_timeout(settings: DevServerSettings):
    with pytest.raises(DevServerError) as caught:
        _server(settings).ensure_started(timeout_s=0)
    assert caught.value.code == "dev_server_timeout_invalid"


# -- stop ----------------------------------------------------------------------------------


def test_stop_refuses_to_touch_a_foreign_server(settings: DevServerSettings):
    """Claiming to have stopped someone else's server would be a lie; killing it is worse."""
    server = _server(settings, probe=lambda host, port: True)
    with pytest.raises(DevServerError) as caught:
        server.stop()
    assert caught.value.code == "dev_server_foreign"
    assert caught.value.details["address"] == settings.address


def test_stop_is_a_no_op_when_nothing_is_running(settings: DevServerSettings):
    server = _server(settings, probe=lambda host, port: False)
    status = server.stop()
    assert not status.running


def test_stop_clears_a_stale_pid_file_without_signalling_anything(settings: DevServerSettings):
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": DEAD_PID}), encoding="utf-8")
    server = _server(settings, probe=lambda host, port: False)
    status = server.stop()
    assert not status.running
    assert not settings.pid_path.exists()


def test_stop_terminates_the_process_it_started(settings: DevServerSettings):
    """A REAL process this test owns, so the SIGTERM path is exercised rather than mocked."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
        settings.pid_path.write_text(json.dumps({"pid": process.pid}), encoding="utf-8")
        server = _server(settings, probe=lambda host, port: False)
        status = server.stop(timeout_s=5.0)
        assert not status.running
        assert process.wait(timeout=5) is not None
        assert not settings.pid_path.exists()
    finally:
        if process.poll() is None:  # pragma: no cover - only on a failed assertion above
            process.kill()
            process.wait(timeout=5)


def test_stop_reports_a_server_still_answering_after_the_pid_was_terminated(settings: DevServerSettings):
    """A port that stays open after the process is gone is a different server: say so."""
    settings.pid_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(json.dumps({"pid": DEAD_PID}), encoding="utf-8")
    server = _server(settings, probe=lambda host, port: True)
    with pytest.raises(DevServerError) as caught:
        server.stop()
    assert caught.value.code == "dev_server_stop_timeout"


# -- the real probe ------------------------------------------------------------------------


def test_the_real_probe_sees_a_listening_socket():
    """The readiness probe is the only thing standing between "started" and "connectable"."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        settings = DevServerSettings(
            host="127.0.0.1",
            port=port,
            namespace="codna",
            db_path=Path("/tmp/unused.db"),
            bin_dir=Path("/tmp/unused-bin"),
            log_path=Path("/tmp/unused.log"),
            pid_path=Path("/tmp/unused.json"),
        )
        server = DevServer(settings)
        assert server.probe() is True
    finally:
        listener.close()


def test_the_real_probe_reports_a_closed_port():
    """A port that was just released must read as DOWN, or `stop` would never return."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    settings = DevServerSettings(
        host="127.0.0.1",
        port=port,
        namespace="codna",
        db_path=Path("/tmp/unused.db"),
        bin_dir=Path("/tmp/unused-bin"),
        log_path=Path("/tmp/unused.log"),
        pid_path=Path("/tmp/unused.json"),
    )
    assert DevServer(settings).probe() is False


def test_the_readiness_poll_interval_is_short_enough_to_feel_immediate():
    """Pinned: the loop runs once per this interval, so a large value would stall every command."""
    assert 0 < POLL_INTERVAL_S <= 0.25


# -- the pin as a SHIPPED artifact ----------------------------------------------------------


def test_the_pin_would_ship_in_the_wheel():
    """`temporal_cli.lock.json` is data the installed package reads at runtime.

    A wheel built without it can only fail closed (`dev_server_pin_missing`), which would make
    `codna run` unusable for every user who installed from a package rather than a checkout. Asserted
    against pyproject rather than by building a wheel: this is the declaration that decides it.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert "temporal_cli.lock.json" in package_data["codna.durable"]
    assert "codna.durable" in data["tool"]["setuptools"]["packages"]


def test_the_pin_file_is_present_beside_the_module():
    """The path `default_lock_path()` computes must actually exist in a checkout."""
    from codna.durable.dev_server import default_lock_path

    assert default_lock_path().is_file(), default_lock_path()
    payload = json.loads(default_lock_path().read_text(encoding="utf-8"))
    assert payload["cli_version"]
    assert payload["archive_url"].startswith("https://")
