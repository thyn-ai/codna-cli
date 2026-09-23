from __future__ import annotations

import json

import pytest

from codna.local_mojo_daemon import (
    LocalMojoDaemonError,
    local_mojo_daemon_state_path,
    stop_local_mojo_daemon,
)
from codna.runtime.config import resolve_runtime_config
import codna.local_mojo_daemon as local_mojo_daemon_module


def test_stop_local_mojo_daemon_reports_permission_denied(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state_path = local_mojo_daemon_state_path(config)
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"pid": 123}), encoding="utf-8")

    monkeypatch.setattr(local_mojo_daemon_module, "_daemon_state_matches", lambda _config, _state: True)
    monkeypatch.setattr(local_mojo_daemon_module, "_request_daemon", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(local_mojo_daemon_module, "pid_is_alive", lambda _pid: True)

    def deny_stop(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(local_mojo_daemon_module, "terminate_pid", deny_stop)

    with pytest.raises(LocalMojoDaemonError) as excinfo:
        stop_local_mojo_daemon(config)

    assert excinfo.value.code == "local_mojo_daemon_stop_permission_denied"
    assert excinfo.value.details["pid"] == 123
    assert excinfo.value.details["manual_stop_command"] == "kill 123"


# ---- idle auto-exit (the daemon used to live forever; 70 leaked daemon+worker pairs on one laptop) ----

class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _TimingOutServer:
    """accept() never yields a client; each call advances the clock, then raises TimeoutError."""

    def __init__(self, clock: _Clock, step: float, stop_after: int | None = None, stop_event=None) -> None:
        self.clock, self.step, self.stop_after, self.stop_event, self.calls = clock, step, stop_after, stop_event, 0

    def accept(self):
        self.calls += 1
        self.clock.t += self.step
        if self.stop_after is not None and self.calls >= self.stop_after and self.stop_event is not None:
            self.stop_event.set()
        raise TimeoutError


def test_idle_exit_seconds_defaults_and_parsing(monkeypatch) -> None:
    monkeypatch.delenv("CODNA_LOCAL_MOJO_DAEMON_IDLE_S", raising=False)
    assert local_mojo_daemon_module.local_mojo_daemon_idle_exit_seconds() == 1800.0
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_IDLE_S", "90")
    assert local_mojo_daemon_module.local_mojo_daemon_idle_exit_seconds() == 90.0
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_IDLE_S", "0")
    assert local_mojo_daemon_module.local_mojo_daemon_idle_exit_seconds() == 0.0
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_IDLE_S", "not-a-number")
    assert local_mojo_daemon_module.local_mojo_daemon_idle_exit_seconds() == 1800.0
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_IDLE_S", "-5")
    assert local_mojo_daemon_module.local_mojo_daemon_idle_exit_seconds() == 0.0


def test_accept_loop_exits_once_idle_for_the_configured_time(capsys) -> None:
    """REGRESSION: the loop only ever ended on an explicit shutdown request, so a daemon whose
    client (a test run) simply exited kept itself and its simulate worker alive forever."""
    import threading

    clock = _Clock()
    stop = threading.Event()
    server = _TimingOutServer(clock, step=4.0)
    activity = local_mojo_daemon_module._Activity(clock)
    why = local_mojo_daemon_module._accept_loop(server, object(), stop, idle_exit_s=10.0, activity=activity)
    assert why == "idle"
    assert stop.is_set()
    assert server.calls == 3                      # 4 s, 8 s, 12 s -> idle >= 10 s on the third timeout
    assert "local_mojo_daemon_idle_exit" in capsys.readouterr().err


def test_accept_loop_with_idle_exit_disabled_only_stops_on_request() -> None:
    import threading

    clock = _Clock()
    stop = threading.Event()
    server = _TimingOutServer(clock, step=1000.0, stop_after=3, stop_event=stop)
    activity = local_mojo_daemon_module._Activity(clock)
    why = local_mojo_daemon_module._accept_loop(server, object(), stop, idle_exit_s=0.0, activity=activity)
    assert why == "stopped" and server.calls == 3


def test_accept_loop_never_idles_out_while_a_request_is_running() -> None:
    """A long invoke must not be cut off by the idle timer: in-flight requests pin the daemon."""
    import threading

    clock = _Clock()
    stop = threading.Event()
    server = _TimingOutServer(clock, step=1000.0, stop_after=3, stop_event=stop)
    activity = local_mojo_daemon_module._Activity(clock)
    activity.begin()                              # a request is in flight the whole time
    why = local_mojo_daemon_module._accept_loop(server, object(), stop, idle_exit_s=10.0, activity=activity)
    assert why == "stopped" and server.calls == 3  # 3000 s "idle" on the clock, yet no idle exit


def test_serve_client_brackets_the_request_in_activity(monkeypatch) -> None:
    import threading
    from unittest.mock import MagicMock

    clock = _Clock()
    activity = local_mojo_daemon_module._Activity(clock)
    clock.t = 50.0
    seen = []
    monkeypatch.setattr(local_mojo_daemon_module, "_read_frame", lambda _h: {"type": "ping"})
    monkeypatch.setattr(local_mojo_daemon_module, "_handle_request", lambda _c, _req, _stop: {"ok": True})
    monkeypatch.setattr(local_mojo_daemon_module, "_write_frame", lambda _h, resp: seen.append(resp))
    handle = MagicMock()
    handle.__enter__ = lambda self: self
    handle.__exit__ = lambda self, *a: None
    local_mojo_daemon_module._serve_client(handle, object(), threading.Event(), activity)
    assert seen and seen[0]["ok"] is True
    assert activity._in_flight == 0
    assert activity.idle_for() == 0.0             # the request just ended at t=50 -> idle resets
    clock.t = 61.0
    assert activity.idle_for() == 11.0


class _BrokenServer:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def accept(self):
        raise self.exc


def test_accept_loop_ends_cleanly_when_the_listening_socket_breaks(capsys) -> None:
    """EMFILE / EBADF on accept() used to escape the loop as an exception with the stop event never
    set; now the loop ends with reason "error" and the caller's finally still shuts the pool."""
    import errno as errno_module
    import threading

    stop = threading.Event()
    activity = local_mojo_daemon_module._Activity(_Clock())
    why = local_mojo_daemon_module._accept_loop(
        _BrokenServer(OSError(errno_module.EMFILE, "Too many open files")), object(), stop, idle_exit_s=10.0, activity=activity,
    )
    assert why == "error" and stop.is_set()
    assert "local_mojo_daemon_accept_error" in capsys.readouterr().err


def test_serve_client_ends_activity_even_if_begin_fails(monkeypatch) -> None:
    import threading
    from unittest.mock import MagicMock

    activity = local_mojo_daemon_module._Activity(_Clock())
    calls = {"begin": 0}

    def failing_begin():
        calls["begin"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(activity, "begin", failing_begin)
    handle = MagicMock()
    handle.__enter__ = lambda self: self
    handle.__exit__ = lambda self, *a: None
    clock = activity._clock
    clock.t = 5.0                                  # time passes before the doomed request
    with pytest.raises(RuntimeError):
        local_mojo_daemon_module._serve_client(handle, object(), threading.Event(), activity)
    assert calls["begin"] == 1
    assert activity._in_flight == 0                # nothing left in flight: the idle timer is not wedged
    assert activity.idle_for() == 5.0              # and end() did NOT run: the idle clock was not reset


class _FlakyThenTimingOutServer:
    """First accept() raises a per-connection error (client vanished); later ones time out."""

    def __init__(self, clock: _Clock, first_errno: int) -> None:
        self.clock, self.first_errno, self.calls = clock, first_errno, 0

    def accept(self):
        self.calls += 1
        if self.calls == 1:
            raise OSError(self.first_errno, "client went away")
        self.clock.t += 100.0
        raise TimeoutError


@pytest.mark.parametrize("code", ["ECONNABORTED", "EINTR", "EAGAIN"])
def test_accept_loop_retries_per_connection_errors_instead_of_exiting(code) -> None:
    """ECONNABORTED etc. mean one client vanished between connect() and accept(); the listening
    socket is healthy, so the loop must keep serving (here: until the idle timer fires)."""
    import errno as errno_module
    import threading

    clock = _Clock()
    stop = threading.Event()
    server = _FlakyThenTimingOutServer(clock, getattr(errno_module, code))
    activity = local_mojo_daemon_module._Activity(clock)
    why = local_mojo_daemon_module._accept_loop(server, object(), stop, idle_exit_s=10.0, activity=activity)
    assert why == "idle"                           # survived the per-connection error, idled out later
    assert server.calls == 2
