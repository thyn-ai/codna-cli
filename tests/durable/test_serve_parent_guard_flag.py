"""`codna serve --guard-parent-pid`: the run API stops with the app that spawned it.

The desktop app now starts and supervises the loopback Run API, so the supervision contract the worker
already carries has to hold for it too. Three properties are asserted here, all at `cmd_serve`'s boundary
(no Temporal, no socket, no real parent):

  1. A VALUE MUST NAME THE ACTUAL PARENT. The guard is opt-in through a pid, and a pid that names some
     other process is a REFUSAL rather than a guard that protects the wrong thing — the worker's own
     refusal codes, reused, so a client branching on them does not have to know which command ran.
  2. A HAND-RUN SERVER KEEPS ITS OLD BEHAVIOUR. No flag means no guard: a human who typed `codna serve`
     has no supervisor, and a terminal closing must not abandon an API another window is driving.
  3. THE GUARD IS DISARMED WHEN SERVING ENDS. A clean stop is not a lost parent, and a timer left armed in
     a process that is shutting down is a thread that could signal the shutdown itself.

The fourth — that a lost parent moves the server onto its ORDINARY SIGTERM path — is asserted directly:
the callback sends SIGTERM to this very process id rather than calling `os._exit`, so the server's own
stream drain and stop reporting run.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

import codna.durable.cli  # noqa: F401 — importing the durable package before `run_cli` resolves its re-export cycle
from codna.durable import run_cli, serve_cli
from codna.durable.errors import DurableCliError
import codna.durable.serve_guard as serve_guard_module
from codna.durable.serve_guard import arm_parent_guard, guard_parent_pid


class _RecordingServer:
    """A stand-in for `RunApiServer` that records the calls `_serve` makes on it."""

    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8790"
        self.port = 8790
        self.stop_reason = None
        self.api = SimpleNamespace(capabilities_payload=lambda: {"mode": "durable"})
        self.bound = False
        self.served = False
        self.stopped = False

    def bind(self) -> Any:
        self.bound = True
        return self

    def serve_forever(self) -> None:
        self.served = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def recording_serve(monkeypatch: pytest.MonkeyPatch):
    """`_serve` with every external effect replaced, so `cmd_serve` runs end to end in-process."""
    recorded: dict[str, Any] = {"guards": [], "printed": []}
    server = _RecordingServer()

    monkeypatch.setattr(run_cli._cli, "_config", lambda args: SimpleNamespace(
        build_id="codna-0.0.0-test-g1",
        address="127.0.0.1:7233",
        namespace="codna",
        task_queue="codna.agent.test",
        graph_engine="plain",
        runs_root="/tmp/runs",
        sidecar_url="http://127.0.0.1:18601",
        profile="local",
        is_cloud=False,
        api_key_env=None,
        pool=None,
        connect_options=lambda: {},
    ))
    monkeypatch.setattr(run_cli._cli, "_print", lambda payload: recorded["printed"].append(payload))
    monkeypatch.setattr(serve_cli, "_LedgerOnlyBackend", lambda: SimpleNamespace(
        capabilities=lambda: SimpleNamespace(temporal_address=None)
    ))
    import codna.durable.run_api as run_api_module
    import codna.durable.run_api_support as run_api_support

    monkeypatch.setattr(run_api_module, "RunApiServer", lambda **kwargs: server)

    class _Token:
        token_path = "/tmp/run-api.token"
        host = "127.0.0.1"
        max_streams = 8

        @classmethod
        def for_config(cls, *_args: Any, **_kwargs: Any) -> Any:
            return cls()

    monkeypatch.setattr(run_api_support, "RunApiSettings", _Token)
    monkeypatch.setattr(run_api_support, "load_or_create_token", lambda _path: "0" * 64)

    class _Identity:
        mode = "token"

    import codna.durable.identity as identity_module

    monkeypatch.setattr(identity_module, "IdentityResolver", lambda _config: _Identity())

    def fake_arm(parent_pid: int | None):
        import contextlib

        @contextlib.contextmanager
        def armed():
            recorded["guards"].append(("armed", parent_pid))
            try:
                yield
            finally:
                recorded["guards"].append(("disarmed", parent_pid))

        return armed()

    monkeypatch.setattr(serve_guard_module, "arm_parent_guard", fake_arm)
    monkeypatch.setattr(serve_guard_module, "guard_parent_pid", guard_parent_pid)
    return recorded, server


def _args(**overrides: Any) -> SimpleNamespace:
    base = dict(
        no_temporal=True,
        start_dev_server=False,
        start_timeout=60.0,
        host="127.0.0.1",
        port=8790,
        heartbeat_interval=15.0,
        max_streams=8,
        auth_mode="",
        oidc_issuer="",
        oidc_audience="",
        oidc_role_claim="roles",
        oidc_jwks_uri="",
        local_role=None,
        local_identity="",
        guard_parent_pid=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_guard_pid_that_is_not_the_actual_parent_is_refused_before_anything_starts(
    recording_serve: tuple[dict[str, Any], _RecordingServer],
) -> None:
    recorded, server = recording_serve
    # Above any real process table entry, so the mismatch is deterministic and never signals anyone.
    wrong_pid = 1 << 22
    with pytest.raises(DurableCliError) as excinfo:
        serve_cli.cmd_serve(_args(guard_parent_pid=wrong_pid))
    # The WORKER's codes, because it is the same contract: a client branching on why a guarded codna
    # process refused must not need to know which command ran.
    assert excinfo.value.code == "worker_parent_pid_mismatch"
    assert recorded["guards"] == [], "the guard was armed for a pid that was never validated"
    assert server.served is False


def test_a_non_positive_guard_pid_is_refused(
    recording_serve: tuple[dict[str, Any], _RecordingServer],
) -> None:
    for bad in (0, -1, True):
        with pytest.raises(DurableCliError) as excinfo:
            serve_cli.cmd_serve(_args(guard_parent_pid=bad))
        assert excinfo.value.code == "worker_parent_pid_invalid"


def test_the_actual_parent_pid_arms_the_guard_for_exactly_the_serving_window(
    recording_serve: tuple[dict[str, Any], _RecordingServer],
) -> None:
    recorded, server = recording_serve
    parent = os.getppid()
    assert serve_cli.cmd_serve(_args(guard_parent_pid=parent)) == 0
    # Armed with the VALIDATED pid before serving, and disarmed after it: the guard covers startup and the
    # listen window, and never outlives the shutdown it is meant to be distinct from.
    assert recorded["guards"] == [("armed", parent), ("disarmed", parent)]
    assert server.bound is True
    assert server.served is True
    assert server.stopped is True


def test_no_flag_arms_nothing_so_a_hand_run_server_keeps_its_behaviour(
    recording_serve: tuple[dict[str, Any], _RecordingServer],
) -> None:
    recorded, server = recording_serve
    assert serve_cli.cmd_serve(_args()) == 0
    assert recorded["guards"] == [("armed", None), ("disarmed", None)]
    assert server.served is True


def test_the_flag_is_registered_on_both_spellings_of_serve() -> None:
    import codna.cli as top_cli

    parsed = top_cli.build_parser().parse_args(["serve", "--guard-parent-pid", "4242"])
    assert parsed.guard_parent_pid == 4242
    grouped = top_cli.build_parser().parse_args(["durable", "serve", "--guard-parent-pid", "4242"])
    assert grouped.guard_parent_pid == 4242
    # Absent means absent: a hand-run server is unguarded, not guarded against pid 0.
    assert top_cli.build_parser().parse_args(["serve"]).guard_parent_pid is None


def test_a_lost_parent_sends_SIGTERM_to_this_process_rather_than_exiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run API drains its streams and reports its own stop; `os._exit` would skip both."""
    import signal as signal_module

    import codna.durable.serve_guard as guard_module

    killed: list[tuple[int, int]] = []
    logged: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    import codna.durable.worker as worker_module

    monkeypatch.setattr(worker_module, "_log_json", lambda event, **fields: logged.append((event, fields)))

    guard_module._signal_self_for_parent_loss("parent-reparented")
    assert killed == [(os.getpid(), signal_module.SIGTERM)]
    assert logged[0][0] == "run_api.parent_lost"
    assert logged[0][1]["signal"] == "SIGTERM"


def test_the_guard_context_manager_installs_and_stops_one_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import codna.durable.parent_guard as parent_guard_module

    calls: dict[str, Any] = {}

    class _Guard:
        def stop(self) -> None:
            calls["stopped"] = True

    def fake_install(on_lost: Any, *, poll_ms: int = 0, parent_pid: int | None = None) -> Any:
        calls["parent_pid"] = parent_pid
        calls["on_lost"] = on_lost
        return _Guard()

    monkeypatch.setattr(parent_guard_module, "install_parent_guard", fake_install)
    with arm_parent_guard(777):
        assert calls["parent_pid"] == 777
        assert "stopped" not in calls, "the guard was disarmed before the window it protects closed"
    assert calls["stopped"] is True
    # The disarmed guard cannot fire: `None` installs nothing at all.
    calls.clear()
    with arm_parent_guard(None):
        pass
    assert calls == {}
