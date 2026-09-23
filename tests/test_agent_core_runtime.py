from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from codna.agent_core_runtime import (
    AgentCoreEndpoint,
    ensure_agent_core_running,
    inspect_agent_core_runtime,
    stop_agent_core_runtime,
)
from codna.runtime.config import resolve_runtime_config
from codna.runtime.local_stack import LocalRuntimeError
from codna.runtime.ports import ListenerInfo
import codna.agent_core_runtime as agent_core_runtime
import codna.local_mojo_daemon as local_mojo_daemon_module


@pytest.fixture(autouse=True)
def _clear_engine_overrides(monkeypatch) -> None:
    for key in (
        "CODNA_ENGINE_URL",
        "ALGENTA_ENGINE_URL",
        "ALGENTA_BASE_URL",
        "CODNA_ALLOW_LOOPBACK_ENGINE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "0")


@contextmanager
def _unlocked(*_args, **_kwargs):
    yield


def _unassignable_pid(offset: int) -> int:
    """A PID the kernel can never have handed to a running process.

    NEVER put a small literal here. This file used to use 123/456, which are free on macOS
    but are live, root-owned kernel threads on the Linux CI fleet. There `os.kill(pid, 0)`
    raises PermissionError, so `pid_is_alive` reports True (processes.py treats EPERM as
    "alive"), the runtime concludes its sidecar is stale, tries to terminate a process it
    does not own, and fails with EPERM -- surfacing as `agent_core_stop_permission_denied`
    inside a test about *adoption*. Whether this file passed depended on the host and not
    on the code under test: green on every developer Mac, red on every fleet runner.

    Going above the kernel maximum is deliberately stronger than picking a PID that merely
    happens to be free right now. A free PID can be recycled between module import and the
    signal, and losing that race means sending a real SIGTERM to an unrelated process on a
    shared machine. Above the maximum the kernel has nothing to match, so every probe is
    ProcessLookupError, forever.
    """
    try:
        limit = int(Path("/proc/sys/kernel/pid_max").read_text())
    except (OSError, ValueError):
        limit = 99999  # macOS PID_MAX; there is no /proc to read it from
    candidate = limit + 1 + offset
    # Enforced, not assumed. If a platform ever hands out PIDs above the limit it reports,
    # the tests below would signal a real process -- so fail loudly right here instead.
    try:
        os.kill(candidate, 0)
    except ProcessLookupError:
        return candidate
    except OSError:
        pass
    raise RuntimeError(
        f"pid {candidate} is above the reported maximum ({limit}) yet the kernel still "
        "recognises it; refusing to hand a possibly-live pid to tests that may signal it"
    )


# Distinct so a test can tell "the listener" from "its parent" -- exactly what
# test_stop_owned_sidecar_never_auto_kills_the_reported_parent_pid asserts.
_LISTENER_PID = _unassignable_pid(0)
_PARENT_PID = _unassignable_pid(1)
_ORG_B_PID = _unassignable_pid(2)
_NEW_RUNTIME_PID = _unassignable_pid(3)


def _listener(port: int, *, parent_pid: int | None = None) -> ListenerInfo:
    return ListenerInfo(
        pid=_LISTENER_PID,
        command="node",
        host="127.0.0.1",
        port=port,
        raw_name=f"127.0.0.1:{port}",
        parent_pid=parent_pid,
    )


def _healthy_probe(runtime_id: str, *, build_id: str = "test-build") -> tuple[bool, dict]:
    return (
        True,
        {
            "health": {
                "payload": {
                    "service": "codna-sidecar",
                    "runtime_id": runtime_id,
                    "build_id": build_id,
                }
            },
            "ready": {"payload": {"status": "ready"}},
        },
    )


def test_agent_core_start_reports_missing_sidecar_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_SIDECAR_DIR", str(tmp_path / "missing-agent-core"))
    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda _port: None)

    with pytest.raises(LocalRuntimeError) as excinfo:
        ensure_agent_core_running()

    assert excinfo.value.code == "agent_core_runtime_not_installed"
    assert excinfo.value.details["sidecar_dir"] == str(tmp_path / "missing-agent-core")
    assert excinfo.value.details["expected_entrypoint"].endswith("run-server.mjs")


def test_agent_core_ready_timeout_defaults_to_cold_start_budget(monkeypatch) -> None:
    monkeypatch.delenv(agent_core_runtime.AGENT_CORE_READY_TIMEOUT_ENV, raising=False)  # noqa: SLF001

    assert agent_core_runtime._agent_core_ready_timeout_seconds() == 120.0  # noqa: SLF001


@pytest.mark.parametrize("value", ["abc", "4.9", "301"])
def test_agent_core_ready_timeout_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv(agent_core_runtime.AGENT_CORE_READY_TIMEOUT_ENV, value)  # noqa: SLF001

    with pytest.raises(LocalRuntimeError) as excinfo:
        agent_core_runtime._agent_core_ready_timeout_seconds()  # noqa: SLF001

    assert excinfo.value.code == "agent_core_ready_timeout_invalid"
    assert excinfo.value.details["env"] == agent_core_runtime.AGENT_CORE_READY_TIMEOUT_ENV  # noqa: SLF001


def test_agent_core_ready_timeout_accepts_valid_override(monkeypatch) -> None:
    monkeypatch.setenv(agent_core_runtime.AGENT_CORE_READY_TIMEOUT_ENV, "120")  # noqa: SLF001

    assert agent_core_runtime._agent_core_ready_timeout_seconds() == 120.0  # noqa: SLF001


def test_agent_core_log_tail_is_bounded_and_redacted(tmp_path) -> None:
    log_path = tmp_path / "local-sidecar.log"
    log_path.write_text(
        "\n".join(
            [
                "startup line",
                "Authorization: Bearer secret-token-value",
                "OPENAI_API_KEY=sk-secretvalue",
                "PYPI_TOKEN=" + "pypi" + "-secretvalue",
                "ready failed",
            ]
        ),
        encoding="utf-8",
    )

    payload = agent_core_runtime._read_log_tail(log_path)  # noqa: SLF001

    assert payload["available"] is True
    rendered = "\n".join(payload["recent_lines"])
    assert "ready failed" in rendered
    assert "secret-token-value" not in rendered
    assert "sk-secretvalue" not in rendered
    assert "pypi" + "-secretvalue" not in rendered
    assert "[redacted]" in rendered


def test_agent_core_reuses_owned_runtime_only_when_fingerprint_matches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
    }
    terminated: list[int] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(
        agent_core_runtime,
        "_single_listener",
        lambda port: _listener(port, parent_pid=_PARENT_PID),
    )
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-1",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: terminated.append(pid))
    monkeypatch.setattr(
        agent_core_runtime,
        "_start_sidecar",
        lambda *_args, **_kwargs: AgentCoreEndpoint(
            url="http://127.0.0.1:9999",
            port=9999,
            runtime_id="new-runtime",
            pid=_NEW_RUNTIME_PID,
        ),
    )

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-1"
    assert endpoint.pid == _LISTENER_PID
    assert terminated == []


def test_agent_core_adopts_owned_local_stack_sidecar_when_agent_state_missing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    stack_state = {
        "runtime_id": "runtime-stack",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
        "sidecar": {
            "pid": _LISTENER_PID,
            "pid_create_time": None,
            "listener_pid": _LISTENER_PID,
            "listener_pid_create_time": None,
            "port": config.sidecar_port,
        },
    }
    starts: list[bool] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(
        agent_core_runtime,
        "_single_listener",
        lambda port: _listener(port, parent_pid=_PARENT_PID),
    )
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: None)
    monkeypatch.setattr(agent_core_runtime, "_load_local_stack_state", lambda _config: stack_state)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-stack",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(
        agent_core_runtime,
        "_start_sidecar",
        lambda *_args, **_kwargs: starts.append(True),
    )

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-stack"
    assert endpoint.pid == _LISTENER_PID
    assert starts == []


def test_agent_core_restarts_owned_runtime_when_fingerprint_is_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": "old-build",
        "runtime_config_hash": config.runtime_config_hash,
    }
    terminated: list[int] = []
    starts: list[bool] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port, parent_pid=_PARENT_PID))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-1",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: terminated.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)

    def _start(*_args, **_kwargs) -> AgentCoreEndpoint:
        starts.append(True)
        return AgentCoreEndpoint(
            url="http://127.0.0.1:28601",
            port=28601,
            runtime_id="runtime-2",
            pid=_PARENT_PID,
        )

    monkeypatch.setattr(agent_core_runtime, "_start_sidecar", _start)

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-2"
    assert endpoint.pid == _PARENT_PID
    # ONLY the verified listener is auto-killed -- never the reported parent_pid, which
    # on a machine running several codna invocations at once could be a live, unrelated process
    # (see test_stop_owned_sidecar_never_auto_kills_the_reported_parent_pid for the direct case).
    assert terminated == [_LISTENER_PID]
    assert starts == [True]


def test_agent_core_prefers_http_shutdown_for_owned_stale_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": "old-build",
        "runtime_config_hash": config.runtime_config_hash,
    }
    starts: list[bool] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port, parent_pid=_PARENT_PID))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-1",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "terminate_pid",
        lambda *_args, **_kwargs: pytest.fail("signal fallback should not run after HTTP shutdown"),
    )

    def _start(*_args, **_kwargs) -> AgentCoreEndpoint:
        starts.append(True)
        return AgentCoreEndpoint(
            url="http://127.0.0.1:28601",
            port=28601,
            runtime_id="runtime-2",
            pid=_PARENT_PID,
        )

    monkeypatch.setattr(agent_core_runtime, "_start_sidecar", _start)

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-2"
    assert starts == [True]


def test_agent_core_reports_stop_permission_denied_for_owned_stale_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": "old-build",
        "runtime_config_hash": config.runtime_config_hash,
    }

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port, parent_pid=_PARENT_PID))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-1",
            build_id=active_config.sidecar_build_id,
        ),
    )

    def _deny_stop(*_args, **_kwargs) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(agent_core_runtime, "terminate_pid", _deny_stop)
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)

    with pytest.raises(LocalRuntimeError) as excinfo:
        ensure_agent_core_running()

    assert excinfo.value.code == "agent_core_stop_permission_denied"
    assert excinfo.value.details["pid"] == _LISTENER_PID
    assert excinfo.value.details["port"] == config.sidecar_port
    assert excinfo.value.details["manual_stop_command"] == f"kill {_PARENT_PID} {_LISTENER_PID}"
    assert "respawn" in excinfo.value.details["manual_stop_note"]


def test_agent_core_restarts_health_proven_sidecar_when_state_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    terminated: list[int] = []
    starts: list[bool] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: None)
    monkeypatch.setattr(agent_core_runtime, "_load_local_stack_state", lambda _config: None)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-orphan",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: terminated.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)

    def _start(*_args, **_kwargs) -> AgentCoreEndpoint:
        starts.append(True)
        return AgentCoreEndpoint(
            url="http://127.0.0.1:28601",
            port=28601,
            runtime_id="runtime-2",
            pid=_PARENT_PID,
        )

    monkeypatch.setattr(agent_core_runtime, "_start_sidecar", _start)

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-2"
    assert terminated == [_LISTENER_PID]
    assert starts == [True]


def test_agent_core_restarts_owned_runtime_when_health_build_mismatches(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
    }
    terminated: list[int] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda _config: _healthy_probe("runtime-1", build_id="old-build"),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: terminated.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)
    monkeypatch.setattr(
        agent_core_runtime,
        "_start_sidecar",
        lambda *_args, **_kwargs: AgentCoreEndpoint(
            url="http://127.0.0.1:28601",
            port=28601,
            runtime_id="runtime-2",
            pid=_PARENT_PID,
        ),
    )

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-2"
    assert terminated == [_LISTENER_PID]


def test_agent_core_inspection_reports_healthy_owned_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-1",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
    }

    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-1",
            build_id=active_config.sidecar_build_id,
        ),
    )

    payload = inspect_agent_core_runtime()

    assert payload["status"] == "healthy_owned"
    assert payload["recovery"] == {
        "action": "none",
        "reason": "sidecar is healthy and owned",
    }


def test_agent_core_inspection_reports_stale_recovery_for_missing_state(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    monkeypatch.setattr(
        agent_core_runtime,
        "_single_listener",
        lambda port: _listener(port, parent_pid=_PARENT_PID),
    )
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: None)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda _config: _healthy_probe("runtime-orphan", build_id="old-build"),
    )

    payload = inspect_agent_core_runtime()

    assert payload["status"] == "stale_codna_sidecar"
    assert payload["stale_reasons"] == ["state_missing", "health_build_mismatch"]
    assert payload["recovery"] == {
        "action": "restart",
        "reason": "healthy Codna sidecar is not reusable by the current checkout",
        "codna_command": "codna doctor --start-stack",
        "manual_stop_command": f"kill {_PARENT_PID} {_LISTENER_PID}",
    }


def test_provider_mismatch_never_kills_the_reported_parent_pid(tmp_path, monkeypatch) -> None:
    """The actual P0: two concurrent `codna fix` processes share a sidecar by design (same
    runtime_config_hash), and org B's own BYOK provider key differs from org A's -- a completely
    routine, EXPECTED situation on a shared webhook worker, not a corruption. That must tear down
    the sidecar to restart it with B's env, but must NEVER touch the parent pid -- on a
    machine running concurrent fixes, that parent is frequently org A's own still-running `codna fix`
    process, not an orphaned launcher. Reproduced directly against ensure_agent_core_running with
    no mocked shortcuts around _stop_owned_sidecar itself.
    """
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    state = {
        "runtime_id": "runtime-org-a",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
        # Deliberately NO provider_env_hash -- org A's fix set this from ITS OWN provider key, and
        # the absence here (rather than a matching one) is what drives both
        # _state_provider_env_matches AND the legacy-hash fallback to False, landing on the exact
        # branch that calls _stop_owned_sidecar(reason="state_fingerprint_or_provider_mismatch").
    }
    terminated: list[int] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    # _PARENT_PID stands in for org A's OWN live `codna fix` process -- the one that spawned this sidecar
    # and is still running it, exactly as `lsof -R` reports on a real machine while that process
    # (start_new_session=True) hasn't exited yet.
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port, parent_pid=_PARENT_PID))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(agent_core_runtime, "_state_provider_env_matches", lambda _state, _keys: False)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe("runtime-org-a", build_id=active_config.sidecar_build_id),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: terminated.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)
    monkeypatch.setattr(
        agent_core_runtime, "_start_sidecar",
        lambda *_a, **_k: AgentCoreEndpoint(url="http://127.0.0.1:28601", port=28601,
                                            runtime_id="runtime-org-b", pid=_ORG_B_PID),
    )

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-org-b"      # org B correctly got its own fresh sidecar
    assert terminated == [_LISTENER_PID]                          # ONLY the verified sidecar listener
    assert _PARENT_PID not in terminated, "org A's own live process must never be auto-killed"


def test_agent_core_stop_health_proven_stale_sidecar_stops_parent_then_child(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    stopped: list[int] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(
        agent_core_runtime,
        "_single_listener",
        lambda port: _listener(port, parent_pid=_PARENT_PID),
    )
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: None)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-orphan",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: stopped.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)

    payload = stop_agent_core_runtime()

    assert payload["status"] == "stopped"
    assert payload["listener"]["pid"] == _LISTENER_PID
    assert payload["listener"]["parent_pid"] == _PARENT_PID
    # The parent pid is reported in the payload for diagnostics, but never auto-killed --
    # only the verified listener is.
    assert stopped == [_LISTENER_PID]


def test_agent_core_stop_falls_back_without_runtime_lock_for_health_proven_sidecar(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    stopped: list[int] = []

    def _deny_lock(*_args, **_kwargs):
        raise PermissionError("runtime root not writable")

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _deny_lock)
    monkeypatch.setattr(
        agent_core_runtime,
        "_single_listener",
        lambda port: _listener(port, parent_pid=_PARENT_PID),
    )
    monkeypatch.setattr(
        agent_core_runtime,
        "_load_state",
        lambda _config: pytest.fail("state must not be authority without the runtime lock"),
    )
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-orphan",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(agent_core_runtime, "terminate_pid", lambda pid, **_kwargs: stopped.append(pid))
    monkeypatch.setattr(agent_core_runtime, "_request_owned_sidecar_shutdown", lambda _config: False)

    payload = stop_agent_core_runtime()

    assert payload["status"] == "stopped"
    assert payload["lock_acquired"] is False
    assert payload["listener"]["pid"] == _LISTENER_PID
    assert payload["listener"]["parent_pid"] == _PARENT_PID
    assert stopped == [_LISTENER_PID]




def test_agent_core_stop_not_running_still_stops_mojo_daemon(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda _port: None)
    stopped = []

    def stop_mojo(config):
        stopped.append(config.sidecar_port)
        return {"status": "not_running", "state_path": "state", "socket_path": "socket"}

    monkeypatch.setattr(local_mojo_daemon_module, "stop_local_mojo_daemon", stop_mojo)

    payload = stop_agent_core_runtime()

    assert payload["status"] == "not_running"
    assert stopped == [18601]
    assert payload["mojo_daemon"]["status"] == "not_running"

def test_agent_core_stop_reports_mojo_daemon_error_after_stopping_sidecar(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port, parent_pid=_PARENT_PID))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: None)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe("runtime-1", build_id=active_config.sidecar_build_id),
    )
    sidecar_shutdowns = []
    monkeypatch.setattr(
        agent_core_runtime,
        "_request_owned_sidecar_shutdown",
        lambda config: sidecar_shutdowns.append(config.sidecar_port) or True,
    )

    def fail_stop(_config):
        raise local_mojo_daemon_module.LocalMojoDaemonError(
            "local_mojo_daemon_stop_permission_denied",
            "Codna could not stop its owned local Mojo daemon.",
            {"pid": _LISTENER_PID, "manual_stop_command": f"kill {_LISTENER_PID}"},
        )

    monkeypatch.setattr(local_mojo_daemon_module, "stop_local_mojo_daemon", fail_stop)

    payload = stop_agent_core_runtime()

    assert payload["status"] == "stopped"
    assert sidecar_shutdowns == [18601]
    assert payload["mojo_daemon"]["status"] == "error"
    assert payload["mojo_daemon"]["error"]["code"] == "local_mojo_daemon_stop_permission_denied"
    assert payload["mojo_daemon"]["error"]["details"]["manual_stop_command"] == f"kill {_LISTENER_PID}"


def test_agent_core_provider_hash_tracks_telys_runtime_config_not_request_mode(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_CORE_TELYS_LOCALIZE_ENABLED", raising=False)
    monkeypatch.delenv("AGENT_CORE_TELYS_SKIP_LINE_TARGETS", raising=False)
    monkeypatch.delenv("AGENT_CORE_TELYS_TOP_K", raising=False)
    baseline = agent_core_runtime._provider_env_hash(None)  # noqa: SLF001

    monkeypatch.setenv("AGENT_CORE_TELYS_LOCALIZE_ENABLED", "0")
    disabled = agent_core_runtime._provider_env_hash(None)  # noqa: SLF001

    monkeypatch.setenv("AGENT_CORE_TELYS_LOCALIZE_ENABLED", "1")
    enabled = agent_core_runtime._provider_env_hash(None)  # noqa: SLF001

    monkeypatch.setenv("AGENT_CORE_TELYS_TOP_K", "8")
    configured = agent_core_runtime._provider_env_hash(None)  # noqa: SLF001

    monkeypatch.delenv("AGENT_CORE_TELYS_TOP_K", raising=False)
    monkeypatch.setenv("AGENT_CORE_TELYS_SKIP_LINE_TARGETS", "1")
    skip_line_targets = agent_core_runtime._provider_env_hash(None)  # noqa: SLF001

    assert disabled == baseline
    assert enabled == baseline
    assert configured != baseline
    assert skip_line_targets != baseline


def test_agent_core_adopts_legacy_telys_request_mode_hash(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("AGENT_CORE_TELYS_LOCALIZE_ENABLED", "1")
    config = resolve_runtime_config()
    legacy_hash = agent_core_runtime._provider_env_hash_for_keys(  # noqa: SLF001
        None,
        (
            *agent_core_runtime.PROVIDER_RUNTIME_ENV_KEYS,  # noqa: SLF001
            *agent_core_runtime.LEGACY_REQUEST_MODE_ENV_KEYS,  # noqa: SLF001
        ),
        overrides={"AGENT_CORE_TELYS_LOCALIZE_ENABLED": "0"},
    )
    state = {
        "runtime_id": "runtime-legacy",
        "codna_cli_version": config.codna_cli_version,
        "sidecar_build_id": config.sidecar_build_id,
        "runtime_config_hash": config.runtime_config_hash,
        "provider_env_hash": legacy_hash,
    }
    writes: list[dict] = []

    monkeypatch.setattr(agent_core_runtime, "runtime_lock", _unlocked)
    monkeypatch.setattr(agent_core_runtime, "_single_listener", lambda port: _listener(port))
    monkeypatch.setattr(agent_core_runtime, "_load_state", lambda _config: state)
    monkeypatch.setattr(agent_core_runtime, "_state_owns_listener", lambda _state, _listener: True)
    monkeypatch.setattr(
        agent_core_runtime,
        "_service_is_codna_sidecar",
        lambda active_config: _healthy_probe(
            "runtime-legacy",
            build_id=active_config.sidecar_build_id,
        ),
    )
    monkeypatch.setattr(
        agent_core_runtime,
        "_start_sidecar",
        lambda *_args, **_kwargs: pytest.fail("legacy request-mode hash should be adopted"),
    )
    monkeypatch.setattr(
        agent_core_runtime,
        "_stop_owned_sidecar",
        lambda *_args, **_kwargs: pytest.fail("legacy request-mode hash should not rotate"),
    )

    def _write_state(_config, *, runtime_id, pid, keys):
        payload = {
            "runtime_id": runtime_id,
            "provider_env_hash": agent_core_runtime._provider_env_hash(keys),  # noqa: SLF001
            "sidecar": {"pid": pid},
        }
        writes.append(payload)
        return payload

    monkeypatch.setattr(agent_core_runtime, "_write_state", _write_state)

    endpoint = ensure_agent_core_running()

    assert endpoint.runtime_id == "runtime-legacy"
    assert endpoint.pid == _LISTENER_PID
    assert writes == [
        {
            "runtime_id": "runtime-legacy",
            "provider_env_hash": agent_core_runtime._provider_env_hash(None),  # noqa: SLF001
            "sidecar": {"pid": _LISTENER_PID},
        }
    ]


def test_agent_core_skips_optional_mojo_daemon_without_import_backed_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "1")
    monkeypatch.delenv("CODNA_REQUIRE_LOCAL_MOJO_POOL", raising=False)
    config = resolve_runtime_config()
    endpoint = AgentCoreEndpoint(
        url="http://127.0.0.1:18601",
        port=18601,
        runtime_id="runtime-packaged",
        pid=_LISTENER_PID,
    )

    monkeypatch.setattr(local_mojo_daemon_module, "local_mojo_backend_available", lambda _config: False)
    monkeypatch.setattr(
        local_mojo_daemon_module,
        "ensure_local_mojo_daemon",
        lambda _config: pytest.fail("optional packaged install must not start the import-backed Mojo daemon"),
    )

    assert agent_core_runtime._with_mojo_daemon(config, endpoint) is endpoint  # noqa: SLF001


def test_agent_core_fails_closed_when_required_mojo_backend_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "1")
    monkeypatch.setenv("CODNA_REQUIRE_LOCAL_MOJO_POOL", "1")
    config = resolve_runtime_config()
    endpoint = AgentCoreEndpoint(
        url="http://127.0.0.1:18601",
        port=18601,
        runtime_id="runtime-packaged",
        pid=_LISTENER_PID,
    )

    monkeypatch.setattr(local_mojo_daemon_module, "local_mojo_backend_available", lambda _config: False)

    with pytest.raises(LocalRuntimeError) as excinfo:
        agent_core_runtime._with_mojo_daemon(config, endpoint)  # noqa: SLF001

    assert excinfo.value.code == "local_mojo_backend_unavailable"
    assert excinfo.value.details["expected_import"] == "apps.api_server.compute.mojo_pool"
