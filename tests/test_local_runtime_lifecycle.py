from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from codna.runtime.config import resolve_runtime_config
from codna.runtime.local_stack import LocalRuntimeError, ensure_running, stop_runtime
from codna.runtime.ports import ListenerInfo
import codna.runtime.local_stack as local_stack_module


@pytest.fixture(autouse=True)
def _clear_runtime_override_env(monkeypatch) -> None:
    for key in (
        "CODNA_ENGINE_URL",
        "ALGENTA_ENGINE_URL",
        "ALGENTA_BASE_URL",
        "CODNA_ALLOW_LOOPBACK_ENGINE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_fake_engine_checkout(tmp_path: Path) -> Path:
    engine_dir = tmp_path / "decision-engine"
    api_server_dir = engine_dir / "apps" / "api_server"
    api_server_dir.mkdir(parents=True)
    (api_server_dir / "main.py").write_text("app = object()\n", encoding="utf-8")
    return engine_dir

def test_ensure_running_full_restarts_for_sidecar_only_build_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        assert timeout_s > 0
        yield

    inspections = iter(
        [
            {
                "status": "owned_partial_or_stale",
                "runtime_id": "runtime-42",
                "fingerprint": {
                    "matches_state": False,
                    "codna_cli_matches_state": True,
                    "engine_build_id_matches_state": True,
                    "sidecar_build_id_matches_state": False,
                    "python_executable_matches_state": True,
                    "runtime_config_hash_matches_state": True,
                },
                "engine_health": {"ok": True},
                "engine_ready": {"ok": True},
                "sidecar_health": {
                    "ok": True,
                    "payload": {"service": "codna-sidecar", "runtime_id": "runtime-42"},
                },
                "sidecar_ready": {"ok": True},
                "state": {
                    "runtime_id": "runtime-42",
                    "engine": {"pid": 1001},
                    "sidecar": {"pid": 1002},
                },
            },
        ]
    )
    stopped: list[str] = []
    started: list[bool] = []

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(local_stack_module, "inspect_runtime", lambda **_kwargs: next(inspections))
    monkeypatch.setattr(
        local_stack_module,
        "_stop_proven_owned_runtime",
        lambda _config, inspection: stopped.append(str(inspection["runtime_id"])),
    )
    monkeypatch.setattr(
        local_stack_module,
        "_start_runtime",
        lambda *_args, **_kwargs: (
            started.append(True)
            or local_stack_module.RuntimeEndpoint(
                engine_url="http://127.0.0.1:18600",
                sidecar_url="http://127.0.0.1:18601",
                local=True,
                port_base=18600,
                runtime_id="runtime-43",
            )
        ),
    )

    endpoint = ensure_running()

    assert endpoint.engine_url == "http://127.0.0.1:18600"
    assert endpoint.sidecar_url == "http://127.0.0.1:18601"
    assert endpoint.runtime_id == "runtime-43"
    assert stopped == ["runtime-42"]
    assert started == [True]


def test_ensure_running_refreshes_state_when_listener_pid_changes_but_identity_matches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        assert timeout_s > 0
        yield

    inspections = iter(
        [
            {
                "status": "owned_partial_or_stale",
                "runtime_id": "runtime-42",
                "fingerprint": {"matches_state": True},
                "engine_listener": {"pid": 1001, "port": 18600},
                "sidecar_listener": {"pid": 3003, "port": 18601},
                "engine_health": {"ok": True},
                "engine_ready": {"ok": True},
                "sidecar_health": {
                    "ok": True,
                    "payload": {
                        "service": "codna-sidecar",
                        "runtime_id": "runtime-42",
                        "build_id": "build-42",
                    },
                },
                "sidecar_ready": {"ok": True},
                "state": {
                    "runtime_id": "runtime-42",
                    "sidecar_build_id": "build-42",
                    "engine": {"pid": 1001},
                    "sidecar": {"pid": 2002},
                },
            },
            {
                "status": "healthy_owned",
                "runtime_id": "runtime-42",
            },
        ]
    )
    writes: list[tuple[str, int, int]] = []

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(local_stack_module, "inspect_runtime", lambda **_kwargs: next(inspections))
    monkeypatch.setattr(
        local_stack_module,
        "_write_state",
        lambda _config, runtime_id, engine_pid, sidecar_pid: writes.append(
            (runtime_id, engine_pid, sidecar_pid)
        ),
    )
    monkeypatch.setattr(
        local_stack_module,
        "_stop_proven_owned_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not stop full runtime")),
    )

    endpoint = ensure_running()

    assert endpoint.runtime_id == "runtime-42"
    assert writes == [("runtime-42", 1001, 3003)]


def test_ensure_running_restarts_health_proven_legacy_pair(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        assert timeout_s > 0
        yield

    stopped: list[tuple[int, str]] = []

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(
        local_stack_module,
        "inspect_runtime",
        lambda **_kwargs: {
            "status": "legacy_codna_fixed_pair",
            "runtime_id": "runtime-legacy",
            "engine_listener": {"pid": 301, "port": 18600},
            "sidecar_listener": {"pid": 302, "port": 18601},
        },
    )
    monkeypatch.setattr(
        local_stack_module,
        "_terminate_owned_pid",
        lambda pid, *, service: stopped.append((pid, service)),
    )
    monkeypatch.setattr(
        local_stack_module,
        "_start_runtime",
        lambda config, **_kwargs: local_stack_module.RuntimeEndpoint(
            engine_url=config.engine_url,
            sidecar_url=config.sidecar_url,
            local=True,
            port_base=config.port_base,
            runtime_id="runtime-new",
        ),
    )

    endpoint = ensure_running()

    assert endpoint.runtime_id == "runtime-new"
    assert stopped == [(301, "engine"), (302, "sidecar")]


def test_write_state_canonicalizes_live_listener_pid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = resolve_runtime_config()
    create_times = {
        1001: 1001.0,
        3003: 3003.0,
    }

    monkeypatch.setattr(
        local_stack_module,
        "_single_listener",
        lambda port: ListenerInfo(
            pid=1001 if port == config.engine_port else 3003,
            command="uvicorn apps.api_server.main:app" if port == config.engine_port else "node run-server.mjs",
            host="127.0.0.1",
            port=port,
            raw_name=f"127.0.0.1:{port}",
        ),
    )
    monkeypatch.setattr(local_stack_module, "pid_create_time", lambda pid: create_times.get(pid))

    payload = local_stack_module._write_state(config, "runtime-1", 1001, 2002)
    persisted = json.loads(config.paths.state_path.read_text(encoding="utf-8"))

    assert payload["sidecar"]["pid"] == 3003
    assert payload["sidecar"]["listener_pid"] == 3003
    assert payload["sidecar"]["pid_create_time"] == 3003.0
    assert persisted["sidecar"]["pid"] == 3003
    assert persisted["sidecar"]["listener_pid"] == 3003


def test_ensure_running_bypasses_local_runtime_for_remote_override(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_ENGINE_URL", "https://api.codna.ai")
    monkeypatch.setattr(
        local_stack_module,
        "runtime_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("lock should not be used")),
    )

    endpoint = ensure_running()

    assert endpoint.local is False
    assert endpoint.engine_url == "https://api.codna.ai"
    assert endpoint.sidecar_url is None


def test_ensure_running_raises_deterministic_collision(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        yield

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(
        local_stack_module,
        "inspect_runtime",
        lambda **_kwargs: {
            "status": "collision",
            "engine_listener": {"pid": 999, "command": "python -m other", "port": 18600},
            "sidecar_listener": None,
        },
    )

    with pytest.raises(LocalRuntimeError) as excinfo:
        ensure_running()

    assert excinfo.value.code == "port_collision"


def test_ensure_running_wraps_lock_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        raise TimeoutError(f"timed out after {timeout_s}")
        yield

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)

    with pytest.raises(LocalRuntimeError) as excinfo:
        ensure_running()

    assert excinfo.value.code == "runtime_lock_timeout"


def test_ensure_running_wraps_runtime_permission_denied(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        raise PermissionError("EPERM")
        yield

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)

    with pytest.raises(LocalRuntimeError) as excinfo:
        ensure_running()

    assert excinfo.value.code == "runtime_permission_denied"


def test_stop_runtime_wraps_permission_denied_for_owned_process(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        yield

    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(
        local_stack_module,
        "inspect_runtime",
        lambda **_kwargs: {
            "status": "owned_partial_or_stale",
            "runtime_id": "runtime-1",
            "state": {
                "engine": {"pid": 111},
                "sidecar": {"pid": 222},
            },
        },
    )
    monkeypatch.setattr(
        local_stack_module,
        "terminate_pid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("EPERM")),
    )

    with pytest.raises(LocalRuntimeError) as excinfo:
        stop_runtime()

    assert excinfo.value.code == "runtime_stop_permission_denied"


def test_stop_runtime_terminates_live_listener_pid_for_stale_sidecar_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    @contextmanager
    def fake_lock(_path, *, timeout_s):
        yield

    terminated: list[int] = []
    monkeypatch.setattr(local_stack_module, "runtime_lock", fake_lock)
    monkeypatch.setattr(
        local_stack_module,
        "inspect_runtime",
        lambda **_kwargs: {
            "status": "owned_partial_or_stale",
            "runtime_id": "runtime-1",
            "state": {
                "engine": {"pid": 111, "listener_pid": 111},
                "sidecar": {"pid": 222, "listener_pid": 333},
            },
        },
    )
    monkeypatch.setattr(
        local_stack_module,
        "terminate_pid",
        lambda pid, **_kwargs: terminated.append(pid),
    )

    payload = stop_runtime()

    assert payload["status"] == "stopped"
    assert payload["runtime_id"] == "runtime-1"
    assert terminated == [111, 333, 222]
