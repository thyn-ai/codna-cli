from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from codna.local_client import LocalCodnaRuntimeClient
from codna.runtime.config import ConfigValue
import codna.local_client as local_client_module
import codna.local_mojo_daemon as mojo_daemon_module
import codna.local_mojo_pool as mojo_pool_module
import codna.packaged_agent_runner as packaged_agent_runner_module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_packaged_agent_patch_capture_excludes_generated_untracked_artifacts(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def price():\n    return 1\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "app.py")
    _git(repo, "-c", "user.email=a@example.com", "-c", "user.name=Tester", "commit", "-m", "initial")
    (repo / "app.py").write_text("def price():\n    return 2\n", encoding="utf-8")
    (repo / "new_module.py").write_text("def tax():\n    return 0\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"cache")
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "README.md").write_text("cache\n", encoding="utf-8")

    patch = packaged_agent_runner_module._capture_patch(repo)

    assert "diff --git a/app.py b/app.py" in patch
    assert "diff --git a/new_module.py b/new_module.py" in patch
    assert "__pycache__" not in patch
    assert ".pytest_cache" not in patch
    assert ".pyc" not in patch


class _Dumpable:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, *, mode="python"):
        assert mode == "json"
        return dict(self.payload)


class _Request:
    def __init__(self, payload):
        self.payload = dict(payload)

    @classmethod
    def model_validate(cls, payload):
        return cls(payload)


class _ConnectorType:
    def __init__(self, value):
        if value not in {"local_repo", "github_repo", "repo_archive"}:
            raise ValueError(value)
        self.value = value


def _fake_modules(core):
    data_connector = SimpleNamespace(
        ConnectorType=_ConnectorType,
        ConnectorStatus=SimpleNamespace(LIVE="live"),
        ConnectorVisibility=SimpleNamespace(PRIVATE=SimpleNamespace(value="private")),
        DataConnector=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    schemas = SimpleNamespace(
        RepositorySnapshotCreateRequest=_Request,
        RepositoryTriageRequest=_Request,
        RepositoryGraphQueryRequest=_Request,
        RepositoryDecisionPlanCreateRequest=_Request,
        RepositorySimulationRequest=_Request,
        RepositoryApplyRequest=_Request,
    )
    return local_client_module._RepositoryModules(
        core=core,
        schemas=schemas,
        data_connector=data_connector,
    )


def _raise_missing_apps_backend(_config):
    try:
        raise ModuleNotFoundError("No module named 'apps'", name="apps")
    except ModuleNotFoundError as exc:
        raise local_client_module.LocalCodnaClientError(
            "local_repository_import_failed",
            "Codna could not import the local Algenta repository-intelligence SDK/core modules.",
            {"reason": "ModuleNotFoundError: No module named 'apps'"},
        ) from exc


def _raise_broken_default_apps_backend(_config):
    try:
        raise AssertionError("dev checkout import failed")
    except AssertionError as exc:
        raise local_client_module.LocalCodnaClientError(
            "local_repository_import_failed",
            "Codna could not import the local Algenta repository-intelligence SDK/core modules.",
            {"reason": "AssertionError: dev checkout import failed"},
        ) from exc


def _reset_local_mojo_pool_state() -> None:
    with mojo_pool_module.LOCAL_MOJO_POOL_LOCK:
        mojo_pool_module.LOCAL_MOJO_POOL_STATE = None
        mojo_pool_module.LOCAL_MOJO_POOL_ATTEMPTED = False
        mojo_pool_module.LOCAL_MOJO_POOL_START_EVENT = None
        mojo_pool_module.LOCAL_MOJO_POOL_START_ERROR = None


@pytest.fixture(autouse=True)
def _local_mojo_pool_test_isolation(monkeypatch):
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_PREWARM", "0")
    _reset_local_mojo_pool_state()
    yield
    _reset_local_mojo_pool_state()

def test_local_client_prewarms_mojo_runtime_on_start(tmp_path, monkeypatch) -> None:
    calls = []

    def prewarm_runtime(config):
        calls.append(config.paths.root)

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_prewarm_local_mojo_runtime", prewarm_runtime)

    LocalCodnaRuntimeClient()

    assert calls == [(tmp_path / ".codna").resolve()]


def test_local_client_propagates_required_prewarm_error(tmp_path, monkeypatch) -> None:
    calls = []

    def prewarm_runtime(config):
        calls.append(config.paths.root)
        raise RuntimeError("prewarm failed")

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_prewarm_local_mojo_runtime", prewarm_runtime)

    with pytest.raises(RuntimeError, match="prewarm failed"):
        LocalCodnaRuntimeClient()

    assert calls == [(tmp_path / ".codna").resolve()]


def test_local_mojo_pool_waits_for_in_progress_start(tmp_path, monkeypatch) -> None:
    starts = []
    starter_entered = threading.Event()
    release_start = threading.Event()
    second_returned = threading.Event()
    errors: list[BaseException] = []

    def start_pool(config):
        starts.append(config.paths.root)
        starter_entered.set()
        assert release_start.wait(timeout=2.0)
        return mojo_pool_module.LocalMojoPoolState(
            pool=SimpleNamespace(running=True),
            loop=SimpleNamespace(is_closed=lambda: False),
            thread=threading.Thread(),
            set_pool=lambda _pool: None,
            config=config,
        )

    def run_ensure(config, done: threading.Event | None = None) -> None:
        try:
            mojo_pool_module.ensure_local_mojo_pool(config)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_START_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(mojo_pool_module, "start_local_mojo_pool_state", start_pool)
    config = local_client_module.resolve_runtime_config()

    first = threading.Thread(target=run_ensure, args=(config,))
    first.start()
    assert starter_entered.wait(timeout=1.0)

    second = threading.Thread(target=run_ensure, args=(config, second_returned))
    second.start()
    assert not second_returned.wait(timeout=0.05)

    release_start.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert starts == [(tmp_path / ".codna").resolve()]
    assert second_returned.is_set()


def test_local_mojo_runtime_prewarm_starts_daemon_in_background(tmp_path, monkeypatch) -> None:
    calls = []
    started = threading.Event()

    def ensure_daemon(config):
        calls.append(config.paths.root)
        started.set()

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "1")
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_PREWARM", "1")
    monkeypatch.delenv("CODNA_REQUIRE_LOCAL_MOJO_POOL", raising=False)
    monkeypatch.setattr(mojo_daemon_module, "local_mojo_backend_available", lambda _config: True)
    monkeypatch.setattr(mojo_daemon_module, "ensure_local_mojo_daemon", ensure_daemon)
    config = local_client_module.resolve_runtime_config()

    mojo_daemon_module.prewarm_local_mojo_runtime(config)

    assert started.wait(timeout=1.0)
    assert calls == [(tmp_path / ".codna").resolve()]


def test_local_mojo_runtime_required_prewarm_is_synchronous(tmp_path, monkeypatch) -> None:
    calls = []

    def ensure_daemon(config):
        calls.append(config.paths.root)

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "1")
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_PREWARM", "1")
    monkeypatch.setenv("CODNA_REQUIRE_LOCAL_MOJO_POOL", "1")
    monkeypatch.setattr(mojo_daemon_module, "local_mojo_backend_available", lambda _config: True)
    monkeypatch.setattr(mojo_daemon_module, "ensure_local_mojo_daemon", ensure_daemon)
    config = local_client_module.resolve_runtime_config()

    mojo_daemon_module.prewarm_local_mojo_runtime(config)

    assert calls == [(tmp_path / ".codna").resolve()]


def test_local_mojo_runtime_background_prewarm_records_failure(tmp_path, monkeypatch) -> None:
    attempted = threading.Event()

    def fail_daemon(_config):
        attempted.set()
        raise mojo_daemon_module.LocalMojoDaemonError(
            "local_mojo_daemon_start_failed",
            "Codna local Mojo daemon did not become ready.",
            {"phase": "test"},
        )

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "1")
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_PREWARM", "1")
    monkeypatch.delenv("CODNA_REQUIRE_LOCAL_MOJO_POOL", raising=False)
    monkeypatch.setattr(mojo_daemon_module, "local_mojo_backend_available", lambda _config: True)
    monkeypatch.setattr(mojo_daemon_module, "ensure_local_mojo_daemon", fail_daemon)
    config = local_client_module.resolve_runtime_config()

    mojo_daemon_module.prewarm_local_mojo_runtime(config)

    assert attempted.wait(timeout=1.0)
    log_path = mojo_daemon_module.local_mojo_daemon_log_path(config)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not log_path.exists():
        time.sleep(0.01)
    log_text = log_path.read_text(encoding="utf-8")
    assert "local_mojo_daemon_prewarm_failed" in log_text
    assert "local_mojo_daemon_start_failed" in log_text


def test_local_mojo_runtime_can_use_in_process_pool_when_daemon_disabled(tmp_path, monkeypatch) -> None:
    calls = []

    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("CODNA_LOCAL_MOJO_DAEMON_ENABLED", "0")
    monkeypatch.setattr(mojo_daemon_module, "ensure_local_mojo_pool", lambda config: calls.append(config.paths.root))
    config = local_client_module.resolve_runtime_config()

    mojo_daemon_module.ensure_local_mojo_runtime(config)

    assert calls == [(tmp_path / ".codna").resolve()]


def test_local_mojo_pool_compute_warmup_invokes_persistent_pool(monkeypatch) -> None:
    seen = {}
    loop = asyncio_loop = mojo_pool_module.asyncio.new_event_loop()
    thread = threading.Thread(
        target=mojo_pool_module.run_local_mojo_loop,
        args=(asyncio_loop,),
        daemon=True,
    )

    class FakePool:
        async def invoke(self, payload, *, timeout=None):
            seen["payload"] = payload
            seen["timeout"] = timeout
            return {"summary": {"mean": 100.0}}

    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_WARMUP_TIMEOUT_SECONDS", "2")
    thread.start()
    try:
        mojo_pool_module.warm_local_mojo_pool(FakePool(), loop)
    finally:
        mojo_pool_module.stop_loop(loop, thread)

    assert seen["payload"]["engine_type"] == "monte_carlo"
    assert seen["payload"]["variables"][0]["params"]["value"] == 100.0
    assert seen["timeout"] == 2.0


def test_local_mojo_pool_compute_warmup_can_be_disabled(monkeypatch) -> None:
    calls = []
    loop = mojo_pool_module.asyncio.new_event_loop()

    class FakePool:
        async def invoke(self, payload, *, timeout=None):
            calls.append(payload)
            return {"summary": {"mean": 100.0}}

    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_COMPUTE_WARMUP", "0")

    mojo_pool_module.warm_local_mojo_pool(FakePool(), loop)

    assert calls == []


def test_local_mojo_pool_compute_warmup_rejects_zero_mean() -> None:
    loop = asyncio_loop = mojo_pool_module.asyncio.new_event_loop()
    thread = threading.Thread(
        target=mojo_pool_module.run_local_mojo_loop,
        args=(asyncio_loop,),
        daemon=True,
    )

    class FakePool:
        async def invoke(self, payload, *, timeout=None):
            return {"summary": {"mean": 0.0}}

    thread.start()
    try:
        with pytest.raises(mojo_pool_module.LocalMojoPoolError) as excinfo:
            mojo_pool_module.warm_local_mojo_pool(FakePool(), loop)
    finally:
        mojo_pool_module.stop_loop(loop, thread)

    assert excinfo.value.code == "local_mojo_pool_warmup_failed"


def test_local_mojo_pool_configures_owned_socket_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    fake_mojo_pool = SimpleNamespace(_SOCKET_DIR=tmp_path / "old")
    monkeypatch.setattr(
        mojo_pool_module,
        "current_mojo_pool_module",
        lambda _config: fake_mojo_pool,
    )

    config = local_client_module.resolve_runtime_config()
    mojo_pool_module.configure_local_mojo_socket_dir(config)

    expected = (mojo_pool_module.local_mojo_socket_base_dir(config) / f"p-{os.getpid()}").resolve()
    assert fake_mojo_pool._SOCKET_DIR == expected
    assert expected.is_dir()


def test_local_mojo_pool_stale_cleanup_kills_only_owned_socket_pids(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    state_path = mojo_pool_module.local_mojo_worker_state_path(config)
    socket_root = mojo_pool_module.local_mojo_socket_dir(config)
    owned_socket = socket_root / "worker-0-owned.sock"
    outside_socket = tmp_path / "outside.sock"
    owned_socket.parent.mkdir(parents=True)
    owned_socket.write_text("", encoding="utf-8")
    outside_socket.write_text("", encoding="utf-8")
    mojo_pool_module.write_atomic_json(
        state_path,
        {
            "workers": [
                {"pid": 111, "socket_path": str(owned_socket)},
                {"pid": 222, "socket_path": str(outside_socket)},
            ],
        },
    )
    killed: list[tuple[int, int]] = []

    def pids_for_socket(path):
        if path == owned_socket:
            return [111]
        if path == outside_socket:
            return [222]
        return []

    def kill(pid, sig):
        killed.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(mojo_pool_module, "pids_for_socket_path", pids_for_socket)
    monkeypatch.setattr(mojo_pool_module.os, "kill", kill)
    monkeypatch.setattr(mojo_pool_module.time, "sleep", lambda _seconds: None)

    mojo_pool_module.cleanup_stale_local_mojo_workers(config)

    assert killed == [(111, signal.SIGTERM), (111, 0)]
    assert not owned_socket.exists()
    assert not socket_root.exists()
    assert not state_path.exists()
    assert outside_socket.exists()


def test_local_mojo_pool_stale_cleanup_skips_live_parent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    owned_socket = mojo_pool_module.local_mojo_socket_dir(config) / "worker-0-owned.sock"
    owned_socket.parent.mkdir(parents=True)
    owned_socket.write_text("", encoding="utf-8")
    mojo_pool_module.write_atomic_json(
        mojo_pool_module.local_mojo_worker_state_path(config),
        {
            "parent_pid": 999999,
            "socket_dir": str(owned_socket.parent),
            "workers": [{"pid": 111, "socket_path": str(owned_socket)}],
        },
    )
    monkeypatch.setattr(mojo_pool_module, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        mojo_pool_module,
        "pids_for_socket_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("should not inspect live parent workers")),
    )

    mojo_pool_module.cleanup_stale_local_mojo_workers(config)

    assert owned_socket.exists()
    assert mojo_pool_module.local_mojo_worker_state_path(config).exists()


def test_local_mojo_pool_stale_cleanup_reports_permission_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    owned_socket = mojo_pool_module.local_mojo_socket_dir(config) / "worker-0-owned.sock"
    owned_socket.parent.mkdir(parents=True)
    owned_socket.write_text("", encoding="utf-8")
    mojo_pool_module.write_atomic_json(
        mojo_pool_module.local_mojo_worker_state_path(config),
        {"workers": [{"pid": 111, "socket_path": str(owned_socket)}]},
    )

    monkeypatch.setattr(mojo_pool_module, "pids_for_socket_path", lambda _path: [111])
    monkeypatch.setattr(mojo_pool_module.os, "kill", lambda _pid, _sig: (_ for _ in ()).throw(PermissionError("nope")))

    with pytest.raises(mojo_pool_module.LocalMojoPoolError) as excinfo:
        mojo_pool_module.cleanup_stale_local_mojo_workers(config)

    assert excinfo.value.code == "local_mojo_pool_cleanup_failed"
    assert mojo_pool_module.local_mojo_worker_state_path(config).exists()


def test_local_mojo_pool_shutdown_clears_owned_worker_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    state_path = mojo_pool_module.local_mojo_worker_state_path(config)
    socket_dir = mojo_pool_module.local_mojo_socket_dir(config)
    socket_dir.mkdir(parents=True)
    mojo_pool_module.write_atomic_json(state_path, {"workers": []})
    with mojo_pool_module.LOCAL_MOJO_POOL_LOCK:
        mojo_pool_module.LOCAL_MOJO_POOL_STATE = mojo_pool_module.LocalMojoPoolState(
            pool=SimpleNamespace(running=False),
            loop=SimpleNamespace(is_closed=lambda: True),
            thread=threading.Thread(),
            set_pool=lambda _pool: None,
            config=config,
        )

    mojo_pool_module.shutdown_local_mojo_pool()

    assert not state_path.exists()
    assert not socket_dir.exists()


def test_legacy_global_mojo_worker_diagnostics_reports_clean(monkeypatch) -> None:
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert mojo_pool_module.legacy_global_mojo_worker_diagnostics() == {
        "status": "clean",
        "owned_by_current_runtime": False,
        "process_count": 0,
        "socket_count": 0,
    }


def test_legacy_global_mojo_worker_diagnostics_reports_unowned_legacy(monkeypatch) -> None:
    lsof_output = "\n".join(
        [
            "simulate 551 angel 3u unix 0x1 0t0 /tmp/algenta-mojo/worker-0-a.sock",
            "simulate 552 angel 3u unix 0x2 0t0 /tmp/algenta-mojo/worker-1-b.sock",
            "simulate 551 angel 3u unix 0x3 0t0 /tmp/algenta-mojo/worker-0-a.sock",
            "simulate 999 angel 3u unix 0x4 0t0 /tmp/codna-mojo-abcd/p-1/worker-0.sock",
            "python 777 angel 3u unix 0x5 0t0 /tmp/algenta-mojo/worker-ignored.sock",
        ]
    )
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=lsof_output, stderr=""),
    )

    payload = mojo_pool_module.legacy_global_mojo_worker_diagnostics()

    assert payload["status"] == "legacy_workers_detected"
    assert payload["owned_by_current_runtime"] is False
    assert payload["process_count"] == 2
    assert payload["socket_count"] == 2
    assert payload["socket_roots"] == ["/tmp/algenta-mojo"]
    assert payload["sample_pids"] == [551, 552]
    assert payload["cleanup_policy"] == "manual_only_unowned_legacy"
    assert payload["manual_cleanup_command"] == (
        "pids=$(lsof -nP -U | awk '$1==\"simulate\" && $0 ~ "
        "/\\/algenta-mojo\\/worker-/ {print $2}' | sort -u); "
        "[ -n \"$pids\" ] && printf '%s\\n' \"$pids\" | xargs kill"
    )


def test_legacy_global_mojo_worker_diagnostics_reports_lsof_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stdout="", stderr="denied"),
    )

    payload = mojo_pool_module.legacy_global_mojo_worker_diagnostics()

    assert payload == {
        "status": "unavailable",
        "owned_by_current_runtime": False,
        "reason": "denied",
    }


def test_owned_local_mojo_worker_diagnostics_reports_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    config = local_client_module.resolve_runtime_config()

    payload = mojo_pool_module.owned_local_mojo_worker_diagnostics(config)

    assert payload["status"] == "clean"
    assert payload["owned_by_current_runtime"] is True
    assert payload["state_present"] is False
    assert payload["recorded_worker_count"] == 0
    assert payload["live_process_count"] == 0
    assert payload["live_socket_count"] == 0


def test_owned_local_mojo_worker_diagnostics_reports_active_owned(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    socket_base = mojo_pool_module.local_mojo_socket_base_dir(config)
    socket_path = socket_base / "p-999" / "worker-0.sock"
    mojo_pool_module.write_atomic_json(
        mojo_pool_module.local_mojo_worker_state_path(config),
        {
            "parent_pid": 999,
            "workers": [{"pid": 111, "socket_path": str(socket_path)}],
        },
    )
    monkeypatch.setattr(mojo_pool_module, "pid_is_alive", lambda pid: pid == 999)
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"simulate 111 angel 3u unix 0x1 0t0 {socket_path}\n",
            stderr="",
        ),
    )

    payload = mojo_pool_module.owned_local_mojo_worker_diagnostics(config)

    assert payload["status"] == "active_owned_workers"
    assert payload["state_present"] is True
    assert payload["recorded_worker_count"] == 1
    assert payload["live_process_count"] == 1
    assert payload["live_socket_count"] == 1
    assert payload["sample_pids"] == [111]
    assert payload["parent_pid"] == 999
    assert payload["parent_alive"] is True


def test_owned_local_mojo_worker_diagnostics_reports_stale_owned(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    socket_base = mojo_pool_module.local_mojo_socket_base_dir(config)
    socket_path = socket_base / "p-999" / "worker-0.sock"
    mojo_pool_module.write_atomic_json(
        mojo_pool_module.local_mojo_worker_state_path(config),
        {
            "parent_pid": 999,
            "workers": [{"pid": 111, "socket_path": str(socket_path)}],
        },
    )
    monkeypatch.setattr(mojo_pool_module, "pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(
        mojo_pool_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"simulate 111 angel 3u unix 0x1 0t0 {socket_path}\n",
            stderr="",
        ),
    )

    payload = mojo_pool_module.owned_local_mojo_worker_diagnostics(config)

    assert payload["status"] == "stale_owned_workers"
    assert payload["cleanup_policy"] == "next_pool_start_or_process_shutdown"
    assert payload["state_present"] is True
    assert payload["recorded_worker_count"] == 1
    assert payload["live_process_count"] == 1
    assert payload["live_socket_count"] == 1
    assert payload["parent_alive"] is False


def test_mojo_worker_diagnostics_scans_unix_sockets_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    owned_socket = mojo_pool_module.local_mojo_socket_base_dir(config) / "p-1" / "worker-0.sock"
    calls = []

    def run(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    f"simulate 111 angel 3u unix 0x1 0t0 {owned_socket}",
                    "simulate 222 angel 3u unix 0x2 0t0 /tmp/algenta-mojo/worker-0-a.sock",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(mojo_pool_module.subprocess, "run", run)

    payload = mojo_pool_module.mojo_worker_diagnostics(config)

    assert len(calls) == 1
    assert payload["owned_local"]["status"] == "stale_owned_workers"
    assert payload["owned_local"]["live_process_count"] == 1
    assert payload["legacy_global"]["status"] == "legacy_workers_detected"
    assert payload["legacy_global"]["process_count"] == 1


def test_mojo_worker_diagnostics_shares_scan_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    config = local_client_module.resolve_runtime_config()
    calls = []

    def run(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(returncode=2, stdout="", stderr="lsof denied")

    monkeypatch.setattr(mojo_pool_module.subprocess, "run", run)

    payload = mojo_pool_module.mojo_worker_diagnostics(config)

    assert len(calls) == 1
    assert payload["owned_local"] == {
        "status": "unavailable",
        "owned_by_current_runtime": True,
        "reason": "lsof denied",
        "state_path": str(mojo_pool_module.local_mojo_worker_state_path(config)),
        "socket_base": str(mojo_pool_module.local_mojo_socket_base_dir(config)),
    }
    assert payload["legacy_global"] == {
        "status": "unavailable",
        "owned_by_current_runtime": False,
        "reason": "lsof denied",
    }


def test_local_mojo_pool_publish_refreshes_current_engine_singleton(tmp_path, monkeypatch) -> None:
    published = []
    simulation_service = SimpleNamespace(invoke_engine=object())

    def current_setter(config):
        assert config.paths.root == (tmp_path / ".codna").resolve()
        return lambda pool: published.append(pool)

    pool = SimpleNamespace(running=True)
    config = local_client_module.resolve_runtime_config()
    with mojo_pool_module.LOCAL_MOJO_POOL_LOCK:
        mojo_pool_module.LOCAL_MOJO_POOL_STATE = mojo_pool_module.LocalMojoPoolState(
            pool=pool,
            loop=SimpleNamespace(is_closed=lambda: False),
            thread=threading.Thread(),
            set_pool=lambda _pool: None,
            config=config,
        )
    monkeypatch.setattr(mojo_pool_module, "current_mojo_pool_setter", current_setter)
    monkeypatch.setattr(
        mojo_pool_module,
        "current_simulation_service_module",
        lambda _config: simulation_service,
    )
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    mojo_pool_module.publish_local_mojo_pool_state(local_client_module.resolve_runtime_config())

    assert published == [pool]
    assert simulation_service.invoke_engine is mojo_pool_module.local_mojo_invoke_engine_sync_bridge


def test_local_mojo_pool_size_is_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("CODNA_LOCAL_MOJO_POOL_SIZE", raising=False)
    monkeypatch.delenv("MOJO_POOL_SIZE", raising=False)

    assert mojo_pool_module.local_mojo_pool_size() == 1

    monkeypatch.setenv("MOJO_POOL_SIZE", "3")
    assert mojo_pool_module.local_mojo_pool_size() == 3

    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_SIZE", "2")
    assert mojo_pool_module.local_mojo_pool_size() == 2


def test_local_mojo_pool_skips_cross_interpreter_engine_site_packages(tmp_path) -> None:
    engine_dir = tmp_path / "decision-engine"
    current_major, current_minor = sys.version_info[:2]
    compatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor}"
        / "site-packages"
    )
    incompatible = (
        engine_dir
        / ".venv"
        / "lib"
        / f"python{current_major}.{current_minor + 1}"
        / "site-packages"
    )
    compatible.mkdir(parents=True)
    incompatible.mkdir(parents=True)

    assert mojo_pool_module.compatible_engine_site_packages(engine_dir) == [compatible]


def test_local_mojo_backend_available_treats_missing_dependency_as_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    engine_dir = tmp_path / "decision-engine"
    mojo_pool = engine_dir / "apps" / "api_server" / "compute" / "mojo_pool.py"
    mojo_pool.parent.mkdir(parents=True)
    mojo_pool.write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("ALGENTA_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    def raise_missing_dependency(_config):
        raise ModuleNotFoundError("No module named 'structlog'", name="structlog")

    monkeypatch.setattr(mojo_pool_module, "current_mojo_pool_module", raise_missing_dependency)

    assert mojo_pool_module.local_mojo_backend_available(local_client_module.resolve_runtime_config()) is False


def test_local_mojo_pool_size_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_LOCAL_MOJO_POOL_SIZE", "0")

    with pytest.raises(mojo_pool_module.LocalMojoPoolError) as excinfo:
        mojo_pool_module.local_mojo_pool_size()

    assert excinfo.value.code == "invalid_local_mojo_pool_config"


def test_verified_plan_uses_sidecar_not_loopback_engine_and_scopes_keys(tmp_path, monkeypatch) -> None:
    captured = {}

    async def create_repository_decision_plan(*, connector, request):
        captured["agent_core_url"] = os.environ.get("ALGENTA_AGENT_CORE_URL")
        captured["engine_url"] = os.environ.get("ALGENTA_ENGINE_URL")
        captured["codna_engine_url"] = os.environ.get("CODNA_ENGINE_URL")
        captured["openai_key"] = os.environ.get("OPENAI_API_KEY")
        captured["agent_token_budget"] = os.environ.get("ALGENTA_AGENT_TOKEN_BUDGET")
        captured["agent_max_turns"] = os.environ.get("ALGENTA_AGENT_MAX_TURNS")
        captured["agent_tool_profile"] = os.environ.get("ALGENTA_REPOSITORY_AGENT_TOOL_PROFILE")
        return _Dumpable({"repository_id": str(connector.id), "decision_plan_id": "plan-1"})

    core = SimpleNamespace(create_repository_decision_plan=create_repository_decision_plan)
    keys = {
        "OPENAI_API_KEY": ConfigValue("OPENAI_API_KEY", "from-keys", "keys_txt"),
        "CODNA_ENGINE_URL": ConfigValue("CODNA_ENGINE_URL", "http://127.0.0.1:8002", "keys_txt"),
    }
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ALGENTA_AGENT_CORE_URL", raising=False)
    monkeypatch.delenv("ALGENTA_ENGINE_URL", raising=False)
    monkeypatch.delenv("CODNA_ENGINE_URL", raising=False)
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))
    monkeypatch.setattr(
        local_client_module,
        "ensure_agent_core_running",
        lambda *, keys=None: SimpleNamespace(url="http://127.0.0.1:18601"),
    )

    client = LocalCodnaRuntimeClient(keys=keys)
    connector = client.create_connector(name="local-test", connector_type="local_repo", config={"path": str(tmp_path)})
    result = client.create_repository_decision_plan(
        connector["id"],
        {
            "workspace_evidence_bundle_ref": "bundle-1",
            "model": "repository.verified_agentic_v1",
        },
    )

    assert result["decision_plan_id"] == "plan-1"
    assert captured == {
        "agent_core_url": "http://127.0.0.1:18601",
        "engine_url": None,
        "codna_engine_url": None,
        "openai_key": "from-keys",
        "agent_token_budget": "6000",
        "agent_max_turns": "4",
        "agent_tool_profile": "no_exploration_local_validation",
    }
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("ALGENTA_AGENT_CORE_URL") is None
    assert os.environ.get("ALGENTA_AGENT_MAX_TURNS") is None


def test_verified_plan_preserves_explicit_agent_core_overrides(tmp_path, monkeypatch) -> None:
    captured = {}

    async def create_repository_decision_plan(*, connector, request):
        captured["agent_token_budget"] = os.environ.get("ALGENTA_AGENT_TOKEN_BUDGET")
        captured["agent_max_turns"] = os.environ.get("ALGENTA_AGENT_MAX_TURNS")
        captured["agent_tool_profile"] = os.environ.get("ALGENTA_REPOSITORY_AGENT_TOOL_PROFILE")
        return _Dumpable({"repository_id": str(connector.id), "decision_plan_id": "plan-1"})

    core = SimpleNamespace(create_repository_decision_plan=create_repository_decision_plan)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setenv("ALGENTA_AGENT_TOKEN_BUDGET", "9000")
    monkeypatch.setenv("ALGENTA_AGENT_MAX_TURNS", "3")
    monkeypatch.setenv("ALGENTA_REPOSITORY_AGENT_TOOL_PROFILE", "no_exploration_no_shell")
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))
    monkeypatch.setattr(
        local_client_module,
        "ensure_agent_core_running",
        lambda *, keys=None: SimpleNamespace(url="http://127.0.0.1:18601"),
    )

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="local-test", connector_type="local_repo", config={"path": str(tmp_path)})
    result = client.create_repository_decision_plan(
        connector["id"],
        {
            "workspace_evidence_bundle_ref": "bundle-1",
            "model": "repository.verified_agentic_v1",
        },
    )

    assert result["decision_plan_id"] == "plan-1"
    assert captured == {
        "agent_token_budget": "9000",
        "agent_max_turns": "3",
        "agent_tool_profile": "no_exploration_no_shell",
    }


def test_verified_plan_recovers_one_agent_core_transport_drop(tmp_path, monkeypatch) -> None:
    attempts = []
    stops = []
    ensures = []

    async def create_repository_decision_plan(*, connector, request):
        attempts.append(os.environ.get("ALGENTA_AGENT_CORE_URL"))
        if len(attempts) == 1:
            raise RuntimeError(
                "agent-core sidecar unreachable after 3 attempt(s): All connection attempts failed"
            )
        return _Dumpable({"repository_id": str(connector.id), "decision_plan_id": "plan-1"})

    core = SimpleNamespace(create_repository_decision_plan=create_repository_decision_plan)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))
    monkeypatch.setattr(
        local_client_module,
        "ensure_agent_core_running",
        lambda *, keys=None: ensures.append(keys) or SimpleNamespace(url="http://127.0.0.1:18601"),
    )
    monkeypatch.setattr(
        local_client_module,
        "stop_agent_core_runtime",
        lambda *, keys=None: stops.append(keys) or {"status": "stopped"},
    )

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(name="local-test", connector_type="local_repo", config={"path": str(tmp_path)})
    result = client.create_repository_decision_plan(
        connector["id"],
        {
            "workspace_evidence_bundle_ref": "bundle-1",
            "model": "repository.verified_agentic_v1",
        },
    )

    assert result["decision_plan_id"] == "plan-1"
    assert attempts == ["http://127.0.0.1:18601", "http://127.0.0.1:18601"]
    assert len(ensures) == 2
    assert len(stops) == 1
    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    assert len(artifacts) == 1
    row = pq.read_table(artifacts[0]).to_pylist()[0]
    assert row["step_name"] == "decision_plan"
    assert row["status"] == "succeeded"
    assert row["decision_plan_id"] == "plan-1"


def test_local_client_records_failed_step_artifact(tmp_path, monkeypatch) -> None:
    def triage_repository(*, connector, request):
        raise RuntimeError("synthetic triage failure")

    core = SimpleNamespace(triage_repository=triage_repository)
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    monkeypatch.setattr(local_client_module, "_import_repository_modules", lambda _config: _fake_modules(core))

    client = LocalCodnaRuntimeClient()
    connector = client.create_connector(
        name="local-test",
        connector_type="local_repo",
        config={"path": str(tmp_path)},
    )

    with pytest.raises(RuntimeError, match="synthetic triage failure"):
        client.triage_repository(connector["id"], {"issue": "broken parser"})

    artifacts = list((tmp_path / ".codna" / "repository-intelligence" / "steps").glob("**/step.parquet"))
    assert len(artifacts) == 1
    row = pq.read_table(artifacts[0]).to_pylist()[0]
    assert row["step_name"] == "triage"
    assert row["status"] == "failed"
    assert row["error_code"] == "RuntimeError"
    assert row["error_message"] == "synthetic triage failure"
