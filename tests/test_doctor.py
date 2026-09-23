from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import codna.cli as cli_module


def _clean_mojo_workers() -> dict[str, object]:
    return {
        "owned_local": {
            "status": "clean",
            "owned_by_current_runtime": True,
            "process_count": 0,
            "socket_count": 0,
        },
        "legacy_global": {
            "status": "clean",
            "owned_by_current_runtime": False,
            "process_count": 0,
            "socket_count": 0,
        },
    }


def test_doctor_read_only_does_not_start_agent_core(monkeypatch, capsys) -> None:
    def forbidden_start(*_args, **_kwargs):
        raise AssertionError("read-only doctor must not start any runtime")
    def forbidden_keychain_read():
        raise AssertionError("read-only doctor must not read OS keychain secrets")

    from codna import keystore
    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)
    monkeypatch.setattr(cli_module, "ensure_agent_core_running", forbidden_start)
    monkeypatch.setattr(
        cli_module,
        "inspect_agent_core_runtime",
        lambda *, keys=None: {
            "mode": "local_agent_core",
            "healthy": False,
            "repository_intelligence": "in_process_algenta_sdk",
            "database": "not_required",
        },
    )
    monkeypatch.setattr(cli_module, "_doctor_mojo_worker_diagnostics", lambda _keys: _clean_mojo_workers())

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=False, stop_stack=False))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "local_agent_core"
    assert payload["database"] == "not_required"
    assert payload["mojo_workers"] == _clean_mojo_workers()


def test_doctor_start_stack_starts_agent_core_only(monkeypatch, capsys) -> None:
    def forbidden_full_stack(*_args, **_kwargs):
        raise AssertionError("doctor must not start the legacy full local engine stack")

    def forbidden_keychain_read():
        raise AssertionError("doctor --start-stack must not read OS keychain secrets")

    from codna import keystore

    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)
    monkeypatch.setattr(cli_module, "ensure_running", forbidden_full_stack)
    monkeypatch.setattr(
        cli_module,
        "ensure_agent_core_running",
        lambda *, keys=None: SimpleNamespace(
            url="http://127.0.0.1:18601",
            port=18601,
            runtime_id="runtime-1",
            pid=123,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "inspect_agent_core_runtime",
        lambda *, keys=None: {
            "mode": "local_agent_core",
            "repository_intelligence": "in_process_algenta_sdk",
            "database": "not_required",
        },
    )
    monkeypatch.setattr(cli_module, "_doctor_mojo_worker_diagnostics", lambda _keys: _clean_mojo_workers())

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=True, stop_stack=False))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["started"] == {
        "url": "http://127.0.0.1:18601",
        "port": 18601,
        "runtime_id": "runtime-1",
        "pid": 123,
    }
    assert payload["database"] == "not_required"


def test_doctor_start_stack_does_not_read_keychain_for_agent_runtime(monkeypatch, capsys) -> None:
    from codna import keystore

    def forbidden_keychain_read():
        raise AssertionError("doctor --start-stack must not read OS keychain secrets")

    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)
    monkeypatch.setattr(
        cli_module,
        "ensure_agent_core_running",
        lambda *, keys=None: SimpleNamespace(url="http://127.0.0.1:18601", port=18601, runtime_id="runtime-1", pid=123),
    )
    monkeypatch.setattr(
        cli_module,
        "inspect_agent_core_runtime",
        lambda *, keys=None: {"mode": "local_agent_core", "database": "not_required"},
    )
    monkeypatch.setattr(cli_module, "_doctor_mojo_worker_diagnostics", lambda _keys: _clean_mojo_workers())

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=True, stop_stack=False))

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["started"]["runtime_id"] == "runtime-1"


def test_engine_url_key_rejects_explicit_loopback_override(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_ENGINE_URL", "http://127.0.0.1:9999")
    monkeypatch.delenv("CODNA_ALLOW_LOOPBACK_ENGINE_URL", raising=False)

    with pytest.raises(cli_module.RuntimeConfigError) as excinfo:
        cli_module._engine_url_key()

    assert excinfo.value.code == "loopback_override_rejected"
