from __future__ import annotations

import json
from types import SimpleNamespace

import codna.cli as cli_module


def _clean_mojo_workers() -> dict:
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


def test_doctor_start_stack_uses_agent_core_runtime(monkeypatch, capsys) -> None:
    def forbidden_full_stack(*_args, **_kwargs):
        raise AssertionError("doctor must not start the legacy full local engine stack")

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

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "local_agent_core"
    assert payload["database"] == "not_required"
    assert payload["started"] == {
        "url": "http://127.0.0.1:18601",
        "port": 18601,
        "runtime_id": "runtime-1",
        "pid": 123,
    }
    assert payload["mojo_workers"] == _clean_mojo_workers()


def test_doctor_read_only_uses_agent_core_inspection(monkeypatch, capsys) -> None:
    def forbidden_start(*_args, **_kwargs):
        raise AssertionError("read-only doctor must not start any runtime")

    monkeypatch.setattr(cli_module, "ensure_agent_core_running", forbidden_start)
    monkeypatch.setattr(
        cli_module,
        "inspect_agent_core_runtime",
        lambda *, keys=None: {
            "mode": "local_agent_core",
            "healthy": False,
            "database": "not_required",
        },
    )
    monkeypatch.setattr(cli_module, "_doctor_mojo_worker_diagnostics", lambda _keys: _clean_mojo_workers())

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=False))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "mode": "local_agent_core",
        "healthy": False,
        "database": "not_required",
        "mojo_workers": _clean_mojo_workers(),
    }


def test_doctor_stop_stack_uses_agent_core_stop(monkeypatch, capsys) -> None:
    def forbidden_start(*_args, **_kwargs):
        raise AssertionError("stop-stack must not start any runtime")

    monkeypatch.setattr(cli_module, "ensure_agent_core_running", forbidden_start)
    monkeypatch.setattr(
        cli_module,
        "stop_agent_core_runtime",
        lambda *, keys=None: {
            "status": "stopped",
            "url": "http://127.0.0.1:18601",
            "port": 18601,
        },
    )
    monkeypatch.setattr(cli_module, "_doctor_mojo_worker_diagnostics", lambda _keys: _clean_mojo_workers())

    rc = cli_module.cmd_doctor(SimpleNamespace(start_stack=False, stop_stack=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "stopped",
        "url": "http://127.0.0.1:18601",
        "port": 18601,
        "mojo_workers": _clean_mojo_workers(),
    }
