from __future__ import annotations

import pytest

import codna.runtime.ports as ports_module
from codna.runtime.ports import (
    PortInspectionError,
    _lsof_binary,
    _parse_lsof_fields,
    _port_inspection_timeout_s,
    listeners_on_port,
)


def test_port_inspection_timeout_default_and_env_override(monkeypatch) -> None:
    monkeypatch.delenv("CODNA_PORT_INSPECTION_TIMEOUT_S", raising=False)
    assert _port_inspection_timeout_s() == 10.0  # not the old too-tight 2.0s
    monkeypatch.setenv("CODNA_PORT_INSPECTION_TIMEOUT_S", "25")
    assert _port_inspection_timeout_s() == 25.0
    for bad in ("", "0", "-3", "abc"):  # invalid/non-positive falls back to the default
        monkeypatch.setenv("CODNA_PORT_INSPECTION_TIMEOUT_S", bad)
        assert _port_inspection_timeout_s() == 10.0


def test_parse_lsof_fields_includes_parent_pid() -> None:
    listeners = _parse_lsof_fields(
        [
            "p21793",
            "R8581",
            "cbun.exe",
            "n127.0.0.1:18601",
        ]
    )

    assert len(listeners) == 1
    assert listeners[0].pid == 21793
    assert listeners[0].parent_pid == 8581
    assert listeners[0].command == "bun.exe"
    assert listeners[0].host == "127.0.0.1"
    assert listeners[0].port == 18601


def test_lsof_binary_uses_absolute_fallback_when_path_lookup_fails(tmp_path, monkeypatch) -> None:
    fallback = tmp_path / "lsof"
    fallback.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(ports_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ports_module, "LSOF_FALLBACK_PATHS", (fallback,))

    assert _lsof_binary() == str(fallback)


def test_lsof_binary_reports_structured_missing_dependency(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ports_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ports_module, "LSOF_FALLBACK_PATHS", (tmp_path / "missing-lsof",))

    with pytest.raises(PortInspectionError) as excinfo:
        _lsof_binary()

    assert excinfo.value.code == "port_inspection_unavailable"
    assert excinfo.value.details["required_binary"] == "lsof"


def test_listeners_on_port_times_out_as_empty_when_loopback_is_closed(monkeypatch) -> None:
    def timeout_run(*_args, **_kwargs):
        raise ports_module.subprocess.TimeoutExpired(cmd=["lsof"], timeout=2.0)

    monkeypatch.setattr(ports_module.subprocess, "run", timeout_run)
    monkeypatch.setattr(ports_module, "_loopback_port_accepts_connection", lambda _port: False)

    assert listeners_on_port(18601) == []


def test_listeners_on_port_reports_timeout_when_listener_may_exist(monkeypatch) -> None:
    def timeout_run(*_args, **_kwargs):
        raise ports_module.subprocess.TimeoutExpired(cmd=["lsof"], timeout=2.0)

    monkeypatch.setattr(ports_module.subprocess, "run", timeout_run)
    monkeypatch.setattr(ports_module, "_loopback_port_accepts_connection", lambda _port: True)

    with pytest.raises(PortInspectionError) as excinfo:
        listeners_on_port(18601)

    assert excinfo.value.code == "port_inspection_timeout"
    assert excinfo.value.details["port"] == 18601
