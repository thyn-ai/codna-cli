from __future__ import annotations

from codna.runtime import RuntimeEndpoint
import codna.supervisor as supervisor_module


def test_start_detached_uses_owned_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "ensure_running",
        lambda: RuntimeEndpoint(
            engine_url="http://127.0.0.1:18600",
            sidecar_url="http://127.0.0.1:18601",
            local=True,
            port_base=18600,
            runtime_id="runtime-123",
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "inspect_runtime",
        lambda: {"status": "healthy_owned", "state_path": "/tmp/local-stack.json"},
    )

    payload = supervisor_module.start_detached()

    assert payload["status"] == "healthy_owned"
    assert payload["started"] == {
        "engine_url": "http://127.0.0.1:18600",
        "sidecar_url": "http://127.0.0.1:18601",
        "local": True,
        "port_base": 18600,
        "runtime_id": "runtime-123",
    }


def test_status_payload_proxies_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "inspect_runtime",
        lambda: {"status": "not_running", "state_path": "/tmp/local-stack.json"},
    )

    assert supervisor_module.status_payload() == {
        "status": "not_running",
        "state_path": "/tmp/local-stack.json",
    }


def test_stop_running_proxies_runtime_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "stop_runtime",
        lambda: {"status": "stopped", "state_path": "/tmp/local-stack.json"},
    )

    assert supervisor_module.stop_running() == {
        "status": "stopped",
        "state_path": "/tmp/local-stack.json",
    }
