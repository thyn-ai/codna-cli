from __future__ import annotations

import signal
from types import SimpleNamespace
from pathlib import Path

import codna.runtime.processes as processes_module
from codna.runtime.processes import read_json, terminate_pid, write_json_atomic


def test_write_json_atomic_replaces_target_without_leaving_temp_file(tmp_path: Path) -> None:
    state_path = tmp_path / "runtime" / "local-stack.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text('{"stale": true}', encoding="utf-8")

    write_json_atomic(state_path, {"runtime_id": "runtime-1", "port_base": 18600})

    assert read_json(state_path) == {"runtime_id": "runtime-1", "port_base": 18600}
    assert not (state_path.parent / "local-stack.json.tmp").exists()


def test_terminate_pid_falls_back_to_kill_executable_on_permission_error(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    alive_checks = iter([True, False])

    monkeypatch.setattr(processes_module, "pid_is_alive", lambda _pid: next(alive_checks))

    def fake_os_kill(pid: int, sig: signal.Signals) -> None:
        calls.append(("os.kill", sig))
        raise PermissionError("EPERM")

    def fake_run(command: list[str], **_kwargs):
        calls.append(("subprocess.run", tuple(command)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(processes_module.os, "kill", fake_os_kill)
    monkeypatch.setattr(
        processes_module.shutil,
        "which",
        lambda name: "/bin/kill" if name == "kill" else ("/bin/zsh" if name == "zsh" else None),
    )
    monkeypatch.setattr(processes_module.subprocess, "run", fake_run)

    terminate_pid(12345, grace_timeout_s=0.01, force_timeout_s=0.01)

    assert calls == [
        ("os.kill", signal.SIGTERM),
        ("subprocess.run", ("/bin/kill", "-TERM", "12345")),
    ]
