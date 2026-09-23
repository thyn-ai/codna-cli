from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev_check.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dev_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_can_import_returns_false_for_missing_interpreter():
    tool = _load_script()

    assert tool._can_import(Path("/definitely/missing/codna-python"), ()) is False


def test_resolve_python_reports_required_modules_for_missing_dependency():
    tool = _load_script()

    with pytest.raises(tool.DevCheckError) as excinfo:
        tool.resolve_python(("codna_missing_dev_check_dependency",))

    message = str(excinfo.value)
    assert "codna_missing_dev_check_dependency" in message
    assert "Create .venv" in message


def test_where_prints_repo_root(capsys):
    tool = _load_script()

    tool.where()

    assert capsys.readouterr().out.strip() == str(ROOT)


def test_build_cline_sdk_serializes_shared_bun_setup(monkeypatch):
    tool = _load_script()
    events = []

    @contextlib.contextmanager
    def fake_lock(name, path, *, timeout_s):
        events.append(("lock_enter", name, path, timeout_s))
        yield
        events.append(("lock_exit", name, path, timeout_s))

    def fake_run(command, *, cwd=tool.ROOT):
        events.append(("run", command, cwd))

    monkeypatch.setattr(tool, "dev_check_lock", fake_lock)
    monkeypatch.setattr(tool, "cline_setup_lock_path", lambda: Path("/tmp/codna-test.lock"))
    monkeypatch.setattr(tool, "run", fake_run)

    tool.build_cline_sdk("bun")

    cline_dir = tool.ROOT / "agent-core" / "vendor" / "cline"
    assert events == [
        (
            "lock_enter",
            "vendored Cline SDK setup",
            Path("/tmp/codna-test.lock"),
            tool.CLINE_SETUP_LOCK_TIMEOUT_SECONDS,
        ),
        ("run", ["bun", "install"], cline_dir),
        ("run", ["bun", "run", "build:sdk"], cline_dir),
        ("run", ["bash", "algenta/postbuild.sh"], cline_dir),
        (
            "lock_exit",
            "vendored Cline SDK setup",
            Path("/tmp/codna-test.lock"),
            tool.CLINE_SETUP_LOCK_TIMEOUT_SECONDS,
        ),
    ]


def test_lint_excludes_broken_fixture_and_vendored_memengine(monkeypatch):
    tool = _load_script()
    commands = []

    monkeypatch.setattr(tool, "resolve_python", lambda modules: Path("/opt/lint/bin/python"))
    monkeypatch.setattr(tool, "run", lambda command, *, cwd=tool.ROOT: commands.append(command))

    tool.lint()

    ruff = commands[0]
    assert ruff[:5] == ["/opt/lint/bin/python", "-m", "ruff", "check", "cli"]
    excludes = [ruff[i + 1] for i, arg in enumerate(ruff) if arg == "--exclude"]
    # Both exclusions come from the script's own constants; the vendored, Telys-owned memengine
    # is replaced wholesale at publish time and must never be linted as codna code.
    assert excludes == [tool.BROKEN_FIXTURE, tool.VENDORED_MEMENGINE]
    assert tool.VENDORED_MEMENGINE == "cli/memengine"
    assert commands[1] == ["/opt/lint/bin/python", "scripts/cline_fork_guard.py"]


def test_dev_check_lock_uses_directory_fallback_without_fcntl(tmp_path, monkeypatch):
    tool = _load_script()
    lock_path = tmp_path / "cline.lock"

    monkeypatch.setattr(tool, "fcntl", None)

    with tool.dev_check_lock("test lock", lock_path, timeout_s=1):
        assert lock_path.with_suffix(".lock.d").is_dir()

    assert not lock_path.with_suffix(".lock.d").exists()
