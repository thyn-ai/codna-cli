"""The bench drivers must import on a machine without the tools they drive.

``bench_env`` validates the operator-supplied binaries (``CODEX_BIN`` / ``CURSOR_BIN`` /
``CODNA_BIN``) and the security manifest (``SEC_MANIFEST``) before they reach ``subprocess``.
PR #559 ran that validation at module level, so merely importing ``benchmark_codex``,
``benchmark_cursor``, ``benchmark_fix`` or ``benchmark_security`` (tests, tooling, docs
generators) raised ``SystemExit`` wherever codex / cursor-agent / codna was not installed.
The validation now runs when a driver starts: import always succeeds, the spawn path still
fails clearly (before any work) on a missing or unvetted binary, and a vetted executable
resolves exactly as before.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

# Every env var the drivers read through bench_env; cleared so the host's own overrides never leak in.
BENCH_ENV_VARS = ("CODEX_BIN", "CURSOR_BIN", "CODNA_BIN", "SEC_MANIFEST", "CODEX_MODEL", "CURSOR_MODEL")

DRIVERS = [
    {"module": "benchmark_codex", "getter": "_codex_bin", "env_var": "CODEX_BIN", "name": "codex",
     "missing": "CODEX_BIN: 'codex' not found on PATH"},
    {"module": "benchmark_cursor", "getter": "_cursor_bin", "env_var": "CURSOR_BIN", "name": "cursor-agent",
     "missing": "CURSOR_BIN='~/.local/bin/cursor-agent': not an existing executable file"},
    {"module": "benchmark_fix", "getter": "_codna_bin", "env_var": "CODNA_BIN", "name": "codna",
     "missing": "CODNA_BIN: 'codna' not found on PATH"},
    {"module": "benchmark_security", "getter": "_codna_bin", "env_var": "CODNA_BIN", "name": "codna",
     "missing": "CODNA_BIN: 'codna' not found on PATH"},
]
DRIVER_IDS = [d["module"] for d in DRIVERS]
MODEL_DRIVERS = [d | {"model_var": v} for d, v in zip(DRIVERS[:2], ("CODEX_MODEL", "CURSOR_MODEL"))]


@pytest.fixture
def bare_machine(tmp_path, monkeypatch):
    """A host without the tools: nothing on PATH, an empty HOME, no bench overrides, no bench repos."""
    empty_path = tmp_path / "empty-path"
    home = tmp_path / "home"
    repos = tmp_path / "bench-repos"
    for d in (empty_path, home, repos):
        d.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCH_REPOS", str(repos))  # never touch the host's /tmp/bench-repos
    monkeypatch.setenv("CURSOR_API_KEY", "not-a-real-key")  # benchmark_cursor.main() checks presence only
    for var in BENCH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _fresh_import(module: str):
    """Re-execute the driver's module-level code under the current environment."""
    sys.modules.pop(module, None)
    return importlib.import_module(module)


def _make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _no_spawn(*args, **kwargs):
    pytest.fail(f"a subprocess was spawned before the operator overrides were validated: {args or kwargs}")


def _run_main(mod, driver: dict, bare_machine: Path) -> int:
    if driver["module"] == "benchmark_fix":
        return mod.main([str(bare_machine / "bench-repos")])
    return mod.main()


@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_driver_imports_without_the_tool_installed(bare_machine, driver):
    mod = _fresh_import(driver["module"])
    assert callable(getattr(mod, driver["getter"]))
    # The regression: a module-level constant holding the binary / manifest resolved at import.
    for constant in ("CODEX", "CURSOR", "CODNA", "MANIFEST"):
        assert not hasattr(mod, constant), f"{driver['module']}.{constant} is resolved at import"


@pytest.mark.parametrize("module", DRIVER_IDS)
def test_driver_imports_in_a_clean_interpreter_without_the_tools(bare_machine, module):
    """The same proof as a real process: `python -c 'import <driver>'` on the bare machine exits 0."""
    env = {k: v for k, v in os.environ.items() if k not in BENCH_ENV_VARS}
    env["PYTHONPATH"] = str(BENCH_DIR)
    p = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=str(bare_machine), env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "Traceback" not in p.stderr


@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_spawn_path_fails_clearly_before_any_work_when_the_tool_is_missing(bare_machine, monkeypatch, driver):
    mod = _fresh_import(driver["module"])
    with pytest.raises(SystemExit, match=re.escape(driver["missing"])):
        getattr(mod, driver["getter"])()
    # main() validates before any throwaway dir, repo copy, git call or report: nothing is spawned.
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    with pytest.raises(SystemExit, match=re.escape(driver["missing"])):
        _run_main(mod, driver, bare_machine)


@pytest.mark.parametrize("driver", DRIVERS, ids=DRIVER_IDS)
def test_resolution_succeeds_with_a_vetted_executable(bare_machine, monkeypatch, driver):
    mod = _fresh_import(driver["module"])
    resolve = getattr(mod, driver["getter"])
    # Bare allowlisted name, found on PATH.
    exe = _make_exe(bare_machine / "bin" / driver["name"])
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.setenv(driver["env_var"], driver["name"])
    assert resolve() == str(exe)
    # Absolute path override.
    other = _make_exe(bare_machine / "elsewhere" / driver["name"])
    monkeypatch.setenv(driver["env_var"], str(other))
    assert resolve() == str(other)
    # The allowlist still applies at spawn time.
    _make_exe(bare_machine / "bin" / "rm")
    monkeypatch.setenv(driver["env_var"], "rm")
    with pytest.raises(SystemExit, match="not an allowed command name"):
        resolve()


def test_cursor_default_path_is_resolved_under_home_at_run_time(bare_machine):
    mod = _fresh_import("benchmark_cursor")
    exe = _make_exe(Path(os.environ["HOME"]) / ".local" / "bin" / "cursor-agent")
    assert mod._cursor_bin() == str(exe)


def test_security_manifest_is_resolved_at_run_time_not_at_import(bare_machine, monkeypatch):
    monkeypatch.setenv("SEC_MANIFEST", str(bare_machine / "missing.yaml"))
    mod = _fresh_import("benchmark_security")  # imports although the manifest override points nowhere
    with pytest.raises(SystemExit, match="SEC_MANIFEST=.*file not found"):
        mod._manifest()
    monkeypatch.delenv("SEC_MANIFEST")
    assert mod._manifest() == str(BENCH_DIR / "fixtures" / "codna-security.yaml")
    manifest = bare_machine / "codna-security.yaml"
    manifest.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setenv("SEC_MANIFEST", str(manifest))
    assert mod._manifest() == str(manifest)
    # codna present but the manifest missing: main() still stops before any work.
    exe = _make_exe(bare_machine / "bin" / "codna")
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.setenv("SEC_MANIFEST", str(bare_machine / "missing.yaml"))
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    with pytest.raises(SystemExit, match="SEC_MANIFEST=.*file not found"):
        mod.main()


@pytest.mark.parametrize("driver", MODEL_DRIVERS, ids=[d["module"] for d in MODEL_DRIVERS])
def test_model_override_is_validated_at_run_time_not_at_import(bare_machine, monkeypatch, driver):
    monkeypatch.setenv(driver["model_var"], "--full-auto")  # option-like: rejected, but only when running
    mod = _fresh_import(driver["module"])
    exe = _make_exe(bare_machine / "bin" / driver["name"])
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.setenv(driver["env_var"], driver["name"])
    monkeypatch.setattr(subprocess, "run", _no_spawn)
    with pytest.raises(SystemExit, match="not a model identifier"):
        mod.main()
