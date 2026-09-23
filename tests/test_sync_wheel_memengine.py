"""Tests for scripts/sync_wheel_memengine_from_staged_runtime.py (option-B wheel-engine pairing).

The contract: at publish time the codna wheel's memengine is REPLACED by the engine staged from the
signed telys bundle — and anything short of a fully-provenanced staged engine fails closed, so a
publish can never silently ship the repo's vendored copy.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sync_wheel_memengine_from_staged_runtime.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sync_wheel_memengine", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stage(tmp_path: Path, *, provenance: dict | None, files: dict[str, bytes]) -> tuple[Path, Path]:
    """Build a staged runtime dir + a fake cli dir; return (cli_dir, runtime_dir)."""
    runtime_dir = tmp_path / "runtime"
    manifest: dict = {"schema_version": 1, "artifact": "libame_kernel.so"}
    if provenance is not None:
        manifest["memengine"] = provenance
    runtime_dir.mkdir()
    (runtime_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, payload in files.items():
        destination = runtime_dir / "memengine" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    cli_dir = tmp_path / "cli"
    (cli_dir / "memengine").mkdir(parents=True)
    (cli_dir / "memengine" / "runtime.py").write_bytes(b"STALE VENDORED COPY\n")
    return cli_dir, runtime_dir


def test_sync_replaces_vendored_memengine_with_staged_engine(tmp_path: Path):
    tool = _load_script()
    files = {"__init__.py": b"__version__ = '0.1.3'\n", "runtime.py": b"def build(*args, texts=None): ...\n"}
    provenance = {"wheel": "telys_runtime_native-0.1.3-py3-none-any.whl",
                  "sha256": "a" * 64, "size_bytes": 1234, "file_count": len(files)}
    cli_dir, runtime_dir = _stage(tmp_path, provenance=provenance, files=files)

    payload = tool.sync(cli_dir, runtime_dir)

    assert (cli_dir / "memengine" / "runtime.py").read_bytes() == files["runtime.py"]
    assert (cli_dir / "memengine" / "__init__.py").read_bytes() == files["__init__.py"]
    assert b"STALE VENDORED COPY" not in (cli_dir / "memengine" / "runtime.py").read_bytes()
    assert payload["source_wheel"] == provenance["wheel"]
    assert payload["sha256"] == provenance["sha256"]
    assert payload["files"] == 2


def test_sync_removes_vendored_files_the_bundle_engine_no_longer_has(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "b" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "obsolete_module.py").write_bytes(b"gone in the new engine\n")

    tool.sync(cli_dir, runtime_dir)

    assert not (cli_dir / "memengine" / "obsolete_module.py").exists()


def test_sync_fails_closed_without_memengine_provenance(tmp_path: Path):
    # Legacy bundle (no runtime wheel in the signed manifest): publishing the repo's vendored copy
    # silently would pair an arbitrary engine with the staged kernel — refuse.
    tool = _load_script()
    cli_dir, runtime_dir = _stage(tmp_path, provenance=None, files={})

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "no memengine provenance" in str(excinfo.value)
    # The vendored copy is untouched on refusal.
    assert (cli_dir / "memengine" / "runtime.py").read_bytes() == b"STALE VENDORED COPY\n"


def test_sync_fails_when_manifest_missing(tmp_path: Path):
    tool = _load_script()
    cli_dir = tmp_path / "cli"
    (cli_dir / "memengine").mkdir(parents=True)

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, tmp_path / "no-runtime-dir")

    assert "manifest is missing" in str(excinfo.value)


def test_sync_fails_on_staged_file_count_mismatch(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "c" * 64, "size_bytes": 1, "file_count": 5},
        files={"__init__.py": b""},
    )

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "file count" in str(excinfo.value)


def test_sync_fails_when_staged_package_dir_missing(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(tmp_path, provenance={
        "wheel": "w.whl", "sha256": "d" * 64, "size_bytes": 1, "file_count": 1}, files={})

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "staged memengine package is missing" in str(excinfo.value)


def test_main_cli_reports_json(tmp_path: Path):
    tool = _load_script()
    files = {"__init__.py": b""}
    cli_dir, runtime_dir = _stage(tmp_path, provenance={
        "wheel": "w.whl", "sha256": "e" * 64, "size_bytes": 1, "file_count": 1}, files=files)

    rc = tool.main(["--cli-dir", str(cli_dir), "--runtime-dir", str(runtime_dir)])

    assert rc == 0


def test_main_cli_exit_1_without_provenance(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(tmp_path, provenance=None, files={})

    rc = tool.main(["--cli-dir", str(cli_dir), "--runtime-dir", str(runtime_dir)])

    assert rc == 1
