"""Tests for scripts/sync_wheel_memengine_from_staged_runtime.py (option-B wheel-engine pairing).

The contract: at publish time the codna wheel's memengine is REPLACED by the engine staged from the
signed telys bundle — and anything short of a fully-provenanced staged engine fails closed, so a
publish can never silently ship the repo's vendored copy.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
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
    _write_overlay_manifest(cli_dir, names=["_pynative.py"])
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"# fallback\n")

    payload = tool.sync(cli_dir, runtime_dir)

    assert (cli_dir / "memengine" / "runtime.py").read_bytes() == files["runtime.py"]
    assert (cli_dir / "memengine" / "__init__.py").read_bytes() == files["__init__.py"]
    assert b"STALE VENDORED COPY" not in (cli_dir / "memengine" / "runtime.py").read_bytes()
    assert payload["source_wheel"] == provenance["wheel"]
    assert payload["sha256"] == provenance["sha256"]
    assert payload["files"] == 4  # two staged files + the preserved fallback + the overlay record


def test_sync_removes_vendored_files_the_bundle_engine_no_longer_has(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "b" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    _write_overlay_manifest(cli_dir, names=["_pynative.py"])
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"# fallback\n")
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
    _write_overlay_manifest(cli_dir, names=["_pynative.py"])
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"# fallback\n")

    rc = tool.main(["--cli-dir", str(cli_dir), "--runtime-dir", str(runtime_dir)])

    assert rc == 0


def test_main_cli_exit_1_without_provenance(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(tmp_path, provenance=None, files={})

    rc = tool.main(["--cli-dir", str(cli_dir), "--runtime-dir", str(runtime_dir)])

    assert rc == 1


# ── downstream overlay: the staged bundle is a mirror of a package that lacks codna's fallback ──────
def _write_overlay_manifest(cli_dir: Path, *, names: list[str]) -> None:
    (cli_dir / "memengine" / "_codna_overlay.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preserve": [{"path": name, "role": "reference"} for name in names],
                "overlaid_modules": [],
            }
        ),
        encoding="utf-8",
    )


def test_sync_preserves_codna_only_files_the_staged_bundle_never_had(tmp_path: Path):
    """A wholesale replace would drop the NumPy reference and leave mojo_backend importing it.

    The released wheel would then raise ImportError on any platform with no Mojo shared library, instead
    of degrading to the documented pure-NumPy path (AGENTS.md section 24.8).
    """
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "f" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    for name in ("_pynative.py", "_pykernel.py"):
        (cli_dir / "memengine" / name).write_bytes(f"# {name} downstream\n".encode())
    _write_overlay_manifest(cli_dir, names=["_pynative.py", "_pykernel.py"])

    payload = tool.sync(cli_dir, runtime_dir)

    for name in ("_pynative.py", "_pykernel.py"):
        assert (cli_dir / "memengine" / name).read_bytes() == f"# {name} downstream\n".encode()
    assert sorted(payload["preserved"]) == ["_pykernel.py", "_pynative.py"]
    # The preserved files and their record are part of the wheel, so the reported count must include them.
    assert payload["files"] == 4


def test_sync_fails_closed_when_the_repo_copy_lost_an_overlay_file(tmp_path: Path):
    """Publishing a wheel with no fallback is the failure this guard exists to prevent — refuse instead."""
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "g" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"present\n")
    _write_overlay_manifest(cli_dir, names=["_pynative.py", "_pykernel.py"])

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "_pykernel.py" in str(excinfo.value)


def test_sync_with_an_overlay_manifest_preserving_only_its_record(tmp_path: Path):
    """The manifest survives even when it names no other codna-only file, and the wheel ships the record."""
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "h" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    _write_overlay_manifest(cli_dir, names=["runtime.py"])

    payload = tool.sync(cli_dir, runtime_dir)

    assert payload["preserved"] == ["runtime.py"]
    assert (cli_dir / "memengine" / "_codna_overlay.json").is_file()
    assert (cli_dir / "memengine" / "__init__.py").read_bytes() == b""


def test_sync_refuses_a_corrupt_overlay_manifest(tmp_path: Path):
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "i" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "_codna_overlay.json").write_text("{nope", encoding="utf-8")

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "invalid JSON" in str(excinfo.value)


# ── the overlay record itself must survive the swap ─────────────────────────────────────────────────
def test_sync_carries_the_overlay_manifest_into_the_published_package(tmp_path: Path):
    """The swap rmtree's the directory the manifest lives in; the manifest is not in its own 'preserve'.

    Without an explicit carry, the released wheel ships _pynative.py and _pykernel.py with no record of
    why they are there — and the NEXT publish reads an empty policy and strips them (the bootstrap paradox).
    """
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "j" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    for name in ("_pynative.py", "_pykernel.py"):
        (cli_dir / "memengine" / name).write_bytes(f"# {name}\n".encode())
    _write_overlay_manifest(cli_dir, names=["_pynative.py", "_pykernel.py"])
    before = (cli_dir / "memengine" / "_codna_overlay.json").read_bytes()

    tool.sync(cli_dir, runtime_dir)

    manifest = cli_dir / "memengine" / "_codna_overlay.json"
    assert manifest.is_file(), "the overlay manifest did not survive the swap"
    assert manifest.read_bytes() == before, "the overlay manifest was not carried across byte-for-byte"


def test_a_second_publish_still_preserves_the_codna_only_files(tmp_path: Path):
    """Red on the pre-fix code: the first swap deleted the record, so the second found nothing to keep.

    This is the one-release-later recurrence the whole guard exists to prevent, so it is asserted as a
    sequence rather than file-by-file: one publish must not disarm the next.
    """
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "k" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    for name in ("_pynative.py", "_pykernel.py"):
        (cli_dir / "memengine" / name).write_bytes(f"# {name}\n".encode())
    _write_overlay_manifest(cli_dir, names=["_pynative.py", "_pykernel.py"])

    tool.sync(cli_dir, runtime_dir)
    second = tool.sync(cli_dir, runtime_dir)

    assert sorted(second["preserved"]) == ["_pykernel.py", "_pynative.py"]
    for name in ("_pynative.py", "_pykernel.py"):
        assert (cli_dir / "memengine" / name).read_bytes() == f"# {name}\n".encode()


def test_sync_fails_closed_when_the_overlay_manifest_is_absent(tmp_path: Path):
    """An absent record must not read as 'preserve nothing' in a publish path — that strips the fallback."""
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "l" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"present\n")
    (cli_dir / "memengine" / "_pykernel.py").write_bytes(b"present\n")

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "overlay manifest is missing" in str(excinfo.value)
    assert (cli_dir / "memengine" / "runtime.py").read_bytes() == b"STALE VENDORED COPY\n"


def test_sync_fails_closed_when_the_vendored_package_is_absent(tmp_path: Path):
    """No package means no record to carry across, and a wheel with no fallback at all."""
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "m" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    shutil.rmtree(cli_dir / "memengine")

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert "vendored memengine package is missing" in str(excinfo.value)


@pytest.mark.parametrize(
    "entry",
    [
        "../../etc/passwd",
        "../escape.py",
        "/etc/hosts",
        "sub/../../escape.py",
        "..\\escape.py",
        "sub\\escape.py",
    ],
)
def test_sync_rejects_an_overlay_entry_that_escapes_the_package(tmp_path: Path, entry: str):
    """The manifest is a repo file, so this is a malicious-or-malformed PR: a publish must not ship it.

    Pre-fix, only non-empty strings were checked, so `read_bytes()` would have copied the target's contents
    into the wheel under the entry's name.
    """
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "n" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"present\n")
    _write_overlay_manifest(cli_dir, names=[entry, "_pynative.py"])

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)

    assert repr(entry) in str(excinfo.value)
    assert (cli_dir / "memengine" / "runtime.py").read_bytes() == b"STALE VENDORED COPY\n"


def test_sync_accepts_a_nested_overlay_entry_inside_the_package(tmp_path: Path):
    """Confinement must reject escapes without rejecting the package's own subdirectories."""
    tool = _load_script()
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "o" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    (cli_dir / "memengine" / "kernels").mkdir()
    (cli_dir / "memengine" / "kernels" / "fallback.py").write_bytes(b"# nested downstream\n")
    _write_overlay_manifest(cli_dir, names=["kernels/fallback.py"])

    payload = tool.sync(cli_dir, runtime_dir)

    assert payload["preserved"] == ["kernels/fallback.py"]
    assert (cli_dir / "memengine" / "kernels" / "fallback.py").read_bytes() == b"# nested downstream\n"


def test_the_post_swap_check_fails_loudly_when_the_manifest_is_not_restored(tmp_path: Path, monkeypatch):
    """Belt and braces: even if the carry step were removed, the published package is re-inspected.

    The carry in ``sync`` is unconditional; this asserts the independent verification after it, so a future
    edit that drops the carry cannot produce a silent, manifest-less wheel.
    """
    tool = _load_script()
    package = tmp_path / "direct-package"
    package.mkdir()

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool._require_overlay_manifest(package)
    assert "did not survive the swap" in str(excinfo.value)

    # And through sync(): with the carry removed, the published package must not be reported as a success.
    cli_dir, runtime_dir = _stage(
        tmp_path,
        provenance={"wheel": "w.whl", "sha256": "p" * 64, "size_bytes": 1, "file_count": 1},
        files={"__init__.py": b""},
    )
    _write_overlay_manifest(cli_dir, names=["_pynative.py"])
    (cli_dir / "memengine" / "_pynative.py").write_bytes(b"present\n")
    monkeypatch.setattr(tool, "_restore", lambda package, name, payload: None)

    with pytest.raises(tool.SyncMemengineError) as excinfo:
        tool.sync(cli_dir, runtime_dir)
    assert "did not survive the swap" in str(excinfo.value)
