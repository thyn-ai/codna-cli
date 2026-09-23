from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_telys_runtime.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_telys_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_writes_manifest_and_verify_passes(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / tool.expected_artifact_name()
    source.write_bytes(b"kernel-bytes")
    runtime_dir = tmp_path / "runtime"

    manifest = tool.stage_kernel(source, runtime_dir)
    verified = tool.verify_runtime(runtime_dir)

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == tool.expected_artifact_name()
    assert "source_path" not in manifest
    assert verified["sha256"] == manifest["sha256"]
    assert verified["path"] == str(runtime_dir / tool.expected_artifact_name())


def test_stage_supports_explicit_target_platform(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / "libame_kernel.so"
    source.write_bytes(b"kernel-bytes")
    runtime_dir = tmp_path / "runtime"

    manifest = tool.stage_kernel(source, runtime_dir, target_platform="linux-x86_64")
    verified = tool.verify_runtime(runtime_dir, target_platform="linux-x86_64")

    assert manifest["artifact"] == "libame_kernel.so"
    assert manifest["platform"] == "linux-x86_64"
    assert verified["path"] == str(runtime_dir / "libame_kernel.so")


def test_stage_records_auxiliary_runtime_artifacts(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / "libame_kernel.dylib"
    aux = tmp_path / "libKGENCompilerRTShared.dylib"
    source.write_bytes(b"kernel-bytes")
    aux.write_bytes(b"aux-bytes")

    manifest = tool.stage_kernel(
        source,
        tmp_path / "runtime",
        target_platform="macos-arm64",
        auxiliary_artifacts=[aux],
    )
    verified = tool.verify_runtime(tmp_path / "runtime", target_platform="macos-arm64")

    assert manifest["auxiliary_artifacts"][0]["name"] == aux.name
    assert verified["auxiliary_artifacts"][0]["name"] == aux.name
    assert (tmp_path / "runtime" / aux.name).read_bytes() == b"aux-bytes"


def test_verify_rejects_missing_auxiliary_runtime_artifact(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / "libame_kernel.dylib"
    aux = tmp_path / "libKGENCompilerRTShared.dylib"
    source.write_bytes(b"kernel-bytes")
    aux.write_bytes(b"aux-bytes")
    runtime_dir = tmp_path / "runtime"
    tool.stage_kernel(source, runtime_dir, target_platform="macos-arm64", auxiliary_artifacts=[aux])
    (runtime_dir / aux.name).unlink()

    with pytest.raises(tool.RuntimePrepError) as excinfo:
        tool.verify_runtime(runtime_dir, target_platform="macos-arm64")

    assert "auxiliary artifact is missing" in str(excinfo.value)


def test_stage_removes_stale_kernels_from_other_platforms(tmp_path: Path):
    tool = _load_script()
    runtime_dir = tmp_path / "runtime"
    linux = tmp_path / "libame_kernel.so"
    macos = tmp_path / "libame_kernel.dylib"
    linux.write_bytes(b"linux")
    macos.write_bytes(b"macos")

    tool.stage_kernel(linux, runtime_dir, target_platform="linux-x86_64")
    tool.stage_kernel(macos, runtime_dir, target_platform="macos-arm64")

    assert not (runtime_dir / "libame_kernel.so").exists()
    assert (runtime_dir / "libame_kernel.dylib").read_bytes() == b"macos"


def test_verify_rejects_tampered_kernel(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / tool.expected_artifact_name()
    source.write_bytes(b"kernel-bytes")
    runtime_dir = tmp_path / "runtime"
    tool.stage_kernel(source, runtime_dir)
    (runtime_dir / tool.expected_artifact_name()).write_bytes(b"tampered")

    with pytest.raises(tool.RuntimePrepError) as excinfo:
        tool.verify_runtime(runtime_dir)

    assert "size does not match" in str(excinfo.value) or "sha256 does not match" in str(excinfo.value)


def test_stage_rejects_wrong_artifact_name(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / "wrong_kernel_name.dylib"
    source.write_bytes(b"kernel-bytes")

    with pytest.raises(tool.RuntimePrepError) as excinfo:
        tool.stage_kernel(source, tmp_path / "runtime")

    assert "kernel filename must be" in str(excinfo.value)


def test_verify_rejects_manifest_with_source_path(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / tool.expected_artifact_name()
    source.write_bytes(b"kernel-bytes")
    runtime_dir = tmp_path / "runtime"
    tool.stage_kernel(source, runtime_dir)
    manifest_path = runtime_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_path"] = str(source)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(tool.RuntimePrepError) as excinfo:
        tool.verify_runtime(runtime_dir)

    assert "source_path" in str(excinfo.value)


def test_verify_rejects_platform_mismatch(tmp_path: Path):
    tool = _load_script()
    source = tmp_path / tool.expected_artifact_name()
    source.write_bytes(b"kernel-bytes")
    runtime_dir = tmp_path / "runtime"
    tool.stage_kernel(source, runtime_dir)
    manifest_path = runtime_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["platform"] = "wrong-platform"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(tool.RuntimePrepError) as excinfo:
        tool.verify_runtime(runtime_dir)

    assert "platform mismatch" in str(excinfo.value)
