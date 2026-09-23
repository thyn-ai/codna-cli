from __future__ import annotations

from pathlib import Path

import pytest

from codna._wheel_guard import (
    has_native_sidecar,
    requires_platform_wheel,
    staged_runtime_files,
    validate_wheel_filename,
)


def test_requires_platform_wheel_false_without_runtime_artifacts(tmp_path: Path):
    assert requires_platform_wheel(tmp_path / "runtime") is False


def test_requires_platform_wheel_true_with_sidecar_binary(tmp_path: Path):
    # The compiled agent-core sidecar is a native artifact (sibling of the Telys runtime dir), so a
    # wheel that ships it must be platform-specific even with no Telys kernel present.
    package_root = tmp_path / "codna"
    sidecar_dir = package_root / "_agent_core_runtime"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "codna-sidecar").write_bytes(b"\x7fELF native binary")

    assert has_native_sidecar(sidecar_dir) is True
    assert requires_platform_wheel(package_root / "_telys_runtime") is True


def test_requires_platform_wheel_true_with_manifest(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "manifest.json").write_text("{}", encoding="utf-8")

    manifest, kernels = staged_runtime_files(runtime_dir)

    assert manifest == runtime_dir / "manifest.json"
    assert kernels == ()
    assert requires_platform_wheel(runtime_dir) is True


def test_requires_platform_wheel_true_with_kernel(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    kernel = runtime_dir / "libame_kernel.dylib"
    kernel.write_bytes(b"kernel")

    manifest, kernels = staged_runtime_files(runtime_dir)

    assert manifest is None
    assert kernels == (kernel,)
    assert requires_platform_wheel(runtime_dir) is True


def test_validate_wheel_filename_rejects_universal_native_wheel(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_wheel_filename("codna-0.1.0-py3-none-any.whl", runtime_dir)

    assert "platform-specific wheel tag" in str(excinfo.value)


def test_validate_wheel_filename_accepts_platform_native_wheel(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "manifest.json").write_text("{}", encoding="utf-8")

    validate_wheel_filename("codna-0.1.0-py3-none-macosx_14_0_arm64.whl", runtime_dir)
