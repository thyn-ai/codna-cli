from __future__ import annotations

from pathlib import Path

MANIFEST_NAME = "manifest.json"
KERNEL_GLOB = "libame_kernel.*"
SIDECAR_RUNTIME_DIRNAME = "_agent_core_runtime"
SIDECAR_BINARY_NAMES = ("codna-sidecar", "codna-sidecar.exe")


def package_runtime_dir() -> Path:
    return Path(__file__).resolve().parent / "_telys_runtime"


def sidecar_runtime_dir() -> Path:
    return Path(__file__).resolve().parent / SIDECAR_RUNTIME_DIRNAME


def has_native_sidecar(sidecar_dir: Path | None = None) -> bool:
    directory = sidecar_dir if sidecar_dir is not None else sidecar_runtime_dir()
    return directory.is_dir() and any((directory / name).is_file() for name in SIDECAR_BINARY_NAMES)


def staged_runtime_files(runtime_dir: Path) -> tuple[Path | None, tuple[Path, ...]]:
    manifest = runtime_dir / MANIFEST_NAME
    kernels = tuple(sorted(runtime_dir.glob(KERNEL_GLOB))) if runtime_dir.is_dir() else ()
    return (manifest if manifest.exists() else None), kernels


def requires_platform_wheel(runtime_dir: Path) -> bool:
    # A wheel that ships ANY native artifact — the Telys kernel OR the compiled agent-core sidecar —
    # must be platform-specific, never py3-none-any. The sidecar dir is the Telys dir's sibling, so
    # this stays a pure function of runtime_dir's location.
    manifest, kernels = staged_runtime_files(runtime_dir)
    sidecar_dir = runtime_dir.parent / SIDECAR_RUNTIME_DIRNAME
    return manifest is not None or bool(kernels) or has_native_sidecar(sidecar_dir)


def validate_wheel_filename(wheel_name: str, runtime_dir: Path) -> None:
    if requires_platform_wheel(runtime_dir) and wheel_name.endswith("-py3-none-any.whl"):
        raise ValueError(
            "native package-runtime artifacts (Telys kernel or agent-core sidecar) "
            "require a platform-specific wheel tag"
        )
