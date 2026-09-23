"""Validated Codna-packaged Telys runtime artifact discovery.

This module is intentionally only about local files: kernel naming, package
runtime manifests, and native dependency validation. It has no Telys imports and
does not mutate process environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

PACKAGE_RUNTIME_DIRNAME = "_telys_runtime"
PACKAGE_RUNTIME_MANIFEST = "manifest.json"
PACKAGE_RUNTIME_SCHEMA_VERSION = 1
NATIVE_LIBRARY_SUFFIXES = (".dylib", ".so", ".dll")


class TelysRuntimeArtifactError(Exception):
    """Raised when a packaged Telys runtime artifact is incomplete or invalid."""


def kernel_filename() -> str:
    if sys.platform == "darwin":
        return "libame_kernel.dylib"
    if sys.platform.startswith("win"):
        return "libame_kernel.dll"
    return "libame_kernel.so"


def source_checkout_root(package_file: str | Path) -> Path:
    return Path(package_file).resolve().parents[2]


def runtime_platform_tag() -> str:
    machine = platform.machine().lower() or "unknown"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    if sys.platform.startswith("linux"):
        return f"linux-{machine}"
    if sys.platform.startswith("win"):
        return f"windows-{machine}"
    return f"{sys.platform}-{machine}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_native_runtime_library_name(name: str) -> bool:
    if "/" in name or "\\" in name or name in {".", ".."} or ".." in Path(name).parts:
        return False
    return name.startswith("lib") and name.endswith(NATIVE_LIBRARY_SUFFIXES)


def package_runtime_dir(package_file: str | Path) -> Path:
    return Path(package_file).resolve().parent / PACKAGE_RUNTIME_DIRNAME


def _fail(error_cls: type[Exception], message: str, cause: Exception | None = None) -> None:
    if cause is None:
        raise error_cls(message)
    raise error_cls(message) from cause


def load_package_runtime_manifest(
    path: Path,
    *,
    error_cls: type[Exception] = TelysRuntimeArtifactError,
) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        _fail(error_cls, f"Codna packaged Telys runtime manifest is missing: {path}", exc)
    except json.JSONDecodeError as exc:
        _fail(error_cls, f"Codna packaged Telys runtime manifest is invalid JSON: {path}", exc)
    if not isinstance(payload, dict):
        _fail(error_cls, f"Codna packaged Telys runtime manifest must be a JSON object: {path}")
    return payload


def validate_package_runtime_manifest(
    runtime_dir: Path,
    manifest: dict,
    *,
    error_cls: type[Exception] = TelysRuntimeArtifactError,
) -> dict:
    expected_artifact = kernel_filename()
    if manifest.get("schema_version") != PACKAGE_RUNTIME_SCHEMA_VERSION:
        _fail(error_cls, "Codna packaged Telys runtime manifest has unsupported schema_version")
    if manifest.get("artifact") != expected_artifact:
        _fail(
            error_cls,
            f"Codna packaged Telys runtime artifact must be {expected_artifact}; "
            f"manifest has {manifest.get('artifact')!r}",
        )
    if manifest.get("platform") != runtime_platform_tag():
        _fail(error_cls, f"Codna packaged Telys runtime platform mismatch: {manifest.get('platform')!r}")
    for key in ("sha256", "size_bytes", "source_basename"):
        if key not in manifest:
            _fail(error_cls, f"Codna packaged Telys runtime manifest is missing {key}")
    if "source_path" in manifest:
        _fail(error_cls, "Codna packaged Telys runtime manifest must not store source_path")

    artifact = runtime_dir / expected_artifact
    if not artifact.is_file():
        _fail(error_cls, f"Codna packaged Telys runtime kernel is missing: {artifact}")
    _validate_artifact_bytes(artifact, manifest, "kernel", error_cls=error_cls)
    auxiliary_names = _validate_auxiliary_artifacts(runtime_dir, manifest, error_cls=error_cls)
    return {
        "manifest_path": str((runtime_dir / PACKAGE_RUNTIME_MANIFEST).resolve()),
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
        "platform": manifest["platform"],
        "auxiliary_artifacts": auxiliary_names,
    }


def _validate_artifact_bytes(
    path: Path,
    manifest_entry: dict,
    label: str,
    *,
    error_cls: type[Exception],
) -> None:
    if path.stat().st_size != manifest_entry["size_bytes"]:
        _fail(error_cls, f"Codna packaged Telys runtime {label} size does not match manifest")
    if sha256_file(path) != manifest_entry["sha256"]:
        _fail(error_cls, f"Codna packaged Telys runtime {label} sha256 does not match manifest")


def _validate_auxiliary_artifacts(
    runtime_dir: Path,
    manifest: dict,
    *,
    error_cls: type[Exception],
) -> list[str]:
    auxiliary = manifest.get("auxiliary_artifacts", [])
    if not isinstance(auxiliary, list):
        _fail(error_cls, "Codna packaged Telys runtime auxiliary_artifacts must be a list")
    auxiliary_names: list[str] = []
    for entry in auxiliary:
        if not isinstance(entry, dict):
            _fail(error_cls, "Codna packaged Telys runtime auxiliary artifact entry must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not is_native_runtime_library_name(name):
            _fail(error_cls, f"Codna packaged Telys runtime auxiliary artifact has invalid name: {name!r}")
        for key in ("sha256", "size_bytes", "source_basename"):
            if key not in entry:
                _fail(error_cls, f"Codna packaged Telys runtime auxiliary artifact {name} is missing {key}")
        aux_path = runtime_dir / name
        if not aux_path.is_file():
            _fail(error_cls, f"Codna packaged Telys runtime auxiliary artifact is missing: {aux_path}")
        _validate_artifact_bytes(aux_path, entry, f"auxiliary artifact {name}", error_cls=error_cls)
        auxiliary_names.append(name)
    return auxiliary_names


def package_runtime_candidate(
    package_file: str | Path,
    *,
    error_cls: type[Exception] = TelysRuntimeArtifactError,
) -> tuple[Path, dict] | None:
    runtime_dir = package_runtime_dir(package_file)
    manifest_path = runtime_dir / PACKAGE_RUNTIME_MANIFEST
    runtime_files = sorted(runtime_dir.glob("lib*")) + sorted(runtime_dir.glob("*.dll")) if runtime_dir.is_dir() else []
    kernels = [path for path in runtime_files if path.name.startswith("libame_kernel.")]
    if not manifest_path.exists() and not kernels:
        return None
    expected = kernel_filename()
    unexpected = [path.name for path in kernels if path.name != expected]
    if unexpected:
        _fail(
            error_cls,
            "Codna packaged Telys runtime contains unsupported kernel artifact(s): "
            + ", ".join(unexpected),
        )
    manifest = load_package_runtime_manifest(manifest_path, error_cls=error_cls)
    metadata = validate_package_runtime_manifest(runtime_dir, manifest, error_cls=error_cls)
    return runtime_dir / expected, metadata


def candidate_codna_install_roots(package_file: str | Path, *, install_root_env_name: str) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    import os

    raw = os.environ.get(install_root_env_name)
    if raw and raw.strip():
        roots.append((f"env:{install_root_env_name}", Path(raw).expanduser()))
    roots.append(("codna:source-build", source_checkout_root(package_file) / "build" / "local-telys"))
    return roots
