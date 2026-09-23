"""Offline tests for the memory wrapper's friendly failure modes (no Telys installed/needed)."""
from __future__ import annotations

import os
import json
import sys
import types

import pytest

from codna import memory as memory_module
from codna.memory import CodeMemoryError, _require_telys


def _clear_kernel_env(monkeypatch):
    for name in (
        "TELYS_KERNEL",
        "AME_KERNEL",
        "CODNA_TELYS_KERNEL",
        "CODNA_TELYS_INSTALL_ROOT",
        "TELYS_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(memory_module, "_LAST_KERNEL_RESOLUTION", None)


def test_require_telys_missing_sdk_is_friendly(monkeypatch):
    # Force `from telys import ...` to fail regardless of whether the package is installed correctly.
    monkeypatch.setitem(sys.modules, "telys", None)
    monkeypatch.setitem(sys.modules, "telys.embedding", None)
    with pytest.raises(CodeMemoryError) as ei:
        _require_telys()
    message = str(ei.value)
    assert "Telys SDK shipped with base `codna`" in message
    assert "platform wheel" in message
    assert "codna[memory]" not in message


def test_require_telys_missing_embedding_runtime_is_friendly(monkeypatch):
    _clear_kernel_env(monkeypatch)
    fake_telys = types.ModuleType("telys")
    fake_telys.__path__ = []
    fake_telys.Telys = object
    fake_telys.scope_key = lambda *_args: "scope"
    monkeypatch.setitem(sys.modules, "telys", fake_telys)
    monkeypatch.setitem(sys.modules, "telys.embedding", None)

    with pytest.raises(CodeMemoryError) as ei:
        _require_telys()

    assert "Telys kernel could not be loaded" in str(ei.value)
    assert "Codna platform wheel" in str(ei.value)


def test_embedder_missing_runtime_import_is_friendly(monkeypatch):
    class MissingKernelEmbedder:
        def __init__(self):
            raise ImportError("No module named 'telys.embedding._kernel'")

    monkeypatch.setattr(memory_module, "_engine_embed_config", lambda: None)
    monkeypatch.setattr(memory_module, "_require_telys", lambda: (object, object, MissingKernelEmbedder))
    monkeypatch.setattr(memory_module, "_current_kernel_resolution", lambda: {"checked": []})

    with pytest.raises(CodeMemoryError) as ei:
        memory_module._embedder()  # noqa: SLF001

    assert "Telys kernel could not be loaded" in str(ei.value)
    assert "Codna platform wheel" in str(ei.value)


def test_code_memory_missing_runtime_is_friendly(monkeypatch, tmp_path):
    class RuntimeNotInstalled(Exception):
        pass

    class FakeTelys:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeNotInstalled("runtime missing")

    class FakeProfile:
        model_id = "test.embedder"
        dimension = 3

    class FakeEmbedder:
        profile = FakeProfile()

    monkeypatch.setattr(memory_module, "_require_telys", lambda: (FakeTelys, lambda *_args: "scope", object))
    monkeypatch.setattr(memory_module, "_load_telys_license_metadata", lambda: {"configured": False})
    monkeypatch.setattr(memory_module, "_current_kernel_resolution", lambda: {"found": False})
    monkeypatch.setattr(memory_module, "_embedder", lambda: FakeEmbedder())

    with pytest.raises(CodeMemoryError) as ei:
        memory_module.CodeMemory(str(tmp_path))

    message = str(ei.value)
    assert "Telys runtime is not installed" in message
    assert "Codna-packaged runtime artifact" in message
    assert "CODNA_TELYS_INSTALL_ROOT" in message


def test_code_memory_error_is_not_the_builtin():
    # Deliberately distinct from the Python builtin MemoryError (avoid shadowing / ambiguous excepts).
    assert CodeMemoryError is not MemoryError and issubclass(CodeMemoryError, Exception)


def test_kernel_resolution_accepts_explicit_env_path(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    kernel = tmp_path / memory_module._kernel_filename()  # noqa: SLF001
    kernel.write_bytes(b"kernel")
    monkeypatch.setenv("TELYS_KERNEL", str(kernel))

    result = memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert result == {
        "found": True,
        "source": "env:TELYS_KERNEL",
        "path": str(kernel.resolve()),
    }


def test_kernel_resolution_rejects_bad_explicit_env_without_fallback(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    install_root = tmp_path / "local-telys"
    valid_kernel = install_root / "kernel" / memory_module._kernel_filename()  # noqa: SLF001
    valid_kernel.parent.mkdir(parents=True)
    valid_kernel.write_bytes(b"kernel")
    monkeypatch.setenv("CODNA_TELYS_INSTALL_ROOT", str(install_root))
    monkeypatch.setenv("TELYS_KERNEL", str(tmp_path / "missing-kernel"))

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert "TELYS_KERNEL" in str(excinfo.value)
    assert str(valid_kernel) not in str(excinfo.value)


def test_kernel_resolution_uses_codna_install_root(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    package_root = tmp_path / "cli" / "codna"
    install_root = tmp_path / "local-telys"
    kernel = install_root / "kernel" / memory_module._kernel_filename()  # noqa: SLF001
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    monkeypatch.setattr(memory_module, "__file__", str(package_root / "memory.py"))
    monkeypatch.setenv("CODNA_TELYS_INSTALL_ROOT", str(install_root))

    result = memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert result["source"] == "env:CODNA_TELYS_INSTALL_ROOT"
    assert result["path"] == str(kernel.resolve())
    assert os.environ["TELYS_KERNEL"] == str(kernel.resolve())


def test_kernel_resolution_uses_packaged_runtime_candidate(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    package_root = tmp_path / "cli" / "codna"
    kernel = package_root / "_telys_runtime" / memory_module._kernel_filename()  # noqa: SLF001
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    manifest = {
        "schema_version": 1,
        "artifact": kernel.name,
        "sha256": memory_module._sha256_file(kernel),  # noqa: SLF001
        "size_bytes": kernel.stat().st_size,
        "platform": memory_module._runtime_platform_tag(),  # noqa: SLF001
        "source_basename": kernel.name,
    }
    (kernel.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(memory_module, "__file__", str(package_root / "memory.py"))

    result = memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert result["source"] == "codna:package-runtime"
    assert result["path"] == str(kernel.resolve())
    assert result["manifest_path"] == str((kernel.parent / "manifest.json").resolve())
    assert result["sha256"] == manifest["sha256"]
    assert os.environ["TELYS_KERNEL"] == str(kernel.resolve())


def test_kernel_resolution_rejects_packaged_runtime_without_manifest(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    package_root = tmp_path / "cli" / "codna"
    kernel = package_root / "_telys_runtime" / memory_module._kernel_filename()  # noqa: SLF001
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    fallback = tmp_path / "build" / "local-telys" / "kernel" / memory_module._kernel_filename()  # noqa: SLF001
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")
    monkeypatch.setattr(memory_module, "__file__", str(package_root / "memory.py"))

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert "manifest is missing" in str(excinfo.value)
    assert "TELYS_KERNEL" not in os.environ


def test_kernel_resolution_rejects_tampered_packaged_runtime_without_fallback(
    monkeypatch,
    tmp_path,
):
    _clear_kernel_env(monkeypatch)
    package_root = tmp_path / "cli" / "codna"
    kernel = package_root / "_telys_runtime" / memory_module._kernel_filename()  # noqa: SLF001
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    manifest = {
        "schema_version": 1,
        "artifact": kernel.name,
        "sha256": "0" * 64,
        "size_bytes": kernel.stat().st_size,
        "platform": memory_module._runtime_platform_tag(),  # noqa: SLF001
        "source_basename": kernel.name,
    }
    (kernel.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fallback = tmp_path / "build" / "local-telys" / "kernel" / memory_module._kernel_filename()  # noqa: SLF001
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")
    monkeypatch.setattr(memory_module, "__file__", str(package_root / "memory.py"))

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert "sha256 does not match" in str(excinfo.value)
    assert str(fallback) not in str(excinfo.value)


def test_kernel_resolution_uses_source_checkout_candidate(monkeypatch, tmp_path):
    _clear_kernel_env(monkeypatch)
    package_root = tmp_path / "cli" / "codna"
    kernel = tmp_path / "build" / "local-telys" / "kernel" / memory_module._kernel_filename()  # noqa: SLF001
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"kernel")
    monkeypatch.setattr(memory_module, "__file__", str(package_root / "memory.py"))

    result = memory_module._resolve_telys_kernel(configure_env=True)  # noqa: SLF001

    assert result["source"] == "codna:source-build"
    assert result["path"] == str(kernel.resolve())
    assert os.environ["TELYS_KERNEL"] == str(kernel.resolve())
