from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage_codna_telys_runtime.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage_codna_telys_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_key


def _sign(private_key: bytes, payload: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    key = serialization.load_pem_private_key(private_key, password=None)
    return key.sign(payload, padding.PKCS1v15(), hashes.SHA256())


def _bundle(
    path: Path,
    *,
    private_key: bytes,
    artifact_name: str,
    artifact: bytes,
    platform: str,
    extra_artifacts: dict[str, bytes] | None = None,
) -> None:
    artifacts = {artifact_name: artifact, **(extra_artifacts or {})}
    manifest = {
        "format_version": 1,
        "platform": platform,
        "artifacts": [
            {
                "name": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in artifacts.items()
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = _sign(private_key, manifest_bytes)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (
            ("telys_manifest.json", manifest_bytes),
            ("telys_manifest.sig", signature),
            *artifacts.items(),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _wheel(files: dict[str, bytes]) -> bytes:
    """Minimal wheel (zip) bytes for the given ``{member_name: content}`` mapping."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


_MEMENGINE_WHEEL_NAME = "telys_runtime_native-0.1.3-py3-none-any.whl"


def _memengine_wheel() -> bytes:
    return _wheel(
        {
            "memengine/__init__.py": b"__version__ = '0.1.3'\n",
            "memengine/runtime.py": b"def build(*args, texts=None):\n    return None\n",
            "telys_runtime_native-0.1.3.dist-info/METADATA": b"Name: telys-runtime-native\n",
        }
    )


def test_stage_from_signed_telys_bundle(tmp_path: Path):
    tool = _load_script()
    private_key, public_key = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    wheel_payload = _memengine_wheel()
    _bundle(
        bundle,
        private_key=private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
        extra_artifacts={"libKGENCompilerRTShared.so": b"runtime-lib", _MEMENGINE_WHEEL_NAME: wheel_payload},
    )

    result = tool.stage_from_bundle(
        bundle,
        tmp_path / "runtime",
        target_platform="linux-x86_64",
        release_pubkey=public_key_path,
    )

    assert result["source"] == "telys-bundle"
    assert result["codna_manifest"]["artifact"] == "libame_kernel.so"
    assert result["codna_manifest"]["platform"] == "linux-x86_64"
    assert result["codna_manifest"]["auxiliary_artifacts"][0]["name"] == "libKGENCompilerRTShared.so"
    assert (tmp_path / "runtime" / "libame_kernel.so").read_bytes() == b"kernel"
    assert (tmp_path / "runtime" / "libKGENCompilerRTShared.so").read_bytes() == b"runtime-lib"
    # The wheel itself is never laid down — only its memengine package is staged.
    assert not (tmp_path / "runtime" / _MEMENGINE_WHEEL_NAME).exists()
    assert (tmp_path / "runtime" / "memengine" / "__init__.py").read_bytes() == b"__version__ = '0.1.3'\n"
    assert b"texts=None" in (tmp_path / "runtime" / "memengine" / "runtime.py").read_bytes()
    assert not (tmp_path / "runtime" / "telys_runtime_native-0.1.3.dist-info").exists()
    # Provenance of the staged engine rides in the codna manifest (sha256 of the signed wheel).
    provenance = result["codna_manifest"]["memengine"]
    assert provenance["wheel"] == _MEMENGINE_WHEEL_NAME
    assert provenance["sha256"] == hashlib.sha256(wheel_payload).hexdigest()
    assert provenance["size_bytes"] == len(wheel_payload)
    assert provenance["file_count"] == 2
    assert result["memengine"]["file_count"] == 2
    # The amended manifest is what landed on disk.
    on_disk = json.loads((tmp_path / "runtime" / "manifest.json").read_text())
    assert on_disk["memengine"]["sha256"] == provenance["sha256"]


def test_stage_from_bundle_without_wheel_stages_no_memengine(tmp_path: Path):
    # Legacy bundles (pre-option-B) carry no wheel — staging must still succeed.
    tool = _load_script()
    private_key, public_key = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    _bundle(
        bundle,
        private_key=private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
    )

    result = tool.stage_from_bundle(
        bundle,
        tmp_path / "runtime",
        target_platform="linux-x86_64",
        release_pubkey=public_key_path,
    )

    assert result["memengine"] is None
    assert "memengine" not in result["codna_manifest"]
    assert not (tmp_path / "runtime" / "memengine").exists()


def test_stage_from_bundle_rejects_wheel_without_memengine(tmp_path: Path):
    tool = _load_script()
    private_key, public_key = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    _bundle(
        bundle,
        private_key=private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
        extra_artifacts={_MEMENGINE_WHEEL_NAME: _wheel({"other/__init__.py": b""})},
    )

    with pytest.raises(tool.StageRuntimeError) as excinfo:
        tool.stage_from_bundle(
            bundle,
            tmp_path / "runtime",
            target_platform="linux-x86_64",
            release_pubkey=public_key_path,
        )

    assert "does not contain a memengine package" in str(excinfo.value)


def test_stage_from_bundle_rejects_invalid_wheel_zip(tmp_path: Path):
    tool = _load_script()
    private_key, public_key = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    _bundle(
        bundle,
        private_key=private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
        extra_artifacts={_MEMENGINE_WHEEL_NAME: b"not-a-zip"},
    )

    with pytest.raises(tool.StageRuntimeError) as excinfo:
        tool.stage_from_bundle(
            bundle,
            tmp_path / "runtime",
            target_platform="linux-x86_64",
            release_pubkey=public_key_path,
        )

    assert "not a valid zip" in str(excinfo.value)


def test_stage_from_bundle_rejects_multiple_wheels(tmp_path: Path):
    tool = _load_script()
    private_key, public_key = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    _bundle(
        bundle,
        private_key=private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
        extra_artifacts={
            _MEMENGINE_WHEEL_NAME: _memengine_wheel(),
            "telys_runtime_native-0.1.2-py3-none-any.whl": _memengine_wheel(),
        },
    )

    with pytest.raises(tool.StageRuntimeError) as excinfo:
        tool.stage_from_bundle(
            bundle,
            tmp_path / "runtime",
            target_platform="linux-x86_64",
            release_pubkey=public_key_path,
        )

    assert "at most one .whl artifact" in str(excinfo.value)


def test_stage_from_bundle_rejects_bad_signature(tmp_path: Path):
    tool = _load_script()
    private_key, public_key = _keypair()
    other_private_key, _ = _keypair()
    public_key_path = tmp_path / "release_pub.pem"
    public_key_path.write_bytes(public_key)
    bundle = tmp_path / "runtime.bundle"
    _bundle(
        bundle,
        private_key=other_private_key,
        artifact_name="libame_kernel.so",
        artifact=b"kernel",
        platform="linux-x86_64",
    )

    with pytest.raises(tool.StageRuntimeError) as excinfo:
        tool.stage_from_bundle(
            bundle,
            tmp_path / "runtime",
            target_platform="linux-x86_64",
            release_pubkey=public_key_path,
        )

    assert "signature does not verify" in str(excinfo.value)
