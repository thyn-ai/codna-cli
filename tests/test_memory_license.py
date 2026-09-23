from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from codna import memory as memory_module
from codna.memory import CodeMemoryError


def _install_fake_verifier(monkeypatch, *, claims=None, error: Exception | None = None) -> None:
    class FakeVerifier:
        @staticmethod
        def verify_license(token: str, *, now: int):
            assert token == "test-license-token"
            assert isinstance(now, int)
            if error is not None:
                raise error
            return claims or {
                "license_id": "oem-codna-2026",
                "customer": "codna",
                "org_id": "codna",
                "deployment_class": "oem",
                "products": {
                    "telys": {
                        "tier": "team",
                        "features": ["local_runtime", "partitioned_vector"],
                    }
                },
            }

    monkeypatch.setitem(sys.modules, "telys", SimpleNamespace(verify=FakeVerifier))


def test_license_env_is_verified_and_returns_safe_metadata(monkeypatch):
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", " test-license-token ")
    monkeypatch.setenv("CODNA_TELYS_LICENSE_REQUIRED", "1")
    _install_fake_verifier(monkeypatch)

    meta = memory_module._load_telys_license_metadata()

    assert meta == {
        "configured": True,
        "required": True,
        "source": "env:CODNA_TELYS_LICENSE_JWT",
        "license_id": "oem-codna-2026",
        "customer": "codna",
        "org_id": "codna",
        "deployment_class": "oem",
        "tier": "team",
        "features": ["local_runtime", "partitioned_vector"],
    }
    assert "test-license-token" not in repr(meta)


def test_license_path_is_supported(monkeypatch, tmp_path):
    path = tmp_path / "license.jwt"
    path.write_text("test-license-token\n", encoding="utf-8")
    monkeypatch.setenv("CODNA_TELYS_LICENSE_PATH", str(path))
    _install_fake_verifier(monkeypatch)

    meta = memory_module._load_telys_license_metadata()

    assert meta["configured"] is True
    assert meta["source"] == "path:CODNA_TELYS_LICENSE_PATH"


def _clear_license_env(monkeypatch, tmp_path) -> None:
    for name in (
        "CODNA_TELYS_LICENSE_JWT",
        "TELYS_LICENSE_JWT",
        "CODNA_TELYS_LICENSE_PATH",
        "TELYS_LICENSE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    # Also neutralize the per-device onboarding license ($TELYS_HOME/login_license.jwt): on a machine
    # that has run `codna login`/`telys login` it exists at the default ~/.telys path and would win over
    # the OEM package license these fall-through tests exercise. Point it at a guaranteed-absent path.
    monkeypatch.setattr(
        memory_module, "_onboarding_license_path", lambda: tmp_path / "__absent_onboarding_license__.jwt"
    )


def test_required_license_without_token_fails(monkeypatch, tmp_path):
    _clear_license_env(monkeypatch, tmp_path)
    # Isolate from any release-baked bundled license so the test is deterministic in every channel.
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: tmp_path / "absent.jwt")
    monkeypatch.setenv("CODNA_TELYS_LICENSE_REQUIRED", "true")

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._load_telys_license_metadata()

    assert "Telys license is required but not configured" in str(excinfo.value)


def test_bundled_oem_license_is_used_with_zero_config(monkeypatch, tmp_path):
    """A release wheel that bakes the OEM license activates base codna with no env/path set."""
    _clear_license_env(monkeypatch, tmp_path)
    monkeypatch.delenv("CODNA_TELYS_LICENSE_REQUIRED", raising=False)
    bundled = tmp_path / "oem-license.jwt"
    bundled.write_text("test-license-token\n", encoding="utf-8")
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: bundled)
    _install_fake_verifier(monkeypatch)

    token, source = memory_module._configured_license_token()
    assert token == "test-license-token"
    assert source == "codna:package-license"

    meta = memory_module._load_telys_license_metadata()
    assert meta["configured"] is True
    assert meta["source"] == "codna:package-license"
    assert meta["customer"] == "codna"


def test_env_license_overrides_bundled_oem_license(monkeypatch, tmp_path):
    """An explicit env token always wins over the release-baked default."""
    _clear_license_env(monkeypatch, tmp_path)
    bundled = tmp_path / "oem-license.jwt"
    bundled.write_text("different-bundled-token\n", encoding="utf-8")
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: bundled)
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", "test-license-token")
    _install_fake_verifier(monkeypatch)

    token, source = memory_module._configured_license_token()
    assert token == "test-license-token"
    assert source == "env:CODNA_TELYS_LICENSE_JWT"


def test_empty_bundled_oem_license_fails(monkeypatch, tmp_path):
    _clear_license_env(monkeypatch, tmp_path)
    bundled = tmp_path / "oem-license.jwt"
    bundled.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: bundled)

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._configured_license_token()
    assert "empty" in str(excinfo.value)


def test_invalid_license_fails_without_leaking_token(monkeypatch):
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", "test-license-token")
    _install_fake_verifier(monkeypatch, error=ValueError("bad test-license-token"))

    with pytest.raises(CodeMemoryError) as excinfo:
        memory_module._load_telys_license_metadata()

    msg = str(excinfo.value)
    assert "invalid: ValueError" in msg
    assert "test-license-token" not in msg
