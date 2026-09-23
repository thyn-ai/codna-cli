from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "download_telys_runtime_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("download_telys_runtime_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clear_telys_envs(monkeypatch):
    for name in (
        "CODNA_TELYS_API_KEY",
        "CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN",
        "CODNA_TELYS_OEM_DOWNLOAD_TOKEN",
        "TELYS_OEM_DOWNLOAD_TOKEN",
        "CODNA_TELYS_LICENSE_JWT",
        "TELYS_LICENSE_JWT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_packages_host_requires_oem_download_token(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)

    with pytest.raises(tool.DownloadBundleError) as excinfo:
        tool.require_auth_for_telys_host(
            "https://packages.telys.ai/runtime/latest/macos-arm64.bundle",
            None,
            "CODNA_TELYS_OEM_DOWNLOAD_TOKEN",
        )

    msg = str(excinfo.value)
    assert "OEM Telys runtime download token" in msg
    assert "CODNA_TELYS_OEM_DOWNLOAD_TOKEN" in msg


def test_missing_token_error_names_oem_download_token_when_old_api_key_env_named(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)

    with pytest.raises(tool.DownloadBundleError) as excinfo:
        tool.resolve_bearer_token("CODNA_TELYS_API_KEY")
    assert "CODNA_TELYS_OEM_DOWNLOAD_TOKEN" in str(excinfo.value)


def test_oem_download_token_env_resolves_as_bearer(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)
    monkeypatch.setenv("CODNA_TELYS_OEM_DOWNLOAD_TOKEN", " telys-oem-download-token ")

    assert tool.resolve_bearer_token("CODNA_TELYS_OEM_DOWNLOAD_TOKEN") == "telys-oem-download-token"
    assert tool.resolve_bearer_token(None) == "telys-oem-download-token"


def test_telys_oem_download_token_is_fallback(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)
    monkeypatch.setenv("TELYS_OEM_DOWNLOAD_TOKEN", "fallback-download-token")

    assert tool.resolve_bearer_token("CODNA_TELYS_OEM_DOWNLOAD_TOKEN") == "fallback-download-token"


def test_legacy_runtime_download_token_is_rejected(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)

    with pytest.raises(tool.DownloadBundleError) as excinfo:
        tool.resolve_bearer_token("CODNA_TELYS_RUNTIME_DOWNLOAD_TOKEN")
    assert "not a valid Codna release credential" in str(excinfo.value)


def test_custom_https_bundle_url_does_not_require_token(monkeypatch):
    tool = _load_script()
    _clear_telys_envs(monkeypatch)

    tool.require_auth_for_telys_host(
        "https://example.com/codna/telys/macos-arm64.bundle",
        None,
        "CODNA_TELYS_LICENSE_JWT",
    )


def test_rejects_non_https_bundle_url():
    tool = _load_script()

    with pytest.raises(tool.DownloadBundleError) as excinfo:
        tool.validate_url("http://packages.telys.ai/runtime/latest/macos-arm64.bundle")

    assert "must use https" in str(excinfo.value)


def test_build_request_adds_bearer_header_without_logging_token():
    tool = _load_script()
    request = tool.build_request("https://packages.telys.ai/runtime/latest/macos-arm64.bundle", "license.jwt")

    assert request.get_header("Authorization") == "Bearer license.jwt"
    assert request.get_header("User-agent") == "CodnaRelease/0.1 (+https://github.com/thyn-ai/codna)"
    assert request.get_method() == "GET"
