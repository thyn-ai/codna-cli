"""codna's own device-code account login (codna.login) — the codna-cli cutover."""
from __future__ import annotations

import pytest

from codna import login


def test_onboard_uses_generic_product_route_and_returns_key(monkeypatch):
    calls: dict = {}

    def fake_request(method, url, *, bearer=None, body=None, timeout=30.0):
        calls.update(method=method, url=url, bearer=bearer, body=body)
        return 201, {"raw_key": "ak_live_xyz"}

    monkeypatch.setattr(login, "_request", fake_request)
    key = login.onboard(api="https://api.codna.ai", access_token="jwt")
    assert key == "ak_live_xyz"
    assert calls["url"].endswith("/v1/products/codna/onboard")  # generic product-onboard, not /v1/telys/onboard
    assert calls["body"]["product"] == "codna"
    assert calls["bearer"] == "jwt"


def test_onboard_raises_loginerror_on_http_error(monkeypatch):
    monkeypatch.setattr(login, "_request", lambda *a, **k: (403, {"error": {"message": "forbidden"}}))
    with pytest.raises(login.LoginError):
        login.onboard(api="https://api.codna.ai", access_token="jwt")


def test_login_with_token_skips_browser_and_stores_codna_api_key(monkeypatch):
    # A supplied token skips device_authorize; onboard mints a key; it is stored as CODNA_API_KEY.
    monkeypatch.setattr(login, "onboard", lambda *, api, access_token: "ak_live_stored")
    stored: dict = {}
    from codna import keystore
    monkeypatch.setattr(keystore, "set_key", lambda name, val: stored.__setitem__(name, val))

    result = login.login(access_token="jwt", open_browser=False)
    assert result["ok"] is True
    assert stored["CODNA_API_KEY"] == "ak_live_stored"   # control-plane key in the OS keychain
    assert result["api_key_stored"] == "keychain"
    # the result carries NO key material / prefix — only the storage location (CodeQL: clear-text logging)
    assert not any("ak_" in str(v) for v in result.values())


def test_client_id_matches_registry_codna_cli():
    # Must equal the algenta product-registry accounts_client_id for device_login to resolve to codna.
    assert login._CLIENT_ID == "codna-cli"
