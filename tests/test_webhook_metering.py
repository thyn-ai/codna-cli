"""Per-installation engine-credential resolution + the changeable managed allowance."""
from __future__ import annotations

import json

import pytest

from codna import webhook_metering as m


@pytest.fixture(autouse=True)
def _no_live_bridge_by_default(monkeypatch):
    """The static-store tests below exercise the self-hosted fallback path in isolation — make
    sure ambient env vars never accidentally route them through the live bridge instead."""
    monkeypatch.delenv(m._ENGINE_URL_ENV, raising=False)
    monkeypatch.delenv(m._ENGINE_INTERNAL_SECRET_ENV, raising=False)


def test_resolve_engine_key_from_inline_env_json(monkeypatch):
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", raising=False)
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "org-42-key", "99": "org-99-key"}))
    assert m.resolve_engine_key(42) == "org-42-key"
    assert m.resolve_engine_key(99) == "org-99-key"
    assert m.resolve_engine_key(7) is None       # unknown install -> fail closed
    assert m.resolve_engine_key(None) is None


def test_resolve_engine_key_from_file_takes_precedence(tmp_path, monkeypatch):
    store = tmp_path / "install_keys.json"
    store.write_text(json.dumps({"42": "from-file"}), encoding="utf-8")
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", str(store))
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "from-env"}))
    assert m.resolve_engine_key(42) == "from-file"   # file store wins over inline env


def test_resolve_engine_key_is_none_with_no_store(monkeypatch):
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", raising=False)
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS", raising=False)
    assert m.resolve_engine_key(42) is None          # nothing provisioned -> fail closed
    # malformed JSON also fails closed, never raises
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", "{not json")
    assert m.resolve_engine_key(42) is None


# ── the live account-linking bridge (decision-engine) ────────────────────────────────────────
class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def test_resolve_engine_key_live_unconfigured_falls_through_to_static_store(monkeypatch):
    """No engine URL / secret set -> the live bridge is skipped entirely, static store still tried."""
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", raising=False)
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "org-42-key"}))
    assert m.resolve_engine_key(42) == "org-42-key"


def test_resolve_engine_key_live_success_takes_precedence_over_static_store(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "stale-static-key"}))
    calls = []

    def _fake_get(url, *, params, headers, timeout):
        calls.append((url, params, headers))
        return _FakeResponse(200, {"engine_key": "fresh-live-key", "org_id": "org-1"})

    monkeypatch.setattr("httpx.get", _fake_get)
    assert m.resolve_engine_key(42) == "fresh-live-key"
    assert calls[0][0] == "https://api.algenta.internal/internal/codna-webhook/engine-key"
    assert calls[0][1] == {"installation_id": 42}
    assert calls[0][2]["Authorization"] == "Bearer shh"


def test_resolve_engine_key_live_not_linked_falls_through_to_static_store(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "static-fallback-key"}))
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(404, {"detail": "not linked"}))
    assert m.resolve_engine_key(42) == "static-fallback-key"


def _bridge_configured_without_static_store(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", raising=False)
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS", raising=False)


def test_resolve_engine_key_live_network_error_is_no_answer_not_unlinked(monkeypatch):
    """REGRESSION: a connection error used to resolve to None == "not linked" and the job posted
    the link-your-account prompt to a linked org. No answer is a distinct outcome the worker retries."""
    _bridge_configured_without_static_store(monkeypatch)

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.get", _boom)
    with pytest.raises(m.BridgeUnavailable, match="OSError"):
        m.resolve_engine_key(42)


@pytest.mark.parametrize("status", [500, 502, 503, 504, 429])
def test_resolve_engine_key_live_transient_http_status_is_no_answer(monkeypatch, status):
    """502/503 is exactly what the control plane serves for a minute or two on every redeploy
    (the 2026-09-18 01:44Z incident: a review in that window got the link prompt instead)."""
    _bridge_configured_without_static_store(monkeypatch)
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(status, {"detail": "upstream"}))
    with pytest.raises(m.BridgeUnavailable, match=f"HTTP {status}"):
        m.resolve_engine_key(42)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_resolve_engine_key_live_definite_non_200_still_fails_closed(monkeypatch, status):
    """A misconfigured secret or an unlinked installation IS an answer: fail closed as before."""
    _bridge_configured_without_static_store(monkeypatch)
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(status, {"detail": "no"}))
    assert m.resolve_engine_key(42) is None


def test_resolve_engine_key_static_store_still_answers_while_the_bridge_is_down(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.delenv("CODNA_WEBHOOK_INSTALL_KEYS_PATH", raising=False)
    monkeypatch.setenv("CODNA_WEBHOOK_INSTALL_KEYS", json.dumps({"42": "self-hosted-key"}))
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(503, {"detail": "upstream"}))
    assert m.resolve_engine_key(42) == "self-hosted-key"


def test_byok_and_kill_switch_lookups_keep_their_fail_open_defaults_while_the_bridge_is_down(monkeypatch):
    """Only the credential gate retries; these two are consulted after it succeeded and keep the
    behavior they always had on a hiccup (no BYOK key -> managed allowance; fixes stay enabled)."""
    _bridge_configured_without_static_store(monkeypatch)
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(503, {"detail": "upstream"}))
    assert m.resolve_provider_credentials(42) == (None, None)
    assert m.resolve_fix_enabled(42) is True


# ── the org's own BYOK provider key (Anthropic / OpenAI / Google) ────────────────────────────
def test_resolve_provider_credentials_surfaces_configured_backend(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(
            200, {"engine_key": "fresh-live-key", "provider": "openai", "provider_key": "sk-org-key"}
        ),
    )
    assert m.resolve_provider_credentials(42) == ("openai", "sk-org-key")


def test_resolve_provider_credentials_none_when_org_has_no_byok(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: _FakeResponse(200, {"engine_key": "fresh-live-key", "provider": None, "provider_key": None}),
    )
    assert m.resolve_provider_credentials(42) == (None, None)


def test_resolve_provider_credentials_none_when_unconfigured_or_unlinked(monkeypatch):
    assert m.resolve_provider_credentials(None) == (None, None)
    monkeypatch.delenv(m._ENGINE_URL_ENV, raising=False)  # bridge unconfigured entirely
    assert m.resolve_provider_credentials(42) == (None, None)
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResponse(404, {"detail": "not linked"}))
    assert m.resolve_provider_credentials(42) == (None, None)  # not linked -> no provider either


def test_resolve_provider_credentials_never_raises_on_network_error(monkeypatch):
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    monkeypatch.setattr("httpx.get", lambda *a, **k: (_ for _ in ()).throw(OSError("connection refused")))
    assert m.resolve_provider_credentials(42) == (None, None)


def test_resolve_engine_key_and_resolve_provider_credentials_never_share_stale_state(monkeypatch):
    """Regression: an earlier bad design cached the last live response across calls, so a SECOND
    job for the same installation_id (a very common real pattern) could silently reuse the FIRST
    job's response instead of asking again. Prove two sequential calls each hit the network."""
    monkeypatch.setenv(m._ENGINE_URL_ENV, "https://api.algenta.internal")
    monkeypatch.setenv(m._ENGINE_INTERNAL_SECRET_ENV, "shh")
    calls = []

    def _fake_get(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse(200, {"engine_key": "key-1", "provider": "openai", "provider_key": "k1"})
        return _FakeResponse(200, {"engine_key": "key-2", "provider": "anthropic", "provider_key": "k2"})

    monkeypatch.setattr("httpx.get", _fake_get)
    assert m.resolve_engine_key(42) == "key-1"
    assert m.resolve_provider_credentials(42) == ("anthropic", "k2")  # its OWN fresh call, not #1's
    assert len(calls) == 2


def test_included_allowance_is_env_configurable(monkeypatch):
    monkeypatch.delenv("CODNA_INCLUDED_MONTHLY_USD", raising=False)
    assert m.included_monthly_usd() == "5"           # default
    monkeypatch.setenv("CODNA_INCLUDED_MONTHLY_USD", "20")
    assert m.included_monthly_usd() == "20"          # changeable without a code change
    assert "$20" in m.unlinked_summary()
    assert "accounts.thyn.ai/login?app=codna" in m.unlinked_summary()