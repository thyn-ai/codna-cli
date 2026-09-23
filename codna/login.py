"""`codna login` — codna's own CLI device-login (RFC 8628), consistent with `telys`/`algenta login`.

Signs this machine into a **codna** account through the shared accounts portal (OAuth 2.0 device
authorization grant), then self-serve onboards the free tier and mints the first codna API key via the
generic product-onboard route (`POST /v1/products/codna/onboard`). The key is stored in the OS keychain
as ``CODNA_API_KEY`` (codna.keystore) for control-plane calls.

Unlike telys/algenta, codna's runtime license is **HS256** and its Telys memory runtime is already
bundled in the wheel (+ the embedded OEM umbrella license), so login does NOT register an RS256
per-device license or download a runtime — it is purely codna account auth + the API key. BYOK provider
keys stay local and are handled separately (`codna key`).

Hosts are env-overridable (``CODNA_ACCOUNTS_URL`` / ``CODNA_API_URL``) for staging or a local mock; the
defaults are the shared accounts portal and codna's public API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_CLIENT_ID = "codna-cli"                 # must match the product registry accounts_client_id (device_login=True)
_SCOPE = "codna"
_DEFAULT_ACCOUNTS_URL = "https://accounts.thyn.ai"   # shared identity portal (Supabase OAuth / device-code)
_DEFAULT_API_URL = "https://api.codna.ai"            # codna public API (product registry public_api_base)
_MAX_RESP_BYTES = 1_000_000


class LoginError(RuntimeError):
    """A step of the codna device-login / onboarding flow failed (surfaced with remediation)."""


def accounts_url() -> str:
    return (os.environ.get("CODNA_ACCOUNTS_URL") or _DEFAULT_ACCOUNTS_URL).rstrip("/")


def api_url() -> str:
    return (os.environ.get("CODNA_API_URL") or _DEFAULT_API_URL).rstrip("/")


def _request(method: str, url: str, *, bearer: str | None = None, body: dict | None = None,
             timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 (https default)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(_MAX_RESP_BYTES + 1)
            status = resp.status
    except urllib.error.HTTPError as e:  # 4xx/5xx — the device flow signals errors here
        raw = b""
        try:
            raw = e.read(_MAX_RESP_BYTES + 1)
        except Exception:  # noqa: BLE001
            pass
        status = e.code
    except urllib.error.URLError as e:
        raise LoginError(f"{method} {url}: {e.reason}") from e
    if len(raw) > _MAX_RESP_BYTES:
        raise LoginError(f"{url}: response too large")
    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        payload = {}
    return status, (payload if isinstance(payload, dict) else {})


# ── OAuth 2.0 device authorization grant (RFC 8628) — mirrors telys/algenta ──────────────────────────────
def device_authorize(*, accounts: str, open_browser: bool = True, max_wait: float = 300.0) -> str:
    _, start = _request("POST", f"{accounts}/oauth/device/code",
                        body={"client_id": _CLIENT_ID, "scope": _SCOPE})
    device_code = start.get("device_code")
    user_code = start.get("user_code")
    verify = start.get("verification_uri_complete") or start.get("verification_uri")
    interval = float(start.get("interval") or 5)
    if not (device_code and user_code and verify):
        raise LoginError("accounts device-code response missing fields (is codna-cli enabled on accounts?)")
    print(f"\n  Open {verify}\n  and enter code:  {user_code}\n")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(verify)
        except Exception:  # noqa: BLE001 — headless boxes have no browser; the code is printed above
            pass
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        time.sleep(interval)
        status, tok = _request("POST", f"{accounts}/oauth/device/token",
                               body={"client_id": _CLIENT_ID, "device_code": device_code,
                                     "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
        access = tok.get("access_token")
        if access:
            return access
        error = tok.get("error")
        if error in ("authorization_pending", "slow_down") or status in (400, 428):
            continue
        raise LoginError(f"device authorization failed: {error or f'HTTP {status}'}")
    raise LoginError("timed out waiting for device authorization")


# ── control-plane: generic product-onboard (POST /v1/products/codna/onboard) ─────────────────────────────
def onboard(*, api: str, access_token: str) -> str:
    """Self-serve onboard: resolve-or-create the org on codna's free tier + mint the first API key.

    Uses the Supabase-JWT product-onboard route (POST /v1/api-keys does NOT accept a Supabase JWT — it
    needs an existing key — which is why onboarding has its own route). Returns the raw API key.
    """
    status, resp = _request("POST", f"{api}/v1/products/codna/onboard", bearer=access_token,
                            body={"name": "codna-cli", "product": "codna"})
    key = resp.get("raw_key") or resp.get("key") or resp.get("api_key")
    if status >= 400 or not key:
        detail = (resp.get("error") or {}).get("message") if isinstance(resp, dict) else None
        raise LoginError(f"codna onboarding failed (HTTP {status})" + (f": {detail}" if detail else ""))
    return key


# ── orchestration: the whole `codna login` in one call ───────────────────────────────────────────────────
def login(*, access_token: str | None = None, open_browser: bool = True) -> dict:
    """Device-login → onboard → store CODNA_API_KEY. Returns a status dict (never echoes the key)."""
    accounts, api = accounts_url(), api_url()
    token = access_token or os.environ.get("CODNA_TOKEN")
    if token:
        print("using supplied token (headless)")
    else:
        print(f"authenticating via {accounts} …")
        token = device_authorize(accounts=accounts, open_browser=open_browser)

    print("creating your Codna API key …")
    api_key = onboard(api=api, access_token=token)

    # Store the control-plane key in the OS keychain (codna.keystore manages CODNA_API_KEY). Never printed.
    from . import keystore

    try:
        keystore.set_key("CODNA_API_KEY", api_key)
        stored = "keychain"
    except keystore.KeystoreError:
        # No keychain (e.g. CODNA_DISABLE_KEYCHAIN / headless) — surface the env var so the user can persist it.
        os.environ["CODNA_API_KEY"] = api_key
        stored = "env (this process) — set CODNA_API_KEY to persist; no OS keychain available"

    # Return ONLY where the key was stored (a literal location) — never the key or any prefix derived from
    # it, so no credential-tainted value can reach a logging sink (CodeQL: clear-text logging of secrets).
    return {"ok": True, "api_key_stored": stored}
