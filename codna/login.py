"""`codna login` — codna's own CLI device-login (RFC 8628), consistent with `telys`/`algenta login`.

Signs this machine into a **codna** account through the shared accounts portal (OAuth 2.0 device
authorization grant), then self-serve onboards the free tier and mints the first codna API key via the
generic product-onboard route (`POST /v1/products/codna/onboard`). The key is stored in the OS keychain
as ``CODNA_API_KEY`` (codna.keystore) for control-plane calls.

The runtime-provisioning half of `codna login` is deliberately NOT in :func:`login`: that stays
purely codna account auth + the API key. :func:`run` below (the `codna login` command body) chains
it with ``telys_onboarding`` (per-device RS256 license + signed runtime install), reusing this
flow's Supabase access token so the user authorizes ONCE — the uniform fleet model
(`telys login` / `algenta login` / `sqai login`). BYOK provider keys stay local and are handled
separately (`codna key`).

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
    """Device-login → onboard → store CODNA_API_KEY. Returns a status dict (never echoes the key).

    The dict also carries the device flow's Supabase ``access_token``: ``cmd_login`` reuses it for
    the same-command runtime provisioning (telys_onboarding) so the user authorizes ONCE. It is a
    credential — callers must never print, log, or persist it beyond this process.
    """
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
    # it — plus the device flow's access token for the provisioning step. No credential-tainted value may
    # reach a logging sink (CodeQL: clear-text logging of sensitive information).
    return {"ok": True, "api_key_stored": stored, "access_token": token}


# ── `codna login` command body (kept here so cli.py stays under the module-size ceiling) ───────────
def run(args, *, runtime_keys) -> int:
    """One-time device authorization that ALSO installs the on-device runtime — one command.

    Uniform with `telys login` / `algenta login` / `sqai login`: device-code auth via the accounts
    portal → self-serve free-tier onboard → the first Codna API key (stored in the OS keychain as
    CODNA_API_KEY) → a local LLM provider key check so `codna fix` runs on-device → the per-device
    Telys license + signed memory runtime (telys_onboarding), reusing the SAME device-flow access
    token so the user authorizes once. After this one command every `codna mcp` tool — including
    `codna_recall` — works, fully offline thereafter.

    Idempotent: an already fully provisioned device skips re-provisioning (no network). A partial
    failure (Codna key stored, runtime fetch failed) is completed by re-running `codna login`.
    Exit codes: 0 = signed in AND provisioned (or already provisioned); 1 = sign-in failed;
    2 = signed in but runtime provisioning incomplete (needs network on first run — re-run).

    ``runtime_keys`` is the CLI's key-resolution callable (passed in, never imported, so this
    module stays free of cli.py).
    """
    import sys

    try:
        # The return is bound for exactly one field — the device flow's access token, handed to the
        # provisioning step below so the user authorizes ONCE. It is a credential and never reaches
        # output: the status JSON at the end is built from literals + non-secret fields only (the
        # account key itself is stored by login() and never returned). CodeQL: clear-text logging
        # of sensitive information.
        login_result = login(
            access_token=getattr(args, "token", None),
            open_browser=not getattr(args, "no_browser", False),
        )
    except LoginError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    # After sign-in, make sure a local LLM provider key exists so `codna fix` can run on-device. The
    # provider key stays on the machine (OS keychain) — never sent to the cloud (distinct from the account
    # API key just minted). See `codna key`.
    from . import byok_cli
    provider_status = byok_cli.ensure_local_provider_key(
        interactive=sys.stdin.isatty() and sys.stdout.isatty(),
        runtime_keys=runtime_keys(include_keychain=False),
    )

    # The fleet-uniform half: provision the per-device license + signed on-device runtime so
    # `codna_recall` works after ONE command (not only triage/secure). Already provisioned → an
    # offline no-op. Offline first run → a clear, actionable error (never a traceback); re-running
    # resumes provisioning without corrupting state.
    from . import memory
    from . import telys_onboarding

    try:
        provisioning = telys_onboarding.ensure_provisioned(
            access_token=login_result.get("access_token"),
            open_browser=not getattr(args, "no_browser", False),
        )
    except telys_onboarding.OnboardingError as exc:
        print(json.dumps({
            "ok": False,
            "error": str(exc),
            "hint": ("You are signed in and your Codna key is stored, but the on-device runtime is "
                     "not fully provisioned — that step needs network access on this first run. "
                     "Check your connection and re-run `codna login` (safe to re-run: it finishes "
                     "provisioning without corrupting state). Everything runs offline after that."),
        }, indent=2), file=sys.stderr)
        return 2
    except memory.CodeMemoryError as exc:
        # A local runtime misconfig (e.g. TELYS_KERNEL pointing at a missing file, or a corrupt
        # packaged-runtime manifest): not a network problem — surface the actionable message the
        # memory layer already wrote, cleanly (never a traceback).
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    # login() raised on failure and provisioning returned, so reaching here means signed-in AND
    # provisioned. Report ONLY sign-in + BYOK + non-secret provisioning status — never anything on
    # the account-key dataflow (login() already stored the key in the keychain; no credential-tainted
    # value reaches this sink). CodeQL: clear-text logging of sensitive information.
    print(json.dumps({
        "ok": True,
        "provider_key": provider_status,
        "runtime": provisioning,
    }, indent=2))
    return 0
