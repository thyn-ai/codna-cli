"""Per-installation metering for Codna's GitHub App webhook.

Codna's webhook is keyless: it runs the packaged Codna CLI with the encapsulated local
runtime and injects only the linked org's Codna run credential (plus, when the org has
configured one, their own BYOK provider key — see :func:`resolve_provider_credentials`). The
monthly managed-model allowance (the changeable "$5/mo"), Stripe plan tier, and device seats
are enforced by the Codna account plane for that org.

This module resolves that per-installation Codna credential and carries the (changeable)
included-allowance config. It FAILS CLOSED: with no credential for an installation, the
webhook must NOT run a fix on a shared house key — it prompts the user to link/authorize.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_INCLUDED_MONTHLY_USD_ENV = "CODNA_INCLUDED_MONTHLY_USD"
_ACCOUNT_LINK_URL_ENV = "CODNA_ACCOUNT_LINK_URL"
# The account-linking bridge: decision-engine (accounts.thyn.ai) is the one place that actually
# knows an installation's org/entitlement, so it's asked live, per call -- no static store to
# keep in sync, no volume, no "did the sync job run" failure mode. Both unset (the thyn-ai
# hosted default is to have them set) falls through to the static store below, which stays for
# self-hosted operators who provision it directly rather than pointing at a decision-engine.
_ENGINE_URL_ENV = "CODNA_ENGINE_URL"
_ENGINE_INTERNAL_SECRET_ENV = "CODNA_WEBHOOK_INTERNAL_SECRET"
# How long the QUEUE holds a job back before it can be claimed again after the bridge gave no
# answer (webhook_queue.complete(retry_after_s=...)); no worker thread sleeps through it. The
# control plane is unreachable for ~1-2 min on each of its redeploys; two holds plus the retries
# themselves cover that window.
BRIDGE_RETRY_WAIT_S = 30.0


class BridgeUnavailable(RuntimeError):
    """The account-linking bridge is configured but could not answer right now.

    Distinct from "not linked" (a definite 404) and from a misconfigured secret (401/403): those
    ARE answers about the installation. A connection error, a timeout, a 429 or any 5xx is not --
    the control plane answers 502/503 for a minute or two on every redeploy, and on 2026-09-18
    01:44Z a review that landed in that window was reported to a linked org as "link your
    account" (a neutral Check Run in place of the required review). The worker retries instead,
    and on its last attempt posts :func:`unreachable_summary` rather than the link prompt.
    """


def included_monthly_usd() -> str:
    """The changeable managed-model allowance shown to users (the "$5/mo").

    Read at call time from ``CODNA_INCLUDED_MONTHLY_USD`` (default ``5``) so it can be changed
    without a code change. Authoritative enforcement is engine-side per org (shared billing
    platform); this value only drives the webhook's link/upgrade message.
    """
    return os.environ.get(_INCLUDED_MONTHLY_USD_ENV, "5")


def account_link_url() -> str:
    """Where users link their GitHub install to their Codna account (+ manage plan/BYOK)."""
    return os.environ.get(_ACCOUNT_LINK_URL_ENV, "https://accounts.thyn.ai/login?app=codna")


def resolve_engine_key(installation_id: int | None) -> str | None:
    """Return the org Codna run credential for a GitHub installation, or None (fail closed).

    Tries the live account-linking bridge first (decision-engine mints a fresh, short-lived
    credential per call — see :func:`_fetch_live_credentials`); falls back to a static store
    for self-hosted operators who provision one directly instead of pointing at a
    decision-engine (:func:`_resolve_engine_key_from_static_store`).
    """
    if not installation_id:
        return None
    try:
        live = _fetch_live_credentials(installation_id, raise_unavailable=True)
    except BridgeUnavailable:
        # A self-hosted static store still answers while the bridge is down; without one there
        # is no answer, and "no answer" must not become "not linked" (see BridgeUnavailable).
        static = _resolve_engine_key_from_static_store(installation_id)
        if static:
            return static
        raise
    if live is not None:
        key = live.get("engine_key")
        if key:
            return str(key)
    return _resolve_engine_key_from_static_store(installation_id)


def resolve_provider_credentials(installation_id: int | None) -> tuple[str | None, str | None]:
    """Return ``(provider, provider_key)`` for the org's own BYOK key (whichever of Anthropic /
    OpenAI / Google they configured), or ``(None, None)`` if none is set / the installation isn't
    linked / this is a self-hosted static-store install (that path carries no provider concept).

    The local agent-core sidecar always looks for ONE specific provider's raw key -- without
    this, every fix silently falls back to whatever the sidecar defaults to (Anthropic) even for
    an org that configured OpenAI or Google, and fails outright for an org with no local key at
    all. A second, independent bridge call from :func:`resolve_engine_key`'s (each mints its own
    fresh short-lived credential) -- deliberately not cached: a stale cross-job reuse of another
    call's response is a correctness bug, not an optimization, since the same installation_id
    recurs across many separate jobs over a webhook's lifetime.
    """
    if not installation_id:
        return None, None
    live = _fetch_live_credentials(installation_id)
    if live is None:
        return None, None
    provider = live.get("provider")
    provider_key = live.get("provider_key")
    if not provider or not provider_key:
        return None, None
    return str(provider), str(provider_key)


def resolve_fix_enabled(installation_id: int | None) -> bool:
    """Return whether the linked org wants automatic fixes to run at all.

    This is a CONVENIENCE kill switch (an admin's "pause Codna" toggle), not a security
    boundary -- unlike :func:`resolve_engine_key`, which correctly fails closed because it
    gates spend/credentials. This one FAILS OPEN: no installation, no live bridge response,
    or no explicit ``fix_enabled`` value on the response all default to True, so a bridge
    hiccup or a missing/older field never silently disables automatic fixes for every org.
    Only an explicit ``fix_enabled: false`` from the live bridge turns fixes off.
    """
    if not installation_id:
        return True
    live = _fetch_live_credentials(installation_id)
    if live is None:
        return True
    value = live.get("fix_enabled")
    if value is None:
        return True
    return bool(value)


def _fetch_live_credentials(installation_id: int, *, raise_unavailable: bool = False) -> dict[str, Any] | None:
    """Ask decision-engine, live, whether this installation is linked to a metered Codna
    account. Unconfigured (no engine URL / no shared secret) and definite non-200 answers (401
    misconfigured secret, 404 not linked) fall through to None. A transient failure -- connection
    error, timeout, 429, any 5xx -- also returns None by default, or raises
    :class:`BridgeUnavailable` when ``raise_unavailable`` is set (the credential gate wants to
    retry; the BYOK and kill-switch lookups keep their fail-open defaults). Always a fresh call,
    never cached (see :func:`resolve_provider_credentials`)."""
    base_url = os.environ.get(_ENGINE_URL_ENV)
    secret = os.environ.get(_ENGINE_INTERNAL_SECRET_ENV)
    if not base_url or not secret:
        return None
    try:
        import httpx
    except Exception:  # noqa: BLE001 — httpx missing is a config problem, not a fatal one here
        return None
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/internal/codna-webhook/engine-key",
            params={"installation_id": installation_id},
            headers={"Authorization": f"Bearer {secret}"},
            timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001 — a network hiccup must fail closed, not crash the job
        if raise_unavailable:
            raise BridgeUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return None
    if resp.status_code == 429 or resp.status_code >= 500:
        if raise_unavailable:
            raise BridgeUnavailable(f"HTTP {resp.status_code}") from None
        return None
    if resp.status_code != 200:
        return None
    try:
        parsed = resp.json()
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_engine_key_from_static_store(installation_id: int) -> str | None:
    """Self-hosted escape hatch: a manually-provisioned ``installation_id -> credential`` JSON
    file on a volume (``CODNA_WEBHOOK_INSTALL_KEYS_PATH``) or inline (``CODNA_WEBHOOK_INSTALL_KEYS``),
    for operators who meter directly rather than pointing at a decision-engine."""
    raw: str | None = None
    path = os.environ.get("CODNA_WEBHOOK_INSTALL_KEYS_PATH")
    if path and Path(path).is_file():
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError:
            raw = None
    if raw is None:
        raw = os.environ.get("CODNA_WEBHOOK_INSTALL_KEYS")
    if not raw:
        return None
    try:
        store = json.loads(raw)
    except ValueError:
        return None
    value = store.get(str(installation_id)) if isinstance(store, dict) else None
    return str(value) if value else None


def unlinked_summary() -> str:
    """The Check Run message when an installation has no linked, metered account."""
    return (
        f"Codna includes a managed-model allowance (~${included_monthly_usd()}/mo) of verified "
        f"fixes. Link this GitHub install to your Codna account to enable cloud fixes: "
        f"{account_link_url()}  (or add your own model key there for uncapped, self-billed usage). "
        f"The Codna CLI runs on your machine and never needs this."
    )


def unreachable_summary(kind: str, detail: str) -> str:
    """The Check Run / comment text when the bridge gave no answer on the job's last attempt."""
    return (
        "Codna could not reach its account service to check this installation's plan "
        f"({detail}), so nothing ran and nothing was spent. Push again or comment "
        f"`@codna {kind}` to retry."
    )
