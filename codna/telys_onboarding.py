"""Codna per-device onboarding — drives Telys device-code auth (RFC 8628) as ONE application.

Codna users never obtain a Telys license or interact with Telys directly. ``codna login`` (and the
onboarding hint surfaced when memory is used unprovisioned) authorizes THIS device through Codna's
accounts portal, registers the device, receives a device-scoped offline license, and provisions the
protected on-device runtime when a packaged runtime is not already available. After that, Codna's local
memory runs fully offline. Free for one device; a device must always authorize through Codna.

Release builds use Codna's OEM Telys license to stage native runtime artifacts into Codna wheels. Users
do not publish, install, or license Telys as a separate product.

The heavy lifting lives in ``telys.login`` (shipped in the public ``telys`` SDK); Codna only points it
at the right hosts and provides a one-app UX. Every ``telys`` import is lazy so importing this module
(and ``codna`` core) never requires the optional ``codna[memory]`` extra.

Hosts are overridable via ``CODNA_TELYS_ACCOUNTS_URL`` / ``CODNA_TELYS_API_URL`` /
``CODNA_TELYS_PACKAGES_URL`` (mapped onto the ``TELYS_*_URL`` envs the SDK reads) — used to target a
staging control plane or a local mock. When unset, the Telys SDK defaults apply.
"""
from __future__ import annotations

import os
from pathlib import Path

# Codna-branded host overrides -> the TELYS_* envs telys.paths reads. Codna does not hardcode the
# production hostnames here (the control-plane/packages hosts are deploy-time config); when a codna
# override is unset the telys SDK's own defaults stand.
_HOST_ENV_MAP = {
    "CODNA_TELYS_ACCOUNTS_URL": "TELYS_ACCOUNTS_URL",
    "CODNA_TELYS_API_URL": "TELYS_API_URL",
    "CODNA_TELYS_PACKAGES_URL": "TELYS_PACKAGES_URL",
}

_LOGIN_LICENSE_FILENAME = "login_license.jwt"
_DEFAULT_PLAN = "telys_developer"


class OnboardingError(RuntimeError):
    """Codna device onboarding failed (surfaced to the user with remediation)."""


def _telys_home() -> Path:
    """Root of Telys on-disk state — mirrors ``telys.paths.telys_home()`` without importing telys."""
    return Path(os.path.abspath(os.environ.get("TELYS_HOME") or os.path.join(os.path.expanduser("~"), ".telys")))


def login_license_path() -> Path:
    """Absolute path where ``telys.login`` writes this device's offline license."""
    return _telys_home() / _LOGIN_LICENSE_FILENAME


def apply_telys_hosts(env: dict | None = None) -> dict:
    """Copy any ``CODNA_TELYS_*_URL`` overrides onto the ``TELYS_*_URL`` envs the SDK reads.

    Returns the mapping actually applied (telys env name -> value). Idempotent; a codna override always
    wins over a pre-set ``TELYS_*`` value so ``codna login`` and the downloader agree on hosts.
    """
    target = os.environ if env is None else env
    applied: dict[str, str] = {}
    for codna_key, telys_key in _HOST_ENV_MAP.items():
        value = (target.get(codna_key) or "").strip()
        if value:
            target[telys_key] = value
            applied[telys_key] = value
    return applied


def run_login(
    *,
    token: str | None = None,
    plan: str = _DEFAULT_PLAN,
    install: bool = True,
    open_browser: bool | None = None,
) -> dict:
    """Authorize this device via Telys device-code onboarding and provision the runtime.

    Thin wrapper over ``telys.login.login`` (never reimplements device-code). ``token`` supplies a
    headless access token for CI (falls back to ``$TELYS_TOKEN``). ``open_browser`` defaults to whether
    stdout is a TTY. After a successful login, points Codna's offline license resolver at the freshly
    written device license so memory works with no further configuration.
    """
    apply_telys_hosts()
    try:
        from telys import login as telys_login
    except ImportError as exc:  # broken or incomplete codna install
        raise OnboardingError(
            "Codna device onboarding needs the Telys SDK shipped with base `codna`; reinstall or "
            "upgrade `codna` from a platform wheel"
        ) from exc

    if open_browser is None:
        import sys

        open_browser = bool(getattr(sys.stdout, "isatty", lambda: False)())

    try:
        result = telys_login.login(
            plan=plan, access_token=token, install=install, open_browser=open_browser
        )
    except getattr(telys_login, "LoginError", Exception) as exc:
        raise OnboardingError(f"Codna device authorization failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface any onboarding failure with remediation
        # Non-LoginError failures past the auth step are provisioning failures (the signed-runtime
        # download/verify, e.g. telys.installer.InstallError "could not reach install host" when
        # offline) — name them accurately so `codna login` can point at the network, not at auth.
        raise OnboardingError(f"Codna runtime provisioning failed: {type(exc).__name__}: {exc}") from exc

    # Make the device license discoverable by codna.memory in this process (belt-and-suspenders; the
    # cross-process default is the codna:onboarding-license source in memory._configured_license_token).
    lic = login_license_path()
    if lic.is_file():
        os.environ.setdefault("CODNA_TELYS_LICENSE_PATH", str(lic))
    return result


def provision_status() -> dict:
    """Offline, non-raising precheck: does this device already have a Telys license + a resolvable
    kernel? Same two memory seams :func:`ensure_licensed` uses, but safe to call anywhere — no
    network, no device flow, no raise. ``codna login`` uses it to stay idempotent: an already
    fully provisioned device skips re-provisioning (and its download) entirely."""
    from codna import memory

    try:
        token, source = memory._configured_license_token()
    except memory.CodeMemoryError:
        token, source = None, None
    kernel = memory._resolve_telys_kernel(configure_env=False)
    return {
        "provisioned": bool(token and kernel.get("found")),
        "license_source": source if token else None,
        "kernel_source": kernel.get("source") if kernel.get("found") else None,
    }


def ensure_provisioned(*, access_token: str | None = None, plan: str = _DEFAULT_PLAN,
                       open_browser: bool | None = None) -> dict:
    """Guarantee this device has its per-device license AND the signed on-device runtime.

    This is the provisioning half of `codna login` — the uniform fleet model (`telys login` /
    `algenta login` / `sqai login` all authorize AND provision in one command). When the device is
    already fully provisioned this is a pure offline status read (no network). Otherwise it runs
    :func:`run_login`, reusing ``access_token`` from the codna device flow so the user authorizes
    ONCE, and downloads the runtime only when no kernel already resolves. Re-running after a
    partial failure is safe: the flow re-mints the device credentials and the runtime installer
    promotes atomically, so a second `codna login` completes provisioning without corrupting state.

    Returns a non-secret status dict (safe to print). Raises :class:`OnboardingError` with
    remediation when provisioning cannot complete (e.g. offline first run).
    """
    status = provision_status()
    if status["provisioned"]:
        status["already_provisioned"] = True
        return status
    result = run_login(
        token=access_token,
        plan=plan,
        install=status["kernel_source"] is None,
        open_browser=open_browser,
    )
    return {
        "provisioned": True,
        "already_provisioned": False,
        "runtime_installed": bool(result.get("installed")),
        "tier": result.get("tier"),
    }


def ensure_licensed(*, allow_interactive_login: bool = False) -> dict:
    """Precheck before using codna memory: confirm this device is provisioned, or steer the user.

    Returns a status dict when the device already has both an offline license and a resolvable kernel.
    Otherwise: if ``allow_interactive_login`` and attached to a TTY, runs :func:`run_login`; else raises
    an actionable :class:`~codna.memory.CodeMemoryError` pointing at ``codna login`` (never hangs a
    non-interactive/MCP caller waiting on a browser).
    """
    from codna import memory

    status = provision_status()
    if status["provisioned"]:
        return status

    if allow_interactive_login:
        import sys

        interactive = bool(
            getattr(sys.stdin, "isatty", lambda: False)()
            and getattr(sys.stdout, "isatty", lambda: False)()
        )
        if interactive:
            run_login(install=True)
            return {"provisioned": True, "onboarded": True}

    raise memory.CodeMemoryError(
        "Codna memory needs a one-time device authorization. Run `codna login` to authorize this "
        "device (free for one device), or for CI set CODNA_TELYS_LICENSE_JWT/CODNA_TELYS_LICENSE_PATH "
        "+ CODNA_TELYS_INSTALL_ROOT."
    )
