from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from codna import memory as memory_module
from codna import telys_onboarding as onb
from codna.telys_onboarding import OnboardingError


class _FakeLoginError(RuntimeError):
    pass


def _install_fake_telys_login(monkeypatch, *, login_impl):
    fake_login_mod = SimpleNamespace(login=login_impl, LoginError=_FakeLoginError)
    # `from telys import login` resolves telys.login off the telys package object
    monkeypatch.setitem(sys.modules, "telys", SimpleNamespace(login=fake_login_mod))
    monkeypatch.setitem(sys.modules, "telys.login", fake_login_mod)
    return fake_login_mod


# ---- host config (#3) --------------------------------------------------------------------------

def test_apply_telys_hosts_maps_codna_overrides():
    env = {
        "CODNA_TELYS_ACCOUNTS_URL": "https://acc.example",
        "CODNA_TELYS_API_URL": "https://api.example",
        "CODNA_TELYS_PACKAGES_URL": "https://pkg.example",
        "TELYS_API_URL": "https://stale.example",  # codna override must win
    }
    applied = onb.apply_telys_hosts(env)
    assert env["TELYS_ACCOUNTS_URL"] == "https://acc.example"
    assert env["TELYS_API_URL"] == "https://api.example"
    assert env["TELYS_PACKAGES_URL"] == "https://pkg.example"
    assert applied == {
        "TELYS_ACCOUNTS_URL": "https://acc.example",
        "TELYS_API_URL": "https://api.example",
        "TELYS_PACKAGES_URL": "https://pkg.example",
    }


def test_apply_telys_hosts_noop_when_unset():
    env = {"TELYS_API_URL": "https://telys-default.example"}
    applied = onb.apply_telys_hosts(env)
    assert applied == {}
    assert env["TELYS_API_URL"] == "https://telys-default.example"  # untouched


def test_login_license_path_respects_telys_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))
    assert onb.login_license_path() == tmp_path / "login_license.jwt"


# ---- run_login (#1) ----------------------------------------------------------------------------

def test_run_login_calls_telys_login_and_wires_license(monkeypatch, tmp_path):
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))
    monkeypatch.delenv("CODNA_TELYS_LICENSE_PATH", raising=False)
    monkeypatch.setenv("CODNA_TELYS_API_URL", "https://api.example")

    calls = {}

    def fake_login(*, plan, access_token, install, open_browser):
        calls.update(plan=plan, access_token=access_token, install=install, open_browser=open_browser)
        # telys.login writes the device license here
        (tmp_path / "login_license.jwt").write_text("device.jwt", encoding="utf-8")
        return {"device_id": "abcdef123456", "installed": True, "tier": "team", "api_key_prefix": "ak_…"}

    _install_fake_telys_login(monkeypatch, login_impl=fake_login)

    result = onb.run_login(token="tok", plan="telys_developer", install=True, open_browser=False)

    assert calls == {"plan": "telys_developer", "access_token": "tok", "install": True, "open_browser": False}
    assert result["tier"] == "team"
    # hosts were applied and license discovered
    assert __import__("os").environ["TELYS_API_URL"] == "https://api.example"
    assert __import__("os").environ["CODNA_TELYS_LICENSE_PATH"] == str(tmp_path / "login_license.jwt")


def test_run_login_wraps_login_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))

    def boom(*, plan, access_token, install, open_browser):
        raise _FakeLoginError("device_flow authorization_pending timed out")

    _install_fake_telys_login(monkeypatch, login_impl=boom)

    with pytest.raises(OnboardingError) as excinfo:
        onb.run_login(open_browser=False)
    assert "device authorization failed" in str(excinfo.value).lower()


def test_run_login_without_telys_raises_onboarding_error(monkeypatch):
    # make `from telys import login` fail
    monkeypatch.setitem(sys.modules, "telys", None)
    with pytest.raises(OnboardingError) as excinfo:
        onb.run_login(open_browser=False)
    message = str(excinfo.value)
    assert "reinstall or upgrade `codna`" in message
    assert "codna[memory]" not in message


# ---- ensure_licensed hook ----------------------------------------------------------------------

def test_ensure_licensed_returns_status_when_provisioned(monkeypatch):
    monkeypatch.setattr(memory_module, "_configured_license_token", lambda: ("device.jwt", "codna:onboarding-license"))
    monkeypatch.setattr(memory_module, "_resolve_telys_kernel", lambda *, configure_env: {"found": True, "source": "telys:installed-runtime"})
    status = onb.ensure_licensed()
    assert status["provisioned"] is True
    assert status["license_source"] == "codna:onboarding-license"


def test_ensure_licensed_non_interactive_raises_actionable(monkeypatch):
    monkeypatch.setattr(memory_module, "_configured_license_token", lambda: (None, None))
    monkeypatch.setattr(memory_module, "_resolve_telys_kernel", lambda *, configure_env: {"found": False})
    with pytest.raises(memory_module.CodeMemoryError) as excinfo:
        onb.ensure_licensed(allow_interactive_login=False)
    msg = str(excinfo.value)
    assert "codna login" in msg
    assert "one-time device authorization" in msg


# ---- memory: per-device onboarding license discovery + precedence ------------------------------

def _clear_license_env(monkeypatch):
    for name in (
        "CODNA_TELYS_LICENSE_JWT", "TELYS_LICENSE_JWT",
        "CODNA_TELYS_LICENSE_PATH", "TELYS_LICENSE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_onboarding_license_discovered_from_telys_home(monkeypatch, tmp_path):
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))
    (tmp_path / "login_license.jwt").write_text("device.jwt\n", encoding="utf-8")
    # ensure the embedded OEM license is absent so we isolate the onboarding source
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: tmp_path / "nope.jwt")

    token, source = memory_module._configured_license_token()
    assert token == "device.jwt"
    assert source == "codna:onboarding-license"


def test_onboarding_license_wins_over_embedded_oem(monkeypatch, tmp_path):
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))
    (tmp_path / "login_license.jwt").write_text("device.jwt", encoding="utf-8")
    bundled = tmp_path / "oem.jwt"
    bundled.write_text("oem-umbrella.jwt", encoding="utf-8")
    monkeypatch.setattr(memory_module, "_package_license_path", lambda: bundled)

    token, source = memory_module._configured_license_token()
    assert source == "codna:onboarding-license"  # per-device wins over the OEM umbrella
    assert token == "device.jwt"


def test_env_license_wins_over_onboarding(monkeypatch, tmp_path):
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TELYS_HOME", str(tmp_path))
    (tmp_path / "login_license.jwt").write_text("device.jwt", encoding="utf-8")
    monkeypatch.setenv("CODNA_TELYS_LICENSE_JWT", "explicit.jwt")

    token, source = memory_module._configured_license_token()
    assert token == "explicit.jwt"
    assert source == "env:CODNA_TELYS_LICENSE_JWT"


def test_kernel_and_runtime_hints_point_at_codna_login():
    assert "codna login" in memory_module._runtime_install_hint()
    monkeypatch_free_hint = memory_module._kernel_hint()
    assert "codna login" in monkeypatch_free_hint


# ---- cli `codna login` -------------------------------------------------------------------------

def test_cmd_login_prints_ok_and_no_raw_key(monkeypatch, capsys):
    # `codna login` is codna's own device-code account login (codna.login) PLUS runtime provisioning
    # (telys_onboarding.ensure_provisioned) — both seams faked here.
    from codna import cli
    from codna import keystore

    def fake_login(*, access_token, open_browser):
        # returns where the account key was stored — never key material to the caller/output.
        return {"ok": True, "api_key_prefix": "ak_…", "api_key_stored": "keychain",
                "access_token": "supa-jwt"}

    def forbidden_keychain_read():
        raise AssertionError("codna login status reporting must not read OS keychain secrets")

    monkeypatch.setattr("codna.login.login", fake_login)
    monkeypatch.setattr(keystore, "config_values", forbidden_keychain_read)
    monkeypatch.setattr("codna.byok_cli.ensure_local_provider_key", lambda **kwargs: "present")
    monkeypatch.setattr(
        "codna.telys_onboarding.ensure_provisioned",
        lambda **kwargs: {"provisioned": True, "already_provisioned": True},
    )

    args = SimpleNamespace(token=None, no_browser=True)
    rc = cli.cmd_login(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert '"ok": true' in out
    assert "provider_key" in out   # BYOK status; the account-key dataflow is never printed
    assert "runtime" in out        # provisioning status is reported (non-secret fields only)
    # never echo anything key-derived (CodeQL: clear-text logging of sensitive info)
    assert "ak_" not in out
    assert "api_key" not in out
    assert "supa-jwt" not in out   # the device-flow access token is never printed either


def test_cmd_login_reports_login_error(monkeypatch, capsys):
    from codna import cli
    from codna.login import LoginError

    def fake_login(**kwargs):
        raise LoginError("device authorization failed: invalid_client")

    def provisioning_must_not_run(**kwargs):
        raise AssertionError("sign-in failed — runtime provisioning must not be attempted")

    monkeypatch.setattr("codna.login.login", fake_login)
    monkeypatch.setattr("codna.telys_onboarding.ensure_provisioned", provisioning_must_not_run)
    args = SimpleNamespace(token=None, no_browser=True)
    rc = cli.cmd_login(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert '"ok": false' in err
    assert "invalid_client" in err
