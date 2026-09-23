"""`codna login` provisions the on-device runtime — one command, uniform with telys/algenta/sqai.

Covers the wiring (cmd_login → telys_onboarding.ensure_provisioned → run_login), idempotency
(already-provisioned devices skip the network entirely), the offline first-run failure message
(clear and actionable, never a traceback), partial-failure rerun (Codna key stored, runtime fetch
failed → a second `codna login` completes provisioning), and the end-to-end login → codna_recall
path — with the platform/control-plane mocked exactly like the existing onboarding tests
(fake `codna.login.login` + a fake `telys.login` module; no production calls).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from types import SimpleNamespace

import pytest

from codna import cli as cli_module
from codna import memory as memory_module
from codna import telys_onboarding as onb
from codna.mcp_server import recall_json


def _args(**overrides):
    base = {"token": None, "no_browser": True}
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def fresh_machine(monkeypatch, tmp_path):
    """A clean device: no Telys state, no license/kernel envs, no Codna key, no real keychain."""
    monkeypatch.setenv("TELYS_HOME", str(tmp_path / "telys-home"))
    for name in (
        "CODNA_TELYS_LICENSE_JWT", "TELYS_LICENSE_JWT",
        "CODNA_TELYS_LICENSE_PATH", "TELYS_LICENSE_PATH",
        "TELYS_KERNEL", "AME_KERNEL", "CODNA_TELYS_KERNEL",
        "CODNA_TELYS_INSTALL_ROOT", "CODNA_KEYS_FILE", "CODNA_TOKEN",
        "CODNA_API_KEY", "ALGENTA_API_KEY", "DE_API_KEY",
        "CODNA_TELYS_ACCOUNTS_URL", "CODNA_TELYS_API_URL", "CODNA_TELYS_PACKAGES_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    # The gate must see only what the test provisions — never the developer's real OS keychain.
    monkeypatch.setattr("codna.keystore.config_values", lambda: {})
    monkeypatch.setattr("codna.byok_cli.ensure_local_provider_key", lambda **kwargs: "present")
    return tmp_path


class _FakeLoginError(RuntimeError):
    pass


def _install_fake_telys_login(monkeypatch, tmp_path, *, install_root, calls, fail_times=0):
    """Fake the `telys.login` module (same pattern as test_telys_onboarding).

    The fake provisions like the real flow: writes the per-device license to $TELYS_HOME and lays
    the signed-runtime kernel under ``install_root`` (activated via CODNA_TELYS_INSTALL_ROOT — the
    CI mechanism the onboarding error message itself documents). ``fail_times`` simulates the
    control plane / packages host being unreachable for the first N attempts.

    Every env touch goes through monkeypatch so nothing leaks into later tests: the pre-set
    CODNA_TELYS_LICENSE_PATH also makes the real run_login's raw ``os.environ.setdefault`` a no-op
    (it points at the very license file the fake flow writes).
    """
    monkeypatch.setenv("CODNA_TELYS_LICENSE_PATH",
                       str(tmp_path / "telys-home" / "login_license.jwt"))
    state = {"attempts": 0}

    def fake_login(*, plan, access_token, install, open_browser):
        state["attempts"] += 1
        calls.append({"plan": plan, "access_token": access_token,
                      "install": install, "open_browser": open_browser})
        if state["attempts"] <= fail_times:
            raise RuntimeError(
                "could not reach install host https://packages.telys.ai/runtime/latest/"
                "macos-arm64.bundle: [Errno 8] nodename nor servname provided"
            )
        telys_home = tmp_path / "telys-home"
        telys_home.mkdir(parents=True, exist_ok=True)
        (telys_home / "login_license.jwt").write_text("device.jwt", encoding="utf-8")
        if install:
            kernel_dir = install_root / "kernel"
            kernel_dir.mkdir(parents=True, exist_ok=True)
            (kernel_dir / memory_module._kernel_filename()).write_bytes(b"fake-signed-kernel")
            monkeypatch.setenv("CODNA_TELYS_INSTALL_ROOT", str(install_root))
        return {"device_id": "d" * 32, "installed": bool(install),
                "tier": "telys_developer", "api_key_prefix": "ak_…"}

    fake_login_mod = SimpleNamespace(login=fake_login, LoginError=_FakeLoginError)
    monkeypatch.setitem(sys.modules, "telys", SimpleNamespace(login=fake_login_mod))
    monkeypatch.setitem(sys.modules, "telys.login", fake_login_mod)
    return state


def _fake_codna_login(monkeypatch, *, token="supa-jwt"):
    """Fake `codna.login.login`: stores the Codna key via the same env fallback the real no-keychain
    path uses, and returns the device flow's access token for the provisioning step."""
    def fake_login(*, access_token, open_browser):
        monkeypatch.setenv("CODNA_API_KEY", "codna-test-key")
        return {"ok": True, "api_key_stored": "env", "access_token": token}
    monkeypatch.setattr("codna.login.login", fake_login)


def _forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network is forbidden after provisioning — recall runs fully offline")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    import socket
    monkeypatch.setattr(socket, "create_connection", forbidden)


# ---- wiring: one command does sign-in AND provisioning --------------------------------------------

def test_cmd_login_provisions_runtime_with_the_same_access_token(monkeypatch, capsys, fresh_machine):
    install_root = fresh_machine / "install-root"
    calls: list[dict] = []
    _install_fake_telys_login(monkeypatch, fresh_machine, install_root=install_root, calls=calls)
    _fake_codna_login(monkeypatch)

    rc = cli_module.cmd_login(_args())

    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["runtime"]["provisioned"] is True
    assert payload["runtime"]["already_provisioned"] is False
    assert payload["runtime"]["runtime_installed"] is True
    # ONE authorization: the telys provisioning reused the codna device flow's access token.
    assert calls == [{"plan": "telys_developer", "access_token": "supa-jwt",
                      "install": True, "open_browser": False}]
    # The device really is provisioned: license + kernel resolve through the real memory seams.
    assert (fresh_machine / "telys-home" / "login_license.jwt").is_file()
    kernel = memory_module._resolve_telys_kernel(configure_env=False)
    assert kernel["found"] is True


# ---- idempotency: already provisioned → offline no-op ----------------------------------------------

def test_cmd_login_already_provisioned_is_an_offline_noop(monkeypatch, capsys, fresh_machine):
    telys_home = fresh_machine / "telys-home"
    telys_home.mkdir()
    (telys_home / "login_license.jwt").write_text("device.jwt", encoding="utf-8")
    kernel_dir = fresh_machine / "install-root" / "kernel"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / memory_module._kernel_filename()).write_bytes(b"fake-signed-kernel")
    monkeypatch.setenv("CODNA_TELYS_INSTALL_ROOT", str(fresh_machine / "install-root"))
    _fake_codna_login(monkeypatch)
    _forbid_network(monkeypatch)

    def telys_login_must_not_run(**kwargs):
        raise AssertionError("already provisioned — provisioning must not hit the network")

    fake_mod = SimpleNamespace(login=telys_login_must_not_run, LoginError=_FakeLoginError)
    monkeypatch.setitem(sys.modules, "telys", SimpleNamespace(login=fake_mod))
    monkeypatch.setitem(sys.modules, "telys.login", fake_mod)

    rc = cli_module.cmd_login(_args())

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["runtime"]["provisioned"] is True
    assert payload["runtime"]["already_provisioned"] is True


# ---- offline first run: clear, actionable failure — never a traceback ------------------------------

def test_cmd_login_offline_first_run_fails_actionably(monkeypatch, capsys, fresh_machine):
    calls: list[dict] = []
    _install_fake_telys_login(monkeypatch, fresh_machine,
                              install_root=fresh_machine / "install-root", calls=calls, fail_times=1)
    _fake_codna_login(monkeypatch)

    rc = cli_module.cmd_login(_args())

    captured = capsys.readouterr()
    assert rc == 2  # signed in, but provisioning incomplete — 0 is reserved for BOTH steps done
    assert "Traceback" not in captured.err
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "provisioning failed" in payload["error"]
    assert "could not reach install host" in payload["error"]
    assert "network" in payload["hint"]
    assert "re-run `codna login`" in payload["hint"]


# ---- partial failure: a second `codna login` completes provisioning ---------------------------------

def test_cmd_login_partial_failure_rerun_completes_provisioning(monkeypatch, capsys, fresh_machine):
    install_root = fresh_machine / "install-root"
    calls: list[dict] = []
    # First attempt: the packages host is unreachable mid-flow (Codna key already stored).
    _install_fake_telys_login(monkeypatch, fresh_machine, install_root=install_root,
                              calls=calls, fail_times=1)
    _fake_codna_login(monkeypatch)

    rc_first = cli_module.cmd_login(_args())
    assert rc_first == 2
    assert not (fresh_machine / "telys-home" / "login_license.jwt").exists()

    rc_second = cli_module.cmd_login(_args())
    payload = json.loads(capsys.readouterr().out)
    assert rc_second == 0
    assert payload["ok"] is True
    assert payload["runtime"]["provisioned"] is True
    # Both attempts ran the full flow (idempotent, nothing corrupted): the license landed exactly
    # once per successful attempt and resolves through the real memory seam.
    assert len(calls) == 2
    assert all(c["access_token"] == "supa-jwt" for c in calls)
    monkeypatch.delenv("CODNA_TELYS_LICENSE_PATH", raising=False)  # see the cross-process source
    token, source = memory_module._configured_license_token()
    assert (token, source) == ("device.jwt", "codna:onboarding-license")


# ---- ensure_provisioned unit semantics ---------------------------------------------------------------

def test_cmd_login_reports_local_runtime_misconfig_without_a_traceback(monkeypatch, capsys,
                                                                      fresh_machine):
    # A broken explicit kernel override (TELYS_KERNEL → missing file) is a LOCAL misconfig, not a
    # network problem: the memory layer's actionable CodeMemoryError is surfaced cleanly.
    _fake_codna_login(monkeypatch)
    monkeypatch.setenv("TELYS_KERNEL", str(fresh_machine / "no-such-kernel.dylib"))

    rc = cli_module.cmd_login(_args())

    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.err
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "TELYS_KERNEL" in payload["error"]
    assert "hint" not in payload  # the network hint would be wrong here


def test_ensure_provisioned_fast_path_never_calls_run_login(monkeypatch):
    monkeypatch.setattr(memory_module, "_configured_license_token",
                        lambda: ("device.jwt", "codna:onboarding-license"))
    monkeypatch.setattr(memory_module, "_resolve_telys_kernel",
                        lambda *, configure_env: {"found": True, "source": "telys:installed-runtime"})

    def run_login_must_not_run(**kwargs):
        raise AssertionError("already provisioned — run_login must not run")

    monkeypatch.setattr(onb, "run_login", run_login_must_not_run)
    status = onb.ensure_provisioned(access_token="tok")
    assert status == {"provisioned": True, "already_provisioned": True,
                      "license_source": "codna:onboarding-license",
                      "kernel_source": "telys:installed-runtime"}


def test_ensure_provisioned_skips_download_when_a_kernel_already_resolves(monkeypatch):
    # License missing but a kernel is already resolvable (e.g. a platform wheel): authorize the
    # device without re-downloading the runtime.
    monkeypatch.setattr(memory_module, "_configured_license_token", lambda: (None, None))
    monkeypatch.setattr(memory_module, "_resolve_telys_kernel",
                        lambda *, configure_env: {"found": True, "source": "codna:package-runtime"})
    captured: dict = {}
    monkeypatch.setattr(onb, "run_login",
                        lambda **kwargs: captured.update(kwargs) or {"installed": False,
                                                                     "tier": "telys_developer"})
    status = onb.ensure_provisioned(access_token="tok")
    assert captured == {"token": "tok", "plan": "telys_developer", "install": False,
                        "open_browser": None}
    assert status["provisioned"] is True
    assert status["already_provisioned"] is False


# ---- login(): the access token is returned for the provisioning step --------------------------------

def test_login_returns_the_device_flow_access_token(monkeypatch):
    from codna import keystore
    from codna import login as login_module

    monkeypatch.delenv("CODNA_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "device_authorize", lambda *, accounts, open_browser: "flow-jwt")
    monkeypatch.setattr(login_module, "onboard", lambda *, api, access_token: "the-key")
    monkeypatch.setattr(keystore, "set_key", lambda name, value: "keychain")

    result = login_module.login(open_browser=False)
    assert result["access_token"] == "flow-jwt"
    assert "the-key" not in json.dumps(result)  # the API key itself is never in the return


def test_login_echoes_a_supplied_headless_token(monkeypatch):
    from codna import keystore
    from codna import login as login_module

    monkeypatch.delenv("CODNA_TOKEN", raising=False)
    monkeypatch.setattr(login_module, "onboard", lambda *, api, access_token: "the-key")
    monkeypatch.setattr(keystore, "set_key", lambda name, value: "keychain")
    result = login_module.login(access_token="supplied-jwt", open_browser=False)
    assert result["access_token"] == "supplied-jwt"


# ---- end-to-end: codna login → codna_recall works, fully offline thereafter --------------------------

def test_login_then_recall_works_offline_end_to_end(monkeypatch, capsys, fresh_machine):
    # BEFORE login: the uniform gate (PR #626) blocks codna_recall on a clean machine.
    out = recall_json(repo=str(fresh_machine), query="anything")
    assert "requires the one-time free `codna login`" in out

    # ONE command: sign-in (fake control plane) + provisioning (fake telys device/register/install).
    install_root = fresh_machine / "install-root"
    calls: list[dict] = []
    _install_fake_telys_login(monkeypatch, fresh_machine, install_root=install_root, calls=calls)
    _fake_codna_login(monkeypatch)
    rc = cli_module.cmd_login(_args())
    assert rc == 0
    capsys.readouterr()

    # The provisioning is REAL on disk: the per-device license and the kernel resolve through the
    # actual memory seams (cross-process view: drop the in-process env hint run_login sets).
    monkeypatch.delenv("CODNA_TELYS_LICENSE_PATH", raising=False)
    token, source = memory_module._configured_license_token()
    assert (token, source) == ("device.jwt", "codna:onboarding-license")
    kernel = memory_module._resolve_telys_kernel(configure_env=False)
    assert kernel["found"] is True

    # Offline thereafter: hard-block all network, then recall must still work. The login-minted key
    # passes the gate with no validation call (offline by design); memory is faked at the same seam
    # the existing recall tests use (the native engine itself is out of scope in CI).
    _forbid_network(monkeypatch)

    class FakeMemory:
        def __init__(self, repo, db_path=None, *, service=None) -> None:
            self.repo = repo

        def is_empty(self) -> bool:
            return False

        def recall(self, query, *, service=None, language=None, final_k=8):
            return {"symbols": [{"id": "mem.py::open_pr_when_ci_fails", "score": 0.9}],
                    "explain": {"plan": "mock"}, "candidate_count": 1}

    monkeypatch.setattr(memory_module, "CodeMemory", FakeMemory)
    out = recall_json(repo=str(fresh_machine), query="open a pull request when CI fails")
    data = json.loads(out)
    assert data["symbols"][0]["id"] == "mem.py::open_pr_when_ci_fails"
    assert "error" not in out
