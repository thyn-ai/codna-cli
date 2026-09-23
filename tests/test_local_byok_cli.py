from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from codna import byok_cli, keystore


@pytest.fixture
def mem_keystore(monkeypatch):
    """Back keystore with an in-process dict so cli wiring is exercised without a real OS keychain."""
    store: dict[str, str] = {}
    monkeypatch.setattr(keystore, "set_key", lambda name, value: (store.__setitem__(keystore.normalize_key_name(name), value.strip()) or keystore.normalize_key_name(name)))
    monkeypatch.setattr(keystore, "get_key", lambda name: store.get(keystore.normalize_key_name(name)))
    monkeypatch.setattr(keystore, "stored_key_names", lambda: sorted(store))
    monkeypatch.setattr(keystore, "config_values", lambda: {
        n: keystore.ConfigValue(key=n, value=v, source="keychain") for n, v in store.items()
    })
    monkeypatch.setattr(keystore, "delete_key", lambda name: bool(store.pop(keystore.normalize_key_name(name), None)))
    monkeypatch.setattr(keystore, "available", lambda: True)
    return store


def _rk():
    """The runtime key map as cli._runtime_keys would produce it (keychain-sourced here)."""
    return keystore.config_values()


def _clear_provider_env(monkeypatch):
    for name in keystore.MANAGED_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_cmd_key_set_reads_stdin_and_stores(mem_keystore, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.readline", lambda: "sk-ant-xyz\n")
    rc = byok_cli.cmd_key(SimpleNamespace(key_action="set", provider="anthropic", stdin=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert '"stored": "ANTHROPIC_API_KEY"' in out
    assert "sk-ant-xyz" not in out  # value never printed
    assert mem_keystore["ANTHROPIC_API_KEY"] == "sk-ant-xyz"


def test_cmd_key_list_shows_names_not_values(mem_keystore, capsys):
    mem_keystore["ANTHROPIC_API_KEY"] = "sk-ant-1"
    rc = byok_cli.cmd_key(SimpleNamespace(key_action="list"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "ANTHROPIC_API_KEY" in out
    assert "sk-ant-1" not in out


def test_cmd_key_list_with_disabled_keychain_is_safe(monkeypatch, capsys):
    monkeypatch.setenv("CODNA_DISABLE_KEYCHAIN", "1")

    rc = byok_cli.cmd_key(SimpleNamespace(key_action="list"))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload == {"ok": True, "keys": [], "keychain": False}


def test_cmd_key_rm(mem_keystore, capsys):
    mem_keystore["OPENAI_API_KEY"] = "sk-oai"
    rc = byok_cli.cmd_key(SimpleNamespace(key_action="rm", provider="openai"))
    assert rc == 0
    assert '"removed": true' in capsys.readouterr().out
    assert "OPENAI_API_KEY" not in mem_keystore


def test_cmd_key_set_bad_provider_errors(mem_keystore, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.readline", lambda: "whatever\n")
    rc = byok_cli.cmd_key(SimpleNamespace(key_action="set", provider="nonsense", stdin=True))
    err = capsys.readouterr().err
    assert rc == 1
    assert '"ok": false' in err


def test_provider_key_present_detects_env(mem_keystore, monkeypatch):
    _clear_provider_env(monkeypatch)
    assert byok_cli.provider_key_present(_rk()) is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert byok_cli.provider_key_present(_rk()) is True


def test_provider_key_present_detects_keychain(mem_keystore, monkeypatch):
    _clear_provider_env(monkeypatch)
    assert byok_cli.provider_key_present(_rk()) is False
    mem_keystore["ANTHROPIC_API_KEY"] = "sk-kc"
    assert byok_cli.provider_key_present(_rk()) is True


def test_engine_key_alone_is_not_a_provider_key(mem_keystore, monkeypatch):
    _clear_provider_env(monkeypatch)
    mem_keystore["CODNA_API_KEY"] = "engine-key"  # engine auth, NOT an LLM model key
    assert byok_cli.provider_key_present(_rk()) is False


def test_empty_valued_key_is_not_treated_as_present(mem_keystore, monkeypatch):
    """A value-less keys.txt entry (ANTHROPIC_API_KEY=) is dropped by the engine, so it must NOT
    count as present (else onboarding skips the prompt and `codna fix` has no usable key)."""
    _clear_provider_env(monkeypatch)
    runtime_keys = {"ANTHROPIC_API_KEY": keystore.ConfigValue(key="ANTHROPIC_API_KEY", value="", source="keys_txt")}
    assert byok_cli.provider_key_present(runtime_keys) is False
    assert "codna key set" in byok_cli.ensure_local_provider_key(interactive=False, runtime_keys=runtime_keys)
    # whitespace-only env var also does not count
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert byok_cli.provider_key_present({}) is False


def test_ensure_local_provider_key_non_interactive_hint(mem_keystore, monkeypatch):
    _clear_provider_env(monkeypatch)
    status = byok_cli.ensure_local_provider_key(interactive=False, runtime_keys=_rk())
    assert "codna key set" in status


def test_ensure_local_provider_key_present(mem_keystore, monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert byok_cli.ensure_local_provider_key(interactive=False, runtime_keys=_rk()) == "present"
