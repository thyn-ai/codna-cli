from __future__ import annotations

import types

import pytest

from codna import keystore


class _FakeKeyring:
    """In-memory keyring backend for tests (service, username) -> secret."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str):
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self.store[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            from keyring.errors import PasswordDeleteError

            raise PasswordDeleteError("not found")
        del self.store[(service, username)]


@pytest.fixture
def kr(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    fake = _FakeKeyring()
    mod = types.SimpleNamespace(
        get_password=fake.get_password,
        set_password=fake.set_password,
        delete_password=fake.delete_password,
    )
    monkeypatch.setattr(keystore, "_keyring", lambda: mod)
    return fake


def test_normalize_provider_alias_and_envname():
    assert keystore.normalize_key_name("anthropic") == "ANTHROPIC_API_KEY"
    assert keystore.normalize_key_name("OPENAI_API_KEY") == "OPENAI_API_KEY"
    assert keystore.normalize_key_name("engine") == "CODNA_API_KEY"
    with pytest.raises(keystore.KeystoreError):
        keystore.normalize_key_name("not-a-provider")
    with pytest.raises(keystore.KeystoreError):
        keystore.normalize_key_name("")


def test_set_get_roundtrip_and_index(kr):
    stored = keystore.set_key("anthropic", "  sk-ant-123  ")
    assert stored == "ANTHROPIC_API_KEY"
    assert keystore.get_key("anthropic") == "sk-ant-123"  # trimmed
    assert keystore.get_key("ANTHROPIC_API_KEY") == "sk-ant-123"
    assert keystore.stored_key_names() == ["ANTHROPIC_API_KEY"]


def test_set_rejects_empty(kr):
    with pytest.raises(keystore.KeystoreError):
        keystore.set_key("openai", "   ")


def test_config_values_are_keychain_sourced(kr):
    keystore.set_key("anthropic", "sk-ant-1")
    keystore.set_key("openai", "sk-oai-2")
    cvs = keystore.config_values()
    assert set(cvs) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    assert all(cv.source == "keychain" for cv in cvs.values())
    assert cvs["ANTHROPIC_API_KEY"].value == "sk-ant-1"


def test_delete_is_idempotent(kr):
    keystore.set_key("groq", "gk-1")
    assert keystore.delete_key("groq") is True
    assert keystore.get_key("groq") is None
    assert keystore.delete_key("groq") is False  # already gone, no raise
    assert "GROQ_API_KEY" not in keystore.stored_key_names()


def test_get_and_config_values_never_raise_without_keychain(monkeypatch):
    def boom():
        raise keystore.KeystoreError("no backend")

    monkeypatch.setattr(keystore, "_keyring", boom)
    # reads degrade to absent; the fix hot path must never crash on a missing keychain
    assert keystore.get_key("anthropic") is None
    assert keystore.config_values() == {}
    assert keystore.stored_key_names() == []
    assert keystore.available() is False
    # but an explicit write surfaces the actionable error
    with pytest.raises(keystore.KeystoreError):
        keystore.set_key("anthropic", "x")


def test_disable_keychain_env_blocks_all_keyring_access(monkeypatch):
    monkeypatch.setenv("CODNA_DISABLE_KEYCHAIN", "1")

    assert keystore.available() is False
    assert keystore.get_key("anthropic") is None
    assert keystore.config_values() == {}
    assert keystore.stored_key_names() == []
    with pytest.raises(keystore.KeystoreError, match="CODNA_DISABLE_KEYCHAIN"):
        keystore.set_key("anthropic", "x")


def test_no_local_index_does_not_probe_keychain(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))

    def forbidden_keyring():
        raise AssertionError("runtime key discovery must not probe an unindexed OS keychain")

    monkeypatch.setattr(keystore, "_keyring", forbidden_keyring)

    assert keystore.stored_key_names() == []
    assert keystore.config_values() == {}


def test_stored_key_names_does_not_read_secret_values(monkeypatch, tmp_path):
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / ".codna"))
    keystore._write_local_index(["ANTHROPIC_API_KEY"])

    def forbidden_keyring():
        raise AssertionError("listing keys must not read OS keychain secret values")

    monkeypatch.setattr(keystore, "_keyring", forbidden_keyring)

    assert keystore.stored_key_names() == ["ANTHROPIC_API_KEY"]


def test_index_only_holds_names_never_secrets(kr):
    keystore.set_key("anthropic", "sk-ant-SECRET")
    index_raw = kr.store[(keystore.SERVICE, keystore._INDEX_USERNAME)]
    assert "ANTHROPIC_API_KEY" in index_raw
    assert "sk-ant-SECRET" not in index_raw
