"""Codec MODE resolution and key lifecycle: `auto` / `required` / `off`, and safe rotation.

The important negative: `auto` with no key must be PLAINTEXT **and say so** — not pretend to encrypt
— and `required` with no key must refuse to start. `off` is an explicit opt-out, never the default.
"""
from __future__ import annotations

import json

import pytest

from codna.durable.codec import (
    generate_key_material,
    encode_key_for_keystore,
    payload_codec_from_keystore,
)
from codna.durable.codec_config import (
    CODEC_ENV,
    CodecUnavailable,
    resolve_codec,
)

CURRENT = "CODNA_TEMPORAL_CODEC_KEY"


def _entries(material=None, retired=()) -> dict[str, str]:
    entries: dict[str, str] = {}
    if material is not None:
        entries[CURRENT] = encode_key_for_keystore(material)
    for item in retired:
        entries[f"{CURRENT}__{item.key_id}"] = encode_key_for_keystore(item)
    return entries


def _resolve(entries: dict[str, str], mode: str | None = None):
    env = {} if mode is None else {CODEC_ENV: mode}
    retired_names = [name for name in entries if name.startswith(f"{CURRENT}__")]
    return resolve_codec(env, lookup=entries.get, retired_names=retired_names)


def test_auto_with_no_key_is_PLAINTEXT_and_says_so() -> None:
    resolution = _resolve({})
    assert resolution.mode == "plaintext"
    assert resolution.encrypting is False
    assert "no codec key" in resolution.reason
    # The reason names the stricter mode, so an operator knows how to close it.
    assert "required" in resolution.reason


def test_auto_with_a_key_encrypts_and_reports_the_key_id() -> None:
    material = generate_key_material()
    resolution = _resolve(_entries(material))
    assert resolution.mode == "aes-256-gcm"
    assert resolution.key_id == material.key_id
    assert resolution.encrypting is True


def test_required_with_no_key_REFUSES_to_start() -> None:
    with pytest.raises(CodecUnavailable) as caught:
        _resolve({}, "required")
    assert caught.value.code == "codec_key_missing"
    assert "keygen" in caught.value.details["hint"]


def test_off_is_an_explicit_opt_out_even_when_a_key_exists() -> None:
    resolution = _resolve(_entries(generate_key_material()), "off")
    assert resolution.mode == "plaintext"
    assert "opt-out" in resolution.reason


def test_an_unknown_mode_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(CodecUnavailable) as caught:
        _resolve({}, "encrypt-maybe")
    assert caught.value.code == "codec_mode_invalid"


def test_resolution_is_serializable_for_a_banner() -> None:
    payload = _resolve(_entries(generate_key_material())).to_dict()
    assert set(payload) == {"mode", "key_id", "requested", "reason"}
    json.dumps(payload)  # must not raise


# ── the CLI lifecycle: keygen -> retire -> keygen, and what each step makes readable ──────────


class FakeKeychain:
    """The three keystore calls `codec_cli` makes, backed by a dict."""

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.entries.get(name)

    def set(self, name: str, value: str) -> str:
        self.entries[name] = value
        return name

    def delete(self, name: str) -> bool:
        return self.entries.pop(name, None) is not None

    def names(self) -> list[str]:
        return sorted(self.entries)


@pytest.fixture()
def keychain(monkeypatch) -> FakeKeychain:
    from codna.durable import codec_cli

    fake = FakeKeychain()
    monkeypatch.setattr(codec_cli, "get_key", fake.get)
    monkeypatch.setattr(codec_cli, "set_key", fake.set)
    monkeypatch.setattr(codec_cli, "delete_key", fake.delete)
    monkeypatch.setattr(codec_cli, "stored_key_names", fake.names)
    return fake


def _run(keychain: FakeKeychain, action: str, *, force: bool = False) -> tuple[int, dict]:
    from codna.durable import codec_cli
    from argparse import Namespace

    args = Namespace(codec_action=action, force=force)
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = codec_cli.cmd_durable_codec(args)
    return code, json.loads(buffer.getvalue())


def test_keygen_then_status(keychain: FakeKeychain) -> None:
    code, body = _run(keychain, "keygen")
    assert code == 0
    assert body["algorithm"] == "AES-256-GCM"
    key_id = body["key_id"]
    # The VALUE must never be printed.
    assert encode_key_for_keystore(generate_key_material()) not in buffer_text(body)
    assert keychain.entries[CURRENT]

    code, status = _run(keychain, "status")
    assert code == 0
    assert status["key_id"] == key_id


def buffer_text(body: dict) -> str:
    return json.dumps(body, sort_keys=True)


def test_keygen_refuses_to_overwrite_an_existing_key(keychain: FakeKeychain) -> None:
    _run(keychain, "keygen")
    from codna.durable.cli import DurableCliError

    with pytest.raises(DurableCliError) as caught:
        _run(keychain, "keygen")
    assert caught.value.code == "codec_key_exists"
    # The hint must name the safe path (retire first), not just refuse.
    assert "retire" in caught.value.details["hint"]


def test_retire_then_keygen_keeps_old_history_readable(keychain: FakeKeychain) -> None:
    """THE rotation contract: the old key is retained, so its payloads still decrypt."""
    _code, first = _run(keychain, "keygen")
    old_id = first["key_id"]

    code, retired = _run(keychain, "retire")
    assert code == 0
    assert retired["key_id"] == old_id
    assert keychain.get(CURRENT) is None, "the current entry is removed only after retiring"

    _code, second = _run(keychain, "keygen")
    new_id = second["key_id"]
    assert new_id != old_id

    ring = payload_codec_from_keystore(lookup=keychain.get, retired_names=keychain.names()).key_ring
    assert ring.current.key_id == new_id
    assert [m.key_id for m in ring.retired] == [old_id]


def test_retire_is_idempotent(keychain: FakeKeychain) -> None:
    _run(keychain, "keygen")
    _run(keychain, "retire")
    # After retiring, there is no CURRENT key, so a second retire reports the missing key rather
    # than duplicating the retired entry.
    from codna.durable.cli import DurableCliError

    with pytest.raises(DurableCliError) as caught:
        _run(keychain, "retire")
    assert caught.value.code == "codec_key_missing"


def test_retire_with_no_key_refuses(keychain: FakeKeychain) -> None:
    from codna.durable.cli import DurableCliError

    with pytest.raises(DurableCliError) as caught:
        _run(keychain, "retire")
    assert caught.value.code == "codec_key_missing"


def test_status_reports_plaintext_when_no_key_is_stored(keychain: FakeKeychain, monkeypatch) -> None:
    monkeypatch.delenv(CODEC_ENV, raising=False)
    code, status = _run(keychain, "status")
    assert code == 0
    assert status["mode"] == "plaintext"
    assert status["key_id"] is None


def test_the_worker_banner_payload_reports_the_mode_not_just_a_boolean() -> None:
    """`mode`/`reason` are what make a PLAINTEXT worker visible rather than merely non-encrypted."""
    plain = _resolve({}).to_dict()
    assert plain["mode"] == "plaintext"
    assert plain["reason"], "a plaintext worker must explain WHY, or an operator has to guess"
    encrypted = _resolve(_entries(generate_key_material())).to_dict()
    assert encrypted["mode"] == "aes-256-gcm"
    assert encrypted["key_id"]
