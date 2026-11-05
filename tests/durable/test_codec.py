"""AES-GCM `PayloadCodec`: nonce uniqueness, authentication, key secrecy, and rotation.

The negatives matter more than the round trip:
  * a tampered ciphertext must be REJECTED, not decrypted to garbage;
  * a nonce must never repeat (reuse under one GCM key is catastrophic);
  * the key must appear in no part of an encoded payload.
"""
from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

import pytest

from codna.durable.codec import (
    CODEC_ENCODING,
    AesGcmPayloadCodec,
    CodecKeyError,
    CodecKeyRing,
    KeyMaterial,
    NullPayloadCodec,
    PayloadDecryptionError,
    codec_key_ring_from_keystore,
    decode_key_from_keystore,
    encode_key_for_keystore,
    generate_key_material,
    key_id_for,
)

# A minimal stand-in for `temporalio.api.common.v1.Payload`: the codec only touches
# `data` and `metadata`, and importing the real message would make this suite need the extra.
class FakePayload:
    def __init__(self, data: bytes = b"", metadata: dict[str, bytes] | None = None) -> None:
        self.data = data
        self.metadata = dict(metadata or {})

    def CopyFrom(self, other: "FakePayload") -> None:
        self.data = other.data
        self.metadata = dict(other.metadata)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture()
def ring() -> CodecKeyRing:
    return CodecKeyRing(current=generate_key_material())


@pytest.fixture()
def codec(ring: CodecKeyRing) -> AesGcmPayloadCodec:
    return AesGcmPayloadCodec(ring)


def test_round_trip_and_the_encoding_label(codec: AesGcmPayloadCodec) -> None:
    original = FakePayload(b'{"prompt":"fix the bug","api_key":"sk-secret"}')
    encoded = run(codec.encode([original]))[0]
    assert encoded.metadata["encoding"] == CODEC_ENCODING.encode()
    assert encoded.data != original.data, "the payload must actually be encrypted"

    decoded = run(codec.decode([encoded]))[0]
    assert decoded.data == original.data
    assert "encoding" not in decoded.metadata, "metadata must be restored, not left encoded"
    assert decoded.metadata == {}


def test_every_payload_is_encrypted_not_a_denylist(codec: AesGcmPayloadCodec) -> None:
    """A codec that encrypted only "sensitive-looking" fields would leak the next field added."""
    payloads = [
        FakePayload(b"run-1"),
        FakePayload(b"/Users/angel/repo"),
        FakePayload(b"anthropic"),
        FakePayload(b'{"usage":{"inputTokens":10}}'),
    ]
    encoded = run(codec.encode(payloads))
    for original, item in zip(payloads, encoded):
        assert item.data != original.data
        assert original.data not in item.data


def test_nonces_are_unique_across_many_encryptions_of_the_SAME_plaintext(
    codec: AesGcmPayloadCodec,
) -> None:
    """GCM nonce reuse under one key leaks plaintext XOR. 500 draws must all differ."""
    plaintext = b"identical every time"
    nonces = set()
    for _ in range(500):
        encoded = run(codec.encode([FakePayload(plaintext)]))[0]
        nonces.add(encoded.data[:12])
    assert len(nonces) == 500


def test_ciphertexts_differ_for_identical_plaintext(codec: AesGcmPayloadCodec) -> None:
    """A deterministic ciphertext would leak equality between turns."""
    first = run(codec.encode([FakePayload(b"same")]))[0]
    second = run(codec.encode([FakePayload(b"same")]))[0]
    assert first.data != second.data


# ── NEGATIVE PROOF (b): tampering is REJECTED, never decrypted to garbage ─────────────────────


def test_a_flipped_ciphertext_bit_is_rejected(codec: AesGcmPayloadCodec) -> None:
    encoded = run(codec.encode([FakePayload(b"important payload")]))[0]
    tampered = FakePayload(bytearray(encoded.data), encoded.metadata)
    # Flip a bit in the CIPHERTEXT (past the 12-byte nonce, before the tag).
    tampered.data[20] ^= 0x01
    with pytest.raises(PayloadDecryptionError) as caught:
        run(codec.decode([tampered]))
    assert caught.value.code == "payload_authentication_failed"


def test_a_flipped_tag_bit_is_rejected(codec: AesGcmPayloadCodec) -> None:
    encoded = run(codec.encode([FakePayload(b"important payload")]))[0]
    tampered = FakePayload(bytearray(encoded.data), encoded.metadata)
    tampered.data[-1] ^= 0x01  # last byte is inside the 16-byte tag
    with pytest.raises(PayloadDecryptionError) as caught:
        run(codec.decode([tampered]))
    assert caught.value.code == "payload_authentication_failed"


def test_a_flipped_nonce_bit_is_rejected(codec: AesGcmPayloadCodec) -> None:
    encoded = run(codec.encode([FakePayload(b"important payload")]))[0]
    tampered = FakePayload(bytearray(encoded.data), encoded.metadata)
    tampered.data[0] ^= 0x01
    with pytest.raises(PayloadDecryptionError):
        run(codec.decode([tampered]))


def test_a_RELABELED_key_id_is_rejected(codec: AesGcmPayloadCodec) -> None:
    """The key id is AAD: renaming it must break the tag, not silently pick another key."""
    other = generate_key_material()
    ring = CodecKeyRing(current=codec._ring.current, retired=(other,))
    widened = AesGcmPayloadCodec(ring)
    encoded = run(codec.encode([FakePayload(b"payload")]))[0]
    relabeled = FakePayload(encoded.data, encoded.metadata)
    relabeled.metadata["codna.key_id"] = other.key_id.encode()
    with pytest.raises(PayloadDecryptionError) as caught:
        run(widened.decode([relabeled]))
    assert caught.value.code == "payload_authentication_failed"


def test_truncated_data_is_rejected(codec: AesGcmPayloadCodec) -> None:
    encoded = run(codec.encode([FakePayload(b"x")]))[0]
    with pytest.raises(PayloadDecryptionError) as caught:
        run(codec.decode([FakePayload(encoded.data[:10], encoded.metadata)]))
    assert caught.value.code == "payload_corrupt"


# ── the key never leaks into the payload ─────────────────────────────────────────────────────


def test_the_key_does_not_appear_in_the_encoded_payload(ring: CodecKeyRing) -> None:
    codec = AesGcmPayloadCodec(ring)
    encoded = run(codec.encode([FakePayload(b"x" * 64)]))[0]
    key = ring.current.key
    # Every common representation: raw, hex, base64, and the key id's full hash.
    representations = {
        key,
        key.hex().encode(),
        base64.b64encode(key),
        repr(key).encode(),
        key.hex()[:16].encode(),
    }
    blobs = [bytes(encoded.data), *[bytes(v) for v in encoded.metadata.values()]]
    for blob in blobs:
        for representation in representations:
            assert representation not in blob, (
                f"the key material ({representation!r}) leaked into the payload"
            )


def test_the_key_does_not_appear_in_metadata_keys_or_repr(ring: CodecKeyRing) -> None:
    codec = AesGcmPayloadCodec(ring)
    encoded = run(codec.encode([FakePayload(b"x")]))[0]
    key_hex = ring.current.key.hex()
    assert key_hex not in repr(encoded.metadata)
    assert key_hex not in repr(ring)
    assert key_hex not in repr(ring.current)
    assert "redacted" in repr(ring.current)


def test_the_key_id_is_a_hash_prefix_not_the_key(ring: CodecKeyRing) -> None:
    assert ring.current.key_id == key_id_for(ring.current.key)
    assert ring.current.key_id not in ring.current.key.hex()


# ── rotation: what happens to OLD payloads ───────────────────────────────────────────────────


def test_rotation_keeps_old_history_readable_when_the_old_key_is_retained() -> None:
    old = generate_key_material()
    historical = run(AesGcmPayloadCodec(CodecKeyRing(current=old)).encode([FakePayload(b"old")]))[0]

    new = generate_key_material()
    rotated = AesGcmPayloadCodec(CodecKeyRing(current=new, retired=(old,)))
    # The new key encrypts NEW payloads...
    fresh = run(rotated.encode([FakePayload(b"new")]))[0]
    assert fresh.metadata["codna.key_id"] == new.key_id.encode()
    # ...and old payloads still decode, because the ring kept their key.
    assert run(rotated.decode([historical]))[0].data == b"old"


def test_rotation_WITHOUT_the_old_key_refuses_rather_than_corrupting() -> None:
    """The stated behaviour: a loud `payload_key_unavailable`, never a garbage decrypt."""
    old = generate_key_material()
    historical = run(AesGcmPayloadCodec(CodecKeyRing(current=old)).encode([FakePayload(b"old")]))[0]

    new = generate_key_material()
    rotated_only = AesGcmPayloadCodec(CodecKeyRing(current=new))
    with pytest.raises(PayloadDecryptionError) as caught:
        run(rotated_only.decode([historical]))
    assert caught.value.code == "payload_key_unavailable"
    # The error names the key an operator must restore — not just "decryption failed".
    assert caught.value.details["key_id"] == old.key_id
    assert old.key_id in caught.value.details["hint"]


def test_a_ring_with_a_duplicate_key_id_is_refused() -> None:
    material = generate_key_material()
    with pytest.raises(CodecKeyError) as caught:
        CodecKeyRing(current=material, retired=(material,))
    assert caught.value.code == "codec_key_duplicate"


def test_encoding_a_payload_twice_does_not_double_wrap(codec: AesGcmPayloadCodec) -> None:
    once = run(codec.encode([FakePayload(b"x")]))[0]
    twice = run(codec.encode([once]))[0]
    assert twice.data == once.data
    assert run(codec.decode([twice]))[0].data == b"x"


def test_the_payloads_OWN_encoding_survives_a_round_trip(codec: AesGcmPayloadCodec) -> None:
    """Regression: a decode used to strip the payload's own `encoding` label.

    Encrypting REPLACES `encoding` with the codec's marker, so the original (`json/plain`) has to be
    carried alongside — otherwise a decoded payload has no encoding at all and the SDK's payload
    converter raises `KeyError: Unknown payload encoding <unknown>` the moment a workflow reads its
    own decoded argument. Found by driving a real `DataConverter` (D6); this pins the property at the
    codec level so a future refactor cannot reintroduce it.
    """
    original = FakePayload(b'{"prompt":"fix the retry"}', {"encoding": b"json/plain"})
    encoded = run(codec.encode([original]))[0]
    # The label is NOT readable from the ciphertext alone: it travels in metadata, beside the key id.
    assert encoded.metadata["encoding"] == CODEC_ENCODING.encode()
    assert codec.current_key_id.encode() not in encoded.data

    decoded = run(codec.decode([encoded]))[0]
    assert decoded.data == original.data
    assert decoded.metadata == {"encoding": b"json/plain"}, (
        "the payload's own encoding must be restored, and none of the codec's metadata may remain"
    )
    # ...and re-encoding the decoded payload is a FRESH encryption, not a pass-through: it must not be
    # mistaken for one this codec already produced.
    re_encoded = run(codec.encode([decoded]))[0]
    assert re_encoded.metadata["encoding"] == CODEC_ENCODING.encode()
    assert run(codec.decode([re_encoded]))[0].data == original.data


def test_a_payload_with_no_encoding_comes_back_with_no_encoding(codec: AesGcmPayloadCodec) -> None:
    """A raw binary payload must not gain a label it never had (the codec invents nothing)."""
    decoded = run(codec.decode(run(codec.encode([FakePayload(b"\x00\x01raw")]))))[0]
    assert decoded.data == b"\x00\x01raw"
    assert "encoding" not in decoded.metadata
    assert decoded.metadata == {}


def test_a_plaintext_payload_passes_through_decode(codec: AesGcmPayloadCodec) -> None:
    """History written before the codec must still be readable — that is not a permission."""
    plain = FakePayload(b"pre-codec history")
    assert run(codec.decode([plain]))[0].data == b"pre-codec history"


def test_null_codec_is_a_real_opt_out_not_a_fallback() -> None:
    codec = NullPayloadCodec()
    payload = FakePayload(b"plain")
    encoded = run(codec.encode([payload]))[0]
    assert encoded is payload
    assert encoded.data == b"plain"


# ── key material / keystore encoding ────────────────────────────────────────────────────────


def test_key_material_rejects_the_wrong_length() -> None:
    with pytest.raises(CodecKeyError) as caught:
        KeyMaterial(key=os.urandom(16))
    assert caught.value.code == "codec_key_invalid"


def test_keystore_encoding_round_trips() -> None:
    material = generate_key_material()
    text = encode_key_for_keystore(material)
    assert decode_key_from_keystore(text).key == material.key
    assert material.key.hex() not in text, "the keystore entry must not contain the raw hex key"


@pytest.mark.parametrize("value", ["", "   ", "not base64!", base64.b64encode(b"short").decode()])
def test_bad_keystore_entries_are_refused(value: str) -> None:
    with pytest.raises(CodecKeyError):
        decode_key_from_keystore(value)


def test_the_ring_is_assembled_from_an_injected_lookup() -> None:
    current = generate_key_material()
    retired = generate_key_material()
    entries = {
        "CODNA_TEMPORAL_CODEC_KEY": encode_key_for_keystore(current),
        f"CODNA_TEMPORAL_CODEC_KEY__{retired.key_id}": encode_key_for_keystore(retired),
    }
    ring = codec_key_ring_from_keystore(
        lookup=entries.get,
        retired_names=[f"CODNA_TEMPORAL_CODEC_KEY__{retired.key_id}"],
    )
    assert ring.current.key_id == current.key_id
    assert [m.key_id for m in ring.retired] == [retired.key_id]


def test_a_missing_current_key_is_a_named_startup_failure() -> None:
    with pytest.raises(CodecKeyError) as caught:
        codec_key_ring_from_keystore(lookup=lambda _name: None)
    assert caught.value.code == "codec_key_missing"
    assert "keygen" in caught.value.details["hint"]


def test_a_mis_named_retired_entry_is_refused() -> None:
    current = generate_key_material()
    other = generate_key_material()
    bogus_name = f"CODNA_TEMPORAL_CODEC_KEY__{key_id_for(os.urandom(32))}"
    entries = {
        "CODNA_TEMPORAL_CODEC_KEY": encode_key_for_keystore(current),
        # The suffix claims one id, the value is a different key.
        bogus_name: encode_key_for_keystore(other),
    }
    with pytest.raises(CodecKeyError) as caught:
        codec_key_ring_from_keystore(lookup=entries.get, retired_names=[bogus_name])
    assert caught.value.code == "codec_key_id_mismatch"
