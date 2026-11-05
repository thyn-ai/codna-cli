"""Temporal `PayloadCodec`: AES-256-GCM over every payload a durable run sends or receives.

## What is encrypted

EVERY payload, unconditionally, not just the sensitive ones. A codec that encrypts "the fields we
think are secrets" is a denylist, and a denylist misses the next field someone adds — prompt text,
a repository path, a provider header. Encrypting everything also means the key-id scheme below
covers the whole history rather than a subset, so rotation is one property to reason about.

## AES-GCM, used correctly

* **A fresh 12-byte nonce per encryption**, from `os.urandom`. It is never derived from a counter, a
  timestamp, or the plaintext, and it is never reused: `_encrypt` draws a new one and there is no
  path that can reuse one. (Reusing a GCM nonce under one key is catastrophic — it leaks the XOR of
  the plaintexts and breaks the authentication key.) A test drives 500 encryptions of the SAME
  plaintext and asserts every nonce is distinct.
* **A 16-byte authentication tag**, verified on decrypt. A tampered ciphertext, tag, nonce or
  key-id is REJECTED with `PayloadDecryptionError`; it is never returned as garbage. `cryptography`'s
  `AESGCM.decrypt` raises `InvalidTag`, which is what "authenticated" means operationally.
* **A 32-byte key** (AES-256).
* The **additional authenticated data (AAD)** is the key id, so a payload cannot be relabeled as
  belonging to a different key — a downgrade attack that would otherwise let an attacker make a
  worker try an old, weaker key.

## The wire format (one JSON `encoding` blob, no plaintext metadata)

    encoding = "codna:enc:v1"
    data     = nonce(12) || ciphertext || tag(16)          # the KEY ID IS NOT IN `data`
    metadata = {"codna.key_id": "<8-byte hex prefix of SHA-256(key)>", "codna.scheme": "AES-256-GCM"}

The key id is a HASH PREFIX of the key, never the key itself: it lets a reader pick the right key
from a keystore without exposing which key was used in any reversible way, and it does not permit
recovering the key. The key bytes never appear in `encoding`, `data` or `metadata` — a test scans
all three for the key material and fails if it is found.

## Key rotation — what happens to OLD payloads (stated plainly)

Each payload records the key id it was encrypted with. `decode` looks that id up in the keystore's
key RING: the current key plus any retired keys the operator kept. So:

* **Rotating the key does NOT corrupt existing history** as long as the retired key remains in the
  ring. Old payloads keep decoding with their own key id.
* **A payload whose key id is NOT in the ring is REFUSED** (`payload_key_unavailable`) — it is not
  decrypted with the current key "to try", because that would either fail confusingly or, worse,
  succeed against the wrong material. The error names the missing key id so an operator knows
  exactly which retired key to restore.
* **There is no re-encryption here.** Rotation without keeping retired keys therefore makes history
  UNREADABLE, not silently wrong. That is the deliberate trade: a loud refusal, never a garbage
  decrypt. Re-encrypting history is a separate, explicit operation (Temporal's
  `reset`/re-encrypt tooling), not something a codec does behind the operator's back.

## Where the key comes from

A `CodecKeyRing`, resolved from the KEYSTORE — never an inline literal, never `os.environ` read at
import time. The `durable` extra already carries the OS keychain (`codna.keystore`); this module adds
a `CODEX_`-prefixed entry whose value is base64 of 32 random bytes:

    keystore entry key:  CODNA_TEMPORAL_CODEC_KEY            (current)
    keystore entry key:  CODNA_TEMPORAL_CODEC_KEY__<key_id>  (a retired key, kept for old history)

Reading happens when the client is built (`codec_key_ring_from_keystore()`), so an import of this
module needs no keychain — which is what lets the base CLI import the durable package without the
extra. A keychain with no codec entry and no `--no-codec` opt-out is a STARTUP failure with an
actionable hint (`codec_key_missing`), not a silent plaintext run.

## Unconfigured / opt-out

`NullPayloadCodec` is the explicit opt-out: it returns payloads untouched. It exists so a
`--no-codec` flag has a real object to carry (and tests can prove the encrypting path is not
accidentally a no-op), NOT as a fallback. Nothing selects it because a key is missing.
"""
from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..keystore import CODEC_KEY_NAME, CODEC_RETIRED_PREFIX

__all__ = [
    "AesGcmPayloadCodec",
    "CODEC_ENCODING",
    "CodecKeyError",
    "CodecKeyRing",
    "KeyMaterial",
    "NullPayloadCodec",
    "PayloadDecryptionError",
    "codec_key_ring_from_keystore",
    "generate_key_material",
    "payload_codec_from_keystore",
]

CODEC_ENCODING = "codna:enc:v1"
_KEY_ID_METADATA = "codna.key_id"
_SCHEME_METADATA = "codna.scheme"
# The payload's OWN `encoding` before we replaced it (usually `json/plain`). Without it a decode
# returns a payload whose encoding is gone, and the SDK's payload converter raises
# `KeyError: Unknown payload encoding <unknown>` on the way back — measured. It is a METADATA label,
# never plaintext, so it is safe to keep beside the ciphertext.
_ORIGINAL_ENCODING_METADATA = "codna.original.encoding"
_SCHEME = "AES-256-GCM"
_NONCE_BYTES = 12
_TAG_BYTES = 16
_KEY_BYTES = 32
#: Keystore entry names, imported from `keystore` so the CLI and the codec cannot drift apart.
KEYSTORE_CURRENT = CODEC_KEY_NAME
KEYSTORE_RETIRED_PREFIX = CODEC_RETIRED_PREFIX


class CodecKeyError(RuntimeError):
    """The codec could not obtain a usable key. Always carries a stable code."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class PayloadDecryptionError(RuntimeError):
    """A payload could not be authenticated or decrypted. NEVER a silent garbage return."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _require_cryptography() -> Any:
    """Import AESGCM lazily so importing the durable package never needs the dependency."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise CodecKeyError(
            "the Temporal payload codec needs the `cryptography` package",
            code="codec_dependency_missing",
            details={"package": "cryptography", "hint": "pip install 'codna[durable]'"},
        ) from exc
    return AESGCM


def key_id_for(key: bytes) -> str:
    """A stable, non-reversible identifier for a key: 8 hex chars of its SHA-256.

    A truncated hash is enough to tell keys apart and reveals nothing useful about the key. It is NOT
    a substitute for authentication — the GCM tag is — it is only how a reader picks a key.
    """
    return hashlib.sha256(key).hexdigest()[:16]


@dataclass(frozen=True)
class KeyMaterial:
    """One AES-256 key and its id. `__repr__` never prints the key."""

    key: bytes
    key_id: str = ""

    def __post_init__(self) -> None:
        if len(self.key) != _KEY_BYTES:
            raise CodecKeyError(
                f"an AES-256 key must be {_KEY_BYTES} bytes, got {len(self.key)}",
                code="codec_key_invalid",
                details={"expected_bytes": _KEY_BYTES, "received_bytes": len(self.key)},
            )
        if not self.key_id:
            object.__setattr__(self, "key_id", key_id_for(self.key))

    def __repr__(self) -> str:  # pragma: no cover - exercised by the secrecy test
        return f"KeyMaterial(key_id={self.key_id!r}, key=<redacted>)"


@dataclass(frozen=True)
class CodecKeyRing:
    """The current key plus retired keys, enough to READ history written before a rotation.

    `current` is used for every ENCRYPTION. `retired` is consulted only by `decode`, and only by the
    key id recorded in the payload — so rotation keeps old history readable without ever encrypting
    new data under an old key.
    """

    current: KeyMaterial
    retired: tuple[KeyMaterial, ...] = ()

    def __post_init__(self) -> None:
        ids = {self.current.key_id, *(m.key_id for m in self.retired)}
        if len(ids) != 1 + len(self.retired):
            raise CodecKeyError(
                "two keys in the codec ring share an id; refusing an ambiguous ring",
                code="codec_key_duplicate",
                details={"key_ids": sorted(ids)},
            )

    def by_id(self) -> dict[str, KeyMaterial]:
        ring = {self.current.key_id: self.current}
        for material in self.retired:
            ring[material.key_id] = material
        return ring

    def __repr__(self) -> str:  # pragma: no cover - exercised by the secrecy test
        return (
            f"CodecKeyRing(current=<{self.current.key_id}>, "
            f"retired={[m.key_id for m in self.retired]})"
        )


def generate_key_material() -> KeyMaterial:
    """A fresh random AES-256 key. Used by `codna durable codec keygen` and by tests."""
    return KeyMaterial(key=os.urandom(_KEY_BYTES))


def encode_key_for_keystore(material: KeyMaterial) -> str:
    """The base64 text a keystore entry holds. The ONLY representation written to disk/keychain."""
    return base64.b64encode(material.key).decode("ascii")


def decode_key_from_keystore(value: str) -> KeyMaterial:
    """Parse a keystore entry, refusing anything that is not exactly 32 base64'd bytes."""
    raw = (value or "").strip()
    if not raw:
        raise CodecKeyError("the codec key entry is empty", code="codec_key_invalid")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same refusal
        raise CodecKeyError(
            "the codec key entry is not valid base64",
            code="codec_key_invalid",
            details={"received_chars": len(raw)},
        ) from exc
    return KeyMaterial(key=decoded)


class AesGcmPayloadCodec:
    """A Temporal `PayloadCodec` that AES-256-GCM-encrypts every payload.

    Structurally compatible with `temporalio.converter.PayloadCodec` (an `encode`/`decode` pair over
    `Sequence[Payload]`) without importing it: this module must be importable without the `durable`
    extra, and the codec is a duck-typed seam either way.
    """

    def __init__(self, ring: CodecKeyRing) -> None:
        self._ring = ring
        self._aesgcm = _require_cryptography()
        self._by_id = ring.by_id()

    @property
    def current_key_id(self) -> str:
        return self._ring.current.key_id

    @property
    def key_ring(self) -> CodecKeyRing:
        """The ring this codec decrypts with. Exposed so a caller can report which keys are usable."""
        return self._ring

    async def encode(self, payloads: Sequence[Any]) -> list[Any]:
        return [self._encode_one(payload) for payload in payloads]

    async def decode(self, payloads: Sequence[Any]) -> list[Any]:
        return [self._decode_one(payload) for payload in payloads]

    # -- the primitives, sync so tests can drive them without an event loop ----------------

    def _encode_one(self, payload: Any) -> Any:
        material = self._ring.current
        # A payload that is ALREADY encoded is passed through unchanged. Temporal may encode the
        # same payload twice (request body + search attributes); re-encrypting would be wasteful but,
        # more importantly, would nest a ciphertext inside a ciphertext and make the key id in the
        # outer blob the only recoverable one.
        if _is_encoded(payload):
            return payload
        nonce = os.urandom(_NONCE_BYTES)  # fresh, every time; never derived, never reused
        aesgcm = self._aesgcm(material.key)
        # The key id as AAD binds the ciphertext to the key that produced it, so relabelling a
        # payload to another key id invalidates the tag instead of being silently accepted.
        ciphertext = aesgcm.encrypt(nonce, bytes(payload.data), material.key_id.encode("utf-8"))
        cloned = type(payload)()
        cloned.CopyFrom(payload)
        # Remember the payload's own encoding BEFORE replacing it. It is a label (e.g. `json/plain`),
        # not content: the SDK's payload converter needs it to interpret the decrypted bytes, and a
        # decode that dropped it produced `KeyError: Unknown payload encoding <unknown>`.
        original_encoding = payload.metadata.get("encoding")
        cloned.data = nonce + ciphertext
        cloned.metadata[_KEY_ID_METADATA] = material.key_id.encode("utf-8")
        cloned.metadata[_SCHEME_METADATA] = _SCHEME.encode("utf-8")
        cloned.metadata["encoding"] = CODEC_ENCODING.encode("utf-8")
        if original_encoding is not None:
            cloned.metadata[_ORIGINAL_ENCODING_METADATA] = bytes(original_encoding)
        return cloned

    def _decode_one(self, payload: Any) -> Any:
        if not _is_encoded(payload):
            # Not ours: a payload written before the codec, or a plaintext produced by a caller that
            # opted out. Returned untouched — that is not a permission decision, it is "nothing to
            # decrypt", and refusing it would make enabling the codec a breaking change.
            return payload
        key_id_raw = payload.metadata.get(_KEY_ID_METADATA)
        if not key_id_raw:
            raise PayloadDecryptionError(
                "an encoded payload carries no key id",
                code="payload_key_id_missing",
                details={"encoding": CODEC_ENCODING},
            )
        key_id = bytes(key_id_raw).decode("utf-8", errors="strict")
        material = self._by_id.get(key_id)
        if material is None:
            # NOT "try the current key". An old payload needs its old key; guessing corrupts history
            # in the most confusing way possible (or, with GCM, just fails). Name the missing id.
            raise PayloadDecryptionError(
                f"this payload was encrypted with key {key_id!r}, which is not in the key ring",
                code="payload_key_unavailable",
                details={
                    "key_id": key_id,
                    "available_key_ids": sorted(self._by_id),
                    "hint": (
                        "a rotated key must be kept as a retired key to read old history; restore it "
                        f"under {KEYSTORE_RETIRED_PREFIX}{key_id}"
                    ),
                },
            )
        blob = bytes(payload.data)
        if len(blob) < _NONCE_BYTES + _TAG_BYTES:
            raise PayloadDecryptionError(
                "an encoded payload is too short to contain a nonce and a tag",
                code="payload_corrupt",
                details={"bytes": len(blob), "minimum": _NONCE_BYTES + _TAG_BYTES},
            )
        nonce = blob[:_NONCE_BYTES]
        ciphertext = blob[_NONCE_BYTES:]
        aesgcm = self._aesgcm(material.key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, key_id.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - InvalidTag and friends are all the same refusal
            # This is the AUTHENTICATED half: a flipped bit in the ciphertext, the tag, the nonce or
            # the key id lands here. Returning the raw bytes "best effort" is exactly the garbage
            # decryption the track forbids.
            raise PayloadDecryptionError(
                "a payload failed authentication; it has been tampered with or was encrypted with a "
                "different key",
                code="payload_authentication_failed",
                details={"key_id": key_id, "cause": type(exc).__name__},
            ) from exc
        cloned = type(payload)()
        cloned.CopyFrom(payload)
        cloned.data = plaintext
        # Restore the payload's OWN metadata: the encoding label it carried before encryption, and
        # nothing of the codec's. Leaving `encoding` set would make a re-encode treat it as
        # already-encoded and skip it, so a payload could travel unencrypted after a round trip; and
        # DROPPING the original label makes the SDK's payload converter fail with
        # `KeyError: Unknown payload encoding` on a payload that travelled through a workflow.
        for name in (_KEY_ID_METADATA, _SCHEME_METADATA):
            cloned.metadata.pop(name, None)
        original_encoding = cloned.metadata.pop(_ORIGINAL_ENCODING_METADATA, None)
        if original_encoding is not None:
            cloned.metadata["encoding"] = bytes(original_encoding)
        else:
            # A payload that arrived with NO encoding (a raw binary payload) must come back the same
            # way: the codec's own label is removed and nothing is invented in its place.
            cloned.metadata.pop("encoding", None)
        return cloned


class NullPayloadCodec:
    """The explicit opt-out. Returns payloads untouched — never selected because a key is missing."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "NullPayloadCodec()"

    async def encode(self, payloads: Sequence[Any]) -> list[Any]:
        return list(payloads)

    async def decode(self, payloads: Sequence[Any]) -> list[Any]:
        return list(payloads)


def _is_encoded(payload: Any) -> bool:
    metadata = getattr(payload, "metadata", None) or {}
    raw = metadata.get("encoding")
    if raw is None:
        return False
    try:
        return bytes(raw).decode("utf-8") == CODEC_ENCODING
    except Exception:  # noqa: BLE001 - an undecodable encoding label is not ours
        return False


# ── keystore resolution ──────────────────────────────────────────────────────────────────────


def codec_key_ring_from_keystore(
    lookup: Any | None = None,
    retired_names: Iterable[str] | None = None,
) -> CodecKeyRing:
    """Build the key ring from the OS keychain.

    Two injectable seams, because assembly needs both a read and an enumeration:

    * `lookup(name)` defaults to `codna.keystore.get_key`, which returns None rather than raising
      when the keychain is unavailable;
    * `retired_names` defaults to the keystore's own non-secret name index. A caller that supplied
      its own `lookup` MUST also supply the names, since a lookup function cannot enumerate.

    The caller (`payload_codec_from_keystore`) turns a `None` current key into `codec_key_missing` —
    this function only assembles what is present.
    """
    if lookup is None:
        from ..keystore import get_key as lookup  # type: ignore[assignment]
    if retired_names is None:
        retired_names = _retired_entry_names()

    current_raw = lookup(KEYSTORE_CURRENT)
    if not current_raw:
        raise CodecKeyError(
            "no Temporal payload-codec key is stored on this device",
            code="codec_key_missing",
            details={
                "keystore_entry": KEYSTORE_CURRENT,
                "hint": "run `codna durable codec keygen` to create one on this device",
            },
        )
    current = decode_key_from_keystore(current_raw)
    retired: list[KeyMaterial] = []
    # Retired keys are named by their OWN id suffix, so the ring can be assembled without a separate
    # manifest: the operator adds `<PREFIX><key_id>` and the payloads referencing that id decode again.
    for name in retired_names:
        if not name.startswith(KEYSTORE_RETIRED_PREFIX):
            continue
        value = lookup(name)
        if not value:
            continue
        material = decode_key_from_keystore(value)
        if material.key_id != name[len(KEYSTORE_RETIRED_PREFIX) :]:
            raise CodecKeyError(
                f"keystore entry {name} does not hold the key its name claims",
                code="codec_key_id_mismatch",
                details={"entry": name, "actual_key_id": material.key_id},
            )
        if material.key_id == current.key_id:
            continue
        retired.append(material)
    # Deterministic order so a ring's repr and a test's assertion do not depend on index order.
    retired.sort(key=lambda m: m.key_id)
    return CodecKeyRing(current=current, retired=tuple(retired))


def _retired_entry_names() -> Iterable[str]:
    """Retired-key entry names from the keystore's non-secret index."""
    try:
        from ..keystore import stored_key_names  # type: ignore
    except Exception:  # noqa: BLE001 - no keystore module: no retired keys, not an error
        return ()
    try:
        names = stored_key_names()
    except Exception:  # noqa: BLE001 - a broken index must not break the ring
        return ()
    return [name for name in names if name.startswith(KEYSTORE_RETIRED_PREFIX)]


def payload_codec_from_keystore(
    lookup: Any | None = None,
    retired_names: Iterable[str] | None = None,
) -> AesGcmPayloadCodec:
    """The codec a real client installs. Raises `CodecKeyError` when no key is configured."""
    return AesGcmPayloadCodec(codec_key_ring_from_keystore(lookup, retired_names))


def data_converter_for(codec: Any) -> Any:
    """A `DataConverter` carrying `codec`, or a bare codec seam for a caller that cannot import it.

    Kept here so the client construction in `config.py` names ONE helper instead of repeating the
    keyword, which is how a `Client.connect` in a second file forgets the codec and writes plaintext.
    """
    from temporalio.converter import DataConverter

    return DataConverter(payload_codec=codec)
