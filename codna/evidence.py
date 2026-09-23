"""Evidence attestation — the signed, replay-resistant handoff between the analysis worker
and the PR writer (gate condition G9).

The worker binds the original commit, patch digest, resulting tree, scanner outputs, proof
results, test logs, Mojo verdict, and policy decision into one payload, stamps a single-use
nonce, and signs it. The writer (which cannot execute repo code) re-verifies the signature,
checks every required field is present, and consumes the nonce so a prior valid attestation
cannot be replayed against a moved base or an already-opened PR (ATT-REPLAY).

The signer is pluggable (`Signer` protocol). `HmacSigner` (stdlib `hmac`) is used for tests
and self-hosted runs; a production deployment can swap an ed25519/KMS/Sigstore signer — the
verify/binding/replay logic is identical. Key custody for the real signer is out of scope
here (see the plan's residual-risks section).
"""
from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from .findings import stable_json

REQUIRED_FIELDS = (
    "original_commit",
    "patch_digest",
    "resulting_tree",
    "scanner_outputs",
    "proof_results",
    "test_logs",
    "mojo_verdict",
    "policy_decision",
    "nonce",
)


class AttestationError(Exception):
    pass


class Signer(Protocol):
    @property
    def key_id(self) -> str: ...
    def sign(self, data: bytes) -> str: ...
    def verify(self, data: bytes, signature: str) -> bool: ...


class HmacSigner:
    """Deterministic keyed-MAC signer (stdlib). Same payload + key -> same signature."""

    def __init__(self, key: bytes, key_id: str = "worker-key"):
        self._key = key
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, data: bytes) -> str:
        return hmac.new(self._key, data, sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(data), signature or "")


def new_nonce() -> str:
    """Fresh single-use nonce. Injected explicitly in tests for determinism."""
    return os.urandom(16).hex()


@dataclass(frozen=True)
class Attestation:
    payload: dict
    signature: str
    signer_id: str

    def to_dict(self) -> dict:
        return {"payload": self.payload, "signature": self.signature, "signer_id": self.signer_id}


def _signing_bytes(payload: dict) -> bytes:
    # Canonical (sorted-key) serialization -> order-independent, deterministic (ATT-05).
    return stable_json(payload).encode("utf-8")


def build_attestation(signer: Signer, *, nonce: str, **fields) -> Attestation:
    missing = [f for f in REQUIRED_FIELDS if f != "nonce" and f not in fields]
    if missing:
        raise AttestationError(f"cannot build attestation, missing fields: {missing}")
    payload = {**fields, "nonce": nonce}
    return Attestation(payload, signer.sign(_signing_bytes(payload)), signer.key_id)


class NonceStore:
    """Single-use enforcement. `use` returns False if the nonce was already consumed."""

    def __init__(self):
        self._seen: set[str] = set()

    def use(self, nonce: str) -> bool:
        if nonce in self._seen:
            return False
        self._seen.add(nonce)
        return True


def attestation_from_dict(d: dict) -> Attestation:
    return Attestation(d["payload"], d["signature"], d["signer_id"])


def write_evidence(evidence_dir: str, *, attestation: Attestation, patch_diff: str, finding=None) -> None:
    """Persist the worker's signed handoff so a separate, unprivileged writer job can consume
    it: the attestation, the patch, and a minimal finding descriptor (identity for the branch)."""
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "attestation.json"), "w", encoding="utf-8") as fh:
        fh.write(stable_json(attestation.to_dict()))
    with open(os.path.join(evidence_dir, "patch.diff"), "w", encoding="utf-8") as fh:
        fh.write(patch_diff)
    if finding is not None:
        with open(os.path.join(evidence_dir, "finding.json"), "w", encoding="utf-8") as fh:
            fh.write(stable_json({
                "canonical_id": finding.canonical_id,
                "rule_id": finding.rule_id,
                "finding_kind": finding.finding_kind.value,
            }))


def read_evidence(evidence_dir: str) -> tuple[Attestation, str, dict]:
    """Inverse of `write_evidence`: (attestation, patch_diff, finding_descriptor)."""
    with open(os.path.join(evidence_dir, "attestation.json"), encoding="utf-8") as fh:
        att = attestation_from_dict(json.load(fh))
    with open(os.path.join(evidence_dir, "patch.diff"), encoding="utf-8") as fh:
        diff = fh.read()
    descriptor: dict = {}
    fpath = os.path.join(evidence_dir, "finding.json")
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as fh:
            descriptor = json.load(fh)
    return att, diff, descriptor


def verify_attestation(
    att: Attestation,
    *,
    trusted_signers: dict[str, Signer],
    nonce_store: NonceStore | None = None,
) -> tuple[bool, str]:
    """Verify an attestation before the writer acts on it. Checks: all required fields
    present, signer is trusted, signature valid over the canonical payload, and (if a
    nonce store is given) the nonce is fresh. Returns (ok, reason)."""
    for f in REQUIRED_FIELDS:
        if f not in att.payload:
            return False, f"missing required field {f!r}"
    signer = trusted_signers.get(att.signer_id)
    if signer is None:
        return False, f"untrusted signer {att.signer_id!r}"
    if not signer.verify(_signing_bytes(att.payload), att.signature):
        return False, "signature invalid (payload tampered or wrong key)"
    if nonce_store is not None and not nonce_store.use(att.payload["nonce"]):
        return False, "attestation replayed / nonce already used"
    return True, "ok"
