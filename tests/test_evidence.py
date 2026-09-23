"""Evidence-attestation tests (A–Z test plan: ATT-01..05, ATT-REPLAY, G9)."""
from __future__ import annotations


from codna.evidence import (
    Attestation,
    HmacSigner,
    NonceStore,
    _signing_bytes,
    build_attestation,
    verify_attestation,
)

WORKER = HmacSigner(b"worker-secret", key_id="worker-key")
ATTACKER = HmacSigner(b"attacker-secret", key_id="attacker-key")
TRUSTED = {WORKER.key_id: WORKER}


def _fields(**over):
    base = dict(
        original_commit="a" * 40,
        patch_digest="sha256:patch",
        resulting_tree="sha256:tree",
        scanner_outputs={"baseline": "...", "patched": "..."},
        proof_results={"closure_status": "closed"},
        test_logs="all passed",
        mojo_verdict={"recommended_action": "apply_patch"},
        policy_decision={"eligible": True, "classification": "exploitable"},
    )
    base.update(over)
    return base


def test_att01_binds_all_fields_and_verifies():
    att = build_attestation(WORKER, nonce="n1", **_fields())
    ok, reason = verify_attestation(att, trusted_signers=TRUSTED)
    assert ok is True, reason
    assert att.signer_id == "worker-key"


def test_att02_missing_field_fails_verification():
    att = build_attestation(WORKER, nonce="n1", **_fields())
    # drop a bound field and re-sign so the signature itself is valid — verify must still
    # reject on the structural check, not merely on a signature mismatch.
    payload = dict(att.payload)
    del payload["mojo_verdict"]
    forged = Attestation(payload, WORKER.sign(_signing_bytes(payload)), WORKER.key_id)
    ok, reason = verify_attestation(forged, trusted_signers=TRUSTED)
    assert ok is False
    assert "mojo_verdict" in reason


def test_att03_tampered_payload_fails_signature():
    att = build_attestation(WORKER, nonce="n1", **_fields())
    tampered = Attestation({**att.payload, "patch_digest": "sha256:evil"}, att.signature, att.signer_id)
    ok, reason = verify_attestation(tampered, trusted_signers=TRUSTED)
    assert ok is False
    assert "signature" in reason


def test_att04_untrusted_signer_rejected_legit_accepted():
    attacker_att = build_attestation(ATTACKER, nonce="n1", **_fields())
    ok, reason = verify_attestation(attacker_att, trusted_signers=TRUSTED)
    assert ok is False
    assert "untrusted signer" in reason
    # positive control: the rejection is enforcement, not an empty allowlist
    assert len(TRUSTED) == 1 and WORKER.key_id in TRUSTED
    legit = build_attestation(WORKER, nonce="n2", **_fields())
    assert verify_attestation(legit, trusted_signers=TRUSTED)[0] is True


def test_att05_canonicalization_order_independent():
    a = build_attestation(WORKER, nonce="n1", **_fields())
    # build with the same fields supplied in a different order
    reordered = dict(reversed(list(_fields().items())))
    b = build_attestation(WORKER, nonce="n1", **reordered)
    assert a.signature == b.signature  # order-independent
    c = build_attestation(WORKER, nonce="n1", **_fields(patch_digest="sha256:other"))
    assert a.signature != c.signature  # content-sensitive


def test_att_replay_rejected_by_nonce_store():
    store = NonceStore()
    att = build_attestation(WORKER, nonce="single-use", **_fields())
    first_ok, _ = verify_attestation(att, trusted_signers=TRUSTED, nonce_store=store)
    assert first_ok is True
    # an intact, validly-signed, trusted attestation replayed -> rejected
    second_ok, reason = verify_attestation(att, trusted_signers=TRUSTED, nonce_store=store)
    assert second_ok is False
    assert "replay" in reason or "nonce" in reason
