"""Evidence (de)serialization tests — the worker→writer artifact handoff (G9, two-job Action)."""
from __future__ import annotations

import pytest

from codna.cli import main as cli_main
from codna.evidence import (
    HmacSigner,
    attestation_from_dict,
    build_attestation,
    read_evidence,
    verify_attestation,
    write_evidence,
)
from codna.secure_open_pr_cli import SecureOpenPrCliError, evidence_bundle_dirs
from codna.findings import FindingKind, NormalizedFinding, Severity

SIGNER = HmacSigner(b"k", key_id="worker-key")
TRUSTED = {SIGNER.key_id: SIGNER}
DIFF = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"


def _att(nonce="n1"):
    return build_attestation(
        SIGNER, nonce=nonce, original_commit="a" * 40, patch_digest="sha256:p",
        resulting_tree="sha256:t", scanner_outputs={"baseline": 1}, proof_results={"c": "exploitable"},
        test_logs="ok", mojo_verdict={"ok": True}, policy_decision={"eligible": True},
    )


def _finding():
    f = NormalizedFinding(
        rule_id="js/sqli", message="", scanner="codeql", finding_kind=FindingKind.TAINT,
        severity=Severity.CRITICAL, original_severity="9.1", severity_rationale="x", primary_location=None,
    )
    f.canonical_id = "sha256:abc123"
    return f


def test_attestation_dict_round_trip_preserves_signature():
    att = _att()
    back = attestation_from_dict(att.to_dict())
    assert back.payload == att.payload and back.signature == att.signature
    assert verify_attestation(back, trusted_signers=TRUSTED)[0] is True


def test_write_and_read_evidence_round_trip(tmp_path):
    att = _att()
    write_evidence(str(tmp_path), attestation=att, patch_diff=DIFF, finding=_finding())
    back_att, back_diff, desc = read_evidence(str(tmp_path))
    assert back_diff == DIFF
    assert desc == {"canonical_id": "sha256:abc123", "rule_id": "js/sqli", "finding_kind": "taint"}
    # the reconstructed attestation still verifies against the worker's signer
    assert verify_attestation(back_att, trusted_signers=TRUSTED)[0] is True


def test_minimal_finding_reconstruction():
    f = NormalizedFinding.minimal(canonical_id="sha256:zzz", rule_id="r", finding_kind="sca")
    assert f.canonical_id == "sha256:zzz" and f.rule_id == "r"
    assert f.finding_kind is FindingKind.SCA


def test_indexed_evidence_bundle_dirs_are_resolved_safely(tmp_path):
    (tmp_path / "001-a").mkdir()
    (tmp_path / "002-b").mkdir()
    (tmp_path / "evidence-index.json").write_text(
        '{"schema_version":1,"bundles":[{"path":"001-a"},{"path":"002-b"}]}',
        encoding="utf-8",
    )

    assert evidence_bundle_dirs(str(tmp_path)) == [tmp_path / "001-a", tmp_path / "002-b"]


def test_indexed_evidence_rejects_path_escape(tmp_path):
    (tmp_path / "evidence-index.json").write_text(
        '{"schema_version":1,"bundles":[{"path":"../escape"}]}',
        encoding="utf-8",
    )

    with pytest.raises(SecureOpenPrCliError, match="unsafe bundle path"):
        evidence_bundle_dirs(str(tmp_path))


def test_secure_open_pr_empty_index_noops_before_credentials(tmp_path, monkeypatch, capsys):
    (tmp_path / "evidence-index.json").write_text(
        '{"schema_version":1,"bundles":[]}',
        encoding="utf-8",
    )
    monkeypatch.delenv("CODNA_ATTESTATION_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    rc = cli_main(["secure-open-pr", "--evidence", str(tmp_path), "--repo-slug", "o/r"])

    assert rc == 0
    assert "no evidence bundles" in capsys.readouterr().out


def test_secure_open_pr_empty_evidence_dir_noops_before_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CODNA_ATTESTATION_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    rc = cli_main(["secure-open-pr", "--evidence", str(tmp_path), "--repo-slug", "o/r"])

    assert rc == 0
    assert "no evidence bundles" in capsys.readouterr().out
