"""GitHubWriter tests (A–Z test plan: WRITER-01/02/04/07, ATT-REPLAY at the writer, WRITER-REF
branch-injection safety, TOCTOU base revalidation). Offline: create_pr + head_resolver stubbed."""
from __future__ import annotations

import json
import re

import pytest

from codna.evidence import HmacSigner, build_attestation
from codna.findings import FindingKind, NormalizedFinding, Severity, digest_of
from codna.sarif import ingest_sarif
from codna.secure import Patch
from codna.writer import GitHubWriter, WriterError

SIGNER = HmacSigner(b"worker-secret", key_id="worker-key")
ATTACKER = HmacSigner(b"attacker", key_id="attacker-key")
TRUSTED = {SIGNER.key_id: SIGNER}
COMMIT = "a" * 40
DIFF = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x\n+y\n"


def _finding():
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0",
                                "rules": [{"id": "js/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}]}},
            "versionControlProvenance": [{"revisionId": COMMIT, "repositoryUri": "https://x/y"}],
            "results": [{"ruleId": "js/sqli", "message": {"text": "x"},
                         "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.py"}, "region": {"startLine": 1}}}]}],
        }],
    }
    ing = ingest_sarif(json.dumps(doc))
    return ing, ing.findings[0]


def _attestation(signer=SIGNER, nonce="n1", diff=DIFF, original_commit=COMMIT):
    return build_attestation(
        signer, nonce=nonce, original_commit=original_commit, patch_digest=digest_of(diff),
        resulting_tree="sha256:tree", scanner_outputs={}, proof_results={"classification": "exploitable"},
        test_logs="", mojo_verdict={"ok": True}, policy_decision={"eligible": True},
    )


class StubCreatePR:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return f"https://github.com/o/r/pull/{len(self.calls)}"


def _writer(create_pr=None, head=COMMIT, **kw):
    return GitHubWriter(
        trusted_signers=TRUSTED,
        create_pr=create_pr or StubCreatePR(),
        head_resolver=lambda base: head,
        **kw,
    )


def test_happy_path_opens_draft_pr():
    ing, finding = _finding()
    create = StubCreatePR()
    w = _writer(create_pr=create)
    assert w.base_unchanged(ing) is True
    url = w.open_draft_pr(finding, Patch(digest_of(DIFF), diff=DIFF), _attestation())
    assert url.endswith("/pull/1")
    assert len(create.calls) == 1
    call = create.calls[0]
    assert call["draft"] is True
    assert call["base"] == "main"
    assert call["branch"].startswith("codna/secure/")
    assert call["patch_diff"] == DIFF


def test_base_changed_aborts():
    _, finding = _finding()
    create = StubCreatePR()
    w = _writer(create_pr=create, head="b" * 40)  # head moved
    with pytest.raises(WriterError, match="base commit changed"):
        w.open_draft_pr(finding, Patch(digest_of(DIFF), diff=DIFF), _attestation())
    assert create.calls == []  # no PR opened


def test_patch_digest_mismatch_aborts():
    _, finding = _finding()
    create = StubCreatePR()
    w = _writer(create_pr=create)
    tampered = Patch(digest_of(DIFF), diff=DIFF + "\n+evil()\n")  # diff changed after attestation
    with pytest.raises(WriterError, match="patch digest"):
        w.open_draft_pr(finding, tampered, _attestation())
    assert create.calls == []


def test_untrusted_signer_rejected():
    _, finding = _finding()
    create = StubCreatePR()
    w = _writer(create_pr=create)
    with pytest.raises(WriterError, match="attestation rejected"):
        w.open_draft_pr(finding, Patch(digest_of(DIFF), diff=DIFF), _attestation(signer=ATTACKER))
    assert create.calls == []


def test_replayed_attestation_rejected():
    _, finding = _finding()
    create = StubCreatePR()
    w = _writer(create_pr=create)
    att = _attestation(nonce="single-use")
    w.open_draft_pr(finding, Patch(digest_of(DIFF), diff=DIFF), att)
    with pytest.raises(WriterError):  # nonce already consumed
        w.open_draft_pr(finding, Patch(digest_of(DIFF), diff=DIFF), att)
    assert len(create.calls) == 1


def test_branch_name_is_injection_safe():
    create = StubCreatePR()
    w = _writer(create_pr=create)
    evil = NormalizedFinding(
        rule_id="js/sqli", message="", scanner="codeql", finding_kind=FindingKind.TAINT,
        severity=Severity.CRITICAL, original_severity="9.1", severity_rationale="x",
        primary_location=None, canonical_id="sha256:../../evil branch;rm -rf /",
    )
    w.open_draft_pr(evil, Patch(digest_of(DIFF), diff=DIFF), _attestation())
    branch = create.calls[0]["branch"]
    assert re.fullmatch(r"codna/secure/[A-Za-z0-9._-]+", branch), branch
    assert " " not in branch and ".." not in branch and ";" not in branch
