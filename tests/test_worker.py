"""LocalWorker composition tests (A–Z test plan: privilege separation, baseline reproduction,
G4 patch integrity, G6 scanner confirmation + degraded-scan, attestation). Real worktree, no
network; scanner/build runners injected."""
from __future__ import annotations

import json
import os
import subprocess


from codna.evidence import HmacSigner, verify_attestation
from codna.manifest import VerificationManifest
from codna.sandbox import SandboxResult
from codna.sarif import ingest_sarif
from codna.worker import LocalWorker

SIGNER = HmacSigner(b"worker-secret", key_id="worker-key")
TRUSTED = {SIGNER.key_id: SIGNER}

MANIFEST = VerificationManifest.from_dict({
    "scanner": {"id": "codeql", "image": "r/codeql@sha256:" + "a" * 64, "command": ["codeql"],
                "output": "results.sarif", "accepted_exit_codes": [0], "rules_digest": "sha256:" + "b" * 64},
    "verification": {"build": [["build"]], "tests": [["test"]]},
    "sandbox": {"network": "deny"},
    "policy": {},
})


# --------------------------------------------------------------- SARIF / ingests

def _doc(with_result=True, n_rules=1):
    rules = [{"id": "js/sqli", "properties": {"security-severity": "9.1", "tags": ["security"]}}] if n_rules else []
    results = [{
        "ruleId": "js/sqli", "message": {"text": "sqli"},
        "locations": [{"physicalLocation": {"artifactLocation": {"uri": "src/app.py"}, "region": {"startLine": 2}}}],
        "codeFlows": [{"threadFlows": [{"locations": [
            {"location": {"physicalLocation": {"artifactLocation": {"uri": "src/in.py"}, "region": {"startLine": 1}}}}]}]}],
    }] if with_result else []
    return json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "CodeQL", "version": "2.15.0", "rules": rules}},
            "versionControlProvenance": [{"revisionId": "a" * 40, "repositoryUri": "https://x/y"}],
            "results": results,
        }],
    })


FINDING = ingest_sarif(_doc(True)).findings[0]
BASELINE = ingest_sarif(_doc(True))
CLEAN_PATCHED = ingest_sarif(_doc(False))          # finding gone, rules unchanged
STILL_REPORTS = ingest_sarif(_doc(True))           # finding still present
DEGRADED_PATCHED = ingest_sarif(_doc(False, n_rules=0))  # finding gone but rules dropped


# --------------------------------------------------------------- helpers / stubs

def _run(argv, cwd):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True)


def _init_repo(tmp_path):
    d = tmp_path / "repo"
    (d / "src").mkdir(parents=True)
    (d / "src" / "app.py").write_text('def q(u):\n    return "X=" + u\n')
    _run(["git", "init", "-q"], d)
    _run(["git", "config", "user.email", "a@b.c"], d)
    _run(["git", "config", "user.name", "t"], d)
    _run(["git", "add", "-A"], d)
    _run(["git", "commit", "-q", "-m", "init"], d)
    sha = _run(["git", "rev-parse", "HEAD"], d).stdout.strip()
    return str(d), sha


class Scan:
    def __init__(self, baseline, patched):
        self.baseline, self.patched = baseline, patched

    def __call__(self, cwd, phase):
        return self.baseline if phase == "baseline" else self.patched


class FakeRun:
    def __init__(self, rc=0):
        self.rc = rc

    def __call__(self, argv, cwd):
        return SandboxResult(self.rc, "out", "", False, list(argv), cwd, "deny")


def gen_happy(finding, cwd):
    p = os.path.join(cwd, "src", "app.py")
    original = open(p).read()
    open(p, "w").write(original.replace('"X=" + u', "safe(u)"))
    diff = _run(["git", "diff"], cwd).stdout
    _run(["git", "checkout", "--", "."], cwd)
    return diff


def gen_evasion(finding, cwd):
    path = ".github/workflows/ci.yml"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n on: push\n+    # tampered\n"


def _worker(tmp_path, *, scan=None, gen=gen_happy, run=None, base_env=None, mojo=None):
    repo, sha = _init_repo(tmp_path)
    return LocalWorker(
        repo_dir=repo, commit=sha, manifest=MANIFEST, signer=SIGNER,
        patch_generator=gen, scan=scan or Scan(BASELINE, CLEAN_PATCHED),
        run_cmd=run or FakeRun(0), mojo_gate=mojo or (lambda d: True),
        base_env=base_env if base_env is not None else {},
    )


# --------------------------------------------------------------------- tests

def test_happy_path_full_flow(tmp_path):
    w = _worker(tmp_path)
    try:
        assert w.reproduce_baseline(FINDING) is True
        patch = w.remediate(FINDING)
        assert patch.integrity_ok is True and patch.patch_digest.startswith("sha256:")
        assert "safe(u)" in open(os.path.join(w._wt.path, "src", "app.py")).read()

        vr = w.verify(FINDING, patch)
        assert (vr.build_passed, vr.tests_passed, vr.mojo_ok,
                vr.scanner_confirms_gone, vr.scan_not_degraded) == (True, True, True, True, True)

        att = w.attest(FINDING, patch, {"classification": "exploitable"})
        ok, reason = verify_attestation(att, trusted_signers=TRUSTED)
        assert ok is True, reason
        assert att.payload["original_commit"] == w.commit
    finally:
        w.close()
    assert w._wt is None


def test_worker_scrubs_write_token_from_sandbox_env(tmp_path):
    # The ambient env carries a write token (as CI does); the worker must scrub it so the
    # sandbox that runs untrusted code never sees it (PRIV-01).
    w = _worker(tmp_path, base_env={"GITHUB_TOKEN": "ghp_" + "a" * 36, "PATH": "/usr/bin:/bin"})
    try:
        assert "GITHUB_TOKEN" not in w.sandbox.env
        assert all("ghp_" not in str(v) for v in w.sandbox.env.values())
    finally:
        w.close()


def test_evasion_patch_is_not_applied(tmp_path):
    w = _worker(tmp_path, gen=gen_evasion)
    try:
        w.reproduce_baseline(FINDING)
        patch = w.remediate(FINDING)
        assert patch.integrity_ok is False
        # the worktree source is untouched (the evasion diff was refused)
        assert "safe(u)" not in open(os.path.join(w._wt.path, "src", "app.py")).read()
    finally:
        w.close()


def test_scanner_still_reports_fails_confirmation(tmp_path):
    w = _worker(tmp_path, scan=Scan(BASELINE, STILL_REPORTS))
    try:
        w.reproduce_baseline(FINDING)
        vr = w.verify(FINDING, w.remediate(FINDING))
        assert vr.scanner_confirms_gone is False
    finally:
        w.close()


def test_degraded_scan_detected(tmp_path):
    w = _worker(tmp_path, scan=Scan(BASELINE, DEGRADED_PATCHED))
    try:
        w.reproduce_baseline(FINDING)
        vr = w.verify(FINDING, w.remediate(FINDING))
        assert vr.scanner_confirms_gone is True  # finding gone...
        assert vr.scan_not_degraded is False     # ...but coverage shrank, so G6 must fail
    finally:
        w.close()


def test_build_and_test_failure_surfaced(tmp_path):
    w = _worker(tmp_path, run=FakeRun(1))
    try:
        w.reproduce_baseline(FINDING)
        vr = w.verify(FINDING, w.remediate(FINDING))
        assert vr.build_passed is False and vr.tests_passed is False
    finally:
        w.close()
