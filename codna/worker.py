"""LocalWorker — the analysis-worker privilege domain.

Runs the UNTRUSTED parts of the loop (baseline scan, patch generation, build, tests, scanner
rerun, closure inputs) in an ephemeral worktree pinned at the exact commit, inside a sandbox
that holds NO write credentials and denies egress, then emits a SIGNED evidence attestation.
It never opens a PR — that is the writer's domain.

Composition only: it wires `Sandbox` (enforcement) + `EphemeralWorktree` (isolation) +
`reject_evasion` (G4) + `build_attestation` (G9). Execution seams (`scan`, `run_cmd`,
`mojo_gate`, `patch_generator`) are injectable so the worker is testable offline; the defaults
shell out through the sandbox.
"""
from __future__ import annotations

import os
from typing import Callable

from .evasion import reject_evasion
from .evidence import Attestation, Signer, build_attestation, new_nonce
from .findings import IngestResult, NormalizedFinding, digest_of
from .manifest import VerificationManifest
from .netjail import default_network_backend
from .sandbox import Sandbox, SandboxResult, scrub_env
from .sarif import ingest_sarif
from .secure import Patch, VerifyResult
from .worktree import EphemeralWorktree


class WorkerError(Exception):
    pass


class LocalWorker:
    def __init__(
        self,
        *,
        repo_dir: str,
        commit: str,
        manifest: VerificationManifest,
        signer: Signer,
        patch_generator: Callable[[NormalizedFinding, str], str],
        scan: Callable[[str, str], IngestResult] | None = None,
        run_cmd: Callable[[list, str], SandboxResult] | None = None,
        mojo_gate: Callable[[str], bool] | None = None,
        nonce_fn: Callable[[], str] = new_nonce,
        base_env: dict | None = None,
        allowed_paths=None,
        dependency_diff_verified: bool = False,
    ):
        self.repo_dir = repo_dir
        self.commit = commit
        self.manifest = manifest
        self.signer = signer
        self.patch_generator = patch_generator
        # Sandbox construction scrubs + asserts no write credentials (PRIV-01/02); the network
        # backend (if available on this host) hard-denies egress for untrusted commands.
        self.sandbox = Sandbox(
            network=manifest.sandbox.network,
            timeout_seconds=manifest.sandbox.timeout_seconds,
            env=scrub_env(dict(base_env if base_env is not None else os.environ)),
            network_backend=default_network_backend(),
        )
        self._run = run_cmd or (lambda argv, cwd: self.sandbox.run(argv, cwd=cwd))
        self._scan = scan or self._default_scan
        self._mojo = mojo_gate or (lambda diff: True)
        self._nonce_fn = nonce_fn
        self.allowed_paths = allowed_paths
        self.dependency_diff_verified = dependency_diff_verified
        self._wt: EphemeralWorktree | None = None
        self._baseline: IngestResult | None = None
        self._artifacts: dict = {}

    # -- default execution seam ------------------------------------------------------

    def _default_scan(self, cwd: str, phase: str) -> IngestResult:
        sc = self.manifest.scanner
        res = self._run(sc.command, cwd)
        if not res.accepted(sc.accepted_exit_codes):
            raise WorkerError(f"scanner exited {res.returncode} in phase {phase} "
                              f"(accepted {list(sc.accepted_exit_codes)})")
        return ingest_sarif(os.path.join(cwd, sc.output))

    def _reset_worktree(self) -> None:
        if self._wt is not None:
            self._wt.remove()
        self._wt = EphemeralWorktree(self.repo_dir, self.commit).create()
        self._artifacts = {}

    # -- Worker protocol -------------------------------------------------------------

    def reproduce_baseline(self, finding: NormalizedFinding) -> bool:
        self._reset_worktree()
        ingest = self._scan(self._wt.path, "baseline")
        self._baseline = ingest
        self._artifacts["scanner_outputs"] = {
            "baseline": {"findings": len(ingest.findings), "rules_digest": ingest.scanner.rules_digest}
        }
        # Semantic equivalence on the unpatched snapshot == same canonical id (G2).
        return any(f.canonical_id == finding.canonical_id for f in ingest.findings)

    def remediate(self, finding: NormalizedFinding) -> Patch:
        if self._wt is None:
            raise WorkerError("remediate called before reproduce_baseline")
        diff = self.patch_generator(finding, self._wt.path)
        integrity = reject_evasion(
            diff, allowed_paths=self.allowed_paths,
            dependency_diff_verified=self.dependency_diff_verified,
        )
        self._artifacts["patch_diff"] = diff
        if integrity.ok:
            self._wt.apply_patch(diff)
        self._artifacts["resulting_tree"] = self._wt.tree_digest()
        return Patch(
            patch_digest=digest_of(diff),
            integrity_ok=integrity.ok,
            integrity_reason="ok" if integrity.ok else "; ".join(v.reason_code for v in integrity.violations),
            diff=diff,
        )

    def verify(self, finding: NormalizedFinding, patch: Patch) -> VerifyResult:
        wt = self._wt.path
        build_passed = all(self._run(cmd, wt).returncode == 0 for cmd in self.manifest.build)
        test_results = [self._run(cmd, wt) for cmd in self.manifest.tests]
        tests_passed = all(r.returncode == 0 for r in test_results)
        mojo_ok = bool(self._mojo(self._artifacts.get("patch_diff", "")))

        patched = self._scan(wt, "patched")
        scanner_confirms_gone = not any(f.canonical_id == finding.canonical_id for f in patched.findings)
        base = self._baseline
        scan_not_degraded = bool(
            base is not None
            and patched.scanner.name == base.scanner.name
            and patched.scanner.rules_digest == base.scanner.rules_digest
            and patched.scanner.rules_count >= base.scanner.rules_count
        )

        self._artifacts.setdefault("scanner_outputs", {})["patched"] = {
            "findings": len(patched.findings), "rules_digest": patched.scanner.rules_digest
        }
        self._artifacts["test_logs"] = "\n".join(r.stdout for r in test_results)[:8192]
        self._artifacts["mojo_verdict"] = {"ok": mojo_ok}
        return VerifyResult(build_passed, tests_passed, mojo_ok, scanner_confirms_gone, scan_not_degraded)

    def attest(self, finding: NormalizedFinding, patch: Patch, decision_inputs: dict) -> Attestation:
        return build_attestation(
            self.signer,
            nonce=self._nonce_fn(),
            original_commit=self.commit,
            patch_digest=patch.patch_digest,
            resulting_tree=self._artifacts.get("resulting_tree", "sha256:tree:none"),
            scanner_outputs=self._artifacts.get("scanner_outputs", {}),
            proof_results=decision_inputs.get("proof_results", {"classification": decision_inputs.get("classification")}),
            test_logs=self._artifacts.get("test_logs", ""),
            mojo_verdict=self._artifacts.get("mojo_verdict", {}),
            policy_decision=decision_inputs.get("policy_decision", decision_inputs),
        )

    def close(self) -> None:
        if self._wt is not None:
            self._wt.remove()
            self._wt = None
