"""`run_local_fix` — the self-hosted detect→fix→verify→close loop behind
`codna secure --fix --engine local`.

No remote engine and no SAST scanner image required: reachability and closure both come from the
bounded `LocalReferenceEngine`, and the patch comes from the agentic Cline SDK
(`ClinePatchGenerator`). It is the local analogue of `run_secure` + `LocalWorker` for air-gapped
or scanner-less environments, and it keeps the same discipline:

  * ineligible findings (per policy) are reported but NEVER fixed;
  * every generated patch is subject to the G4 anti-evasion check BEFORE it is applied;
  * the patched proof tree must pass the manifest's build/test commands;
  * the original obligation must be independently re-proven `closed` after the fix; and
  * CLI fix mode applies the verified diff to the user's checkout and re-runs verification there.

It never opens PRs itself (that needs the privilege-separated writer + a scoped token, which
`--open-pr` wires through `run_secure`).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

from .evasion import reject_evasion
from .findings import IngestResult, digest_of
from .policy import Policy
from .secure import Patch, SecurityEngine
from .worktree import EphemeralWorktree


@dataclass
class TargetApplyResult:
    applied: bool
    tests_passed: bool | None = None
    reason: str = ""


@dataclass
class LocalFixOutcome:
    canonical_id: str
    rule_id: str
    classification: str
    eligible: bool
    reason: str
    fix_applied: bool = False
    integrity_ok: bool = False
    integrity_reason: str = ""
    tests_passed: bool | None = None  # None = not run (no test commands configured)
    closure: str = "n/a"
    diff: str = ""
    target_applied: bool | None = None  # None = target apply was not requested.
    target_tests_passed: bool | None = None
    target_apply_reason: str = ""

    @property
    def remediated(self) -> bool:
        """A finding is remediated only when the fix applied cleanly, passed the integrity
        check, did not regress the tests, and the obligation re-proves `closed`."""
        return (
            self.fix_applied
            and self.integrity_ok
            and self.tests_passed is not False
            and self.closure == "closed"
            and self.target_applied is not False
            and self.target_tests_passed is not False
        )


@dataclass
class LocalFixReport:
    findings_total: int
    eligible: int
    outcomes: list[LocalFixOutcome]

    @property
    def remediated(self) -> list[LocalFixOutcome]:
        return [o for o in self.outcomes if o.remediated]


def run_local_fix(
    ingest: IngestResult,
    *,
    engine: SecurityEngine,
    patch_generator,
    repo_dir: str,
    commit: str,
    policy: Policy,
    test_cmds: list[list[str]] | None = None,
    apply_to_repo: bool = False,
) -> LocalFixReport:
    """Run detect→fix→verify→close for every eligible finding.

    The agent always works against an isolated worktree pinned to the SARIF-bound commit. When
    ``apply_to_repo`` is true, the verified diff is then applied to the caller's checkout with
    ``git apply --check`` first and verification is run again on the target checkout.
    """
    outcomes: list[LocalFixOutcome] = []
    eligible = 0

    for finding in ingest.findings:
        if finding.quarantined:
            outcomes.append(LocalFixOutcome(
                finding.canonical_id, finding.rule_id, "quarantined", False, "artifact path escape"))
            continue

        verdict = engine.analyze(ingest, finding)
        is_eligible, reason = policy.is_autofix_eligible(verdict.classification)
        if not is_eligible:
            outcomes.append(LocalFixOutcome(
                finding.canonical_id, finding.rule_id, verdict.classification.value, False, reason))
            continue
        eligible += 1

        outcome = LocalFixOutcome(
            finding.canonical_id, finding.rule_id, verdict.classification.value, True, reason)
        with EphemeralWorktree(repo_dir, commit) as wt:
            diff = patch_generator(finding, wt.path)
            integrity = reject_evasion(diff)
            outcome.diff = diff
            outcome.integrity_ok = integrity.ok
            outcome.integrity_reason = (
                "ok" if integrity.ok
                else "; ".join(v.reason_code for v in integrity.violations)
            )
            if not integrity.ok:
                # Patch rejected by anti-evasion — never apply it.
                outcomes.append(outcome)
                continue

            wt.apply_patch(diff)
            outcome.fix_applied = True
            if test_cmds:
                outcome.tests_passed = _run_verification_commands(test_cmds, cwd=wt.path)
            closure = engine.reprove_closure(finding, Patch(digest_of(diff), diff=diff))
            outcome.closure = closure.closure_status.value
            if apply_to_repo and _proof_passed(outcome):
                target = _apply_verified_patch_to_repo(repo_dir, diff, test_cmds or [])
                outcome.target_applied = target.applied
                outcome.target_tests_passed = target.tests_passed
                outcome.target_apply_reason = target.reason
        outcomes.append(outcome)

    return LocalFixReport(len(ingest.findings), eligible, outcomes)


def _proof_passed(outcome: LocalFixOutcome) -> bool:
    return (
        outcome.fix_applied
        and outcome.integrity_ok
        and outcome.tests_passed is not False
        and outcome.closure == "closed"
    )


def _run_verification_commands(commands: list[list[str]], *, cwd: str) -> bool:
    return all(
        subprocess.run(_verification_command(cmd), cwd=cwd, capture_output=True).returncode == 0
        for cmd in commands
    )


def _apply_verified_patch_to_repo(repo_dir: str, diff: str, test_cmds: list[list[str]]) -> TargetApplyResult:
    check = _git_apply(repo_dir, diff, "--check")
    if check.returncode != 0:
        return TargetApplyResult(False, reason=_format_git_apply_error("target apply check failed", check))

    applied = _git_apply(repo_dir, diff)
    if applied.returncode != 0:
        return TargetApplyResult(False, reason=_format_git_apply_error("target apply failed", applied))

    if not test_cmds:
        return TargetApplyResult(True)

    if _run_verification_commands(test_cmds, cwd=repo_dir):
        return TargetApplyResult(True, tests_passed=True)

    rollback = _git_apply(repo_dir, diff, "-R")
    if rollback.returncode == 0:
        return TargetApplyResult(
            False,
            tests_passed=False,
            reason="target verification failed after apply; patch rolled back",
        )
    return TargetApplyResult(
        True,
        tests_passed=False,
        reason=_format_git_apply_error("target verification failed after apply; rollback failed", rollback),
    )


def _git_apply(repo_dir: str, diff: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_dir, "apply", "--whitespace=nowarn", *args, "-"],
        input=diff,
        capture_output=True,
        text=True,
    )


def _format_git_apply_error(prefix: str, proc: subprocess.CompletedProcess) -> str:
    detail = (proc.stderr or proc.stdout or "").strip()
    if not detail:
        return prefix
    return f"{prefix}: {detail[:500]}"


def _verification_command(cmd: list[str]) -> list[str]:
    """Run manifest `python ...` commands with the interpreter executing Codna when needed.

    macOS and minimal containers often expose only `python3`, while virtualenv console scripts run
    under an exact interpreter path. Treat bare `python` as the current interpreter if it is not on
    PATH; explicit paths, `python3`, and every other command remain operator-controlled.
    """
    if not cmd or cmd[0] != "python" or shutil.which("python"):
        return cmd
    return [sys.executable, *cmd[1:]]
