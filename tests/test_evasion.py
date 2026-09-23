"""Anti-evasion / patch-integrity tests (A–Z test plan: G4, PI-EV-*, PI-SCA-*, PI-SCOPE-*,
G4-HOOK-INJECT)."""
from __future__ import annotations

import posixpath

import pytest

from codna.evasion import _PATH_CATEGORIES, reject_evasion


def _diff(path, added="return safe(u)"):
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n context\n+{added}\n"
    )


def _matches(pattern, path):
    """Mirror `_categorize`: a pattern hits on the full path OR on the basename."""
    return bool(pattern.search(path) or pattern.search(posixpath.basename(path)))


# thyn-ai/security-toolchain files: rule packs / pins (scanner config) and allowlists /
# ratchet baselines (suppressions). Kept separate so the scope tests below can iterate them.
TOOLCHAIN_PROTECTED = [
    (".gitleaks.toml", "scanner_config_or_rulepack"),
    ("security/opengrep/no-shell-true.yml", "scanner_config_or_rulepack"),
    ("security/toolchain.lock", "scanner_config_or_rulepack"),
    (".gitleaksignore", "suppression_or_baseline"),
    ("osv-scanner.toml", "suppression_or_baseline"),
    ("security/baseline/opengrep.txt", "suppression_or_baseline"),
]

# One oracle per forbidden path pattern in `_PATH_CATEGORIES` (asserted below, so a pattern
# can never be added without a case that proves it fires).
FORBIDDEN_CASES = [
    (".semgrep.yml", "scanner_config_or_rulepack"),
    (".semgrep/rules.yml", "scanner_config_or_rulepack"),
    (".github/codeql/config.yml", "scanner_config_or_rulepack"),
    ("codeql-config.yml", "scanner_config_or_rulepack"),
    (".snyk", "scanner_config_or_rulepack"),
    ("trivy.yaml", "scanner_config_or_rulepack"),
    (".qlpack.yml", "scanner_config_or_rulepack"),
    (".semgrepignore", "suppression_or_baseline"),
    (".codeqlignore", "suppression_or_baseline"),
    (".trivyignore", "suppression_or_baseline"),
    ("baseline-2024.sarif", "suppression_or_baseline"),
    (".sarif-baseline", "suppression_or_baseline"),
    (".github/workflows/ci.yml", "ci_or_test_config"),
    (".gitlab-ci.yml", "ci_or_test_config"),
    ("azure-pipelines.yml", "ci_or_test_config"),
    ("pytest.ini", "ci_or_test_config"),
    ("tox.ini", "ci_or_test_config"),
    ("conftest.py", "ci_or_test_config"),
    ("jest.config.ts", "ci_or_test_config"),
    ("codna-security.yaml", "codna_policy_or_model"),
    (".codna/policy.yml", "codna_policy_or_model"),
    ("codna/evidence.py", "evidence_generation"),
    ("codna/gate.py", "evidence_generation"),
    (".git/hooks/pre-push", "executable_hook"),
    (".pre-commit-config.yaml", "executable_hook"),
    ("Makefile", "executable_hook"),
    (".husky/pre-commit", "executable_hook"),
] + TOOLCHAIN_PROTECTED


def test_clean_source_only_patch_allowed():
    res = reject_evasion(_diff("src/app.py"))
    assert res.ok is True and res.violations == []
    assert res.changed_paths == ["src/app.py"]


@pytest.mark.parametrize("path,reason", FORBIDDEN_CASES)
def test_forbidden_file_categories_rejected(path, reason):
    res = reject_evasion(_diff(path))
    assert res.ok is False
    assert any(v.reason_code == reason for v in res.violations), res.violations


def test_every_forbidden_category_has_an_oracle():
    assert {reason for reason, _ in _PATH_CATEGORIES} == {reason for _, reason in FORBIDDEN_CASES}


def test_every_forbidden_pattern_has_an_oracle():
    # Derived from the source of truth: every regex in `_PATH_CATEGORIES` must be hit by at least
    # one case, whatever the module currently declares -- never a hardcoded count.
    uncovered = [
        (reason, pat.pattern)
        for reason, patterns in _PATH_CATEGORIES
        for pat in patterns
        if not any(_matches(pat, path) for path, _ in FORBIDDEN_CASES)
    ]
    assert uncovered == []


@pytest.mark.parametrize("path,reason", TOOLCHAIN_PROTECTED)
def test_toolchain_files_rejected_at_root_nested_and_in_scope(path, reason):
    # Categorization precedes the scope check: even a candidate whose allow-list names the very
    # file is rejected, at the repository root and in a nested monorepo package alike.
    for candidate in (path, f"packages/api/{path}"):
        res = reject_evasion(_diff(candidate), allowed_paths=[candidate])
        assert res.ok is False, candidate
        assert [v.reason_code for v in res.violations] == [reason], res.violations
    # Only an explicit operator authorization lets it through, and that is recorded, never silent.
    authorized = reject_evasion(_diff(path), policy_authorized=[reason], policy_digest="sha256:pol")
    assert authorized.ok is True
    assert authorized.authorization_used == [f"{reason}@sha256:pol"]


@pytest.mark.parametrize("token", [
    "nosemgrep", "# nosec", "NOSONAR", "checkov:skip=CKV_AWS_1",
    "eslint-disable-next-line", "trivy:ignore:AVD-AWS-0089", "snyk:ignore",
    "lgtm[py/sql-injection]", "gitleaks:allow",
])
def test_inline_suppressions_rejected(token):
    res = reject_evasion(_diff("src/app.py", added=f"x = 1  # {token}"))
    assert res.ok is False
    assert any(v.reason_code == "inline_suppression" for v in res.violations)


def test_lockfile_requires_dependency_diff_verification():
    unverified = reject_evasion(_diff("package-lock.json", added='"lodash": "4.17.21"'))
    assert unverified.ok is False
    assert any(v.reason_code == "lockfile_unverified" for v in unverified.violations)

    verified = reject_evasion(_diff("package-lock.json"), dependency_diff_verified=True)
    assert verified.ok is True


def test_change_scope_enforced():
    res = reject_evasion(_diff("infra/deploy.tf"), allowed_paths=["src/"])
    assert res.ok is False
    assert any(v.reason_code == "change_scope_exceeded" for v in res.violations)
    # in-scope edit passes under the same allow-list
    assert reject_evasion(_diff("src/app.py"), allowed_paths=["src/"]).ok is True


def test_explicit_policy_authorization_is_recorded_not_silent():
    res = reject_evasion(
        _diff(".semgrep.yml"),
        policy_authorized=["scanner_config_or_rulepack"],
        policy_digest="sha256:pol",
    )
    assert res.ok is True
    assert res.authorization_used == ["scanner_config_or_rulepack@sha256:pol"]
