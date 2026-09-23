"""Anti-evasion patch integrity (gate condition G4).

A generated fix must remediate the vulnerability, not "fix" the finding by WEAKENING the
verification system. `reject_evasion` inspects a unified diff and rejects (unless explicitly
policy-authorized) any patch that touches scanner config/rule packs, suppression/baseline
files, inline suppression comments, test selection/CI verification steps, codna policy/model
config, evidence-generation code, or git hooks / build-script targets. Lockfile changes stay
allowed for legitimate SCA remediation but only when a dependency-diff verification has run.
Edits outside the permitted change scope are rejected too.

Pure stdlib — operates on the diff text. Heavily exercised by PI-EV-*, PI-SCA-*, PI-SCOPE-*,
and G4-HOOK-INJECT.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

# --- forbidden file categories (path patterns) -------------------------------------
# Each entry: (reason_code, list of regexes matched against the changed path).
_PATH_CATEGORIES: list[tuple[str, list[re.Pattern]]] = [
    ("scanner_config_or_rulepack", [
        re.compile(r"(^|/)\.?semgrep(\.ya?ml)?$"),
        re.compile(r"(^|/)\.semgrep/"),
        re.compile(r"(^|/)\.github/codeql/"),
        re.compile(r"(^|/)codeql-config\.ya?ml$"),
        re.compile(r"(^|/)\.snyk$"),
        re.compile(r"(^|/)trivy\.ya?ml$"),
        re.compile(r"\.qlpack\.ya?ml$"),
        # thyn-ai/security-toolchain: gitleaks rules + path allowlist, repo-local Opengrep rule
        # packs, and the pinned scanner versions + sha256 digests every binary is verified against.
        re.compile(r"(^|/)\.gitleaks\.toml$"),
        re.compile(r"(^|/)security/opengrep/"),
        re.compile(r"(^|/)security/toolchain\.lock$"),
    ]),
    ("suppression_or_baseline", [
        re.compile(r"(^|/)\.semgrepignore$"),
        re.compile(r"(^|/)\.codeqlignore$"),
        re.compile(r"(^|/)\.trivyignore$"),
        re.compile(r"(^|/)\.snyk$"),
        re.compile(r"baseline.*\.sarif$"),
        re.compile(r"(^|/)\.sarif-baseline"),
        # thyn-ai/security-toolchain: gitleaks fingerprint allowlist, osv-scanner ignore/override
        # config, and the CI-measured `security/baseline/<tool>.txt` ratchet files.
        re.compile(r"(^|/)\.gitleaksignore$"),
        re.compile(r"(^|/)osv-scanner\.toml$"),
        re.compile(r"(^|/)security/baseline/"),
    ]),
    ("ci_or_test_config", [
        re.compile(r"(^|/)\.github/workflows/"),
        re.compile(r"(^|/)\.gitlab-ci\.ya?ml$"),
        re.compile(r"(^|/)azure-pipelines\.ya?ml$"),
        re.compile(r"(^|/)pytest\.ini$"),
        re.compile(r"(^|/)tox\.ini$"),
        re.compile(r"(^|/)conftest\.py$"),
        re.compile(r"(^|/)jest\.config\.[jt]s$"),
    ]),
    ("codna_policy_or_model", [
        re.compile(r"(^|/)codna-security\.ya?ml$"),
        re.compile(r"(^|/)\.codna/"),
    ]),
    ("evidence_generation", [
        re.compile(r"(^|/)codna/evidence\.py$"),
        re.compile(r"(^|/)codna/(gate|policy|manifest)\.py$"),
    ]),
    ("executable_hook", [
        re.compile(r"(^|/)\.git/hooks/"),
        re.compile(r"(^|/)\.pre-commit-config\.ya?ml$"),
        re.compile(r"(^|/)Makefile$"),
        re.compile(r"(^|/)\.husky/"),
    ]),
]

_LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "go.sum", "Cargo.lock", "Gemfile.lock", "composer.lock",
}

# Inline suppression directives across scanner dialects (matched on ADDED lines only).
# No surrounding \b anchors — several directives start/end with non-word chars (#, [, :).
_INLINE_SUPPRESSIONS = re.compile(
    r"(nosemgrep|nosem|lgtm\s*\[|#\s*nosec|NOSONAR|checkov:skip|eslint-disable"
    r"|trivy:ignore|snyk:ignore|gitleaks:allow|//\s*codeql[^\n]*\bignore)",
    re.IGNORECASE,
)


@dataclass
class IntegrityViolation:
    path: str
    reason_code: str
    detail: str


@dataclass
class IntegrityResult:
    ok: bool
    violations: list[IntegrityViolation] = field(default_factory=list)
    authorization_used: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)


def _changed_paths(diff: str) -> list[str]:
    paths: list[str] = []
    for m in re.finditer(r"^\+\+\+ [ab]/(.+)$", diff, re.MULTILINE):
        p = m.group(1).strip()
        if p != "/dev/null":
            paths.append(posixpath.normpath(p))
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", diff, re.MULTILINE):
        paths.append(posixpath.normpath(m.group(2)))
    # de-dup preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _added_lines(diff: str) -> list[str]:
    return [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def changed_paths(diff: str) -> list[str]:
    """Public: the file paths a unified diff touches."""
    return _changed_paths(diff)


def added_lines(diff: str) -> list[str]:
    """Public: the added (`+`) content lines of a unified diff."""
    return _added_lines(diff)


def _categorize(path: str) -> str | None:
    base = posixpath.basename(path)
    for reason, patterns in _PATH_CATEGORIES:
        for pat in patterns:
            if pat.search(path) or pat.search(base):
                return reason
    return None


def _within_scope(path: str, allowed_paths) -> bool:
    if not allowed_paths:
        return True
    for allowed in allowed_paths:
        a = posixpath.normpath(allowed).rstrip("/")
        if path == a or path.startswith(a + "/"):
            return True
    return False


def reject_evasion(
    diff: str,
    *,
    allowed_paths=None,
    dependency_diff_verified: bool = False,
    policy_authorized=(),
    policy_digest: str | None = None,
) -> IntegrityResult:
    """Return an `IntegrityResult`. `ok` is False if the patch weakens verification or escapes
    scope. `policy_authorized` lists reason_codes an operator explicitly allowed (recorded in
    `authorization_used`, never silently). Lockfile edits require `dependency_diff_verified`."""
    paths = _changed_paths(diff)
    violations: list[IntegrityViolation] = []
    authorization_used: list[str] = []
    authorized = {str(c) for c in policy_authorized}

    for path in paths:
        base = posixpath.basename(path)

        if base in _LOCKFILES:
            if not dependency_diff_verified:
                violations.append(IntegrityViolation(path, "lockfile_unverified",
                                                     "lockfile change requires dependency-diff verification"))
            continue  # a verified lockfile change is legitimate SCA remediation

        reason = _categorize(path)
        if reason:
            if reason in authorized:
                authorization_used.append(f"{reason}@{policy_digest or '?'}")
            else:
                violations.append(IntegrityViolation(path, reason, f"patch modifies {reason}"))
            continue

        if not _within_scope(path, allowed_paths):
            violations.append(IntegrityViolation(path, "change_scope_exceeded",
                                                 f"{path} is outside the permitted change scope"))

    for line in _added_lines(diff):
        m = _INLINE_SUPPRESSIONS.search(line)
        if m:
            violations.append(IntegrityViolation("(added line)", "inline_suppression",
                                                 f"adds suppression directive {m.group(0)!r}"))

    return IntegrityResult(
        ok=not violations,
        violations=violations,
        authorization_used=authorization_used,
        changed_paths=paths,
    )
