"""Verification-manifest tests (A–Z test plan: MANIFEST-TAMPER, PROV-02, G1)."""
from __future__ import annotations


import re

import pytest

from codna.manifest import ManifestError, VerificationManifest


def _manifest_dict():
    return {
        "scanner": {
            "id": "semgrep",
            "image": "registry.example/semgrep@sha256:" + "a" * 64,
            "command": ["semgrep", "--config", "p/ci", "--sarif", "--output", "results.sarif"],
            "accepted_exit_codes": [0, 1],
            "output": "results.sarif",
            "rules_digest": "sha256:" + "b" * 64,
        },
        "verification": {
            "build": [["npm", "run", "build"]],
            "tests": [["npm", "test", "--", "--runInBand"]],
        },
        "sandbox": {"network": "deny", "timeout_seconds": 1800, "cpu_limit": 4, "memory_mb": 8192},
        "policy": {"autofix_classifications": ["exploitable"], "block_new_severities": ["high", "critical"]},
    }


def test_from_dict_builds_and_digest_stable():
    m1 = VerificationManifest.from_dict(_manifest_dict())
    m2 = VerificationManifest.from_dict(_manifest_dict())
    assert m1.digest == m2.digest
    assert m1.digest.startswith("sha256:")
    assert m1.scanner.accepted_exit_codes == (0, 1)


def test_manifest_tamper_exit_codes_changes_digest():
    base = VerificationManifest.from_dict(_manifest_dict()).digest
    d = _manifest_dict()
    d["scanner"]["accepted_exit_codes"] = [0, 1, 2]  # smuggle a "success"
    assert VerificationManifest.from_dict(d).digest != base


def test_manifest_tamper_sandbox_limits_changes_digest():
    base = VerificationManifest.from_dict(_manifest_dict()).digest
    for field, val in (("timeout_seconds", 99999), ("cpu_limit", 64), ("memory_mb", 1)):
        d = _manifest_dict()
        d["sandbox"][field] = val
        assert VerificationManifest.from_dict(d).digest != base, field


def test_require_resolved_for_pr_passes_when_pinned():
    VerificationManifest.from_dict(_manifest_dict()).require_resolved_for_pr()  # no raise


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["scanner"].__setitem__("image", "registry.example/semgrep:latest"),  # floating
        lambda d: d["scanner"].__setitem__("rules_digest", None),  # unpinned rules
        lambda d: d["sandbox"].__setitem__("network", "host"),  # egress open
        lambda d: d["verification"].__setitem__("tests", []),  # no tests
    ],
)
def test_require_resolved_for_pr_rejects_unpinned(mutate):
    d = _manifest_dict()
    mutate(d)
    with pytest.raises(ManifestError):
        VerificationManifest.from_dict(d).require_resolved_for_pr()


def test_image_pinned_detection():
    d = _manifest_dict()
    assert VerificationManifest.from_dict(d).scanner.image_pinned is True
    d["scanner"]["image"] = "semgrep:1.2.3"
    assert VerificationManifest.from_dict(d).scanner.image_pinned is False


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda d: d["scanner"].__setitem__("command", ["semgrep", False]), "scanner.command[1]"),
        (lambda d: d["verification"].__setitem__("build", [["npm", True]]), "verification.build[0][1]"),
        (lambda d: d["verification"].__setitem__("tests", [[False]]), "verification.tests[0][0]"),
        (lambda d: d["scanner"].__setitem__("accepted_exit_codes", [0, True]), "scanner.accepted_exit_codes[1]"),
    ],
)
def test_manifest_rejects_non_string_commands_and_bool_exit_codes(mutate, expected):
    d = _manifest_dict()
    mutate(d)
    with pytest.raises(ManifestError, match=re.escape(expected)):
        VerificationManifest.from_dict(d)
