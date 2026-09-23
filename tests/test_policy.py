"""Autofix-policy tests (A–Z test plan: G3 eligibility, severity overrides, G8 block set)."""
from __future__ import annotations

import pytest

from codna.findings import Classification, Severity
from codna.policy import Policy


def test_g3_exploitable_eligible_by_default():
    ok, reason = Policy().is_autofix_eligible(Classification.EXPLOITABLE)
    assert ok is True
    assert "default" in reason


def test_g3_production_reachable_needs_explicit_override():
    ok, reason = Policy().is_autofix_eligible(Classification.PRODUCTION_REACHABLE)
    assert ok is False
    assert "explicit policy override" in reason

    ok2, reason2 = Policy(
        autofix_classifications=("exploitable", "production-reachable")
    ).is_autofix_eligible("production-reachable")
    assert ok2 is True
    assert "override" in reason2


@pytest.mark.parametrize("verdict", [Classification.UNREACHABLE, Classification.UNKNOWN])
def test_g3_unreachable_and_unknown_never_eligible(verdict):
    # even if (mis)configured into the autofix set, they must never get a PR
    pol = Policy(autofix_classifications=("exploitable", "unreachable", "unknown"))
    ok, reason = pol.is_autofix_eligible(verdict)
    assert ok is False
    assert "never" in reason


def test_policy_digest_deterministic_and_sensitive():
    a = Policy(severity_overrides={"r": Severity.HIGH})
    b = Policy(severity_overrides={"r": Severity.HIGH})
    c = Policy(severity_overrides={"r": Severity.LOW})
    assert a.digest == b.digest
    assert a.digest != c.digest
    assert a.digest.startswith("sha256:")


def test_g8_block_new_severities_default():
    pol = Policy()
    assert pol.blocks_new_severity(Severity.HIGH) is True
    assert pol.blocks_new_severity("critical") is True
    assert pol.blocks_new_severity(Severity.LOW) is False
