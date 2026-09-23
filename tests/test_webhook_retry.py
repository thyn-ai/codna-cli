"""Retry policy for webhook jobs (codna.webhook_retry): per-kind budgets, jittered exponential
backoff, and the transient/deterministic split that decides whether an attempt is worth repeating."""
from __future__ import annotations

import random

import pytest

from codna import webhook_retry as wr
from codna.webhook_pg_schema import KINDS
from codna.webhook_summaries import _NON_RETRYABLE_CODES


def test_every_kind_has_a_policy_and_review_gets_the_most_attempts():
    budgets = {kind: wr.budget_for(kind) for kind in KINDS}
    assert set(budgets) == set(KINDS) and all(b >= 2 for b in budgets.values())
    # A review gates merges and is cheap to repeat; a fix runs a repository's tests and opens a PR.
    assert budgets["review"] > budgets["fix"] and budgets["queue"] >= budgets["review"]
    assert wr.budget_for("something-new") == wr._DEFAULT_POLICY.budget


@pytest.mark.parametrize("kind", KINDS)
def test_backoff_grows_geometrically_and_is_capped(kind):
    policy = wr.policy_for(kind)
    ceilings = [policy.wait_s(n) for n in range(1, 12)]
    assert ceilings[0] == policy.base_s
    for earlier, later in zip(ceilings, ceilings[1:]):
        assert later >= earlier
        assert later <= policy.cap_s
    assert ceilings[-1] == policy.cap_s  # the cap is reached well within any plausible budget
    assert policy.wait_s(0) == policy.base_s  # defensive: "no failed attempts yet" is the base


@pytest.mark.parametrize("kind", KINDS)
def test_backoff_is_full_jitter_within_the_ceiling(kind):
    policy = wr.policy_for(kind)
    rng = random.Random(7)
    draws = [wr.backoff_s(kind, 2, rng=rng) for _ in range(200)]
    assert all(0.0 <= d <= policy.wait_s(2) for d in draws)
    assert len(set(draws)) > 100  # jittered, not a fixed schedule: a burst does not retry in lockstep
    assert min(draws) < policy.wait_s(2) * 0.25 and max(draws) > policy.wait_s(2) * 0.75


@pytest.mark.parametrize("code", sorted(_NON_RETRYABLE_CODES))
def test_the_clis_deterministic_error_codes_are_never_retried(code):
    assert wr.classify(f"codna fix failed ({code}): bad --issue text") == "deterministic"
    assert wr.classify("anything", error_code=code) == "deterministic"
    assert wr.NON_RETRYABLE_CODES == _NON_RETRYABLE_CODES  # one truth, two modules


@pytest.mark.parametrize("reason", sorted(wr.DETERMINISTIC_REASONS))
def test_worker_side_terminal_reasons_are_deterministic(reason):
    assert wr.classify(None, error_code=reason) == "deterministic"


@pytest.mark.parametrize("text", [
    "provider returned 429 Too Many Requests",
    "anthropic: overloaded_error",
    "GitHub API 502 Bad Gateway",
    "httpx.ConnectError: [Errno 111] connection refused",
    "codna review timed out after 1800s",
    "account bridge unavailable (503) -- will retry",
    "OSError: [Errno 28] No space left on device",
    "codna fix exit 137",
    "codna review crashed unexpectedly (RemoteProtocolError)",
])
def test_outage_shaped_failures_are_transient_and_look_like_an_outage(text):
    assert wr.classify(text) == "transient"
    assert wr.looks_like_outage(text) is True


def test_unknown_failures_are_transient_but_do_not_look_like_an_outage():
    """The budget bounds the cost of being wrong about a retry; the post-outage sweep is stricter."""
    assert wr.classify("codna review failed (review_failed): the agent produced no findings JSON") == "transient"
    assert wr.looks_like_outage("the agent produced no findings JSON") is False
    assert wr.looks_like_outage(None, exit_code=137) is True
    assert wr.is_transient("random text") is True


def test_a_deterministic_code_wins_over_transient_looking_text():
    assert wr.classify("codna fix failed (cli_error): clone timed out after 30s (503)") == "deterministic"


def test_dead_letter_summary_names_the_job_and_disclaims_the_code():
    text = wr.dead_letter_summary("codna review", 4, 123, retrigger="Comment `@codna review`.")
    assert "could not complete after 4 attempt(s)" in text
    assert "job #123" in text and "not a verdict on your code" in text
    assert "Comment `@codna review`." in text and "Operators have been alerted" in text
