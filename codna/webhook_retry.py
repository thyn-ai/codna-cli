"""Retry policy for webhook jobs: per-kind attempt budgets, exponential backoff with full jitter,
and the transient-vs-deterministic classification that decides whether a failed attempt is worth
another one.

Pure functions, no I/O. The SQLite queue (``webhook_queue``) keeps its historical flat cap of
``_MAX_ATTEMPTS = 3`` with no backoff so the ``sqlite`` backend behaves exactly as it always has;
the Postgres backend (``webhook_pg_queue``) stamps ``max_attempts`` from :data:`RETRY_POLICY` at
enqueue and asks :func:`backoff_s` how long to hold a retry back.

Why a budget per kind: a ``review`` gates merges (it is the REQUIRED check on every public repo
ruleset) and is cheap to repeat, so it gets the most attempts with a short first wait; a ``fix``
runs a repository's own tests for up to the whole job timeout and opens a pull request, so it gets
two attempts and a long wait, on top of the 24 h one-fix-per-head rule the queue already enforces.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Mapping

# Error codes the CLI emits whose failure is a property of the inputs, never of the moment. Kept in
# lockstep with webhook_summaries._NON_RETRYABLE_CODES (imported there from this module's view of
# the world would create a cycle: summaries imports webhook, which must stay import-light).
NON_RETRYABLE_CODES = frozenset({"cli_error", "usage_error", "invalid_input", "input_error"})

# Worker-side reasons that are terminal by construction: the queue must never spend another
# attempt on them because the next attempt can only produce the same answer.
DETERMINISTIC_REASONS = frozenset({
    "review_requires_pr", "secure_requires_sarif", "fix_requires_head_ref", "fix_requires_issue_text",
    "unknown_job_kind", "installation_token_failed", "github_app_not_configured",
    "interrupted_after_resume", "orphaned_after_max_attempts", "superseded", "cancelled",
})


@dataclass(frozen=True)
class RetryPolicy:
    """``budget`` attempts in total; the wait before attempt ``n+1`` is
    ``base_s * factor**(n-1)`` capped at ``cap_s``, then drawn uniformly from ``[0, wait]``
    (full jitter), so a burst of failures does not retry in lockstep."""

    budget: int
    base_s: float
    factor: float
    cap_s: float

    def wait_s(self, attempts: int) -> float:
        """The un-jittered ceiling of the wait after ``attempts`` attempts have failed."""
        n = max(0, int(attempts) - 1)
        return min(self.cap_s, self.base_s * (self.factor ** n))


RETRY_POLICY: Mapping[str, RetryPolicy] = {
    "review": RetryPolicy(budget=4, base_s=15.0, factor=3.0, cap_s=600.0),
    # A merge queue gives every required check a bounded window and kicks the PR when it lapses;
    # the group check has to land quickly or not at all.
    "queue": RetryPolicy(budget=5, base_s=10.0, factor=2.0, cap_s=120.0),
    "secure": RetryPolicy(budget=3, base_s=60.0, factor=3.0, cap_s=900.0),
    "fix": RetryPolicy(budget=2, base_s=120.0, factor=3.0, cap_s=1800.0),
}
_DEFAULT_POLICY = RetryPolicy(budget=3, base_s=30.0, factor=2.0, cap_s=600.0)


def policy_for(kind: str) -> RetryPolicy:
    return RETRY_POLICY.get(kind, _DEFAULT_POLICY)


def budget_for(kind: str) -> int:
    return policy_for(kind).budget


def backoff_s(kind: str, attempts: int, *, rng: random.Random | None = None) -> float:
    """Seconds to hold the next attempt back after ``attempts`` failed ones (full jitter)."""
    ceiling = policy_for(kind).wait_s(attempts)
    draw = (rng or random).random()
    return round(ceiling * draw, 3)


# --- classification --------------------------------------------------------------------------
# Provider / GitHub / network conditions that resolve on their own. Matched case-insensitively
# against the worker's one-line summary and the structured error's code and message.
_TRANSIENT_RE = re.compile(
    r"("
    r"\b429\b|too many requests|rate.?limit|secondary rate|retry-after|"
    r"\b50[0-9]\b|\b502\b|\b503\b|\b504\b|bad gateway|service unavailable|gateway time.?out|"
    r"overloaded|overloaded_error|internal server error|server error|"
    r"timed? ?out|timeout|deadline exceeded|"
    r"connection (reset|refused|aborted|error)|connecterror|readerror|remoteprotocolerror|"
    r"temporarily unavailable|try again|econnreset|enotfound|eai_again|network is unreachable|"
    r"bridge unavailable|bridgeunavailable|account bridge|"
    r"enospc|no space left|out of memory|oom|killed \(137\)|exit 137|signal 9|"
    r"sidecar (was )?not ready|sidecar unreachable|agent-core (start|readiness)"
    r")",
    re.IGNORECASE,
)
_CODE_IN_SUMMARY = re.compile(r"failed \(([^)]+)\)", re.IGNORECASE)


def classify(summary: str | None = None, *, error_code: str | None = None,
             exit_code: int | None = None) -> str:
    """``"transient"`` when another attempt could plausibly succeed, ``"deterministic"`` when it
    cannot. Deterministic wins over any transient-looking text (a bad input that also mentions a
    timeout is still a bad input). Unknown failures are TRANSIENT: the budget bounds the cost of
    being wrong, and a deterministic failure mislabelled transient costs one retry, whereas a
    transient failure mislabelled deterministic costs the user their review."""
    text = summary or ""
    code = (error_code or "").strip()
    if not code:
        m = _CODE_IN_SUMMARY.search(text)
        code = m.group(1).strip() if m else ""
    if code in NON_RETRYABLE_CODES or code in DETERMINISTIC_REASONS:
        return "deterministic"
    return "transient"


def looks_like_outage(summary: str | None, *, error_code: str | None = None,
                      exit_code: int | None = None) -> bool:
    """True when the failure text names a condition that resolves on its own (a provider 429/5xx,
    a GitHub 5xx, a network error, the account bridge, ENOSPC, an OOM kill). This is the stricter
    question the post-outage sweep asks before running a dead-lettered job again: not "could a
    retry help" (:func:`classify` answers that generously) but "was this OUR outage"."""
    text = f"{summary or ''} {error_code or ''}"
    return exit_code in (137, -9) or bool(_TRANSIENT_RE.search(text))


def is_transient(summary: str | None = None, *, error_code: str | None = None,
                 exit_code: int | None = None) -> bool:
    return classify(summary, error_code=error_code, exit_code=exit_code) == "transient"


def dead_letter_summary(kind_name: str, attempts: int, job_id: int, *, retrigger: str) -> str:
    """The Check Run text for a job whose transient-failure budget is exhausted. Says plainly that
    this is Codna's failure, not a verdict on the code, and how to run it again."""
    return (f"{kind_name} could not complete after {attempts} attempt(s) because of a Codna-side "
            f"error (job #{job_id}). This is not a verdict on your code. {retrigger} "
            "Operators have been alerted.")
