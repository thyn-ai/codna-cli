"""A failed `codna review` has to say WHY, not just that it failed.

Measured on live check runs 2026-08-26/27, every `codna review` failure across the org
serialised as:

    {"error": {"code": "review_error",
               "message": "review agent failed: Local agent-core did not complete the
                           packaged fix run successfully.",
               "details": {}}}

`details` was empty for a structural reason, not an unlucky one: `ReviewError.details` was a
bare CLASS attribute that nothing ever assigned to, so it was the same empty dict for every
instance, forever. Meanwhile the wrapped `PackagedRepositoryAdvancedError` carried
`sidecar_url`, `status`, `terminal_state` and `error` -- everything needed to tell a stub
runtime from a genuinely failed run from an unreachable sidecar -- and the stringifying
`except` dropped all of it.

The result was a check that names a symptom and withholds every fact about it.
"""

from __future__ import annotations

import codna.review as review


class _RuntimeFailure(Exception):
    """Shaped like the runtime errors review wraps: a code plus structured detail."""

    def __init__(self) -> None:
        super().__init__("Local agent-core did not complete the packaged fix run successfully.")
        self.code = "agent_core_run_failed"
        self.details = {
            "sidecar_url": "http://127.0.0.1:28601",
            "status": "failed",
            "terminal_state": "failed",
            "error": "provider request rejected",
        }


def test_details_are_per_instance_not_shared_across_errors() -> None:
    """The original bug in one line: a class attribute is one object for every instance.

    Nothing wrote to it, so it stayed empty; had anything mutated it in place, one review's
    detail would have leaked into every other review's error -- across organizations, on a
    shared webhook worker.
    """
    first = review.ReviewError("first", {"a": 1})
    second = review.ReviewError("second")

    assert first.details == {"a": 1}
    assert second.details == {}
    assert first.details is not second.details


def test_a_wrapped_runtime_failure_keeps_its_structured_detail() -> None:
    """The fields that make the failure actionable survive the wrap."""
    details = review._wrapped_details(_RuntimeFailure())  # noqa: SLF001

    assert details["terminal_state"] == "failed"
    assert details["status"] == "failed"
    assert details["error"] == "provider request rejected"
    assert details["sidecar_url"] == "http://127.0.0.1:28601"


def test_the_cause_code_is_preserved_so_failures_can_be_told_apart() -> None:
    """`review_error` is the outer code for every failure, so it distinguishes nothing.

    The inner code is what separates agent_core_run_failed from
    agent_core_stub_runtime_rejected from agent_core_final_frame_missing.
    """
    assert review._wrapped_details(_RuntimeFailure())["cause_code"] == "agent_core_run_failed"  # noqa: SLF001


def test_an_error_carrying_nothing_still_produces_an_empty_mapping() -> None:
    """Plain exceptions must not blow up the error path -- it is already the failure path."""
    assert review._wrapped_details(ValueError("boom")) == {}  # noqa: SLF001


def test_a_non_mapping_details_attribute_is_ignored_rather_than_trusted() -> None:
    """Defensive: `details` is only meaningful if it is actually a mapping."""

    class Odd(Exception):
        details = "not-a-mapping"
        code = "odd_failure"

    out = review._wrapped_details(Odd("x"))  # noqa: SLF001
    assert out == {"cause_code": "odd_failure"}


def test_fail_attaches_details_to_the_raised_error() -> None:
    """End to end through the helper the review path actually calls."""
    try:
        review._fail("review agent failed: boom", {"terminal_state": "failed"})  # noqa: SLF001
    except review.ReviewError as exc:
        assert exc.details == {"terminal_state": "failed"}
        assert exc.code == "review_error"
    else:  # pragma: no cover - _fail must always raise
        raise AssertionError("_fail did not raise")
