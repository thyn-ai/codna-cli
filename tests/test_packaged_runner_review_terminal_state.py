"""A read-only review terminates in `apply_blocked`, and that is success.

`codna review` failed on EVERY pull request across the org with one identical message:

    review agent failed: Local agent-core did not complete the packaged fix run successfully.

It was not flaky. It was categorically impossible to pass, and the two halves of the reason
live in different repositories:

  * codna dispatches a review with NO write scope, deliberately -- `_sidecar_payload` sets
    `allowed_write_paths` and `allowed_write_scope` to `[]` when task_kind == "review",
    because a review analyses and must not modify the repository.
  * agent-core emits terminal_state `apply_blocked` in exactly that situation
    (vendor/cline/algenta/server/artifacts.ts: `allowedWriteScope.length === 0` AND some tool
    call succeeded).

So the better a review worked -- the more tool calls it completed -- the more certainly it
landed in `apply_blocked`, while `_parse_final_frame` accepted only "succeeded".

The frame below is the real payload observed on thyn-ai/algenta#926, recoverable only after
thyn-ai/codna#492 stopped discarding the error details that named `terminal_state`.
"""

from __future__ import annotations

import json

import pytest

from codna.packaged_agent_runner import (
    _parse_final_frame,  # noqa: SLF001
    _terminal_states_meaning_success,  # noqa: SLF001
)
from codna.packaged_repository_advanced import PackagedRepositoryAdvancedError

# Verbatim from the failing check run on thyn-ai/algenta#926.
_PRODUCTION_REVIEW_FRAME = json.dumps(
    {
        "type": "final",
        "status": "failed",
        "terminal_state": "apply_blocked",
        "error": None,
        "artifacts": {"findings": []},
        "runtime": {"is_stub": False},
    }
)


def test_a_review_that_was_blocked_from_applying_is_not_a_failure():
    """The exact production frame must now parse instead of raising."""
    final = _parse_final_frame(
        _PRODUCTION_REVIEW_FRAME, sidecar_url="http://127.0.0.1:34048", task_kind="review"
    )
    assert final["terminal_state"] == "apply_blocked"


def test_the_same_frame_is_still_a_failure_for_a_fix():
    """A fix that was not allowed to write has not fixed anything.

    This is the half of the behaviour that must NOT change: widening `apply_blocked` for
    every task kind would turn a silently-unapplied fix into a green check, which is far
    worse than the bug being fixed here.
    """
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(
            _PRODUCTION_REVIEW_FRAME, sidecar_url="http://127.0.0.1:34048", task_kind="fix"
        )
    assert excinfo.value.code == "agent_core_run_failed"
    assert excinfo.value.details["terminal_state"] == "apply_blocked"
    assert excinfo.value.details["task_kind"] == "fix"


def test_the_default_task_kind_stays_strict():
    """An unspecified caller must get the conservative behaviour, not the permissive one."""
    assert _terminal_states_meaning_success("fix") == {"succeeded"}
    assert _terminal_states_meaning_success("triage") == {"succeeded"}
    assert "apply_blocked" in _terminal_states_meaning_success("review")


def test_a_genuinely_failed_review_still_fails():
    """`apply_blocked` is accepted; arbitrary failure is not.

    Without this, the fix would read as "reviews can never fail", which would make the check
    worthless rather than working.
    """
    frame = json.dumps(
        {"type": "final", "status": "failed", "terminal_state": "failed", "error": "boom"}
    )
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(frame, sidecar_url="http://127.0.0.1:1", task_kind="review")
    assert excinfo.value.details["terminal_state"] == "failed"


def test_an_aborted_review_still_fails():
    """The other terminal states agent-core can emit stay failures for a review too."""
    for state in ("aborted", "stopped", "engine_evidence_missing", "plugin_evidence_missing"):
        frame = json.dumps({"type": "final", "status": "failed", "terminal_state": state})
        with pytest.raises(PackagedRepositoryAdvancedError):
            _parse_final_frame(frame, sidecar_url="http://127.0.0.1:1", task_kind="review")


def test_the_error_names_what_it_accepted():
    """When it does fail, the payload must say which states would have counted.

    The whole reason this bug survived so long is that the failure reported a symptom and
    withheld the facts (thyn-ai/codna#492).
    """
    frame = json.dumps({"type": "final", "status": "failed", "terminal_state": "failed"})
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(frame, sidecar_url="http://127.0.0.1:1", task_kind="review")
    assert excinfo.value.details["accepted_terminal_states"] == ["apply_blocked", "succeeded"]


def test_a_stub_runtime_is_still_refused_for_a_review():
    """Accepting `apply_blocked` must not accidentally let a stub runtime through.

    The stub check runs AFTER the terminal-state check, so widening the latter is exactly the
    kind of change that could expose it. Pinned here so it cannot regress silently.
    """
    frame = json.dumps(
        {
            "type": "final",
            "status": "failed",
            "terminal_state": "apply_blocked",
            "runtime": {"is_stub": True},
        }
    )
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(frame, sidecar_url="http://127.0.0.1:1", task_kind="review")
    assert excinfo.value.code == "agent_core_stub_runtime_rejected"
