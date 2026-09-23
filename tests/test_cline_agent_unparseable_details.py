"""When the review agent's reply cannot be parsed, say what it actually said.

Observed live on thyn-ai/algenta-sdk#16, after thyn-ai/codna#493 fixed the `apply_blocked`
failure and a *different* one surfaced underneath:

    {"error": {"code": "review_error",
               "message": "review agent failed: review agent did not return parseable findings JSON",
               "details": {}}}

`details` was empty because `ClineAgentError` carried no detail and the raw reply was discarded
at the raise site. Three quite different problems then report the identical sentence:

  * the agent answered in prose (a refusal, or a preamble before the JSON)
  * the JSON was truncated by a token limit
  * the reply was empty

Head AND tail are both kept because they show different things: truncation is only visible at
the end, a refusal or preamble only at the start. The text is redacted before it is attached,
because it is echoed into a check summary on a public pull request.
"""

from __future__ import annotations

import json

import pytest

from codna.cline_agent import (
    ClineAgentError,
    _RAW_EXCERPT_CHARS,  # noqa: SLF001
    _unparseable_output_details,  # noqa: SLF001
)


def test_details_are_per_instance_not_shared() -> None:
    """The class-attribute bug fixed in codna#492, pinned here so it is not reintroduced."""
    first = ClineAgentError("a", {"x": 1})
    second = ClineAgentError("b")
    assert first.details == {"x": 1}
    assert second.details == {}
    assert first.details is not second.details


def test_a_prose_refusal_is_visible_in_the_head() -> None:
    """The most common real case: the agent explains instead of emitting JSON."""
    details = _unparseable_output_details("I'm unable to review this diff because ...")
    assert "unable to review" in details["raw_head"]
    assert details["raw_chars"] == len("I'm unable to review this diff because ...")


def test_truncated_json_is_visible_in_the_tail() -> None:
    """A token-limit cut is only detectable at the END, which is why the tail is kept."""
    body = '{"findings": [' + ('{"a": "' + "x" * 200 + '"},') * 6
    details = _unparseable_output_details(body)
    assert details["raw_tail"], "a long reply must carry a tail"
    assert details["raw_tail"].endswith(body[-10:])


def test_an_empty_reply_is_distinguishable_from_a_long_one() -> None:
    """Zero chars is itself the diagnosis, and must not look like anything else."""
    details = _unparseable_output_details("")
    assert details["raw_chars"] == 0
    assert details["raw_head"] == ""
    assert details["raw_tail"] == ""


def test_a_short_reply_carries_no_redundant_tail() -> None:
    """Head already contains everything; duplicating it would just pad the check summary."""
    details = _unparseable_output_details("nope")
    assert details["raw_head"] == "nope"
    assert details["raw_tail"] == ""


def test_the_excerpt_is_bounded_so_a_runaway_reply_cannot_flood_the_summary() -> None:
    huge = "y" * 100_000
    details = _unparseable_output_details(huge)
    assert len(details["raw_head"]) == _RAW_EXCERPT_CHARS
    assert len(details["raw_tail"]) == _RAW_EXCERPT_CHARS
    assert details["raw_chars"] == 100_000


def test_secrets_are_redacted_before_being_attached() -> None:
    """This text lands in a check summary on a public pull request.

    Attaching the raw reply without redaction would trade one diagnosability bug for a
    disclosure one, which is a strictly worse deal.
    """
    details = _unparseable_output_details(
        "OPENAI_API_KEY=sk-secretvalue\nAuthorization: Bearer secret-token-value\nnot json"
    )
    rendered = json.dumps(details)
    assert "sk-secretvalue" not in rendered
    assert "secret-token-value" not in rendered
    assert "[redacted]" in rendered


def test_the_strategies_tried_are_named() -> None:
    """Saying WHAT was attempted turns "unparseable" into something actionable."""
    tried = _unparseable_output_details("x")["looked_for"]
    assert any("whole message" in s for s in tried)
    assert any("fence" in s for s in tried)
    assert any("balanced" in s for s in tried)


def test_the_error_carries_the_details_through() -> None:
    """End to end: raising with these details keeps them on the exception."""
    with pytest.raises(ClineAgentError) as excinfo:
        raise ClineAgentError("boom", _unparseable_output_details("prose, not json"))
    assert excinfo.value.details["raw_head"] == "prose, not json"
