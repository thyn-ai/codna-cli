"""The plan/implement prompts must offer BOTH outcomes: a change, or an answer.

THE MEASURED DEFECT. A user typed ``Hi`` into the Codna chat. The durable run planned it, and because
the plan prompt's only instruction about the SHAPE of the answer was

    "Answer with an ordered list of steps, each naming the files it touches."

the turn had no way to reply conversationally. It produced the only thing that shape allows — a plan
for doing nothing — and the product showed a work report to someone who said hello:

    run-20261115T194227Z (the user's own run, brief ``hi``):
      'The task is a greeting ("hi") with no actionable change, bug, or feature request. Per the
       no-hallucination policy, I will not invent work to perform.\\n\\nPlan:\\n\\n1. **No-op — no
       changes required.** … Files touched: none.'

That is NOT a string bug to patch with a `hi` special case: the same shape appeared for every
non-task message (measured independently on run-20261115T151026Z and run-20261114T215127Z). The cause
is the prompt, so the fix is the prompt, and these tests pin the property that makes it general:

  * every plan/implement prompt states the NO-CHANGE outcome explicitly, so a greeting, a question or
    a request answerable from the conversation has something to reply with;
  * the change outcome is unchanged, so a real brief is planned exactly as it always was;
  * the prompt is a PURE function of its inputs — no clock, no environment, no randomness — because
    the workflow records it and a retry must plan the same thing.

A test that asserted "the prompt mentions Hi" would be worthless: it would pass for a prompt that
special-cased the one word and fail the next greeting. These assert the RULE.
"""
from __future__ import annotations

import pytest

from codna.durable.contracts import implement_prompt, plan_prompt


def test_the_plan_prompt_offers_the_no_change_outcome():
    prompt = plan_prompt("hi")
    assert "asks for NO change" in prompt
    # …and says plainly what the turn should do instead, so the model is not left to invent a shape.
    assert "answer it directly and briefly" in prompt
    # The prohibition that made the old answer a refusal is still stated: nothing is fabricated either.
    assert "do not invent work to fill one" in prompt


def test_the_plan_prompt_keeps_the_change_outcome_unchanged():
    """A real brief must plan exactly as it did — the fix cannot regress the task path."""
    prompt = plan_prompt("Fix the flaky retry in client.ts")
    assert "If the task asks for a change, answer with an ordered list of steps" in prompt
    assert "each naming the files it touches" in prompt


def test_the_implement_prompt_also_offers_the_no_change_outcome():
    """The second paid turn must be able to ANSWER a greeting instead of reporting an empty diff."""
    prompt = implement_prompt("hi")
    assert "asks for NO change" in prompt
    assert "do not write a work report about a diff you did not make" in prompt


@pytest.mark.parametrize(
    "brief",
    [
        "hi",
        "Hi there!",
        "what does app.js do?",
        "thanks",
        "Fix the flaky retry",
        "",
        "  \n ",
        "Ünïcode — 你好",
        "a" * 5_000,
    ],
    ids=["greeting", "greeting-long", "question", "thanks", "task", "empty", "whitespace", "unicode", "long"],
)
def test_both_prompts_are_pure_functions_of_their_inputs(brief: str):
    """Same input, same text, every call: the replay guarantee the recorder depends on."""
    assert plan_prompt(brief) == plan_prompt(brief)
    assert plan_prompt(brief, ("note",)) == plan_prompt(brief, ("note",))
    assert implement_prompt(brief) == implement_prompt(brief)
    # An empty brief still produces a prompt the contract accepts (non-empty after strip of the
    # template), and it carries the same two-outcome rule as any other.
    assert "asks for NO change" in plan_prompt(brief)
    assert "asks for NO change" in implement_prompt(brief)


def test_steering_and_goal_context_are_APPENDED_after_the_outcomes():
    """The append-only rule that makes a recorded prompt replay byte for byte is preserved."""
    plain = plan_prompt("Fix it")
    with_steer = plan_prompt("Fix it", ("do it carefully",))
    with_goal = plan_prompt("Fix it", (), "\n\nRun objective:\nShip it")
    assert with_steer.startswith(plain)
    assert with_goal.startswith(plain)
    assert with_steer.endswith("do it carefully")
    assert with_goal.endswith("Ship it")
