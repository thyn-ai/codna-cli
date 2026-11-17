"""A keyless provider's turn is FREE, not UNKNOWN — the sidecar side of the unpriced-spend defect.

THE MEASURED DEFECT. On `provider: ollama`, `model: glm-5.3-flash:cloud`, every turn reported
`usage.totalCost: 0` — the engine prices local compute at zero, stated rather than omitted — and the
turn driver accepted a price only when it was `> 0`. So the whole token count was folded in as
`unpriced`, `Budget.unpriced_tokens` matched `spent_tokens`, and the ledger carried two notices per
run saying the cost caps could not see the spend:

    run-20261115T194545Z  (before)
      session.turn_completed → spend {"tokens": 9402, "unpriced_tokens": 9402, "usd": 0}
      session.budget_unpriced (session) → "…could not price part of this session's tokens…"
      session.budget_unpriced (turn)    → "…could not price this turn's tokens…"

The rule these tests pin, and the reason it is `>= 0` rather than an invented price:

  * `0` IS a price. A local provider bills nothing Codna can see, so the cap is not blind to it —
    it is not being charged. Treating "the catalog has no entry" as "unknown" is what produced the
    alarm, and inventing a dollar figure would make the cap lie in the other direction.
  * everything else keeps the OLD behaviour exactly: a hosted provider whose model cannot be priced
    is still reported as unpriced, because a cap that silently assumed $0 there would under-enforce.
"""
from __future__ import annotations

import pytest

# THE `durable` EXTRA IS OPTIONAL HERE, as it is for every other file in this directory: the offline
# CI job installs pytest and the runtime deps but NOT temporalio/langgraph, and a module-level import
# of either aborts COLLECTION for the whole run rather than skipping one file. `turn_driver` pulls in
# the activity surface, so this file needs the same gate its siblings use (see
# `test_approval_driver.py`, `test_ask_parks_run.py`).
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.state import Budget, TaskState

# `turn_driver` is re-exported BY `activity_support` (its documented entry point: the module imports
# the runtime types and then re-exports the driver), so importing the driver first would close the
# cycle. Importing the surface first is the shape every caller uses.
from codna.durable import activity_support  # noqa: F401  (import order: see above)

from codna.durable.turn_driver import _spend_from_event, is_keyless_provider


def _state(*, spent_usd: float = 0.0, spent_tokens: int = 0) -> TaskState:
    return TaskState(
        run_id="run-test",
        task_id="task-1",
        budget=Budget(max_cost_usd=25.0, spent_usd=spent_usd, max_tokens=250_000, spent_tokens=spent_tokens),
    )


# -- the provider predicate ------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    ["ollama", "lmstudio", "openai-compatible", "litellm", "OLLAMA", " Ollama "],
)
def test_keyless_providers_are_recognised(provider: str):
    assert is_keyless_provider(provider) is True


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "openai", "gemini", "openrouter", "", None],
)
def test_hosted_and_absent_providers_are_not_keyless(provider: object):
    # An ABSENT provider is not keyless: "we were not told" is not "there is no bill".
    assert is_keyless_provider(provider) is False  # type: ignore[arg-type]


# -- the accounting ---------------------------------------------------------------------------


def test_a_local_turns_zero_cost_is_a_PRICE_not_an_unknown():
    """The exact measured payload: 5,716 tokens and `totalCost: 0` on ollama."""
    spend = _spend_from_event(
        {"usage": {"inputTokens": 5_418, "outputTokens": 298, "totalCost": 0}},
        state=_state(),
        provider="ollama",
    )
    assert spend.tokens == 5_716
    assert spend.priced is True
    assert spend.cost_usd == 0.0
    # …so folding it in can never produce the "the cap cannot see 5,716 tokens" notice.
    budget = _state().budget.with_spend(cost_usd=spend.cost_usd, tokens=spend.tokens, priced=spend.priced)
    assert budget.unpriced_tokens == 0
    assert budget.spent_tokens == 5_716
    assert budget.spent_usd == 0.0


def test_a_local_turn_with_no_usage_OBJECT_at_all_is_still_free():
    """A keyless provider cannot become a blind spot by omitting the usage block."""
    spend = _spend_from_event({}, state=_state(), provider="ollama")
    assert (spend.tokens, spend.priced, spend.cost_usd) == (0, True, 0.0)


def test_a_hosted_turn_that_cannot_be_priced_is_STILL_unpriced():
    """The fix must not leak: a hosted model with no catalog price stays a genuine blind spot."""
    spend = _spend_from_event(
        {"usage": {"inputTokens": 1_000, "outputTokens": 100}},
        state=_state(),
        provider="anthropic",
    )
    assert spend.tokens == 1_100
    assert spend.priced is False
    budget = _state().budget.with_spend(cost_usd=spend.cost_usd, tokens=spend.tokens, priced=spend.priced)
    assert budget.unpriced_tokens == 1_100


def test_a_reported_positive_cost_still_wins_for_every_provider():
    """A provider that DOES report a cost is believed, whether or not it is keyless."""
    spend = _spend_from_event(
        {"usage": {"inputTokens": 1_000, "outputTokens": 0, "totalCost": 0.25}},
        state=_state(),
        provider="ollama",
    )
    assert spend.priced is True
    assert spend.cost_usd == 0.25
