"""Why a task is parked, and what kind of gate that is — a LEAF module.

`park_reason`/`park_kind` were written in `workflow_support.py`, which imports `.graph` (it needs
`ACTIVITY_RETRIES` and `STEP_TIMEOUT_S`). `graph.py` therefore could not read them back: importing
`workflow_support` from a workflow-facing module would be a cycle, and — the reason this matters more
— `workflow_support` reaches `codna.runtime` layers that Temporal's workflow sandbox refuses (see
`tests/durable/test_import_hygiene.py`).

The consequence was a real, measured defect rather than a tidiness problem. The graph engine's
`wait_for_human` node had no way to say WHY it parked, so `_await_decision` hardcoded
``open_pr_ready`` for every non-budget park. `run-20261113T225225Z` shows what that costs: three
`verify` records with ``passed: false`` and ``returncode: 127``, then an approval recorded with
reason ``open_pr_ready`` — the audit trail of a human gate reporting the work as ready while the
run's own verification had never once passed.

So the vocabulary lives here, where BOTH engines and both layers can reach it, and it imports only
`state` (itself a leaf) because a reason is a judgement about a `TaskState`.
"""
from __future__ import annotations

from typing import Final

from .state import TaskState

__all__ = ["ParkReason", "PARK_REASONS", "park_kind", "park_reason"]

#: The closed vocabulary, in precedence order. Nothing else may be recorded as a park reason: an
#: audit trail with free-form reasons cannot be counted, and the UI switches on these names.
PARK_REASONS: Final[tuple[str, ...]] = (
    "answer_required",
    "budget_exhausted",
    "plan_not_produced",
    "verify_failed",
    "review_not_approved",
)

ParkReason = str


def park_reason(state: TaskState) -> str:
    """Why a task is parking, in the vocabulary `status` and the ledger use.

    Order matters, and every rung is a different repair:

    * ``answer_required`` first because it is the only reason that is NOT a judgement about the work.
      The work has not happened yet — the agent is waiting for a decision — and reporting it as
      ``plan_not_produced`` would send a human looking for a broken plan.
    * ``budget_exhausted`` before the step reasons because a spent budget is why the step stopped
      mattering; the step reason would describe the last attempt, not the reason for the park.
    * ``verify_failed`` is read from `verify_ref`, so an exhausted attempt budget after three red
      verifies reports ``verify_failed`` rather than a generic "needs approval". That distinction is
      the whole point: an approval granted here is a human overriding a FAILED verification, and the
      record has to say so.
    """
    if state.ask is not None:
        return "answer_required"
    if state.budget.exhausted:
        return "budget_exhausted"
    if state.plan_ref is None:
        return "plan_not_produced"
    if state.verify_ref is None or not state.verify_ref.passed:
        return "verify_failed"
    return "review_not_approved"


def park_kind(state: TaskState) -> str:
    """The kind of gate this park presents.

    ``ask`` is the third kind `PendingApproval` declares: a parked run whose approval releases a
    specific tool call, not a whole task. Keeping it distinct is what lets a UI render the engine's
    question instead of a generic "approve this task".
    """
    if state.ask is not None:
        return "ask"
    return "budget" if state.budget.exhausted else "approve"