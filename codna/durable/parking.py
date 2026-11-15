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

── THE SECOND DEFECT THIS MODULE CARRIES, AND THE CORRECTION A FIRST FIX GOT WRONG ─────────────

`run-cancelrepro-before`, verbatim from the ledger:

    seq 404  kind=review  payload={verdict:"approved", sha:"1686ad67...", conclusion:"success"}
    seq 405  kind=note    publish.skipped (no origin remote)
    seq 406  kind=note    approval.requested "task task-1 is waiting on a human (review_not_approved)"

The contradiction is REAL and the cause is the fallback rung below, but the fix is NOT to stop the
park. **The park is the design.** ``wait_for_human`` is the graph's last node and every route ends
there; the review verdict approves the WORK, and the human gate approves RELEASING it. A published or
publishable task is therefore expected to reach the gate with ``review_verdict == "approved"`` — see
`test_task_workflow_time_skipping.py::test_plain_engine_walks_the_full_route_the_plan_names`, which
asserts exactly that route (``review`` -> ``open_pr`` -> gate -> human approval -> ``approved``).

What was wrong is the WORD. The fallback returned the literal ``review_not_approved``, which is a
sentence about the reviewer's verdict — and one the reviewer had just contradicted. A gate reading
"not approved" for a task whose review record says ``approved`` tells a human that the work was
rejected and to look for a bad review that does not exist. Deleting the park would instead have
removed the human gate from every published task, which is a far worse bug than the one reported.

So the reason for "work is reviewed and ready; a human must release it" is now its own name,
``ready_for_approval``, and ``review_not_approved`` is reserved for the state it actually describes.
That is the closed-vocabulary discipline this module exists for: a name means ONE thing, and the value
the gate reads is the value the reviewer wrote.
"""
from __future__ import annotations

from typing import Final

from .state import TaskState

__all__ = ["ParkReason", "PARK_REASONS", "park_kind", "park_message", "park_reason"]

#: The closed vocabulary, in precedence order. Nothing else may be recorded as a park reason: an
#: audit trail with free-form reasons cannot be counted, and the UI switches on these names.
PARK_REASONS: Final[tuple[str, ...]] = (
    "answer_required",
    "budget_exhausted",
    "plan_not_produced",
    "verify_failed",
    "review_not_approved",
    # The work PASSED its own review and the gate is asking a human to release it. Its own name
    # because the alternative — reporting it as `review_not_approved` — is a false statement about a
    # record in the same ledger (see this module's docstring).
    "ready_for_approval",
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
    * ``review_not_approved`` is read from `review_verdict`, the value the review step WROTE. An
      absent or unrecognized verdict is not an approval (the same rule `after_review` applies), so it
      lands here; an ``approved`` verdict falls through to the last rung instead.
    * ``ready_for_approval`` last, and it is the normal case for a completed task: the work passed,
      and releasing it is a human's decision. It is NOT a complaint about the work, and the message a
      human reads must not read like one.
    """
    if state.ask is not None:
        return "answer_required"
    if state.budget.exhausted:
        return "budget_exhausted"
    if state.plan_ref is None:
        return "plan_not_produced"
    if state.verify_ref is None or not state.verify_ref.passed:
        return "verify_failed"
    if state.review_verdict != "approved":
        return "review_not_approved"
    return "ready_for_approval"


def park_kind(state: TaskState) -> str:
    """The kind of gate this park presents.

    ``ask`` is the third kind `PendingApproval` declares: a parked run whose approval releases a
    specific tool call, not a whole task. Keeping it distinct is what lets a UI render the engine's
    question instead of a generic "approve this task".
    """
    if state.ask is not None:
        return "ask"
    return "budget" if state.budget.exhausted else "approve"


def park_message(state: TaskState, reason: str) -> str:
    """The sentence a human reads on the gate. One reading per reason, so a name cannot mislead.

    ``park_reason`` is the CLOSED name a UI branches on; this is the prose. They are separate because
    a terse name cannot carry a sentence and a sentence cannot be switched on — and a reason whose
    message contradicts it is exactly the defect this module was fixed for.
    """
    if reason == "ready_for_approval":
        if state.pr is None and state.publish_skip is not None:
            return (
                "the work is planned, verified and reviewed; approving releases it. No pull request "
                "was opened (the repository has nothing to publish to), so it releases the commits on "
                "this task's branch"
            )
        return "the work is planned, verified and reviewed; approving releases it"
    if reason == "review_not_approved":
        return "the review did not approve the work"
    if reason == "verify_failed":
        return "verification did not pass"
    if reason == "budget_exhausted":
        return "the task's budget is spent"
    if reason == "plan_not_produced":
        return "no plan was produced"
    return "the agent is waiting on a decision"
