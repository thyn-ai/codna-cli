"""The ONE approval driver the durable scenarios share, with its benign races spelled out.

Every scenario that parks a child on a human decision needs the same three steps: ask the child
whether it is parked, approve it if it is, and tolerate the two things that legitimately go wrong on
the way. That used to be copy-pasted into five files, each with a blanket ``except Exception`` and a
comment saying "a concurrent iteration already approved it". The copy-paste is why the same latent
hazard lived in four of them: the ``except`` also swallowed a programming error, and a swallowed error
here does not fail a test — it leaves every child parked, the run RUNNING, and the caller dead on a
timeout that names nothing.
This module exists so the narrowing happens once. What the helper absorbs is enumerated, not implied:

* a child that has NOT STARTED YET — ``RPCError(NOT_FOUND)`` from both the status query and the
  approval update (measured against the time-skipping server, not assumed);
* a child that CLOSED between the query and the update — the same ``RPCError(NOT_FOUND)``;
* an approval already applied by a concurrent iteration — ``WorkflowUpdateFailedError`` whose cause is
  the workflow validator's ApplicationError saying the task "is not waiting on a decision". The cause
  is inspected rather than the exception type alone, so a validator refusal with a DIFFERENT reason (a
  malformed identity, say) is a harness bug and propagates.

Everything else propagates. That is the point: ``AttributeError``/``TypeError``/``NameError`` from
this helper or its caller must fail at the call, because a hang can never satisfy a caller and never
explains itself. `test_approval_driver.py` pins exactly that.

``approval_cls`` is the workflow class of the child (``TaskWorkflowPlain`` or ``TaskWorkflow``), and a
wrong one is REFUSED before any await instead of being queried: ``None.status`` raising inside a
driver is precisely the bug this module was extracted to make impossible to repeat.
"""
from __future__ import annotations

from typing import Any, Iterable

from temporalio.client import WorkflowUpdateFailedError
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError, RPCStatusCode

# The validator's own words when a decision is already recorded (both engines raise this text). The
# message is part of the contract this helper relies on: the SAME exception type is also how a
# malformed payload is refused, and those must not be confused.
_ALREADY_DECIDED = "not waiting on a decision"


def require_approval_class(approval_cls: Any) -> Any:
    """Refuse anything that cannot drive an approval, naming the offender.

    A hang, not a red assertion, is what a wrong class used to produce: ``None.status`` raised
    ``AttributeError`` inside a driver whose blanket ``except`` swallowed it, so no approval was ever
    sent, every child stayed parked at its ``interrupt()``, and the run sat RUNNING until a timeout.
    Asserting the preconditions turns that into a failure at the call that got it wrong.
    """
    if not isinstance(approval_cls, type):
        raise TypeError(
            f"an approval needs the child workflow class, got {approval_cls!r}: a None/live object here "
            "cannot query or update, and swallowing that failure leaves every child parked"
        )
    for handler in ("status", "approve"):
        if not hasattr(approval_cls, handler):
            raise TypeError(
                f"{approval_cls!r} cannot drive an approval: it needs the 'status' query and the "
                f"'approve' update, and it has no {handler!r}"
            )
    return approval_cls


def _is_not_started_or_closed(exc: RPCError) -> bool:
    """True for the server's "that execution/update is not there" answer, and nothing else.

    NOT_FOUND is what both benign absences produce (measured): a query before the child's first
    workflow task, and an Update aimed at an execution that closed in between. Every other RPC status
    — UNAVAILABLE, DEADLINE_EXCEEDED, PERMISSION_DENIED — is an environment or harness problem that a
    test must see, so it propagates.
    """
    return exc.status == RPCStatusCode.NOT_FOUND


def _is_already_decided(exc: WorkflowUpdateFailedError) -> bool:
    """True when the refusal IS the benign race: someone else already recorded the decision.

    Checked on the CAUSE, because the same exception type also carries a validator's refusal of a
    malformed Update. Absorbing that would hide a harness bug behind a comment — the exact trade this
    module was extracted to stop making.
    """
    cause = exc.cause
    return isinstance(cause, ApplicationError) and _ALREADY_DECIDED in str(cause)


async def approve_if_parked(
    client: Any,
    workflow_id: str,
    approval_cls: Any,
    *,
    identity: str,
) -> bool:
    """Approve ``workflow_id`` if it is parked and undecided. Returns whether an Update was sent.

    Skipping is NOT a failure: a child that has not started, one that closed, and one whose decision a
    concurrent driver iteration already recorded are all normal in a polling loop. Anything outside
    that list propagates, so a bug fails here rather than as a timeout.
    """
    require_approval_class(approval_cls)
    child = client.get_workflow_handle(workflow_id)
    try:
        status = await child.query(approval_cls.status)
    except RPCError as exc:
        if _is_not_started_or_closed(exc):  # not started yet, or already closed
            return False
        raise
    if not status.get("awaiting_approval"):
        return False
    try:
        await child.execute_update(approval_cls.approve, {"identity": identity})
    except RPCError as exc:
        if _is_not_started_or_closed(exc):  # closed between the query and the update
            return False
        raise
    except WorkflowUpdateFailedError as exc:  # a concurrent iteration already approved it
        if _is_already_decided(exc):
            return False
        raise
    return True


async def approve_all_parked(
    client: Any,
    workflow_ids: Iterable[str],
    approval_cls: Any,
    *,
    identity: str,
) -> int:
    """`approve_if_parked` over every id a run may have created. Returns how many were approved.

    The ids are enumerated by the caller because a fix-CI task's workflow id is
    ``task-<run>-<task>-ci<n>``: the set of children a run can start is not derivable from its task
    count. The return value is for the caller's own assertions; a polling loop may ignore it.
    """
    require_approval_class(approval_cls)
    approved = 0
    for workflow_id in workflow_ids:
        if await approve_if_parked(client, workflow_id, approval_cls, identity=identity):
            approved += 1
    return approved
