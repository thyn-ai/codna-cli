"""GitHub events → durable runs, with the workflow id as the idempotency key (D5).

The App's ingress used to hand a delivery to a SQLite queue, and the queue's claim / retry /
recover semantics were what made "one webhook, one job" true. Temporal already IS that machine, so
the durable path replaces the queue rather than running beside it:

    workflow id = <X-GitHub-Delivery>

`temporalio` guarantees a workflow id is an **idempotency key**: a `start_workflow` for an id that is
already running either fails, is refused, or — with ``USE_EXISTING`` — attaches to the execution that
is already running. That is what makes a GitHub REDELIVERY harmless: GitHub retries a delivery it
thinks failed, and the second attempt must not spend a second run.

**Signal-with-Start is why the signal exists.** ``start_signal`` is delivered as part of the start
request, atomically, so there is no window in which a run is created without knowing which delivery
woke it. With ``USE_EXISTING``, a redelivery's signal lands on the run that is already working, and
`WebhookTrigger` on the workflow side records it as a DUPLICATE (the delivery id it has already seen)
rather than queueing the same work twice.

**The brief is derived from the event, deterministically.** ``webhook_brief`` is a pure function of
the delivery, so a redelivery that somehow did start a second run would ask for the same work — and a
test can assert what a given payload produces without a server.

This module is CLIENT code: it builds a payload and calls ``start_workflow``. Nothing here is
imported by a workflow, so it may touch `webhook.py` (which imports the App's policy).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Mapping

from .goal import GoalSpec, StopCondition

__all__ = [
    "NAME_WEBHOOK_SIGNAL",
    "WEBHOOK_TRIGGER_VERSION",
    "WebhookDispatch",
    "WebhookTriggerError",
    "dispatch_webhook",
    "goal_for_webhook",
    "run_id_for_delivery",
    "webhook_brief",
]

# The signal Temporal delivers WITH the start request. Named once, here, and mirrored by
# `RunWorkflow.webhook`'s decorator — a mismatch is a signal the server accepts and the workflow
# ignores, which is silent, so a test asserts the two agree.
NAME_WEBHOOK_SIGNAL: Final = "webhook"

# The trigger's payload schema. Bumped when the shape a workflow receives changes, so a run started
# by an older trigger can be told apart from one started by this build.
WEBHOOK_TRIGGER_VERSION: Final = 1

# What a "fix" job asks for by default: verify and review must both hold. CI is NOT required for an
# App-triggered fix, because the App's own fix pipeline already gates on the check suite it was
# reacting to — requiring it again would make a fix for a red check wait on that same red check.
_DEFAULT_STOP = StopCondition(verify=True, review=True, ci=False)

_RUN_ID_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class WebhookTriggerError(RuntimeError):
    """A delivery that could not be turned into a run. Always carries a stable ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass(frozen=True)
class WebhookDispatch:
    """One delivery, as the thing that starts (or attaches to) a run.

    ``delivery_id`` is both the workflow id and the event's identity. It is REQUIRED and validated:
    a delivery without one cannot be idempotent, and a made-up id (a timestamp, a random uuid) would
    turn every redelivery into a second run — the failure this whole module exists to prevent.
    """

    delivery_id: str
    kind: str
    repo_full_name: str
    run_payload: dict[str, Any]
    # What the workflow's `webhook` signal receives. Small and reference-only: the delivery id, the
    # kind and the ref facts a redelivery has to be recognised by.
    signal_payload: dict[str, Any]

    def validate(self) -> None:
        validate_delivery_id(self.delivery_id)
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise WebhookTriggerError(
                "a trigger needs a non-empty kind",
                code="webhook_trigger_invalid",
                details={"field": "kind"},
            )
        if not isinstance(self.repo_full_name, str) or "/" not in self.repo_full_name:
            raise WebhookTriggerError(
                "repo_full_name must be owner/repo",
                code="webhook_trigger_invalid",
                details={"field": "repo_full_name", "value": repr(self.repo_full_name)},
            )
        if not isinstance(self.run_payload, dict) or not self.run_payload:
            raise WebhookTriggerError(
                "a trigger needs a run payload",
                code="webhook_trigger_invalid",
                details={"field": "run_payload"},
            )
        if not isinstance(self.signal_payload, dict):
            raise WebhookTriggerError(
                "a trigger needs a signal payload",
                code="webhook_trigger_invalid",
                details={"field": "signal_payload"},
            )
        goal = self.run_payload.get("goal")
        if goal is not None:
            GoalSpec.from_payload(goal)


def validate_delivery_id(value: str) -> str:
    """A GitHub delivery id, as it reaches a workflow id.

    GitHub's delivery ids are GUIDs, but the accepted class is the same token class every other
    codna identifier uses: this value becomes a Temporal workflow id, and a workflow id is also what
    an operator types. Refusing an exotic one here is better than a started run whose id cannot be
    looked up.
    """
    if not isinstance(value, str) or not _RUN_ID_TOKEN.match(value):
        raise WebhookTriggerError(
            "delivery_id must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ (it is the workflow id)",
            code="webhook_delivery_invalid",
            details={"delivery_id": value if isinstance(value, str) else type(value).__name__},
        )
    return value


def run_id_for_delivery(delivery_id: str, kind: str) -> str:
    """The run's own id, derived from the delivery. Deterministic, so a redelivery names the same run.

    Derived rather than random for the same reason the workflow id is the delivery: a run id is what
    the user sees, what the run directory is keyed on and what the ledger is named after, so a
    redelivery that attached to an existing execution must resolve to the SAME directory — otherwise
    the attached signal would write its duplicate note into a second, empty ledger.
    """
    validate_delivery_id(delivery_id)
    kind_token = (kind or "job").strip().lower()
    if not _RUN_ID_TOKEN.match(kind_token):
        kind_token = re.sub(r"[^a-z0-9_.-]+", "-", kind_token).strip("-") or "job"
    return f"gh-{kind_token}-{delivery_id}"


def webhook_brief(
    *,
    kind: str,
    repo_full_name: str,
    number: int | None,
    reason: str,
    extra: str = "",
) -> str:
    """The task brief a delivery asks for. PURE: the same delivery produces the same brief.

    Built from the event rather than from a model's summary of it, for the same reason the CI fix
    brief is: the brief is what the plan turn reads, so it must say what the event said. ``extra``
    carries event-specific evidence (a review comment's body, an issue's title) that the worker
    already extracted.
    """
    target = f"#{number}" if number else "the repository"
    lines = [
        f"Address this GitHub {kind} event on {repo_full_name} {target} ({reason}).",
        "",
        "Make the minimal correct change, run the repository's checks, and keep existing tests green.",
    ]
    if extra.strip():
        lines.extend(["", "Event details:", extra.strip()[:8000]])
    return "\n".join(lines)


def goal_for_webhook(*, objective: str, stop_condition: StopCondition | None = None) -> GoalSpec:
    """The `GoalSpec` an App-triggered run uses. One definition, so every trigger agrees.

    ``max_ci_attempts`` is set low on purpose: a delivery-triggered run has a human on the other end
    of a check suite, so three automatic repair attempts at somebody's PR is already generous.
    """
    return GoalSpec(
        objective=objective,
        stop_condition=stop_condition or _DEFAULT_STOP,
        max_tasks=4,
        max_ci_attempts=2,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    import json

    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


async def dispatch_webhook(
    client: Any,
    config: Any,
    dispatch: WebhookDispatch,
    *,
    run_timeout_s: int = 7 * 24 * 60 * 60,
) -> dict[str, Any]:
    """Start the run for this delivery, or ATTACH to the one already running for it.

    The three properties, and where each comes from:

    1. **One delivery, one run.** ``id=delivery_id`` plus ``USE_EXISTING``. A redelivery is answered by
       the run that already exists; the signal it carries is recorded by the workflow as a duplicate.
    2. **No window without the trigger.** ``start_signal`` delivers the trigger atomically with the
       start, so there is never an execution that exists without knowing which event woke it.
    3. **Nothing is silently dropped.** The report says whether this call STARTED a run or ATTACHED to
       one, and a caller that needed "started" can see it did not get it.
    """
    from temporalio.common import WorkflowIDConflictPolicy

    from .run_workflow import RunWorkflow

    dispatch.validate()
    try:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[dispatch.run_payload, None],
            id=dispatch.delivery_id,
            task_queue=config.task_queue,
            run_timeout=timedelta(seconds=run_timeout_s),
            start_signal=NAME_WEBHOOK_SIGNAL,
            start_signal_args=[dict(dispatch.signal_payload)],
            # ATTACH rather than refuse: a GitHub redelivery is a normal event, not an error, and the
            # second attempt must reach the run that is already working.
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with the delivery, which is the diagnosis
        raise WebhookTriggerError(
            f"could not dispatch delivery {dispatch.delivery_id}: {exc}",
            code="webhook_dispatch_failed",
            details={
                "delivery_id": dispatch.delivery_id,
                "kind": dispatch.kind,
                "task_queue": config.task_queue,
                "cause_code": str(getattr(exc, "code", "") or ""),
                "hint": "a duplicate id means the run already exists; check `codna durable status --run`",
            },
        ) from exc
    return {
        "delivery_id": dispatch.delivery_id,
        "workflow_id": dispatch.delivery_id,
        "run_id": str(dispatch.run_payload.get("run_id") or ""),
        "kind": dispatch.kind,
        "task_queue": config.task_queue,
        # A stable digest of what was asked for, so "this delivery asked for the same thing twice" is
        # answerable from the report.
        "payload_fingerprint": _fingerprint(dispatch.run_payload),
        # `handle` carries no "was this a start or an attach" flag; the workflow's own duplicate
        # detection (`WebhookSignal.received`) is the authoritative answer and the ledger holds it.
        "first_run_id": getattr(handle, "first_execution_run_id", None),
    }
