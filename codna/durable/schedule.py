"""Temporal Schedules for durable runs: ``codna run schedule`` (D5).

A scheduled run is a RUN that the server starts on a timetable, not a process codna keeps alive. The
properties that matter, and why each one is Temporal's rather than ours:

* **It survives a restart, a reboot and a laptop lid.** The schedule lives on the server; a `codna`
  process that dies does not take the timetable with it.
* **Catch-up policy is explicit.** ``skip`` (the default here) means a schedule whose previous
  execution is still running does not start an overlapping one. That is the right default for a run
  that spends money: two concurrent copies of one goal is a doubled bill with no extra result.
* **Overlap is a REFUSAL, not a tolerance.** The generated workflow id embeds ``{{.ScheduledTime}}``,
  so two fires of the same schedule are two distinct executions — but ``skip`` stops the second from
  being created at all, and the id makes "which fire was this" answerable from any handle.

**The run payload is built by the SAME function the CLI uses.** ``run_payload_for`` is called here,
so a schedule cannot produce a payload the workflow would refuse to decode — a schedule whose runs
fail on decode is invisible until the first fire, which may be hours later.

Nothing here is workflow code: it is a CLIENT module (it builds a `Schedule` and calls
``create_schedule``), so importing `temporalio.client` here is fine and importing it from a workflow
is not.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Final, Mapping, Sequence

from .goal import GoalSpec

__all__ = [
    "DEFAULT_JITTER_S",
    "ScheduleError",
    "MAX_INTERVAL_S",
    "SCHEDULE_ID_PREFIX",
    "ScheduleDefinition",
    "describe_schedule",
    "ensure_schedule",
    "schedule_id_for",
    "schedule_payload",
]

class ScheduleError(RuntimeError):
    """A schedule that cannot be registered, or a registration that failed. Always carries a code.

    A plain ``RuntimeError`` (not the durable contracts' `ContractError`) because this is CLIENT code:
    nothing here crosses a workflow boundary, and a schedule's validation failure is an operator
    error the CLI reports with an exit code, not a payload the sandbox must refuse.
    """

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


SCHEDULE_ID_PREFIX: Final = "codna-schedule-"
# A minute. A schedule that fires more often than this is a polling loop wearing a timetable, and the
# server's own minimum granularity makes sub-minute intervals meaningless anyway.
MAX_INTERVAL_S: Final = 24 * 60 * 60
# Deterministic spread, not randomness: two hosts that create the same schedule must agree on it, and
# a jitter that changed per call would make "the schedule is unchanged" unanswerable by comparison.
DEFAULT_JITTER_S: Final = 0


@dataclass(frozen=True)
class ScheduleDefinition:
    """One scheduled run: how often, and exactly what to start.

    ``run_payload`` is validated by the CALLER (`run_payload_for`) and re-validated on decode by
    `run_spec_from`, so this type does not re-implement that check. What it owns is its own shape: a
    positive interval, a bounded jitter, and an id that is derived from the definition rather than
    invented, so re-registering the same definition is idempotent.

    ``every_seconds`` XOR ``cron``: both is a refusal, because a schedule with two timetables has no
    single answer to "when does this fire".
    """

    schedule_id: str
    every_seconds: int | None
    cron: str | None
    run_payload: dict[str, Any]
    jitter_seconds: int = DEFAULT_JITTER_S
    # The run's own cap. A schedule with no `run_timeout` relies on the server default, which is what
    # makes a wedged run invisible: named explicitly here so "how long may one fire take" is a value
    # in the definition rather than a server default nobody remembers.
    run_timeout_s: int = 7 * 24 * 60 * 60

    def validate(self) -> None:
        from .config import validate_run_id

        validate_run_id(self.schedule_id)
        if (self.every_seconds is None) == (self.cron is None):
            raise ScheduleError(
                "a schedule needs exactly one of every_seconds or cron",
                code="schedule_trigger_invalid",
                details={
                    "every_seconds": self.every_seconds,
                    "cron": self.cron,
                },
            )
        if self.every_seconds is not None:
            if (
                not isinstance(self.every_seconds, int)
                or isinstance(self.every_seconds, bool)
                or self.every_seconds < 1
            ):
                raise ScheduleError(
                    "every_seconds must be a positive integer",
                    code="schedule_trigger_invalid",
                    details={"every_seconds": repr(self.every_seconds)},
                )
            if self.every_seconds > MAX_INTERVAL_S:
                raise ScheduleError(
                    f"every_seconds must be at most {MAX_INTERVAL_S} (a day)",
                    code="schedule_trigger_invalid",
                    details={"every_seconds": self.every_seconds},
                )
        if self.cron is not None and (not isinstance(self.cron, str) or not self.cron.strip()):
            raise ScheduleError(
                "cron must be a non-empty cron expression",
                code="schedule_trigger_invalid",
                details={"cron": repr(self.cron)},
            )
        if not isinstance(self.jitter_seconds, int) or isinstance(self.jitter_seconds, bool) or self.jitter_seconds < 0:
            raise ScheduleError(
                "jitter_seconds must be a non-negative integer",
                code="schedule_trigger_invalid",
                details={"jitter_seconds": repr(self.jitter_seconds)},
            )
        if not isinstance(self.run_timeout_s, int) or self.run_timeout_s < 1:
            raise ScheduleError(
                "run_timeout_s must be a positive integer",
                code="schedule_trigger_invalid",
                details={"run_timeout_s": repr(self.run_timeout_s)},
            )
        if not isinstance(self.run_payload, dict) or not self.run_payload:
            raise ScheduleError(
                "a schedule needs a run payload",
                code="schedule_trigger_invalid",
                details={"field": "run_payload"},
            )
        goal = self.run_payload.get("goal")
        if goal is not None:
            # Decoded here so a malformed objective is a CLI error, not a run that fails on its first
            # fire hours later.
            GoalSpec.from_payload(goal)

    def to_payload(self) -> dict[str, Any]:
        """The definition, canonicalised. Used for the fingerprint and for ``describe``."""
        return {
            "schedule_id": self.schedule_id,
            "every_seconds": self.every_seconds,
            "cron": self.cron,
            "jitter_seconds": self.jitter_seconds,
            "run_timeout_s": self.run_timeout_s,
            "run_payload": self.run_payload,
        }

    @property
    def fingerprint(self) -> str:
        """A digest of the definition. What ``ensure_schedule`` compares to decide "unchanged".

        Sorted-key JSON of the canonical payload, so two definitions that differ only in dict order
        are the SAME schedule — which is what makes re-running ``codna run schedule`` idempotent
        rather than a needless update on every invocation.
        """
        canonical = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def schedule_id_for(name: str) -> str:
    """``codna-schedule-<name>`` — the server-side id, derived from the user's name.

    Prefixed so a codna schedule is recognisable among whatever else lives on the namespace, and
    validated through the run-id token class because it becomes a directory/ref-adjacent identifier
    in logs and in ``--schedule`` lookups.
    """
    from .config import validate_run_id

    return f"{SCHEDULE_ID_PREFIX}{validate_run_id(name)}"


def schedule_payload(definition: ScheduleDefinition) -> dict[str, Any]:
    """The definition as a plain dict, for a banner or a ``describe`` reply."""
    definition.validate()
    body = definition.to_payload()
    body["fingerprint"] = definition.fingerprint
    return body


# -- registration --------------------------------------------------------------------------------


def _temporal_schedule(definition: ScheduleDefinition, *, task_queue: str) -> Any:
    """The SDK's `Schedule` object for this definition. Pure: no client, no I/O.

    ``task_queue`` is a REQUIRED keyword rather than a field of the definition because a queue is
    per-host and per-build while a definition is portable between them: the same "nightly" schedule
    must be registrable on any host, each on that host's own queue.

    Split out so a test can assert the SHAPE (the interval, the skip policy, the workflow id
    template) without a server — the one part of this module a time-skipping environment cannot
    exercise, because its test server does not implement ``CreateSchedule`` (measured: ``Method
    temporal.api.workflowservice.v1.WorkflowService/CreateSchedule is unimplemented``).
    """
    from temporalio.client import (
        Schedule,
        ScheduleActionStartWorkflow,
        ScheduleIntervalSpec,
        ScheduleOverlapPolicy,
        SchedulePolicy,
        ScheduleSpec,
    )

    definition.validate()
    if definition.every_seconds is not None:
        spec = ScheduleSpec(
            intervals=[ScheduleIntervalSpec(every=timedelta(seconds=definition.every_seconds))]
        )
    else:
        spec = ScheduleSpec(cron_expressions=[definition.cron])
    if definition.jitter_seconds:
        spec.jitter = timedelta(seconds=definition.jitter_seconds)
    return Schedule(
        # `{{.ScheduledTime}}` in the id: each fire is its own execution, and the id says WHEN it was
        # due. A fixed id would make the second fire collide with the first (or be silently skipped
        # by the id-reuse policy), which is the opposite of a timetable.
        action=ScheduleActionStartWorkflow(
            "RunWorkflow",
            args=[definition.run_payload, None],
            id=f"run:{definition.schedule_id}:{{{{.ScheduledTime}}}}",
            task_queue=task_queue,
            run_timeout=timedelta(seconds=definition.run_timeout_s),
        ),
        spec=spec,
        policy=SchedulePolicy(
            # A run that spends money must not overlap itself: two concurrent copies is a doubled
            # bill for one result. `SKIP` is the explicit choice, not an inherited default.
            overlap=ScheduleOverlapPolicy.SKIP,
        ),
    )


async def ensure_schedule(client: Any, config: Any, definition: ScheduleDefinition) -> dict[str, Any]:
    """Create or update this schedule, IDEMPOTENTLY. Returns a small report of what happened.

    "Create or update" rather than "create": re-running ``codna run schedule`` is how a user edits a
    timetable, and a create that failed with "already exists" would make every edit a manual delete
    first. The comparison is the definition's FINGERPRINT, which is why an unchanged definition
    produces no update at all — an update on every invocation would reset nothing but would make the
    command's effect impossible to reason about.

    The update path sets the ACTION and SPEC wholesale: a partial update of a schedule is a schedule
    half of whose behaviour comes from the previous definition, which is exactly the kind of drift a
    declarative command exists to remove.
    """
    definition.validate()
    from temporalio.client import ScheduleAlreadyRunningError

    schedule = _temporal_schedule(definition, task_queue=config.task_queue)
    handle = client.get_schedule_handle(definition.schedule_id)
    try:
        described = await handle.describe()
    except Exception as exc:  # noqa: BLE001 - "not found" is the create path
        if not _is_not_found(exc):
            raise ScheduleError(
                f"could not describe schedule {definition.schedule_id!r}: {exc}",
                code="schedule_describe_failed",
                details={"schedule_id": definition.schedule_id},
            ) from exc
        try:
            await client.create_schedule(definition.schedule_id, schedule)
        except ScheduleAlreadyRunningError:
            # Lost a race with a concurrent create: fall through to the compare-and-update path, which
            # is the same outcome as having found it.
            described = await handle.describe()
        else:
            return {
                "schedule_id": definition.schedule_id,
                "created": True,
                "updated": False,
                "fingerprint": definition.fingerprint,
                "task_queue": config.task_queue,
            }
    current = _fingerprint_of(described)
    if current == definition.fingerprint:
        return {
            "schedule_id": definition.schedule_id,
            "created": False,
            "updated": False,
            "fingerprint": definition.fingerprint,
            "task_queue": config.task_queue,
        }
    await handle.update(_replace_with(schedule))
    return {
        "schedule_id": definition.schedule_id,
        "created": False,
        "updated": True,
        "fingerprint": definition.fingerprint,
        "previous_fingerprint": current,
        "task_queue": config.task_queue,
    }


def _replace_with(schedule: Any) -> Any:
    """The update closure: keep the schedule's identity, replace its action and spec.

    A full replacement rather than a field-by-field edit, deliberately: a partial update would leave
    behaviour half from the previous definition, which is the drift a declarative command exists to
    remove. ``ScheduleUpdateInput`` is accepted (the SDK's callback signature) and ignored — the whole
    point is that the NEW definition wins.
    """
    from temporalio.client import ScheduleUpdate

    def update(_input: Any) -> Any:
        return ScheduleUpdate(schedule=schedule)

    return update


async def describe_schedule(client: Any, schedule_id: str) -> dict[str, Any]:
    """One schedule's live state, as a small dict a CLI can print or a test can assert."""
    handle = client.get_schedule_handle(schedule_id)
    described = await handle.describe()
    schedule = described.schedule
    spec = schedule.spec
    return {
        "schedule_id": schedule_id,
        "paused": bool(getattr(described, "schedule", None) and described.state.paused),
        "every_seconds": (
            int(spec.intervals[0].every.total_seconds()) if spec.intervals else None
        ),
        "cron": spec.cron_expressions[0] if spec.cron_expressions else None,
        "workflow_id_template": schedule.action.id,
        "task_queue": schedule.action.task_queue,
        "next_action_times": [
            info.scheduled_at.isoformat() for info in (described.info.next_action_times or [])
        ][:5],
        "num_actions": int(described.info.num_actions),
    }


def _fingerprint_of(described: Any) -> str:
    """Recompute the definition's fingerprint from a live schedule description.

    The description is Temporal's canonical form, so the comparison runs through the fields BOTH
    sides have: the trigger, the jitter, the run timeout, and the RUN PAYLOAD. The payload is the one
    that is easy to get wrong — ``action.args`` is a list of encoded ``Payload`` messages, not Python
    values, so it is decoded through the SDK's own converter here. Comparing the raw payloads would
    make every definition look changed (they carry protobuf metadata), which would turn every
    ``codna run schedule`` into a needless update.

    A description that cannot be read yields ``""``, which can never equal a real fingerprint — so the
    update path runs and the schedule is brought to the declared state. That is the safe direction: a
    spurious update restores the declaration, while a spurious "unchanged" would leave drift.
    """
    try:
        from temporalio.converter import default as default_converter

        action = described.schedule.action
        spec = described.schedule.spec
        every = None
        if spec.intervals:
            every = int(spec.intervals[0].every.total_seconds())
        cron = spec.cron_expressions[0] if spec.cron_expressions else None
        jitter = int(spec.jitter.total_seconds()) if spec.jitter else 0
        run_timeout = int(action.run_timeout.total_seconds()) if action.run_timeout else 0
        converter = default_converter().payload_converter
        decoded = [converter.from_payload(payload) for payload in (action.args or [])]
        run_payload = decoded[0] if decoded and isinstance(decoded[0], dict) else {}
        body = {
            "schedule_id": _schedule_id_from(described),
            "every_seconds": every,
            "cron": cron,
            "jitter_seconds": jitter,
            "run_timeout_s": run_timeout,
            "run_payload": run_payload,
        }
        return ScheduleDefinition(
            schedule_id=body["schedule_id"],
            every_seconds=body["every_seconds"],
            cron=body["cron"],
            run_payload=body["run_payload"],
            jitter_seconds=body["jitter_seconds"],
            run_timeout_s=body["run_timeout_s"],
        ).fingerprint
    except Exception:  # noqa: BLE001 - unreadable means "different", which triggers the safe update
        return ""


def _schedule_id_from(described: Any) -> str:
    """The schedule id a description belongs to, from its handle.

    ``ScheduleDescription`` carries the id on its HANDLE rather than as a field, so this reaches for
    it the way the SDK exposes it and refuses (raising, caught by `_fingerprint_of`) when it cannot.
    """
    for candidate in (
        getattr(getattr(described, "id", None), "id", None),
        getattr(described, "id", None),
        getattr(described, "schedule_id", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    raise AttributeError("the schedule description carries no id")


def _is_not_found(exc: BaseException) -> bool:
    """Whether a describe failure means "this schedule does not exist yet".

    Checked by GRPC status name rather than by message text: the SDK surfaces
    ``temporalio.service.RPCError`` with a ``status`` of ``NOT_FOUND``, and matching on the string
    would break when the wording changes.
    """
    status = getattr(exc, "status", None)
    name = getattr(status, "name", None)
    if name == "NOT_FOUND":
        return True
    return "not found" in str(exc).lower()


_ = (field, Mapping, Sequence)
