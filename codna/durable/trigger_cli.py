"""`codna run schedule` and `codna run trigger`: the two TRIGGER surfaces (D5).

Both are RUN DECLARATIONS, not a different kind of command — a schedule starts the same run a manual
invocation does, on a timetable, and a trigger starts it for a GitHub delivery. That is why they live
here, next to `run`, and why they share `run_payload_for` rather than re-deriving a payload: a
schedule that produced a payload the workflow would refuse to decode fails at its FIRST FIRE, often
hours later, which is the least observable possible moment.

Split out of `run_cli.py` at the repository's 1000-line ceiling (`test_modularity.py`), on a real
boundary: `run_cli` answers "how does a run begin interactively", this answers "how does a run begin
without a human". The shared helpers (`_cli._config`, `run_payload_for`, `_worktree_choice`) are
reached through their modules rather than copied, so there is still one definition of each.

Neither handler imports `temporalio` at module scope: the base CLI must answer `codna --help` without
the durable extra.

`run_cli` is reached LAZILY (inside the functions that need it) because it imports this module for the
registrations — a module-scope import back into it would close a cycle at import time, which fails the
whole durable surface rather than one command.
"""
from __future__ import annotations

from typing import Any

from . import cli as _cli
from .config import DurableConfig, validate_run_id

def _error(message: str, *, code: str, details: dict[str, object] | None = None, hint: str = "") -> Any:
    """The CLI's error type, reached through `run_cli` at CALL time.

    A module-scope import back into `run_cli` would close the cycle (`run_cli` imports this module for
    its registrations), so it is resolved per call. Reaching the module object rather than the class
    also keeps ONE definition: a test that swaps `cli.DurableCliError` still sees the change here.
    """
    from .run_cli import _error as _make

    return _make(message, code=code, details=details, hint=hint)


__all__ = [
    "_cmd_schedule_entry",
    "add_schedule_flags",
    "add_trigger_flags",
    "cmd_schedule",
    "cmd_schedule_list",
    "cmd_trigger",
    "schedule_definition_for",
]


# -- schedules (D5) ---------------------------------------------------------------------------
#
# `codna run schedule` registers a TEMPORAL SCHEDULE: the server owns the timetable, and a codna
# process that dies does not take it with it. The flags are the same run flags (a schedule starts the
# same run a manual invocation does), plus the trigger and a name.


def schedule_definition_for(args: Any, config: DurableConfig) -> Any:
    """Build and validate the schedule this invocation declares. Nothing is registered here.

    Split out (and pure) so the definition — the interval, the payload, the run id shape — can be
    asserted without a server or an event loop. A schedule is a thing an operator declares; making its
    validation reachable only through a live namespace would put the cheapest possible check behind
    the most expensive possible setup.
    """
    from .run_cli import _error, run_payload_for
    from .run_cli import _worktree_choice
    from .schedule import ScheduleDefinition, ScheduleError, schedule_id_for

    every = getattr(args, "every", None)
    cron = getattr(args, "cron", None)
    if (every is None) == (cron is None):
        raise _error(
            "a schedule needs exactly one of --every or --cron.",
            code="schedule_trigger_invalid",
            details={"every_seconds": every, "cron": cron},
        )
    name = getattr(args, "schedule_name", None)
    if not name:
        raise _error(
            "--name is required: it is the schedule's identity on the namespace.",
            code="schedule_name_required",
            details={"flag": "--name"},
        )
    repo_dir = _cli._repo_dir(args.repo)
    base_run_id = validate_run_id(args.run_id or _cli._new_run_id())
    payload = run_payload_for(
        config=config,
        run_id=base_run_id,
        repo_dir=repo_dir,
        briefs=list(args.brief or ()),
        base_branch=args.base_branch,
        verify_commands=list(args.verify or ()),
        max_cost_usd=args.max_cost,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        engine=getattr(args, "engine", None) or config.graph_engine,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        provider_key_source=getattr(args, "provider_key_source", None),
        worktree=_worktree_choice(args),
    )
    payload["run_id"] = f"{base_run_id}-scheduled"
    definition = ScheduleDefinition(
        schedule_id=schedule_id_for(name),
        every_seconds=every,
        cron=cron,
        run_payload=payload,
        jitter_seconds=int(getattr(args, "jitter", 0) or 0),
        run_timeout_s=int(args.run_timeout),
    )
    try:
        definition.validate()
    except ScheduleError as exc:
        raise _error(str(exc), code=exc.code, details=dict(exc.details)) from exc
    return definition


def cmd_schedule(args: Any) -> int:
    """`codna run schedule`: register (or update) this run as a Temporal Schedule.

    Idempotent by FINGERPRINT: an unchanged definition creates nothing and updates nothing, and a
    changed one replaces the schedule's action and spec wholesale. That is what makes re-running the
    command the way a user edits a timetable — rather than a delete-and-recreate they have to
    remember to do.
    """

    async def command() -> int:
        from .schedule import ScheduleError, ensure_schedule

        client = await _cli._connect(config)
        try:
            report = await ensure_schedule(client, config, definition)
        except ScheduleError as exc:
            raise _error(str(exc), code=exc.code, details=dict(exc.details)) from exc
        _cli._print(
            {
                "schedule": {
                    **report,
                    "every_seconds": definition.every_seconds,
                    "cron": definition.cron,
                    "run_id": definition.run_payload.get("run_id"),
                    "workflow_id_template": f"run:{definition.schedule_id}:{{{{.ScheduledTime}}}}",
                    "overlap_policy": "skip",
                    # The address is reported so "which server did I just register on" never has to
                    # be inferred from the environment.
                    "address": config.address,
                    "namespace": config.namespace,
                }
            }
        )
        return 0

    from . import require_durable

    require_durable()
    config = _cli._config(args)
    definition = schedule_definition_for(args, config)
    return int(_cli._in_loop(command))


def cmd_schedule_list(args: Any) -> int:
    """`codna run schedule --list`: every codna schedule on this namespace, with its next fire."""

    async def command() -> int:
        from .schedule import describe_schedule

        client = await _cli._connect(config)
        rows: list[dict[str, Any]] = []
        async for description in client.list_schedules():
            schedule_id = getattr(getattr(description, "id", None), "id", None) or getattr(
                description, "id", None
            )
            if not isinstance(schedule_id, str) or not schedule_id:
                continue
            rows.append(await describe_schedule(client, schedule_id))
        _cli._print({"schedules": rows, "address": config.address, "namespace": config.namespace})
        return 0

    from . import require_durable

    require_durable()
    config = _cli._config(args)
    return int(_cli._in_loop(command))


def cmd_trigger(args: Any) -> int:
    """`codna run trigger`: dispatch ONE GitHub delivery as a durable run (Signal-with-Start).

    The workflow id is the delivery id, so a redelivery ATTACHES to the run already working instead
    of starting a second. This is the local, single-delivery form of what the App's ingress does; it
    exists so the property is exercisable (and testable) without a live GitHub App.
    """
    from .run_cli import _error, run_payload_for, _worktree_choice
    from .trigger import (
        WebhookDispatch,
        WebhookTriggerError,
        dispatch_webhook,
        goal_for_webhook,
        run_id_for_delivery,
        webhook_brief,
    )

    async def command() -> int:
        client = await _cli._connect(config)
        try:
            report = await dispatch_webhook(client, config, dispatch)
        except WebhookTriggerError as exc:
            raise _error(str(exc), code=exc.code, details=dict(exc.details)) from exc
        _cli._print({"webhook": report})
        return 0

    from . import require_durable

    require_durable()
    config = _cli._config(args)
    repo_dir = _cli._repo_dir(args.repo)
    delivery = (getattr(args, "delivery_id", None) or "").strip()
    if not delivery:
        raise _error(
            "a trigger needs --delivery-id: it IS the workflow id, and what makes a redelivery safe.",
            code="webhook_delivery_required",
            details={"flag": "--delivery-id"},
        )
    kind = getattr(args, "kind", None) or "fix"
    brief = webhook_brief(
        kind=kind,
        repo_full_name=getattr(args, "repo_full_name", None) or "unknown/unknown",
        number=getattr(args, "number", None),
        reason="manual trigger",
    )
    run_id = run_id_for_delivery(delivery, kind)
    payload = run_payload_for(
        config=config,
        run_id=run_id,
        repo_dir=repo_dir,
        briefs=[brief],
        base_branch=args.base_branch,
        verify_commands=list(getattr(args, "verify", None) or ()),
        max_cost_usd=getattr(args, "max_cost", None),
        max_tokens=getattr(args, "max_tokens", None),
        max_attempts=getattr(args, "max_attempts", 3),
        engine=getattr(args, "engine", None) or config.graph_engine,
        worktree=_worktree_choice(args),
    )
    payload["goal"] = goal_for_webhook(objective=brief).to_payload()
    dispatch = WebhookDispatch(
        delivery_id=delivery,
        kind=kind,
        repo_full_name=getattr(args, "repo_full_name", None) or "unknown/unknown",
        run_payload=payload,
        signal_payload={
            "schema_version": 1,
            "delivery_id": delivery,
            "kind": kind,
            "repo_full_name": getattr(args, "repo_full_name", None) or "unknown/unknown",
            "number": getattr(args, "number", None),
        },
    )
    return int(_cli._in_loop(command))


def add_schedule_flags(schedule: Any) -> None:
    """The `schedule` flags: everything `run` takes, plus the TIMETABLE and the schedule's name.

    Sharing `add_run_flags` is deliberate — a schedule starts the same run a manual invocation does,
    so a flag added to one must not go missing from the other. The trigger flags are the schedule's
    own and are mutually exclusive by VALIDATION rather than by argparse (``--every 0`` must be a
    structured refusal with a code, not an argparse message that reads like a typo).
    """
    from .run_cli import add_run_flags

    # `brief_required=False` because `--list` takes no run declaration: the listing mode has no brief
    # to require, and the registration path enforces one through `run_payload_for`.
    add_run_flags(schedule)
    schedule.add_argument(
        "--name", dest="schedule_name",
        help="the schedule's identity on the namespace (required; used as `codna-schedule-<name>`)",
    )
    schedule.add_argument(
        "--every", type=int, default=None, metavar="SECONDS",
        help="fire every N seconds (at most a day); mutually exclusive with --cron",
    )
    schedule.add_argument(
        "--cron", default=None, metavar="EXPR",
        help="a cron expression, e.g. '0 9 * * 1-5'; mutually exclusive with --every",
    )
    schedule.add_argument(
        "--jitter", type=int, default=0, metavar="SECONDS",
        help="deterministic spread applied to each fire (default: 0, no spread)",
    )


def add_trigger_flags(trigger: Any) -> None:
    """The `trigger` flags: ONE delivery, described. The delivery id is what makes it idempotent."""
    trigger.add_argument(
        "--delivery-id", required=True,
        help="the GitHub X-GitHub-Delivery value; it IS the workflow id, and the idempotency key",
    )
    trigger.add_argument("--kind", default="fix", help="what the delivery asks for (default: fix)")
    trigger.add_argument(
        "--repo-full-name", default=None, metavar="OWNER/REPO",
        help="the repository the event names (required for a real dispatch)",
    )
    trigger.add_argument("--number", type=int, default=None,
                         help="the PR or issue number the event is about")
    trigger.add_argument("--repo", help="checkout to run against (default: the current directory)")
    trigger.add_argument("--base-branch", default="main", help="branch a task's work is based on")
    trigger.add_argument("--verify", action="append", default=[], metavar="CMD",
                         help="command that must pass to verify a task (repeatable)")
    trigger.add_argument("--max-cost", type=float, help="run-level cap, in USD")
    trigger.add_argument("--max-tokens", type=int, help="per-task token cap")
    trigger.add_argument("--max-attempts", type=int, default=3, help="plan/implement attempts per task")
    trigger.add_argument("--engine", choices=("langgraph", "plain"), help="graph engine")
    trigger.add_argument("--worktree", action="store_true",
                         help="give each task its own git worktree instead of the shared checkout")
    trigger.add_argument("--no-worktree", action="store_true",
                         help="work in the shared checkout (the default)")


def _cmd_schedule_entry(args: Any) -> int:
    """Route `codna run schedule`: register, or list when ``--list`` was given.

    One parser, two modes, chosen HERE rather than by a nested subparser group: the flags describe a
    RUN, and `--every` in a `schedule list` group would be a flag with no meaning in half the space.
    """
    if getattr(args, "list_schedules", False):
        return cmd_schedule_list(args)
    return cmd_schedule(args)

