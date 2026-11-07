"""The run and worker commands: start a run, execute it, wait for it.

Split out of `cli.py` for the repository's 1000-line ceiling (`tests/test_modularity.py`), by
concern rather than by size: everything that STARTS or EXECUTES work lives here, and `cli.py`
keeps the commands that OBSERVE or STEER an already-started run (status, approvals, control,
ledger) plus the parser registration. A reader looking for "how does a run begin" has one file.

**The two entrances, one implementation.** ``codna durable run`` (start only, beside a worker) and
``codna run`` (start AND execute, top level) are the same command with one default flipped. They
share `run_payload_for` and `_start_run`, so a flag cannot exist on one surface and not the other.

**Nothing here imports `temporalio` at module scope.** The base CLI must answer ``codna --help``
without the durable extra, so the client and the worker are imported inside the functions that use
them. `run_payload_for` is deliberately extra-free: it validates a run spec before anything is
started, and it is what the base suite exercises.

**Shared helpers are reached through the `cli` module object**, not copied. ``cli._config``,
``cli._connect``, ``cli._print`` and friends are the CLI's one definition of "resolve the config",
"connect", "print JSON", and the tests substitute them by name. Importing them would freeze the
binding at import time and silently defeat those substitutions; ``from . import cli as _cli`` keeps
one object and one seam.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from . import cli as _cli
from .config import DurableConfig, validate_run_id
from .errors import cli_error
# The wait loop and the exit-code contract live in their own module (see `run_wait.py`); these
# names are RE-EXPORTED because every caller of `codna durable run --wait` addresses this module,
# and because `cli.py` exposes them as the CLI's test seams.
from .run_wait import (  # noqa: F401 - re-exported
    EXIT_NOT_SUCCESS,
    _exit_for_closed_run,
    _run_is_finished,
    _wait_run,
    query_workflow,
    workflow_completion,
    workflow_failure_report,
    workflow_outcome,
)
# The closed set of credential sources, imported at module scope: `contracts` is a sandbox-clean leaf
# (stdlib + dataclasses), so this keeps `run_cli` importable without the durable extra — and it means
# a `--provider-key-source` the contract would refuse cannot be typed into the parser at all.
from .contracts import PROVIDER_KEY_SOURCES
from .swarm import SwarmFlagError, add_delegation_flags, swarm_choice
# The Run API's backend seam, for the `serve` command. Imported at module scope (unlike the server
# itself) because `_LedgerOnlyBackend` subclasses it and because it imports nothing heavier than
# `config` — which this module already imports.
# `serve` lives in its own module for the ceiling; re-exported so `cli.py`'s registration and the
# tests keep reaching ONE name.
# `_LedgerOnlyBackend` is re-exported for `test_run_api.py`, which asserts the ledger-only
# server's answers through `run_cli` (the module that used to own it).
from .serve_cli import (  # noqa: F401 - re-exported
    _LedgerOnlyBackend,
    add_serve_flags,
    cmd_serve,
)
# The trigger surfaces (`run schedule` / `run trigger`) live in their own module for the ceiling; the
# registrations below reach them through here so both parser surfaces still register ONE command.
from .trigger_cli import (
    _cmd_schedule_entry,
    add_schedule_flags,
    add_trigger_flags,
    cmd_trigger,
)
# `run verify-ledger` lives in its own module: it only READS a ledger, which is not part of starting
# or executing a run, and it must PRINT divergence rather than raise it.
from .verify_ledger_cli import add_verify_ledger_flags, cmd_verify_ledger

__all__ = [
    "DEFAULT_RUN_TIMEOUT_S",
    "DEFAULT_WAIT_POLL_S",
    "add_run_flags",
    "add_serve_flags",
    "add_worker_flags",
    "cmd_run",
    "cmd_serve",
    "cmd_verify_ledger",
    "cmd_worker",
    "query_workflow",
    "register_top_level",
    "run_payload_for",
]

# Re-exported from `cli.py` so a caller importing "the run command's defaults" gets them here too.
# A run may legitimately park on a human for a long time; the workflow's own `run_timeout` (set from
# `--run-timeout`) is the real bound. This is only the default for the CLI's own `--wait` poll.
DEFAULT_RUN_TIMEOUT_S: Final = 7 * 24 * 60 * 60
DEFAULT_WAIT_POLL_S: Final = 2.0

# Exit code for "the command ran, the run is not in a success state". Defined in `run_wait.py` (which
# computes it) and re-exported above, so there is ONE number rather than two that agree for now.
# `Final` is not re-stated here: this name is an import, not a second definition.


def _error(message: str, *, code: str, details: dict[str, object] | None = None, hint: str = "") -> Any:
    """The CLI's error type, from the ONE definition in `errors.py`.

    Bound here as a module-level alias rather than re-implemented: `run_cli` already imports `errors`
    for the class, and a second copy is how the hint or the code style drifts between commands.
    """
    return cli_error(message, code=code, details=details, hint=hint)


# -- the run spec --------------------------------------------------------------------------


def run_payload_for(
    *,
    config: DurableConfig,
    run_id: str,
    repo_dir: Path,
    briefs: Sequence[str],
    base_branch: str,
    verify_commands: Sequence[str],
    max_cost_usd: float | None,
    max_tokens: int | None,
    max_attempts: int,
    engine: str,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    provider_key_source: str | None = None,
    worktree: bool = False,
    mcp_servers: Sequence[Mapping[str, Any]] = (),
    enable_spawn_agent: bool = False,
    enable_agent_teams: bool = False,
    max_parallel_tasks: int = 1,
    worktree_branch: str = "",
) -> dict[str, Any]:
    """The `RunSpec` payload for one invocation, validated BEFORE anything is started.

    Building and validating locally (not letting the workflow fail on decode) means a typo in a
    brief or a relative repo path is a CLI error with an exit code, not a started workflow that
    fails inside Temporal's history.

    ``worktree`` gives every task its own checkout (``<runs_root>/<run>/wt-<task>``, branch
    ``codna/run-<run>/task-<n>``) instead of the shared ``repo_dir``. Default ``False`` — see
    `add_run_flags` for why the default is the shared checkout, and what ``--no-worktree`` means.
    """
    from .contracts import RunSpec, TaskSpec

    if not briefs:
        raise _error(
            "a run needs at least one --brief.", code="durable_brief_required", details={"flag": "--brief"}
        )
    if not 1 <= len(briefs) <= 50:
        raise _error(
            f"a run supports 1..50 tasks (got {len(briefs)}).",
            code="durable_task_count_invalid",
            details={"tasks": len(briefs)},
        )
    if not base_branch or not base_branch.strip():
        raise _error(
            "base branch must not be empty.", code="durable_base_branch_invalid", details={}
        )
    tasks = tuple(
        TaskSpec(
            run_id=run_id,
            task_id=f"task-{index}",
            task_number=index,
            brief=brief,
            repo_dir=str(repo_dir),
            base_branch=base_branch,
            verify_commands=tuple(verify_commands),
            max_attempts=max_attempts,
            max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
            engine=engine,  # type: ignore[arg-type]  # validated by RunSpec.validate below
            provider=provider,
            model=model,
            base_url=base_url,
            provider_key_source=provider_key_source,
            worktree=worktree,
            mcp_servers=tuple(dict(server) for server in mcp_servers),
            enable_spawn_agent=enable_spawn_agent,
            enable_agent_teams=enable_agent_teams,
            worktree_branch=worktree_branch.strip(),
        )
        for index, brief in enumerate(briefs, start=1)
    )
    spec = RunSpec(
        run_id=run_id,
        repo_dir=str(repo_dir),
        base_branch=base_branch,
        tasks=tasks,
        max_cost_usd=max_cost_usd,
        engine=engine,  # type: ignore[arg-type]
        # The RUN-level flag decides whether the run sweeps its worktrees at the end, so it has to be
        # set here and not only on each task.
        worktree=worktree,
        # D7/MULTITASK: validated against `worktree` by `RunSpec.validate` (concurrent tasks need one
        # checkout each), so a `--parallel 4` without `--worktree` is a CLI error here.
        max_parallel_tasks=max_parallel_tasks,
    )
    try:
        spec.validate()
    except Exception as exc:  # noqa: BLE001 - ContractError and friends, re-shaped for the CLI
        raise _error(
            f"the run specification is invalid: {exc}",
            code=str(getattr(exc, "code", "durable_run_spec_invalid")),
            details={"cause": getattr(exc, "details", None) or {}},
        ) from exc
    payload = {
        "schema_version": spec.schema_version,
        "run_id": spec.run_id,
        "repo_dir": spec.repo_dir,
        "base_branch": spec.base_branch,
        "tasks": [task.to_payload() for task in spec.tasks],
        "max_cost_usd": spec.max_cost_usd,
        "engine": spec.engine,
        "idle_between_tasks_s": spec.idle_between_tasks_s,
        "continue_as_new_tasks": spec.continue_as_new_tasks,
        "continue_as_new_history_events": spec.continue_as_new_history_events,
        "worktree": spec.worktree,
        "max_parallel_tasks": spec.max_parallel_tasks,
    }
    _ = config  # the payload is config-independent today; the parameter keeps call sites explicit
    return payload


# -- run -----------------------------------------------------------------------------------


def _pin_choice(args: Any) -> bool | None:
    """Translate ``--pin``/``--no-pin`` into the tri-state `_start_run` takes.

    Default ``None`` = "pin when a matching worker is registered", which is the behaviour that makes
    a build bump safe without breaking a worker that never registered a deployment version. The two
    flags exist for the cases where that automatic answer is wrong: ``--no-pin`` for a plain worker,
    ``--pin`` when a run MUST be owned by this build or not start at all.
    """
    if getattr(args, "no_pin", False) and getattr(args, "pin", False):
        raise _error(
            "--pin and --no-pin are mutually exclusive.",
            code="durable_pin_conflict",
            details={"flags": ["--pin", "--no-pin"]},
        )
    if getattr(args, "no_pin", False):
        return False
    if getattr(args, "pin", False):
        return True
    return None


def _worktree_choice(args: Any) -> bool:
    """Whether this run gives each task its own worktree (``--worktree`` / ``--no-worktree``).

    The two spellings are a TRI-STATE on purpose, exactly like ``--pin``/``--no-pin``: ``None`` means
    the flag was not given and the default applies, ``True``/``False`` are explicit choices. Both at
    once is a refusal rather than "last one wins", because a script that passes both is a script whose
    author does not know which checkout it will edit.

    **The default is ``--no-worktree``** (the shared checkout), deliberately:

    * it is the behaviour every existing run already has, so the default changes nothing for a caller
      who did not ask;
    * a worktree costs a full checkout of the repository on disk for every task, which is the wrong
      thing to impose on a run pointed at a large repository;
    * it is the only mode in which the checkpoint's pre-D5 branch behaviour is byte-identical, so the
      replay cassette and the in-flight runs are unaffected.

    ``--no-worktree`` means exactly one thing: each task works in, commits to, verifies in and reviews
    ``repo_dir`` itself — the SHARED checkout, as before. Tasks then serialise on that one tree, so two
    tasks cannot be worked on at once and task 2 starts from whatever task 1 left behind. It does NOT
    create a temporary checkout somewhere else, and it does NOT disable the task branches
    (``codna/run-<run>/task-<n>``): those are still created by the first checkpoint, in the shared tree.
    """
    if getattr(args, "worktree", False) and getattr(args, "no_worktree", False):
        raise _error(
            "--worktree and --no-worktree are mutually exclusive.",
            code="durable_worktree_conflict",
            details={"flags": ["--worktree", "--no-worktree"]},
        )
    if getattr(args, "no_worktree", False):
        return False
    return bool(getattr(args, "worktree", False))


def _swarm_choice(args: Any, capability: str) -> bool:
    """`swarm.swarm_choice`, translated into this package's structured CLI error (see `swarm.py`)."""
    try:
        return swarm_choice(args, capability)
    except SwarmFlagError as exc:
        raise _error(str(exc), code=exc.code, details=exc.details) from exc


def _worker_choice(args: Any) -> bool:
    """Whether this invocation runs the worker in-process.

    ``--worker`` and ``--no-worker`` are explicit; with neither, the TOP-LEVEL spelling runs a worker
    (``codna run`` with nothing else polling would otherwise start a run and sit forever) and the
    ``durable`` group's spelling does not (it is the lower-level surface used beside a worker).
    """
    if getattr(args, "worker", False) and getattr(args, "no_worker", False):
        raise _error(
            "--worker and --no-worker are mutually exclusive.",
            code="durable_worker_conflict",
            details={"flags": ["--worker", "--no-worker"]},
        )
    if getattr(args, "worker", False):
        return True
    if getattr(args, "no_worker", False):
        return False
    return bool(getattr(args, "run_worker_by_default", False))


def _require_repository_policy(repo_dir: Path) -> None:
    """Refuse to start a durable run whose repository policy does not validate.

    FAIL CLOSED: this raises for every non-allowing verdict, including a MISSING file. A durable run
    is unattended, so "no policy" is not a neutral state — it is the case where an agent would run
    for hours under whatever the approval profile happens to permit. The error carries the reader's
    own code (``policy_unknown_key``, ``policy_unparseable``, …) so the fix is obvious.

    The sidecar enforces the same file again on the turn path. Two consumers, one implementation
    (`codna.policy_document`): this call cannot disagree with the sidecar's verdict because it is the
    same function over the same bytes.
    """
    from ..policy_document import policy_path_for, policy_verdict

    policy_path = policy_path_for(repo_dir)
    verdict = policy_verdict(policy_path, required=True)
    if verdict.allowed:
        return
    raise _error(
        f"the repository policy does not permit this run: {verdict.reason}",
        code=verdict.code,
        details={
            "policy_path": str(policy_path),
            "policy_mode": verdict.mode,
            "policy_digest": verdict.policy_digest,
            **{k: v for k, v in verdict.details.items() if k != "source"},
        },
        hint=(
            f"add or fix {policy_path}; a durable run requires a policy that permits something, "
            "and an unparseable or unknown-key document is refused rather than ignored"
        ),
    )


def _run_start_error(run_id: str, config: DurableConfig, exc: BaseException) -> Any:
    """The one structured error for every way ``start_workflow`` can fail."""
    return _error(
        f"could not start run {run_id}: {exc}",
        code="durable_run_start_failed",
        details={
            "run_id": run_id,
            "workflow_id": config.run_workflow_id(run_id),
            "task_queue": config.task_queue,
            "cause_code": str(getattr(exc, "code", "") or ""),
        },
        hint="a duplicate workflow id means the run already exists; use `codna durable status`",
    )


async def _start_run(
    client: Any,
    config: DurableConfig,
    payload: Mapping[str, Any],
    *,
    run_id: str,
    run_timeout_s: int,
    pin_build: bool | None = None,
) -> Any:
    """Start the run workflow, PINNED to this build when a worker is actually polling it.

    ``pin_build=None`` (the default) asks the server: if the task queue has a poller registered under
    ``config.build_id``, the run is pinned to that version — so a build bump cannot migrate a run that
    is already mid-task. Otherwise the run starts unpinned, which is the only form an unversioned
    worker accepts (a pinned start there is refused outright). ``pin_build=False`` skips the check;
    ``True`` demands a pin and fails loudly when no matching build is registered.
    """
    from datetime import timedelta

    from .run_workflow import RunWorkflow
    from .worker import WorkerError, registered_build_ids, worker_settings_from_config

    versioning: Any | None = None
    pinned = False
    pin_report: dict[str, object] = {"requested": pin_build is not False}
    if pin_build is not False:
        try:
            settings = worker_settings_from_config(config, ensure_sidecar=False)
            builds = await registered_build_ids(client, settings)
        except WorkerError as exc:
            # "We cannot tell which builds are polling" is not "this run cannot start". Pinning is a
            # rollout SAFETY feature, not a precondition: an older server (the time-skipping test
            # server, a pre-1.24 cluster) has no DescribeTaskQueue, and failing there would make
            # `codna run` unusable against a perfectly working worker. Report it, start unpinned —
            # unless the caller DEMANDED a pin, in which case their requirement cannot be met.
            if pin_build is True:
                raise _run_start_error(run_id, config, exc) from exc
            pin_report.update({"pinned": False, "reason": exc.code, "cause": exc.details})
            builds = set()
        except Exception as exc:  # noqa: BLE001 - same reasoning as above
            if pin_build is True:
                raise _run_start_error(run_id, config, exc) from exc
            pin_report.update({"pinned": False, "reason": "inspection_failed", "cause": str(exc)})
            builds = set()
        if config.build_id in builds:
            from temporalio.common import PinnedVersioningOverride
            from temporalio.worker import WorkerDeploymentVersion

            versioning = PinnedVersioningOverride(
                WorkerDeploymentVersion(deployment_name="codna", build_id=config.build_id)
            )
            pinned = True
        elif pin_build is True:
            raise _error(
                f"no worker is registered on {config.task_queue!r} under build {config.build_id!r}, "
                "so the run cannot be pinned to it.",
                code="durable_run_pin_unavailable",
                details={
                    "run_id": run_id,
                    "task_queue": config.task_queue,
                    "build_id": config.build_id,
                    "registered_build_ids": sorted(builds),
                },
                hint="start `codna worker` first, or pass --no-pin to start the run unpinned",
            )
    try:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[dict(payload), None],
            id=config.run_workflow_id(run_id),
            task_queue=config.task_queue,
            run_timeout=timedelta(seconds=run_timeout_s),
            versioning_override=versioning,
        )
    except Exception as exc:  # noqa: BLE001 - e.g. a duplicate workflow id
        error = _run_start_error(run_id, config, exc)
        error.details["pinned"] = pinned
        error.details["build_id"] = config.build_id
        raise error from exc
    # The server's handle is what callers expect back (and what a test may poke at), so the pin
    # outcome rides along on it rather than changing the return type. A run that started unpinned
    # when a pin was expected is a fact a rollout must be able to SEE, not infer.
    pin_report["pinned"] = pinned
    pin_report["build_id"] = config.build_id
    setattr(handle, "codna_pin_report", pin_report)  # noqa: B010 - attribute on a foreign object
    return handle


async def _run_with_worker(
    client: Any,
    config: DurableConfig,
    args: Any,
    payload: Mapping[str, Any],
    *,
    run_id: str,
    banner: Mapping[str, Any],
    pin_build: bool | None,
) -> int:
    """Start a worker in THIS process, start the run, wait for it, then drain the worker.

    The top-level ``codna run`` contract: one command that executes what it starts. The worker is
    awaited to "started" before the run is started, so the run cannot be dispatched to a queue nobody
    polls; it is drained after the run reaches a terminal outcome (or the command is interrupted), so
    an in-flight activity finishes instead of being abandoned and re-billed.

    Ordering matters twice: ``on_started`` fires only after the deployment version is registered (so
    the pin check in `_start_run` can see this build), and the drain happens in ``finally`` so a failed
    or interrupted run still stops the worker cleanly.
    """
    import asyncio

    from .worker import run_worker, worker_settings_from_config

    settings = worker_settings_from_config(
        config,
        ensure_sidecar=not getattr(args, "no_sidecar", False),
        grace_period_s=getattr(args, "grace_period", 10.0),
        promote_current=not getattr(args, "no_promote", False),
    )
    stop = asyncio.Event()
    started = asyncio.Event()

    async def worker_main() -> None:
        await run_worker(
            settings,
            stop=stop,
            install_signal_handlers=True,
            client=client,
            on_started=lambda _banner: started.set(),
        )

    worker_task = asyncio.ensure_future(worker_main())
    try:
        try:
            await asyncio.wait_for(started.wait(), timeout=max(args.start_timeout, 1.0))
        except asyncio.TimeoutError:
            raise _error(
                f"the worker did not finish registering within {args.start_timeout:g}s.",
                code="durable_worker_start_timeout",
                details={"task_queue": config.task_queue, "build_id": config.build_id},
                hint="run `codna worker --start-dev-server` in another terminal and inspect its output",
            ) from None
        await _start_run(
            client,
            config,
            payload,
            run_id=run_id,
            run_timeout_s=args.run_timeout,
            pin_build=pin_build,
        )
        _cli._print(banner)
        return await _wait_run(
            client,
            config,
            run_id,
            poll_s=args.poll_interval,
            timeout_s=args.wait_timeout,
            stop=stop,
        )
    finally:
        stop.set()
        with contextlib.suppress(BaseException):
            await worker_task


def cmd_run(args: Any) -> int:
    """`codna run` / `codna durable run`: validate, ensure a server, start, optionally wait."""
    from . import require_durable

    require_durable()
    config = _cli._config(args)
    run_id = validate_run_id(args.run_id or _cli._new_run_id())
    repo_dir = _cli._repo_dir(args.repo)
    # E3 policy-as-code: a durable run is GOVERNED. Validate the repository policy here, in the
    # worker's own process, BEFORE any Temporal history exists — so a typo'd policy is a CLI error
    # an operator can fix immediately, not a run that starts and then fails every turn. The sidecar
    # reads the same file through the same module on the turn path; this is the second consumer of
    # one implementation, not a second implementation.
    _require_repository_policy(repo_dir)
    payload = run_payload_for(
        config=config,
        run_id=run_id,
        repo_dir=repo_dir,
        briefs=list(args.brief or ()),
        base_branch=args.base_branch,
        verify_commands=list(args.verify or ()),
        max_cost_usd=args.max_cost,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        engine=config.graph_engine,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        provider_key_source=getattr(args, "provider_key_source", None),
        worktree=_worktree_choice(args),
        # D7/SWARM: `--swarm` turns on BOTH delegation fields; the per-capability flags select one.
        # Validated against the approval profile inside `TaskSpec.validate`, so a forbidden pair is a
        # CLI error rather than a run that fails at its first session create.
        enable_spawn_agent=_swarm_choice(args, "spawn_agent"),
        enable_agent_teams=_swarm_choice(args, "agent_teams"),
        max_parallel_tasks=int(getattr(args, "parallel", 1) or 1),
    )
    # An implicit `ensure_started` is for the LOCAL profile only. Against Cloud there is nothing to
    # start, and a local start would happily succeed while the run then failed to connect — a
    # failure whose cause (wrong server) would be reported as the symptom (unreachable Cloud).
    started_local_server = False
    if not args.no_start_server and config.owns_local_server:
        with _cli._dev_server_errors():
            _cli._dev_server(config).ensure_started(timeout_s=args.start_timeout)
        started_local_server = True

    banner: dict[str, Any] = {
        "started": {
            "run_id": run_id,
            "workflow_id": config.run_workflow_id(run_id),
            "task_queue": config.task_queue,
            "engine": config.graph_engine,
            "tasks": len(payload["tasks"]),
            "repo_dir": str(repo_dir),
            "address": config.address,
            "temporal_profile": config.profile,
            "namespace": config.namespace,
            "local_server_started": started_local_server,
            # The provider and checkout mode the run will use, echoed from the VALIDATED spec so
            # "which model is this spending on, and does it create worktrees" is answerable from the
            # start banner rather than only from the ledger or from disk later.
            "provider": payload["tasks"][0].get("provider") if payload["tasks"] else None,
            "model": payload["tasks"][0].get("model") if payload["tasks"] else None,
            "base_url": payload["tasks"][0].get("base_url") if payload["tasks"] else None,
            "worktree": payload.get("worktree", False),
        }
    }
    run_worker_here = _worker_choice(args)
    pin_build = _pin_choice(args)
    banner["started"]["worker_in_process"] = run_worker_here

    async def command() -> int:
        client = await _cli._connect(config)
        if not run_worker_here:
            handle = await _start_run(
                client,
                config,
                payload,
                run_id=run_id,
                run_timeout_s=args.run_timeout,
                pin_build=pin_build,
            )
            banner["started"]["versioning"] = getattr(handle, "codna_pin_report", {})
            _cli._print(banner)
            if not args.wait:
                return 0
            return await _wait_run(
                client, config, run_id, poll_s=args.poll_interval, timeout_s=args.wait_timeout
            )
        return await _run_with_worker(
            client, config, args, payload, run_id=run_id, banner=banner, pin_build=pin_build
        )

    return int(_cli._in_loop(command))


# -- worker --------------------------------------------------------------------------------


def cmd_worker(args: Any) -> int:
    """`codna worker` / `codna durable worker`: poll this host's queue until a signal."""
    from . import require_durable
    from .worker import run_worker_sync, worker_settings_from_config

    require_durable()
    config = _cli._config(args)
    # Both spellings mean the same thing: "--dev-server" is the plan's name for it, and
    # "--start-dev-server" is what earlier commands already used. Refusing neither keeps a script that
    # used one working after the other was added.
    #
    # `--start-dev-server` on a Cloud config is a REFUSAL rather than a no-op: the flag asks for a
    # local server, and silently ignoring it would leave an operator believing one is running. (A
    # Cloud worker does not need it — Cloud is already up.)
    if getattr(args, "start_dev_server", False) or getattr(args, "dev_server", False):
        server = _cli._dev_server(config)  # raises `durable_dev_server_not_applicable` for cloud
        with _cli._dev_server_errors():
            server.ensure_started(timeout_s=args.start_timeout)
    settings = worker_settings_from_config(
        config,
        max_concurrent_activities=args.max_concurrent_activities,
        ensure_sidecar=not args.no_sidecar,
        grace_period_s=args.grace_period,
        promote_current=not args.no_promote,
        engine=getattr(args, "engine", None),
    )
    _cli._print({"worker": settings.to_payload()})
    return run_worker_sync(settings)


# -- parser registration -------------------------------------------------------------------


def add_worker_execution_flags(worker: Any) -> None:
    """The flags that control HOW a worker executes — the ones a run also needs in-process.

    Disjoint from `add_run_flags`: a parser that receives both sets must not see one option string
    twice (`--start-timeout` and `--engine` belong to the run set), because argparse refuses a
    duplicate outright. Keeping the split here means each flag has exactly one owner.
    """
    worker.add_argument("--max-concurrent-activities", type=int, default=2,
                        help="how many steps may run at once (default: 2)")
    worker.add_argument("--no-sidecar", action="store_true",
                        help="do not ensure agent-core is running (connect to what is there)")
    worker.add_argument("--grace-period", type=float, default=10.0,
                        help="seconds to let in-flight steps finish on shutdown")
    worker.add_argument("--no-promote", action="store_true",
                        help="do not make this build the deployment's current version when none is")
    worker.add_argument("--start-dev-server", action="store_true",
                        help="start the local Temporal dev server if it is not running")
    worker.add_argument("--dev-server", action="store_true",
                        help="start the local dev server if it is not running "
                             "(the same thing as --start-dev-server)")


def add_worker_flags(worker: Any) -> None:
    """The full `worker` flag set. One definition for both surfaces (`codna worker`, `durable worker`)."""
    add_worker_execution_flags(worker)
    worker.add_argument("--start-timeout", type=float, default=60.0, help="dev-server start timeout")
    worker.add_argument("--engine", choices=("langgraph", "plain"),
                        help="graph engine (default from config)")
    # WHICH SERVER. `--temporal-profile cloud` is the flag form of `CODNA_TEMPORAL_PROFILE=cloud`, so
    # a worker can be pointed at Temporal Cloud without editing a file or exporting a variable into
    # the shell that later runs something else. `None` means "decide from the environment" (see
    # `config._resolve_profile`), which is what keeps the local default byte-for-byte unchanged.
    # It belongs to the WORKER set (not the execution set) because `codna run` deliberately keeps the
    # server choice for itself: an in-process worker must poll the same server the run was started on.
    worker.add_argument("--temporal-profile", choices=("local", "cloud"), default=None,
                        help="which Temporal server: 'local' (dev server) or 'cloud' (TLS + API key); "
                             "default: inferred from TEMPORAL_* / CODNA_TEMPORAL_PROFILE")


def add_run_flags(run: Any, *, brief_required: bool = False) -> None:
    """The `run` flags. One definition for both surfaces (`codna run` and `codna durable run`).

    Two spellings of the same command would otherwise drift — a flag added to one and not the other
    is the classic way a "documented" option turns out not to exist on the path users take. The only
    difference between the surfaces is the handler's default (top-level `run` starts a worker for
    you), so the flags are shared and the defaults are set by each registrar.
    """
    run.add_argument("--repo", help="checkout to run against (default: the current directory)")
    # NOT argparse-required, deliberately: `codna run schedule` / `codna run trigger` are MODES of
    # `run` whose arguments come after the subcommand, so a parser-level requirement here would make
    # both unparseable. The requirement is enforced by `run_payload_for` (`durable_brief_required`),
    # which is also where an empty brief is refused — one rule, one place, one structured code.
    run.add_argument(
        "--brief",
        action="append",
        default=None,
        required=brief_required,
        help="what the task should do (repeatable; required unless a mode supplies it)",
    )
    run.add_argument("--base-branch", default="main", help="branch a task's work is based on")
    run.add_argument("--verify", action="append", default=[], metavar="CMD",
                     help="command that must pass to verify a task (repeatable)")
    run.add_argument("--max-cost", type=float, help="run-level cap, in USD")
    run.add_argument("--max-tokens", type=int, help="per-task token cap")
    run.add_argument("--max-attempts", type=int, default=3, help="plan/implement attempts per task")
    run.add_argument("--engine", choices=("langgraph", "plain"), help="graph engine (default from config)")
    # -- provider selection ---------------------------------------------------------------------
    #
    # These select WHICH provider the run's turns execute on. Without them the engine falls back to
    # its own default (`anthropic`), which is a credential error on a host that only has
    # `OPENAI_API_KEY` and makes a keyless local provider unusable for a durable run.
    #
    # There is deliberately NO `--provider-api-key`. The provider coordinates become a `TaskSpec`,
    # which enters workflow history and is mirrored into the ledger; a key there would be a secret in
    # an artifact people paste into tickets. ``--provider-key-source environment`` is the whole
    # supported story: the worker reads the key from its own process environment at turn time.
    run.add_argument(
        "--provider",
        default=None,
        help="model provider for this run's turns (e.g. anthropic, openai, ollama, openai-compatible)",
    )
    run.add_argument("--model", default=None, help="model id (default: the provider's own default)")
    run.add_argument(
        "--base-url",
        default=None,
        help="provider endpoint override; http(s), and plain http must point at loopback",
    )
    run.add_argument(
        "--provider-key-source",
        default=None,
        choices=PROVIDER_KEY_SOURCES,
        help="where the provider credential comes from; 'environment' makes the worker read it "
        "from its own process env (the only supported source — a key never enters the run spec)",
    )
    # -- per-task checkouts (D5) ----------------------------------------------------------------
    #
    # `--worktree` gives every task its own checkout: `<runs_root>/<run>/wt-<task>` on branch
    # `codna/run-<run>/task-<n>`, created by the task's first step and removed when the run ends. That
    # is what lets two tasks be worked on without sharing a tree, and what stops task 2 from starting
    # in whatever task 1 left behind.
    #
    # The DEFAULT is the shared checkout (`--no-worktree`): a worktree is a full extra checkout per
    # task, and imposing that by default on a large repository would be a surprise. See
    # `_worktree_choice` for exactly what `--no-worktree` means.
    run.add_argument(
        "--worktree",
        action="store_true",
        help="give each task its own git worktree (one full checkout per task, removed when the run "
        "ends) instead of working in the shared checkout",
    )
    run.add_argument(
        "--no-worktree",
        action="store_true",
        help="work in the shared checkout (the default): tasks serialise on `repo_dir` and task N+1 "
        "starts from whatever task N left behind",
    )
    # D7/SWARM + D7/MULTITASK. Registered in `swarm.py`, which owns the capability decision.
    add_delegation_flags(run)
    run.add_argument("--run-id", help="explicit run id (default: a UTC timestamp)")
    run.add_argument("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT_S,
                     help="server-side run timeout in seconds (default: 7 days)")
    run.add_argument("--wait", action="store_true", help="poll until the run finishes")
    run.add_argument("--wait-timeout", type=float, default=DEFAULT_RUN_TIMEOUT_S,
                     help="how long --wait may block (default: 7 days)")
    run.add_argument("--poll-interval", type=float, default=DEFAULT_WAIT_POLL_S,
                     help="poll period for --wait")
    run.add_argument("--no-start-server", action="store_true",
                     help="fail instead of starting the local dev server")
    run.add_argument("--start-timeout", type=float, default=60.0, help="dev-server start timeout")
    run.add_argument("--temporal-profile", choices=("local", "cloud"), default=None,
                     help="which Temporal server: 'local' (dev server) or 'cloud' (TLS + API key); "
                          "default: inferred from TEMPORAL_* / CODNA_TEMPORAL_PROFILE")
    run.add_argument("--pin", action="store_true",
                     help="pin the run to this build's worker version; fail if it is not registered")
    run.add_argument("--no-pin", action="store_true",
                     help="start the run unpinned (for a worker that registered no deployment version)")
    run.add_argument("--worker", action="store_true",
                     help="also run a worker in this process while the run executes "
                          "(top-level `codna run`; the default there)")
    run.add_argument("--no-worker", action="store_true",
                     help="only start the run; expect a separate `codna worker` to execute it")


def register_top_level(sub: Any) -> None:
    """Attach `codna worker` and `codna run` to the top-level parser.

    The plan names these as first-class commands ("codna worker / codna run CLI"), not only as
    `codna durable` subcommands: a durable run is a Codna feature, not a sub-feature of an optional
    namespace. They are registered separately from the `durable` group so the top-level help reads
    like a product surface, and they call the SAME handlers — one implementation, two entrances.

    The only real difference is `run`'s default — the top-level spelling runs a worker in the same
    process, because "codna run --brief ..." with nothing else running would sit forever on a queue
    nobody polls. The `durable` group's spelling defaults to the opposite (start only) because it is
    the lower-level surface used when a worker is already up.
    """
    worker = sub.add_parser(
        "worker",
        help="Run the durable worker: poll this host's queue and execute run steps.",
        description="The long-lived process that executes durable runs. Ctrl-C drains in-flight work.",
    )
    add_worker_flags(worker)
    worker.set_defaults(func=cmd_worker, uses_config=False)

    run = sub.add_parser(
        "run",
        help="Run a task durably: start a run (and, by default, the worker that executes it).",
        description="Durable runs: Temporal owns execution, the task graph owns the steps.",
    )
    add_run_flags(run)
    add_worker_execution_flags(run)
    # `run` reads the durable config, which reads the shared runtime config, so `uses_config` must be
    # true for the CLI's own config handling to run exactly as it does for `durable run`.
    # `run_worker_by_default` is what makes `codna run` execute what it starts, while the `durable`
    # group's spelling stays "start only" for callers who already have a worker.
    run.set_defaults(func=cmd_run, uses_config=True, run_worker_by_default=True)

    # `codna run schedule` / `codna run trigger`: the plan names these as MODES of `run`, and they are
    # — a schedule and a trigger both START the same run a manual invocation does, with the same
    # flags. A SUBPARSER under `run` (rather than a sibling top-level command) is what makes
    # `codna run schedule --brief ...` read as one sentence, and `dest` being optional keeps the
    # plain `codna run --brief ...` form working unchanged.
    run_actions = run.add_subparsers(dest="run_action")

    verify_ledger = run_actions.add_parser(
        "verify-ledger",
        help="Walk a run's hash-chained ledger and report every divergence (it prints, never crashes).",
        description="The audit check over the D2 ledger: each record links to its predecessor by "
        "hash, so an edited or removed record is located by index with both hashes. Exit 0 = "
        "intact, 2 = diverged.",
    )
    add_verify_ledger_flags(verify_ledger, flag="--run-id")
    verify_ledger.set_defaults(
        func=cmd_verify_ledger, uses_config=True, run_worker_by_default=False
    )

    schedule = run_actions.add_parser(
        "schedule",
        help="Register this run as a Temporal Schedule (or list the existing ones).",
        description="A scheduled run is a RUN the server starts on a timetable: it survives a "
        "restart, and an overlapping fire is skipped rather than doubled.",
    )
    add_schedule_flags(schedule)
    # `--list` is a MODE of the same command rather than a `schedule list` subcommand: the flags are
    # the same parser (a schedule IS a run declaration), and a nested group would make `--every`
    # meaningless in half of it.
    schedule.add_argument(
        "--list", action="store_true", dest="list_schedules",
        help="list the codna schedules on this namespace instead of registering one",
    )
    schedule.set_defaults(func=_cmd_schedule_entry, uses_config=True, run_worker_by_default=False)

    trigger = run_actions.add_parser(
        "trigger",
        help="Dispatch one GitHub delivery as a durable run (idempotent by delivery id).",
        description="Signal-with-Start: workflow id = the delivery id, so a redelivery attaches to "
        "the run already working instead of starting a second.",
    )
    add_trigger_flags(trigger)
    add_worker_execution_flags(trigger)
    trigger.set_defaults(func=cmd_trigger, uses_config=True, run_worker_by_default=False)

    serve = sub.add_parser(
        "serve",
        help="Serve the loopback Run API the IDE drives a run through.",
        description="A local HTTP surface over the same Temporal calls `codna run` makes. "
        "Loopback only, bearer-token authenticated.",
    )
    add_serve_flags(serve)
    # The Run API resolves the durable config (address, namespace, queue, runs root), so `uses_config`
    # is true for the same reason it is on `run`.
    serve.set_defaults(func=cmd_serve, uses_config=True)


def register_into(actions: Any) -> None:
    """Attach the `run`, `worker` and `serve` subcommands to a `codna durable` subparser group.

    These dispatch through `cmd_durable` (`durable_action`), like every other member of the group:
    one dispatcher for the namespace keeps the "unknown action" and "missing extra" reporting in
    ONE place. The top-level `codna run`/`codna worker`/`codna serve` set `func` directly — they are
    not inside the group and have no `durable_action` to route on.
    """
    run = actions.add_parser("run", help="Start a run (one task per --brief).")
    # This spelling has NO subcommands, so `--brief` is argparse-required here. The top-level `codna
    # run` cannot be (its `schedule`/`trigger` modes take their run declaration after the
    # subcommand), so the requirement is expressed where it can be: at the point of parsing.
    add_run_flags(run, brief_required=True)
    # The `durable` group's spelling starts the run only: it is the lower-level surface, used when a
    # worker is already running beside it. `codna run` (top level) defaults to running one here.
    run.set_defaults(func=_cli.cmd_durable, durable_action="run", run_worker_by_default=False)
    worker = actions.add_parser("worker", help="Poll this host's task queue and execute the steps.")
    add_worker_flags(worker)
    worker.set_defaults(func=_cli.cmd_durable, durable_action="worker")
    serve = actions.add_parser("serve", help="Serve the loopback Run API the IDE drives a run through.")
    add_serve_flags(serve)
    serve.set_defaults(func=_cli.cmd_durable, durable_action="serve")

    schedule = actions.add_parser("schedule", help="Register this run as a Temporal Schedule.")
    add_schedule_flags(schedule)
    schedule.add_argument(
        "--list", action="store_true", dest="list_schedules",
        help="list the codna schedules on this namespace instead of registering one",
    )
    schedule.set_defaults(func=_cmd_schedule_entry, uses_config=True)

    trigger = actions.add_parser("trigger", help="Dispatch one GitHub delivery as a durable run.")
    add_trigger_flags(trigger)
    add_worker_execution_flags(trigger)
    trigger.set_defaults(func=cmd_trigger, uses_config=True)
