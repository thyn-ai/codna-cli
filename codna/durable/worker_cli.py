"""The `codna worker` command surface: flags, the parent-guard contract, and the settings translation.

Split out of `run_cli.py` at the repository's 1000-line ceiling (`tests/test_modularity.py`), on the
boundary the plan names: the commands that START work stay there; what decides HOW A WORKER EXECUTES —
including the one contract that makes it the app's worker rather than a hand-run one — lives here.

**Nothing here imports `temporalio` at module scope**, so the base CLI keeps answering `codna worker
--help` without the durable extra installed.
"""
from __future__ import annotations

import os
import signal
from typing import Any

from . import cli as _cli
from .errors import DurableCliError, cli_error  # noqa: F401 - re-exported for callers/tests

__all__ = [
    "add_worker_execution_flags",
    "add_worker_flags",
    "cmd_worker",
]


def _validate_guard_parent_pid(raw: Any) -> int | None:
    """The supervision contract, validated BEFORE anything is imported or started.

    A pid is required rather than a bare flag so the child can never be guarding a process it did not
    name, and so the banner can report it. The value must name THIS process's parent: a worker whose
    parent changed between spawn and startup refuses rather than guard an unrelated pid.
    """
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise _error(
            "--guard-parent-pid must be a positive process id.",
            code="worker_parent_pid_invalid",
            details={"value": repr(raw)},
        )
    actual_parent_pid = os.getppid()
    if raw != actual_parent_pid:
        raise _error(
            f"--guard-parent-pid {raw} does not name this worker's parent "
            f"(actual parent pid {actual_parent_pid}).",
            code="worker_parent_pid_mismatch",
            details={"expected_parent_pid": raw, "actual_parent_pid": actual_parent_pid},
            hint="the desktop starts this flag itself; remove it when running a worker by hand",
        )
    return raw


def _error(message: str, *, code: str, details: dict[str, object] | None = None, hint: str = "") -> Any:
    return cli_error(message, code=code, details=details, hint=hint)


def cmd_worker(args: Any) -> int:
    """`codna worker` / `codna durable worker`: poll this host's queue until a signal."""
    raw_guard_parent_pid = _validate_guard_parent_pid(getattr(args, "guard_parent_pid", None))
    guard = None
    if raw_guard_parent_pid is not None:
        # ARM BEFORE importing Temporal, reading config, or starting a sidecar. An app crash during
        # startup is just as capable of leaving an orphan as one during a run. The callback sends
        # SIGTERM, never os._exit, so once the worker's normal handlers are installed parent loss uses
        # the same drain path as a graceful quit.
        #
        # THE VALIDATED PID, not a fresh `os.getppid()`: `_validate_guard_parent_pid` proved the value
        # names this worker's parent, and re-reading it here would open a window in which the OS answers
        # with a different process — the guard would then watch a pid nobody checked.
        from .parent_guard import install_parent_guard

        guard = install_parent_guard(_signal_self_for_parent_loss, parent_pid=raw_guard_parent_pid)

    try:
        return _run_worker_command(args, raw_guard_parent_pid)
    finally:
        if guard is not None:
            guard.stop()


def _run_worker_command(args: Any, raw_guard_parent_pid: int | None) -> int:
    """Resolve config and run one worker, after `cmd_worker` armed its parent guard."""
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
        # THE APP'S WORKER NEVER MOVES TRAFFIC BY DEFAULT. A worker the IDE starts is one of possibly
        # several codna workers on this machine (a dev checkout's, a teammate's, a container's), and
        # promoting it would silently re-route queued runs away from a build the operator chose. So the
        # flag's absence means `--no-promote` HERE, derived from the same switch that marks this worker
        # as app-owned. An explicit `--no-promote` is still honoured on every other path, and a hand-run
        # `codna worker` keeps its documented default of promoting a first build.
        promote_current=not (args.no_promote or raw_guard_parent_pid is not None),
        engine=getattr(args, "engine", None),
        guard_parent=raw_guard_parent_pid is not None,
        guard_parent_pid=raw_guard_parent_pid,
    )
    banner = settings.to_payload()
    if settings.guard_parent:
        # REPORTED, not implied: the banner says this worker will stop with its parent, so an operator
        # reading the listener table can tell an app-owned worker from a hand-started one.
        banner["guard_parent_pid"] = raw_guard_parent_pid
    _cli._print({"worker": banner})
    return run_worker_sync(settings)


def _signal_self_for_parent_loss(reason: str) -> None:
    """Move a guarded worker onto its standard SIGTERM/drain path when its desktop parent dies."""
    from .worker import _log_json

    _log_json("worker.parent_lost", reason=reason, signal="SIGTERM")
    os.kill(os.getpid(), signal.SIGTERM)


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
    worker.add_argument("--temporal-profile", choices=("local", "cloud"), default=None,
                        help="which Temporal server: 'local' (dev server) or 'cloud' (TLS + API key); "
                             "default: inferred from TEMPORAL_* / CODNA_TEMPORAL_PROFILE")
    # THE SUPERVISION CONTRACT, stated by the SPAWNER. When the desktop app forks this worker it passes
    # its own pid, and the worker then stops (through the ordinary drain path) the moment that pid is
    # gone — which is what makes "the app owns its worker" true across a SIGKILL of the app rather than
    # only across a graceful quit.
    #
    # It is OPT-IN: a worker a human types has no supervisor, and losing the terminal must not abandon
    # an in-flight paid turn. That asymmetry is the whole reason this is a flag and not the default.
    worker.add_argument("--guard-parent-pid", type=int, default=None, metavar="PID",
                        help="stop when the process PID (the app that spawned this worker) goes away; "
                             "set by the Codna desktop app, never needed for a hand-run worker")
