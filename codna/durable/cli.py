"""The `codna durable` command surface.

Split out of `codna/cli.py` for the same reason as `ci_cli`/`mcp_cli`/`webhook_cli`: the top-level
parser is held under a hard line ceiling (`tests/test_modularity.py`), and a command's parser plus
its handler belong together. `cli.py` carries one import and one call.

**Every command here is a thin translator.** It parses flags, resolves `DurableConfig`, and calls
ONE function from the durable package: `dev_server.DevServer`, `worker.run_worker_sync`,
`workflows.registered_workflows`, or a Temporal client. No orchestration happens in this module —
that is what keeps the CLI reproducible in a test (the pieces it calls are individually injectable)
and keeps a flag from quietly changing workflow semantics.

**Nothing here imports `temporalio` at module scope.** The base CLI must answer `codna durable
status` with a structured "the extra is not installed" rather than a `ModuleNotFoundError` traceback,
so the client is imported inside `_connect` and every entry point calls `require_durable` first
(which produces the structured error).

**Failures are loud and structured.** Errors carry a stable ``code`` and ``details``; `codna/cli.py`
already renders any exception with a ``code`` attribute as ``{"error": {...}}`` on stderr, so a
durable failure looks like every other Codna failure.

Commands:

    codna durable run           start a run (one child task per --brief) and optionally wait
    codna durable worker        poll this host's task queue and execute the durable activities
    codna durable dev-server    start/stop/status the local Temporal dev server
    codna durable status        one run's, or one task's, live status
    codna durable approve       approve a parked task (records WHO)
    codna durable reject        reject a parked task
    codna durable steer         add a steering note the next turn carries
    codna durable set-budget    raise a task's cost cap
    codna durable control       pause/resume/stop a run
    codna durable ledger        verify, summarise or roll up a run's ledger
"""
from __future__ import annotations

import contextlib
import getpass
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Final, Iterator, Mapping

from .config import DurableConfig, DurableConfigError, validate_run_id
from .checkout import looks_remote as _looks_remote, scm_url_for as _scm_url_for
from .dev_server import DevServer, DevServerError, DevServerSettings
from .errors import DurableCliError
from .repo_validation import existing_git_repo as _existing_git_repo
from .repo_validation import http_repo_dir as _validated_http_repo_dir
# The run/worker surface lives in its own module (1000-line ceiling): the commands that START work
# there, the commands that OBSERVE or STEER it here. Re-exported so callers keep one import.
from .run_cli import (
    DEFAULT_RUN_TIMEOUT_S,
    DEFAULT_WAIT_POLL_S,
    add_run_flags,
    add_serve_flags,
    add_worker_flags,
    cmd_run,
    cmd_serve,
    cmd_worker,
    register_into as _register_run_worker,
    register_top_level,
    run_payload_for,
)
# `query_workflow` lives with the wait loop it serves (see `run_wait.py`), and is re-exported here
# under its historical private name because `cli.py`'s own status command queries through it.
from .run_wait import query_workflow as _query
# The run command's internals are re-exported under their original private names because they are
# the CLI's TEST SEAMS: `_wait_run` is exercised directly (its poll/exit-code contract is not
# reachable without a server), and both spellings of the durable surface must reach the same
# function. Keeping them here means a test keeps addressing the module it always did.
#
# `run_cli` imports THIS module (for the shared error type), so these names are resolved from
# `run_cli` at import time and must exist there first — they do, because `run_cli` is imported above
# and defines every one of them. The block is a RE-EXPORT, not a second definition.
from .run_cli import (  # noqa: F401 - re-exported test seams, see above
    _exit_for_closed_run,
    _pin_choice,
    _run_is_finished,
    _run_with_worker,
    _start_run,
    _swarm_choice,
    _wait_run,
    _worker_choice,
    workflow_completion,
    workflow_failure_report,
    workflow_outcome,
)

__all__ = [
    "DEFAULT_RUN_TIMEOUT_S",
    "DEFAULT_WAIT_POLL_S",
    "DurableCliError",
    "add_run_flags",
    "add_serve_flags",
    "add_worker_flags",
    "cmd_durable",
    "cmd_run",
    "cmd_serve",
    "cmd_worker",
    "register",
    "register_top_level",
    "run_payload_for",
]

# Exit codes: 1 = the command failed, 2 = it ran but the run is not in a success state.
# (`DEFAULT_RUN_TIMEOUT_S`/`DEFAULT_WAIT_POLL_S` live in `run_cli.py` with the commands that use
# them, and are re-exported at the top of this module for callers that expect them here.)
EXIT_FAILURE: Final = 1
EXIT_NOT_SUCCESS: Final = 2


# -- small helpers -------------------------------------------------------------------------


def _print(payload: Mapping[str, Any]) -> None:
    from ..doctor import redact_display_text

    print(redact_display_text(json.dumps(dict(payload), indent=2, sort_keys=True)))


def _config(args: Any) -> DurableConfig:
    """Resolve the durable config, translating its structured error into this module's type.

    `load_durable_config` is imported here rather than at module scope because it reaches
    `codna.runtime.config` → `httpx`; a CLI module that pulls the HTTP stack in to parse a flag is a
    measurable startup cost on every `codna --help`.
    """
    from .config import load_durable_config

    engine = getattr(args, "engine", None)
    env = dict(os.environ)
    if engine:
        from .config import ENV_ENGINE

        env[ENV_ENGINE] = engine
    try:
        # `profile` is passed as its own argument rather than folded into `env`: a flag must outrank
        # the environment, and a value the environment never said must not appear to have come
        # from it (the inference from TEMPORAL_* stays testable in isolation).
        return load_durable_config(env=env, profile=getattr(args, "temporal_profile", None))
    except DurableConfigError as exc:
        raise DurableCliError(str(exc), code=exc.code, details=exc.details) from exc


def _repo_dir(raw: str | None, *, allowed_roots: tuple[Path, ...] | None = None) -> Path:
    """The checkout to run against: absolute, existing, and a git work tree — or a loud error.

    A REMOTE URL is cloned with the platform deploy key (D6); local paths are resolved and validated.
    HTTP callers must supply operator-approved roots. The key never enters a command line.
    """
    if allowed_roots is not None:
        return _http_repo_dir(raw, allowed_roots=allowed_roots)
    if raw and _looks_remote(raw):
        return _clone_remote(raw)
    if raw and "\0" in raw:
        raise DurableCliError(
            "repo directory contains a NUL byte.",
            code="durable_repo_missing",
            details={"repo_dir": "<NUL>"},
        )
    resolved_text = os.path.realpath(os.path.expanduser(raw)) if raw else str(Path.cwd())
    return _existing_git_repo(resolved_text)


def _http_repo_dir(raw: str | None, *, allowed_roots: tuple[Path, ...]) -> Path:
    """The HTTP repository boundary: remote clone, or an explicitly contained local path."""
    if raw and _looks_remote(raw):
        return _clone_remote(raw)
    return _validated_http_repo_dir(raw, allowed_roots=allowed_roots)


def _remote_directory_name(remote: str) -> str:
    """The clone directory for a remote: its repository name, without `.git` or a userinfo prefix.

    Derived from the SSH form so an `https://` and a `git@` spelling of the same repository produce
    ONE directory — otherwise the same repo would be cloned twice under two names and the second
    `codna run` would edit a tree the first never touched.
    """
    url = _scm_url_for(remote).rstrip("/")
    tail = url.rsplit("/", 1)[-1]
    # `git@github.com:o/r.git` has no `/` after the host, so the name is the part after the last `:`.
    name = tail.rsplit(":", 1)[-1] if "/" not in tail else tail
    return re.sub(r"\.git$", "", name)


def _clone_remote(raw: str) -> Path:
    """Clone a hosted repository with the deploy key, then behave exactly like a local `--repo`.

    The destination is derived from a fixed repos root and the repository NAME, so two runs of the
    same repo share one clone and a retried deploy does not clone twice. An existing clone is REUSED
    only after its `origin` is verified to be the same repository — reusing by name alone would edit
    someone else's checkout.
    """
    from .checkout import CheckoutError, checkout_repository, deploy_key_from_env, scm_url_for

    name = _remote_directory_name(raw)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", name) is None:
        raise DurableCliError(
            f"cannot derive a safe directory name from --repo {raw!r}.",
            code="durable_repo_missing",
            details={"repo": raw},
        )
    root_value = os.environ.get("CODNA_DURABLE_REPO_ROOT") or str(Path.home() / ".codna" / "repos")
    root = os.path.realpath(os.path.expanduser(root_value))
    destination = os.path.realpath(os.path.join(root, name))
    root_prefix = root if root.endswith(os.sep) else f"{root}{os.sep}"
    if not destination.startswith(root_prefix):
        raise DurableCliError(
            f"refusing to clone outside the durable repository root: {destination}",
            code="durable_repo_missing",
            details={"destination": destination, "root": root},
        )
    if os.path.isdir(destination) and os.listdir(destination):
        origin = subprocess.run(
            ["git", "-C", destination, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=False,
        )
        if origin.returncode != 0 or scm_url_for(origin.stdout.strip()) != scm_url_for(raw):
            raise DurableCliError(
                f"{destination} already holds a different repository, so it is not reused.",
                code="durable_repo_conflict",
                details={"destination": str(destination), "expected": scm_url_for(raw)},
                hint="set CODNA_DURABLE_REPO_ROOT to a fresh directory, or remove that clone",
            )
        return Path(destination)
    try:
        return checkout_repository(raw, Path(destination), deploy_key=deploy_key_from_env())
    except CheckoutError as exc:
        raise DurableCliError(str(exc), code=exc.code, details=exc.details) from exc


def _identity(args: Any) -> str:
    """The approver's identity. Required, never guessed: an anonymous approval is not an audit record."""
    raw = getattr(args, "actor", None)
    if not isinstance(raw, str) or not raw.strip():
        raise DurableCliError(
            "an approval needs an identity: pass --as you@example.com.",
            code="durable_identity_required",
            details={"flag": "--as"},
            hint="approvals are audit records; an anonymous one is refused by the workflow validator",
        )
    return raw.strip()


def _default_actor() -> str:
    """A suggested identity for `--help` only. Never used implicitly."""
    try:
        return f"{getpass.getuser()}@{os.uname().nodename}"
    except (OSError, KeyError):  # pragma: no cover - exotic platforms
        return "unknown"


def _new_run_id() -> str:
    """A UTC-timestamped run id. Time-derived because a run id must be unique, and logged as such.

    Overridable with ``--run-id``; the timestamp form is only a default so two invocations a second
    apart cannot collide.
    """
    return time.strftime("run-%Y%m%dT%H%M%SZ", time.gmtime())


def _dev_server(config: DurableConfig) -> DevServer:
    """The local dev server, or a refusal when this config targets Temporal Cloud.

    Refusing HERE (rather than letting ``DevServer`` complain about a non-loopback bind) is what
    keeps the error honest: the fix is "this command does not apply to Cloud", not "pick a loopback
    address". Every caller of this helper — the three dev-server actions and ``run``/``worker``'s
    ``--start-dev-server`` — therefore cannot start a local server behind a Cloud config.
    """
    if config.is_cloud:
        raise DurableCliError(
            "there is no local dev server to manage for a Temporal Cloud profile.",
            code="durable_dev_server_not_applicable",
            details={
                "temporal_profile": config.profile,
                "address": config.address,
                "namespace": config.namespace,
            },
            hint=(
                "Cloud is already running; drop --start-dev-server/--dev-server, or set "
                "CODNA_TEMPORAL_PROFILE=local to manage a dev server"
            ),
        )
    return DevServer(DevServerSettings.from_config(config))


async def _connect(config: DurableConfig) -> Any:
    """A Temporal client on the configured address, connected inside the caller's loop.

    ``temporalio`` is imported here, not at module scope: the base CLI must parse and answer
    ``codna durable --help`` without the extra. The client is created INSIDE the loop that will use
    it — a client built by one ``asyncio.run`` and used by another is bound to a closed loop and
    fails on its first RPC, which is why every command below is one async function start to finish.

    ``config.connect_options()`` is what puts a Cloud connection on TLS with its API key. For the
    local profile it returns nothing, and this call is exactly what it always was.
    """
    from temporalio.client import Client

    try:
        options = config.connect_options()
    except DurableConfigError as exc:
        # A key that vanished between resolution and connection. Raised before the RPC so the cause
        # is this process's environment, not an "unauthorised" answer from the server.
        raise DurableCliError(str(exc), code=exc.code, details=exc.details) from exc
    # E3: the payload codec is installed on EVERY client, not just the worker's. A client that only
    # STARTS a run still writes history (the input, the memo, the search attributes), and a client
    # that only QUERIES still has to decode it — so a codec on one side of a connection is half a
    # codec. `resolve_codec` reports the mode, which the caller surfaces.
    from .codec_config import CodecUnavailable, data_converter_kwargs, resolve_codec

    try:
        resolution = resolve_codec()
    except CodecUnavailable as exc:
        raise DurableCliError(str(exc), code=exc.code, details=exc.details) from exc
    options.update(data_converter_kwargs(resolution))
    try:
        return await Client.connect(config.address, namespace=config.namespace, **options)
    except Exception as exc:  # noqa: BLE001 - reported with the address and a fix
        raise DurableCliError(
            f"could not connect to the Temporal server at {config.address}: {exc}",
            code="durable_server_unreachable",
            details={
                "address": config.address,
                "namespace": config.namespace,
                "temporal_profile": config.profile,
                "tls": config.requires_tls,
            },
            hint=(
                "check TEMPORAL_ENDPOINT/NAMESPACE/API_KEY and that this namespace grants the key"
                if config.is_cloud
                else "start it with `codna durable dev-server start`, or set CODNA_TEMPORAL_ADDRESS"
            ),
        ) from exc


def _in_loop(command: Any) -> Any:
    """Run one async command, turning any durable error into the type `codna/cli.py` prints.

    A single loop per command (rather than per RPC) is what keeps the connected client usable: the
    client, every query, every update and the poll loop all live in the same loop, and the process
    exits when the command is done.
    """
    import asyncio

    try:
        return asyncio.run(command())
    except DurableCliError:
        raise
    except DevServerError as exc:
        raise _dev_server_error(exc) from exc


# -- dev server ----------------------------------------------------------------------------


@contextlib.contextmanager
def _dev_server_errors() -> Iterator[None]:
    """Translate `DevServerError` into this module's error type, preserving code and hint.

    One translation point for four call sites (`run`, `worker`, and the three dev-server actions),
    so the structured body a caller sees does not depend on which line raised.
    """
    try:
        yield
    except DevServerError as exc:
        raise _dev_server_error(exc) from exc


def _cmd_dev_server(args: Any) -> int:
    from . import require_durable

    require_durable()
    config = _config(args)
    server = _dev_server(config)
    action = getattr(args, "dev_action", "status")
    with _dev_server_errors():
        if action == "start":
            status = server.ensure_started(timeout_s=args.start_timeout, headless=args.headless)
        elif action == "stop":
            status = server.stop(timeout_s=args.stop_timeout)
        elif action == "status":
            status = server.status()
        else:  # pragma: no cover - the parser only offers the three
            raise DurableCliError(
                f"unknown dev-server action {action!r}", code="durable_dev_action_invalid", details={}
            )
    _print({"dev_server": status.to_payload()})
    return 0


# -- status --------------------------------------------------------------------------------


def _cmd_status(args: Any) -> int:
    from . import require_durable

    require_durable()
    config = _config(args)
    run_id = _require_run(args)

    if args.task:
        task_id = _task_id(args)

        async def task_status() -> int:
            return await _print_task_status(await _connect(config), config, run_id, task_id)

        return int(_in_loop(task_status))

    async def run_status() -> int:
        from .run_workflow import RunWorkflow

        client = await _connect(config)
        workflow_id = config.run_workflow_id(run_id)
        status = await _query(client, workflow_id, RunWorkflow.status)
        _print({"status": status, "workflow_id": workflow_id})
        return 0

    return int(_in_loop(run_status))


def _require_run(args: Any) -> str:
    raw = getattr(args, "run", None)
    if not raw:
        raise DurableCliError(
            "this command needs --run <run-id>.", code="durable_run_required", details={"flag": "--run"}
        )
    return validate_run_id(raw)


def _task_workflow_id(run_id: str, task_id: str) -> str:
    """The child's id, mirroring `RunWorkflow._run_task` exactly (``task-<run>-<task>``).

    Duplicated as a pure function here rather than imported: the workflow module imports
    ``temporalio``, and a CLI that cannot even parse ``status`` without the extra installed would
    turn "not installed" into a traceback. A test asserts the two agree.
    """
    return f"task-{validate_run_id(run_id)}-{task_id}"


async def _print_task_status(client: Any, config: DurableConfig, run_id: str, task_id: str) -> int:
    """A task's status, discovering which engine started it by asking each workflow class.

    The `status` QUERY is defined on both engines with the same name, so the query itself is
    addressed by NAME and only the declaration (the class) is engine-specific. Asking the wrong
    class would raise "no query named status", which is why both are tried in order.
    """
    from .workflows import workflow_class_for

    workflow_id = _task_workflow_id(run_id, task_id)
    engine_status: dict[str, Any] | None = None
    last_error: Exception | None = None
    for engine in ("langgraph", "plain"):
        cls = workflow_class_for(engine)
        try:
            engine_status = await _query(client, workflow_id, cls.status)
            break
        except DurableCliError as exc:
            last_error = exc
            continue
    if engine_status is None:
        raise DurableCliError(
            f"could not read status for task {task_id} of run {run_id}.",
            code="durable_task_status_failed",
            details={"workflow_id": workflow_id, "last_error": str(last_error) if last_error else None},
        )
    _print({"status": engine_status, "workflow_id": workflow_id})
    return 0


# -- approvals -----------------------------------------------------------------------------


def _cmd_decide(args: Any) -> int:
    """approve / reject / steer / set-budget: one Update against a task's workflow.

    The engine is discovered by trying both, because the update is addressed to a workflow id and the
    CLI is not told which engine started it. Missing runs are reported as one error, not two.
    """
    from . import require_durable

    require_durable()
    config = _config(args)
    _require_run(args)
    task_id = _task_id(args)
    identity = _identity(args)
    action = args.decide_action
    payload: dict[str, Any] = {"identity": identity}
    if action == "steer":
        payload["note"] = _note(args)
    elif action == "set-budget":
        payload["max_cost_usd"] = _positive(args.max_cost)

    workflow_id = _task_workflow_id(args.run, task_id)

    async def command() -> int:
        client = await _connect(config)
        result = await _send_update(client, workflow_id, action, payload)
        _print({"update": action, "workflow_id": workflow_id, "result": result})
        return 0

    return int(_in_loop(command))


def _task_id(args: Any) -> str:
    raw = getattr(args, "task", None)
    if not raw or not str(raw).strip():
        raise DurableCliError(
            "this command needs --task <task-id> (for example task-1).",
            code="durable_task_required",
            details={"flag": "--task"},
        )
    value = str(raw).strip()
    if "/" in value or ".." in value:
        raise DurableCliError(
            "task id must not contain a path separator.", code="durable_task_invalid", details={"task": value}
        )
    return value


def _note(args: Any) -> str:
    note = (getattr(args, "note", None) or "").strip()
    if not note:
        raise DurableCliError(
            "steer needs --note <text>.", code="durable_note_required", details={"flag": "--note"}
        )
    return note


def _positive(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise DurableCliError(
            "--max-cost must be a positive number for set-budget.",
            code="durable_cost_invalid",
            details={"value": repr(value)},
        )
    return float(value)


async def _send_update(client: Any, workflow_id: str, action: str, payload: Mapping[str, Any]) -> Any:
    """Send one Update, discovering the engine by trying both workflow classes.

    The CLI is not told which engine started a task, and an Update's name is only resolvable
    against the declaring class. A *validator* refusal is the answer we want to report ("this task
    is not waiting on a decision"), so the LAST refusal is raised with its own text preserved.

    **An identity is a precondition of this function, not of the workflow.** Both engines validate it
    too (an anonymous approval is refused there as well), and this check exists so an Update cannot
    even leave the CLI without one — belt and braces, because "who approved" is the audit fact and a
    second place that refuses to omit it costs nothing.
    """
    from .workflows import workflow_class_for

    raw_identity = payload.get("identity")
    if not isinstance(raw_identity, str) or not raw_identity.strip():
        raise DurableCliError(
            f"a {action} Update needs an identity: who is acting?",
            code="durable_identity_required",
            details={"action": action, "workflow_id": workflow_id},
            hint="pass --as you@example.com",
        )
    method_name = {"set-budget": "set_budget"}.get(action, action)
    handle = client.get_workflow_handle(workflow_id)
    last: Exception | None = None
    for engine in ("langgraph", "plain"):
        method = getattr(workflow_class_for(engine), method_name, None)
        if method is None:  # pragma: no cover - both engines declare all four
            continue
        try:
            return await handle.execute_update(method, dict(payload))
        except Exception as exc:  # noqa: BLE001 - retried on the other engine
            last = exc
    if last is None:  # pragma: no cover - unreachable unless a workflow class lost a handler
        raise DurableCliError(
            f"no workflow class declares the {action} update.",
            code="durable_update_unknown",
            details={"action": action},
        )
    raise DurableCliError(
        f"the workflow refused {action}: {last}",
        code=_refusal_code(last),
        details={"workflow_id": workflow_id, "action": action},
        hint="a refusal is a validator's answer: the task may not be parked, or already decided",
    ) from last


def _refusal_code(exc: BaseException) -> str:
    text = str(exc).lower()
    if "not found" in text or "not_found" in text:
        return "durable_workflow_not_found"
    if "not waiting" in text or "not parked" in text:
        return "durable_not_parked"
    return "durable_update_refused"


# -- control -------------------------------------------------------------------------------


def _cmd_control(args: Any) -> int:
    """pause / resume / stop a run, as Signals (the run's own handlers).

    Signals, not Updates: these are fire-and-forget control actions and their effect is visible in
    the next `status`, so waiting for an ack would only add a failure mode.
    """
    from . import require_durable

    require_durable()
    config = _config(args)
    run_id = _require_run(args)
    action = args.control_action
    workflow_id = config.run_workflow_id(run_id)

    async def command() -> int:
        client = await _connect(config)
        await _signal(client, workflow_id, action)
        _print({"signalled": action, "workflow_id": workflow_id})
        return 0

    return int(_in_loop(command))


async def _signal(client: Any, workflow_id: str, action: str) -> None:
    from .run_workflow import RunWorkflow

    method = getattr(RunWorkflow, action, None)
    if method is None:  # pragma: no cover - the parser only offers pause/resume/stop
        raise DurableCliError(
            f"unknown control action {action!r}", code="durable_control_invalid", details={}
        )
    try:
        await client.get_workflow_handle(workflow_id).signal(method)
    except Exception as exc:  # noqa: BLE001
        raise DurableCliError(
            f"could not send {action} to {workflow_id}: {exc}",
            code="durable_signal_failed",
            details={"workflow_id": workflow_id, "action": action},
        ) from exc


# -- events (reattach) ---------------------------------------------------------------------


def _cmd_events(args: Any) -> int:
    """`codna durable events --run X [--after N]`: the run's ledger records, oldest first.

    Reattach is a LEDGER read, not a workflow read. The ledger is the append-only, hash-chained record
    that survives everything else (Temporal history expires, the sidecar keeps sessions in memory), so
    it is what a client that was disconnected for ten minutes re-reads — and `--after` is the resume
    point it holds. Following (`--follow`) re-reads from the highest seq it has emitted, which is the
    same number a client would have persisted, so a client can be restarted and pass it back here.

    Exit code 2 when the chain does not verify: an auditor must not read a tampered ledger as truth.
    """
    from . import require_durable
    from .ledger import load_ledger, records_after

    require_durable()
    config = _config(args)
    run_id = _require_run(args)
    # Caller input is validated BEFORE the filesystem is consulted: a bad --after is a usage error on
    # every run, not only on a run whose ledger happens to exist (which made the same mistake report
    # two different codes depending on unrelated state).
    after = _after_seq(args)
    path = config.ledger_path(run_id)
    if not path.is_file():
        raise DurableCliError(
            f"run {run_id} has no ledger at {path}.",
            code="durable_ledger_missing",
            details={"run_id": run_id, "path": str(path)},
            hint="the ledger is written by the activities; a run that has not executed has none",
        )
    follow = bool(getattr(args, "follow", False))
    interval = float(getattr(args, "interval", 1.0))
    if interval <= 0:
        raise DurableCliError(
            "--interval must be positive.", code="durable_interval_invalid", details={"interval": interval}
        )
    emitted = after
    try:
        while True:
            records, problems = load_ledger(path)
            for record in records_after(records, emitted):
                _print({"record": record.to_payload()})
                emitted = record.seq
            if problems:
                raise DurableCliError(
                    f"the ledger for run {run_id} does not verify: {'; '.join(problems[:3])}",
                    code="durable_ledger_invalid",
                    details={"run_id": run_id, "path": str(path), "problems": problems},
                )
            if not follow:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - an interactive stop, not a failure
        return 0


def _after_seq(args: Any) -> int:
    """The ``--after`` resume point: a non-negative integer, refusing anything else.

    Validated rather than coerced because a mistyped resume point silently SKIPS records — and the
    records it would skip are exactly the ones the caller has not seen.
    """
    raw = getattr(args, "after", None)
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise DurableCliError(
            "--after must be a non-negative integer (the last seq you already have).",
            code="durable_after_invalid",
            details={"after": repr(raw)},
        )
    return raw


# -- ledger --------------------------------------------------------------------------------


def _cmd_codec(args: Any) -> int:
    """Payload-codec key lifecycle. Delegated so key handling lives in one module."""
    from .codec_cli import cmd_durable_codec

    return cmd_durable_codec(args)


def _cmd_ledger(args: Any) -> int:
    from . import require_durable
    from .ledger import RunLedger, load_ledger, verify_chain

    require_durable()
    config = _config(args)
    run_id = _require_run(args)
    path = config.ledger_path(run_id)
    if not path.is_file():
        raise DurableCliError(
            f"run {run_id} has no ledger at {path}.",
            code="durable_ledger_missing",
            details={"run_id": run_id, "path": str(path)},
            hint="the ledger is written by the activities; a run that has not executed has none",
        )
    problems = verify_chain(path)
    ledger = RunLedger.for_run(config.paths, run_id)
    stats = ledger.stats()
    records, load_problems = load_ledger(path)
    # The ledger's spend is the sum of the per-record `cost_usd` deltas the activities appended —
    # derived here rather than stored, so the total cannot disagree with the records it summarises.
    spent_usd = sum(float(record.payload.get("cost_usd") or 0.0) for record in records)
    body: dict[str, Any] = {
        "run_id": run_id,
        "path": str(path),
        "valid": not problems and not load_problems,
        "problems": [*problems, *load_problems],
        "records": stats.records,
        "sessions": len({record.session_id for record in records if record.session_id}),
        "spent_usd": round(spent_usd, 6),
        "last_seq": stats.last_seq,
    }
    if args.rollup:
        body["parquet"] = str(ledger.rollup_parquet())
    if args.tail:
        body["tail"] = [record.to_payload() for record in records[-args.tail :]]
    _print(body)
    return 0 if body["valid"] else EXIT_NOT_SUCCESS


# -- registration --------------------------------------------------------------------------


def _cmd_verify_ledger(args: Any) -> int:
    """`codna durable verify-ledger`: the audit check, delegated to its own module.

    A delegator rather than a re-implementation so `codna run verify-ledger` and this spelling are
    one behaviour — the property the plan asks for ("`codna run verify-ledger` over the D2 hash
    chain") must not depend on which surface invoked it.
    """
    from .verify_ledger_cli import cmd_verify_ledger

    return cmd_verify_ledger(args)


def _cmd_audit(args: Any) -> int:
    """`codna durable audit`: SIEM retention + a sweep of every retained ledger."""
    from .audit_cli import cmd_audit

    return cmd_audit(args)


def cmd_durable(args: Any) -> int:
    action = getattr(args, "durable_action", None)
    handlers = {
        "run": cmd_run,
        "worker": cmd_worker,
        "serve": cmd_serve,
        "dev-server": _cmd_dev_server,
        "status": _cmd_status,
        "approve": _cmd_decide,
        "reject": _cmd_decide,
        "steer": _cmd_decide,
        "set-budget": _cmd_decide,
        "control": _cmd_control,
        "ledger": _cmd_ledger,
        "verify-ledger": _cmd_verify_ledger,
        "audit": _cmd_audit,
        "events": _cmd_events,
        "codec": _cmd_codec,
    }
    handler = handlers.get(action or "")
    if handler is None:
        raise DurableCliError(
            "codna durable needs a subcommand.",
            code="durable_action_required",
            details={"known": sorted(handlers)},
        )
    return int(handler(args))


def register(sub: Any) -> None:
    """Attach `codna durable {run,worker,dev-server,status,approve,...}` to the top-level parsers."""
    p = sub.add_parser(
        "durable",
        help="Run a long task durably: survives a crash, a sleep, and a worker restart.",
        description="Durable runs: Temporal owns execution, the task graph owns the steps, and an "
        "append-only ledger owns the audit trail.",
    )
    actions = p.add_subparsers(dest="durable_action")
    _register_run_worker(actions)  # run + worker, from run_cli.py (where they execute)
    _register_dev_server(actions)
    _register_status(actions)
    for name, help_text in (
        ("approve", "Approve a task parked on a human decision."),
        ("reject", "Reject a task parked on a human decision."),
        ("steer", "Add a note the next turn carries (never interrupts a running turn)."),
        ("set-budget", "Raise a task's cost cap (never lowers it)."),
    ):
        _register_decide(actions, name, help_text)
    _register_control(actions)
    _register_ledger(actions)
    _register_events(actions)
    from .codec_cli import register as _register_codec

    _register_codec(actions)
    p.set_defaults(func=cmd_durable, durable_action=None, uses_config=False)


# -- per-command registration --------------------------------------------------------------


def _register_dev_server(actions: Any) -> None:
    dev = actions.add_parser("dev-server", help="Manage the local Temporal dev server.")
    dev_actions = dev.add_subparsers(dest="dev_action")
    start = dev_actions.add_parser("start", help="Start it (idempotent: does nothing if it is up).")
    start.add_argument("--start-timeout", type=float, default=60.0)
    start.add_argument("--headless", action="store_true", help="do not serve the Temporal Web UI")
    start.set_defaults(func=cmd_durable, durable_action="dev-server", dev_action="start")
    stop = dev_actions.add_parser("stop", help="Stop the one codna started (never a foreign server).")
    stop.add_argument("--stop-timeout", type=float, default=15.0)
    stop.set_defaults(func=cmd_durable, durable_action="dev-server", dev_action="stop")
    status = dev_actions.add_parser("status", help="Report whether it is running and who owns it.")
    status.set_defaults(func=cmd_durable, durable_action="dev-server", dev_action="status")
    dev.set_defaults(func=cmd_durable, durable_action="dev-server", dev_action="status")


def _register_status(actions: Any) -> None:
    status = actions.add_parser("status", help="One run's or one task's live status.")
    status.add_argument("--run", required=True, help="run id")
    status.add_argument("--task", help="task id (e.g. task-1) for a task-level status")
    status.set_defaults(func=cmd_durable, durable_action="status")


def _register_decide(actions: Any, name: str, help_text: str) -> None:
    parser = actions.add_parser(name, help=help_text)
    parser.add_argument("--run", required=True, help="run id")
    parser.add_argument("--task", required=True, help="task id (e.g. task-1)")
    parser.add_argument("--as", dest="actor", metavar="IDENTITY", default=None,
                        help=f"who is acting (required; e.g. {_default_actor()})")
    if name == "steer":
        parser.add_argument("--note", required=True, help="the steering note")
    if name == "set-budget":
        parser.add_argument("--max-cost", type=float, required=True, help="the new cost cap, in USD")
    # `decide_action` (not `durable_action`) because the four commands share ONE handler and the
    # handler must not confuse "which command was typed" with "which subcommand dispatched it".
    parser.set_defaults(func=cmd_durable, durable_action=name, decide_action=name)


def _register_control(actions: Any) -> None:
    control = actions.add_parser("control", help="Pause, resume or stop a run.")
    control.add_argument("--run", required=True, help="run id")
    control.add_argument("control_action", choices=("pause", "resume", "stop"))
    control.set_defaults(func=cmd_durable, durable_action="control")


def _register_ledger(actions: Any) -> None:
    ledger = actions.add_parser("ledger", help="Verify, summarise or roll up a run's ledger.")
    ledger.add_argument("--run", required=True, help="run id")
    ledger.add_argument("--tail", type=int, default=0, help="include the last N records")
    ledger.add_argument("--rollup", action="store_true", help="write the Parquet rollup as well")
    ledger.set_defaults(func=cmd_durable, durable_action="ledger")

    # The audit check, in the namespace where a ledger is managed. It is a separate command from
    # `ledger` because it is the one that must never crash: `ledger` summarises and fails on a
    # malformed line (which is right for a summary), `verify-ledger` reports the malformation.
    from .verify_ledger_cli import add_verify_ledger_flags

    verify = actions.add_parser(
        "verify-ledger",
        help="Report every divergence in a run's hash chain (prints; never crashes).",
    )
    add_verify_ledger_flags(verify)
    verify.set_defaults(func=cmd_durable, durable_action="verify-ledger")

    # The operational half of E2's audit story: retention over the SIEM export, and a sweep that
    # verifies every retained ledger. It needs no credential and no Temporal — it only reads and
    # unlinks files this host already owns.
    from .audit_cli import add_audit_flags

    audit = actions.add_parser(
        "audit",
        help="Apply the SIEM retention window and verify every retained run ledger.",
        description="Temporal Cloud keeps a closed run's history for a bounded period; this is what "
        "outlives it. Retention prunes the export, never a ledger, and the sweep reports a chain "
        "that no longer verifies.",
    )
    add_audit_flags(audit)
    audit.set_defaults(func=cmd_durable, durable_action="audit")


def _register_events(actions: Any) -> None:
    events = actions.add_parser(
        "events", help="Reattach: print a run's ledger records (optionally after a seq)."
    )
    events.add_argument("--run", required=True, help="run id")
    events.add_argument("--after", type=int, default=None,
                        help="only records with seq greater than this (the resume point)")
    events.add_argument("--follow", action="store_true",
                        help="keep printing new records as they arrive")
    events.add_argument("--interval", type=float, default=1.0,
                        help="seconds between reads with --follow (default: 1.0)")
    events.set_defaults(func=cmd_durable, durable_action="events")


def _dev_server_error(exc: DevServerError) -> DurableCliError:
    return DurableCliError(str(exc), code=exc.code, details=exc.details, hint=exc.hint)


# -- top-level `codna worker` and `codna run` ----------------------------------------------
#
# The plan names these as first-class commands ("codna worker / codna run CLI"), not only as
# `codna durable` subcommands: a durable run is a Codna feature, not a sub-feature of an optional
# namespace. `run_cli.py` owns their implementation (it is where work is started and executed); this
# module only re-exports its registration so `codna/cli.py` keeps ONE import for the durable surface.
