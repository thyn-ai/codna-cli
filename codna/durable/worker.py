"""The durable worker: one process that owns a task queue and every dependency an activity needs.

A worker is where the injected `ActivityRuntime` becomes REAL — the sidecar URL, the run directory
layout, the ledger, the GitHub token. Everything above this module was written against an injected
runtime (which is what made it testable); this is the single place that decides what those
injections are in production. Keeping the decision in one function (`build_activity_runtime`) is
the reason no activity reads `os.environ` for a provider key.

**Registration is the other half.** `Worker(workflows=..., activities=..., plugins=...)` is
assembled here, from the two tuples the durable package already owns
(`workflows.registered_workflows()`, `activities.ACTIVITIES`) plus the LangGraph plugin for the
selected engine. Registering the plugin for the plain engine would make the worker reject a
duplicate activity name; registering neither would leave a run started under the graph engine
waiting until its task-queue timeout, so the engine choice is explicit and validated.

**Startup fails loudly, in this order**, because each failure has a different fix:

* ``worker_extra_missing`` — ``codna[durable]`` is not installed (`require_durable`);
* ``worker_server_unreachable`` — nothing is listening at the configured address;
* ``worker_sidecar_unavailable`` — the agent-core sidecar could not be started or is not ready;
* ``worker_registration_failed`` — Temporal refused the registration (a duplicate name, or a
  workflow outside the sandbox), quoted with the cause.

**Registration is a DEPLOYMENT VERSION, not a bare build id.** ``WorkerDeploymentConfig`` with the
build id ``codna-<version>-g<GRAPH_VERSION>`` as its version, ``use_worker_versioning=True`` and
``default_versioning_behavior=PINNED`` is what makes the plan's risk mitigation real: a graph change
bumps ``GRAPH_VERSION``, which is a new build id, which registers as a NEW deployment version, and
runs already in flight stay pinned to the build they started on. A bare ``build_id=`` registers no
deployment version at all (measured: ``describe_worker_deployment`` says "no Worker Deployment
found"), so nothing would be protected. Three measured consequences drive the design here:

* A PINNED worker only receives runs that are pinned to ITS version. A run started without a pin is
  routed to the deployment's current version and, on a server with no current version, sits until it
  times out. So ``run_worker`` promotes its own version to current when nothing is current yet
  (``allow_no_pollers=True``) — the first worker on a fresh server — and leaves an existing current
  version alone. ``--no-promote`` turns that off for a host that must not move traffic.
* A run started with an explicit pin against a worker that registered NO deployment version fails at
  ``start_workflow`` ("Pinned version ... is not present in task queue"). That is why the CLI's
  ``run`` asks the worker's own registration path what to pin to, instead of inventing a version.
* A child workflow inherits its parent's version, so a pinned run's task children land on the same
  worker without their own override. Measured on both dev-server modes.

**Shutdown is a first-class path.** SIGINT/SIGTERM set the stop event; the worker then calls
``Worker.shutdown`` and AWAITS ``worker.run()`` to completion. Temporal's shutdown lets an in-flight
activity finish (up to ``graceful_shutdown_timeout``) instead of abandoning it, which is what keeps a
paid turn from being re-delivered and billed twice. A worker that exits on the signal without
awaiting that path would leave a paid turn running on a dead process.

``temporalio`` is imported inside functions only. The base CLI (and the base test suite) imports
this module for its settings builders without the extra installed, and must keep working.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Mapping

from .config import ENV_CLOUD_API_KEY
from .publish import GITHUB_TOKEN_ENV as _GITHUB_TOKEN_ENV

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import DurableConfig

__all__ = [
    "DEFAULT_MAX_CONCURRENT_ACTIVITIES",
    "DEPLOYMENT_NAME",
    "WorkerError",
    "WorkerSettings",
    "build_activity_runtime",
    "deployment_version",
    "deployment_version_config",
    "github_token_provider",
    "plugins_for",
    "promote_current_version",
    "registered_build_ids",
    "run_worker",
    "run_worker_sync",
    "wait_until_registered",
    "worker_settings_from_config",
]

# The Temporal Worker Deployment this worker belongs to. One per product (not per host): the build
# id already carries the version, so a deployment per host would only fragment "which build is
# current" into one question per laptop. Versioning routes BY TASK QUEUE, so two hosts on different
# builds still each get their own queue's traffic — the deployment name is just the family.
DEPLOYMENT_NAME: Final = "codna"

# What a deployment's routing config holds before any version has been promoted. Temporal's own
# sentinel; treated as "nothing is current" rather than as a version name.
UNVERSIONED: Final = "__unversioned__"

# How long the startup banner waits for this build's deployment version to appear before reporting
# that it could not be promoted. A deployment only exists once pollers attach, so some wait is normal.
PROMOTION_WAIT_S: Final = 10.0
# How long "is my build polling yet?" may wait. Pollers attach a beat after `run()` starts; the wait
# is short because a healthy worker registers in milliseconds and a broken one should fail fast.
REGISTRATION_WAIT_S: Final = 15.0

# Two concurrent activities per worker: the plan's cap, and each slot can hold an open sidecar
# stream. Bounded deliberately rather than left at Temporal's default (unbounded).
DEFAULT_MAX_CONCURRENT_ACTIVITIES: Final = 2

# Environment variables that may carry the GitHub write token, in precedence order. Re-exported
# from `publish` (a LEAF module) because that is where the publish step's refusal reads them: one
# definition, so a token that resolves for the step cannot be invisible to this provider or vice
# versa.
GITHUB_TOKEN_ENV: Final[tuple[str, ...]] = _GITHUB_TOKEN_ENV

# The sidecar's internal bearer (see `agent-core/.../server/http.ts`): absent means "local dev is
# open", present means every request must carry it. Read, never invented.
SIDECAR_TOKEN_ENV: Final = "AGENT_CORE_AUTH_TOKEN"

# D6: how many activities one worker may run at once, when the flag does not say. A container has no
# per-machine argument, so a pool's sizing has to be an env value; see `worker_settings_from_config`.
CONCURRENCY_ENV: Final = "CODNA_WORKER_MAX_CONCURRENT_ACTIVITIES"
# D6: how long in-flight activities may finish after SIGTERM. An env value for the same reason: a
# container's shutdown grace is a per-deployment decision, and it has to fit the platform's own
# kill_timeout (Fly's maximum is 300 s), which no flag can be arranged to match by hand.
GRACE_PERIOD_ENV: Final = "CODNA_WORKER_GRACE_PERIOD_S"

# The signature default for `grace_period_s`. A module constant so "the caller passed nothing" is
# distinguishable from "the caller asked for 10 s" — see `worker_settings_from_config`.
_DEFAULT_GRACE_PERIOD_S: Final = 10.0


def _grace_from_env(env: Mapping[str, str]) -> float:
    """The shutdown grace from the environment. A malformed value is refused, never ignored.

    A non-numeric value silently treated as unset would leave a container running the 10 s default
    while its platform waits 300 s — the machine would exit early and hand every in-flight turn to a
    retry for a reason nobody could see in a config file.
    """
    raw = env.get(GRACE_PERIOD_ENV)
    if raw is None:  # pragma: no cover - the caller only asks when it is present
        return _DEFAULT_GRACE_PERIOD_S
    text = raw.strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise WorkerError(
            f"{GRACE_PERIOD_ENV} must be a number of seconds (got {raw!r}).",
            code="worker_settings_invalid",
            details={"field": "grace_period_s", "value": raw, "source": GRACE_PERIOD_ENV},
        ) from exc
    # NaN fails every comparison, including the `< 0` check below, so it is refused explicitly rather
    # than slipping through as "not negative".
    if value != value or value < 0:
        raise WorkerError(
            f"{GRACE_PERIOD_ENV} must be a non-negative number of seconds (got {raw!r}).",
            code="worker_settings_invalid",
            details={"field": "grace_period_s", "value": raw, "source": GRACE_PERIOD_ENV},
        )
    return value


def _concurrency_from_env(env: Mapping[str, str]) -> int:
    """The concurrency from the environment, or the default. A MALFORMED value is refused.

    Refused rather than ignored: a container whose `CODNA_WORKER_MAX_CONCURRENT_ACTIVITIES=two` was
    silently treated as "unset" would run at the default and an operator debugging a memory
    regression would be reading a number the machine never used.
    """
    raw = env.get(CONCURRENCY_ENV)
    if raw is None:
        return DEFAULT_MAX_CONCURRENT_ACTIVITIES
    text = raw.strip()
    if not text.isdigit():
        raise WorkerError(
            f"{CONCURRENCY_ENV} must be a positive integer (got {raw!r}).",
            code="worker_settings_invalid",
            details={"field": "max_concurrent_activities", "value": raw, "source": CONCURRENCY_ENV},
        )
    value = int(text)
    if value < 1:
        raise WorkerError(
            f"{CONCURRENCY_ENV} must be at least 1 (got {value}).",
            code="worker_settings_invalid",
            details={"field": "max_concurrent_activities", "value": raw, "source": CONCURRENCY_ENV},
        )
    return value


class WorkerError(RuntimeError):
    """The worker could not be configured or started. Always carries a stable ``code``."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
        if self.hint:
            body["hint"] = self.hint
        return {"error": body}


@dataclass(frozen=True)
class WorkerSettings:
    """What the worker polls, with what, and how much of it.

    Frozen and explicit: the CLI resolves it once from `DurableConfig` (plus flags), and `run_worker`
    does not read the environment again. A setting that changed mid-run would be a worker whose
    registration and whose activities disagree.
    """

    address: str
    namespace: str
    task_queue: str
    engine: str
    build_id: str
    runs_root: Path
    sidecar_url: str
    sidecar_token: str | None = None
    max_concurrent_activities: int = DEFAULT_MAX_CONCURRENT_ACTIVITIES
    ensure_sidecar: bool = True
    grace_period_s: float = 10.0
    # Promote this build to the deployment's current version when nothing is current yet. On by
    # default so the first worker on a fresh server actually receives runs; a host that must not
    # move cluster traffic turns it off and promotes deliberately.
    promote_current: bool = True
    # `local` (loopback dev server, cleartext) or `cloud` (TLS + API key). Defaults to `local` so a
    # caller that only cares about a dev server keeps working unchanged.
    profile: str = "local"
    # The env var the API key is read from at connect time, or None. NEVER the key itself: this
    # dataclass is frozen with a generated `repr`, and `to_payload()` is printed.
    api_key_env: str | None = None
    # D6: the pool this worker belongs to, or None for a per-host queue. Reported (never secret) so
    # an operator reading the banner can tell a pooled worker from a laptop one.
    pool_name: str | None = None
    pool_workers: int | None = None
    pool_state: str | None = None
    # THE APP'S SUPERVISION CONTRACT. True when this process was started BY the IDE and must therefore
    # stop when that app goes away (`--guard-parent-pid`). False for a worker a human typed, which must
    # behave exactly as it always did: a terminal running `codna worker` is not a supervisor and losing
    # it is not a reason to abandon an in-flight turn. See `parent_guard.py` for the mechanism.
    guard_parent: bool = False
    # The exact parent pid the desktop named. `True` without this value is not a supervision contract: it
    # could silently guard a different process if the child was reparented between spawn and startup.
    guard_parent_pid: int | None = None
    # D6: HOW to authenticate, resolved from the config and called at connect time. A zero-argument
    # callable rather than a resolved dict so the credential is read as late as possible (a rotation
    # between settings-resolution and connect is honoured), and a plain function rather than a
    # `DurableConfig` so this frozen dataclass never holds a credential-shaped object at all.
    connect_options: Callable[[], Mapping[str, object]] | None = None

    @property
    def is_cloud(self) -> bool:
        return self.profile == "cloud"

    def to_payload(self) -> dict[str, object]:
        """The startup banner. No token, no key, no path to one: this is printed and may be logged."""
        return {
            "address": self.address,
            "namespace": self.namespace,
            "task_queue": self.task_queue,
            "engine": self.engine,
            "build_id": self.build_id,
            "runs_root": str(self.runs_root),
            "sidecar_url": self.sidecar_url,
            "sidecar_auth": self.sidecar_token is not None,
            "max_concurrent_activities": self.max_concurrent_activities,
            "ensure_sidecar": self.ensure_sidecar,
            "deployment": DEPLOYMENT_NAME,
            "version": f"{DEPLOYMENT_NAME}.{self.build_id}",
            "promote_current": self.promote_current,
            "grace_period_s": self.grace_period_s,
            # The PROFILE, so an operator can tell "this worker is polling Cloud" from its own
            # output, plus the NAME of the credential variable — never the credential.
            "temporal_profile": self.profile,
            "tls": self.is_cloud,
            "api_key_env": ENV_CLOUD_API_KEY if self.api_key_env == ENV_CLOUD_API_KEY else None,
            # D6: which POOL this worker polls. `None` for a per-host queue, which is the D2 shape.
            "pool": self.pool_name,
            "pool_workers": self.pool_workers,
            "pool_state": self.pool_state,
            # The supervision contract, stated in the banner: True means this worker stops with the app
            # that forked it (`--guard-parent-pid`), False means it is a hand-run worker with no owner.
            "guard_parent": self.guard_parent,
            "guard_parent_pid": self.guard_parent_pid,
        }


def worker_settings_from_config(
    config: "DurableConfig",
    *,
    max_concurrent_activities: int | None = None,
    ensure_sidecar: bool = True,
    grace_period_s: float = 10.0,
    promote_current: bool = True,
    engine: str | None = None,
    env: Mapping[str, str] | None = None,
    guard_parent: bool = False,
    guard_parent_pid: int | None = None,
) -> WorkerSettings:
    """Resolve the worker's settings from the durable config, validating the concurrency cap.

    The cap is refused when it is not a positive integer rather than clamped: a worker launched with
    ``--max-concurrent-activities 0`` by a typo would poll and never execute anything, and the run
    would look like a server problem.
    ``concurrency`` comes from the flag, else `CODNA_WORKER_MAX_CONCURRENT_ACTIVITIES`, else the
    default. An env fallback exists because a CONTAINER has no argument-per-machine: the image's
    sizing is a per-deployment decision, so the pool's Fly toml sets it once for every machine the
    same way. A value from the flag always wins, so a manual run is unaffected by a stale env var.
    """
    resolved_env = os.environ if env is None else env
    cap = (
        _concurrency_from_env(resolved_env)
        if max_concurrent_activities is None
        else max_concurrent_activities
    )
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise WorkerError(
            "max_concurrent_activities must be an integer >= 1.",
            code="worker_settings_invalid",
            details={
                "field": "max_concurrent_activities",
                "value": repr(max_concurrent_activities),
                "source": "flag" if max_concurrent_activities is not None else CONCURRENCY_ENV,
            },
        )
    # The default grace is 10 s (a CLI where a Ctrl-C should not wait). A container's default has to
    # fill the platform's kill_timeout, so an ENV value is read when the caller passed nothing. The
    # signature keeps its float default for every existing caller; `None` is what means "not given".
    resolved_grace = grace_period_s
    if grace_period_s == _DEFAULT_GRACE_PERIOD_S and GRACE_PERIOD_ENV in resolved_env:
        resolved_grace = _grace_from_env(resolved_env)
    if resolved_grace < 0:
        raise WorkerError(
            "grace_period_s must be a non-negative number.",
            code="worker_settings_invalid",
            details={"field": "grace_period_s", "value": repr(resolved_grace)},
        )
    # An explicit engine wins over the config's default — `--engine` exists so one host can run the
    # plain fallback without editing its config for every other command. Validated by `plugins_for`
    # at startup, so an unknown name fails there rather than being carried into registration.
    chosen_engine = engine if engine is not None else config.graph_engine
    token = resolved_env.get(SIDECAR_TOKEN_ENV) or None
    return WorkerSettings(
        address=config.address,
        namespace=config.namespace,
        task_queue=config.task_queue,
        engine=chosen_engine,
        build_id=config.build_id,
        runs_root=config.runs_root,
        sidecar_url=config.sidecar_url,
        sidecar_token=token,
        max_concurrent_activities=cap,
        ensure_sidecar=ensure_sidecar,
        grace_period_s=resolved_grace,
        promote_current=promote_current,
        profile=config.profile,
        api_key_env=config.api_key_env,
        pool_name=config.pool.name if config.pool else None,
        pool_workers=config.pool.workers if config.pool else None,
        pool_state=config.pool.state if config.pool else None,
        connect_options=config.connect_options,
        guard_parent=guard_parent,
        guard_parent_pid=guard_parent_pid,
    )


def github_token_provider(env: Mapping[str, str] | None = None) -> Callable[[], str | None]:
    """A lazily-resolved GitHub write token: environment first, then the OS keystore.

    Lazy on purpose. Resolving at startup would make a worker that only plans (nobody ever opens a
    pull request) require a token it never uses; resolving per call keeps the failure where it
    belongs — in the ``open_pr`` activity, which already reports ``activity_pr_token_missing``.
    """

    def resolve() -> str | None:
        resolved_env = os.environ if env is None else env
        for name in GITHUB_TOKEN_ENV:
            value = resolved_env.get(name)
            if value and value.strip():
                return value.strip()
        try:
            from .. import keystore

            for name in GITHUB_TOKEN_ENV:
                value = keystore.get_key(name)
                if value and value.strip():
                    return value.strip()
        except Exception:  # noqa: BLE001 - a keystore that cannot open is "no token", not a crash
            return None
        return None

    return resolve


def _log_json(event: str, **fields: Any) -> None:
    """One JSON line per worker event, on stderr. Context first, always machine-readable."""
    payload = {"event": event, **fields}
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


def build_activity_runtime(
    settings: WorkerSettings,
    *,
    sidecar_factory: Callable[[str], Any] | None = None,
    ledger_clock: Callable[[], float] | None = None,
    github_token: Callable[[], str | None] | None = None,
    log: Callable[[str], None] | None = None,
) -> Any:
    """The REAL `ActivityRuntime`: the sidecar, the ledger and the run directory layout.

    The sidecar factory creates one client per activity and the step layer closes it in a
    ``finally`` (see `activity_steps._close`), so this returns a fresh client each call. It is the
    only place the sidecar's bearer token is attached, and the token never appears in the runtime's
    ``repr`` because `SidecarClient` keeps it private.
    """
    from .activities import ActivityRuntime

    from .ledger import RunLedger

    runs_root = Path(settings.runs_root)

    def default_sidecar(owner: str) -> Any:
        """A client scoped to ONE run's owner.

        The owner must match the one the session was created with: the engine stamps the session at
        CREATE time and rejects a later request whose ``x-session-owner`` disagrees. A client built
        without an owner sends ``anonymous`` and is refused on its own session's first turn
        (``403 session_owner_mismatch``), so the owner is a required argument rather than a default.
        """
        from .sidecar_client import SidecarClient

        return SidecarClient(
            settings.sidecar_url, auth_token=settings.sidecar_token, owner=owner
        )

    def ledger_for(run_id: str) -> Any:
        return RunLedger.for_run(_paths_for(settings), run_id, clock=ledger_clock)

    return ActivityRuntime(
        runs_root=runs_root,
        sidecar_factory=sidecar_factory or default_sidecar,
        ledger_factory=ledger_for,
        clock=ledger_clock or _monotonic_clock(),
        log=log or (lambda line: print(line, file=sys.stderr, flush=True)),
        github_token=github_token or github_token_provider(),
    )


def _monotonic_clock() -> Callable[[], float]:
    """The activity clock: wall time, because it stamps ledger records that outlive the process."""
    import time

    return time.time


def _paths_for(settings: WorkerSettings) -> Any:
    """A `DurablePaths` for the ledger, derived from the settings the worker was given.

    The ledger only needs ``runs_root`` and ``root`` (its siblings are used by the dev-server and the
    CLI), but constructing the full value keeps one definition of the layout. ``root`` is the runs
    root's parent, which is what `config.load_durable_config` produces.
    """
    from .config import DurablePaths

    runs_root = Path(settings.runs_root)
    root = runs_root.parent
    return DurablePaths(
        root=root,
        temporal_dir=root / "temporal",
        dev_db_path=root / "temporal" / "dev.db",
        dev_server_bin_dir=root / "temporal" / "bin",
        runs_root=runs_root,
        logs_dir=root / "logs",
    )


def plugins_for(engine: str) -> list[Any]:
    """The Temporal plugins to register for an engine.

    A FRESH graph per call: `LangGraphPlugin.__init__` consumes the graph's node metadata, so a
    reused graph raises on the second plugin (D2.0 spike, finding 2). The plain engine registers no
    plugin at all — adding one would be harmless at best and a duplicate-activity error at worst.
    """
    if engine == "plain":
        return []
    if engine != "langgraph":
        raise WorkerError(
            f"unknown graph engine {engine!r}; expected one of langgraph, plain.",
            code="worker_engine_invalid",
            details={"engine": engine, "allowed": ["langgraph", "plain"]},
        )
    from .graph import build_task_graph

    return [build_task_graph().plugin]


def deployment_version(settings: WorkerSettings) -> Any:
    """The ``WorkerDeploymentVersion`` this worker registers: the deployment name plus its build id.

    ``settings.build_id`` is ``codna-<cli version>-g<GRAPH_VERSION>`` (see `config.build_id_for`), so
    a graph change is automatically a new deployment version — the property the replay histories and
    the plan's "old runs finish on the old build" risk mitigation both depend on. Imported lazily so
    the base CLI can call `worker_settings_from_config` without the extra.
    """
    from temporalio.worker import WorkerDeploymentVersion

    return WorkerDeploymentVersion(deployment_name=DEPLOYMENT_NAME, build_id=settings.build_id)


def deployment_version_config(settings: WorkerSettings) -> Any:
    """The ``WorkerDeploymentConfig`` that registers this build as a PINNED deployment version.

    ``use_worker_versioning=True`` is required for the version to exist at all; PINNED is the default
    behaviour so a workflow started with no override still lands on the current build and — once
    started — cannot be moved to a newer one mid-run. A ``build_id`` argument cannot be combined with
    this object (Temporal raises ``ValueError``), so the build id lives INSIDE the version.
    """
    from temporalio.common import VersioningBehavior
    from temporalio.worker import WorkerDeploymentConfig

    return WorkerDeploymentConfig(
        version=deployment_version(settings),
        use_worker_versioning=True,
        default_versioning_behavior=VersioningBehavior.PINNED,
    )


async def promote_current_version(client: Any, settings: WorkerSettings) -> dict[str, object]:
    """Make this build the deployment's CURRENT version when nothing is current yet.

    Called after the worker is polling (a deployment only exists once pollers have attached). The
    promotion is deliberately conservative: if the deployment already has a current version, this
    does NOTHING and reports it. Moving an existing current version is a rollout decision, not
    something a worker should do on its own — a second build registering would otherwise steal
    traffic from runs still in flight on the first.

    Returns a small, JSON-safe report for the startup banner. It never raises on "cannot promote"
    (an older server without deployment support): it reports the reason and lets the worker run,
    because the worker's job is to poll the queue, not to own cluster rollout policy.
    """
    from temporalio.api.workflowservice.v1 import (
        DescribeWorkerDeploymentRequest,
        SetWorkerDeploymentCurrentVersionRequest,
    )

    version = f"{DEPLOYMENT_NAME}.{settings.build_id}"
    # Bounded: a deployment's existence is eventual (pollers must attach first), so a short wait is
    # normal — but a long one would make the worker's banner the slowest part of its startup. The
    # wait only delays the REPORT; the worker is already polling and executing either way.
    deadline = asyncio.get_running_loop().time() + PROMOTION_WAIT_S
    current: str | None = None
    last_reason = "not_described"
    while asyncio.get_running_loop().time() < deadline:
        try:
            described = await client.workflow_service.describe_worker_deployment(
                DescribeWorkerDeploymentRequest(
                    namespace=settings.namespace, deployment_name=DEPLOYMENT_NAME
                )
            )
            info = described.worker_deployment_info
            current = info.routing_config.current_version or ""
            if any(v.version == version for v in info.version_summaries):
                break
            last_reason = "not_registered_yet"
        except Exception as exc:  # noqa: BLE001 - reported, never fatal: the worker still polls
            last_reason = type(exc).__name__
        await asyncio.sleep(0.2)
    else:
        return {"promoted": False, "version": version, "reason": last_reason, "current": current}
    if current and current != UNVERSIONED:
        return {"promoted": False, "version": version, "reason": "already_current", "current": current}
    try:
        await client.workflow_service.set_worker_deployment_current_version(
            SetWorkerDeploymentCurrentVersionRequest(
                namespace=settings.namespace,
                deployment_name=DEPLOYMENT_NAME,
                version=version,
                allow_no_pollers=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {"promoted": False, "version": version, "reason": type(exc).__name__, "current": current}
    return {"promoted": True, "version": version, "reason": "promoted", "current": version}


async def registered_build_ids(client: Any, settings: WorkerSettings) -> set[str]:
    """The build ids of the pollers currently attached to this worker's task queue.

    This is the question a run must answer before it PINS: "is the build that owns this code actually
    polling my queue?" A pinned start against a queue whose pollers registered no deployment version
    is REFUSED by the server (measured: ``Pinned version ... is not present in task queue``), so a CLI
    that pinned unconditionally would fail on every plain worker. The answer is read from the server's
    own ``DescribeTaskQueue`` — each poller carries its ``deployment_options`` — rather than inferred
    from configuration, so a worker that failed to start cannot masquerade as one that is registered.

    An unversioned poller has no ``deployment_options``, so its build id is absent. An empty set means
    "no build is registered on this queue" — pinning there would fail, so the caller must not.

    Raises `WorkerError` (``worker_queue_inspection_failed``) when the question CANNOT be asked — a
    server too old to implement ``DescribeTaskQueue``, or one that is unreachable. The caller decides
    what that means for it; this function never guesses an answer.
    """
    from temporalio.api.enums.v1 import TaskQueueType
    from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest

    try:
        described = await client.workflow_service.describe_task_queue(
            DescribeTaskQueueRequest(
                namespace=settings.namespace,
                task_queue={"name": settings.task_queue},
                task_queue_type=TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW,
                report_task_reachability=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - reported by the caller as "cannot pin"
        raise WorkerError(
            f"could not read the pollers on task queue {settings.task_queue!r}: {exc}",
            code="worker_queue_inspection_failed",
            details={
                "task_queue": settings.task_queue,
                "address": settings.address,
                "cause": type(exc).__name__,
            },
            hint="the server may predate DescribeTaskQueue; pass --no-pin to start the run unpinned",
        ) from exc
    builds: set[str] = set()
    for poller in described.pollers:
        options = getattr(poller, "deployment_options", None)
        if options is None or not getattr(options, "build_id", ""):
            continue
        builds.add(str(options.build_id))
    return builds


async def wait_until_registered(
    client: Any, settings: WorkerSettings, *, timeout_s: float = REGISTRATION_WAIT_S
) -> dict[str, object]:
    """Wait (bounded) until THIS build's poller is visible on the task queue.

    A worker is not "ready to receive a run" the moment ``worker.run()`` is scheduled: its pollers
    attach a beat later, and Temporal only creates the deployment version when a versioned poller
    arrives (measured: the first describe shows 0 pollers, the next shows the build). A caller that
    started a run at that instant would be told "no worker is registered under this build" — which is
    true for a moment and wrong in substance.

    Returns a JSON-safe report either way; ``ready`` is what a caller branches on.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    last: set[str] = set()
    while asyncio.get_running_loop().time() < deadline:
        try:
            last = await registered_build_ids(client, settings)
        except WorkerError as exc:
            return {"ready": False, "reason": exc.code, "builds": [], "build_id": settings.build_id}
        if settings.build_id in last:
            return {"ready": True, "reason": "registered", "builds": sorted(last),
                    "build_id": settings.build_id}
        await asyncio.sleep(0.1)
    return {
        "ready": False,
        "reason": "not_visible",
        "builds": sorted(last),
        "build_id": settings.build_id,
        "timeout_s": timeout_s,
    }


def _ensure_sidecar(settings: WorkerSettings) -> str:
    """Make sure agent-core is answering, and return the URL the activities should use.

    Uses the same `ensure_agent_core_running` the CLI uses, so the worker attaches to the runtime the
    user already has instead of starting a second one on another port. A failure is reported with
    the runtime's own structured error, not a bare "connection refused" from the first activity.
    """
    from ..agent_core_runtime import ensure_agent_core_running

    try:
        endpoint = ensure_agent_core_running()
    except Exception as exc:  # noqa: BLE001 - re-raised as a structured worker failure
        details = getattr(exc, "details", None)
        raise WorkerError(
            f"the agent-core sidecar is not available: {exc}",
            code="worker_sidecar_unavailable",
            details={
                "cause": details if isinstance(details, dict) else {},
                "cause_code": str(getattr(exc, "code", "") or ""),
                "sidecar_url": settings.sidecar_url,
            },
            hint="run `codna doctor` (or `codna login`) and try again",
        ) from exc
    return endpoint.url or settings.sidecar_url


async def run_worker(
    settings: WorkerSettings,
    *,
    stop: asyncio.Event | None = None,
    on_started: Callable[[dict[str, object]], None] | None = None,
    install_signal_handlers: bool = True,
    runtime: Any | None = None,
    client: Any | None = None,
) -> None:
    """Connect, register, poll until ``stop`` is set, then shut down cleanly.

    ``stop``/``on_started``/``runtime``/``client`` are injectable so a test can drive the whole
    lifecycle — including registration against a real server — without a terminal signal, a real
    sidecar or a spawned binary. The CLI passes none of them.
    """
    from . import require_durable

    require_durable()  # structured `durable_extra_missing` before any import below

    plugins = plugins_for(settings.engine)
    if settings.ensure_sidecar and runtime is None:
        sidecar_url = _ensure_sidecar(settings)
        if sidecar_url != settings.sidecar_url:
            settings = _with_sidecar_url(settings, sidecar_url)

    from .activities import ACTIVITIES, activity_executor, configure_default_runtime
    from .workflows import registered_workflows

    effective_runtime = runtime if runtime is not None else build_activity_runtime(settings)
    configure_default_runtime(effective_runtime)

    from temporalio.client import Client
    from temporalio.worker import Worker

    if client is not None:
        connected = client
        # An injected client decides its own converter; report that we are not the ones setting it.
        codec_payload: dict[str, object] = {"mode": "caller_supplied", "note": "codec set by the injected client"}
    else:
        # The profile decides TLS + CREDENTIAL SHAPE in ONE place: `DurableConfig.connect_options`
        # (D6 hands the choice to `credentials.resolve_temporal_credentials`, which supports both an
        # API key and mTLS), resolved by the caller and handed in as a callable. It is called HERE so
        # the credential is read as late as possible — a key rotated into the process between
        # settings-resolution and connect is the one that gets used, which is what an operator
        # restarting a worker expects. For the local profile the options are empty and this call is
        # byte-for-byte what it was.
        options: dict[str, object] = (
            dict(settings.connect_options()) if settings.connect_options is not None else {}
        )
        # E3: the payload codec is decided HERE, beside the connection, because a client without it
        # writes PLAINTEXT history. `resolve_codec` reports which mode it chose, so a run with no key
        # is visibly unencrypted rather than silently so.
        from .codec_config import CodecUnavailable, data_converter_kwargs, resolve_codec

        try:
            resolution = resolve_codec()
        except CodecUnavailable as exc:
            raise WorkerError(
                f"the Temporal payload codec is required but unusable: {exc}",
                code=exc.code,
                details=exc.details,
            ) from exc
        options.update(data_converter_kwargs(resolution))
        # Reported in the banner: a worker running PLAINTEXT must say so, because "history is not
        # encrypted" is a fact an operator has to be able to see rather than infer from a missing key.
        codec_payload = resolution.to_dict()
        try:
            connected = await Client.connect(
                settings.address, namespace=settings.namespace, **options
            )
        except Exception as exc:  # noqa: BLE001 - wrapped with the address and a fix
            raise WorkerError(
                f"could not connect to the Temporal server at {settings.address}: {exc}",
                code="worker_server_unreachable",
                details={
                    "address": settings.address,
                    "namespace": settings.namespace,
                    "temporal_profile": settings.profile,
                    "tls": settings.is_cloud,
                },
                hint=(
                    "check TEMPORAL_ENDPOINT/NAMESPACE/API_KEY and that this namespace grants the key"
                    if settings.is_cloud
                    else "start it with `codna durable dev-server start`, or set CODNA_TEMPORAL_ADDRESS"
                ),
            ) from exc

    stop_event = stop if stop is not None else asyncio.Event()
    executor = activity_executor(max_workers=settings.max_concurrent_activities)
    # A deployment version only exists while pollers are attached, so "register" means "construct and
    # start". The banner is therefore emitted AFTER `worker.run()` has begun polling.
    config = deployment_version_config(settings)
    worker = _register(
        Worker,
        connected,
        settings=settings,
        workflows=list(registered_workflows()),
        activities=list(ACTIVITIES),
        activity_executor=executor,
        plugins=plugins,
        deployment_config=config,
        # THE bound that makes the drain meaningful. Temporal's default is ZERO, which cancels
        # in-flight activities the instant shutdown is called — a paid turn abandoned mid-stream, then
        # re-delivered and paid for again. This is what `--grace-period` configures.
        graceful_shutdown_timeout=timedelta(seconds=settings.grace_period_s),
    )
    try:
        with contextlib.ExitStack() as stack:
            stack.callback(executor.shutdown, wait=False)
            if install_signal_handlers:
                _install_signal_handlers(stop_event, stack)
            run_task = asyncio.ensure_future(worker.run())
            banner = settings.to_payload()
            banner["workflows"] = [cls.__name__ for cls in registered_workflows()]
            banner["activities"] = [activity.__name__ for activity in ACTIVITIES]
            # E3: which payload-encryption mode this worker actually installed. A plaintext worker is
            # visible here, next to the workflows it registered.
            banner["codec"] = codec_payload
            if settings.promote_current:
                banner["promotion"] = await promote_current_version(connected, settings)
            # "Started" must mean "a run started now will reach me": pollers attach a beat after
            # `run()` is scheduled, so the banner reports the registration it OBSERVED. This is the
            # difference between `codna run --worker` working on the first try and a run that is told
            # "no worker is registered under this build" for the 200ms it takes a poller to arrive.
            banner["registration"] = await wait_until_registered(connected, settings)
            _log_json("worker.starting", **banner)
            if on_started is not None:
                on_started(banner)
            try:
                await _wait_for_stop(stop_event, run_task)
            finally:
                await _drain(worker, run_task, grace_period_s=settings.grace_period_s)
    except BaseException:
        executor.shutdown(wait=False)
        raise
    _log_json("worker.stopped", task_queue=settings.task_queue, build_id=settings.build_id)


def _register(worker_cls: Any, client: Any, *, settings: WorkerSettings, **kwargs: Any) -> Any:
    """Construct the Worker, turning a refused registration into a structured startup error.

    Registration is where an inconsistent workflow/activity set surfaces (a duplicate activity name,
    a workflow the sandbox refuses, a sync activity with no executor). All three are code bugs, and
    all three read as "the worker will not start" — so they get one code and one hint.
    """
    try:
        return worker_cls(client, task_queue=settings.task_queue, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a registration error is a startup error
        raise WorkerError(
            f"Temporal refused the worker registration: {exc}",
            code="worker_registration_failed",
            details={"task_queue": settings.task_queue, "engine": settings.engine},
            hint="this is a code bug: the workflow/activity sets are inconsistent",
        ) from exc


async def _drain(worker: Any, run_task: "asyncio.Future[Any]", *, grace_period_s: float) -> None:
    """Stop accepting work and let in-flight activities finish, then return.

    ``Worker.shutdown`` asks Temporal to stop delivering new tasks and waits up to the worker's
    ``graceful_shutdown_timeout`` (set from ``grace_period_s``) for activities already running. It is
    called even when ``worker.run()`` has already ended on its own, because activities execute on a
    separate thread pool and one may still be mid-turn. A failure from ``run()`` is re-raised after
    the worker is stopped, so a fatal poller error is never reported as "stopped cleanly".

    Exposed as its own function so the drain is testable without a signal, a server or a subprocess —
    the failure it prevents (a re-delivered, twice-paid turn) is invisible in a happy-path test.
    """
    failure: BaseException | None = None
    # `shutdown` is called UNCONDITIONALLY, even when the run loop has already ended. Activities run
    # on a separate thread pool, so a `run()` that raised does not mean the activities are done — a
    # paid turn can still be mid-flight and must not be abandoned. `Worker.shutdown` is documented as
    # safe for repeated invocations and before the worker starts, so calling it here is cheap.
    #
    # Its own exceptions are suppressed: a worker that has already failed cannot be "shut down more",
    # and the REAL failure below is the one worth reporting.
    with contextlib.suppress(Exception):
        await worker.shutdown()
    try:
        await asyncio.wait_for(asyncio.shield(run_task), timeout=max(grace_period_s, 0.0) + 60.0)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        run_task.cancel()
        with contextlib.suppress(BaseException):
            await run_task
    except BaseException as exc:  # noqa: BLE001 - re-raised after the worker is fully stopped
        failure = exc
    if failure is not None:
        _log_json(
            "worker.run_failed",
            task_queue=worker.task_queue,
            error=type(failure).__name__,
            message=str(failure),
        )
        raise failure


def _with_sidecar_url(settings: WorkerSettings, sidecar_url: str) -> WorkerSettings:
    from dataclasses import replace

    return replace(settings, sidecar_url=sidecar_url)


async def _wait_for_stop(stop: asyncio.Event, run_task: "asyncio.Future[Any]") -> None:
    """Wait for a stop request, or for ``worker.run()`` to end by itself.

    Both are awaited, not just the event: a `worker.run()` that raises (a fatal poller error, a
    lost server) must end the process instead of leaving the CLI hanging on an event that will
    never be set.
    """
    stop_waiter = asyncio.ensure_future(stop.wait())
    try:
        done, _pending = await asyncio.wait(
            {stop_waiter, run_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if run_task in done:
            exc = run_task.exception()
            if exc is not None:
                raise exc
    finally:
        stop_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_waiter


def _install_signal_handlers(stop: asyncio.Event, stack: contextlib.ExitStack) -> None:
    """SIGINT/SIGTERM set the stop event. Installed only where the loop supports it (not Windows).

    Registered handlers are removed on exit so a CLI invoked twice in one process (the tests, an
    embedded caller) does not accumulate them.
    """
    loop = asyncio.get_running_loop()
    if not hasattr(loop, "add_signal_handler"):
        return  # pragma: no cover - Windows; KeyboardInterrupt still ends the process
    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - restricted environments
            continue
        stack.callback(_remove_signal_handler, loop, number)


def _remove_signal_handler(loop: asyncio.AbstractEventLoop, number: int) -> None:
    with contextlib.suppress(NotImplementedError, RuntimeError):
        loop.remove_signal_handler(number)


def run_worker_sync(settings: WorkerSettings) -> int:
    """The CLI's entry point: run one worker until a signal or a fatal error.

    A thin synchronous wrapper, kept here so `cli.py` carries one call and no asyncio knowledge.
    Exit status is 0 for a clean stop; a startup failure raises `WorkerError` before this returns.
    """
    asyncio.run(run_worker(settings))
    return 0
