"""The Temporal half of the Run API: one client, one loop, four operations.

Split out of `run_api.py` for the repository's 1000-line ceiling (`tests/test_modularity.py`), on
the same seam the rest of this package uses: the TRANSPORT (routing, framing, HTTP status codes)
lives in `run_api.py`; what it needs to talk to Temporal lives here. The split is also what makes
the HTTP surface testable without Temporal at all — `RunBackend` is the seam, and a scripted
implementation of four methods is all a route test needs.

**One client, one loop, one thread.** A `temporalio` client is bound to the event loop that created
it, and `ThreadingHTTPServer` runs each request on its own thread. So the client cannot live where
the requests are: a dedicated thread owns a dedicated loop, requests submit a coroutine with
``run_coroutine_threadsafe`` and wait for the result with a TIMEOUT. That timeout is the reason this
module exists as more than a convenience: a Temporal server that stops answering must become a
bounded 504, not request threads pinned forever by a thread-per-request server.

**It calls the CLI's functions, not copies of them.** `_start_run`, `query_workflow`,
`_send_update` and `_signal` are imported from `run_cli`/`cli` and called with the same arguments
`codna run`/`codna durable` pass. A route therefore cannot drift from the command it mirrors — there
is only one implementation of "start a run", "approve a task", "send a signal".
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Final, Mapping

from .config import DurableConfig
from .run_api_support import RunApiError

__all__ = [
    "ALL_ROUTE_KEYS",
    "READ_ROUTE_KEYS",
    "RPC_TIMEOUT_S",
    "TEMPORAL_ROUTE_KEYS",
    "RunBackend",
    "RunCapabilities",
    "TemporalRunBackend",
    "durable_capabilities",
    "is_not_found",
    "ledger_only_capabilities",
    "translate",
]

# How long one Temporal RPC may take before the request is answered with 504. Bounded because the
# alternative — waiting — pins a request thread in a thread-per-request server.
RPC_TIMEOUT_S: Final = 30.0
# How long the loop thread may take to establish the client before the server refuses to start.
CONNECT_TIMEOUT_S: Final = 30.0


# Which routes this surface can serve WITHOUT a durable executor. Deliberately a declaration beside
# the backend seam rather than something derived from the role table: "a viewer may read" and "a read
# needs no Temporal" are two different facts that happen to agree today, and a role added later must
# not silently move a route between the two sets.
#
# `runs.capabilities` belongs here rather than in its own third bucket: it describes the server itself,
# needs no workflow, and is the route a ledger-only server must ANSWER to describe being ledger-only.
READ_ROUTE_KEYS: Final[frozenset[str]] = frozenset(
    {"runs.capabilities", "runs.list", "runs.get", "runs.events"}
)
# Every route that needs a workflow to act on. `tests/durable/test_run_api_capabilities.py` asserts the
# two sets PARTITION `roles.ROUTE_ROLES`, so a route added without a decision here is a red test and
# never a capability document that quietly omits it.
TEMPORAL_ROUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "runs.start",
        "runs.steer",
        "runs.approve",
        "runs.budget",
        "runs.pause",
        "runs.resume",
        "runs.cancel",
    }
)


# The union, computed ONCE. `durable_capabilities` is called on every `GET /runs` (the IDE's poll), and a
# per-request set union is an allocation with no purpose: the set of routes this build can serve is a property of
# the BUILD, not of the request. Declared beside the two halves so a route added to either is in it automatically.
ALL_ROUTE_KEYS: Final[frozenset[str]] = READ_ROUTE_KEYS | TEMPORAL_ROUTE_KEYS


@dataclass(frozen=True)
class RunCapabilities:
    """What this server can ACTUALLY do, as data — not as prose a client has to interpret.

    The problem this type exists for: a Run API started with ``--no-temporal`` is a legitimate mode
    (it serves the run list, the ledger and the event history), but the only way a caller learned that
    was by attempting a mutation and reading a 503 whose message said a flag had been passed. An
    operator who had just been told to "run `codna serve`" read that as a broken install.

    So the server states its capabilities UP FRONT, in the banner it prints and on the list route the
    IDE already calls. ``available_routes`` / ``unavailable_routes`` name route keys rather than
    sentences, because the client must branch on them; ``execution`` is the one-word summary of the
    same fact for a human, and ``reason`` is the coded refusal a mutation would produce.

    ``summary`` is what the MODE means, in the words a person needs: a degraded server is described
    by what still works first, and by the one requirement it is missing second.
    """

    #: True only when this server can start, steer and control runs (i.e. it owns a Temporal client).
    durable_execution: bool
    #: The Temporal address this server is connected to; ``None`` in a ledger-only deployment.
    temporal_address: str | None
    #: Route keys this server serves. A client may call all of them.
    available_routes: frozenset[str]
    #: Route keys this server refuses. Calling one is a 503 with ``reason_code``.
    unavailable_routes: frozenset[str]

    @property
    def mode(self) -> str:
        """``durable`` or ``ledger-only``. One word, so a client does not have to infer a mode."""
        return "durable" if self.durable_execution else "ledger-only"

    @property
    def reason_code(self) -> str | None:
        """The code an unavailable route answers with, or ``None`` when everything is available."""
        return None if self.durable_execution else "run_api_temporal_disabled"

    @property
    def summary(self) -> str:
        """One sentence a human can read: what works first, then what is missing and why."""
        if self.durable_execution:
            return (
                f"Durable execution is available via Temporal at {self.temporal_address}: "
                "runs can be started, steered and controlled."
            )
        return (
            "Ledger-only mode: listing runs and reading run history work; starting and controlling "
            "runs needs Temporal. Start one with `codna durable dev-server start`, then restart "
            "`codna serve` without --no-temporal."
        )

    def as_payload(self) -> dict[str, Any]:
        """The JSON shape. Sorted lists, because a client diffing two banners must get a stable answer."""
        return {
            "mode": self.mode,
            "durable_execution": self.durable_execution,
            "temporal_address": self.temporal_address,
            "available_routes": sorted(self.available_routes),
            "unavailable_routes": sorted(self.unavailable_routes),
            "reason_code": self.reason_code,
            "summary": self.summary,
        }


def durable_capabilities(temporal_address: str) -> RunCapabilities:
    """The capabilities of a server that owns a Temporal client. Every route is servable.

    A function rather than a class constant because the address is part of the answer: a banner that
    said "durable execution is available" without naming WHICH server would leave an operator unable
    to tell a local dev server from a Cloud namespace.
    """
    return RunCapabilities(
        durable_execution=True,
        temporal_address=temporal_address,
        available_routes=ALL_ROUTE_KEYS,
        unavailable_routes=frozenset(),
    )


def ledger_only_capabilities() -> RunCapabilities:
    """The capabilities of a `--no-temporal` server: the ledger routes, and nothing that starts work.

    THE ONE DEFINITION of this mode. The banner, `GET /capabilities`, the copy on `GET /runs` and the
    503 a mutation raises all read it, so an operator and a client cannot be told two different things
    about the same process.
    """
    return RunCapabilities(
        durable_execution=False,
        temporal_address=None,
        available_routes=READ_ROUTE_KEYS,
        unavailable_routes=TEMPORAL_ROUTE_KEYS,
    )


class RunBackend:
    """The operations the Run API needs from a durable executor.

    Deliberately narrow: four verbs plus shutdown. A route that needed a fifth would be a route the
    CLI could not perform, which is the drift this seam exists to prevent.

    An implementation raises `RunApiError` for a condition the surface reports (unknown run) and may
    raise the CLI's own coded error, which `translate` maps onto a status.
    """

    def capabilities(self) -> RunCapabilities:
        """What this backend can do. NO DEFAULT: an implementation has to state it.

        A permissive default here would be the same defect this seam's guard rails exist to prevent
        — a backend that forgot to answer would be described as fully capable, and the client would
        only learn otherwise by failing a mutation.
        """
        raise NotImplementedError

    def start_run(self, *, payload: Mapping[str, Any], run_id: str, run_timeout_s: int) -> dict[str, Any]:
        raise NotImplementedError

    def status(self, run_id: str) -> dict[str, Any] | None:
        """The run's live status, or None when Temporal has no such open execution."""
        raise NotImplementedError

    def update(self, run_id: str, *, task_id: str, action: str, payload: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def signal(self, run_id: str, action: str) -> None:
        """Send a run-level signal; ``action`` is already the workflow's method name."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


# Codes that mean "the CALLER'S request is wrong", so the surface can say 400 with confidence. A
# closed set rather than a prefix: `durable_query_failed` shares the `durable_` prefix and is emphatic-
# ally NOT the caller's fault (a query timeout is this server's problem, answered as 502).
_CALLER_INPUT_CODES: Final[frozenset[str]] = frozenset(
    {
        "durable_brief_required",
        "durable_task_count_invalid",
        "durable_base_branch_invalid",
        "durable_run_spec_invalid",
        "durable_pin_conflict",
        "durable_worker_conflict",
        "durable_pin_unavailable",
        "durable_identity_required",
        "durable_task_required",
        "durable_task_invalid",
        "durable_note_required",
        "durable_cost_invalid",
        "durable_after_invalid",
        "durable_repo_missing",
        "durable_repo_not_git",
        "durable_repo_outside_allowed_roots",
        "durable_interval_invalid",
        "durable_poll_interval_invalid",
        "durable_run_id_invalid",
        "durable_engine_invalid",
        "contract_invalid",
        "contract_contains_secret",
        "contract_schema_unsupported",
        "unsupported_contract_schema",
        "invalid_run_id",
        "invalid_graph_engine",
    }
)


def is_not_found(exc: BaseException) -> bool:
    """Whether an error means "no such execution" rather than "the server is unwell".

    Checked structurally first (``RPCError.status``) and only then by text, because the CLI layer
    wraps a query failure into a ``DurableCliError`` whose ``__cause__`` holds the original RPCError.
    """
    status = getattr(exc, "status", None)
    if status is not None and str(getattr(status, "name", status)).upper() == "NOT_FOUND":
        return True
    text = str(exc).lower()
    return "not found" in text or "notfound" in text or "no such workflow" in text


def translate(exc: BaseException, *, run_id: str) -> RunApiError:
    """Map a CLI/Temporal failure onto this surface's statuses, preserving code, details and hint.

    The mapping is deliberate, not a catch-all:

    * a missing workflow is 404 — the caller's run id names nothing this host has;
    * a validator's refusal is 409 — the task exists but cannot take that action now;
    * a caller-input refusal (``durable_brief_required``, ``contract_contains_secret``, ...) is 400,
      because the request is what is wrong;
    * anything else is 502: this server failed, and the caller did nothing wrong.
    """
    code = str(getattr(exc, "code", "") or "")
    details: dict[str, object] = {"run_id": run_id}
    raw_details = getattr(exc, "details", None)
    if isinstance(raw_details, dict):
        details.update({str(key): value for key, value in raw_details.items()})
    hint = str(getattr(exc, "hint", "") or "")
    cause = getattr(exc, "__cause__", None)
    if code == "durable_workflow_not_found" or is_not_found(exc) or (
        cause is not None and is_not_found(cause)
    ):
        return RunApiError(
            f"run {run_id} has no such workflow, or it has already completed.",
            code="run_not_found",
            status=HTTPStatus.NOT_FOUND,
            details=details,
            hint=hint or "`codna durable status --run <id>` shows what this host still has",
        )
    if code in ("durable_not_parked", "durable_update_refused"):
        return RunApiError(str(exc), code=code, status=HTTPStatus.CONFLICT, details=details, hint=hint)
    if code == "durable_identity_required":
        return RunApiError(
            str(exc),
            code="run_api_identity_required",
            status=HTTPStatus.BAD_REQUEST,
            details=details,
            hint=hint,
        )
    if code == "durable_run_start_failed":
        # The CLI's catch-all for `start_workflow`. Its usual cause is a duplicate workflow id — the
        # run already exists — which is a CONFLICT: a caller that retried a start must be able to
        # tell "already started" apart from "malformed" and from "the server failed".
        details["cause"] = str(getattr(exc, "__cause__", "") or "")[:400]
        if "duplicate" in (str(exc) + details["cause"]).lower():
            return RunApiError(
                f"run {run_id} has already been started.",
                code="run_already_exists",
                status=HTTPStatus.CONFLICT,
                details=details,
                hint=hint,
            )
        return RunApiError(str(exc), code=code, status=HTTPStatus.BAD_GATEWAY, details=details, hint=hint)
    if code in ("durable_server_unreachable", "durable_extra_missing", "durable_plugin_missing"):
        return RunApiError(str(exc), code=code, status=HTTPStatus.SERVICE_UNAVAILABLE,
                           details=details, hint=hint)
    if code in _CALLER_INPUT_CODES:
        return RunApiError(str(exc), code=code, status=HTTPStatus.BAD_REQUEST, details=details, hint=hint)
    return RunApiError(
        f"the durable backend failed: {exc}",
        code="run_api_backend_failed",
        status=HTTPStatus.BAD_GATEWAY,
        details=details,
    )


class TemporalRunBackend(RunBackend):
    """The real backend, over the CLI's own Temporal calls."""

    def capabilities(self) -> RunCapabilities:
        """Connected to Temporal: every route is servable.

        This is a statement about the CONNECTION, which is why it is a method and not a class constant:
        the backend cannot be constructed without a client, so "connected" is true by construction here
        and the answer is derived from the address it is actually connected to.
        """
        return durable_capabilities(self._config.address)

    def __init__(self, config: DurableConfig, *, connect_timeout_s: float = CONNECT_TIMEOUT_S) -> None:
        self._config = config
        self._loop = asyncio.new_event_loop()
        self._client: Any = None
        self._failure: BaseException | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._serve_loop, name="codna-run-api-temporal", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(connect_timeout_s):
            raise RunApiError(
                f"the Temporal client did not connect within {connect_timeout_s:g}s.",
                code="run_api_temporal_unavailable",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                details={"address": config.address, "namespace": config.namespace},
            )
        if self._failure is not None:
            raise RunApiError(
                f"could not connect to Temporal at {config.address}: {self._failure}",
                code="run_api_temporal_unavailable",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                details={"address": config.address, "namespace": config.namespace},
                hint="start it with `codna durable dev-server start`",
            ) from self._failure

    # -- the loop thread -------------------------------------------------------

    def _serve_loop(self) -> None:
        """Own the event loop for the process's lifetime: connect once, then serve requests."""
        from . import cli as durable_cli

        asyncio.set_event_loop(self._loop)
        try:
            self._client = self._loop.run_until_complete(durable_cli._connect(self._config))
        except BaseException as exc:  # noqa: BLE001 - reported through `_failure`
            self._failure = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    def _call(self, coroutine: Any, *, timeout_s: float = RPC_TIMEOUT_S) -> Any:
        """Run one coroutine on the client's loop, waiting a bounded time for the answer."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout_s)
        except TimeoutError as exc:
            future.cancel()
            raise RunApiError(
                "the Temporal call did not answer in time.",
                code="run_api_temporal_timeout",
                status=HTTPStatus.GATEWAY_TIMEOUT,
                details={"address": self._config.address, "timeout_s": timeout_s},
            ) from exc

    # -- the operations --------------------------------------------------------

    def start_run(self, *, payload: Mapping[str, Any], run_id: str, run_timeout_s: int) -> dict[str, Any]:
        """Start the run workflow through the CLI's own `_start_run` (pin check included)."""
        from . import run_cli

        try:
            handle = self._call(
                run_cli._start_run(
                    self._client,
                    self._config,
                    payload,
                    run_id=run_id,
                    run_timeout_s=run_timeout_s,
                )
            )
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a duplicate id, or an unreachable server
            raise translate(exc, run_id=run_id) from exc
        body = dict(payload)
        return {
            "run_id": run_id,
            "workflow_id": self._config.run_workflow_id(run_id),
            "task_queue": self._config.task_queue,
            "engine": body.get("engine", self._config.graph_engine),
            "tasks": len(body.get("tasks") or ()),
            # The pin outcome is reported, never inferred: "this run was pinned to build X" is a
            # fact a rollout must be able to read back, and an unpinned run is not the same thing.
            "versioning": dict(getattr(handle, "codna_pin_report", {}) or {}),
        }

    def status(self, run_id: str) -> dict[str, Any] | None:
        """The run's live status. ``None`` when Temporal has no such execution (a 404 at the edge)."""
        from . import run_cli
        from .run_workflow import RunWorkflow

        async def query() -> dict[str, Any]:
            return await run_cli.query_workflow(
                self._client, self._config.run_workflow_id(run_id), RunWorkflow.status
            )

        try:
            return self._call(query())
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - "gone" is an answer; anything else is not
            if is_not_found(exc) or is_not_found(getattr(exc, "__cause__", None) or exc):
                return None
            raise translate(exc, run_id=run_id) from exc

    def update(self, run_id: str, *, task_id: str, action: str, payload: Mapping[str, Any]) -> Any:
        """Send one Update to a TASK workflow, through the CLI's engine-discovering sender."""
        from . import cli as durable_cli

        workflow_id = durable_cli._task_workflow_id(run_id, task_id)
        try:
            return self._call(durable_cli._send_update(self._client, workflow_id, action, dict(payload)))
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a validator's refusal is the answer
            raise translate(exc, run_id=run_id) from exc

    def signal(self, run_id: str, action: str) -> None:
        """Send one Signal to the RUN workflow, through the CLI's own sender."""
        from . import cli as durable_cli

        try:
            self._call(durable_cli._signal(self._client, self._config.run_workflow_id(run_id), action))
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001
            raise translate(exc, run_id=run_id) from exc

    # -- shutdown --------------------------------------------------------------

    def close(self) -> None:
        """Stop the loop and join its thread. Idempotent: `stop()` may be called twice."""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()
