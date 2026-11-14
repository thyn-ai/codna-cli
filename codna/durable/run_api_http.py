"""The HTTP transport of the Run API: one handler, one request per thread, one socket.

Split from `run_api.py` for the repository's 1000-line ceiling (`tests/test_modularity.py`), on the
seam this package already uses: the RULES (what a route means, what it validates, when it is a 404)
live in `RunApi`; the THINGS THAT TOUCH A SOCKET live here. A reviewer asking "can a hung client
wedge the server?" reads this file, and one asking "is this allowed?" reads the other.

Three details here are decisions, not boilerplate:

* **``daemon_threads`` + ``block_on_close = False``.** A thread per request means a streaming client
  holds a thread; a daemon thread means a stream that ignores every other bound still cannot keep the
  process alive at shutdown. Without both, ``server_close()`` joins a follower and a stuck client
  hangs the exit.
* **Every blocking read/write has a timeout** (``request_timeout_s``/``write_timeout_s``), and the
  follower polls its socket for EOF rather than waiting for a heartbeat to fail. That is what turns
  "the client vanished" from a 15-second delay into a released thread and a freed stream slot.
* **The status line is written BEFORE the stream body.** Once those bytes are out the status cannot
  be changed, so a failure mid-stream has to be a final ``run.stream_error`` data frame. That is the
  only honest option, and the IDE's parser reads it like any other frame.
"""
from __future__ import annotations

import json
import logging
import select
import socket
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable, Final, Mapping
from urllib.parse import parse_qs, urlsplit

from .config import GRAPH_ENGINES, validate_run_id
from .ledger import LedgerError
from .roles import Principal, required_role, route_key
from .run_api_backend import translate
from .run_repo import repo_dir_of
from .run_api_support import (
    DEFAULT_LIST_LIMIT,
    MUTATION_ROUTES,
    ClientGone,
    LedgerTail,
    RunApiError,
    SSE_EVENT_TEMPLATE,
    SSE_HEADERS,
    SSE_HEARTBEAT_TEMPLATE,
    TERMINAL_EVENT_TYPES,
    after_param,
    event_frame_payload,
    follow_param,
    last_seq_of,
    ledger_verdict,
    limit_param,
    optional_bool,
    optional_positive_int,
    optional_positive_int_or_none,
    optional_positive_number,
    optional_string_list,
    reject_unknown_fields,
    require_identity,
    require_positive_number,
    require_string_list,
    require_task_id,
    validated_run_id,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; `run_api` imports THIS module
    from .run_api import RunApi

logger = logging.getLogger("codna.durable.run_api")

__all__ = ["RunApiHandler", "ThreadingRunApiServer"]

_JSON_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("content-type", "application/json"),
    ("cache-control", "no-store"),
)

# Task-level Updates (a task workflow declares them) vs run-level Signals (the run workflow does).
TASK_UPDATES: Final[frozenset[str]] = frozenset({"steer", "approve", "budget"})
# Route name -> the workflow's signal method. `cancel` is the surface's word for the run's `stop`.
SIGNALS: Final[Mapping[str, str]] = {"pause": "pause", "resume": "resume", "cancel": "stop"}


@dataclass(frozen=True)
class Route:
    """One dispatchable route: its role-bearing KEY, the handler, and the argument shape.

    The key is not decoration. `__post_init__` asks `roles.required_role` for it, so a `Route` whose
    key is missing from `ROUTE_ROLES` cannot be constructed at all — and the module-level table below
    is built at import, so "somebody added a route and forgot its role" is an ImportError rather
    than an endpoint that serves whoever can reach it.

    ``handler`` is the bound method NAME, not the method: these are bound per request instance, and a
    frozen dataclass holding a bound method of a short-lived handler is a reference to a dead object.
    """

    key: str
    method: str
    handler: str
    # ``needs_id`` routes receive the run id; ``needs_query`` ones receive the query mapping.
    needs_id: bool = False
    needs_query: bool = False
    needs_action: bool = False

    def __post_init__(self) -> None:
        # Raises UnknownRouteError — deliberately, and at import time.
        required_role(self.key)


# THE route table. Every reachable route is one of these; `_resolve` cannot produce anything else, and
# `_serve` authorizes the key BEFORE it calls the handler. There is no default and no fallback.
ROUTES: Final[tuple[Route, ...]] = (
    Route("runs.capabilities", "GET", "_capabilities"),
    Route("runs.start", "POST", "_create_run"),
    Route("runs.list", "GET", "_list_runs", needs_query=True),
    Route("runs.get", "GET", "_get_run", needs_id=True),
    Route("runs.events", "GET", "_stream_events", needs_id=True, needs_query=True),
)


def _mutation_routes() -> tuple[Route, ...]:
    """One `POST /runs/:id/<action>` route per mutation, keyed through `roles.route_key`.

    Built by expansion rather than hand-listed so a new entry in `roles.MUTATION_ROUTE_KEYS` is a route
    with a role, and a short name the role table does not know fails the import.
    """
    return tuple(
        Route(route_key(action), "POST", "_mutate", needs_id=True, needs_action=True)
        for action in sorted(MUTATION_ROUTES)
    )


ALL_ROUTES: Final[tuple[Route, ...]] = ROUTES + _mutation_routes()

# Path shape -> the route that serves it. Looked up by (method, segment count/name), never by a
# prefix, so an unrecognised shape is a 404 instead of falling through to something else.
_MUTATION_BY_ACTION: Final[Mapping[str, Route]] = {
    route.key.rsplit(".", 1)[-1]: route for route in ALL_ROUTES if route.key.startswith("runs.") and route.needs_action
}



class ThreadingRunApiServer(ThreadingHTTPServer):
    """A threaded HTTP server whose threads cannot keep the process alive.

    ``daemon_threads`` is the decision that makes shutdown bounded: a streaming handler that ignored
    ``stopping`` would otherwise be joined by ``server_close()``, and one stuck client would hang
    the process. ``block_on_close = False`` says the same for the mixin's own bookkeeping.
    """

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True


@dataclass(frozen=True)
class _Resolved:
    """One request, resolved to a route and its already-validated arguments.

    The run id is validated HERE (at resolution), so a malformed id is a 400 before any role check
    or lookup — and every handler below may assume its id is a legal run id.
    """

    route: Route
    run_id: str = ""
    action: str = ""
    query: Mapping[str, list[str]] = field(default_factory=dict)


class RunApiHandler(BaseHTTPRequestHandler):
    """One request per thread. Routing, framing and the body live here; the rules live in `RunApi`."""

    protocol_version = "HTTP/1.1"
    server_version = "codna-run-api/1.0"
    sys_version = ""
    # The bound API, set on the subclass the server creates.
    api: "RunApi"
    # Overridden per instance from the settings before the request is read.
    timeout = 30.0

    _START_FIELDS: Final = frozenset(
        {
            "run_id",
            "repo_dir",
            "briefs",
            "base_branch",
            "verify_commands",
            "max_cost_usd",
            "max_tokens",
            "max_attempts",
            "engine",
            "run_timeout_s",
            # The provider coordinates `codna run` has had all along (`--provider`/`--model`/
            # `--base-url`). They were missing here, so an IDE-submitted run silently got the engine's
            # default provider — a credential error on any host whose key is for another provider.
            # The set stays CLOSED: these three are accepted because they are plumbed to the spec, and
            # anything else (including a key source, which is not a caller's decision) is still a 400.
            "provider",
            "model",
            "base_url",
            # D5: one worktree per task, or the shared checkout. A BOOLEAN, validated below — the set
            # stays closed, and `--no-worktree` is the CLI's spelling of `false` (the default), so the
            # wire has one field rather than two mutually-exclusive ones.
            "worktree",
            # D5/MCP: the servers every task's turns are given. Validated by the same normalizer the
            # spec uses, so a malformed block is a 400 here rather than a run that starts without the
            # server the caller asked for. Transport config + `{"from_env": …}` references only — a
            # literal secret may not travel (the run payload is history).
            "mcp_servers",
            # D7/SWARM: whether every task's turns may delegate. Two BOOLEANS validated against the
            # approval profile below (and again by the spec), so a run cannot ask the wire for a
            # capability its profile forbids and only learn that at session-create time.
            "enable_spawn_agent",
            "enable_agent_teams",
            # D7/MULTITASK: how many tasks may be in flight at once. Validated against `worktree` by
            # the spec (concurrent tasks need one checkout each), so the 400 names the conflict.
            "max_parallel_tasks",
            # D5/X8: run FROM an existing pull request. The work lands on ITS branch, in its own
            # worktree, instead of a fresh `codna/run-...` one. Validated below (and by the spec): the
            # branch must already exist, and `worktree` must be true.
            "worktree_branch",
        }
    )

    # -- plumbing --------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # Method/path/status only, and only at debug: one request header is the bearer token, and a
        # server that logs what it is sent eventually logs that.
        logger.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        self._headers_sent = False
        self._principal: Principal | None = None
        try:
            self.connection.settimeout(self.api.settings.request_timeout_s)
            split = urlsplit(self.path)
            query = parse_qs(split.query, keep_blank_values=True)
            self._route(method, split.path, query)
        except ClientGone:
            self.close_connection = True
        except RunApiError as exc:
            self._send_error(exc)
        except Exception as exc:  # noqa: BLE001 - a route must never take the server down
            logger.exception("unhandled run API failure on %s", method)
            self._send_error(
                RunApiError(
                    "the run API failed to serve this request.",
                    code="run_api_internal_error",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    details={"method": method, "cause": type(exc).__name__},
                )
            )

    def _route(self, method: str, path: str, query: Mapping[str, list[str]]) -> None:
        """Resolve the request to ONE `Route`, authorize it, then serve it.

        The order is the control: the route is known before any credential is looked at (so an
        unknown path is a 404 without an auth probe), the principal is established once, and the
        role check happens BEFORE the handler runs — so no handler can be reached by a caller who
        was never authorized. A route the table does not contain cannot be resolved, which is why
        there is no "unprotected route" to add by accident.
        """
        resolved = self._resolve(method, path)
        principal = self.api.authenticate(self.headers.get("authorization"))
        # Raises `run_api_forbidden` (403) unless the principal holds the route's role. This is the
        # single enforcement point; a handler is unreachable without passing it.
        self.api.require_role(principal, resolved.route.key)
        self._principal = principal
        self._serve(resolved)

    def _serve(self, resolved: "_Resolved") -> None:
        """Call the resolved handler. Arguments are named explicitly so a shape mismatch is a
        `TypeError` at import-adjacent review time rather than a silently different route."""
        route = resolved.route
        handler: Callable[..., None] = getattr(self, route.handler)
        if route.needs_action:
            handler(resolved.run_id, resolved.action)
            return
        if route.needs_id and route.needs_query:
            handler(resolved.run_id, resolved.query)
            return
        if route.needs_id:
            handler(resolved.run_id)
            return
        if route.needs_query:
            handler(resolved.query)
            return
        handler()

    def _resolve(self, method: str, path: str) -> "_Resolved":
        """Path + method -> `(route, run_id, action, query)`. Unknown shapes are 404, never a fallback."""
        parts = [segment for segment in path.strip("/").split("/") if segment]
        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        if parts == ["capabilities"]:
            # The mode probe, OUTSIDE `/runs` on purpose: it is a fact about the server, not about a
            # run, and a client asking "what can you do" must not have to guess a run id. It is still
            # an authenticated, role-checked route like every other one.
            if method != "GET":
                self._method_not_allowed("GET")
            return _Resolved(self._route_by_key("runs.capabilities"), "", "", query)
        if not parts or parts[0] != "runs":
            raise self._not_found(path)
        rest = parts[1:]
        if method == "POST":
            if not rest:
                return _Resolved(self._route_by_key("runs.start"), "", "", query)
            if len(rest) == 2 and rest[1] in _MUTATION_BY_ACTION:
                return _Resolved(
                    _MUTATION_BY_ACTION[rest[1]], validated_run_id(rest[0]), rest[1], query
                )
            if len(rest) <= 2:
                # The path EXISTS for another method; that is a 405, and it is still not a route a
                # caller can reach: the message names the allowed method rather than serving it.
                self._method_not_allowed("GET")
            raise self._not_found(path)
        if not rest:
            return _Resolved(self._route_by_key("runs.list"), "", "", query)
        if len(rest) == 1:
            return _Resolved(self._route_by_key("runs.get"), validated_run_id(rest[0]), "", query)
        if len(rest) == 2 and rest[1] == "events":
            return _Resolved(self._route_by_key("runs.events"), validated_run_id(rest[0]), "", query)
        if len(rest) == 2 and rest[1] in _MUTATION_BY_ACTION:
            self._method_not_allowed("POST")
        raise self._not_found(path)

    @staticmethod
    def _route_by_key(key: str) -> Route:
        for route in ROUTES:
            if route.key == key:
                return route
        raise AssertionError(f"route {key!r} is not in ROUTES")  # pragma: no cover - table bug

    def _not_found(self, path: str) -> RunApiError:
        return RunApiError(
            f"no route for {path}.",
            code="run_api_not_found",
            status=HTTPStatus.NOT_FOUND,
            details={"path": path},
        )

    def _method_not_allowed(self, allowed: str) -> None:
        raise RunApiError(
            f"{self.command} is not allowed on this route.",
            code="run_api_method_not_allowed",
            status=HTTPStatus.METHOD_NOT_ALLOWED,
            details={"method": self.command, "allowed": allowed},
        )

    # -- responses -------------------------------------------------------------

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        for name, value in _JSON_HEADERS:
            self.send_header(name, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self._headers_sent = True
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise ClientGone() from exc

    def _send_error(self, exc: RunApiError) -> None:
        if self._headers_sent:
            # Mid-stream: the status is on the wire already, so the only honest thing left is a final
            # frame naming the failure, then a close. The IDE's parser reads data frames.
            try:
                self._write(
                    SSE_EVENT_TEMPLATE.format(
                        event_index=0,
                        data=json.dumps(
                            {
                                "event_index": 0,
                                "event_type": "run.stream_error",
                                "error": exc.as_body()["error"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )
            except (ClientGone, OSError):
                pass
            self.close_connection = True
            return
        try:
            self._send_json(exc.status, exc.as_body())
        except ClientGone:
            self.close_connection = True
        if exc.status in (HTTPStatus.REQUEST_ENTITY_TOO_LARGE, HTTPStatus.REQUEST_TIMEOUT):
            # The body was not consumed, so this connection cannot be reused.
            self.close_connection = True

    # -- the body --------------------------------------------------------------

    def _read_body(self) -> dict[str, Any]:
        """Read a bounded JSON object body, refusing rather than buffering an oversized one."""
        raw_length = self.headers.get("content-length")
        if raw_length is None:
            raise RunApiError(
                "a request body must declare content-length.",
                code="run_api_length_required",
                status=HTTPStatus.LENGTH_REQUIRED,
                details={"header": "content-length"},
            )
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RunApiError(
                "content-length must be an integer.",
                code="run_api_length_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"content-length": raw_length[:32]},
            ) from exc
        if length < 0:
            raise RunApiError(
                "content-length must not be negative.",
                code="run_api_length_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"content-length": length},
            )
        limit = self.api.settings.max_body_bytes
        if length > limit:
            # Not one byte is read: a body over the limit is a refusal, not something to buffer.
            raise RunApiError(
                f"the request body exceeds {limit} bytes.",
                code="run_api_body_too_large",
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                details={"content_length": length, "max_body_bytes": limit},
            )
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            raise RunApiError(
                "the request body ended before content-length bytes arrived.",
                code="run_api_body_truncated",
                status=HTTPStatus.BAD_REQUEST,
                details={"expected": length, "received": len(data)},
            )
        if not data:
            return {}
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunApiError(
                f"the request body is not valid JSON: {exc}",
                code="run_api_json_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"error": str(exc)},
            ) from exc
        if not isinstance(parsed, dict):
            raise RunApiError(
                "the request body must be a JSON object.",
                code="run_api_body_not_object",
                status=HTTPStatus.BAD_REQUEST,
                details={"type": type(parsed).__name__},
            )
        return parsed

    # -- POST /runs ------------------------------------------------------------

    def _create_run(self) -> None:
        from . import cli as durable_cli
        from . import run_cli
        from .contract_primitives import ContractError
        from .provider import provider_coordinates_from_request

        body = self._read_body()
        reject_unknown_fields(body, self._START_FIELDS, route="POST /runs")
        default_id = durable_cli._new_run_id()
        run_id = validated_run_id(body.get("run_id") or default_id)
        # The provider coordinates are validated BEFORE the repository is touched: they are the
        # request's own shape, and a caller that named a provider the contract refuses should hear
        # that rather than a git error about a checkout. `provider_key_source` is derived here (never
        # read from the body), so a credential cannot arrive on this route at all.
        try:
            coordinates = provider_coordinates_from_request(body)
        except ContractError as exc:
            raise translate(exc, run_id=run_id) from exc
        raw_repo = body.get("repo_dir")
        if raw_repo is not None and not isinstance(raw_repo, str):
            raise RunApiError(
                "repo_dir must be a string path.",
                code="run_api_field_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "repo_dir"},
            )
        briefs = require_string_list(body, "briefs")
        verify_commands = optional_string_list(body, "verify_commands")
        base_branch = body.get("base_branch", "main")
        if not isinstance(base_branch, str) or not base_branch.strip():
            raise RunApiError(
                "base_branch must be a non-empty string.",
                code="run_api_field_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "base_branch"},
            )
        engine = body.get("engine", self.api.config.graph_engine)
        if engine not in GRAPH_ENGINES:
            raise RunApiError(
                f"engine must be one of {', '.join(GRAPH_ENGINES)}.",
                code="run_api_field_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "engine", "value": str(engine)[:32], "allowed": list(GRAPH_ENGINES)},
            )
        max_cost_usd = optional_positive_number(body, "max_cost_usd")
        max_tokens = optional_positive_int_or_none(body, "max_tokens")
        max_attempts = optional_positive_int(body, "max_attempts", default=3)
        worktree = optional_bool(body, "worktree", default=False)
        # D5/X8: the branch a run-from-a-pull-request works on. Empty is the ordinary case.
        raw_branch = body.get("worktree_branch", "")
        if not isinstance(raw_branch, str):
            raise RunApiError(
                "worktree_branch must be a string.",
                code="run_api_field_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "worktree_branch"},
            )
        worktree_branch = raw_branch.strip()
        if worktree_branch and not worktree:
            # Refused rather than ignored: adopting a branch in the SHARED checkout would move the
            # developer's HEAD, which per-task worktrees exist to prevent.
            raise RunApiError(
                "worktree_branch requires worktree=true: an adopted branch needs its own checkout.",
                code="run_api_field_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "worktree_branch"},
            )
        # D5/MCP: validated HERE, before the spec is built, so a malformed block is a 400 with the
        # offending field named rather than a workflow that starts and fails inside an activity.
        from .mcp import McpConfigError, normalize_mcp_servers

        try:
            mcp_servers = normalize_mcp_servers(body.get("mcp_servers"))
        except McpConfigError as exc:
            raise RunApiError(
                str(exc),
                code=exc.code,
                status=HTTPStatus.BAD_REQUEST,
                details={**exc.details, "field": "mcp_servers"},
            ) from exc
        run_timeout_s = optional_positive_int(
            body, "run_timeout_s", default=run_cli.DEFAULT_RUN_TIMEOUT_S
        )
        # D7/SWARM: refused at the wire boundary, naming the profile, rather than at session create.
        from .swarm import SwarmConfigError, normalize_swarm

        try:
            enable_spawn_agent, enable_agent_teams = normalize_swarm(
                enable_spawn_agent=optional_bool(body, "enable_spawn_agent", default=False),
                enable_agent_teams=optional_bool(body, "enable_agent_teams", default=False),
            )
        except SwarmConfigError as exc:
            raise RunApiError(
                str(exc),
                code=exc.code,
                status=HTTPStatus.BAD_REQUEST,
                details={**exc.details, "field": exc.details.get("field", "swarm")},
            ) from exc
        max_parallel_tasks = optional_positive_int(body, "max_parallel_tasks", default=1)
        try:
            repo_dir = durable_cli._http_repo_dir(raw_repo, allowed_roots=self.api.settings.allowed_repo_roots)
            # Preserve the API's existing missing-policy behavior: no policy means no exemption,
            # never a refusal invented by this feature. A present policy is still read fail-closed.
            auto_approve, policy_digest = run_cli._optional_repository_policy(repo_dir)
            payload = run_cli.run_payload_for(
                config=self.api.config,
                run_id=run_id,
                repo_dir=repo_dir,
                briefs=briefs,
                base_branch=base_branch.strip(),
                verify_commands=verify_commands,
                max_cost_usd=max_cost_usd,
                max_tokens=max_tokens,
                max_attempts=max_attempts,
                engine=engine,
                provider=coordinates.provider,
                model=coordinates.model,
                base_url=coordinates.base_url,
                provider_key_source=coordinates.provider_key_source,
                worktree=worktree,
                mcp_servers=mcp_servers,
                enable_spawn_agent=enable_spawn_agent,
                enable_agent_teams=enable_agent_teams,
                max_parallel_tasks=max_parallel_tasks,
                worktree_branch=worktree_branch,
                auto_approve_local_only=auto_approve,
                approval_policy_digest=policy_digest,
            )
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a spec/contract refusal is the caller's fault
            translated = translate(exc, run_id=run_id) if getattr(exc, "code", "") else None
            if translated is not None and translated.status != HTTPStatus.BAD_GATEWAY:
                raise translated from exc
            raise RunApiError(
                f"the run specification is invalid: {exc}",
                code="run_api_run_spec_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"run_id": run_id},
            ) from exc
        try:
            started = self.api.backend.start_run(
                payload=payload,
                run_id=run_id,
                run_timeout_s=run_timeout_s,
            )
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a coded refusal is the caller's answer
            raise translate(exc, run_id=run_id) from exc
        # 202, not 201: the run has been ACCEPTED and executes elsewhere. A 201 would claim a
        # resource whose useful state this response does not carry.
        self.api.note_run_started(run_id, principal=self._principal)
        self._send_json(HTTPStatus.ACCEPTED, {"run": started})

    # -- GET /capabilities -----------------------------------------------------

    def _capabilities(self) -> None:
        """What this server can do, before a caller has to discover it by failing.

        The mode is a fact about the SERVER (`--no-temporal` vs a Temporal client), so it belongs
        next to the address, not inside an error a mutation produces. `GET /runs` also carries it (see
        `_list_runs`) because that is the route the IDE already calls on every refresh; this route
        exists for a client that wants the answer WITHOUT listing runs, which is exactly the client
        composing a new-task sheet.
        """
        self._send_json(HTTPStatus.OK, {"capabilities": self.api.capabilities_payload()})

    # -- GET /runs -------------------------------------------------------------

    def _list_runs(self, query: Mapping[str, list[str]]) -> None:
        limit = limit_param(query, default=DEFAULT_LIST_LIMIT)
        root = self.api.config.runs_root
        entries: list[dict[str, Any]] = []
        if root.is_dir():
            for path in root.iterdir():
                ledger_path = path / "ledger.ndjson"
                if not path.is_dir() or not ledger_path.is_file():
                    continue
                try:
                    run_id = validate_run_id(path.name)
                    stat = ledger_path.stat()
                except Exception:  # noqa: BLE001 - a stray or vanished entry is not a run
                    continue
                # ONE bounded tail pass for ended/outcome/stalled (`ledger_verdict`). The three
                # travel together because a row that said "stalled" while carrying an outcome, or
                # "ended" while its ledger was still being written, would be a contradiction the
                # sidebar could not render honestly.
                verdict = ledger_verdict(ledger_path, updated_ts=stat.st_mtime)
                entries.append(
                    {
                        "run_id": run_id,
                        "updated_ts": round(stat.st_mtime, 6),
                        "ledger_bytes": stat.st_size,
                        "last_seq": last_seq_of(ledger_path),
                        # WHICH REPOSITORY this run works in, so a row can be labelled by the fact rather
                        # than by whichever folder the reading window has open. `None` for a ledger
                        # written before the field existed: a caller must present that as unknown, not
                        # substitute its own workspace and call it the run's.
                        "repo_dir": repo_dir_of(ledger_path),
                        "outcome": verdict["outcome"],
                        # `ended` is the fact that stops a spinner, reported apart from `outcome`
                        # because a ledger can prove its run is over while saying nothing about why.
                        "ended": verdict["ended"],
                        # No terminal AND quiet for longer than any activity may run: not finished,
                        # and not being worked on. The third state, so a caller need not claim
                        # "working" for a run nobody is producing.
                        "stalled": verdict["stalled"],
                        "quiet_for_s": verdict["quiet_for_s"],
                    }
                )
        entries.sort(key=lambda item: (-float(item["updated_ts"]), str(item["run_id"])))
        selected = entries[:limit]
        self._send_json(
            HTTPStatus.OK,
            {
                "runs": selected,
                "count": len(selected),
                "total": len(entries),
                "limit": limit,
                # The mode rides on the route the IDE already polls, so a window that can list runs
                # has the capability facts without a second round trip — and a client that never reads
                # /capabilities still cannot honestly claim it could not tell.
                "capabilities": self.api.capabilities_payload(),
            },
        )

    # -- GET /runs/:id ---------------------------------------------------------

    def _get_run(self, raw_run_id: str) -> None:
        run_id = validated_run_id(raw_run_id)
        status = self.api.require_run(run_id)
        ledger = self.api.ledger_summary(run_id)
        # The live query is authoritative while the run is open; the ledger is what survives it.
        live_outcome = (status or {}).get("outcome")
        self._send_json(
            HTTPStatus.OK,
            {
                "run_id": run_id,
                "workflow_id": self.api.config.run_workflow_id(run_id),
                "task_queue": self.api.config.task_queue,
                "status": status,
                "ledger": ledger,
                "outcome": live_outcome if isinstance(live_outcome, str) else ledger.get("outcome"),
            },
        )

    # -- POST /runs/:id/<mutation> --------------------------------------------

    def _mutate(self, raw_run_id: str, action: str) -> None:
        from .contracts import MAX_STEER_NOTE_CHARS

        run_id = validated_run_id(raw_run_id)
        body = self._read_body()
        reject_unknown_fields(body, self._mutation_fields(action), route=f"POST /runs/:id/{action}")
        # WHO is acting comes from the VERIFIED principal, never from the body. A body `identity` is
        # accepted only as a claim that must MATCH it (the IDE sends one), so a client cannot name
        # somebody else and neither can it be silently ignored while the caller believes it applied.
        identity = self._verified_identity(body)
        # Existence BEFORE the mutation: an Update against an unknown run is a 404, not a Temporal
        # failure the caller has to interpret.
        self.api.require_run(run_id)

        if action in TASK_UPDATES:
            self._task_update(run_id, action, body, identity, MAX_STEER_NOTE_CHARS)
            return

        signal_name = SIGNALS[action]
        try:
            self.api.backend.signal(run_id, signal_name)
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the backend's own coded refusal
            raise translate(exc, run_id=run_id) from exc
        seq = self.api.audit_control(run_id, action=action, identity=identity, principal=self._principal)
        self.api.note_control(run_id, action=action, identity=identity)
        self._send_json(
            HTTPStatus.OK,
            {
                "run_id": run_id,
                "signalled": action,
                "signal": signal_name,
                "identity": identity,
                "ledger_seq": seq,
            },
        )

    def _verified_identity(self, body: Mapping[str, Any]) -> str:
        """The acting identity: the body may CONFIRM the authenticated principal, never assert it.

        The field stays REQUIRED (it is the existing contract, and the IDE already sends it), but it
        can no longer *decide* who acted. Three outcomes, and only three:

        * the body names nobody, or names something empty → 400 (the route's own contract);
        * the body names the principal → proceed, stamped with the principal's identity;
        * the body names somebody else → 403. Not a 400: the request is well-formed; the caller is
          claiming an identity it was not authenticated as, which is exactly what a forged approval
          looks like.

        The returned value is ALWAYS ``principal.identity`` — so the ledger, the workflow's
        ``_approvals`` and the SIEM row cannot disagree about who acted.
        """
        claimed = require_identity(body)
        principal = self._principal
        # `_route` refuses any request without a principal, so this is an internal invariant, not a
        # caller-visible branch.
        assert principal is not None, "a mutation was reached without an authenticated principal"
        if claimed != principal.identity:
            raise RunApiError(
                "the identity in the body is not the one this request was authenticated as.",
                code="run_api_identity_mismatch",
                status=HTTPStatus.FORBIDDEN,
                details={
                    "claimed": claimed[:128],
                    "authenticated": principal.identity[:128],
                    "source": principal.source,
                },
                hint="the server stamps the token's identity; the body may only repeat it",
            )
        return principal.identity

    @staticmethod
    def _mutation_fields(action: str) -> frozenset[str]:
        """The exact body a route accepts. Fewer fields is a refusal, more is one too."""
        if action == "steer":
            return frozenset({"identity", "task_id", "note"})
        if action == "budget":
            return frozenset({"identity", "task_id", "max_cost_usd"})
        if action == "approve":
            return frozenset({"identity", "task_id"})
        return frozenset({"identity"})

    def _task_update(
        self,
        run_id: str,
        action: str,
        body: Mapping[str, Any],
        identity: str,
        max_note_chars: int,
    ) -> None:
        task_id = require_task_id(body)
        payload: dict[str, Any] = {"identity": identity}
        if action == "steer":
            note = body.get("note")
            if not isinstance(note, str) or not note.strip():
                raise RunApiError(
                    "steer needs a non-empty note.",
                    code="run_api_note_invalid",
                    status=HTTPStatus.BAD_REQUEST,
                    details={"field": "note"},
                )
            if len(note) > max_note_chars:
                raise RunApiError(
                    f"note must be at most {max_note_chars} characters.",
                    code="run_api_note_invalid",
                    status=HTTPStatus.BAD_REQUEST,
                    details={"length": len(note), "max": max_note_chars},
                )
            payload["note"] = note.strip()
        elif action == "budget":
            payload["max_cost_usd"] = require_positive_number(body, "max_cost_usd")
        try:
            result = self.api.backend.update(run_id, task_id=task_id, action=action, payload=payload)
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a validator's refusal is the answer
            raise translate(exc, run_id=run_id) from exc
        # The identity reached the workflow in `payload` above; the SIEM row records that this host
        # dispatched it, and under which authentication source. Two records of one act on purpose:
        # the workflow owns the decision, the export owns the delivery.
        self.api.note_task_update(run_id, task_id=task_id, action=action, principal=self._principal)
        self._send_json(
            HTTPStatus.OK,
            {
                "run_id": run_id,
                "task_id": task_id,
                "update": action,
                "identity": identity,
                "applied": True,
                "result": result,
            },
        )

    # -- GET /runs/:id/events -------------------------------------------------

    def _stream_events(self, raw_run_id: str, query: Mapping[str, list[str]]) -> None:
        run_id = validated_run_id(raw_run_id)
        after = after_param(query)
        follow = follow_param(query)
        # A run started a second ago has a workflow but no ledger yet: that is a stream waiting for
        # its first record, not a 404. A run id this host has never heard of is a 404.
        if not self.api.config.ledger_path(run_id).is_file():
            self.api.require_run(run_id)
        self.api.acquire_stream()
        try:
            self._send_sse_head()
            self._pump(run_id, after=after, follow=follow)
        finally:
            self.api.release_stream()

    def _send_sse_head(self) -> None:
        self.send_response(HTTPStatus.OK)
        for name, value in SSE_HEADERS:
            self.send_header(name, value)
        self.end_headers()
        self._headers_sent = True
        # No content-length: the body ends when the connection closes, which is what the sidecar's
        # `response.end()` produces. `close_connection` tells the HTTP layer the same thing.
        self.close_connection = True
        self.connection.settimeout(self.api.settings.write_timeout_s)
        self.wfile.flush()

    def _pump(self, run_id: str, *, after: int, follow: bool) -> None:
        """Replay after ``after``, then follow until the terminal, the client leaves, or stop()."""
        tail = LedgerTail(self.api.config.ledger_path(run_id))
        last_index = after
        next_heartbeat = time.monotonic() + self.api.settings.heartbeat_s
        while True:
            if self.api.stopping.is_set():
                return
            try:
                records = tail.poll()
            except LedgerError as exc:
                self._write_stream_error(last_index, exc)
                return
            for record in records:
                if record.seq <= last_index:
                    continue  # already sent: replay and live share one cursor
                frame = event_frame_payload(record)
                self._write_frame(frame)
                last_index = record.seq
                if frame["event_type"] in TERMINAL_EVENT_TYPES:
                    # Exactly one terminal ends the stream, and one already in the backlog ends the
                    # replay — the rule `streamSessionEventsLive` applies to a turn's terminal.
                    return
            if not follow:
                return
            now = time.monotonic()
            if now >= next_heartbeat:
                self._write(SSE_HEARTBEAT_TEMPLATE.format(now_ms=int(time.time() * 1000)))
                next_heartbeat = now + self.api.settings.heartbeat_s
            if self._peer_closed():
                return
            time.sleep(self.api.settings.poll_interval_s)

    def _write_frame(self, frame: Mapping[str, Any]) -> None:
        self._write(
            SSE_EVENT_TEMPLATE.format(
                event_index=int(frame.get("event_index", 0)),
                data=json.dumps(dict(frame), sort_keys=True, separators=(",", ":")),
            )
        )

    def _write_stream_error(self, last_index: int, exc: LedgerError) -> None:
        """One final frame naming a broken ledger, then a close.

        The stream cannot continue — the records after the break are unreachable — but ending with
        silence would look like a clean end. The code is the ledger's own, so the client can tell a
        truncated append from a tampered file.
        """
        self._write(
            SSE_EVENT_TEMPLATE.format(
                event_index=last_index,
                data=json.dumps(
                    {
                        "event_index": last_index,
                        "event_type": "run.stream_error",
                        "error": {
                            "code": exc.code,
                            "message": str(exc),
                            "details": dict(exc.details),
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    def _write(self, text: str) -> None:
        try:
            self.wfile.write(text.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout) as exc:
            raise ClientGone() from exc
        except OSError as exc:
            raise ClientGone() from exc

    def _peer_closed(self) -> bool:
        """Whether the client has closed its end, detected without writing to it.

        ``select`` + ``MSG_PEEK`` sees an orderly close immediately; waiting for a heartbeat write
        to fail would hold the thread (and its stream slot) for a whole heartbeat after the client
        is already gone.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True


