"""The Run API — the loopback HTTP surface the IDE drives a durable run through (D2.5).

The IDE never talks to Temporal. It talks to this server, which owns one long-lived Temporal client
and translates HTTP into the SAME Temporal calls ``codna run``/``codna durable`` already make
(`run_cli._start_run`, `run_cli.query_workflow`, `cli._send_update`, `cli._signal`). Reusing them is
the point: a route cannot drift from the command it mirrors, because it is not a second
implementation of it.

**The surface.**

    POST   /runs                        start a run (202)                        -> RunWorkflow
    GET    /runs                        the runs whose ledgers are on this host
    GET    /runs/:id                    live status + ledger summary             -> query + ledger
    GET    /runs/:id/events?after=N     the run's ledger as an SSE stream         -> the ledger
    POST   /runs/:id/steer              add a steering note   (update)           -> steer
    POST   /runs/:id/approve            approve a parked task (update)           -> approve
    POST   /runs/:id/budget             raise a task's cap    (update)           -> set_budget
    POST   /runs/:id/pause              pause the run         (signal)           -> pause
    POST   /runs/:id/resume             resume the run        (signal)           -> resume
    POST   /runs/:id/cancel             stop the run          (signal)           -> stop

``steer``/``approve``/``budget`` address a TASK workflow (``task-<run>-<task>``) because that is
where the Update is declared, so their body carries ``task_id``. ``pause``/``resume``/``cancel``
address the RUN workflow, whose signals are ``pause``/``resume``/``stop`` — ``cancel`` is the
route's word, ``stop`` the workflow's.

**SSE framing is copied, not invented.** `agent-core/vendor/cline/algenta/server/http.ts`
(``streamSessionEventsLive``) is what the IDE's `streamClient.ts` parser was proven against, so the
headers, the frame template and the heartbeat are transcribed from it into `run_api_support.py` and
asserted equal to that file's literals by the tests. The stream:

* replays the ledger after ``?after=N`` (``N`` = the last ``seq`` the caller has), then follows;
* ends at exactly ONE terminal — ``run.finished`` — whether that arrives live or in the replay;
* sends ``: heartbeat <ms>`` comments while idle;
* treats a client disconnect as a detach, never as a failure (the run keeps running).

**Security** is in `run_api_support.py` (loopback-only settings, the 0600 token, the constant-time
compare) and enforced here on every route: auth BEFORE any parsing, an identity on every mutation,
and every external input validated. An unknown run is 404, malformed JSON is 400, an oversized body
is 413 *without being buffered*, and an unknown field is 400 rather than silently ignored.

**Threads and shutdown — the part `ThreadingHTTPServer` gets wrong by default.** One thread per
request; a streaming request holds its thread for the life of the stream. Four explicit bounds:

1. ``request_timeout_s`` on the socket, so a slow-loris body cannot pin a thread forever;
2. ``max_streams`` — past it a stream is a 503 rather than unbounded thread growth;
3. streaming writes use ``write_timeout_s`` and the loop polls the socket for EOF, so a client that
   stops reading or disconnects releases its thread instead of blocking on a dead socket;
4. ``stopping`` is checked every poll, so ``stop()`` (another thread) drains streams; the server's
   threads are daemons, so a stream that ignored all of the above cannot hold the process open.
"""
from __future__ import annotations

import logging
import signal
import threading
from http import HTTPStatus
from typing import Any, Final, Mapping

from .config import DurableConfig
from .identity import (
    AUTH_MODE_LOCAL_TRUST,
    AUTH_MODE_TOKEN,
    AuthenticationError,
    IdentityConfig,
    IdentityError,
    IdentityResolver,
    ProviderUnavailable,
)
from .ledger import LedgerError, RunLedger
from .roles import AuthorizationError, Principal
from .run_api_backend import RunBackend, TemporalRunBackend, translate
from .run_api_http import RunApiHandler, ThreadingRunApiServer
from .run_api_support import (
    LedgerTail,
    RunApiError,
    RunApiSettings,
    constant_time_token_equals,
    load_or_create_token,
    parse_bearer,
    terminal_verdict,
)
from .siem import SiemEvent, SiemSink, SiemDeliveryError, build_siem_sink

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MUTATION_ROUTES",
    "RunApi",
    "RunApiError",
    "RunApiServer",
    "RunApiSettings",
    "RunBackend",
    "TemporalRunBackend",
    "constant_time_token_equals",
    "load_or_create_token",
    "parse_bearer",
]

logger = logging.getLogger("codna.durable.run_api")

# Re-exported from the support module so a caller importing "the run API's defaults" gets them here.
DEFAULT_HOST: Final = RunApiSettings.host
DEFAULT_PORT: Final = RunApiSettings.port
MUTATION_ROUTES: Final = frozenset({"steer", "approve", "pause", "resume", "budget", "cancel"})




# -- the routes --------------------------------------------------------------------------------

# The capability field that reports `max_streams` — what `--max-streams` set. One spelling: the composer below, the
# banner `serve` prints and the client's parser all read the same word, so the document cannot disagree with itself.
MAX_STREAMS_FIELD: Final = "max_streams"


def capabilities_payload_for(settings: RunApiSettings, capabilities: Any) -> dict[str, Any]:
    """The capability document: the backend's route/mode facts, plus this PROCESS's stream ceiling.

    THE ONE COMPOSITION. Three callers send it — `GET /capabilities`, `GET /runs`, and the banner `serve`
    prints — and all three go through here, so a surface and an operator cannot be told different things
    about the same process. The ceiling comes from ``settings`` (what `--max-streams` set), which is the
    SAME value `RunApi._streams` was built from: advertise anything else and the client budgets slots the
    server will refuse.

    Free rather than a method so the composition is testable without constructing a backend or a socket.
    """
    payload = capabilities.as_payload()
    payload[MAX_STREAMS_FIELD] = settings.max_streams
    return payload


class RunApi:
    """The routes, over a backend. Transport-agnostic: a test can call it without a socket."""

    def __init__(
        self,
        *,
        config: DurableConfig,
        backend: RunBackend,
        settings: RunApiSettings,
        identity: IdentityResolver | None = None,
        token: str | None = None,
        siem: SiemSink | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.settings = settings
        self.stopping = threading.Event()
        self._streams = threading.BoundedSemaphore(settings.max_streams)
        # The identity layer is REQUIRED, but defaults to the local bearer-token mode with the token
        # `RunApiServer` minted. Being explicit about that default is what keeps every existing
        # caller working while making "no identity resolver" impossible: there is no code path
        # where `authenticate` has nothing to ask.
        if identity is None:
            resolved_token = token if token is not None else load_or_create_token(settings.token_path)
            identity = IdentityResolver(
                IdentityConfig(mode=AUTH_MODE_TOKEN, local_token=resolved_token)
            )
        self.identity = identity
        # The token is no longer the boundary — the resolver is — so nothing here keeps a second
        # copy of it to compare against. `token` remains a constructor argument only so a caller
        # that already has one does not read the file twice.
        self.siem = siem if siem is not None else build_siem_sink()

    # -- auth ------------------------------------------------------------------

    def authenticate(self, header: str | None) -> Principal:
        """The verified `Principal` for this request. Raises `RunApiError` (401/503) on failure.

        Every cause of "not authenticated" becomes ONE 401 body, because distinguishing "no token"
        from "a token that is close" is a free oracle. A provider outage is the one exception: it is
        this server's failure, so it is 503 and says so — sending an operator to hunt a bad token
        while the identity provider is down is the more expensive mistake.
        """
        try:
            principal = self.identity.resolve(header)
        except AuthenticationError as exc:
            raise RunApiError(
                "this endpoint requires a valid bearer token.",
                code="run_api_unauthorized",
                status=HTTPStatus.UNAUTHORIZED,
                details={"header": "authorization", "mode": self.identity.mode},
                hint=(
                    f"read the local token from {self.settings.token_path} (mode 0600) on this host"
                    if self.identity.mode in (AUTH_MODE_TOKEN, AUTH_MODE_LOCAL_TRUST)
                    else "authenticate with the token your identity provider issued (`codna login`)"
                ),
            ) from exc
        except ProviderUnavailable as exc:
            raise RunApiError(
                str(exc),
                code=exc.code,
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                details=dict(exc.details),
                hint="the identity provider is unreachable; this is this server's problem, not yours",
            ) from exc
        except IdentityError as exc:
            # A misconfiguration is a 500: it must be loud, and it must not read as "your token is
            # bad" to a caller who did nothing wrong.
            raise RunApiError(
                str(exc),
                code=exc.code,
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                details=dict(exc.details),
            ) from exc
        self.record_siem(
            SiemEvent.action("auth.succeeded", principal=principal, route="")
        )
        return principal

    def require_role(self, principal: Principal, key: str) -> None:
        """Refuse (403) unless ``principal`` holds the role route ``key`` requires. Fails CLOSED."""
        try:
            principal.require(key)
        except AuthorizationError as exc:
            self.record_siem(
                SiemEvent.action(
                    "auth.denied",
                    principal=principal,
                    route=key,
                    outcome="denied",
                    detail={"required": exc.required},
                )
            )
            raise RunApiError(
                f"this caller may not use {key}: it requires the {exc.required!r} role.",
                code="run_api_forbidden",
                status=HTTPStatus.FORBIDDEN,
                details={
                    "route": key,
                    "required_role": exc.required,
                    "held_roles": sorted(exc.held),
                },
                hint="ask a host operator for the role this route needs",
            ) from exc

    def authorize(self, header: str | None) -> None:
        """Backward-compatible authentication-only entry point. Kept for the tests and embedders that
        only want "is this caller known?"; the transport never uses it (it must also check the
        route's role, which `require_role` does with a route key)."""
        self.authenticate(header)

    def __repr__(self) -> str:  # pragma: no cover - formatting only, and it must never print the token
        return f"RunApi(config={self.config.address!r}, streams={self.settings.max_streams})"

    # -- what this server can do ----------------------------------------------

    def capabilities_payload(self) -> dict[str, Any]:
        """The capability document every route that reports one sends.

        ONE COMPOSER, for the reason this whole seam exists. The document is two facts from two owners:
        the backend knows which routes it can serve, and THIS object owns the stream semaphore, so
        ``max_streams`` is read from the same ``settings`` the semaphore was built from
        (``self._streams = BoundedSemaphore(settings.max_streams)``). A client is therefore told the
        exact number it will be refused at, not a second copy of it.

        Why it is here and not in ``RunCapabilities``: a capability is a property of the backend's
        CONNECTION (a ledger-only server and a durable one differ), while the stream ceiling is a
        property of the PROCESS — ``--max-streams``, which the backend is never told. Putting it on the
        backend would mean passing the settings down to a type that has no semaphore, which is how the
        advertised number and the enforced one drift apart in the first place.

        Delegates to ``capabilities_payload_for`` so a test can assert the composition without a socket.
        """
        return capabilities_payload_for(self.settings, self.backend.capabilities())

    # -- run existence ---------------------------------------------------------

    def require_run(self, run_id: str) -> dict[str, Any] | None:
        """404 unless this host knows the run; returns its live status when Temporal can answer.

        Two sources, because they cover different lifetimes: a finished run's workflow may be gone
        from retention while its ledger is still the audit record, and a run started a second ago
        has a workflow but no ledger until its first activity appends.
        """
        try:
            status = self.backend.status(run_id)
        except RunApiError:
            raise
        except BaseException as exc:  # noqa: BLE001 - a coded refusal is the answer
            raise translate(exc, run_id=run_id) from exc
        if status is None and not self.config.ledger_path(run_id).is_file():
            raise RunApiError(
                f"run {run_id} does not exist on this host.",
                code="run_not_found",
                status=HTTPStatus.NOT_FOUND,
                details={"run_id": run_id, "workflow_id": self.config.run_workflow_id(run_id)},
                hint="GET /runs lists the runs whose ledgers are on disk",
            )
        return status

    def ledger_summary(self, run_id: str) -> dict[str, Any]:
        """A bounded summary of the run's ledger: one read for counts, a tail read for the outcome."""
        from .run_api_support import ledger_problems
        from .run_repo import repo_dir_of

        path = self.config.ledger_path(run_id)
        if not path.is_file():
            return {
                "path": str(path),
                "exists": False,
                "records": 0,
                "last_seq": 0,
                "valid": True,
                "problems": [],
                "terminal": False,
                "outcome": None,
                # WHICH CHECKOUT this run works in — the same fact `GET /runs` carries per row. `None` for a
                # ledger written before the field existed: a caller must present that as unknown rather than
                # substitute whichever folder its window happens to have open (the bug #719 fixed on the list
                # route, which stays fixed here for the detail route that feeds the sidebar and the cursors).
                "repo_dir": None,
            }
        tail = LedgerTail(path)
        try:
            records = tail.poll()
            problems = ledger_problems(records)
        except LedgerError as exc:
            records = []
            problems = [f"{exc.code}: {exc}"]
        ended, outcome = terminal_verdict(path)
        return {
            "path": str(path),
            "exists": True,
            "records": len(records),
            "last_seq": records[-1].seq if records else 0,
            "valid": not problems,
            "problems": problems[:10],
            # `terminal` is the fact that stops a spinner: a run-level `run.finished`, OR a turn
            # terminal the run never wrote past. `outcome` is only reported when the ledger carries
            # (or implies) one. `terminal_verdict` reads BOTH from ONE tail pass: two separate reads
            # could observe two file states if a record landed between them, and a response that says
            # `terminal: false` beside `outcome: "completed"` contradicts itself.
            "terminal": ended,
            "outcome": outcome,
            # The run's OWN repository, from the same bounded head scan the list route uses. One reader,
            # so a run cannot be labelled from its list row and re-labelled from its detail.
            "repo_dir": repo_dir_of(path),
        }

    def audit_control(self, run_id: str, *, action: str, identity: str, principal: Principal | None = None) -> int:
        """Record WHO asked for a run-level signal, as a `note` ledger record.

        A Temporal signal carries no payload, so an identity sent with one would be lost the moment
        it left this process. The ledger is the append-only, hash-chained audit trail the plan names,
        which makes this a record an auditor can check — not a log line that rotates away.
        """
        try:
            record = RunLedger.for_run(self.config.paths, run_id).append(
                "note",
                payload={"event": "control.requested", "action": action, "identity": identity},
            )
        except LedgerError as exc:
            raise RunApiError(
                f"the {action} was applied but could not be recorded: {exc}",
                code="run_api_audit_failed",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                details={"run_id": run_id, "action": action, "cause": str(exc)},
            ) from exc
        if principal is not None:
            self.note_control(run_id, action=action, identity=identity, principal=principal)
        return record.seq if record is not None else 0

    # -- SIEM export ----------------------------------------------------------
    #
    # One helper per kind of act rather than one generic `note(event=...)`: the closed event set in
    # `siem.EVENT_NAMES` is only useful if the call sites cannot invent a name, and a named helper is
    # where "a start is a start" is decided.

    def note_run_started(self, run_id: str, *, principal: Principal | None) -> None:
        self._note_siem("run.started", principal=principal, run_id=run_id, route="runs.start")

    def note_control(
        self,
        run_id: str,
        *,
        action: str,
        identity: str,
        principal: Principal | None = None,
    ) -> None:
        self._note_siem(
            "run.controlled",
            principal=principal,
            run_id=run_id,
            route=f"runs.{action}",
            detail={"action": action, "identity": identity},
        )

    def note_task_update(
        self,
        run_id: str,
        *,
        task_id: str,
        action: str,
        principal: Principal | None,
    ) -> None:
        event = {
            "approve": "run.approved",
            "steer": "run.steered",
            "budget": "run.budget_raised",
        }.get(action, "run.steered")
        self._note_siem(
            event, principal=principal, run_id=run_id, task_id=task_id, route=f"runs.{action}"
        )

    def _note_siem(
        self,
        event: str,
        *,
        principal: Principal | None,
        run_id: str,
        route: str,
        task_id: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if principal is None:  # pragma: no cover - every route holds an authenticated principal
            return
        self.record_siem(
            SiemEvent.action(
                event,
                principal=principal,
                route=route,
                run_id=run_id,
                task_id=task_id,
                detail=detail,
            )
        )

    def record_siem(self, event: SiemEvent) -> None:
        """Deliver one SIEM event, or fail LOUDLY.

        A sink that cannot write is a 500, not a warning: an audit export that silently stops is the
        exact failure this module exists to prevent, and a caller who is not told has no reason to
        look. The act may already have been applied at this point — which is why the message says so
        and the code lets a client distinguish it from "the request was rejected".
        """
        try:
            self.siem.record(event)
        except SiemDeliveryError as exc:
            raise RunApiError(
                f"the action was applied but its SIEM record could not be written: {exc}",
                code=exc.code,
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                details=dict(exc.details),
                hint="check the SIEM export directory is writable; the ledger still holds the record",
            ) from exc

    # -- stream slots ----------------------------------------------------------

    def acquire_stream(self) -> None:
        if not self._streams.acquire(blocking=False):
            raise RunApiError(
                f"this server is already streaming to {self.settings.max_streams} clients.",
                code="run_api_stream_limit",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                details={"max_streams": self.settings.max_streams},
                hint="a slot is released when its client disconnects or its run finishes",
            )

    def release_stream(self) -> None:
        self._streams.release()


# -- the server --------------------------------------------------------------------------------


class RunApiServer:
    """The listener: owns the HTTP socket, the token, and the backend's lifetime."""

    def __init__(
        self,
        *,
        config: DurableConfig,
        backend: RunBackend,
        settings: RunApiSettings,
        identity: IdentityResolver | None = None,
        siem: SiemSink | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.settings = settings
        # The token is minted even in OIDC mode: `--no-temporal` viewers and any local tooling still
        # use it, and a mode switch must not require deleting a file. It is not an authentication
        # input in OIDC/local-trust mode — the resolver is.
        self.token = load_or_create_token(settings.token_path)
        self.identity = identity or IdentityResolver(
            IdentityConfig(mode=AUTH_MODE_TOKEN, local_token=self.token)
        )
        self.siem = siem if siem is not None else build_siem_sink()
        self.api = RunApi(
            config=config,
            backend=backend,
            settings=settings,
            identity=self.identity,
            token=self.token,
            siem=self.siem,
        )
        self._httpd: ThreadingRunApiServer | None = None
        self._thread: threading.Thread | None = None

    def bind(self) -> "RunApiServer":
        """Bind the socket without serving. Raises rather than retrying a port somebody owns."""
        if self._httpd is not None:
            return self
        handler = type(
            "_BoundRunApiHandler",
            (RunApiHandler,),
            {"api": self.api, "timeout": self.settings.request_timeout_s},
        )
        try:
            self._httpd = ThreadingRunApiServer((self.settings.host, self.settings.port), handler)
        except OSError as exc:
            raise RunApiError(
                f"could not bind {self.settings.host}:{self.settings.port}: {exc}",
                code="run_api_bind_failed",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                details={"host": self.settings.host, "port": self.settings.port},
                hint="another `codna durable serve` may already own the port",
            ) from exc
        self._httpd.timeout = self.settings.request_timeout_s
        return self

    def start(self) -> "RunApiServer":
        """Bind and serve on a background thread. For tests and embedding, not for the CLI."""
        self.bind()
        assert self._httpd is not None  # narrowed by `bind`
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="codna-run-api",
            daemon=True,
        )
        self._thread.start()
        return self

    def serve_forever(self, *, stop_on_signals: bool = True) -> None:
        """Serve in the CALLING thread until interrupted. This is the CLI's foreground loop.

        ``serve_forever`` must run in the thread that owns the signal handlers (Python only delivers
        signals to the main thread), and ``shutdown()`` must NOT be called from the thread running
        it — so a signal sets ``stopping`` (which drains the streams) and hands ``shutdown`` to a
        helper thread. Doing the obvious thing instead deadlocks the shutdown.
        """
        self.bind()
        assert self._httpd is not None  # narrowed by `bind`
        httpd = self._httpd
        previous: dict[int, Any] = {}
        if stop_on_signals:
            previous = self._install_signal_handlers(httpd)
        try:
            httpd.serve_forever(poll_interval=0.1)
        finally:
            for signal_number, handler in previous.items():
                signal.signal(signal_number, handler)

    def _install_signal_handlers(self, httpd: ThreadingRunApiServer) -> dict[int, Any]:
        """SIGINT/SIGTERM -> drain the streams, then stop the server. Returns the old handlers."""
        previous: dict[int, Any] = {}

        def handle(signal_number: int, _frame: Any) -> None:
            self.api.stopping.set()
            threading.Thread(
                target=httpd.shutdown, name="codna-run-api-shutdown", daemon=True
            ).start()

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, handle)
            except (ValueError, OSError):  # pragma: no cover - a non-main thread, or Windows
                previous.pop(signal_number, None)
        return previous

    @property
    def port(self) -> int:
        if self._httpd is None:  # pragma: no cover - before start(), or after stop()
            return self.settings.port
        return int(self._httpd.server_address[1])

    @property
    def host(self) -> str:
        return self.settings.host

    @property
    def address(self) -> tuple[str, int]:
        return (self.settings.host, self.port)

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.settings.host == "localhost" else self.settings.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def wait(self, *, timeout_s: float | None = None) -> None:
        """Block until the serving thread ends (the CLI's foreground loop)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def stop(self) -> None:
        """Stop serving and release the backend. Safe to call twice, from any thread.

        ``stopping`` is set BEFORE ``shutdown()`` so a streaming handler notices on its next poll
        and returns, rather than writing to a socket this call is about to close.

        ``shutdown()`` blocks until ``serve_forever`` returns, so it is skipped when the calling
        thread IS the serving thread — that is the "stop from a signal handler" case, where the
        signal path has already asked a helper thread to shut down.
        """
        self.api.stopping.set()
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            if threading.current_thread() is not self._thread:
                httpd.shutdown()
            httpd.server_close()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise on the way out
            logger.warning("the run API backend did not close cleanly", exc_info=True)

    def __enter__(self) -> "RunApiServer":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
