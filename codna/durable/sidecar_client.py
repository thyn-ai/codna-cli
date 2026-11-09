"""HTTP client for the sidecar's ``/v1`` API — the durable worker's only way to run a turn.

The IDE already has this client in TypeScript (`codna-agent-backend/src/streamClient.ts`, over
`server/http.ts` + `run-server.ts`). The CLI had none: `cline_agent.py` shells out to bun and reads
stdout. A durable activity cannot do that — it needs the same three guarantees the IDE relies on,
so this module re-implements exactly those against the same wire, and no more:

* **`turn_key` idempotency.** ``POST /v1/sessions/:id/turns`` takes the key in the
  ``x-codna-turn-key`` header (or ``?turn_key=``). A replay answers ``200`` with
  ``{"turn_id": …, "replayed": true}``; a fresh turn answers ``202``. This is what makes a
  re-executed activity cheap instead of a second paid turn, so the client surfaces the distinction
  (`TurnStart.replayed`) rather than flattening both to "started".
* **SSE replay/resume via ``?after=``.** ``GET /v1/sessions/:id/events?turn_id=…&after=n`` replays
  the backlog after ``n`` and then follows live events until one terminal. A dropped connection
  is not a failed turn — the engine keeps running — so the client reconnects from the highest
  ``event_index`` it has seen. That index is exposed to the caller on every event so it can be
  persisted in `TaskState` and survive a worker restart.
* **Abort vs. unsubscribe.** Closing the stream is an unsubscribe. Cancellation is an explicit
  ``POST /sessions/:id/abort``, and a failed abort must raise: silently swallowing it means the
  user pressed Stop and nothing stopped.

Wire details taken from the server, not guessed: routes and shapes from
``agent-core/vendor/cline/algenta/run-server.ts``, SSE framing (``id:``/``event:``/``data:``, with
CR, LF or CRLF line endings and ``:`` heartbeat comments) from ``agent-core/.../server/http.ts`` and
the TS parser. Frames are separated by a blank line in any of the three splittings; an LF-only
parser silently never yields a frame from a CRLF stream.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

import httpx

__all__ = [
    "CAPABILITIES_PATH",
    "SESSIONS_PATH",
    "SSE_TERMINAL_TURN_EVENTS",
    "SidecarClient",
    "SidecarClientError",
    "SidecarError",
    "SidecarLimits",
    "SidecarSession",
    "SseEvent",
    "TurnHandle",
    "TurnStart",
    "parse_sse_frames",
]

CAPABILITIES_PATH: Final = "/v1/capabilities"
SESSIONS_PATH: Final = "/v1/sessions"
# Legacy collection route. `/v1/sessions` is the canonical one and both go through the same
# handler; kept as a constant so the fallback is explicit rather than a literal in a function.
LEGACY_SESSIONS_PATH: Final = "/sessions"


class _OmitWriteScope:
    """Sentinel for ``start_turn(allowed_write_paths=...)``: "this turn names no scope".

    A sentinel rather than ``None`` because ``None`` is a MEANINGFUL wire value here (JSON ``null`` =
    "no additional restriction"), and because the three states must not be confused:

    * omitted (this sentinel) — inherit the session's scope;
    * ``None`` — explicitly unrestricted for this turn, overriding the session's;
    * ``[]`` — this turn writes nothing.

    The middle one exists because a durable run creates its session READ-ONLY, so its writable
    ``implement`` turn has to lift that scope explicitly rather than inherit it.

    ``repr`` is stable so a failed assertion prints something a reader can act on.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "OMIT_WRITE_SCOPE"


OMIT_WRITE_SCOPE: Final = _OmitWriteScope()

# Terminal turn events, from `server/http.ts::_SSE_TERMINAL`. The stream is expected to end on one
# of these; seeing one is what distinguishes "turn finished" from "connection dropped".
SSE_TERMINAL_TURN_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "session.turn_completed",
        "session.turn_failed",
        "session.turn_aborted",
        "session.aborted",
        "session.stopped",
    }
)

# SSE framing: a frame ends at a blank line; a line ends with CR, LF or CRLF.
_FRAME_DELIM = re.compile(r"\r\n\r\n|\n\n|\r\r")
_LINE_DELIM = re.compile(r"\r\n|\n|\r")

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SidecarClientError(RuntimeError):
    """A transport/contract failure talking to the sidecar. Always carries a code."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "status": self.status,
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class SidecarError:
    """The sidecar's own structured error body (``{"error": {code, message, details}}``)."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _decode_error(payload: Any, status: int) -> SidecarError:
    """Decode the sidecar error shape, tolerating a non-JSON body.

    Every non-2xx from the sidecar is *supposed* to be ``{"error": {code, message, details}}``, but
    a proxy 502 is HTML. Falling back to a status-derived error keeps the failure diagnosable
    instead of raising on the error path.
    """
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return SidecarError(
                code=str(error.get("code") or f"http_{status}"),
                message=str(error.get("message") or ""),
                details=dict(error.get("details") or {}),
            )
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return SidecarError(code=f"http_{status}", message=detail)
    return SidecarError(code=f"http_{status}", message=f"sidecar returned HTTP {status}")


@dataclass(frozen=True)
class SidecarCapabilities:
    """The negotiated feature set (``GET /v1/capabilities``)."""

    protocol_version: int
    events_schema: str
    transports: tuple[str, ...]
    features: dict[str, bool]

    @property
    def supports_turn_key(self) -> bool:
        return bool(self.features.get("turn_key"))

    @property
    def supports_resume_from(self) -> bool:
        return bool(self.features.get("resume_from"))

    def require(self, feature: str) -> None:
        """Fail fast when the engine cannot do something the durable path depends on."""
        if not self.features.get(feature):
            raise SidecarClientError(
                f"the sidecar does not support {feature!r}; durable turns require it.",
                code="sidecar_capability_missing",
                details={"feature": feature, "features": self.features},
            )


@dataclass(frozen=True)
class SidecarSession:
    """A created/resumed engine session."""

    session_id: str
    resumed: bool
    resume_from_ignored: bool
    owner: str | None = None
    state: str | None = None


@dataclass(frozen=True)
class SidecarLimits:
    """Caps in force on a session, and whether a previous breach was lifted.

    The PATCH contract is ``null`` = clear, absent = leave alone. This type only models the patch a
    durable run actually sends (raise a cap at a task boundary), and refuses to represent "send
    everything" so a caller cannot accidentally clear limits it did not mean to touch.
    """

    max_cost_usd: float | None = None
    max_turn_cost_usd: float | None = None
    max_tokens: int | None = None

    def to_patch(self) -> dict[str, object]:
        patch: dict[str, object] = {}
        if self.max_cost_usd is not None:
            patch["maxCostUsd"] = self.max_cost_usd
        if self.max_turn_cost_usd is not None:
            patch["maxTurnCostUsd"] = self.max_turn_cost_usd
        if self.max_tokens is not None:
            patch["maxTokens"] = self.max_tokens
        return patch


@dataclass(frozen=True)
class TurnStart:
    """The outcome of starting a turn.

    ``replayed`` is the whole point of `turn_key`: True means the engine recognized the key and did
    NOT start a second paid turn.
    """

    session_id: str
    turn_id: str
    replayed: bool
    events_url: str | None = None


@dataclass(frozen=True)
class SseEvent:
    """One decoded SSE frame from the events stream.

    ``event_index`` is the sidecar's monotonic per-session sequence — the value to persist and to
    pass back as ``?after=``. ``None`` means the frame carried no index (a lifecycle event outside
    a turn), which must not be treated as "index 0".
    """

    event_id: str | None
    event_name: str | None
    event_type: str
    event_index: int | None
    session_id: str | None
    turn_id: str | None
    payload: dict[str, Any]

    @property
    def is_terminal(self) -> bool:
        return self.event_type in SSE_TERMINAL_TURN_EVENTS


@dataclass
class TurnHandle:
    """A started turn plus its abort path."""

    session_id: str
    turn_id: str
    replayed: bool
    _abort: Callable[[], None] | None = None

    def abort(self) -> None:
        if self._abort is None:  # pragma: no cover - always set by the client
            raise SidecarClientError(
                "this TurnHandle cannot abort (constructed without a client).",
                code="abort_unavailable",
            )
        self._abort()


def parse_sse_frames(buffer: str) -> tuple[list[SseEvent], str]:
    """Parse complete SSE frames out of ``buffer``; return the events and the unconsumed tail.

    Mirrors the TS parser's rules: ``:`` lines are heartbeats, ``id:``/``event:`` are metadata,
    every ``data:`` line of a frame is joined with newlines and JSON-decoded, and a frame whose data
    is not a JSON object is dropped (rather than tearing down the stream — one malformed frame must
    not lose the rest of a paid turn). The remainder is returned so a chunk boundary in the middle
    of a frame cannot lose data.
    """
    events: list[SseEvent] = []
    while True:
        match = _FRAME_DELIM.search(buffer)
        if match is None:
            return events, buffer
        frame = buffer[: match.start()]
        buffer = buffer[match.end() :]
        parsed = _parse_frame(frame)
        if parsed is not None:
            events.append(parsed)


def _parse_frame(frame: str) -> SseEvent | None:
    data_lines: list[str] = []
    event_id: str | None = None
    event_name: str | None = None
    for line in _LINE_DELIM.split(frame):
        if not line or line.startswith(":"):
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip() or None
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:][1:] if line[5:6] == " " else line[5:])
    if not data_lines:
        return None
    try:
        decoded = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    event_type = decoded.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        return None
    raw_index = decoded.get("event_index")
    index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else None
    payload = decoded.get("payload")
    return SseEvent(
        event_id=event_id or _optional_str(decoded.get("event_id")),
        event_name=event_name,
        event_type=event_type,
        event_index=index,
        session_id=_optional_str(decoded.get("session_id")),
        turn_id=_optional_str(decoded.get("turn_id")),
        payload=dict(payload) if isinstance(payload, Mapping) else {},
    )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class SidecarClient:
    """Synchronous httpx client for one sidecar instance.

    Synchronous on purpose: Temporal's synchronous activities are simpler to bound and to cancel
    than an async loop, and a durable turn is I/O-bound on exactly one connection. ``stream_turn``
    is a generator so the activity can heartbeat, append to the ledger and check for cancellation
    between events rather than after the whole turn.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str | None = None,
        owner: str | None = None,
        timeout_s: float = 30.0,
        stream_timeout_s: float = 600.0,
        client: httpx.Client | None = None,
        reconnect_attempts: int = 5,
        backoff_s: float = 0.5,
    ) -> None:
        # Trim trailing slashes without a backtracking regex (ReDoS on adversarial input).
        url = base_url or ""
        while url.endswith("/"):
            url = url[:-1]
        if not url:
            raise SidecarClientError("sidecar base_url must not be empty.", code="sidecar_url_invalid")
        self._base_url = url
        self._auth_token = auth_token
        self._owner = owner
        self._timeout_s = timeout_s
        self._stream_timeout_s = stream_timeout_s
        self._reconnect_attempts = max(0, reconnect_attempts)
        self._backoff_s = max(0.0, backoff_s)
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._owns_client = client is None
        self._capabilities: SidecarCapabilities | None = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SidecarClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = dict(extra or {})
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
        # The sidecar scopes sessions by owner (`x-session-owner`), and a durable run must be its own
        # owner so it can never re-attach to an interactive chat's session.
        if self._owner:
            headers["x-session-owner"] = self._owner
        return headers

    def _raise_for(self, response: httpx.Response, *, what: str) -> None:
        if 200 <= response.status_code < 300:
            return
        try:
            payload: Any = response.json()
        except ValueError:
            payload = None
        error = _decode_error(payload, response.status_code)
        raise SidecarClientError(
            f"{what} failed: {response.status_code} {error.code}: {error.message}".strip(),
            code=error.code,
            status=response.status_code,
            details=error.details,
        )

    # -- capabilities -------------------------------------------------------------

    def capabilities(self, *, refresh: bool = False) -> SidecarCapabilities:
        """Fetch (and cache) the engine's negotiated capabilities."""
        with self._lock:
            if self._capabilities is not None and not refresh:
                return self._capabilities
        response = self._client.get(f"{self._base_url}{CAPABILITIES_PATH}", headers=self._headers())
        self._raise_for(response, what="capabilities")
        payload = response.json()
        features = payload.get("features")
        capabilities = SidecarCapabilities(
            protocol_version=int(payload.get("protocol_version") or 0),
            events_schema=str(payload.get("events_schema") or ""),
            transports=tuple(payload.get("transports") or ()),
            features={str(key): bool(value) for key, value in (features or {}).items()},
        )
        with self._lock:
            self._capabilities = capabilities
        return capabilities

    def negotiate(self, *, require: Sequence[str] = ()) -> SidecarCapabilities:
        """Negotiate before a turn: the durable path needs `turn_key`, so refuse an engine that
        cannot dedupe rather than running turns that a retry would double-charge."""
        capabilities = self.capabilities()
        for feature in require:
            capabilities.require(feature)
        return capabilities

    # -- sessions -----------------------------------------------------------------

    def create_session(
        self,
        *,
        working_dir: str,
        prompt: str = "",
        resume_from: str | None = None,
        spec: Mapping[str, Any] | None = None,
    ) -> SidecarSession:
        """Create (or resume) a session.

        ``working_dir`` is required by the engine's own validator, so it is required here rather
        than defaulted — a session pointing at the wrong directory would edit the wrong tree.
        """
        body: dict[str, Any] = dict(spec or {})
        body["working_dir"] = working_dir
        if prompt:
            body["prompt"] = prompt
        if resume_from:
            body["resume_from"] = resume_from
        response = self._client.post(
            f"{self._base_url}{SESSIONS_PATH}",
            headers=self._headers({"content-type": "application/json"}),
            content=json.dumps(body),
        )
        self._raise_for(response, what="create_session")
        payload = response.json()
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise SidecarClientError(
                "create_session returned no usable session_id.",
                code="sidecar_response_invalid",
                status=response.status_code,
                details={"session_id": repr(session_id)},
            )
        return SidecarSession(
            session_id=session_id,
            resumed=payload.get("resumed") is True,
            resume_from_ignored=payload.get("resume_from_ignored") is True,
            owner=_optional_str(payload.get("owner")),
            state=_optional_str(payload.get("state")),
        )

    # -- turns --------------------------------------------------------------------

    def start_turn(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_key: str | None = None,
        allowed_write_paths: Sequence[str] | None | _OmitWriteScope = OMIT_WRITE_SCOPE,
    ) -> TurnStart:
        """Start a turn, or replay one.

        ``202`` = a turn started; ``200`` with ``replayed: true`` = the key was already used and
        nothing was started. Any other status is an error carrying the sidecar's own code (``429``
        admission and ``409`` start failures are both expected in production).

        ``allowed_write_paths`` is the TURN's write scope, and all three wire states are expressible
        because they mean three different things:

        * :data:`OMIT_WRITE_SCOPE` (the default) — the field is not sent; the session's own scope
          stands. This is what a one-shot caller means.
        * ``None`` — sent as JSON ``null``: NO additional restriction for this turn, explicitly
          overriding the session's. Needed because a durable session is CREATED read-only, so its
          writable ``implement`` turn cannot inherit it.
        * a sequence — sent as an array; ``[]`` means this turn writes nothing.

        A durable task's plan and implement turns share one session, so the scope cannot live only on
        the create request — and ``None``/absent/``[]`` must not collapse into each other.
        """
        _validate_session_id(session_id)
        headers = self._headers({"content-type": "application/json"})
        if turn_key:
            headers["x-codna-turn-key"] = turn_key
        body: dict[str, Any] = {"prompt": prompt}
        if not isinstance(allowed_write_paths, _OmitWriteScope):
            body["allowed_write_paths"] = (
                None if allowed_write_paths is None else list(allowed_write_paths)
            )
        response = self._client.post(
            f"{self._base_url}{SESSIONS_PATH}/{session_id}/turns",
            headers=headers,
            content=json.dumps(body),
        )
        self._raise_for(response, what="start_turn")
        if response.status_code not in (200, 202):
            raise SidecarClientError(
                f"start_turn expected 200 or 202, got {response.status_code}.",
                code="sidecar_response_invalid",
                status=response.status_code,
            )
        payload = response.json()
        turn_id = payload.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            raise SidecarClientError(
                "start_turn returned no turn_id.",
                code="sidecar_response_invalid",
                status=response.status_code,
            )
        replayed = response.status_code == 200 or payload.get("replayed") is True
        return TurnStart(
            session_id=session_id,
            turn_id=turn_id,
            replayed=replayed,
            events_url=_optional_str(payload.get("events_url")),
        )

    def stream_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        after: int = 0,
        should_reconnect: Callable[[int], bool] | None = None,
    ) -> Iterator[SseEvent]:
        """Yield a turn's events from ``after``, reconnecting on a drop until a terminal.

        The generator is deliberately thin: it yields decoded events and stops after the first
        terminal, which is what `streamClient.ts` does (and why a trailing session-level terminal on
        the same connection can never produce a second one). Everything stateful — heartbeat,
        ledger append, budget accounting, cancellation — belongs to the caller, which is why
        ``should_reconnect`` can be used to stop early (cancellation) without this layer knowing
        why.

        A reconnect charges against ``reconnect_attempts`` only when a connection made NO forward
        progress: a stream that keeps delivering events across drops is not stuck, but a
        flapping 200-then-close must eventually fail instead of spinning forever.
        """
        _validate_session_id(session_id)
        last_index = max(0, after)
        attempts = 0
        while True:
            progressed_from = last_index
            terminal_seen = False
            # Iterate explicitly (rather than `yield from`) so this loop owns `last_index`: the
            # reconnect must resume from the highest index actually DELIVERED to the caller, and a
            # delegated generator could not update it.
            stream = self._stream_once(session_id, turn_id, last_index)
            try:
                for event in stream:
                    if event.event_index is not None and event.event_index > last_index:
                        last_index = event.event_index
                    yield event
                    if event.is_terminal:
                        terminal_seen = True
                        break
            except _StreamDropped:
                terminal_seen = False
            finally:
                stream.close()

            if terminal_seen:
                return
            if should_reconnect is not None and not should_reconnect(last_index):
                return
            if last_index > progressed_from:
                # Forward progress resets the budget: a stream that keeps delivering is not stuck.
                attempts = 0
                continue
            attempts += 1
            if attempts > self._reconnect_attempts:
                raise SidecarClientError(
                    "the turn event stream dropped repeatedly without progress.",
                    code="sidecar_stream_lost",
                    details={
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "after": last_index,
                        "attempts": attempts,
                    },
                )
            if self._backoff_s:
                _sleep(self._backoff_s * min(attempts, 4))

    def _stream_once(self, session_id: str, turn_id: str, after: int) -> Iterator[SseEvent]:
        """One connection: replay after ``after``, then follow live events.

        Returns normally when the connection ends (the caller decides whether that was a drop); the
        caller detects a terminal through `SseEvent.is_terminal`, so the terminal signal and the
        index bookkeeping cannot disagree.
        """
        url = f"{self._base_url}{SESSIONS_PATH}/{session_id}/events"
        params = {"turn_id": turn_id, "after": str(after)}
        try:
            with self._client.stream(
                "GET",
                url,
                params=params,
                headers=self._headers({"accept": "text/event-stream"}),
                timeout=self._stream_timeout_s,
            ) as response:
                if not (200 <= response.status_code < 300):
                    response.read()
                    self._raise_for(response, what="events stream")
                buffer = ""
                for chunk in response.iter_text():
                    buffer += chunk
                    events, buffer = parse_sse_frames(buffer)
                    for event in events:
                        yield event
        except httpx.HTTPError as exc:
            raise _StreamDropped(str(exc)) from exc

    def abort(self, session_id: str) -> None:
        """Explicitly abort the running turn.

        ``POST /sessions/:id/abort`` (the legacy path the IDE's client also uses). A failure
        raises: an abort that silently did nothing is worse than a loud one, because the caller
        would report "stopped" for a turn that is still spending.
        """
        _validate_session_id(session_id)
        response = self._client.post(
            f"{self._base_url}/sessions/{session_id}/abort",
            headers=self._headers({"content-type": "application/json"}),
            content=b"{}",
        )
        self._raise_for(response, what="abort")

    # -- session mutations --------------------------------------------------------

    def allow_scope(self, session_id: str, path: str) -> str:
        """Approve an out-of-workspace path for this session (``POST …/scope-allow``).

        Returns the root the engine actually granted — the folder, not the file — because the
        engine resolves it and a caller that assumed otherwise would record the wrong grant.
        """
        _validate_session_id(session_id)
        response = self._client.post(
            f"{self._base_url}{SESSIONS_PATH}/{session_id}/scope-allow",
            headers=self._headers({"content-type": "application/json"}),
            content=json.dumps({"path": path}),
        )
        self._raise_for(response, what="allow_scope")
        payload = response.json()
        root = payload.get("root")
        if not isinstance(root, str) or not root:
            raise SidecarClientError(
                "scope-allow returned no root.",
                code="sidecar_response_invalid",
                status=response.status_code,
            )
        return root

    def update_limits(self, session_id: str, limits: SidecarLimits) -> dict[str, Any]:
        """PATCH a live session's caps (``PATCH …/limits``). Returns the engine's applied limits.

        A ``400`` carries the engine's own explanation of *why* a cap was refused; that text is
        kept in the raised error rather than replaced by a generic message.
        """
        _validate_session_id(session_id)
        patch = limits.to_patch()
        if not patch:
            raise SidecarClientError(
                "update_limits needs at least one cap to change (null clears, absent leaves alone).",
                code="limits_patch_empty",
            )
        response = self._client.patch(
            f"{self._base_url}{SESSIONS_PATH}/{session_id}/limits",
            headers=self._headers({"content-type": "application/json"}),
            content=json.dumps(patch),
        )
        self._raise_for(response, what="update_limits")
        payload = response.json()
        applied = payload.get("limits")
        return dict(applied) if isinstance(applied, Mapping) else {}

    # -- probing ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /health`` — liveness, no auth. Used to wait for a sidecar the worker just spawned."""
        response = self._client.get(f"{self._base_url}/health")
        self._raise_for(response, what="health")
        payload = response.json()
        return dict(payload) if isinstance(payload, Mapping) else {}

    def ready(self) -> dict[str, Any]:
        """``GET /ready`` — readiness (capacity + drain state), distinct from liveness."""
        response = self._client.get(f"{self._base_url}/ready")
        self._raise_for(response, what="ready")
        payload = response.json()
        return dict(payload) if isinstance(payload, Mapping) else {}


class _StreamDropped(Exception):
    """The connection ended without a terminal event: reconnect from the last index."""


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
        raise SidecarClientError(
            f"invalid session id {session_id!r}: must match ^[A-Za-z0-9_-]{{1,64}}$",
            code="sidecar_session_id_invalid",
            details={"session_id": repr(session_id)},
        )
