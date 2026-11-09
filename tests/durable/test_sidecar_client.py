"""`SidecarClient` against a REAL loopback HTTP server that speaks the sidecar's `\v1` wire.

No mocking of httpx and no `respx`: the three behaviours under test — `turn_key` idempotency, SSE
resume via `?after=`, and abort-on-cancel — are all properties of *the wire*, so the test drives a
socket. The server below implements exactly the routes and shapes
`agent-core/vendor/cline/algenta/run-server.ts` exposes (verified against that file), including the
`200 + replayed:true` vs `202` distinction and the `id:`/`event: session_event`/`data:` framing
from `server/http.ts`.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import pytest

from codna.durable.sidecar_client import (
    SSE_TERMINAL_TURN_EVENTS,
    SidecarClient,
    SidecarClientError,
    SidecarLimits,
    parse_sse_frames,
)

AUTH_TOKEN = "test-bearer-token"


@dataclass
class Recorded:
    """What the server actually received. Assertions are made against this, not against mocks."""

    requests: list[dict[str, Any]] = field(default_factory=list)

    def paths(self) -> list[str]:
        return [request["path"] for request in self.requests]

    def find(self, method: str, path: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == method and r["path"] == path]


class FakeSidecar:
    """A minimal, honest sidecar: real HTTP, real SSE, real status codes.

    ``cut_after`` is the behaviour the reconnect test needs: the first events connection writes
    events up to that index and then closes WITHOUT a terminal, exactly as a dropped socket would.
    """

    def __init__(self) -> None:
        self.recorded = Recorded()
        self.turn_keys: dict[str, str] = {}
        self.turns: dict[str, list[dict[str, Any]]] = {}
        self.cut_after: int | None = None
        self._cuts_served = 0
        self.aborts: list[str] = []
        self.scope_roots: dict[str, str] = {}
        self.limits_patches: list[dict[str, Any]] = []
        self.require_auth = True
        self.capabilities_override: dict[str, Any] | None = None
        self.events_for_turn: Callable[[str], list[dict[str, Any]]] = self._default_events

    # -- fixtures ------------------------------------------------------------------

    @staticmethod
    def _default_events(turn_id: str) -> list[dict[str, Any]]:
        events = [
            {"event_type": "session.created", "payload": {}},
            {"event_type": "agent.thinking", "payload": {"text": "..."}},
            {"event_type": "agent.text", "payload": {"text": "done"}},
            {"event_type": "session.turn_completed", "payload": {}},
        ]
        return [
            {
                "schema_version": "cline-session-event/v1",
                "event_id": f"evt_{turn_id}_{index} ",
                "event_index": index,
                "session_id": "session_fake",
                "turn_id": turn_id,
                "observed_at": "2026-10-26T00:00:00.000Z",
                "event_type": event["event_type"],
                "payload": event["payload"],
            }
            for index, event in enumerate(events, start=1)
        ]

    def events_with_index_range(self, turn_id: str, count: int) -> list[dict[str, Any]]:
        """A stream of ``count`` events; only the last is terminal."""
        events = []
        for index in range(1, count + 1):
            event_type = "agent.text" if index < count else "session.turn_completed"
            events.append(
                {
                    "schema_version": "cline-session-event/v1",
                    "event_id": f"evt_{index}",
                    "event_index": index,
                    "session_id": "session_fake",
                    "turn_id": turn_id,
                    "observed_at": "2026-10-26T00:00:00.000Z",
                    "event_type": event_type,
                    "payload": {"n": index},
                }
            )
        return events


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"  # type: ignore[assignment]
    protocol_version = "HTTP/1.0"  # close-delimited bodies: a dropped socket is a clean EOF here

    def log_message(self, *args: object) -> None:  # silence the test output
        return

    # -- helpers -------------------------------------------------------------------

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, code: str, message: str, details: dict | None = None) -> None:
        self._json(status, {"error": {"code": code, "message": message, "details": details or {}}})

    def _record(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        self.server.state.recorded.requests.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": self._body() if method in {"POST", "PATCH"} else {},
            }
        )

    def _authorized(self) -> bool:
        if not self.server.state.require_auth:
            return True
        if self.headers.get("authorization") != f"Bearer {AUTH_TOKEN}":
            self._error(401, "unauthorized", "missing or invalid agent-core bearer token.")
            return False
        return True

    # -- routes --------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self._record("GET", parsed.path, query)
        state = self.server.state

        if parsed.path == "/health":
            self._json(200, {"service": "codna-sidecar", "status": "ok"})
            return
        if parsed.path == "/ready":
            self._json(200, {"service": "codna-sidecar", "status": "ready", "ready": True})
            return
        if not self._authorized():
            return
        if parsed.path == "/v1/capabilities":
            self._json(
                200,
                state.capabilities_override
                or {
                    "protocol_version": 1,
                    "events_schema": "cline-session-event/v1",
                    "transports": ["sse"],
                    "features": {"replay": True, "turn_key": True, "resume_from": True, "limits_patch": True},
                },
            )
            return
        if parsed.path == "/v1/sessions/session_fake/events":
            self._stream_events(query)
            return
        self._error(404, "not_found", f"Unknown /v1 route: {parsed.path}")

    def _stream_events(self, query: dict[str, list[str]]) -> None:
        state = self.server.state
        after = int((query.get("after") or ["0"])[0] or 0)
        turn_id = (query.get("turn_id") or [""])[0]
        events = state.events_for_turn(turn_id)

        start_cut = state.cut_after is not None and state._cuts_served == 0
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        for event in events:
            if event["event_index"] <= after:
                continue
            if start_cut and event["event_index"] > state.cut_after:
                state._cuts_served += 1
                self.close_connection = True
                return  # drop the socket mid-turn, no terminal: the client must resume
            frame = f"id: {event['event_index']}\nevent: session_event\ndata: {json.dumps(event)}\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()
            if event["event_type"] in SSE_TERMINAL_TURN_EVENTS:
                return
        if state.cut_after is not None and state._cuts_served == 0:
            state._cuts_served += 1
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        self._record("POST", parsed.path, parse_qs(parsed.query))
        if not self._authorized():
            return
        state = self.server.state

        if parsed.path == "/v1/sessions":
            body = self.server.state.recorded.requests[-1]["body"]
            if not body.get("working_dir"):
                self._error(400, "session_invalid", "working_dir is required")
                return
            if body.get("resume_from") == "unknown-session":
                self._json(201, {"session_id": "session_fake", "resume_from_ignored": True})
                return
            self._json(
                201,
                {
                    "session_id": "session_fake",
                    "owner": body.get("owner") or self.headers.get("x-session-owner", ""),
                    "state": "created",
                    "resumed": bool(body.get("resume_from")),
                },
            )
            return

        if parsed.path == "/v1/sessions/session_fake/turns":
            body = self.server.state.recorded.requests[-1]["body"]
            if not body.get("prompt"):
                self._error(400, "invalid_request", "prompt is required")
                return
            turn_key = self.headers.get("x-codna-turn-key")
            if turn_key and turn_key in state.turn_keys:
                # D1: a replay is 200 with replayed:true — never 202, which would claim a turn.
                self._json(
                    200,
                    {"session_id": "session_fake", "turn_id": state.turn_keys[turn_key], "replayed": True},
                )
                return
            turn_id = f"turn_{len(state.turn_keys) + 1}"
            if turn_key:
                state.turn_keys[turn_key] = turn_id
            self._json(
                202,
                {
                    "session_id": "session_fake",
                    "turn_id": turn_id,
                    "events_url": f"/v1/sessions/session_fake/events?turn_id={turn_id}",
                },
            )
            return

        if parsed.path == "/v1/sessions/session_fake/scope-allow":
            body = self.server.state.recorded.requests[-1]["body"]
            path = body.get("path")
            if not path:
                self._error(400, "invalid_request", "path is required")
                return
            root = "/Users/dev/project/../outside"
            state.scope_roots[path] = root
            self._json(200, {"ok": True, "root": root})
            return

        if parsed.path == "/sessions/session_fake/abort":
            state.aborts.append("session_fake")
            self._json(200, {"session_id": "session_fake", "state": "aborted"})
            return

        self._error(404, "not_found", f"Unknown route: {parsed.path}")

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        self._record("PATCH", parsed.path, parse_qs(parsed.query))
        if not self._authorized():
            return
        state = self.server.state
        if parsed.path == "/v1/sessions/session_fake/limits":
            patch = state.recorded.requests[-1]["body"]
            if patch.get("maxCostUsd") == 0:
                self._error(400, "invalid_limits", "limits.maxCostUsd must be a positive finite number")
                return
            state.limits_patches.append(patch)
            self._json(200, {"ok": True, "limits": patch, "cleared": False})
            return
        self._error(404, "not_found", f"Unknown route: {parsed.path}")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: FakeSidecar) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.state = state


@pytest.fixture
def sidecar() -> Any:
    state = FakeSidecar()
    server = _Server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.server = server  # type: ignore[attr-defined]
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def client(sidecar: FakeSidecar) -> Any:
    port = sidecar.server.server_address[1]  # type: ignore[attr-defined]
    with SidecarClient(f"http://127.0.0.1:{port}", auth_token=AUTH_TOKEN, owner="durable-run-1", backoff_s=0) as c:
        yield c


def _base_url(sidecar: FakeSidecar) -> str:
    return f"http://127.0.0.1:{sidecar.server.server_address[1]}"  # type: ignore[attr-defined]


# -- capabilities -----------------------------------------------------------------------


def test_capabilities_are_fetched_once_and_cached(client: SidecarClient, sidecar: FakeSidecar):
    first = client.capabilities()
    second = client.capabilities()
    assert first is second
    assert first.supports_turn_key and first.supports_resume_from
    assert first.protocol_version == 1
    assert len(sidecar.recorded.find("GET", "/v1/capabilities")) == 1
    client.capabilities(refresh=True)
    assert len(sidecar.recorded.find("GET", "/v1/capabilities")) == 2


def test_negotiate_refuses_an_engine_without_turn_key(sidecar: FakeSidecar):
    sidecar.capabilities_override = {
        "protocol_version": 1,
        "events_schema": "cline-session-event/v1",
        "transports": ["sse"],
        "features": {"turn_key": False},
    }
    with SidecarClient(_base_url(sidecar), auth_token=AUTH_TOKEN) as client:
        assert client.capabilities().supports_turn_key is False
        with pytest.raises(SidecarClientError) as excinfo:
            client.negotiate(require=["turn_key"])
    assert excinfo.value.code == "sidecar_capability_missing"
    assert excinfo.value.details["feature"] == "turn_key"


def test_negotiate_passes_when_the_feature_is_advertised(client: SidecarClient):
    capabilities = client.negotiate(require=["turn_key", "resume_from"])
    assert capabilities.supports_turn_key


# -- session creation -------------------------------------------------------------------


def test_create_session_sends_working_dir_and_owner(client: SidecarClient, sidecar: FakeSidecar):
    session = client.create_session(working_dir="/repo", prompt="fix it")
    assert session.session_id == "session_fake"
    assert session.resumed is False
    request = sidecar.recorded.find("POST", "/v1/sessions")[0]
    assert request["body"]["working_dir"] == "/repo"
    assert request["body"]["prompt"] == "fix it"
    # The durable run owns its session: it must never re-attach to an interactive chat's session.
    assert request["headers"]["x-session-owner"] == "durable-run-1"
    assert request["headers"]["authorization"] == f"Bearer {AUTH_TOKEN}"


def test_create_session_resumes_by_reference(client: SidecarClient, sidecar: FakeSidecar):
    session = client.create_session(working_dir="/repo", resume_from="session_old")
    assert session.resumed is True
    assert sidecar.recorded.find("POST", "/v1/sessions")[0]["body"]["resume_from"] == "session_old"


def test_a_resume_the_engine_cannot_honour_is_reported_not_hidden(client: SidecarClient):
    session = client.create_session(working_dir="/repo", resume_from="unknown-session")
    assert session.resume_from_ignored is True


def test_create_session_surfaces_the_engines_own_error_code(client: SidecarClient):
    with pytest.raises(SidecarClientError) as excinfo:
        client.create_session(working_dir="")
    assert excinfo.value.code == "session_invalid"
    assert excinfo.value.status == 400
    assert "working_dir" in str(excinfo.value)


def test_a_missing_session_id_is_an_error_not_a_guess(sidecar: FakeSidecar, monkeypatch):
    port = sidecar.server.server_address[1]  # type: ignore[attr-defined]
    with SidecarClient(f"http://127.0.0.1:{port}", auth_token=AUTH_TOKEN) as client:
        class _Bad:
            status_code = 201
            def json(self) -> dict:
                return {"owner": "x"}
        monkeypatch.setattr(client._client, "post", lambda *a, **k: _Bad())  # type: ignore[attr-defined]
        with pytest.raises(SidecarClientError) as excinfo:
            client.create_session(working_dir="/repo")
    assert excinfo.value.code == "sidecar_response_invalid"


# -- turn_key idempotency ---------------------------------------------------------------


def test_turn_key_replay_returns_the_same_turn_id(client: SidecarClient, sidecar: FakeSidecar):
    client.create_session(working_dir="/repo")
    first = client.start_turn("session_fake", "do the thing", turn_key="run-1:task-1:1")
    assert first.replayed is False
    assert first.turn_id == "turn_1"

    # The activity is retried (a worker crash, a timeout) with the SAME key.
    second = client.start_turn("session_fake", "do the thing", turn_key="run-1:task-1:1")
    assert second.turn_id == first.turn_id
    assert second.replayed is True

    # Only one turn was ever created, and the engine saw the key as a header (not the body: reading
    # the body for a key would consume the turn stream).
    posts = sidecar.recorded.find("POST", "/v1/sessions/session_fake/turns")
    assert len(posts) == 2
    assert posts[0]["headers"]["x-codna-turn-key"] == "run-1:task-1:1"
    assert posts[1]["headers"]["x-codna-turn-key"] == "run-1:task-1:1"


def test_a_different_key_starts_a_different_turn(client: SidecarClient, sidecar: FakeSidecar):
    client.create_session(working_dir="/repo")
    first = client.start_turn("session_fake", "one", turn_key="run-1:task-1:1")
    second = client.start_turn("session_fake", "two", turn_key="run-1:task-1:2")
    assert first.turn_id != second.turn_id
    assert second.replayed is False


def test_turn_without_a_key_is_never_marked_replayed(client: SidecarClient):
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "one")
    assert start.replayed is False
    assert start.events_url and "turn_id=" in start.events_url


def test_a_turn_carries_its_own_write_scope_and_all_three_states_differ(client: SidecarClient, sidecar: FakeSidecar):
    """The turn's scope has THREE wire states, and collapsing any pair is a defect.

    * omitted — inherit the session's scope;
    * ``[]`` — this turn writes nothing;
    * ``None`` (JSON ``null``) — explicitly NO additional restriction, overriding the session's.

    The middle one is the original defect (`[]` used to mean "unrestricted"). The third is why
    ``None`` cannot stand in for "omitted": a durable session is created READ-ONLY, so its writable
    ``implement`` turn has to lift that scope explicitly — a silent turn would inherit it.
    """
    client.create_session(working_dir="/repo")

    client.start_turn("session_fake", "plan only", allowed_write_paths=[])
    client.start_turn("session_fake", "implement it", allowed_write_paths=None)
    client.start_turn("session_fake", "one shot")  # OMIT_WRITE_SCOPE

    posts = sidecar.recorded.find("POST", "/v1/sessions/session_fake/turns")
    # Present and empty: a read-only turn.
    assert posts[0]["body"]["allowed_write_paths"] == []
    # Present and null: unrestricted for this turn (JSON-serialised, not a Python `None` that vanished).
    assert "allowed_write_paths" in posts[1]["body"]
    assert posts[1]["body"]["allowed_write_paths"] is None
    # ABSENT (not `[]`, not null): keep the session's scope.
    assert "allowed_write_paths" not in posts[2]["body"]


def test_a_narrowed_turn_sends_its_allow_list(client: SidecarClient, sidecar: FakeSidecar):
    client.create_session(working_dir="/repo")
    client.start_turn("session_fake", "edit calc.py", allowed_write_paths=["calc.py"])
    posts = sidecar.recorded.find("POST", "/v1/sessions/session_fake/turns")
    assert posts[0]["body"]["allowed_write_paths"] == ["calc.py"]


def test_start_turn_reports_an_empty_prompt_precisely(client: SidecarClient):
    with pytest.raises(SidecarClientError) as excinfo:
        client.start_turn("session_fake", "")
    assert excinfo.value.code == "invalid_request"
    assert excinfo.value.status == 400


def test_start_turn_validates_the_session_id_before_sending(client: SidecarClient, sidecar: FakeSidecar):
    with pytest.raises(SidecarClientError) as excinfo:
        client.start_turn("../../etc", "x")
    assert excinfo.value.code == "sidecar_session_id_invalid"
    assert sidecar.recorded.find("POST", "/v1/sessions/../../etc/turns") == []


# -- SSE streaming / resume -------------------------------------------------------------


def test_stream_events_are_decoded_in_order_and_stop_at_the_terminal(client: SidecarClient):
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    events = list(client.stream_turn("session_fake", start.turn_id))
    assert [event.event_index for event in events] == [1, 2, 3, 4]
    assert events[-1].event_type == "session.turn_completed"
    assert events[-1].is_terminal is True
    assert events[0].event_index == 1
    assert events[0].session_id == "session_fake"


def test_a_stream_cut_at_event_7_resumes_with_after_7(client: SidecarClient, sidecar: FakeSidecar):
    sidecar.events_for_turn = lambda turn_id: sidecar.events_with_index_range(turn_id, 10)
    sidecar.cut_after = 7
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")

    events = list(client.stream_turn("session_fake", start.turn_id))

    # Every event exactly once, no duplicates across the two connections, ending on the terminal.
    assert [event.event_index for event in events] == list(range(1, 11))
    assert events[-1].is_terminal is True

    # The reconnect asked for exactly the right window — the assertion the plan names.
    event_requests = [
        request
        for request in sidecar.recorded.find("GET", "/v1/sessions/session_fake/events")
    ]
    assert len(event_requests) == 2
    assert event_requests[0]["query"]["after"] == ["0"]
    assert event_requests[1]["query"]["after"] == ["7"]
    assert event_requests[1]["query"]["turn_id"] == [start.turn_id]


def test_resume_from_a_persisted_index_skips_the_backlog(client: SidecarClient, sidecar: FakeSidecar):
    sidecar.events_for_turn = lambda turn_id: sidecar.events_with_index_range(turn_id, 10)
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    # A worker restarted mid-turn and persisted `last_event_index=7` in TaskState.
    events = list(client.stream_turn("session_fake", start.turn_id, after=7))
    assert [event.event_index for event in events] == [8, 9, 10]
    assert sidecar.recorded.find("GET", "/v1/sessions/session_fake/events")[0]["query"]["after"] == ["7"]


def test_a_stream_that_never_progresses_exhausts_its_budget_and_raises(client: SidecarClient, sidecar: FakeSidecar):
    # Every connection drops at event 1 without delivering anything new after the first cut.
    sidecar.events_for_turn = lambda turn_id: sidecar.events_with_index_range(turn_id, 5)
    sidecar.cut_after = 1
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    # `cut_after` only cuts the first connection; make every connection cut by resetting per call.
    sidecar.events_for_turn = lambda turn_id: []  # empty stream: a 200 that immediately closes
    events: list = []
    with pytest.raises(SidecarClientError) as excinfo:
        for event in client.stream_turn("session_fake", start.turn_id):
            events.append(event)
    assert excinfo.value.code == "sidecar_stream_lost"
    assert excinfo.value.details["attempts"] > 0


def test_should_reconnect_can_stop_the_loop_for_cancellation(client: SidecarClient, sidecar: FakeSidecar):
    sidecar.events_for_turn = lambda turn_id: sidecar.events_with_index_range(turn_id, 10)
    sidecar.cut_after = 3
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    seen: list[int] = []
    for event in client.stream_turn(
        "session_fake", start.turn_id, should_reconnect=lambda last: False
    ):
        seen.append(event.event_index)
    # No third connection was attempted: the caller said "stop" and the generator returned.
    assert seen == [1, 2, 3]
    assert len(sidecar.recorded.find("GET", "/v1/sessions/session_fake/events")) == 1


def test_events_require_auth(client: SidecarClient, sidecar: FakeSidecar):
    sidecar.require_auth = True
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    with SidecarClient(_base_url(sidecar)) as unauthenticated:  # no bearer
        with pytest.raises(SidecarClientError) as excinfo:
            list(unauthenticated.stream_turn("session_fake", start.turn_id))
    assert excinfo.value.code == "unauthorized"
    assert excinfo.value.status == 401


# -- abort ------------------------------------------------------------------------------


def test_cancel_maps_to_abort(client: SidecarClient, sidecar: FakeSidecar):
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    # Cancellation: stop consuming the stream (unsubscribe) and abort explicitly.
    stream = client.stream_turn("session_fake", start.turn_id)
    next(stream)
    stream.close()
    client.abort("session_fake")
    assert sidecar.aborts == ["session_fake"]
    assert len(sidecar.recorded.find("POST", "/sessions/session_fake/abort")) == 1


def test_abort_failure_raises_rather_than_reporting_stopped(sidecar: FakeSidecar):
    # An abort that silently did nothing would let a caller report "stopped" for a turn that is
    # still spending. The route 404s for an unknown session, which is exactly that case.
    port = sidecar.server.server_address[1]  # type: ignore[attr-defined]
    with SidecarClient(f"http://127.0.0.1:{port}", auth_token=AUTH_TOKEN) as client:
        with pytest.raises(SidecarClientError) as excinfo:
            client.abort("session_unknown")
    assert excinfo.value.status == 404


def test_turn_handle_aborts_through_the_client(client: SidecarClient, sidecar: FakeSidecar):
    client.create_session(working_dir="/repo")
    start = client.start_turn("session_fake", "x")
    from codna.durable.sidecar_client import TurnHandle

    handle = TurnHandle(
        session_id=start.session_id,
        turn_id=start.turn_id,
        replayed=start.replayed,
        _abort=lambda: client.abort("session_fake"),
    )
    handle.abort()
    assert sidecar.aborts == ["session_fake"]


# -- scope allow / limits ---------------------------------------------------------------


def test_scope_allow_returns_the_granted_root(client: SidecarClient):
    root = client.allow_scope("session_fake", "/Users/dev/project/../outside/file.txt")
    assert root == "/Users/dev/project/../outside"


def test_scope_allow_without_a_path_is_an_error(client: SidecarClient):
    with pytest.raises(SidecarClientError) as excinfo:
        client.allow_scope("session_fake", "")
    assert excinfo.value.code == "invalid_request"


def test_update_limits_sends_only_the_caps_that_changed(client: SidecarClient, sidecar: FakeSidecar):
    applied = client.update_limits("session_fake", SidecarLimits(max_cost_usd=10.0))
    assert applied == {"maxCostUsd": 10.0}
    assert sidecar.limits_patches == [{"maxCostUsd": 10.0}]


def test_update_limits_keeps_the_engines_rejection_reason(client: SidecarClient):
    with pytest.raises(SidecarClientError) as excinfo:
        client.update_limits("session_fake", SidecarLimits(max_cost_usd=0))
    assert excinfo.value.code == "invalid_limits"
    assert "positive finite number" in str(excinfo.value)


def test_an_empty_limits_patch_is_refused_locally(client: SidecarClient, sidecar: FakeSidecar):
    # `{}` would be a no-op request that looks like a successful raise; refusing it in the client
    # means a caller cannot believe it changed a cap when nothing was sent.
    with pytest.raises(SidecarClientError) as excinfo:
        client.update_limits("session_fake", SidecarLimits())
    assert excinfo.value.code == "limits_patch_empty"
    assert sidecar.recorded.find("PATCH", "/v1/sessions/session_fake/limits") == []


# -- probes -----------------------------------------------------------------------------


def test_health_and_ready_need_no_auth(sidecar: FakeSidecar):
    with SidecarClient(_base_url(sidecar)) as client:
        assert client.health()["status"] == "ok"
        assert client.ready()["ready"] is True


def test_a_502_with_an_html_body_still_produces_a_diagnosable_error():
    import threading as _threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Broken(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(502)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>bad gateway</html>")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Broken)
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with SidecarClient(f"http://127.0.0.1:{server.server_address[1]}") as client:
            with pytest.raises(SidecarClientError) as excinfo:
                client.capabilities()
            assert excinfo.value.code == "http_502"
            assert excinfo.value.status == 502
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# -- SSE framing ------------------------------------------------------------------------


def test_parse_sse_frames_handles_crlf_and_heartbeats():
    buffer = (
        ": heartbeat 1234\r\n\r\n"
        'id: 7\r\nevent: session_event\r\ndata: {"event_type": "agent.text", "event_index": 7, '
        '"session_id": "s", "payload": {"text": "hi"}}\r\n\r\n'
    )
    events, remainder = parse_sse_frames(buffer)
    assert remainder == ""
    assert len(events) == 1
    assert events[0].event_index == 7
    assert events[0].event_id == "7"
    assert events[0].event_name == "session_event"
    assert events[0].payload == {"text": "hi"}


def test_parse_sse_frames_returns_an_incomplete_frame_untouched():
    # A chunk boundary in the middle of a frame must not lose data.
    partial = 'event: session_event\ndata: {"event_type": "agent.text", "event_index": 2, "sess'
    events, remainder = parse_sse_frames(partial)
    assert events == []
    assert remainder == partial

    completed = partial + 'ion_id": "s", "payload": {}}\n\n'
    events, remainder = parse_sse_frames(completed)
    assert remainder == ""
    assert [event.event_index for event in events] == [2]


def test_parse_sse_frames_drops_a_malformed_frame_without_losing_the_next():
    # One bad frame must not tear down the stream: the rest of the turn is still paid for.
    buffer = (
        "data: {not json}\n\n"
        'data: {"event_type": "agent.text", "event_index": 4, "session_id": "s", "payload": {}}\n\n'
    )
    events, _ = parse_sse_frames(buffer)
    assert [event.event_index for event in events] == [4]


def test_parse_sse_frames_ignores_a_frame_without_an_event_type():
    buffer = 'data: {"event_index": 3, "payload": {}}\n\n'
    events, _ = parse_sse_frames(buffer)
    assert events == []


def test_an_event_without_an_index_reports_none_not_zero():
    # The index is what gets persisted for reattach; defaulting it to 0 would rewind the resume.
    buffer = 'data: {"event_type": "session.created", "session_id": "s", "payload": {}}\n\n'
    events, _ = parse_sse_frames(buffer)
    assert events[0].event_index is None
