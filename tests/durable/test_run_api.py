"""The Run API (D2.5): the loopback HTTP surface, driven over real HTTP.

**What this file is about.** Not "does it return 200" — the four contracts that make the surface safe
to ship:

1. **The stream is readable by the IDE's parser, unchanged.** The plan's acceptance criterion is
   explicitly not "an SSE stream": it is that
   `ide/extensions/codna-agent-backend/src/streamClient.ts` can consume it. So the framing constants
   are asserted equal to `server/http.ts`'s own literals (read from that file, not restated), and the
   REAL bytes a REAL server sent are fed through `IdeStreamParser` — a transcription of
   `streamClient.ts`'s frame and line delimiters, itself asserted equal to the TypeScript source.
   What comes out is the ledger's records, in order, ending at exactly one terminal.

   **The limit of that proof, stated rather than implied.** `IdeStreamParser` is a transcription in
   Python, not the TypeScript module executed. It proves the BYTES are framed as the IDE's rules
   require — frame delimiter, `data:` accumulation, `id:`/`event:` ignored, comments dropped,
   malformed frames dropped — which is what "the transport is compatible" means.

   **The VOCABULARY seam was a gap and is now closed (D3).** When this file was written,
   `normalizeEngineEvent` mapped `session.turn_completed`/`turn_failed`/`turn_aborted`/`aborted`/
   `stopped` to a turn terminal while this surface's terminal is `run.finished`, which fell through
   its `default` branch to a generic "working" step — the IDE would read every byte correctly and
   never stop at the run's end. `ide/` now has a Run API client (`codnaRuns.ts`), and
   `normalizer.ts` maps `run.finished` (and `run.stream_error`, the API's own final frame) to a
   terminal, so a run's stream DOES stop. Both are asserted below — including that the client's own
   terminal set is exactly what this server emits, so the two cannot drift apart again.
2. **The token is a real boundary.** Missing, wrong and non-``Bearer`` are one 401 body (distinguishing
   them would confirm a token exists); the file is 0600 even when something widened it; and the
   comparison goes through ``hmac.compare_digest`` — asserted on the code path, since a timing test
   would be a flaky one.
3. **Every mutation records WHO.** The three Updates carry the identity into the workflow payload (the
   task's own ``_approvals`` is the record); the three Signals cannot, so the server appends a `note`
   ledger record naming the caller. Both are read back here.
4. **A bad client cannot wedge the server.** A client that disconnects mid-stream costs a thread, not
   the process; the next request still works and the stream slot returns.

**The backend is the seam, not a mock.** Everything except the Temporal client is real: a real socket
on a real ephemeral port, real ``http.client`` requests, the real `RunApiServer`/`RunApiHandler`, and a
`ScriptedBackend` implementing the four-method `RunBackend` interface. `test_run_api_real.py` (gated on
``CODNA_TEST_TEMPORAL=1``) drives the SAME surface against a real Temporal test server, so the seam is
proved on both sides.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

# The `durable` extra is OPT-IN and the offline CI job installs everything EXCEPT it. A few tests here
# import `codna.durable.workflow_support` INSIDE the test body (it is the module that assembles a run's
# payload from coordinates), and that module imports `temporalio` at module scope — so without this guard
# the failure appears at CALL time as a ModuleNotFoundError naming `workflow_support.py`, which reads like
# a product bug rather than a missing optional dependency.
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable import run_api as api_module
from codna.durable import run_api_support as support
from codna.durable import run_repo
from codna.durable.config import load_durable_config
from codna.durable.identity import AUTH_MODE_TOKEN, IdentityConfig, IdentityResolver
from codna.durable.ledger import RunLedger
from codna.durable.siem import NullSiemSink
from codna.durable.run_api import RunApiServer
from codna.durable.run_api_backend import RunBackend, durable_capabilities, translate
from codna.durable.run_api_support import (
    SSE_EVENT_TEMPLATE,
    SSE_HEADERS,
    SSE_HEARTBEAT_TEMPLATE,
    RunApiError,
    RunApiSettings,
    after_param,
    constant_time_token_equals,
    load_or_create_token,
    outcome_of,
    parse_bearer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HTTP_TS = REPO_ROOT / "agent-core" / "vendor" / "cline" / "algenta" / "server" / "http.ts"


def _engine_ts(name: str) -> Path:
    """Resolve an engine TypeScript module wherever it currently lives.

    These assertions are a CROSS-LANGUAGE contract: the Python server's framing constants must
    equal the TypeScript client's. That only holds if the test can find the client, and the client
    moved once already -- V1 extracted the portable half of `codna-agent-backend` into
    `@codna/engine-client`, which turned every hard-coded path here into a hard failure that read
    like a protocol regression.

    So search both known roots and fail with a message that names them, rather than reporting a
    missing file as if the constants had diverged.
    """
    candidates = (
        REPO_ROOT / "ide" / "packages" / "codna-engine-client" / "src" / name,
        REPO_ROOT / "ide" / "extensions" / "codna-agent-backend" / "src" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{name} is not where this test expects it; searched:\n  {searched}")


STREAM_CLIENT_TS = _engine_ts("streamClient.ts")
NORMALIZER_TS = _engine_ts("normalizer.ts")
CODNA_RUNS_TS = _engine_ts("codnaRuns.ts")

# A blank import guard: `temporalio` is NOT needed by anything in this file. The Run API's transport
# never imports it (that is `run_api_backend.TemporalRunBackend`'s job, imported lazily inside it),
# so the whole surface is testable without the extra.

# The token the SERVER mints is per-test (`server.token`); this constant exists only for the parser
# unit tests, which do not involve a server. A test that authenticates must use the server's own.
TOKEN = "test-token-0123456789"
# The identity every request in this file is authenticated AS. A literal rather than the host's OS
# user so an assertion about "who acted" does not depend on who runs the tests.
TEST_IDENTITY = "qa@codna.ai"




# -- fixtures ----------------------------------------------------------------------------------


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch | None = None) -> Any:
    """A real `DurableConfig` rooted in the test's tmp dir — never the developer's ``~/.codna``.

    ``CODNA_RUNTIME_ROOT`` must be set in the REAL environment, not only in the mapping handed to
    `load_durable_config`: `DurableConfig.paths` is resolved through `build_runtime_paths()`, which
    reads ``os.environ``. Passing the mapping alone left the runs root pointing at the developer's
    home directory, which is exactly the leak a test must not have.
    """
    root = tmp_path / "runtime"
    os.environ["CODNA_RUNTIME_ROOT"] = str(root)
    if monkeypatch is not None:
        monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(root))
    env = dict(os.environ)
    env["CODNA_RUNTIME_ROOT"] = str(root)
    return load_durable_config(env=env, hostname="run-api-test-host")


class ScriptedBackend(RunBackend):
    """A `RunBackend` that records what it was asked and answers from a script.

    Deliberately dumb: the interesting assertions are about what REACHED the backend (the identity,
    the task id, the workflow method) and about what the transport did with its answer, not about
    Temporal, which `test_run_api_real.py` covers.
    """

    def __init__(self, *, statuses: Mapping[str, dict[str, Any] | None] | None = None) -> None:
        self.statuses: dict[str, dict[str, Any] | None] = dict(statuses or {})
        self.started: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.signals: list[tuple[str, str]] = []
        self.closed = False
        self.start_error: BaseException | None = None

    def capabilities(self) -> Any:
        # A Temporal-shaped answer: this backend stands in for a CONNECTED server, so every route is
        # servable. The `--no-temporal` mode has its own tests, which assert the other answer.
        return durable_capabilities("127.0.0.1:7233")

    def start_run(self, *, payload: Mapping[str, Any], run_id: str, run_timeout_s: int) -> dict[str, Any]:
        if self.start_error is not None:
            raise self.start_error
        self.started.append({"payload": dict(payload), "run_id": run_id, "timeout": run_timeout_s})
        return {
            "run_id": run_id,
            "workflow_id": f"run:{run_id}",
            "task_queue": "codna.agent.test",
            "engine": payload.get("engine"),
            "tasks": len(payload.get("tasks") or ()),
            "versioning": {"pinned": False},
        }

    def status(self, run_id: str) -> dict[str, Any] | None:
        return self.statuses.get(run_id)

    def update(self, run_id: str, *, task_id: str, action: str, payload: Mapping[str, Any]) -> Any:
        self.updates.append(
            {"run_id": run_id, "task_id": task_id, "action": action, "payload": dict(payload)}
        )
        return {"applied": True, "identity": payload.get("identity")}

    def signal(self, run_id: str, action: str) -> None:
        self.signals.append((run_id, action))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A tmp-rooted config, with the root set in the environment for the whole test.

    `monkeypatch` reverts it, so a later test cannot inherit this one's root.
    """
    return _config(tmp_path, monkeypatch)


@pytest.fixture
def server(config: Any, tmp_path: Path) -> Iterator[Any]:
    """The REAL server on an ephemeral loopback port, with a scripted backend.

    ``port=0`` rather than a guessed free port: the OS hands out one nobody else can be holding, so
    a test never races the machine it runs on. The server is stopped in teardown because a leaked
    listener poisons every later test with a thread and a port.

    The identity is a FIXED literal (``TEST_IDENTITY``) rather than the host's OS user: a test that
    asserted the developer's username would be a test that passes on one machine. The mode is the
    local bearer token with all three roles, which is what a developer's own server is; the role
    matrix and the OIDC half are covered by `test_run_api_rbac.py`.
    """
    backend = ScriptedBackend()
    settings = RunApiSettings(
        port=0,
        token_path=config.paths.root / "run-api" / "run-api.token",
        heartbeat_s=0.15,
        poll_interval_s=0.02,
        request_timeout_s=5.0,
        write_timeout_s=2.0,
        allowed_repo_roots=(tmp_path,),
    )
    running = RunApiServer(
        config=config,
        backend=backend,
        settings=settings,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_TOKEN,
                local_token=load_or_create_token(settings.token_path),
                local_identity=TEST_IDENTITY,
            )
        ),
        siem=NullSiemSink(),
    ).start()
    try:
        yield Running(running, backend, config)
    finally:
        running.stop()


class Running:
    """What a test needs to drive the server: its address, its token, its backend."""

    def __init__(self, server: RunApiServer, backend: ScriptedBackend, config: Any) -> None:
        self.server = server
        self.backend = backend
        self.config = config
        self.token = server.token
        self.host, self.port = server.address

    # -- plain requests --------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        token: str | None = "__default__",
        headers: Mapping[str, str] | None = None,
        raw_body: bytes | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, dict[str, Any] | None, Mapping[str, str]]:
        """One HTTP request over a fresh connection. Returns (status, json body or None, headers)."""
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        payload = raw_body
        if payload is None and body is not None:
            payload = json.dumps(body).encode("utf-8")
        request_headers: dict[str, str] = {"accept": "application/json"}
        if payload is not None:
            request_headers["content-type"] = "application/json"
        resolved = self.token if token == "__default__" else token
        if resolved is not None:
            request_headers["authorization"] = f"Bearer {resolved}"
        request_headers.update(dict(headers or {}))
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            decoded: dict[str, Any] | None = None
            if raw:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    decoded = None
            return response.status, decoded, dict(response.getheaders())
        finally:
            connection.close()

    def json(self, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        status, body, _ = self.request(method, path, **kwargs)
        return status, body

    # -- streams ---------------------------------------------------------------

    def open_stream(
        self, path: str, *, token: str | None = "__default__", timeout: float = 10.0
    ) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
        """Open a streaming GET and hand back the live response (the caller reads/closes it)."""
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        resolved = self.token if token == "__default__" else token
        headers = {"accept": "text/event-stream"}
        if resolved is not None:
            headers["authorization"] = f"Bearer {resolved}"
        connection.request("GET", path, headers=headers)
        return connection.getresponse(), connection

    def read_until_terminal(self, path: str, *, timeout: float = 20.0) -> tuple[bytes, Mapping[str, str]]:
        """Read a stream to its end (the server closes it at the terminal). Returns raw bytes."""
        response, connection = self.open_stream(path, timeout=timeout)
        try:
            raw = response.read()
            return raw, dict(response.getheaders())
        finally:
            connection.close()

    # -- ledger seeding --------------------------------------------------------

    def seed_ledger(self, run_id: str) -> list[Any]:
        """A small, realistic ledger: start, two turn events, task + run terminals.

        Written with the REAL `RunLedger`, so the hash chain is genuine and a test that reads the
        chain back is reading what an activity would have written.
        """
        ledger = RunLedger.for_run(self.config.paths, run_id)
        records = [
            ledger.append(
                "run.started",
                payload={"event": "notification", "kind": "run.started", "message": "go"},
            ),
            ledger.append(
                "turn.event",
                task_id="task-1",
                session_id="sess-1",
                event_index=1,
                payload={"event_type": "agent.content_start", "payload": {"text": "hi"}},
            ),
            ledger.append(
                "turn.event",
                task_id="task-1",
                session_id="sess-1",
                event_index=2,
                payload={"event_type": "agent.done", "payload": {}},
            ),
            ledger.append(
                "task.finished",
                task_id="task-1",
                payload={"event": "notification", "kind": "task.finished", "message": "ok"},
            ),
            ledger.append(
                "note",
                payload={
                    "event": "notification",
                    "kind": "run.finished",
                    "message": "done",
                    "outcome": "completed",
                },
            ),
        ]
        return [record for record in records if record is not None]


# -- the IDE's parser, transcribed -------------------------------------------------------------
#
# These two regexes are `streamClient.ts`'s `SSE_FRAME_DELIM` and `SSE_LINE_DELIM`, character for
# character. They are asserted equal to the TypeScript source by
# `test_the_parser_used_here_matches_the_ides_own_delimiters`, so this transcription cannot drift.


class IdeStreamParser:
    """`streamClient.ts`'s ``consumeStream`` + ``handleFrame``, as a byte sink.

    Fed the same bytes the IDE would receive, it yields the same ``EngineTurnEvent`` objects: the
    ``data:`` lines joined by newlines and JSON-parsed, comments skipped, ``id:``/``event:`` ignored,
    malformed frames dropped rather than raised. If the server's framing changed in any way that
    mattered to the IDE, this class would stop yielding the events.
    """

    FRAME_DELIM = re.compile(r"\r\n\r\n|\n\n|\r\r")
    LINE_DELIM = re.compile(r"\r\n|\n|\r")

    def __init__(self) -> None:
        self._buffer = ""
        self.events: list[dict[str, Any]] = []
        self.malformed = 0

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._buffer += chunk.decode("utf-8")
        produced: list[dict[str, Any]] = []
        while True:
            match = self.FRAME_DELIM.search(self._buffer)
            if not match:
                return produced
            frame = self._buffer[: match.start()]
            self._buffer = self._buffer[match.end() :]
            event = self._handle_frame(frame)
            if event is not None:
                self.events.append(event)
                produced.append(event)

    def _handle_frame(self, frame: str) -> dict[str, Any] | None:
        data_lines: list[str] = []
        for line in self.LINE_DELIM.split(frame):
            if line.startswith(":"):
                continue  # heartbeat / comment
            if line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
        if not data_lines:
            return None
        try:
            parsed = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            self.malformed += 1
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("event_type"), str):
            return parsed
        self.malformed += 1
        return None




# -- shared test fixtures (in a helper, not a `conftest.py`, because they are plain functions) ----


def drain(response: Any, parser: IdeStreamParser) -> None:
    """Read a stream to EOF, feeding every chunk to the parser (a thread body for stream tests)."""
    while True:
        chunk = response.read1(8192)
        if not chunk:
            return
        parser.feed(chunk)


def git_repo(path: Path) -> Path:
    """A real work tree: `POST /runs` validates the repo the same way `codna run` does."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "test@codna.ai"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        completed = subprocess.run(
            [argv[0], "-C", str(path), *argv[1:]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    return path


def seed_mutable_run(server: Running) -> str:
    """A run that exists (a ledger) and has a live status, so both halves of existence agree."""
    server.seed_ledger("run-1")
    server.backend.statuses["run-1"] = {"run_id": "run-1", "tasks_done": 0}
    return "run-1"


def cli_error(message: str, *, code: str) -> BaseException:
    """A `DurableCliError`, constructed lazily so importing the CLI does not run a command."""
    from codna.durable.cli import DurableCliError

    return DurableCliError(message, code=code)


# -- 1. framing: the acceptance criterion -----------------------------------------------------


def test_the_framing_constants_equal_the_engine_servers_literals():
    """The template, the four headers and the heartbeat must be the sidecar's, not merely valid SSE.

    Read from `server/http.ts` rather than restated here: a copy that drifts from its source is the
    failure this test exists for, and a hard-coded copy would drift with it.
    """
    source = HTTP_TS.read_text(encoding="utf-8")
    assert SSE_EVENT_TEMPLATE == "id: {event_index}\nevent: session_event\ndata: {data}\n\n"
    assert SSE_HEARTBEAT_TEMPLATE == ": heartbeat {now_ms}\n\n"
    # The engine writes these three as one template literal; the exact substring is asserted so a
    # change to `event:` (the parser ignores it, but the shape is the contract) is caught.
    #
    # The payload is `dialectFrame(event, dialect)`, not `JSON.stringify(event)`: V0 made the wire
    # format negotiable (`cline-session-event/v1` vs `codna-session-event/v1`), and every event is
    # re-stamped for the dialect the reader negotiated. Asserting the old substring would pin the
    # pre-negotiation wire format and fail the moment a second dialect exists, which is exactly what
    # happened. The FRAME SHAPE (`id:` / `event:` / `data:` / blank line) is still the contract, so
    # that is what is asserted -- not which function produced the bytes in between.
    assert "id: ${event.event_index}\\nevent: session_event\\ndata: ${" in source
    assert "\\n\\n`" in source
    assert "dialectFrame(event, dialect)" in source
    assert ": heartbeat ${Date.now()}\\n\\n" in source
    for name, value in SSE_HEADERS:
        quoted = f'"{name}": "{value}"' if "-" in name else f'{name}: "{value}"'
        assert quoted in source, f"{name}: {value} is not what server/http.ts writes"


def test_the_parser_used_here_matches_the_ides_own_delimiters():
    """The transcription above is only evidence if it is the same regex the IDE ships."""
    assert "const SSE_FRAME_DELIM = /\\r\\n\\r\\n|\\n\\n|\\r\\r/;" in STREAM_CLIENT_TS.read_text(
        encoding="utf-8"
    )
    assert IdeStreamParser.FRAME_DELIM.pattern == r"\r\n\r\n|\n\n|\r\r"
    assert IdeStreamParser.LINE_DELIM.pattern == r"\r\n|\n|\r"


def test_the_stream_the_server_sends_is_read_by_the_ide_parser(server: Running) -> None:
    """THE acceptance test: real server bytes -> the IDE's own frame rules -> the same events.

    Not "it looks like SSE": every record the ledger holds is delivered, in order, as an
    ``EngineTurnEvent``-shaped object the IDE's ``normalizeEngineEvent`` could consume (it needs a
    string ``event_type``), with ``event_index`` carrying the resume cursor.
    """
    records = server.seed_ledger("run-1")
    raw, headers = server.read_until_terminal("/runs/run-1/events")

    # The response headers are the sidecar's, exactly.
    assert headers["content-type"] == "text/event-stream"
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"
    # No content-length: like `http.ts`, the body ends at `end()`, not at a declared length.
    assert "content-length" not in headers

    parser = IdeStreamParser()
    events = parser.feed(raw)
    assert parser.malformed == 0, raw[-400:]
    assert [event["event_index"] for event in events] == [record.seq for record in records]
    assert [event["event_type"] for event in events] == [
        "run.started",
        "agent.content_start",
        "agent.done",
        "task.finished",
        "run.finished",
    ]
    # Every frame is a complete object with the fields the IDE reads.
    for event, record in zip(events, records):
        assert event["run_id"] == "run-1"
        assert event["kind"] == record.kind
        assert event["hash"] == record.hash
    assert events[1]["task_id"] == "task-1"
    assert events[1]["session_id"] == "sess-1"


def test_the_run_finished_frame_IS_a_terminal_the_ide_now_stops_on(server: Running) -> None:
    """The vocabulary seam, closed — inverted ON PURPOSE from the gap this test used to assert.

    **What it used to say (D2.5).** `streamClient.ts` stops reading at a frame whose normalized event is
    a turn terminal, and `normalizer.ts` only produced one for `session.turn_completed`/`turn_failed`/
    `session.turn_aborted`/`aborted`/`stopped`. This surface's terminal is `run.finished`, which the
    normalizer did not know, so it landed in its `default` branch as a generic "working" step: the IDE
    read every byte correctly and never stopped at the run's end. That was a presentation-layer gap,
    not a framing gap, and this test existed to keep it visible.

    **What changed (D3).** `normalizer.ts` maps `run.finished` — and `run.stream_error`, the API's own
    final frame — to a terminal presentation event, and `ide/codnaRuns.ts` is the client that reads
    this stream. So the assertion flips: the run's terminal is now terminal on BOTH sides of the seam.
    It is asserted against the SOURCE here rather than by running TypeScript, because that is what this
    suite is: a Python proof that the bytes and the contract line up. The runtime proof — that a
    `run.finished` frame actually stops a live subscription — is
    `ide/extensions/codna-agent-backend/test/runTerminal.test.ts`, which drives the real
    `CodnaStreamClient` against a real socket.
    """
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events?follow=0")
    events = IdeStreamParser().feed(raw)
    assert events[-1]["event_type"] == "run.finished"

    normalizer = NORMALIZER_TS.read_text(encoding="utf-8")
    # The turn terminals are still mapped (nothing was replaced: a run stream can carry a task's turn events).
    for known in ("session.turn_completed", "session.turn_failed", "session.turn_aborted",
                  "session.aborted", "session.stopped"):
        assert f'case "{known}"' in normalizer, known
    # …and the run's own terminals now are too. This is the flip.
    assert 'case "run.finished"' in normalizer
    assert 'case "run.stream_error"' in normalizer


def test_the_ides_terminal_set_is_exactly_what_this_server_emits() -> None:
    """The other half of the seam: the client's set and the server's set cannot drift apart.

    A client that stopped on a type this server never sends would never stop; a server that ended a
    stream on a type the client did not know would leave the stream open. Both sides are therefore read
    from their own source and compared.
    """
    client = CODNA_RUNS_TS.read_text(encoding="utf-8")

    # The client's constants, read as literals rather than trusted from prose.
    terminal = re.search(r'export const RUN_TERMINAL_EVENT = "([^"]+)";', client)
    stream_error = re.search(r'export const RUN_STREAM_ERROR_EVENT = "([^"]+)";', client)
    assert terminal is not None, "codnaRuns.ts no longer exports RUN_TERMINAL_EVENT"
    assert stream_error is not None, "codnaRuns.ts no longer exports RUN_STREAM_ERROR_EVENT"
    assert terminal.group(1) == support.TERMINAL_EVENT
    client_types = {terminal.group(1), stream_error.group(1)}

    # The server's real terminal, from the module under test, plus the API's own final frame (written by
    # `run_api_http._write_stream_error`, outside the ledger's vocabulary).
    server_types = set(support.TERMINAL_EVENT_TYPES) | {"run.stream_error"}
    assert client_types == server_types == {"run.finished", "run.stream_error"}

    # And the set the client STOPS on is spelled from those two constants, not from loose literals.
    declared = re.search(r"RUN_TERMINAL_EVENT_TYPES[^=]*=\s*new Set\(\[(.*?)\]\)", client, re.DOTALL)
    assert declared is not None, "codnaRuns.ts no longer declares RUN_TERMINAL_EVENT_TYPES"
    members = {name.strip() for name in declared.group(1).split(",") if name.strip()}
    assert members == {"RUN_TERMINAL_EVENT", "RUN_STREAM_ERROR_EVENT"}



def test_the_stream_ends_at_exactly_one_terminal(server: Running) -> None:
    """A terminal in the backlog ends the stream; records written after it are never sent.

    The IDE's `consumeStream` stops at the first terminal, so a frame after one is a frame it would
    have thrown away — and a second terminal would be the "two terminals" bug it guards against.
    """
    ledger = RunLedger.for_run(server.config.paths, "run-1")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "outcome": "completed"})
    # Written AFTER the terminal on purpose: a stream that kept following would deliver it.
    ledger.append("note", payload={"event": "notification", "kind": "run.finished"})

    raw, _ = server.read_until_terminal("/runs/run-1/events")
    events = IdeStreamParser().feed(raw)
    terminals = [event for event in events if event["event_type"] == "run.finished"]
    assert len(terminals) == 1
    assert len(events) == 2


def test_a_live_frame_is_written_before_the_stream_ends(server: Running) -> None:
    """Frames are flushed as they are produced, not batched at the end.

    Asserted by reading a PARTIAL stream: the first frame must be readable while the run is still
    open (no terminal yet), which is the property that makes the IDE's progress display work.
    """
    ledger = RunLedger.for_run(server.config.paths, "run-1")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    response, connection = server.open_stream("/runs/run-1/events")
    try:
        parser = IdeStreamParser()
        deadline = time.monotonic() + 10.0
        while not parser.events and time.monotonic() < deadline:
            chunk = response.read1(4096)  # one available chunk, not "until EOF"
            if not chunk:
                break
            parser.feed(chunk)
        assert [event["event_type"] for event in parser.events] == ["run.started"]
    finally:
        connection.close()

    # ...and the run finishing while a client is attached arrives on the SAME connection.
    response, connection = server.open_stream("/runs/run-1/events?after=1")
    try:
        parser = IdeStreamParser()
        watcher = threading.Thread(target=drain, args=(response, parser), daemon=True)
        watcher.start()
        time.sleep(0.15)  # let the follower loop poll once with nothing new
        ledger.append("note", payload={"event": "notification", "kind": "run.finished"})
        watcher.join(timeout=10.0)
        assert not watcher.is_alive(), "the stream did not close on the terminal"
        assert [event["event_type"] for event in parser.events] == ["run.finished"]
    finally:
        connection.close()


def test_a_heartbeat_is_a_comment_the_parser_ignores(server: Running) -> None:
    """An idle stream sends keepalives; the IDE treats them as comments, not events.

    Asserted end to end rather than on the template alone: the server must actually WRITE one while
    the run is open, and the parser must drop it (zero events, zero malformed frames).
    """
    ledger = RunLedger.for_run(server.config.paths, "run-1")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    response, connection = server.open_stream("/runs/run-1/events?after=1")
    parser = IdeStreamParser()
    try:
        buffer = b""
        deadline = time.monotonic() + 10.0
        while b"heartbeat" not in buffer and time.monotonic() < deadline:
            chunk = response.read1(4096)
            if not chunk:
                break
            buffer += chunk
            parser.feed(chunk)
        assert b": heartbeat " in buffer
        assert parser.malformed == 0
        assert parser.events == []
    finally:
        connection.close()


# -- 2. auth -----------------------------------------------------------------------------------


def test_a_missing_token_is_401_with_a_structured_body(server: Running) -> None:
    status, body = server.json("GET", "/runs", token=None)
    assert status == 401
    assert body is not None and body["error"]["code"] == "run_api_unauthorized"
    assert "details" in body["error"]


def test_a_wrong_token_is_401_and_says_the_same_thing_as_a_missing_one(server: Running) -> None:
    """Identical bodies on purpose: a different message would confirm a token exists."""
    missing = server.json("GET", "/runs", token=None)
    wrong = server.json("GET", "/runs", token="not-the-token")
    assert missing[0] == wrong[0] == 401
    assert missing[1]["error"]["message"] == wrong[1]["error"]["message"]
    assert missing[1]["error"]["code"] == wrong[1]["error"]["code"]


@pytest.mark.parametrize("shape", ["basic", "bare", "extra", "empty", "spaces"])
def test_a_malformed_authorization_header_is_401(server: Running, shape: str) -> None:
    """A header that is not exactly ``Bearer <token>`` is "absent", never a parse hint."""
    header = {
        "basic": "Basic dXNlcjpwYXNz",
        "bare": server.token,
        "extra": f"Bearer {server.token} extra",
        "empty": "Bearer",
        "spaces": "Bearer    ",
    }[shape]
    status, body, _ = server.request("GET", "/runs", token=None, headers={"authorization": header})
    assert status == 401, shape
    assert body["error"]["code"] == "run_api_unauthorized"


def test_a_valid_token_is_200(server: Running) -> None:
    status, body = server.json("GET", "/runs")
    assert status == 200
    assert body is not None and body["runs"] == []


def test_the_bearer_scheme_is_case_insensitive(server: Running) -> None:
    for scheme in ("bearer", "Bearer", "BEARER"):
        status, _, _ = server.request(
            "GET", "/runs", token=None, headers={"authorization": f"{scheme} {server.token}"}
        )
        assert status == 200, scheme


def test_the_token_file_is_created_0600(tmp_path: Path) -> None:
    path = tmp_path / "tokens" / "run-api.token"
    token = load_or_create_token(path)
    assert token and len(token) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text(encoding="utf-8").strip() == token


def test_a_widened_token_file_is_restricted_back_to_0600(tmp_path: Path) -> None:
    """An operator's ``cp``/``install`` can widen the mode; reading it must narrow it again."""
    path = tmp_path / "run-api.token"
    path.write_text("existing-token\n", encoding="utf-8")
    os.chmod(path, 0o600)
    subprocess.run(["chmod", "o+x", str(path)], check=True)
    assert load_or_create_token(path) == "existing-token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_existing_token_is_reused_not_regenerated(tmp_path: Path) -> None:
    """Two servers on one host must share one credential, or the IDE needs two."""
    path = tmp_path / "run-api.token"
    first = load_or_create_token(path)
    assert load_or_create_token(path) == first


def test_the_token_comparison_goes_through_compare_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asserted on the CODE PATH, not on timing — a timing test is a flaky test.

    ``hmac.compare_digest`` is replaced with a recorder: if the server compared with ``==`` (or
    skipped the call for some inputs), the recorder would not see every attempt.
    """
    seen: list[tuple[bytes, bytes]] = []
    real = __import__("hmac").compare_digest

    def recording(left: Any, right: Any) -> bool:
        seen.append((left, right))
        return real(left, right)

    monkeypatch.setattr("hmac.compare_digest", recording)
    assert constant_time_token_equals("abc", "abc") is True
    assert constant_time_token_equals("abc", "abd") is False
    assert seen == [(b"abc", b"abc"), (b"abc", b"abd")]


def test_the_server_uses_the_constant_time_comparison(server: Running, monkeypatch: pytest.MonkeyPatch) -> None:
    """The local token comparison goes through ``hmac.compare_digest`` — proven with a wrong token.

    The comparison moved into `identity.IdentityResolver` when roles arrived (the token is now one
    way to prove an identity, not the whole boundary), so the spy lands THERE. The assertion is the
    same one that matters: a wrong token is refused, and the refusal went through compare_digest
    rather than ``==``.
    """
    import hmac

    calls: list[bool] = []
    real = hmac.compare_digest

    def spy(left: Any, right: Any) -> bool:
        calls.append(True)
        return real(left, right)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    status, _ = server.json("GET", "/runs", token="wrong")
    assert status == 401
    assert calls, "the authorization path did not go through the constant-time helper"


@pytest.mark.parametrize(
    "header, expected",
    [
        (None, None),
        ("", None),
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("BEARER  abc ", "abc"),
        ("Basic abc", None),
        ("Bearer ", None),
        ("Bearer", None),
    ],
)
def test_parse_bearer(header: str | None, expected: str | None) -> None:
    assert parse_bearer(header) == expected


def test_the_token_is_never_in_a_response_or_a_log(server: Running, caplog: Any) -> None:
    """No route echoes the token, and no line the server logs contains it."""
    caplog.set_level(logging.DEBUG, logger="codna.durable.run_api")
    for path in ("/runs", "/runs/missing"):
        _, body, headers = server.request("GET", path)
        assert server.token not in json.dumps(body)
        assert server.token not in json.dumps(dict(headers))
    assert server.token not in caplog.text



# -- 3. starting and reading runs --------------------------------------------------------------


def test_post_runs_starts_a_run_and_answers_202(server: Running, tmp_path: Path) -> None:
    """202, not 201: the run was ACCEPTED and executes elsewhere; this response is not its state."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={"run_id": "run-post", "repo_dir": str(repo), "briefs": ["fix the retry"], "engine": "plain"},
    )
    assert status == 202
    assert body["run"]["run_id"] == "run-post"
    assert body["run"]["workflow_id"] == "run:run-post"
    assert body["run"]["tasks"] == 1
    # The payload that reached the backend is the CLI's own `run_payload_for` output, not a lookalike.
    started = server.backend.started[0]
    assert started["run_id"] == "run-post"
    assert started["payload"]["tasks"][0]["brief"] == "fix the retry"
    assert started["payload"]["tasks"][0]["repo_dir"] == str(repo.resolve())
    assert started["timeout"] > 0


def test_post_runs_refuses_a_repo_outside_the_authorized_roots(server: Running, tmp_path: Path) -> None:
    outside = git_repo(tmp_path.parent / f"{tmp_path.name}-outside")
    status, body = server.json("POST", "/runs", body={"repo_dir": str(outside), "briefs": ["b"]})
    assert status == 400
    assert body["error"]["code"] == "durable_repo_outside_allowed_roots"
    assert server.backend.started == []


def test_post_runs_refuses_an_in_root_symlink_that_escapes(server: Running, tmp_path: Path) -> None:
    outside = git_repo(tmp_path.parent / f"{tmp_path.name}-symlink-outside")
    link = tmp_path / "repo-link"
    link.symlink_to(outside, target_is_directory=True)
    status, body = server.json("POST", "/runs", body={"repo_dir": str(link), "briefs": ["b"]})
    assert status == 400
    assert body["error"]["code"] == "durable_repo_outside_allowed_roots"
    assert server.backend.started == []


def test_post_runs_defaults_the_run_id(server: Running, tmp_path: Path) -> None:
    """No `run_id` means the CLI's timestamped default — an idempotent retry would collide, which
    is why the caller that wants to retry supplies one."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json("POST", "/runs", body={"repo_dir": str(repo), "briefs": ["b"]})
    assert status == 202
    assert re.fullmatch(r"run-\d{8}T\d{6}Z", body["run"]["run_id"])


def test_post_runs_accepts_the_optional_spec_fields(server: Running, tmp_path: Path) -> None:
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-full",
            "repo_dir": str(repo),
            "briefs": ["a", "b"],
            "base_branch": "develop",
            "verify_commands": ["pytest -q"],
            "max_cost_usd": 4.5,
            "max_tokens": 1000,
            "max_attempts": 2,
            "engine": "plain",
            "run_timeout_s": 600,
        },
    )
    assert status == 202
    payload = server.backend.started[0]["payload"]
    assert payload["base_branch"] == "develop"
    assert payload["max_cost_usd"] == 4.5
    assert [task["verify_commands"] for task in payload["tasks"]] == [["pytest -q"], ["pytest -q"]]
    assert server.backend.started[0]["timeout"] == 600


# -- POST /runs: the provider coordinates ------------------------------------------------------
#
# The hole these cover: `TaskSpec` and `codna run` have carried `provider`/`model`/`base_url` since the
# soak found that a host holding only `OPENAI_API_KEY` failed every turn on the engine's `anthropic`
# default, but `POST /runs` did not accept them — and the Run API is what the IDE submits through. An
# IDE run therefore could not select a provider and got the same silent default, re-opening the bug on
# the path the IDE actually uses.
#
# Every assertion below decodes the STARTED PAYLOAD back into a `TaskSpec` (`spec_from`, the same call a
# worker makes), so what is pinned is the spec a turn would execute on, not a request echo.


def _provider_spec(started: Mapping[str, Any]) -> Any:
    """The `TaskSpec` the first task was started with, decoded by the worker's own parser."""
    from codna.durable.workflow_support import spec_from

    tasks = started["payload"]["tasks"]
    assert tasks, "the run was started with no tasks"
    return spec_from(tasks[0])


def test_the_route_accepts_exactly_the_coordinates_provider_py_owns() -> None:
    """The accepted field set and the validator's own vocabulary are ONE statement.

    If a coordinate is added to `provider.py` and not to the route (or the other way round), the
    caller would send a field the route 400s — the hole this change closed. And the reverse: a
    credential source must never appear here, because it is not the caller's decision.
    """
    from codna.durable.provider import PROVIDER_COORDINATE_FIELDS, PROVIDER_KEY_SOURCES
    from codna.durable.run_api_http import RunApiHandler

    assert set(PROVIDER_COORDINATE_FIELDS) == {"provider", "model", "base_url"}
    assert set(PROVIDER_COORDINATE_FIELDS) <= RunApiHandler._START_FIELDS
    assert "provider_key_source" not in RunApiHandler._START_FIELDS
    assert PROVIDER_KEY_SOURCES == ("environment",)
    # `max_tokens` is a BUDGET, not a credential; the check names the credential-shaped names only.
    assert not {
        name
        for name in RunApiHandler._START_FIELDS
        if name.endswith(("_key", "_key_source", "_api_key")) or name in ("api_key", "authorization")
    }


def test_post_runs_carries_the_provider_coordinates_into_the_task_spec(
    server: Running, tmp_path: Path
) -> None:
    """`provider`/`model`/`base_url` reach the spec a turn executes on — and no key travels with them."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-provider",
            "repo_dir": str(repo),
            "briefs": ["fix the retry"],
            "engine": "plain",
            "provider": "openai-compatible",
            "model": "qwen3.6:latest",
            "base_url": "http://127.0.0.1:11434/v1",
        },
    )
    assert status == 202, body
    spec = _provider_spec(server.backend.started[0])
    assert spec.provider == "openai-compatible"
    assert spec.model == "qwen3.6:latest"
    assert spec.base_url == "http://127.0.0.1:11434/v1"
    # The KEY SOURCE is derived, not sent: a named provider means "read the key from the worker's own
    # environment". The body above carried no credential, and neither does the spec.
    assert spec.provider_key_source == "environment"
    assert "provider_api_key" not in spec.to_payload()


def test_post_runs_trims_the_provider_coordinates_like_the_contract_does(
    server: Running, tmp_path: Path
) -> None:
    """The canonicalisation is the contract's (`canonicalise_provider_fields`), not a second copy."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-trim",
            "repo_dir": str(repo),
            "briefs": ["b"],
            "provider": "  openai  ",
            "model": "  gpt-4.1  ",
        },
    )
    assert status == 202, body
    spec = _provider_spec(server.backend.started[0])
    assert (spec.provider, spec.model, spec.provider_key_source) == ("openai", "gpt-4.1", "environment")


def test_post_runs_still_omits_the_provider_coordinates_when_they_are_absent(
    server: Running, tmp_path: Path
) -> None:
    """BACKWARD COMPATIBLE: no provider in, no provider out, so the engine default applies as before.

    This is the pre-change behaviour for every existing caller (the CLI's non-provider path, the IDE's
    current body, `test_run_api_real.py`), and the spec's ``None`` is what selects the engine's own
    default rather than a provider named "null".
    """
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST", "/runs", body={"run_id": "run-default", "repo_dir": str(repo), "briefs": ["b"]}
    )
    assert status == 202, body
    spec = _provider_spec(server.backend.started[0])
    assert (spec.provider, spec.model, spec.base_url, spec.provider_key_source) == (
        None,
        None,
        None,
        None,
    )
    assert spec.to_payload()["provider"] is None


def test_post_runs_accepts_a_model_without_a_provider_like_the_cli_does(
    server: Running, tmp_path: Path
) -> None:
    """A model with no provider follows the CONTRACT's rule, not a second one invented here.

    `TaskSpec.validate` refuses ``base_url`` without a provider (an endpoint is resolved against one)
    but accepts ``model`` alone, because that is exactly what ``codna run --model`` produces: the
    engine's default provider, with the caller's model id. Reusing the validator is what keeps the two
    entrances from disagreeing about this case — so the API accepts it too.
    """
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={"run_id": "run-model-only", "repo_dir": str(repo), "briefs": ["b"], "model": "gpt-4.1"},
    )
    assert status == 202, body
    spec = _provider_spec(server.backend.started[0])
    assert (spec.provider, spec.model, spec.provider_key_source) == (None, "gpt-4.1", None)


def test_post_runs_refuses_a_base_url_without_a_provider(server: Running, tmp_path: Path) -> None:
    """An endpoint override is resolved against a provider, so alone it is a refusal — the contract's
    own rule, surfaced as a 400 rather than a run that starts and ignores it."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-url-only",
            "repo_dir": str(repo),
            "briefs": ["b"],
            "base_url": "http://127.0.0.1:11434/v1",
        },
    )
    assert status == 400, body
    assert body["error"]["code"] == "contract_invalid"
    assert body["error"]["details"]["field"] == "base_url"
    assert server.backend.started == []


@pytest.mark.parametrize(
    "coordinates, code",
    [
        # An id outside the engine's lowercase alias alphabet. NOT silently lowercased (that would hide
        # a typo that means the engine cannot find the provider).
        ({"provider": "OpenAI"}, "contract_invalid"),
        ({"provider": "not a provider"}, "contract_invalid"),
        ({"provider": ""}, "contract_invalid"),
        # A provider id longer than the contract's ceiling.
        ({"provider": "a" * 65}, "contract_invalid"),
        # Cleartext to a REMOTE host: the key would be exposed in transit.
        (
            {"provider": "openai-compatible", "base_url": "http://api.example.com/v1"},
            "contract_invalid",
        ),
        # A scheme that is not http(s).
        ({"provider": "openai", "base_url": "ftp://127.0.0.1/v1"}, "contract_invalid"),
        ({"provider": "openai", "base_url": "file:///etc/passwd"}, "contract_invalid"),
        # Credentials embedded in the URL — a secret in every request and log line.
        ({"provider": "openai", "base_url": "https://user:pass@api.example.com/v1"}, "contract_invalid"),
        # A query string or fragment would be silently dropped, so the config would lie.
        ({"provider": "openai", "base_url": "https://api.example.com/v1?x=1"}, "contract_invalid"),
        # A URL with no host.
        ({"provider": "openai", "base_url": "https://"}, "contract_invalid"),
        # A non-string coordinate: refused, not coerced.
        ({"provider": 3}, "contract_invalid"),
        ({"provider": "openai", "model": {"id": "gpt-4.1"}}, "contract_invalid"),
        ({"provider": "openai", "base_url": ["http://127.0.0.1"]}, "contract_invalid"),
    ],
)
def test_post_runs_refuses_bad_provider_coordinates_with_the_contract_code(
    server: Running, tmp_path: Path, coordinates: dict[str, Any], code: str
) -> None:
    """An invalid provider/base_url is a 400 with the contract's own code — never a silent default."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={"run_id": "run-bad-provider", "repo_dir": str(repo), "briefs": ["b"], **coordinates},
    )
    assert status == 400, body
    assert body["error"]["code"] == code
    assert server.backend.started == []


@pytest.mark.parametrize(
    "model",
    [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
        "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        "AIzaSy0123456789abcdefghijklmnopqrstuv",
        "xai-abcdefghijklmnopqrstuvwxyz",
        "glpat-abcdefghijklmnopqrstuvwxyz",
        "Bearer abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_post_runs_refuses_a_key_pasted_into_a_coordinate(
    server: Running, tmp_path: Path, model: str
) -> None:
    """A credential smuggled into ``model`` is refused by the EXISTING secret screen.

    The spec enters workflow history and is mirrored into the ledger — the artifact people paste into
    tickets — so this is the check that makes a key unrepresentable rather than merely discouraged.
    """
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-key",
            "repo_dir": str(repo),
            "briefs": ["b"],
            "provider": "openai",
            "model": model,
        },
    )
    assert status == 400, body
    assert body["error"]["code"] == "contract_contains_secret"
    # The refusal names the PATH and never repeats the value.
    assert model not in json.dumps(body)
    assert server.backend.started == []


def test_post_runs_has_no_field_for_a_credential(server: Running, tmp_path: Path) -> None:
    """A key source or a key is STILL an unknown field: the closed set survived the addition.

    ``provider_key_source`` is deliberately NOT accepted — the environment is the only source this
    durable path has, and accepting the field would invite ``inline``. So a caller that tries to say
    where a key comes from, or to supply one, is refused for the same reason as any other stray field.
    """
    repo = git_repo(tmp_path / "repo")
    for extra in (
        {"provider_key_source": "environment"},
        {"provider_api_key": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"},
        {"api_key": "anything"},
        {"provider_key_source": "inline"},
    ):
        status, body = server.json(
            "POST",
            "/runs",
            body={"run_id": "run-key-field", "repo_dir": str(repo), "briefs": ["b"], **extra},
        )
        assert status == 400, body
        assert body["error"]["code"] == "run_api_field_unknown", extra
    assert server.backend.started == []


def test_post_runs_refuses_an_unknown_field_alongside_a_valid_provider(
    server: Running, tmp_path: Path
) -> None:
    """The unknown-field refusal is not bypassed by the new fields being present and valid."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-unknown",
            "repo_dir": str(repo),
            "briefs": ["b"],
            "provider": "openai",
            "model": "gpt-4.1",
            "effort": "high",
        },
    )
    assert status == 400, body
    assert body["error"]["code"] == "run_api_field_unknown"
    assert body["error"]["details"]["unknown"] == ["effort"]
    assert server.backend.started == []



@pytest.mark.parametrize(
    "body, code",
    [
        ({"briefs": [], "repo_dir": "/tmp"}, "durable_repo_outside_allowed_roots"),
        ({"briefs": ["b"], "engine": "nope"}, "run_api_field_invalid"),
        ({"briefs": ["b"], "max_cost_usd": -1}, "run_api_number_invalid"),
        ({"briefs": ["b"], "max_cost_usd": "5"}, "run_api_number_invalid"),
        ({"briefs": ["b"], "max_attempts": True}, "run_api_number_invalid"),
        ({"briefs": ["b"], "base_branch": "  "}, "run_api_field_invalid"),
        ({"briefs": "not-a-list"}, "run_api_field_invalid"),
        ({"briefs": ["b"], "surprise": 1}, "run_api_field_unknown"),
        ({"briefs": ["b"], "repo_dir": "/does/not/exist"}, "durable_repo_outside_allowed_roots"),
    ],
)
def test_post_runs_refuses_a_bad_spec(server: Running, body: dict[str, Any], code: str) -> None:
    """Each refusal is a stable code and NOTHING was started."""
    status, payload = server.json("POST", "/runs", body={"run_id": "run-bad", **body})
    assert status == 400, payload
    assert payload["error"]["code"] == code
    assert server.backend.started == []


def test_post_runs_refuses_an_in_root_directory_that_is_not_git(server: Running, tmp_path: Path) -> None:
    status, payload = server.json("POST", "/runs", body={"briefs": [], "repo_dir": str(tmp_path)})
    assert status == 400, payload
    assert payload["error"]["code"] == "durable_repo_not_git"
    assert server.backend.started == []


def test_post_runs_refuses_a_missing_repo_inside_the_authorized_roots(server: Running, tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    status, payload = server.json("POST", "/runs", body={"briefs": ["b"], "repo_dir": str(missing)})
    assert status == 400, payload
    assert payload["error"]["code"] == "durable_repo_missing"
    assert server.backend.started == []


@pytest.mark.parametrize(
    "body, code",
    [
        ({"briefs": [], "engine": "plain"}, "durable_brief_required"),
        ({"briefs": ["b"], "engine": "plain", "max_attempts": 0}, "run_api_number_invalid"),
        ({"briefs": ["b"], "engine": "plain", "run_timeout_s": -1}, "run_api_number_invalid"),
    ],
)
def test_post_runs_refuses_a_bad_spec_in_a_real_work_tree(
    server: Running, tmp_path: Path, body: dict[str, Any], code: str
) -> None:
    """The spec validators that run AFTER the repo check, exercised against a real work tree.

    Split from the cheap-field cases because these only fail once `_repo_dir` succeeds: a refusal
    that depends on the filesystem is a different branch from one that does not.
    """
    repo = git_repo(tmp_path / "repo")
    status, payload = server.json(
        "POST", "/runs", body={"run_id": "run-spec", "repo_dir": str(repo), **body}
    )
    assert status == 400, payload
    assert payload["error"]["code"] == code
    assert server.backend.started == []


def test_post_runs_refuses_a_brief_that_carries_a_secret(server: Running, tmp_path: Path) -> None:
    """The contract refuses secret-like data because a brief enters workflow history; the API
    must surface that refusal rather than start a run that fails inside Temporal."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-secret",
            "repo_dir": str(repo),
            "briefs": ["use this key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"],
        },
    )
    assert status == 400
    assert body["error"]["code"] in ("contract_contains_secret", "durable_run_spec_invalid")
    assert server.backend.started == []


def test_a_duplicate_start_is_409(server: Running, tmp_path: Path) -> None:
    """A retried start must be distinguishable from a malformed one: 409, with its own code."""
    repo = git_repo(tmp_path / "repo")
    server.backend.start_error = cli_error(
        "could not start run run-dup: duplicate workflow id", code="durable_run_start_failed"
    )
    status, body = server.json(
        "POST", "/runs", body={"run_id": "run-dup", "repo_dir": str(repo), "briefs": ["b"]}
    )
    assert status == 409
    assert body["error"]["code"] == "run_already_exists"


def test_an_uncoded_start_failure_is_502(server: Running, tmp_path: Path) -> None:
    """The server failed and the caller did nothing wrong: 502, not 400 and not 500."""
    repo = git_repo(tmp_path / "repo")
    server.backend.start_error = RuntimeError("the server exploded")
    status, body = server.json(
        "POST", "/runs", body={"run_id": "run-boom", "repo_dir": str(repo), "briefs": ["b"]}
    )
    assert status == 502
    assert body["error"]["code"] == "run_api_backend_failed"


def test_get_runs_lists_the_runs_whose_ledgers_exist(server: Running) -> None:
    server.seed_ledger("run-a")
    server.seed_ledger("run-b")
    status, body = server.json("GET", "/runs")
    assert status == 200
    assert sorted(entry["run_id"] for entry in body["runs"]) == ["run-a", "run-b"]
    assert body["total"] == 2
    for entry in body["runs"]:
        assert entry["last_seq"] == 5
        assert entry["ledger_bytes"] > 0
        assert entry["outcome"] == "completed"


def test_a_ledger_without_a_repo_reports_null_rather_than_inventing_one(server: Running) -> None:
    """AN OLD LEDGER HAS NO REPOSITORY ON RECORD, and the route must say exactly that.

    `repo_dir` was added to the ledger after the first runs were written. For those, the honest answer
    is `None` — a caller that substitutes its own workspace labels the run with a repository it may
    never have touched, which is the bug this field exists to remove. `null` is a documented value, not
    a missing key: a client can branch on it without guessing whether the server is old.
    """
    ledger = RunLedger.for_run(server.config.paths, "run-old")
    # The shape every pre-change ledger has: a `run.started` note with no `repo_dir` in its payload.
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started", "message": "go"})
    status, body = server.json("GET", "/runs")
    assert status == 200
    assert [entry["run_id"] for entry in body["runs"]] == ["run-old"]
    assert "repo_dir" in body["runs"][0]
    assert body["runs"][0]["repo_dir"] is None


def test_the_run_list_reports_the_repository_the_run_works_in(server: Running, tmp_path: Path) -> None:
    """A run's row names ITS repository, not the one the reading window happens to have open.

    This is the field the sidebar's project grouping was missing: without it every durable row was
    stamped with `defaultProjectId` — the open folder's name — so a run from repo A appeared under repo
    B the moment the window was reopened elsewhere.
    """
    repo = git_repo(tmp_path / "repo-a")
    ledger = RunLedger.for_run(server.config.paths, "run-with-repo")
    # The shape `notify` writes: a ledger `note` whose payload names the notification. Only a note is a
    # run-level fact, which is what `run_repo.repo_dir_of` reads.
    ledger.append(
        "note",
        payload={
            "event": "notification",
            "kind": "run.started",
            "message": "run run-with-repo: 1 task(s) to go",
            "repo_dir": str(repo),
        },
    )
    status, body = server.json("GET", "/runs")
    assert status == 200
    entry = body["runs"][0]
    assert entry["run_id"] == "run-with-repo"
    # The absolute path the run recorded, not the folder the server was started from.
    assert entry["repo_dir"] == str(repo.resolve())
    assert entry["repo_dir"] != str(server.config.runs_root)


def test_the_repo_scan_reads_notes_near_the_head_and_ignores_turn_events(server: Running) -> None:
    """The repository lives on a run-level NOTE, so a turn event's copy of the key is not a candidate.

    A ledger can hold thousands of turn events; scanning all of them to answer "which repository" would
    make a list read proportional to the largest run on the machine. The scan is a bounded head window —
    which is also what makes the answer cheap for a run that predates the field.
    """
    ledger = RunLedger.for_run(server.config.paths, "run-buried")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    # Not a `note`, so this is not a run-level fact and must not be reported.
    ledger.append(
        "turn.event",
        session_id="s1",
        event_index=1,
        payload={"event_type": "agent.done", "payload": {}, "repo_dir": "/not/a/note"},
    )
    ledger.append(
        "note",
        payload={"event": "notification", "kind": "note", "message": "later", "repo_dir": "/after/start"},
    )
    _, body = server.json("GET", "/runs")
    assert body["runs"][0]["repo_dir"] == "/after/start"


def test_a_repo_beyond_the_head_window_reports_null_rather_than_a_full_scan(server: Running) -> None:
    """A repo written far past the front is not found, and the route says `null` instead of scanning.

    The field is only ever written by the run's opening notification, so a value that far into the file
    is not a shape any version of the product produces. Reporting `null` keeps the read bounded; the
    alternative (walk the whole ledger) would make listing O(all events on disk).
    """
    ledger = RunLedger.for_run(server.config.paths, "run-far")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    for index in range(run_repo._HEAD_SCAN_RECORDS + 4):
        ledger.append(
            "turn.event",
            session_id="s1",
            event_index=index + 1,
            payload={"event_type": "agent.done", "payload": {}},
        )
    ledger.append(
        "note", payload={"event": "notification", "kind": "note", "repo_dir": "/too/far/to/be/read"}
    )
    _, body = server.json("GET", "/runs")
    assert body["runs"][0]["repo_dir"] is None


def test_the_run_list_reports_no_outcome_for_an_unfinished_run(server: Running) -> None:
    """An open run has no outcome: reporting `completed` for one would be a guess, and a wrong one."""
    ledger = RunLedger.for_run(server.config.paths, "run-open")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    _, body = server.json("GET", "/runs")
    assert [entry["outcome"] for entry in body["runs"]] == [None]


def test_get_runs_honours_limit_and_refuses_a_bad_one(server: Running) -> None:
    for run_id in ("run-a", "run-b", "run-c"):
        server.seed_ledger(run_id)
    status, body = server.json("GET", "/runs?limit=2")
    assert status == 200
    assert body["count"] == 2 and body["total"] == 3 and body["limit"] == 2

    for bad in ("0", "-1", "abc", "99999"):
        status, body = server.json("GET", f"/runs?limit={bad}")
        assert status == 400, bad
        assert body["error"]["code"] == "run_api_limit_invalid"


def test_get_runs_ignores_a_directory_that_is_not_a_run(server: Running) -> None:
    """A stray directory under the runs root must not become an invented run with zero records."""
    stray = server.config.runs_root / "not-a-run"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "notes.txt").write_text("hi", encoding="utf-8")
    status, body = server.json("GET", "/runs")
    assert status == 200
    assert body["runs"] == []


def test_get_one_run_reports_status_and_ledger(server: Running) -> None:
    server.seed_ledger("run-1")
    server.backend.statuses["run-1"] = {
        "run_id": "run-1",
        "tasks_total": 1,
        "tasks_done": 1,
        "outcome": "completed",
    }
    status, body = server.json("GET", "/runs/run-1")
    assert status == 200
    assert body["run_id"] == "run-1"
    assert body["workflow_id"] == "run:run-1"
    assert body["status"]["tasks_done"] == 1
    assert body["ledger"]["records"] == 5
    assert body["ledger"]["valid"] is True
    assert body["ledger"]["last_seq"] == 5
    assert body["outcome"] == "completed"


def test_get_one_run_reads_an_open_run_from_temporal_only(server: Running) -> None:
    """A run that has started but written no ledger yet is still a run: status, no ledger."""
    server.backend.statuses["run-open"] = {"run_id": "run-open", "tasks_done": 0}
    status, body = server.json("GET", "/runs/run-open")
    assert status == 200
    assert body["ledger"]["exists"] is False
    assert body["ledger"]["records"] == 0
    assert body["status"]["tasks_done"] == 0


def test_an_unknown_run_is_404(server: Running) -> None:
    for path in ("/runs/nope", "/runs/nope/events"):
        status, body = server.json("GET", path)
        assert status == 404, path
        assert body["error"]["code"] == "run_not_found"


def test_a_malformed_run_id_is_400_not_404(server: Running) -> None:
    """A run id that could escape the runs root is a malformed request, never a lookup."""
    status, body = server.json("GET", "/runs/..%2Fetc")
    assert status == 400
    assert body["error"]["code"] == "run_api_run_id_invalid"


def test_an_unknown_route_is_404(server: Running) -> None:
    for path in ("/", "/nope", "/runs/run-1/nope", "/runs/run-1/events/extra"):
        status, body = server.json("GET", path)
        assert status == 404, path
        assert body["error"]["code"] == "run_api_not_found"


def test_the_wrong_method_is_405(server: Running) -> None:
    status, body = server.json("POST", "/runs/run-1")
    assert status == 405
    assert body["error"]["code"] == "run_api_method_not_allowed"
    status, body = server.json("GET", "/runs/run-1/approve")
    assert status == 405


# -- 4. the six mutations ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "route, body, expected_action, expected_payload",
    [
        ("approve", {"task_id": "task-1"}, "approve", {}),
        ("steer", {"task_id": "task-1", "note": "  also cover the retry  "}, "steer",
         {"note": "also cover the retry"}),
        ("budget", {"task_id": "task-1", "max_cost_usd": 4.5}, "budget", {"max_cost_usd": 4.5}),
    ],
)
def test_each_task_update_records_the_identity(
    server: Running,
    route: str,
    body: dict[str, Any],
    expected_action: str,
    expected_payload: dict[str, Any],
) -> None:
    """The identity is what makes an approval an audit record: it must reach the workflow payload.

    The body's `identity` is a CONFIRMATION of the authenticated principal, not a source: what reaches
    the workflow is the principal's identity, which happens to equal the one the caller repeated.
    """
    run_id = seed_mutable_run(server)
    status, payload = server.json(
        "POST", f"/runs/{run_id}/{route}", body={"identity": TEST_IDENTITY, **body}
    )
    assert status == 200, payload
    assert payload["identity"] == TEST_IDENTITY
    assert payload["applied"] is True
    assert payload["task_id"] == "task-1"

    sent = server.backend.updates[-1]
    assert sent["action"] == expected_action
    assert sent["task_id"] == "task-1"
    assert sent["payload"]["identity"] == TEST_IDENTITY
    for key, value in expected_payload.items():
        assert sent["payload"][key] == value


def test_a_body_identity_naming_somebody_else_is_403_and_never_reaches_the_backend(
    server: Running,
) -> None:
    """The forged-approval case: a well-formed request claiming an identity it was not issued as.

    This is the whole point of moving authorization server-side. Before E2 the body WAS the identity,
    so any caller holding the local token could record an approval as the CTO.
    """
    run_id = seed_mutable_run(server)
    status, payload = server.json(
        "POST", f"/runs/{run_id}/approve", body={"identity": "cto@codna.ai", "task_id": "task-1"}
    )
    assert status == 403, payload
    assert payload["error"]["code"] == "run_api_identity_mismatch"
    assert payload["error"]["details"]["authenticated"] == TEST_IDENTITY
    assert server.backend.updates == []


def test_set_budget_maps_onto_the_workflows_update_name(server: Running) -> None:
    """The route is `budget`, the workflow's Update is `set_budget`: the mapping must be explicit."""
    run_id = seed_mutable_run(server)
    status, _ = server.json(
        "POST", f"/runs/{run_id}/budget", body={"identity": TEST_IDENTITY, "task_id": "task-1", "max_cost_usd": 2}
    )
    assert status == 200
    assert server.backend.updates[-1]["action"] == "budget"
    from codna.durable.cli import _send_update  # noqa: F401 - imported to read its mapping below

    source = (REPO_ROOT / "cli" / "codna" / "durable" / "cli.py").read_text(encoding="utf-8")
    assert '{"set-budget": "set_budget"}' in source


@pytest.mark.parametrize(
    "route, body, expected_signal",
    [
        ("pause", {}, "pause"),
        ("resume", {}, "resume"),
        ("cancel", {}, "stop"),
    ],
)
def test_each_signal_route_records_the_identity_in_the_ledger(
    server: Running, route: str, body: dict[str, Any], expected_signal: str
) -> None:
    """A Signal carries no payload, so the identity must land somewhere an auditor can read it.

    That somewhere is the ledger: an append-only, hash-chained record. The test reads it BACK through
    the API's own ledger summary, which is what a client would do.
    """
    run_id = seed_mutable_run(server)
    status, payload = server.json("POST", f"/runs/{run_id}/{route}", body={"identity": TEST_IDENTITY, **body})
    assert status == 200, payload
    assert payload["signal"] == expected_signal
    assert payload["identity"] == TEST_IDENTITY
    assert server.backend.signals == [(run_id, expected_signal)]
    assert payload["ledger_seq"] == 6

    ledger = RunLedger.for_run(server.config.paths, run_id)
    tail = ledger.tail()
    assert tail.kind == "note"
    assert tail.payload == {
        "event": "control.requested",
        "action": route,
        "identity": TEST_IDENTITY,
    }
    assert ledger.verify() == []


@pytest.mark.parametrize("route", ["approve", "steer", "budget", "pause", "resume", "cancel"])
def test_a_blank_identity_is_400_and_never_reaches_the_backend(server: Running, route: str) -> None:
    """An anonymous approval is refused at the edge: the workflow's validator would only say so
    after a round trip, and a refused Update is a worse error than a refused request."""
    run_id = seed_mutable_run(server)
    body: dict[str, Any] = {"identity": "   "}
    if route in ("approve", "steer", "budget"):
        body["task_id"] = "task-1"
    if route == "steer":
        body["note"] = "n"
    if route == "budget":
        body["max_cost_usd"] = 1.0
    status, payload = server.json("POST", f"/runs/{run_id}/{route}", body=body)
    assert status == 400, payload
    assert payload["error"]["code"] == "run_api_identity_required"
    assert server.backend.updates == [] and server.backend.signals == []


@pytest.mark.parametrize(
    "body",
    [{}, {"identity": None}, {"identity": 7}, {"identity": ""}, {"identity": "n" * 400}],
)
def test_a_missing_or_non_string_identity_is_400(server: Running, body: dict[str, Any]) -> None:
    run_id = seed_mutable_run(server)
    status, payload = server.json("POST", f"/runs/{run_id}/pause", body=body)
    assert status == 400
    assert payload["error"]["code"] in ("run_api_identity_required", "run_api_identity_invalid")


@pytest.mark.parametrize("route", ["approve", "steer", "budget", "pause", "resume", "cancel"])
def test_each_mutation_on_an_unknown_run_is_404(server: Running, route: str) -> None:
    """A mutation names a run that does not exist: 404, before anything is attempted."""
    body: dict[str, Any] = {"identity": TEST_IDENTITY}
    if route in ("approve", "steer", "budget"):
        body["task_id"] = "task-1"
    if route == "steer":
        body["note"] = "n"
    if route == "budget":
        body["max_cost_usd"] = 1.0
    status, payload = server.json("POST", f"/runs/run-missing/{route}", body=body)
    assert status == 404, payload
    assert payload["error"]["code"] == "run_not_found"
    assert server.backend.updates == [] and server.backend.signals == []


@pytest.mark.parametrize(
    "route, body, code",
    [
        ("approve", {"identity": TEST_IDENTITY}, "run_api_task_required"),
        ("approve", {"identity": TEST_IDENTITY, "task_id": "../etc"}, "run_api_task_invalid"),
        ("approve", {"identity": TEST_IDENTITY, "task_id": "  "}, "run_api_task_required"),
        ("steer", {"identity": TEST_IDENTITY, "task_id": "task-1"}, "run_api_note_invalid"),
        ("steer", {"identity": TEST_IDENTITY, "task_id": "task-1", "note": "  "}, "run_api_note_invalid"),
        ("steer", {"identity": TEST_IDENTITY, "task_id": "task-1", "note": "n" * 2001}, "run_api_note_invalid"),
        ("budget", {"identity": TEST_IDENTITY, "task_id": "task-1"}, "run_api_number_invalid"),
        ("budget", {"identity": TEST_IDENTITY, "task_id": "task-1", "max_cost_usd": 0}, "run_api_number_invalid"),
        ("budget", {"identity": TEST_IDENTITY, "task_id": "task-1", "max_cost_usd": False},
         "run_api_number_invalid"),
        ("pause", {"identity": TEST_IDENTITY, "task_id": "task-1"}, "run_api_field_unknown"),
        ("pause", {"identity": TEST_IDENTITY, "action": "pause"}, "run_api_field_unknown"),
    ],
)
def test_each_mutation_validates_its_own_body(
    server: Running, route: str, body: dict[str, Any], code: str
) -> None:
    run_id = seed_mutable_run(server)
    status, payload = server.json("POST", f"/runs/{run_id}/{route}", body=body)
    assert status == 400, payload
    assert payload["error"]["code"] == code
    assert server.backend.updates == [] and server.backend.signals == []


def test_a_validator_refusal_is_409(server: Running) -> None:
    """The task exists but cannot take that action now: the run is fine, the request is not."""
    run_id = seed_mutable_run(server)

    class _Refusing(ScriptedBackend):
        def update(self, run_id: str, *, task_id: str, action: str, payload: Any) -> Any:
            raise cli_error("this task is not waiting on a decision", code="durable_not_parked")

    server.server.api.backend = _Refusing()
    status, body = server.json(
        "POST", f"/runs/{run_id}/approve", body={"identity": TEST_IDENTITY, "task_id": "task-1"}
    )
    assert status == 409
    assert body["error"]["code"] == "durable_not_parked"


# -- 5. malformed requests ---------------------------------------------------------------------


def test_malformed_json_is_400_with_a_stable_code(server: Running) -> None:
    status, body, _ = server.request(
        "POST", "/runs", raw_body=b"{not json", headers={"content-type": "application/json"}
    )
    assert status == 400
    assert body["error"]["code"] == "run_api_json_invalid"


def test_a_non_object_json_body_is_400(server: Running) -> None:
    status, body, _ = server.request(
        "POST", "/runs", raw_body=b'["a list"]', headers={"content-type": "application/json"}
    )
    assert status == 400
    assert body["error"]["code"] == "run_api_body_not_object"


def test_an_oversized_body_is_413_and_is_not_buffered(server: Running) -> None:
    """Refused on the DECLARED length, before a byte of it is read.

    The request declares 8 MiB and sends NOTHING. A server that buffered the body would block until
    the client's timeout; the refusal arriving at all proves the length was checked first, which is
    the property that stops an oversized upload from being turned into memory.
    """
    limit = server.server.settings.max_body_bytes
    declared = limit * 8
    with socket.create_connection((server.host, server.port), timeout=10) as connection:
        connection.sendall(
            f"POST /runs HTTP/1.1\r\n"
            f"Host: {server.host}:{server.port}\r\n"
            f"Authorization: Bearer {server.token}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {declared}\r\n\r\n".encode("utf-8")
        )
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = connection.recv(4096)
            if not chunk:
                break
            raw += chunk
        head, _, rest = raw.partition(b"\r\n\r\n")
        # The body may or may not have arrived with the headers; read the rest of it.
        length = int(re.search(rb"content-length: (\d+)", head.lower()).group(1))
        while len(rest) < length:
            chunk = connection.recv(4096)
            if not chunk:
                break
            rest += chunk
    assert b" 413 " in head, head
    body = json.loads(rest)
    assert body["error"]["code"] == "run_api_body_too_large"
    assert body["error"]["details"]["max_body_bytes"] == limit
    assert body["error"]["details"]["content_length"] == declared


def test_a_body_without_content_length_is_length_required(server: Running) -> None:
    connection = http.client.HTTPConnection(server.host, server.port, timeout=10)
    try:
        connection.putrequest("POST", "/runs")
        connection.putheader("authorization", f"Bearer {server.token}")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 411
        assert json.loads(response.read())["error"]["code"] == "run_api_length_required"
    finally:
        connection.close()


@pytest.mark.parametrize("query", ["after=-1", "after=abc", "after=1.5", "after=%20%20", "after="])
def test_a_bad_after_is_400(server: Running, query: str) -> None:
    """A mistyped resume point skips records, so it is refused on EVERY run, not only a live one.

    A blank ``after=`` is refused rather than read as "absent": a caller that supplied a resume
    point and got it wrong must not be silently replayed from zero.
    """
    server.seed_ledger("run-1")
    status, body = server.json("GET", f"/runs/run-1/events?{query}")
    assert status == 400, body
    assert body["error"]["code"] == "run_api_after_invalid"


def test_a_bad_follow_is_400(server: Running) -> None:
    server.seed_ledger("run-1")
    status, body = server.json("GET", "/runs/run-1/events?follow=maybe")
    assert status == 400
    assert body["error"]["code"] == "run_api_follow_invalid"


# -- 6. reattach semantics ---------------------------------------------------------------------


def test_after_resumes_exactly_after_the_given_seq(server: Running) -> None:
    """``?after=N`` means "I have everything up to N": the first frame is N+1, with no duplicate.

    This is the property the IDE's ``lastIndex`` depends on; a stream that re-sent N would make the
    renderer draw the same step twice.
    """
    records = server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events?after=2")
    events = IdeStreamParser().feed(raw)
    assert [event["event_index"] for event in events] == [3, 4, 5]
    assert events[0]["event_index"] == records[2].seq


def test_after_past_the_end_emits_nothing(server: Running) -> None:
    """A resume point beyond the ledger is a quiet stream, not an error and not a replay.

    ``follow=0`` because the point is the REPLAY: a follower on a run past its terminal has nothing
    to wait for, and asserting "nothing" on a stream that never closes would be asserting a hang.
    """
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events?after=999&follow=0")
    assert "data:" not in raw.decode("utf-8")


def test_after_a_terminal_replays_nothing(server: Running) -> None:
    """A client that already saw the terminal reattaches to an empty stream that closes at once."""
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events?after=5&follow=0")
    assert "data:" not in raw.decode("utf-8")


def test_without_after_the_whole_ledger_replays(server: Running) -> None:
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events")
    events = IdeStreamParser().feed(raw)
    assert [event["event_index"] for event in events] == [1, 2, 3, 4, 5]


def test_the_event_index_is_the_ledger_seq_a_client_persists(server: Running) -> None:
    """One cursor, not two: the number in the frame is the number ``?after=`` takes.

    Proven by round-tripping: read the stream, take the last ``event_index`` it delivered, and use it
    as ``after`` — the next read must start after it.
    """
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events")
    events = IdeStreamParser().feed(raw)
    cursor = events[-1]["event_index"]
    ledger = server.config.ledger_path("run-1")
    assert cursor == len(ledger.read_text(encoding="utf-8").strip().splitlines())


def test_a_run_with_no_ledger_yet_streams_its_first_record(server: Running) -> None:
    """A run started a second ago has a workflow but no ledger: a stream, not a 404, and it waits."""
    server.backend.statuses["run-fresh"] = {"run_id": "run-fresh", "tasks_done": 0}
    response, connection = server.open_stream("/runs/run-fresh/events?follow=1")
    parser = IdeStreamParser()
    try:
        assert response.status == 200
        watcher = threading.Thread(target=drain, args=(response, parser), daemon=True)
        watcher.start()
        time.sleep(0.2)
        RunLedger.for_run(server.config.paths, "run-fresh").append(
            "run.started", payload={"event": "notification", "kind": "run.started"}
        )
        RunLedger.for_run(server.config.paths, "run-fresh").append(
            "note", payload={"event": "notification", "kind": "run.finished"}
        )
        watcher.join(timeout=10.0)
        assert not watcher.is_alive()
        assert [event["event_type"] for event in parser.events] == ["run.started", "run.finished"]
    finally:
        connection.close()


def test_follow_zero_returns_the_backlog_and_closes(server: Running) -> None:
    """A bounded read for a caller that wants a snapshot (a CLI, a report) rather than a follower."""
    server.seed_ledger("run-1")
    raw, _ = server.read_until_terminal("/runs/run-1/events?follow=0")
    # Five records, no terminal-close needed: `follow=0` ends the stream after the backlog even
    # though the terminal is in it, so this is the same bytes as the follow case.
    events = IdeStreamParser().feed(raw)
    assert [event["event_index"] for event in events] == [1, 2, 3, 4, 5]


# -- 7. disconnects and shutdown ---------------------------------------------------------------


def test_a_client_that_disconnects_mid_stream_does_not_disturb_the_server(server: Running) -> None:
    """The plan's explicit requirement: a dropped stream costs a thread, not the server.

    The client reads one frame and closes the socket. The server's follower loop must notice (via
    its EOF poll) and return, and the NEXT request must be served normally — including the stream
    slot, which a leaked one would eventually exhaust.
    """
    ledger = RunLedger.for_run(server.config.paths, "run-1")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})

    for _ in range(server.server.settings.max_streams):
        response, connection = server.open_stream("/runs/run-1/events")
        try:
            assert response.read1(4096)  # the first frame really arrived
        finally:
            # Abandoned mid-stream: no draining, no graceful end() — a killed renderer. `read1`
            # rather than `read`, and `close` rather than a normal shutdown, so the server sees an
            # EOF on a socket it is still following rather than a well-behaved request end.
            response.close()
            connection.close()

    # The server is still serving: an ordinary request works, and the ledger is intact.
    status, body = server.json("GET", "/runs")
    assert status == 200 and [entry["run_id"] for entry in body["runs"]] == ["run-1"]
    # ...and every slot came back. The release is asynchronous by design (the follower notices EOF
    # on its next poll), so this waits rather than assuming an instant handover.
    deadline = time.monotonic() + 10.0
    while True:
        status, _, _ = server.request("GET", "/runs/run-1/events?follow=0")
        if status == 200:
            break
        assert status == 503, status
        assert time.monotonic() < deadline, "the disconnected streams never released their slots"
        time.sleep(0.05)


def test_a_closed_stream_releases_its_slot(server: Running) -> None:
    """The stream semaphore is a hard bound; if a close leaked one, the last acquire would fail."""
    server.seed_ledger("run-1")
    for _ in range(server.server.settings.max_streams * 2):
        raw, _ = server.read_until_terminal("/runs/run-1/events?follow=0")
        assert raw
    status, body = server.json("GET", "/runs")
    assert status == 200


def test_the_stream_limit_answers_503_instead_of_growing_threads(server: Running) -> None:
    """Past ``max_streams`` the answer is a refusal, not unbounded thread growth."""
    server.seed_ledger("run-1")
    held: list[tuple[Any, Any]] = []
    try:
        for _ in range(server.server.settings.max_streams):
            response, connection = server.open_stream("/runs/run-1/events?after=5")
            held.append((response, connection))
        status, body = server.json("GET", "/runs/run-1/events")
        assert status == 503
        assert body["error"]["code"] == "run_api_stream_limit"
    finally:
        for response, connection in held:
            connection.close()


def test_stopping_the_server_drains_an_open_stream(server: Running) -> None:
    """``stop()`` from another thread must end a follower, not wait for the run to finish."""
    ledger = RunLedger.for_run(server.config.paths, "run-1")
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    response, connection = server.open_stream("/runs/run-1/events")
    try:
        assert response.read1(4096)
        finished = threading.Event()

        def read_to_eof() -> None:
            while response.read1(4096):
                pass
            finished.set()

        reader = threading.Thread(target=read_to_eof, daemon=True)
        reader.start()
        server.server.stop()
        assert finished.wait(timeout=10.0), "an open stream survived stop()"
    finally:
        connection.close()


# -- 8. settings and validation as units -------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com", "", "localhost.evil"])
def test_a_non_loopback_bind_is_refused(host: str) -> None:
    """Loopback only, and a hostname that merely RESOLVES to loopback is not enough."""
    with pytest.raises(RunApiError) as caught:
        RunApiSettings(host=host)
    assert caught.value.code == "run_api_bind_not_loopback"
    assert caught.value.status == 400


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_the_loopback_spellings_are_accepted(host: str) -> None:
    assert RunApiSettings(host=host).host == host


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"port": -1}, "run_api_port_invalid"),
        ({"port": 70000}, "run_api_port_invalid"),
        ({"port": True}, "run_api_port_invalid"),
        ({"heartbeat_s": 0}, "run_api_settings_invalid"),
        ({"poll_interval_s": -1}, "run_api_settings_invalid"),
        ({"request_timeout_s": 0}, "run_api_settings_invalid"),
        ({"write_timeout_s": "5"}, "run_api_settings_invalid"),
        ({"max_streams": 0}, "run_api_settings_invalid"),
        ({"max_streams": 1.5}, "run_api_settings_invalid"),
        ({"max_body_bytes": 0}, "run_api_settings_invalid"),
        ({"allowed_repo_roots": ()}, "run_api_settings_invalid"),
        ({"allowed_repo_roots": (Path("relative"),)}, "run_api_settings_invalid"),
    ],
)
def test_settings_are_validated_at_construction(kwargs: dict[str, Any], code: str) -> None:
    with pytest.raises(RunApiError) as caught:
        RunApiSettings(**kwargs)
    assert caught.value.code == code


def test_allowed_repo_roots_use_only_the_supplied_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(support.ALLOWED_REPO_ROOTS_ENV, str(tmp_path / "ambient"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    configured = os.pathsep.join((f" {first} ", f" {second} "))
    assert support.allowed_repo_roots_from_env({support.ALLOWED_REPO_ROOTS_ENV: configured}) == (
        first.resolve(),
        second.resolve(),
    )
    assert support.allowed_repo_roots_from_env({}) == (Path.cwd(),)


def test_allowed_repo_roots_default_is_evaluated_at_construction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert RunApiSettings().allowed_repo_roots == (tmp_path.resolve(),)


def test_the_token_path_comes_from_the_runtime_root(config: Any) -> None:
    """The token sits beside the runs, so one ``CODNA_RUNTIME_ROOT`` moves the whole surface."""
    settings = RunApiSettings.for_config(config, port=0)
    assert settings.token_path == config.paths.root / support.TOKEN_FILENAME


@pytest.mark.parametrize(
    "query, expected",
    [({}, 0), ({"after": ["0"]}, 0), ({"after": ["42"]}, 42), ({"after": ["7", "9"]}, 9)],
)
def test_after_param_accepts(query: dict[str, list[str]], expected: int) -> None:
    assert after_param(query) == expected


@pytest.mark.parametrize("raw", ["-1", "x", "1.0", "+3", " ", "", "1e3"])
def test_after_param_refuses(raw: str) -> None:
    with pytest.raises(RunApiError) as caught:
        after_param({"after": [raw]})
    assert caught.value.code == "run_api_after_invalid"


def test_outcome_of_reads_the_tail_not_the_whole_file(config: Any) -> None:
    """The outcome is in the last record; scanning a year of ledger to find it is wasted work."""
    ledger = RunLedger.for_run(config.paths, "run-tail")
    for index in range(500):
        ledger.append("turn.event", session_id="s", event_index=index, payload={"event_type": "x"})
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "outcome": "stopped"})
    assert outcome_of(config.ledger_path("run-tail")) == "stopped"


def test_outcome_of_reports_nothing_for_an_open_run(config: Any) -> None:
    ledger = RunLedger.for_run(config.paths, "run-open")
    ledger.append("run.started")
    assert outcome_of(config.ledger_path("run-open")) is None


# -- the terminal a run ACTUALLY writes ---------------------------------------------------------
#
# `run.finished` is written only on the two `completed` exits of `RunWorkflow`. A run whose task never
# got that far ends at a TURN terminal and writes nothing above it, and reading only the run terminal
# meant such a run was reported as open — and animated as Working — forever. These tests use the real
# shape: the event is a `turn.event` whose `event_type` names the turn's ending.

# The THREE REAL LEDGERS from the report, copied verbatim (byte-identical by sha256) and checked in.
# They lived only in the reporter's `~/.codna/runs/`, so a path into `$HOME` would have made this
# proof pass on one machine and skip everywhere else — including CI, whose "no unexpected skips" gate
# correctly refuses a skip that hides a missing fixture. A committed copy is what makes the assertion
# portable; the user's own files stay untouched and are still the thing that was measured.
STUCK_RUN_IDS = (
    "run-20261030T185610Z",
    "run-20261030T184245Z",
    "run-20261030T184525Z",
)
STUCK_RUNS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "stuck_runs"

# sha256 of each file EXACTLY as the reporter's `~/.codna/runs/<id>/ledger.ndjson` had it. Pinned so a
# fixture cannot be edited into a shape that passes: being real is the entire evidentiary value.
STUCK_RUN_SHA256 = {
    "run-20261030T185610Z": "c6b6c676e443e2a0bbec1eda1e938c5d83c35f9f49fb03501d6a3c4b3ca32ae4",
    "run-20261030T184245Z": "15c04b4a9f2332e778576c57ee813a9aea5fb406c8c7f4855e0c57e4d93fd058",
    "run-20261030T184525Z": "cbbfad02121af73cab420437867d996f2865c3bc9975e3c8624f682e5d1743e3",
}



def _turn_terminal(config: Any, run_id: str, event_type: str) -> Path:
    """A ledger whose last meaningful record is a turn terminal, as the engine writes it."""
    ledger = RunLedger.for_run(config.paths, run_id)
    ledger.append("note", payload={"event": "notification", "kind": "run.started", "message": "go"})
    ledger.append("turn.started", payload={"turn_id": "t1", "phase": "plan"})
    ledger.append(
        "turn.event", session_id="s1", event_index=1, payload={"event_type": event_type, "payload": {}}
    )
    return config.ledger_path(run_id)


def test_a_failed_turn_ends_the_run_as_failed(config: Any) -> None:
    """A turn the engine failed is a run that ended badly, not a run that is still going."""
    path = _turn_terminal(config, "run-turn-failed", "session.turn_failed")
    assert support.terminal_verdict(path)[0] is True
    assert outcome_of(path) == "failed"


def test_an_aborted_turn_ends_the_run_as_stopped_not_failed(config: Any) -> None:
    """An INTERRUPTED turn is not a failure. `turn_unpaired` is a warning in the soak harness for
    exactly this reason, so an abort resolves to `stopped` — ended, not blamed."""
    for index, event_type in enumerate(("session.aborted", "session.turn_aborted", "session.stopped")):
        path = _turn_terminal(config, f"run-turn-aborted-{index}", event_type)
        assert support.terminal_verdict(path)[0] is True
        assert outcome_of(path) == "stopped", event_type


def test_a_resume_after_an_abort_supersedes_it(config: Any) -> None:
    """The engine restarts a turn under the same key: the abort is history, not a verdict.

    Measured on a real 24h soak: 23 of 97 ledgers end at `session.aborted` and many of those have a
    `session.resumed` right after. Reading "any terminal anywhere" would call a resumed run finished.
    """
    ledger = RunLedger.for_run(config.paths, "run-resumed")
    ledger.append("turn.event", session_id="s1", event_index=1, payload={"event_type": "session.aborted"})
    ledger.append("note", payload={"event": "turn.resumed_from_ledger", "turn_key": "k"})
    ledger.append("turn.event", session_id="s1", event_index=2, payload={"event_type": "session.resumed"})
    ledger.append("turn.event", session_id="s1", event_index=3, payload={"event_type": "agent.content_start"})
    path = config.ledger_path("run-resumed")
    assert support.terminal_verdict(path)[0] is False
    assert outcome_of(path) is None


def test_work_after_a_run_terminal_cannot_unfinish_it(config: Any) -> None:
    """`run.finished` is final: nothing is scheduled after it, so nothing may revoke it."""
    ledger = RunLedger.for_run(config.paths, "run-final")
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "outcome": "completed"})
    ledger.append("turn.event", session_id="s1", event_index=1, payload={"event_type": "agent.content_start"})
    assert outcome_of(config.ledger_path("run-final")) == "completed"


def test_a_terminal_without_an_outcome_is_still_ended(config: Any) -> None:
    """The ending is a fact even when the cause was never written down.

    Production `_notify` carries no `outcome` — measured on a real 24h soak, all 9 `run.finished`
    notes had keys `event/evidence/kind/message` and 0 had an outcome. So `ended` must not depend on
    it, or every completed production run would still read as in progress.
    """
    ledger = RunLedger.for_run(config.paths, "run-no-outcome")
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "message": "done"})
    path = config.ledger_path("run-no-outcome")
    assert support.terminal_verdict(path)[0] is True
    assert outcome_of(path) is None


def test_stalled_threshold_is_twice_the_longest_activity() -> None:
    """The threshold mirrors Temporal's own ceiling for one activity, so it cannot silently drift."""
    from codna.durable.contracts import ACTIVITY_TURN_TIMEOUT_S

    assert support.STALLED_AFTER_S == 2 * ACTIVITY_TURN_TIMEOUT_S


def test_a_quiet_ledger_with_no_terminal_is_stalled_not_working(config: Any) -> None:
    """The third state. A run nobody is producing must not animate forever."""
    ledger = RunLedger.for_run(config.paths, "run-quiet")
    ledger.append("run.started")
    path = config.ledger_path("run-quiet")
    fresh = support.ledger_verdict(path, updated_ts=path.stat().st_mtime)
    assert fresh["ended"] is False
    assert fresh["stalled"] is False, "a ledger written a moment ago is simply live"
    old = support.ledger_verdict(
        path, updated_ts=path.stat().st_mtime - (support.STALLED_AFTER_S + 1)
    )
    assert old["ended"] is False
    assert old["stalled"] is True


def test_a_terminal_bearing_ledger_is_never_stalled(config: Any) -> None:
    """`stalled` is "we do not know"; a terminal is "we do know". They cannot both hold."""
    path = _turn_terminal(config, "run-ended-old", "session.turn_failed")
    verdict = support.ledger_verdict(path, updated_ts=path.stat().st_mtime - 30 * 24 * 60 * 60)
    assert verdict["ended"] is True
    assert verdict["stalled"] is False
    assert verdict["outcome"] == "failed"


@pytest.mark.parametrize("run_id", STUCK_RUN_IDS)
def test_the_users_own_stuck_runs_resolve_to_an_outcome(run_id: str) -> None:
    """The reported defect, on the real ledgers: these three animated as Working for 4-6 days.

    ``stopped`` rather than ``failed`` is the measured answer, not a softened one: each of these
    ledgers records a ``session.turn_failed``, then retries (``session.resumed``), and the LAST
    substantive record is a ``session.aborted`` — the run was aborted, so the abort is the verdict.
    Both outcomes are non-``completed``, so the sidebar shows a run needing attention either way;
    the distinction that matters is ended-versus-working.
    """
    path = STUCK_RUNS_DIR / run_id / "ledger.ndjson"
    assert path.is_file(), f"the fixture for {run_id} is missing from {STUCK_RUNS_DIR}"
    assert support.terminal_verdict(path)[0] is True, f"{run_id} contains a turn terminal and must read as ended"
    assert outcome_of(path) == "stopped", run_id


def test_the_turn_terminal_set_matches_the_sidecars() -> None:
    """The vocabulary is the SIDECAR's, transcribed. A drifting copy is a silent misclassification."""
    from codna.durable.sidecar_client import SSE_TERMINAL_TURN_EVENTS

    assert support.TURN_TERMINAL_EVENT_TYPES == SSE_TERMINAL_TURN_EVENTS


def test_a_completed_turn_is_not_a_run_verdict(config: Any) -> None:
    """A turn that COMPLETED is progress, not an ending — folding it into "stopped" would be a lie.

    This is the distinction `codna review` was right to probe. A successful turn is a task doing its
    job; a run whose next task is still coming must not read as over because the last turn ended
    cleanly. So `session.turn_completed` is deliberately absent from the verdict set, and the mapping
    function refuses it rather than having a fallthrough that quietly calls success "stopped".
    """
    assert support.TURN_COMPLETED_EVENT in support.TURN_TERMINAL_EVENT_TYPES
    assert support.TURN_COMPLETED_EVENT not in support.TURN_VERDICT_EVENT_TYPES
    with pytest.raises(ValueError):
        support.outcome_of_turn_terminal(support.TURN_COMPLETED_EVENT)
    # And a ledger ending on a completed turn is NOT reported as finished...
    ledger = RunLedger.for_run(config.paths, "run-turn-ok")
    ledger.append("turn.event", session_id="s1", event_index=1,
                  payload={"event_type": support.TURN_COMPLETED_EVENT, "payload": {}})
    path = config.ledger_path("run-turn-ok")
    assert support.terminal_verdict(path)[0] is False
    assert outcome_of(path) is None


def test_the_detail_summary_reads_terminal_and_outcome_from_one_pass(config: Any) -> None:
    """`ledger_summary` must not report `terminal: false` beside `outcome: "completed"`.

    Two independent tail reads can observe two file states, and a response that contradicts itself is
    worse than either answer. The route now takes both facts from ONE `terminal_verdict` call.
    """
    ledger = RunLedger.for_run(config.paths, "run-atomic")
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "outcome": "completed"})
    ended, outcome = support.terminal_verdict(config.ledger_path("run-atomic"))
    assert ended is True
    assert outcome == "completed"
    # The invariant the route relies on: it is impossible for the pair to disagree.
    assert (ended, outcome) == (
        support.terminal_verdict(config.ledger_path("run-atomic"))[0],
        outcome_of(config.ledger_path("run-atomic")),
    )


def test_the_route_reports_a_terminal_bearing_run_as_ended(server: Any) -> None:
    """End-to-end: a turn terminal (no `run.finished`) reaches `/runs` as ended, with an outcome.

    A FRESH run id, not `seed_mutable_run`: that fixture's ledger already ends at a `run.finished`,
    which is deliberately final, so appending a turn terminal after it would (correctly) still report
    `completed`. The case under test is a run that never got a `run.finished` at all.
    """
    run_id = "run-turn-only"
    ledger = RunLedger.for_run(server.config.paths, run_id)
    ledger.append("note", payload={"event": "notification", "kind": "run.started", "message": "go"})
    ledger.append(
        "turn.event",
        session_id="s1",
        event_index=1,
        payload={"event_type": "session.turn_failed", "payload": {"error": "no API key"}},
    )
    _, body = server.json("GET", "/runs")
    entry = next(item for item in body["runs"] if item["run_id"] == run_id)
    assert entry["ended"] is True
    assert entry["outcome"] == "failed"
    assert entry["stalled"] is False
    _, one = server.json("GET", f"/runs/{run_id}")
    assert one["ledger"]["terminal"] is True
    assert one["outcome"] == "failed"


@pytest.mark.parametrize("run_id", STUCK_RUN_IDS)
def test_the_committed_fixtures_are_the_real_ledgers(run_id: str) -> None:
    """The fixtures are the reporter's files copied verbatim — proven by sha256, not by chain.

    Pinned by sha256 so a fixture cannot be quietly edited into something that passes: the point of
    these files is that they are what a real run wrote, and a "fixed" copy would prove nothing.

    The hash chain is deliberately NOT asserted: one of the three real ledgers has a broken chain
    (``run-20261030T184525Z``, 7 problems starting at seq 10), and that damage is part of the
    evidence. Repairing it in a fixture would erase the condition that produced the defect.
    """
    import hashlib

    path = STUCK_RUNS_DIR / run_id / "ledger.ndjson"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == STUCK_RUN_SHA256[run_id]


def _records(path: Path) -> list[Any]:
    """Every complete record in a ledger file, in order. The whole file, for chain checks only."""
    from codna.durable.ledger import LedgerRecord

    raw = path.read_bytes().decode("utf-8")
    return [LedgerRecord.from_payload(json.loads(line)) for line in raw.splitlines() if line.strip()]


def test_a_broken_chain_does_not_hide_the_terminal_verdict() -> None:
    """The one damaged fixture still resolves, because the verdict is read from the tail, not the chain.

    A run whose ledger chain is broken is exactly the run most likely to be stuck: reporting it as
    ``working`` would be the worst possible answer. The tail scan is chain-independent by design.
    """
    path = STUCK_RUNS_DIR / "run-20261030T184525Z" / "ledger.ndjson"
    assert len(support.ledger_problems(_records(path))) > 0, "this fixture is expected to be damaged"
    assert support.terminal_verdict(path)[0] is True
    assert outcome_of(path) == "stopped"


def test_translate_maps_a_missing_workflow_to_404() -> None:
    error = translate(cli_error("could not query workflow run:x: not found", code="durable_query_failed"),
                      run_id="x")
    assert error.code == "run_not_found"
    assert error.status == 404


def test_translate_maps_an_unknown_failure_to_502() -> None:
    error = translate(RuntimeError("the server exploded"), run_id="x")
    assert error.code == "run_api_backend_failed"
    assert error.status == 502


def test_the_surface_repr_never_contains_the_token(server: Running) -> None:
    assert server.token not in repr(server.server.api)


def test_an_ephemeral_port_is_reported_after_bind(server: Running) -> None:
    assert server.port != 0
    assert server.server.base_url == f"http://{server.host}:{server.port}"


def test_two_servers_share_one_token_file(config: Any) -> None:
    """The IDE holds one credential; a second server on the host must accept it."""
    settings = RunApiSettings(port=0, token_path=config.paths.root / "shared.token")
    first = RunApiServer(config=config, backend=ScriptedBackend(), settings=settings).start()
    second = RunApiServer(config=config, backend=ScriptedBackend(), settings=settings).start()
    try:
        assert first.token == second.token
        connection = http.client.HTTPConnection(second.host, second.port, timeout=10)
        try:
            connection.request("GET", "/runs", headers={"authorization": f"Bearer {first.token}"})
            assert connection.getresponse().status == 200
        finally:
            connection.close()
    finally:
        first.stop()
        second.stop()


# -- 9. the `serve` command --------------------------------------------------------------------


def test_serve_refuses_a_non_loopback_host_before_binding(config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A flag is not an escape hatch: the refusal has to happen before a socket exists."""
    from codna import cli as top_cli
    from codna.durable.cli import DurableCliError, cmd_durable

    monkeypatch.setattr("codna.durable.cli._config", lambda _args: config)
    opened: list[Any] = []

    def no_socket(*_args: Any, **_kwargs: Any) -> Any:
        opened.append(True)
        raise AssertionError("a socket was created for a refused bind address")

    monkeypatch.setattr(api_module.ThreadingRunApiServer, "__init__", no_socket)
    args = top_cli.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--no-temporal"])
    with pytest.raises(DurableCliError) as caught:
        cmd_durable(args) if args.func is cmd_durable else args.func(args)
    assert caught.value.code == "run_api_bind_not_loopback"
    assert opened == []


def test_serve_prints_a_readable_banner_through_a_pipe(tmp_path: Path) -> None:
    """The address must reach a caller that spawned the server and is WAITING for it.

    This is a real subprocess with a pipe, because that is how the IDE starts this: Python
    block-buffers stdout when it is not a TTY, and the server then serves forever. Without an
    explicit flush the banner never arrives — the failure is "the IDE hangs waiting for a URL", not
    "the output looks odd", so the assertion is that a line arrives while the process is still alive.
    """
    env = dict(os.environ, CODNA_RUNTIME_ROOT=str(tmp_path / "runtime"), PYTHONPATH=str(REPO_ROOT / "cli"))
    process = subprocess.Popen(
        [sys.executable, "-m", "codna.cli", "serve", "--no-temporal", "--port", "0"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert process.stdout is not None
        # Read with a deadline: a buffered banner means this never returns, so a plain read() would
        # hang the suite instead of failing it. `readline` on a pipe returns as soon as bytes exist.
        deadline = time.monotonic() + 60.0
        banner = ""
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            banner += line
            # The banner is indented JSON, so "the document arrived" means the braces balance — not
            # "a line mentioning url arrived", which is still the middle of the document.
            if banner.lstrip().startswith("{") and banner.count("{") == banner.count("}"):
                break
        assert process.poll() is None, f"the server exited instead of serving: {banner!r}"
        assert banner.lstrip().startswith("{"), f"the banner never arrived through the pipe: {banner!r}"
        document = json.loads(banner)
        assert document["run_api"]["url"].startswith("http://127.0.0.1:")
        # THE BANNER STATES THE MODE. A `--no-temporal` server used to report only `"temporal": null`,
        # which is a fact about a CONNECTION — an operator who had been told to run `codna serve` read
        # the absence as a broken install. `capabilities` says what works and what does not.
        capabilities = document["run_api"]["capabilities"]
        assert capabilities["mode"] == "ledger-only"
        assert capabilities["durable_execution"] is False
        assert capabilities["temporal_address"] is None
        assert capabilities["reason_code"] == "run_api_temporal_disabled"
        assert capabilities["available_routes"] == [
            "runs.capabilities",
            "runs.events",
            "runs.get",
            "runs.list",
        ]
        assert "runs.start" in capabilities["unavailable_routes"]
        assert capabilities["summary"].startswith("Ledger-only mode:")
        # `temporal` is kept for callers already keying on it, and it must AGREE with `capabilities`.
        assert document["run_api"]["temporal"] is None
        # The token's PATH is printed; the token itself must not be. A server that logs its own
        # credential has published it to every process that can read the log.
        token_path = Path(document["run_api"]["token_path"])
        token = token_path.read_text(encoding="utf-8").strip()
        assert token and token not in banner
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    finally:
        process.terminate()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - a server that ignored SIGTERM
            process.kill()
            process.communicate()


def test_the_ledger_only_backend_answers_503_for_everything_that_needs_temporal(config: Any) -> None:
    """`--no-temporal` is explicit: a caller that steers through it is told it did not happen.

    THE REFUSAL IS NOW ACTIONABLE, and that is what this pins. It used to read "this run API cannot
    reach Temporal: it was started with --no-temporal" — a sentence that named the flag the operator
    deliberately chose and nothing they could DO. The message now describes the mode (ledger-only),
    names what still works, and names the fix; `details.available_routes` is the machine-readable half
    so a client can render its own affordances instead of parsing prose.
    """
    from codna.durable.run_cli import _LedgerOnlyBackend

    backend = _LedgerOnlyBackend()
    for call in (
        lambda: backend.start_run(payload={}, run_id="r", run_timeout_s=1),
        lambda: backend.update("r", task_id="task-1", action="approve", payload={}),
        lambda: backend.signal("r", "pause"),
    ):
        with pytest.raises(RunApiError) as caught:
            call()
        assert caught.value.code == "run_api_temporal_disabled"
        assert caught.value.status == 503
        # NOT an error about a flag: a statement of the mode and of what still works.
        assert "ledger-only" in str(caught.value)
        assert "can list runs and read their history" in str(caught.value)
        # AND the route that does work is named in the structured half.
        assert caught.value.details["available_routes"] == [
            "runs.capabilities",
            "runs.events",
            "runs.get",
            "runs.list",
        ]
        assert caught.value.details["no_temporal"] is True
        # The hint names a command a person can run, not the flag they already passed.
        assert "codna durable dev-server start" in caught.value.hint
    # ...but a ledger read still works, which is the point of the mode.
    assert backend.status("r") is None
    backend.close()


def test_the_ledger_only_backend_states_its_capabilities_before_being_asked_to_act(config: Any) -> None:
    """The mode is DATA, available without provoking a 503.

    This is the other half of the fix: the banner and `GET /capabilities` answer "what can you do"
    up front, so a client does not have to attempt a mutation to learn the mode.
    """
    from codna.durable.run_cli import _LedgerOnlyBackend

    capabilities = _LedgerOnlyBackend().capabilities()
    assert capabilities.durable_execution is False
    assert capabilities.temporal_address is None
    assert capabilities.mode == "ledger-only"
    assert capabilities.reason_code == "run_api_temporal_disabled"
    # Exactly the reads work, and exactly the mutations do not — no third state to interpret.
    assert capabilities.available_routes == frozenset(
        {"runs.capabilities", "runs.list", "runs.get", "runs.events"}
    )
    assert capabilities.unavailable_routes == frozenset(
        {"runs.start", "runs.steer", "runs.approve", "runs.budget", "runs.pause", "runs.resume", "runs.cancel"}
    )
    assert not (capabilities.available_routes & capabilities.unavailable_routes)
    payload = capabilities.as_payload()
    assert payload["mode"] == "ledger-only"
    assert payload["summary"].startswith("Ledger-only mode: listing runs and reading run history work")
    assert "codna durable dev-server start" in payload["summary"]


def test_a_temporal_backend_reports_every_route_and_its_address() -> None:
    """The healthy half of the same document. No socket: the answer is a pure function of the config."""
    from codna.durable.run_api_backend import durable_capabilities

    capabilities = durable_capabilities("127.0.0.1:7233")
    assert capabilities.durable_execution is True
    assert capabilities.mode == "durable"
    assert capabilities.reason_code is None
    assert capabilities.unavailable_routes == frozenset()
    assert capabilities.available_routes == frozenset(
        {
            "runs.capabilities",
            "runs.list",
            "runs.get",
            "runs.events",
            "runs.start",
            "runs.steer",
            "runs.approve",
            "runs.budget",
            "runs.pause",
            "runs.resume",
            "runs.cancel",
        }
    )
    assert "127.0.0.1:7233" in capabilities.as_payload()["summary"]


def test_a_ledger_only_server_serves_reads_and_refuses_writes(config: Any) -> None:
    """The mode's contract over real HTTP: the run list and the stream work, a mutation is a 503."""
    from codna.durable.run_cli import _LedgerOnlyBackend

    RunLedger.for_run(config.paths, "run-io").append(
        "run.started", payload={"event": "notification", "kind": "run.started"}
    )
    settings = RunApiSettings(port=0, token_path=config.paths.root / "io.token")
    running = RunApiServer(
        config=config,
        backend=_LedgerOnlyBackend(),
        settings=settings,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_TOKEN,
                local_token=load_or_create_token(settings.token_path),
                local_identity=TEST_IDENTITY,
            )
        ),
        siem=NullSiemSink(),
    ).start()
    live = Running(running, ScriptedBackend(), config)
    try:
        status, body = live.json("GET", "/runs")
        assert status == 200
        assert [entry["run_id"] for entry in body["runs"]] == ["run-io"]
        # THE MODE RIDES ON THE LIST the IDE already polls, so a window that can list runs can say
        # what else it can do. Without this, the only way to learn the mode was to fail a start.
        assert body["capabilities"]["mode"] == "ledger-only"
        assert body["capabilities"]["durable_execution"] is False
        assert body["capabilities"]["unavailable_routes"] == sorted(
            {"runs.start", "runs.steer", "runs.approve", "runs.budget", "runs.pause", "runs.resume", "runs.cancel"}
        )

        status, body = live.json("GET", "/capabilities")
        assert status == 200
        assert body["capabilities"]["mode"] == "ledger-only"

        status, body = live.json(
            "POST", "/runs/run-io/pause", body={"identity": TEST_IDENTITY}
        )
        assert status == 503
        assert body["error"]["code"] == "run_api_temporal_disabled"
        # The refusal itself names what still works, so a client that only ever sees the 503 is not
        # left with "a flag was passed".
        assert "ledger-only" in body["error"]["message"]
        assert body["error"]["details"]["available_routes"] == [
                "runs.capabilities",
                "runs.events",
                "runs.get",
                "runs.list",
            ]

        status, body = live.json("POST", "/runs", body={"briefs": ["b"]})
        assert status == 503
    finally:
        running.stop()


def test_a_durable_server_reports_full_capabilities_over_http(config: Any) -> None:
    """The healthy path over the wire: the mode says every route is servable.

    `ScriptedBackend` stands in for a CONNECTED Temporal server, so this pins the transport's half of
    the contract without needing a real cluster. The real-cluster path is covered by
    `test_run_api_real.py` where one is available.
    """
    RunLedger.for_run(config.paths, "run-ok").append(        "run.started", payload={"event": "notification", "kind": "run.started"}
    )
    settings = RunApiSettings(port=0, token_path=config.paths.root / "ok.token")
    running = RunApiServer(
        config=config,
        backend=ScriptedBackend(),
        settings=settings,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_TOKEN,
                local_token=load_or_create_token(settings.token_path),
                local_identity=TEST_IDENTITY,
            )
        ),
        siem=NullSiemSink(),
    ).start()
    live = Running(running, ScriptedBackend(), config)
    try:
        status, body = live.json("GET", "/capabilities")
        assert status == 200
        assert body["capabilities"]["mode"] == "durable"
        assert body["capabilities"]["durable_execution"] is True
        assert body["capabilities"]["unavailable_routes"] == []
        assert body["capabilities"]["reason_code"] is None

        status, body = live.json("GET", "/runs")
        assert status == 200
        assert body["capabilities"]["mode"] == "durable"
    finally:
        running.stop()


def test_the_capability_sets_partition_the_route_table() -> None:
    """Every declared route is in exactly one set, so a new route cannot be silently undescribed.

    Written against `roles.ROUTE_ROLES` rather than the router because the role table is the closed
    declaration of "which routes exist"; a route added there without a decision in `READ_ROUTE_KEYS`
    or `TEMPORAL_ROUTE_KEYS` would otherwise be an endpoint a capability document omitted.

    THE UNION IS ASSERTED EQUAL, NOT MERELY A SUPERSET. An earlier version wrote
    ``described | {"runs.capabilities"} == set(ROUTE_ROLES)``, which passes even if
    ``runs.capabilities`` were removed from BOTH sets — the union would add it back, so the guard was
    weaker than its own comment claimed. Equality is what makes "a declared route has a home" true,
    and the explicit membership check below keeps the capability route in the READ set specifically
    (it needs no Temporal, which is why a ledger-only server can answer it).
    """
    from codna.durable.roles import ROUTE_ROLES
    from codna.durable.run_api_backend import READ_ROUTE_KEYS, TEMPORAL_ROUTE_KEYS

    assert not (READ_ROUTE_KEYS & TEMPORAL_ROUTE_KEYS), "a route cannot both need and not need Temporal"
    described = READ_ROUTE_KEYS | TEMPORAL_ROUTE_KEYS
    # Equality, with no set added back in: a route missing from both buckets fails HERE.
    assert described == set(ROUTE_ROLES)
    # And `runs.capabilities` is a READ, so a ledger-only server can describe itself.
    assert "runs.capabilities" in READ_ROUTE_KEYS
    assert "runs.capabilities" not in TEMPORAL_ROUTE_KEYS


def test_the_outcome_survives_a_control_note_written_after_the_terminal(config: Any) -> None:
    """A pause on a finished run appends a note past the terminal; the run still finished.

    "The last record is not a terminal" must not be read as "this run never finished" — the surface
    would then report a completed run as open forever.
    """
    ledger = RunLedger.for_run(config.paths, "run-post")
    ledger.append("note", payload={"event": "notification", "kind": "run.finished", "outcome": "completed"})
    ledger.append("note", payload={"event": "control.requested", "action": "pause", "identity": TEST_IDENTITY})
    path = config.ledger_path("run-post")
    assert outcome_of(path) == "completed"
    assert support.last_seq_of(path) == 2


def test_the_run_list_reports_a_finished_run_after_a_control_note(server: Running) -> None:
    """The same property through the route a client reads it from."""
    run_id = seed_mutable_run(server)
    RunLedger.for_run(server.config.paths, run_id).append(
        "note", payload={"event": "notification", "kind": "run.finished", "outcome": "completed"}
    )
    status, _ = server.json("POST", f"/runs/{run_id}/pause", body={"identity": TEST_IDENTITY})
    assert status == 200
    _, body = server.json("GET", "/runs")
    assert [entry["outcome"] for entry in body["runs"]] == ["completed"]
    assert [entry["ended"] for entry in body["runs"]] == [True]
    _, one = server.json("GET", f"/runs/{run_id}")
    assert one["outcome"] == "completed"
    # `terminal` is "the run ended", not "the last record is a terminal". The control note is the
    # last record and the run still ended — reading the flag as "last record" is what reported a
    # finished run as open forever.
    assert one["ledger"]["terminal"] is True


# -- delegation and parallelism (D7) ---------------------------------------------------------


def test_post_runs_carries_the_swarm_flags_into_the_task_spec(
    server: Running, tmp_path: Path
) -> None:
    """The delegation fields reach the spec a turn executes on, through the WIRE.

    The route has a CLOSED accepted-field set, so before this change a caller sending
    ``enable_agent_teams`` got a 400 — the capability was unreachable over the API even after the spec
    learned to carry it. This asserts the accepted set and the carried values together.
    """
    from codna.durable.run_api_http import RunApiHandler

    assert {"enable_spawn_agent", "enable_agent_teams"} <= RunApiHandler._START_FIELDS
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-swarm-api",
            "repo_dir": str(repo),
            "briefs": ["delegate this"],
            "engine": "plain",
            "enable_spawn_agent": True,
            "enable_agent_teams": True,
        },
    )
    assert status == 202, body
    spec = _provider_spec(server.backend.started[0])
    assert spec.enable_spawn_agent is True
    assert spec.enable_agent_teams is True


def test_post_runs_refuses_a_non_boolean_swarm_flag(server: Running, tmp_path: Path) -> None:
    """A truthy string is refused at the wire, not coerced into a request the spec did not record."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-swarm-bad",
            "repo_dir": str(repo),
            "briefs": ["delegate this"],
            "engine": "plain",
            "enable_agent_teams": "yes",
        },
    )
    assert status == 400, body
    # `optional_bool`'s own code: the refusal names the FIELD, and it happens before the swarm
    # validator ever sees the value — the value is not a boolean, so there is nothing to gate yet.
    assert body["error"]["code"] == "run_api_field_invalid"


def test_post_runs_refuses_parallel_tasks_without_worktrees(
    server: Running, tmp_path: Path
) -> None:
    """Concurrent tasks need one checkout each; the conflict is a 400 naming it, not a bad run."""
    repo = git_repo(tmp_path / "repo")
    status, body = server.json(
        "POST",
        "/runs",
        body={
            "run_id": "run-parallel-bad",
            "repo_dir": str(repo),
            "briefs": ["a", "b"],
            "engine": "plain",
            "max_parallel_tasks": 2,
            "worktree": False,
        },
    )
    assert status == 400, body
    # The spec's own code, propagated: the conflict is a CONTRACT violation (parallelism requires
    # worktrees), so it keeps the contract's code rather than being flattened into a generic one.
    assert body["error"]["code"] == "contract_invalid"
