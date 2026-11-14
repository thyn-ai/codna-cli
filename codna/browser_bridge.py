"""The client half of the browser tool: one long-lived Playwright MCP session, and the tool bodies.

**Why a background thread.** A Playwright MCP session is an MCP *client* connection: a child process plus
an ``asyncio`` stream that must stay open between tool calls, because a browser relaunched per call has no
page, no cookies and no history — "navigate, then click what you just saw" would be impossible. Codna's own
tool bodies, however, run inside an already-running event loop, where ``asyncio.run`` is an error. So the
session owns a private thread with its own loop (started lazily, never at import), and each call hops onto
that loop with ``asyncio.run_coroutine_threadsafe``. Calls are serialised by a lock: the browser is one
shared page, and two concurrent clicks on it are not a thing that means anything.

**Why the connection is lazy.** Registering a tool must not spawn a browser. ``codna mcp`` lists its tools
on every client connect, and paying for Node + Chromium to answer ``tools/list`` would make the server slow
to start for users who never browse. Nothing starts until the first browser call, and one call starts it
for the rest of the process's life.

**Failures are structured, never silent.** Every body returns text — a JSON payload on success,
``codna_browser_… error: …`` on failure — exactly like every other tool in ``mcp_server``: a tool answers,
it does not traceback into the host.

**Which Playwright tools are reachable, and why only those.** ``browser_tool.playwright_tool_for`` is the
allowlist and it is enforced here before any call: navigate, snapshot (accessible text), click, type,
screenshot and wait — the six capabilities the tool promises. Playwright also ships ``browser_evaluate``
and ``browser_run_code_unsafe`` (arbitrary in-page JavaScript), ``browser_file_upload`` and
``browser_handle_dialog``; none is exposed, because an agent should not have arbitrary in-page execution or
filesystem uploads by accident.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .browser_tool import (
    BrowserToolConfig,
    BrowserToolConfigError,
    browser_server_args,
    browser_tool_config,
    playwright_tool_for,
    screenshot_output_path,
)

__all__ = [
    "BrowserCallResult",
    "BrowserToolError",
    "PlaywrightClient",
    "browser_click_json",
    "browser_client",
    "browser_close_json",
    "browser_json",
    "browser_navigate_json",
    "browser_screenshot_content",
    "browser_screenshot_json",
    "browser_snapshot_json",
    "browser_type_json",
    "browser_wait_json",
    "close_browser_client",
    "screenshot_paths",
]

#: How long ``npx @playwright/mcp`` may take to answer ``initialize``. Generous on purpose: the first
#: launch on a machine downloads the package, and a cold npx cache is minutes, not seconds. A timeout is a
#: REFUSAL that names Node/npx, never a hang.
DEFAULT_STARTUP_TIMEOUT_S: Final = 180.0
#: Margin over Playwright's own navigation timeout for a single call.
CALL_MARGIN_S: Final = 30.0
#: How long ``close`` waits for the session's thread to unwind before giving up on it.
CLOSE_TIMEOUT_S: Final = 20.0


class BrowserToolError(RuntimeError):
    """A browser-tool failure with a stable ``code``. Raised, never returned as a bare string."""

    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass(frozen=True)
class BrowserCallResult:
    """One Playwright MCP tool answer, flattened to what a Codna tool needs to report.

    ``text`` is Playwright's own response text (its ``### Result`` / ``### Snapshot`` blocks), which is what
    a model reads. ``images`` carry the screenshot PAYLOAD plus the path it was written to: the registered
    MCP tool emits the bytes as real image content, while the JSON body reports the path and the size — a
    base64 blob inside a text payload is unreadable and unbounded.
    """

    tool: str
    text: str
    images: tuple[dict[str, Any], ...]
    is_error: bool
    path: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "text": self.text,
            "is_error": self.is_error,
            "path": self.path,
            "images": [
                {
                    "mime_type": image["mime_type"],
                    "bytes": image["bytes"],
                    "path": image.get("path", ""),
                }
                for image in self.images
            ],
        }


# -- the lifecycle: ONE verdict, decided exactly once ----------------------------------------------
#
# A startup has exactly two possible outcomes, and the defect this structure replaces came from having
# two writers for them: the waiting caller recorded a timeout, the session thread recorded success, and
# whichever landed second won. A guard on the caller's write cannot fix that — the window is the shape.
#
# So the transition has one owner: ``_settle``. It writes ``_outcome`` under a lock, the FIRST verdict
# wins, and every other writer is handed back the verdict that actually took effect. ``_outcome`` is a
# single reference to either ``_Started`` (which CARRIES the session) or the ``BrowserToolError`` that
# ended the attempt, so there is no pair of fields to observe half-set: "failed" and "running" are two
# values of one field, and a client cannot be in both states at once.
#
# The deadline is a COMPETITOR for that transition rather than a special case. A caller whose wait
# expires submits ``browser_startup_timeout`` as its verdict. If the session thread already submitted
# success, the caller loses the race by construction and adopts the LIVE session — a healthy browser is
# never reported as failed. If the caller wins, the thread's late success is refused at ``_settle`` and
# the session it built is torn down by its own exit stack — so a recorded failure never leaves a live
# session behind either. Both directions are pinned in ``tests/test_browser_bridge_lifecycle.py``.


@dataclass(frozen=True)
class _Started:
    """The verdict that a session is published — carrying the session, so the two cannot disagree.

    A boolean flag beside a separate ``session`` attribute is the structure that allowed a caller to
    observe "started" and find nothing behind it. Holding both in one immutable value removes that.
    """

    session: Any
    loop: asyncio.AbstractEventLoop


@dataclass
class _Attempt:
    """One startup attempt: the thread that runs it and the event that reports it settled.

    Mutable only in the two fields that must be assigned before the thread exists — ``thread`` so
    ``close`` can never miss it, and nothing else. The verdict lives in ``_outcome``, not here, so an
    attempt carries no state a reader could mistake for one.
    """

    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


@dataclass(frozen=True)
class BrowserSessionState:
    """The client's lifecycle as one readable value: a status, the reason if failed, and liveness.

    ``session_alive`` is never true together with a ``failure_code`` — that conjunction is exactly the
    defect this class exists to make unrepresentable, so it is reportable rather than avoidable.
    """

    status: str  #: "starting" | "started" | "failed" | "closed"
    failure_code: str = ""
    failure_message: str = ""
    session_alive: bool = False
    #: Whether the session thread is still running. Reported because "the thread was never stopped" is
    #: a defect this state must be able to show, not one a caller should have to infer from a hang.
    thread_alive: bool = False


class PlaywrightClient:
    """A lazily-started, thread-owned Playwright MCP session.

    One instance per process (see ``browser_client``). ``start`` is idempotent and safe to call from any
    thread; ``call`` blocks the calling thread while the session's own thread does the I/O.

    Two owners, each of one thing: the session thread owns the loop and the session, and ``_settle``
    owns the started-or-failed decision. Nothing else writes either, which is why a caller cannot
    observe a failure over a live session (see the lifecycle note above).
    """

    def __init__(self, config: BrowserToolConfig | None = None, *,
                 startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S) -> None:
        self._config = config
        self._startup_timeout_s = startup_timeout_s
        # Serialises the ONE-SHOT lifetime transition (`_outcome`, `_closed`, the thread handle). Never
        # held across I/O, a wait, or a join.
        self._settle_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._outcome: _Started | BrowserToolError | None = None
        self._attempt: _Attempt | None = None
        self._closed = False
        # The stop REQUEST is a threading event created up front, so `close` and a timed-out `start`
        # can always reach the thread — including before it has created its loop or its own event
        # (finding 2: the old `_stop` only existed once `_serve` had already got that far).
        self._stop_requested = threading.Event()
        # Thread-owned. Published for the stopper; the session callers use is the one inside `_Started`.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._stop_async: asyncio.Event | None = None

    # -- lifecycle ------------------------------------------------------------------------------

    @property
    def config(self) -> BrowserToolConfig:
        if self._config is None:
            self._config = browser_tool_config()
        return self._config

    @property
    def state(self) -> BrowserSessionState:
        """What the client is, in one read: `session_alive` and a failure are mutually exclusive."""
        with self._settle_lock:
            closed, outcome = self._closed, self._outcome
            thread = self._attempt.thread if self._attempt is not None else None
        alive = thread is not None and thread.is_alive()
        if isinstance(outcome, _Started):
            # A closed client has already joined the thread, so the session it published is gone.
            return BrowserSessionState(
                status="closed" if closed else "started",
                session_alive=not closed,
                thread_alive=alive,
            )
        if outcome is None:
            return BrowserSessionState(
                status="closed" if closed else "starting", thread_alive=alive)
        return BrowserSessionState(
            status="closed" if closed or outcome.code == "browser_closed" else "failed",
            failure_code=outcome.code,
            failure_message=str(outcome),
            thread_alive=alive,
        )

    def start(self) -> None:
        """Start the session, or raise ``BrowserToolError`` saying exactly why it could not.

        Idempotent, and safe from any thread: the FIRST caller creates the attempt, every later one
        waits on that same attempt, and the started-or-failed verdict is written once.
        """
        attempt = self._begin_attempt()
        if not attempt.ready.wait(self._startup_timeout_s):
            # The caller OFFERS its verdict; it does not record one. Whoever settled first wins, and
            # this thread is handed the winner: a session published just before the deadline is
            # adopted rather than contradicted, so a healthy browser is never reported as failed.
            settled = self._settle(self._timeout_error())
            if isinstance(settled, _Started):
                return
            # The deadline won. The attempt is now abandoned, so stop it rather than letting a cold
            # npx download keep running against a session nobody will ever receive (finding 2's other
            # half: a timed-out thread must be told to stop, not merely eventually joined by `close`).
            self._request_stop()
        else:
            settled = self._settled_outcome()
        if isinstance(settled, _Started):
            return
        raise settled

    def close(self) -> None:
        """Stop the session and its child process. Idempotent; never raises.

        Reaches the thread in EVERY state: the stop flag exists before the thread does, so a thread
        still connecting to npx — the cold-cache case — is cancelled rather than left running.
        """
        with self._settle_lock:
            self._closed = True
            # Set INSIDE the lock: `_serve` reads it under the same lock before it connects, so either
            # this flag is visible to it, or it has already published its stop event for us to signal.
            self._stop_requested.set()
            thread = self._attempt.thread if self._attempt is not None else None
        self._settle(self._closed_error())
        self._request_stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(CLOSE_TIMEOUT_S)

    def _request_stop(self) -> None:
        """Signal the session thread to unwind, whichever phase it is in. Idempotent; never raises.

        Two mechanisms because the thread can be blocked in two different places: the asyncio event ends
        a PUBLISHED session, and cancelling the task ends one still inside ``initialize``. The flag is
        what covers a thread that has not reached either yet.
        """
        self._stop_requested.set()
        with self._settle_lock:
            loop, stop, task = self._loop, self._stop_async, self._serve_task
        if loop is None:
            return
        try:
            if stop is not None:
                loop.call_soon_threadsafe(stop.set)
            if task is not None:
                loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # the loop was already closed by the thread itself
            return

    # -- the one-shot transition ------------------------------------------------------------------

    def _begin_attempt(self) -> "_Attempt":
        """The attempt this client's first `start` created, starting it if this call is the first."""
        with self._settle_lock:
            if self._closed:
                raise self._closed_error()
            if self._attempt is None:
                attempt = _Attempt()
                # Assigned BEFORE the thread starts, so `close` can never see a thread it cannot join.
                self._attempt = attempt
                attempt.thread = threading.Thread(
                    target=self._run, name="codna-browser", daemon=True)
                attempt.thread.start()
            return self._attempt

    def _settle(self, verdict: "_Started | BrowserToolError") -> "_Started | BrowserToolError":
        """Decide the outcome. FIRST WRITE WINS, and every writer receives the winner.

        This is the only function that ever writes ``_outcome``, which is why no caller can observe a
        half-set state: "started" and "failed" are two values of one reference, not two fields.
        """
        with self._settle_lock:
            if self._outcome is None:
                self._outcome = verdict
            settled = self._outcome
            attempt = self._attempt
        if attempt is not None:
            attempt.ready.set()
        return settled

    def _settled_outcome(self) -> "_Started | BrowserToolError":
        """The verdict, read under the lock. Callable only after ``attempt.ready`` is set."""
        with self._settle_lock:
            settled = self._outcome
        if settled is None:  # pragma: no cover - `attempt.ready` is only set after a verdict exists
            raise self._timeout_error()
        return settled

    def _published_session(self) -> tuple[Any, asyncio.AbstractEventLoop]:
        """The live session and its loop, or the reason there is none.

        Deliberately a VERDICT LOOKUP after ``start`` rather than a re-`start`: a client whose startup
        failed must report that failure, not silently block for another ``startup_timeout_s`` on an
        attempt that will never become ready.
        """
        self.start()
        with self._settle_lock:
            settled = self._outcome
        if isinstance(settled, _Started):
            return settled.session, settled.loop
        # `start` only returns on a started verdict, so reaching here means `close` landed in between.
        raise self._closed_error()  # pragma: no cover - a close between start and this read

    def _timeout_error(self) -> BrowserToolError:
        """The deadline's verdict, built without anything that can throw.

        An error handler that can fail hides the failure it was reporting: reading ``self.config`` here
        RESOLVES the environment, and an invalid ``CODNA_BROWSER_*`` setting (a malformed viewport, say)
        raises ``BrowserToolConfigError`` — so a genuine startup timeout reached the user as a config
        error and the real reason was lost.

        Only a config ALREADY resolved through this client is consulted: ``_config`` is assigned in
        ``__init__`` when the caller passed one, or by the ``config`` property once a launch has
        resolved it. When there is none the message says the package is unknown rather than naming one
        it did not read — the launch package is configurable via ``CODNA_BROWSER_PACKAGE``, so
        substituting the pinned default would be a guess presented as a fact.
        """
        package = self._resolved_package()
        timeout = self._startup_timeout_s
        reason = f"the browser server did not answer initialize within {timeout:.0f}s"
        if package:
            message = (
                f"{reason}. The first launch of {package} downloads it — check this machine's network "
                "and that `npx --version` works."
            )
        else:
            # Only what is KNOWN: no config was resolved before the deadline. Naming a cause here would
            # be a guess — the thread may have been slow to resolve it, or resolution may have failed.
            message = (
                f"{reason}. The package it launches is unknown (package unknown: no browser config was "
                "resolved before the deadline), so check the `CODNA_BROWSER_*` settings as well as this "
                "machine's network and that `npx --version` works."
            )
        return BrowserToolError(
            message,
            code="browser_startup_timeout",
            details={
                "package": package,
                "package_resolved": bool(package),
                "timeout_s": timeout,
            },
        )

    def _resolved_package(self) -> str:
        """The launch package if a config was ALREADY resolved, else ``""``. Never resolves, never raises.

        ``_config`` is a plain attribute, so this cannot trigger the environment resolution that
        ``BrowserToolConfigError`` comes from. ``getattr`` keeps it total even if a caller passed an
        object that only resembles the dataclass.
        """
        config = self._config
        if config is None:
            return ""
        package = getattr(config, "package", "")
        return package if isinstance(package, str) else ""

    @staticmethod
    def _closed_error() -> BrowserToolError:
        return BrowserToolError(
            "the browser session was closed; a new process starts a new one",
            code="browser_closed",
        )

    def _resolved_config_or_none(self) -> BrowserToolConfig | None:
        """The config, or ``None`` when resolving it is itself the failure being reported."""
        if self._config is not None:
            return self._config
        try:
            return self.config
        except BrowserToolConfigError:
            return None

    def _run(self) -> None:
        """The session thread's entry point: own the loop for the session's whole life."""
        loop = asyncio.new_event_loop()
        with self._settle_lock:
            self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(self._serve())
            with self._settle_lock:
                self._serve_task = task
            loop.run_until_complete(task)
        except BaseException as exc:  # noqa: BLE001 - normalised into the tool-error family
            # NORMALISED, not passed through raw. `_serve` converts what it can, but it can also raise
            # before its own conversion — a bad setting read by `self.config`, or a `SystemExit` from the
            # asyncio machinery — and a caller catching `BrowserToolError` must never see the raw type.
            # `_settle` refuses this if a verdict already won, so a cancellation during `close` cannot
            # overwrite a session that was published, and a late failure cannot overwrite a timeout.
            self._settle(
                self._closed_error() if self._closed
                else _as_browser_error(exc, self._resolved_config_or_none())
            )
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except BaseException:  # noqa: BLE001 - best-effort cleanup in a daemon thread
                pass
            loop.close()

    async def _serve(self) -> None:
        """Connect the child, publish the session, then serve until stopped.

        Publication is the ONE point where success becomes visible, and it is refused when a failure
        (the startup deadline included) already won. A refused publication returns — unwinding the exit
        stack kills the child — so an abandoned startup leaves nothing behind to leak.
        """
        stop = asyncio.Event()
        with self._settle_lock:
            self._stop_async = stop
            already_stopped = self._stop_requested.is_set()
        if already_stopped:
            # Closed before the connection began: do not spawn npx at all. `close` has already recorded
            # the closed verdict, so there is nothing left for this attempt to decide.
            return

        # Imported here, not at module scope: this module and `browser_tool` must import in the
        # minimal-dependency lane (no `mcp` extra), which exercises their config surface.
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ModuleNotFoundError as exc:  # the extra's absence, reported as a tool error
            raise BrowserToolError(
                "the browser tool needs the `mcp` extra: pip install 'codna[mcp]'",
                code="browser_mcp_missing",
            ) from exc

        config = self.config
        try:
            os.makedirs(config.output_dir, exist_ok=True)
        except OSError as exc:
            raise BrowserToolError(
                f"cannot create the browser output directory {config.output_dir}: {exc}",
                code="browser_output_dir_invalid",
                details={"output_dir": config.output_dir},
            ) from exc

        params = StdioServerParameters(
            command="npx",
            args=browser_server_args(config),
            cwd=config.output_dir,
        )
        async with AsyncExitStack() as stack:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            published = self._settle(_Started(session=session, loop=asyncio.get_running_loop()))
            if not isinstance(published, _Started) or published.session is not session:
                # The deadline (or `close`) decided first. Returning unwinds the stack and tears the
                # child down; the caller already has the real reason.
                return
            await stop.wait()

    # -- calls ----------------------------------------------------------------------------------

    def call(self, tool: str, arguments: dict[str, Any] | None = None, *,
             timeout_s: float | None = None) -> BrowserCallResult:
        """Run one Playwright MCP tool. Blocks until it answers, or raises ``BrowserToolError``."""
        session, loop = self._published_session()
        with self._call_lock:
            bound = timeout_s if timeout_s is not None else (
                self.config.navigation_timeout_ms / 1000.0 + CALL_MARGIN_S)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool(tool, dict(arguments or {})), loop)
            except RuntimeError as exc:  # pragma: no cover - the loop closed underneath us
                raise BrowserToolError(
                    f"the browser session's loop is closed: {exc}", code="browser_loop_closed") from exc
            try:
                raw = future.result(bound)
            except TimeoutError as exc:
                raise BrowserToolError(
                    f"{tool} did not answer within {bound:.0f}s",
                    code="browser_call_timeout",
                    details={"tool": tool, "timeout_s": bound},
                ) from exc
            except BaseException as exc:  # noqa: BLE001 - normalised into the tool-answer contract
                raise _as_browser_error(exc, self.config, tool=tool) from exc
        return _flatten(tool, raw)

    def list_tools(self, *, timeout_s: float = 60.0) -> list[str]:
        """The Playwright tools the running server advertises.

        Runs on the session's OWN loop for the same reason every other call does: a ``ClientSession`` is
        bound to the loop that created it, and calling it from a second loop — which is what
        ``asyncio.run`` in a test would do — is a deadlock, not an error. That is the whole reason this
        lives here instead of in a test helper.
        """
        session, loop = self._published_session()
        with self._call_lock:
            future = asyncio.run_coroutine_threadsafe(session.list_tools(), loop)
            try:
                listed = future.result(timeout_s)
            except TimeoutError as exc:
                raise BrowserToolError(
                    f"tools/list did not answer within {timeout_s:.0f}s",
                    code="browser_call_timeout",
                    details={"tool": "tools/list", "timeout_s": timeout_s},
                ) from exc
            except BaseException as exc:  # noqa: BLE001
                raise _as_browser_error(exc, self.config, tool="tools/list") from exc
        return [str(getattr(tool, "name", "")) for tool in getattr(listed, "tools", ())]

    # -- the six capabilities -------------------------------------------------------------------

    def navigate(self, url: str) -> BrowserCallResult:
        return self.call("browser_navigate", {"url": url})

    def snapshot(self, *, depth: float | None = None) -> BrowserCallResult:
        """Accessible text of the page — the readable form of what is on screen."""
        arguments: dict[str, Any] = {}
        if depth is not None:
            arguments["depth"] = depth
        return self.call("browser_snapshot", arguments)

    def click(self, target: str, *, element: str = "") -> BrowserCallResult:
        arguments: dict[str, Any] = {"target": target}
        if element:
            arguments["element"] = element
        return self.call("browser_click", arguments)

    def type_text(self, target: str, text: str, *, element: str = "",
                  submit: bool = False) -> BrowserCallResult:
        arguments: dict[str, Any] = {"target": target, "text": text}
        if element:
            arguments["element"] = element
        if submit:
            arguments["submit"] = True
        return self.call("browser_type", arguments)

    def wait_for(self, *, text: str = "", text_gone: str = "",
                 time_s: float = 0.0) -> BrowserCallResult:
        if not text and not text_gone and time_s <= 0:
            raise BrowserToolError(
                "wait needs text, text_gone, or a positive time",
                code="browser_wait_invalid",
            )
        arguments: dict[str, Any] = {}
        if text:
            arguments["text"] = text
        if text_gone:
            arguments["textGone"] = text_gone
        if time_s > 0:
            arguments["time"] = time_s
        return self.call("browser_wait_for", arguments)

    def screenshot(self, filename: str = "", *, full_page: bool = False) -> BrowserCallResult:
        """Screenshot to a file under the output directory, and return its bytes as an image.

        Playwright's server answers with a PATH, not a payload (verified against 0.0.85), so the file is
        read back here — that is what makes the image reach the model instead of a filename it cannot open.
        """
        relative, absolute = screenshot_paths(self.config.output_dir, filename)
        arguments: dict[str, Any] = {"scale": "css", "filename": absolute}
        if full_page:
            arguments["fullPage"] = True
        result = self.call("browser_take_screenshot", arguments)
        if result.is_error:
            return result
        try:
            payload = Path(absolute).read_bytes()
        except OSError as exc:
            raise BrowserToolError(
                f"the screenshot was reported but {relative} could not be read: {exc}",
                code="browser_screenshot_unreadable",
                details={"path": absolute},
            ) from exc
        image = {
            "data": base64.b64encode(payload).decode("ascii"),
            "mime_type": _mime_for(absolute),
            "bytes": len(payload),
            "path": relative,
        }
        return BrowserCallResult(
            tool=result.tool,
            text=f"{result.text}\n\nscreenshot: {relative} ({len(payload)} bytes)",
            images=(image,),
            is_error=False,
            path=relative,
        )


def screenshot_paths(output_dir: str, filename: str) -> tuple[str, str]:
    """Resolve a screenshot name INSIDE ``output_dir``, or refuse it.

    A tool that writes wherever a model says is a tool that can overwrite a source file. The name is
    therefore a plain relative path under the browser output directory; an absolute path or ``..`` is
    REFUSED rather than clamped, because silently relocating a write is its own confusion.
    """
    name = (filename or "").strip() or f"page-{time.strftime('%Y%m%dT%H%M%S')}.png"
    try:
        return screenshot_output_path(output_dir, name)
    except BrowserToolConfigError as exc:
        # Same refusal, carried into this module's error family so a caller catching `BrowserToolError`
        # still sees the code and the offending value rather than a foreign exception type.
        raise BrowserToolError(str(exc), code=exc.code, details=exc.details) from exc


def _mime_for(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/png"


def _flatten(tool: str, raw: Any) -> BrowserCallResult:
    """A ``CallToolResult`` as ``BrowserCallResult``, tolerating either spelling of ``isError``."""
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for block in getattr(raw, "content", None) or ():
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(str(getattr(block, "text", "")))
        elif kind == "image":
            data = getattr(block, "data", "")
            if data:
                images.append({
                    "data": str(data),
                    "mime_type": str(getattr(block, "mimeType", "") or "image/png"),
                    "bytes": 0,
                    "path": "",
                })
    is_error = bool(getattr(raw, "isError", getattr(raw, "is_error", False)))
    return BrowserCallResult(
        tool=tool,
        text="\n".join(part for part in texts if part),
        images=tuple(images),
        is_error=is_error,
    )


def _as_browser_error(exc: BaseException, config: BrowserToolConfig | None, *,
                      tool: str = "") -> BrowserToolError:
    """Normalise anything a launch or a call can throw into one structured error family.

    ``config`` is OPTIONAL because one of the failures this must carry is config resolution itself: a bad
    ``CODNA_BROWSER_*`` value raises before a config exists, and turning that into a raw
    ``BrowserToolConfigError`` out of ``start()`` is what the tool bodies cannot catch.
    """
    if isinstance(exc, BrowserToolError):
        return exc
    if isinstance(exc, BrowserToolConfigError):
        # Already structured, with its own code and field. Carried across the family boundary so a caller
        # that catches `BrowserToolError` still sees the setting's name and its refusal.
        return BrowserToolError(str(exc), code=exc.code, details=exc.details)
    package = getattr(config, "package", "") or ""
    # `anyio`/`mcp` wrap a child that exited during startup in a BaseExceptionGroup; the useful reason is
    # the leaf, and "unhandled errors in a TaskGroup" is not something a user can act on.
    leaf = exc
    while True:
        inner = getattr(leaf, "exceptions", None)
        if not inner:
            break
        leaf = inner[0]
    detail = str(leaf) or type(leaf).__name__
    if tool:
        return BrowserToolError(
            f"{tool} failed: {detail}",
            code="browser_tool_call_failed",
            details={"tool": tool, "package": package},
        )
    return BrowserToolError(
        f"the browser server ({package or 'unknown package'}) could not start: {detail}. Codna does not "
        "bundle Node: install Node.js 18+ and confirm `npx @playwright/mcp --help` runs.",
        code="browser_startup_failed",
        details={"package": package},
    )


# -- the process-wide session --------------------------------------------------------------------

_CLIENT: PlaywrightClient | None = None
_CLIENT_LOCK = threading.Lock()


def browser_client(config: BrowserToolConfig | None = None) -> PlaywrightClient:
    """This process's browser session, created on first use and reused for its lifetime."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = PlaywrightClient(config)
        return _CLIENT


def close_browser_client() -> None:
    """Close and forget the process-wide session. Also runs at interpreter exit."""
    global _CLIENT
    with _CLIENT_LOCK:
        client, _CLIENT = _CLIENT, None
    if client is not None:
        client.close()


atexit.register(close_browser_client)


# -- tool bodies ---------------------------------------------------------------------------------
#
# One function per capability, module level, so each is testable without the `mcp` extra and the
# registered tool in `mcp_server` is a thin wrapper over the SAME body (the `fix_json` / `recall_json`
# pattern). Each returns text — a JSON payload on success, `codna_browser_<name> error: …` on failure —
# and never raises, because a tool must answer.


def _render(tool_label: str, result: BrowserCallResult) -> str:
    if result.is_error:
        return f"{tool_label} error: {result.text or 'the browser returned an error'}"
    return json.dumps(result.to_payload(), indent=2)


def browser_json(name: str, action) -> str:
    """Run one capability body: resolve the client, call ``action``, render, and never raise.

    ``name`` is the Codna tool name, used both for the error prefix and for the allowlist check — a name
    outside ``browser_tool.BROWSER_TOOL_NAMES`` is refused before a browser is ever started.
    """
    try:
        playwright_tool_for(name)
    except BrowserToolConfigError as exc:
        return f"{name} error: {exc}"
    try:
        return _render(name, action(browser_client()))
    except (BrowserToolError, BrowserToolConfigError) as exc:
        return f"{name} error: {exc}"
    except Exception as exc:  # noqa: BLE001 - a tool answers; it never tracebacks into the host
        return f"{name} error: {type(exc).__name__}: {exc}"


def browser_navigate_json(url: str = "") -> str:
    """Navigate to ``url`` and return the page's accessible text. ``url`` is required."""
    if not (url or "").strip():
        return "codna_browser_navigate error: url is required"
    return browser_json("codna_browser_navigate", lambda client: client.navigate(url.strip()))


def browser_snapshot_json(depth: float = 0) -> str:
    """Read the current page as accessible text (no JavaScript execution)."""
    return browser_json(
        "codna_browser_snapshot",
        lambda client: client.snapshot(depth=depth if depth and depth > 0 else None),
    )


def browser_click_json(target: str = "", element: str = "") -> str:
    """Click an element by snapshot ref, CSS selector, or another Playwright locator."""
    if not (target or "").strip():
        return "codna_browser_click error: target is required"
    return browser_json(
        "codna_browser_click", lambda client: client.click(target.strip(), element=element))


def browser_type_json(target: str = "", text: str = "", element: str = "",
                      submit: bool = False) -> str:
    """Type ``text`` into an editable element, optionally pressing Enter afterwards."""
    if not (target or "").strip():
        return "codna_browser_type error: target is required"
    return browser_json(
        "codna_browser_type",
        lambda client: client.type_text(target.strip(), text, element=element, submit=submit),
    )


def browser_screenshot_json(filename: str = "", full_page: bool = False) -> str:
    """Screenshot the viewport (or the full page) into the browser output directory."""
    return browser_json(
        "codna_browser_screenshot",
        lambda client: client.screenshot(filename, full_page=full_page),
    )


def browser_screenshot_content(
    filename: str = "", full_page: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """The screenshot body, as ``(json text, image parts)``.

    A screenshot is the one capability whose answer must be an IMAGE, not a description of one, so the
    registered MCP tool needs the payload as well as the JSON text. Both come from one call, so the image
    bytes and the JSON can never describe different files. ``images`` is empty on any failure.
    """
    captured: dict[str, list[dict[str, Any]]] = {"images": []}

    def action(client: PlaywrightClient) -> BrowserCallResult:
        result = client.screenshot(filename, full_page=full_page)
        if not result.is_error:
            captured["images"] = [dict(image) for image in result.images]
        return result

    return browser_json("codna_browser_screenshot", action), captured["images"]


def browser_wait_json(text: str = "", text_gone: str = "", time_s: float = 0.0) -> str:
    """Wait for text to appear, text to disappear, or a fixed number of seconds to pass."""
    return browser_json(
        "codna_browser_wait",
        lambda client: client.wait_for(text=text, text_gone=text_gone, time_s=time_s),
    )


def browser_close_json() -> str:
    """Close the session and its browser. The next browser call starts a fresh one."""
    try:
        close_browser_client()
    except Exception as exc:  # noqa: BLE001 - a tool answers
        return f"codna_browser_close error: {type(exc).__name__}: {exc}"
    return json.dumps({"closed": True}, indent=2)
