"""The session lifecycle: a startup deadline and a startup success can never both be true.

**The defect these tests exist for.** ``PlaywrightClient.start()`` used to record a startup timeout from
the CALLER's thread while the session thread independently recorded success. Two writers, one state, and
a window between them: whichever write landed second won, so a session that did start could be reported
as failed forever (every later call answering ``browser_startup_timeout`` while a healthy browser ran) or
a failed startup could be reported as started (every later call answering ``browser_not_started`` with the
real reason lost). Two earlier fixes edited WHERE the timeout was recorded; neither removed the second
writer, so the window survived both.

**How these tests force the interleaving instead of hoping for it.** The session thread's progress is
controlled by a gate the test releases: a fake ``mcp`` package is installed in ``sys.modules``, so the
REAL ``_serve`` runs — config, launch args, exit stack, ``initialize`` — and only the point where it would
talk to npx is under the test's control. Nothing here patches the client's internals.

**Why the assertions are on behaviour.** They were written against the pre-restructure code and failed
there (see ``_observation`` for the one compatibility shim, which only READS the old private fields). The
assertions themselves are what a caller sees — ``start()`` raises or returns, ``call()`` answers or
refuses, ``close()`` stops the thread — so the same file is a reproduction on the old code and a
regression pin on the new one.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from typing import Any

import pytest

from codna.browser_bridge import BrowserToolError, PlaywrightClient
from codna.browser_tool import browser_tool_config

ON = {"CODNA_BROWSER_TOOL": "1", "CODNA_RUNTIME_ROOT": "/tmp/codna-browser-lifecycle"}


class _FakeResult:
    """Enough of a ``CallToolResult`` for ``_flatten``."""

    isError = False
    content: tuple[Any, ...] = ()


class _Gate:
    """The session thread's release point, settable from the test's thread.

    An ``asyncio.Event`` cannot be set from another thread, and the loop that owns it belongs to the
    client, so the gate keeps a reference to that loop and hops onto it — the same way a real caller
    would reach the session.
    """

    def __init__(self) -> None:
        self.event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def release(self) -> None:
        assert self._loop is not None and self.event is not None, "the gate was never reached"
        self._loop.call_soon_threadsafe(self.event.set)

    async def wait(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.event = asyncio.Event()
        await self.event.wait()


@pytest.fixture
def fake_mcp(monkeypatch):
    """A fake ``mcp`` package whose ``initialize`` waits on a gate the test releases.

    Installed in ``sys.modules`` because ``_serve`` imports it INSIDE the coroutine (deliberately: the
    module must import on the lane without the extra). Replacing the module is therefore the only seam
    that leaves the client's own code untouched.
    """
    gate = _Gate()

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            self._read, self._write = read, write

        async def __aenter__(self) -> "_Session":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def initialize(self) -> None:
            await gate.wait()

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeResult:
            return _FakeResult()

        async def list_tools(self) -> Any:
            return types.SimpleNamespace(tools=())

    class _Stdio:
        async def __aenter__(self) -> tuple[Any, Any]:
            return (object(), object())

        async def __aexit__(self, *exc: object) -> bool:
            return False

    mcp_module = types.ModuleType("mcp")
    mcp_module.ClientSession = _Session  # type: ignore[attr-defined]
    mcp_module.StdioServerParameters = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    stdio_module = types.ModuleType("mcp.client.stdio")
    stdio_module.stdio_client = lambda params: _Stdio()  # type: ignore[attr-defined]
    client_module = types.ModuleType("mcp.client")

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)
    return gate


class _Observation:
    """What a caller can see about a client: whether a session is live, and the recorded failure.

    On the restructured client this reads ``client.state``. On the pre-restructure client — which had no
    such property — it reads the two private fields the race was between, so this same test can be run
    against the old code and FAIL there for the real reason. Read-only: nothing here writes state.
    """

    def __init__(self, client: PlaywrightClient) -> None:
        self._client = client

    @property
    def session_alive(self) -> bool:
        state = getattr(self._client, "state", None)
        if state is not None:
            return bool(state.session_alive)
        return getattr(self._client, "_session", None) is not None

    @property
    def failure_code(self) -> str:
        state = getattr(self._client, "state", None)
        if state is not None:
            return str(state.failure_code)
        failure = getattr(self._client, "_startup_failure", None)
        return str(getattr(failure, "code", "") or "")

    @property
    def thread_alive(self) -> bool:
        state = getattr(self._client, "state", None)
        if state is not None:
            return bool(state.thread_alive)
        thread = getattr(self._client, "_thread", None)
        return thread is not None and thread.is_alive()

    @property
    def settled(self) -> bool:
        """Whether the startup attempt has finished, either way."""
        state = getattr(self._client, "state", None)
        if state is not None:
            return state.status in {"started", "failed", "closed"}
        ready = getattr(self._client, "_ready", None)
        return bool(ready is not None and ready.is_set())


def _settle(client: PlaywrightClient, *, timeout_s: float = 10.0) -> _Observation:
    """Wait until the startup attempt has finished. Never assumes a fixed sleep."""
    observed = _Observation(client)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if observed.settled:
            return observed
        time.sleep(0.01)
    return observed


def test_a_timeout_and_a_late_success_are_never_both_recorded(fake_mcp) -> None:
    """The finding, as an invariant: a reported failure and a live session cannot coexist.

    The deadline expires while ``initialize`` is still gated, so the timeout is committed first. The gate
    is then released and the thread's startup DOES complete. The client must not end up holding a live
    session while telling every caller the startup failed — that is the "permanently poisoned session"
    the reviewer described.
    """
    client = PlaywrightClient(browser_tool_config(ON), startup_timeout_s=0.05)
    try:
        with pytest.raises(BrowserToolError) as timed_out:
            client.start()
        assert timed_out.value.code == "browser_startup_timeout"

        # The thread is now past the deadline and completes a session that nobody asked for any more.
        fake_mcp.release()
        observed = _settle(client)

        assert not (observed.session_alive and observed.failure_code), (
            "a live session was reported as failed; every later call would answer "
            f"{observed.failure_code!r} forever while the browser kept running"
        )
        assert observed.session_alive is False, (
            "the deadline won, so the abandoned session must have been torn down, not left running"
        )
        # The one useful message is not lost after the first call.
        with pytest.raises(BrowserToolError) as later:
            client.start()
        assert later.value.code == "browser_startup_timeout"
        # And a call reports that same reason rather than a misleading "not started".
        with pytest.raises(BrowserToolError) as called:
            client.call("browser_snapshot", {})
        assert called.value.code == "browser_startup_timeout"
    finally:
        client.close()


def test_a_success_that_wins_the_deadline_is_a_working_session(fake_mcp) -> None:
    """The other direction: if the startup completed, a caller must get a WORKING client.

    The gate is released before the deadline can expire, so success is committed first. A client that
    reported a live session but refused every call — or that recorded a timeout it never observed — is
    the same defect mirrored.
    """
    client = PlaywrightClient(browser_tool_config(ON), startup_timeout_s=10.0)
    try:
        released = threading.Timer(0.05, fake_mcp.release)
        released.start()
        client.start()
        released.join()

        observed = _Observation(client)
        assert observed.session_alive is True
        assert observed.failure_code == ""
        # The proof that "started" is not a label over an empty session: a real call goes through.
        assert client.call("browser_snapshot", {}).is_error is False
    finally:
        client.close()


def test_close_reaches_a_thread_that_never_reached_its_stop_signal(fake_mcp) -> None:
    """Finding 2: a thread blocked mid-startup must be stoppable, and ``close`` must not give up on it.

    ``_stop`` used to be created inside the serve coroutine, so ``close`` found ``None`` and returned
    without joining — leaving the npx child downloading with no way to cancel it. Here the thread is
    blocked BEFORE any stop signal could exist, which is exactly that window.
    """
    client = PlaywrightClient(browser_tool_config(ON), startup_timeout_s=0.05)
    with pytest.raises(BrowserToolError):
        client.start()

    assert _Observation(client).thread_alive is True, (
        "the session thread should still be blocked mid-connect"
    )

    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started

    assert _Observation(client).thread_alive is False, (
        "close() returned with the session thread still running"
    )
    assert elapsed < 10.0, f"close() took {elapsed:.1f}s; the thread was not signalled, only waited out"
    # Nothing is reachable afterwards, and it says so with the closed code rather than a stale one.
    with pytest.raises(BrowserToolError) as after:
        client.start()
    assert after.value.code == "browser_closed"
