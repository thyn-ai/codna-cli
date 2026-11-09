"""The graph's step bodies must beat while they run, on the activity's own loop.

WHY THIS EXISTS. `codna.task.<step>` is an ASYNC activity (temporalio's langgraph contrib wraps each
graph node in `async def`), and `activity.heartbeat()` inside an async activity resolves the RUNNING
EVENT LOOP and schedules `_heartbeat_async` on it. The graph used to beat from a bare
`threading.Thread` started inside `asyncio.to_thread`, which cannot work for an async activity in
either shape:

    bare threading.Thread             -> RuntimeError('Not in activity context')
    threading.Thread(ctx.run(...))    -> RuntimeError('no running event loop')
    a task on the activity's own loop -> ok

`_beat_once` swallows `RuntimeError` by design (it also runs in tests with no activity), so the
wrapper never beat and every failure above was silent. The observable consequence: a turn that
streams longer than `ACTIVITY_HEARTBEAT_TIMEOUT_S` (120s) fails the workflow with
"activity Heartbeat timeout" and throws away work already paid for.

These tests assert the PROPERTY — beats are issued from the loop, concurrently with the body — using a
fake `_beat_once` that records the running loop when it fires. They need no dev server, no
`langgraph` and no `durable` extra, so they run in the default suite.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from typing import Any

import pytest

from codna.durable import graph
from codna import durable as durable_package


@pytest.fixture(autouse=True)
def _stub_activities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `codna.durable.activities` with a stub whose `impl_for` each test overrides.

    Two reasons, both load-bearing:

    * `activities.py` imports `temporalio` at module scope, and the OFFLINE CI job installs none —
      importing the real module would fail that job for a reason unrelated to the property.
    * `activities.impl_for` RAISES `ActivityError` for a name it does not know, so a test that
      monkeypatched the already-imported module could pass locally (temporalio present) and fail on
      the durable job (module already imported, patch did not land).

    A module-level stub installed here makes both jobs run the same code path: the test supplies the
    body, and nothing else about `activities` is involved.
    """
    module = types.ModuleType("codna.durable.activities")

    def impl_for(name: str) -> Any:
        raise AssertionError(f"the test did not supply a body for {name!r}")

    module.impl_for = impl_for  # type: ignore[attr-defined]
    # BOTH must be replaced. `graph` does `from . import activities`, and that resolves the
    # attribute on the `codna.durable` PACKAGE before consulting `sys.modules` — so patching only
    # `sys.modules` leaves a real module that some earlier test already imported (and whose
    # `impl_for` raises for `codna.step.fake`). Patching the attribute is what makes the stub win in
    # every job, whatever imported what first.
    monkeypatch.setitem(sys.modules, "codna.durable.activities", module)
    monkeypatch.setattr(durable_package, "activities", module, raising=False)
    # `graph` imports `activities` lazily inside the function, so it re-reads `sys.modules` each call;
    # a cached attribute on the graph module would defeat that, so assert it holds none.
    assert not hasattr(graph, "activities"), (
        "graph must resolve `activities` at call time so this stub is seen"
    )


def _set_body(monkeypatch: pytest.MonkeyPatch, impl_for: object) -> None:
    """Point the stubbed `activities.impl_for` at the test's body."""
    module = sys.modules["codna.durable.activities"]
    monkeypatch.setattr(module, "impl_for", impl_for)


def test_run_step_beats_from_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_run_step` must beat while its body runs, and beat ON THE ACTIVITY'S LOOP.

    This goes through the PUBLIC seam (`_run_step` + `STEPS` + `_node_context`), so it fails on the
    pre-fix code for the right reason: the old `_beat_around` beats from a `threading.Thread`, where
    `asyncio.get_running_loop()` raises — which is exactly why `activity.heartbeat()` could never
    reach Temporal there.
    """
    beats: list[bool] = []
    seen_enough = threading.Event()

    def probe(_details: object) -> None:
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        beats.append(on_loop)
        seen_enough.set()

    monkeypatch.setattr(graph, "_beat_once", probe)
    monkeypatch.setattr(graph, "HEARTBEAT_INTERVAL_S", 0.01)

    class _Outcome:
        state = {"ok": True}

    class _Step:
        activity = "codna.step.fake"

        @staticmethod
        def build(_context: object, _state: object) -> str:
            return "request"

    def fake_body(_request: object) -> _Outcome:
        assert seen_enough.wait(timeout=5), "no heartbeat was issued while the body ran"
        return _Outcome()

    _set_body(monkeypatch, lambda _name: fake_body)
    monkeypatch.setattr(graph, "STEPS", {"fake": (_Step(),)})
    monkeypatch.setattr(graph, "_node_context", lambda: object())

    result = asyncio.run(graph._run_step("fake", {}))

    assert result == {"ok": True}
    assert beats, "the step body was never beaten"
    assert all(beats), (
        "a heartbeat was issued OFF the event loop. `codna.task.*` is an ASYNC activity, so "
        "`activity.heartbeat()` from a thread raises ('Not in activity context' for a bare thread, "
        f"'no running event loop' for a copied one) and is silently swallowed (beats: {beats[:3]!r})"
    )


def test_beat_loop_stops_when_the_body_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The beat task must be cancelled, not left running on the loop."""
    beats: list[float] = []

    monkeypatch.setattr(graph, "_beat_once", lambda _details: beats.append(time.monotonic()))
    monkeypatch.setattr(graph, "HEARTBEAT_INTERVAL_S", 0.01)

    class _Outcome:
        state = {}

    _set_body(monkeypatch, lambda _n: (lambda _r: _Outcome()))

    async def drive() -> None:
        await graph._run_step_with_heartbeat("codna.step.fake", object())

    asyncio.run(drive())
    settled = len(beats)
    time.sleep(0.1)
    assert len(beats) - settled <= 1, f"the beat loop outlived the body: +{len(beats) - settled} beats"


def test_beat_loop_propagates_a_body_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing body still cancels the beat task and surfaces the original error."""
    monkeypatch.setattr(graph, "_beat_once", lambda _details: None)
    monkeypatch.setattr(graph, "HEARTBEAT_INTERVAL_S", 0.01)

    def boom(_request: object) -> None:
        raise RuntimeError("body failed")

    _set_body(monkeypatch, lambda _n: boom)

    async def drive() -> None:
        await graph._run_step_with_heartbeat("codna.step.fake", object())

    with pytest.raises(RuntimeError, match="body failed"):
        asyncio.run(drive())
