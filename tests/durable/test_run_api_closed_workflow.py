"""`GET /runs/:id` must answer for a run that CLOSED, not 502 on the query the CLI chose.

THE DEFECT THIS PINS, MEASURED ON A REAL SERVER. `TemporalRunBackend.status` called
`run_cli.query_workflow`, which sends ``reject_condition=QueryRejectCondition.NOT_OPEN`` (right for
`--wait`: it must not query a closed execution, it reads the result instead). So the moment a run
reached a terminal state, every detail read answered::

    {"error": {"code": "run_api_backend_failed",
               "message": "… could not query workflow run:<id>: Query rejected, status: 2"}}

Measured on `run-20261108T221611Z`, which had just finished `approved_unpublished`. The list route
(user, 2026-11-08 22:29 UTC) kept working because it reads the ledger, so the IDE rendered a *done*
row whose detail pane could not open — and the row would have been correct while opening it was a
502.

WHY THESE TESTS DO NOT NEED TEMPORAL. The bug is a CLASSIFICATION bug: "the execution closed" was
being classified as "the server is unwell". These tests drive `TemporalRunBackend.status` through a
stub client that raises exactly what the SDK raises, so they pin the classification — including that
a real server failure is STILL a 502 and that a gone execution is STILL `None`.
"""
from __future__ import annotations

from typing import Any

import pytest

# The `durable` extra is optional and the offline CI job installs none of it, so this module asks for
# it the same way `test_run_api.py` does. `run_api_backend` reaches `temporalio` through
# `run_workflow`, which is why a test that drives `status()` cannot run without it at all.
pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.run_api import RunApiError  # noqa: E402
from codna.durable.run_api_backend import (  # noqa: E402
    TemporalRunBackend,
    _is_query_rejected,
    is_not_found,
)


class _WorkflowQueryRejectedError(Exception):
    """The SDK's own type name and shape: `WorkflowQueryRejectedError`."""


_WorkflowQueryRejectedError.__name__ = "WorkflowQueryRejectedError"


class _GoneError(Exception):
    def __init__(self) -> None:
        super().__init__("workflow not found for ID: run:r")


class _ApplicationError(Exception):
    """The bottom of a failed workflow's cause chain: the activity's own error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.type = "ActivityError"
        self.details = None


class _WorkflowFailureError(Exception):
    """The SDK's wrapper for a failed `result()`. `cause` is the SDK's chain, not `__cause__`."""

    def __init__(self) -> None:
        super().__init__("workflow execution failed")
        self.cause: Any = None


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _Description:
    def __init__(self, status: str) -> None:
        self.status = _Status(status)
        self.close_time = None


class _Handle:
    def __init__(self, *, describe: Any = None, result: Any = None) -> None:
        self._describe = describe
        self._result = result

    async def describe(self) -> Any:
        if isinstance(self._describe, BaseException):
            raise self._describe
        return self._describe

    async def result(self) -> Any:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _Client:
    def __init__(self, handle: _Handle) -> None:
        self.handle = handle
        self.queried: list[str] = []

    def get_workflow_handle(self, workflow_id: str) -> _Handle:
        self.queried.append(workflow_id)
        return self.handle


def _backend(monkeypatch: pytest.MonkeyPatch, handle: _Handle, *, query_error: BaseException) -> TemporalRunBackend:
    """A TemporalRunBackend whose status query raises `query_error`.

    `_call` is replaced by a dispatcher rather than a Mock so the two coroutines this method can run
    (`query`, then `describe`/`result`) are distinguishable — the point of the fix is that the SECOND
    one is reached at all.
    """
    client = _Client(handle)
    config = type(
        "Config",
        (),
        {"run_workflow_id": staticmethod(lambda run_id: f"run:{run_id}"), "address": "127.0.0.1:7233"},
    )()
    backend = TemporalRunBackend.__new__(TemporalRunBackend)
    backend._client = client  # type: ignore[attr-defined]
    backend._config = config  # type: ignore[attr-defined]

    import asyncio

    def call(coroutine: Any, *, timeout_s: float = 1.0) -> Any:
        try:
            return asyncio.run(coroutine)
        except BaseException:
            raise

    # The query path goes through `run_cli.query_workflow`, which the stub client cannot serve
    # (`get_workflow_handle().query` is not implemented) — so the query is intercepted and the
    # recurring failure is raised exactly where the real one appears.
    import codna.durable.run_cli as run_cli
    async def fake_query(client_: Any, workflow_id: str, query: Any) -> dict[str, Any]:
        raise query_error

    monkeypatch.setattr(run_cli, "query_workflow", fake_query)
    backend._call = call  # type: ignore[attr-defined]
    return backend


def test_a_closed_completed_workflow_answers_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE REGRESSION: a finished run's detail read is an ANSWER, not a 502."""
    handle = _Handle(describe=_Description("COMPLETED"), result={"outcome": "completed"})
    backend = _backend(monkeypatch, handle, query_error=_WorkflowQueryRejectedError("Query rejected, status: 2"))

    status = backend.status("r")
    assert status is not None
    assert status["closed"] is True
    assert status["workflow_status"] == "COMPLETED"
    assert status["outcome"] == "completed"
    assert status["error"] is None


def test_a_closed_failed_workflow_reports_the_runs_own_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAILED run says why it failed; it does not report the query as the failure."""
    # The SDK's shape: `result()` raises `WorkflowFailureError` whose `.cause` chain bottoms out at
    # the activity's own `ApplicationError`. That is what `workflow_failure_report` walks, and this
    # fixture gives it the same two links so the answer is the REAL one.
    activity = _ApplicationError("create_session failed: 403 policy_missing")
    wrapper = _WorkflowFailureError()
    wrapper.cause = activity
    handle = _Handle(describe=_Description("FAILED"), result=wrapper)
    backend = _backend(monkeypatch, handle, query_error=_WorkflowQueryRejectedError("Query rejected, status: 2"))

    status = backend.status("r")
    assert status is not None
    assert status["workflow_status"] == "FAILED"
    assert status["closed"] is True
    # The run's OWN error reaches the caller, not "the query failed".
    assert status["outcome"] == "failed"
    assert status["error"] is not None
    assert "policy_missing" in status["error"]["message"]


def test_a_gone_execution_is_still_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention is still a 404 at the edge, and the ledger is what covers that lifetime."""
    handle = _Handle(describe=_GoneError())
    backend = _backend(monkeypatch, handle, query_error=_GoneError())
    assert backend.status("r") is None


def test_a_real_server_failure_is_still_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE HALF THAT MUST NOT REGRESS: a worker-down/timeout error is NOT reclassified as "closed"."""
    handle = _Handle(describe=_Description("RUNNING"))
    backend = _backend(monkeypatch, handle, query_error=RuntimeError("no poller seen for task queue recently"))

    with pytest.raises(RunApiError) as caught:
        backend.status("r")
    assert caught.value.code == "run_api_backend_failed"
    assert caught.value.status == 502


def test_query_rejection_is_recognised_through_the_cli_wrapper() -> None:
    """The CLI wraps the SDK error, so the cause is inspected, not only the outer type."""


    inner = _WorkflowQueryRejectedError("Query rejected, status: 2")

    class Wrapper(Exception):
        pass

    wrapper = Wrapper("could not query workflow run:r: Query rejected, status: 2")
    wrapper.__cause__ = inner
    assert _is_query_rejected(inner) is True
    assert _is_query_rejected(wrapper) is True
    assert _is_query_rejected(RuntimeError("connection refused")) is False
    # And it is not confused with "gone", which has its own answer.
    assert is_not_found(_WorkflowQueryRejectedError("Query rejected, status: 2")) is False


def test_an_unrelated_rejection_phrase_is_not_read_as_a_closed_workflow() -> None:
    """A REVIEW FINDING, PINNED. The text branch must not match another system's wording.

    The first version matched the bare words "query rejected" anywhere in the message, so a database or
    gateway error that happens to phrase itself that way would be classified as "the workflow closed"
    and its genuine failure would be answered from `_closed_status` instead of raising a 502.
    """
    assert _is_query_rejected(RuntimeError("database query rejected by policy")) is False
    assert _is_query_rejected(RuntimeError("search query rejected: rate limited")) is False
    # The SDK's own sentence still matches, which is what the branch is for.
    assert _is_query_rejected(RuntimeError("Query rejected, status: 2")) is True


def test_an_unreadable_ending_is_a_502_not_a_blank_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A REVIEW FINDING, PINNED — the fall-through that produced `outcome: null, error: null`.

    `workflow_failure_report` walks the SDK's `.cause` chain. An error raised by THIS package (here:
    `durable_run_outcome_unknown`, because the completion's outcome is not in the contract's
    vocabulary) has Python's `__cause__` instead, so the report is None and `is_not_found` is False.
    Before this test the method returned a success-shaped body with no outcome and no error; it must
    raise so the caller learns the read failed.
    """

    class _BadOutcomeHandle(_Handle):
        async def result(self) -> Any:
            return {"outcome": "COMPLETED"}  # not in RunOutcome's vocabulary: refused on purpose

    backend = _backend(
        monkeypatch,
        _BadOutcomeHandle(describe=_Description("COMPLETED")),
        query_error=_WorkflowQueryRejectedError("Query rejected, status: 2"),
    )
    with pytest.raises(RunApiError) as caught:
        backend.status("r")
    # THIS SURFACE reports an unreadable ending as its own 502 (that is `translate`'s contract for
    # anything it does not recognise) — the point is that it RAISES instead of returning a blank
    # success. The message keeps the underlying refusal's own words for whoever debugs it.
    assert caught.value.status == 502
    assert caught.value.code == "run_api_backend_failed"
    assert "no recognized outcome" in str(caught.value)
