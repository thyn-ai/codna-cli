"""Waiting for a run: the poll loop, and the exit code a FINISHED run must produce.

Split out of `run_cli.py` (which is at the repository's 1000-line ceiling) on a real boundary: this
is what a CLIENT does while it waits, as opposed to what it ASKS the server to start. Every function
here is about one question — "how did this run end, and what does that mean for my exit code" — and
none of them starts, configures or registers anything.

Two facts shape the code, both measured rather than assumed:

* **A query only answers while the workflow is OPEN.** `RunWorkflow.status` is a query, so a run that
  completes inside one poll interval never produces a terminal status: the next poll is REJECTED
  ("Query rejected, status: 2"). MEASURED on the acceptance runs for the publish-skip change — the
  ledger recorded `run.finished`, and `codna run --wait` still exited non-zero. So a rejected query is
  not "an error" by itself; the workflow's own RESULT is asked for, and `completed` there is zero.
* **Unreadable is not success.** A result that is missing, untyped in an unexpected way, or carries an
  outcome this build does not know re-raises the ORIGINAL query failure, and an unrecognized outcome
  is refused by name. Nothing is coerced into "fine".

The exit code is `EXIT_NOT_SUCCESS` (2) for every non-`completed` outcome, which is the contract
`test_run_cli.py` pins: a caller scripting around `codna run --wait` must be able to read the code.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

from . import cli as _cli
from .config import DurableConfig
from .errors import cli_error as _error

__all__ = [
    "EXIT_NOT_SUCCESS",
    "_exit_for_closed_run",
    "_run_is_finished",
    "_wait_run",
    "query_workflow",
    "workflow_completion",
    "workflow_failure_report",
    "workflow_outcome",
]

# The exit status for "the command ran, and the run did not succeed". `cli.py` re-exports it, so the
# two spellings of the durable surface answer with one number.
EXIT_NOT_SUCCESS = 2


async def _wait_run(
    client: Any,
    config: DurableConfig,
    run_id: str,
    *,
    poll_s: float,
    timeout_s: float,
    stop: Any | None = None,
) -> int:
    """Poll the run's status until it completes, printing each change as JSON.

    Polling a QUERY (not ``result()``) so a parked task is visible while it is parked: ``result()``
    would block for the whole approval wait with no output, which reads as a hung command.

    ``stop`` (an ``asyncio.Event``) ends the wait when a signal arrives — the in-process worker's
    SIGINT/SIGTERM sets it. The run itself is NOT cancelled: it lives on the server, so "stop waiting"
    must leave a resumable run rather than destroy one the user paid for. The exit code says so.
    """
    from .run_workflow import RunWorkflow

    if poll_s <= 0:
        raise _error(
            "--poll-interval must be positive.",
            code="durable_poll_interval_invalid",
            details={"poll_interval": poll_s},
        )
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while True:
        try:
            status = await query_workflow(client, config.run_workflow_id(run_id), RunWorkflow.status)
        except Exception as exc:  # noqa: BLE001 - "the run is gone" is an ANSWER, not always an error
            # A CLOSED execution is the run having FINISHED between two polls. The query can only
            # answer while the workflow is open, and a run that completes inside one poll interval
            # never produces a terminal status to read — MEASURED on the acceptance runs for the
            # publish-skip change: the ledger showed `run.finished`, and the command still exited
            # non-zero with `durable_query_failed`, so a finished, committed run read as a failure.
            #
            # The RESULT is the server's own record of how it ended, so it is asked for directly
            # rather than inferred from a poll that may not exist. `workflow_outcome` raises the
            # original error when the result cannot be read (or when the execution never completed),
            # so a genuinely vanished run still fails by name.
            if isinstance(exc, _cli.DurableCliError) and exc.code == "durable_query_failed":
                return await _exit_for_closed_run(client, config, run_id, cause=exc, printed=last)
            raise
        if status != last:
            _cli._print({"status": status})
            last = status
        if _run_is_finished(status):
            return 0 if status.get("outcome") == "completed" else EXIT_NOT_SUCCESS
        if stop is not None and stop.is_set():
            _cli._print(
                {
                    "detached": {
                        "run_id": run_id,
                        "workflow_id": config.run_workflow_id(run_id),
                        "note": "the run continues on the server; reattach with "
                        f"`codna durable status --run {run_id} --wait`",
                    }
                }
            )
            return EXIT_NOT_SUCCESS
        if time.monotonic() >= deadline:
            raise _error(
                f"run {run_id} did not finish within {timeout_s:g}s.",
                code="durable_run_wait_timeout",
                details={"run_id": run_id, "timeout_s": timeout_s, "last_status": status},
                hint="the run continues on the server; check `codna durable status --run`",
            )
        await asyncio.sleep(poll_s)


def _run_is_finished(status: Mapping[str, Any]) -> bool:
    """A run reports ``outcome`` only in its completion; a running one has not reached it."""
    return status.get("outcome") in ("completed", "budget_exhausted", "stopped")


async def workflow_completion(client: Any, workflow_id: str) -> Any:
    """The COMPLETION a closed workflow recorded, as whatever shape its client decodes.

    A closed workflow ended in exactly one of two ways, and a caller has to tell them apart:

    * it RETURNED a `RunCompletion` — the run finished, one way or another;
    * it FAILED — ``result()`` raises the workflow's own error, whose cause is the activity failure
      that ended it (``activity_pr_token_missing``, for the refusal this change introduced, and any
      other named code for any other failure).

    Both are ANSWERS about how the run ended, so this raises neither away nor into a generic query
    error: the caller reports the failure's own cause. Raises the SDK's error only when the execution
    is genuinely gone (retention), which is not an answer at all.
    """
    handle = client.get_workflow_handle(workflow_id)
    return await handle.result()


def workflow_outcome(completion: Any, *, workflow_id: str) -> str:
    """The outcome a RETURNED completion records.

    Deliberately strict: a result that is not a mapping or a `RunCompletion`, or carries no recognized
    outcome, RAISES — "we could not read how it ended" must never be reported as success. The
    recognized set is `RunOutcome`'s own vocabulary plus the `failed` the turn-terminal derivation
    already uses; nothing is inferred.
    """
    from .run_contracts import RunOutcome

    # EITHER the contract type OR a plain dict, and both are legitimate: a client that passes no
    # `result_type` (the CLI's own) decodes the payload to a dict, while a typed one gives
    # `RunCompletion`. Reading only the attribute would silently find nothing on the CLI's own path —
    # MEASURED on the acceptance run, where the fallback then re-raised and a finished run was
    # reported as a query failure. Both shapes are checked; neither is coerced.
    if isinstance(completion, Mapping):
        outcome = completion.get("outcome")
    else:
        outcome = getattr(completion, "outcome", None)
    allowed = set(RunOutcome.__args__) | {"failed"}
    if not isinstance(outcome, str) or outcome not in allowed:
        raise _error(
            f"workflow {workflow_id} closed but its result carries no recognized outcome "
            f"(got {outcome!r}).",
            code="durable_run_outcome_unknown",
            details={"workflow_id": workflow_id, "outcome": repr(outcome), "known": sorted(allowed)},
            hint="check `codna durable ledger --run <id>` for what the run recorded",
        )
    return outcome


def workflow_failure_report(error: BaseException) -> dict[str, Any] | None:
    """What a FAILED workflow reported, dug out of the SDK's wrapping — without inventing anything.

    A failed workflow's ``result()`` raises ``WorkflowFailureError``; the useful fact is the
    ``ApplicationError`` at the bottom of its ``cause`` chain, which is the error this package raised
    inside an activity. MEASURED on the live refusal run, the chain is
    ``WorkflowFailureError`` → ``ChildWorkflowError``/``ActivityError`` → ``ApplicationError``, and the
    ``ApplicationError`` carries the activity's own MESSAGE with ``type='ActivityError'``.

    So this reports ``{type, message, details}`` — the deepest error's type and text, plus its
    ``details`` when they are a structured mapping. It deliberately does NOT parse a code out of the
    message: "the code was in the sentence" is a guess, and a guess that happens to be right today
    would silently stop being right when the message is reworded.

    Total: any shape it cannot recognise returns None (the caller reports what it has).
    """
    deepest: Any = None
    node: Any = getattr(error, "cause", None)
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        deepest = node
        node = getattr(node, "cause", None)
    if deepest is None:
        return None
    details = getattr(deepest, "details", None)
    report: dict[str, Any] = {
        "type": str(getattr(deepest, "type", "") or type(deepest).__name__),
        "message": str(deepest),
    }
    if isinstance(details, Mapping) and details:
        report["details"] = dict(details)
    return report


async def _exit_for_closed_run(
    client: Any,
    config: "DurableConfig",
    run_id: str,
    *,
    cause: Exception,
    printed: Mapping[str, Any] | None,
) -> int:
    """The exit code for a run whose query was rejected because the workflow CLOSED.

    Zero only for a `completed` outcome read from the workflow's own result: the run did its job,
    whether or not it published anything (a local-only repository finishes `completed` with its task
    `approved_unpublished`, which is success). Every other ending is `EXIT_NOT_SUCCESS`, and each is
    reported for what it IS:

    * the workflow FAILED — the error that ended it is printed, so a user reads "opening a pull
      request needs a GitHub write token" rather than "query rejected, status: 3", which sends them
      looking for a server problem instead of at their configuration;
    * the workflow RETURNED a non-``completed`` outcome — that outcome is printed;
    * the execution is genuinely gone (retention), or the result cannot be read — ``cause``, the
      original named query failure, is re-raised. Nothing is converted into success.
    """
    workflow_id = config.run_workflow_id(run_id)
    try:
        completion = await workflow_completion(client, workflow_id)
    except Exception as exc:  # noqa: BLE001 - a FAILED run is an answer; a gone one is not
        failure = workflow_failure_report(exc)
        if failure is None:
            raise cause
        _cli._print(
            {
                "failed": {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    **failure,
                    "note": "the run's workflow FAILED; this is the error that ended it",
                    "last_status": dict(printed) if printed else None,
                }
            }
        )
        return EXIT_NOT_SUCCESS
    outcome = workflow_outcome(completion, workflow_id=workflow_id)
    _cli._print(
        {
            "closed": {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "outcome": outcome,
                "note": "the run's workflow had already closed; this outcome is its recorded result",
                "last_status": dict(printed) if printed else None,
            }
        }
    )
    return 0 if outcome == "completed" else EXIT_NOT_SUCCESS


async def query_workflow(client: Any, workflow_id: str, query: Any) -> dict[str, Any]:
    """Read one workflow's QUERY result, reporting a closed/unknown workflow as a CLI error.

    Public (not ``_query``) because `cli.py`'s status/approval commands query too. One definition:
    the "workflow may be gone" message and its hint must not differ between `status` and `--wait`.

    A REJECTED query is exactly the signal `_wait_run` turns into a result read (see the module
    docstring), so this function's error code is part of that contract rather than incidental: the
    caller distinguishes "the execution closed" from a real failure BY this code, and by nothing else.
    """
    from temporalio.common import QueryRejectCondition

    handle = client.get_workflow_handle(workflow_id)
    try:
        result = await handle.query(query, reject_condition=QueryRejectCondition.NOT_OPEN)
    except Exception as exc:  # noqa: BLE001 - a closed/unknown workflow is a CLI-level answer
        raise _error(
            f"could not query workflow {workflow_id}: {exc}",
            code="durable_query_failed",
            details={"workflow_id": workflow_id},
            hint="the workflow may not exist yet, or may have completed and been removed by retention",
        ) from exc
    return dict(result)


