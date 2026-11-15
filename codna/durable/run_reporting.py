"""Run-level reporting: the ONE summary a run writes about tasks that could not publish.

── WHY THIS IS ITS OWN MODULE ──────────────────────────────────────────────────────────────────

`run_workflow.py` is held under this package's 1000-line modularity ceiling
(`tests/test_modularity.py`), and the stop-delivery work pushed it over. The ceiling exists to force
a split before a module sprawls, so the fix is to move a cohesive concern out rather than raise the
number.

── WHAT IT REPORTS ─────────────────────────────────────────────────────────────────────────────

Each task writes its own `publish.skipped` note when it finishes without opening a pull request. A
reader of the RUN's ledger should not have to walk every task's records to answer "did this run open
anything?", so the run writes one summary naming them.

A run whose tasks all published writes nothing new. That is deliberate and load-bearing: an existing
history replays with exactly the record set it recorded, so this function cannot introduce a command
into a replay that did not produce it.

It is a NOTIFICATION, never a failure. The work is done; "the repository has no remote to publish to"
is a fact about the repository, not a defect in the run.

No clock, no I/O, no randomness: importable from the Temporal sandbox.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Protocol


class _HasUnpublished(Protocol):
    """The one property this module reads from a task completion."""

    task_id: str
    unpublished: bool


async def report_publish_skips(
    completed: Iterable[Any],
    run_id: str,
    *,
    notify: Callable[[str, str], Awaitable[None]],
) -> None:
    """Send ONE note naming the tasks that finished without a pull request.

    ``notify`` is the run's own note sink, injected so this stays free of workflow state and testable
    without one.
    """
    skipped = tuple(summary for summary in completed if getattr(summary, "unpublished", False))
    if not skipped:
        return
    names = ", ".join(summary.task_id for summary in skipped)
    await notify(
        "note",
        f"run {run_id}: {len(skipped)} task(s) finished without a pull request "
        f"({names}): the repository has nothing to publish to (no GitHub remote); the commits "
        "are on each task's branch.",
    )
