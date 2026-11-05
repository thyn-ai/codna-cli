"""The task spec the workflow tests and the cassette share.

One module so the recorded history, the replay test and the scenario tests cannot drift: a cassette
recorded against a different ``brief``/``repo_dir`` would still replay (the Replayer compares
commands, not inputs), but the tests that assert WHICH route was taken would then be reading a
cassette for a different route. Keeping the spec in one place makes that impossible.

``TASK_QUEUE`` lives here too: the recorder and the tests must use the same queue, and a queue name
is exactly the kind of string that gets edited in one file and not the other.

No `pytest` import, no activity bodies — this module is imported by the recorder, which constructs a
worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from codna.durable.contracts import TaskSpec

TASK_QUEUE: Final = "codna-durable-workflow-tests"


def task_spec(repo: Path, *, engine: str = "plain", **overrides: Any) -> TaskSpec:
    """A valid spec against ``repo``, with the ids a test or the recorder needs."""
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the flaky retry in the client",
        "repo_dir": str(repo),
        "verify_commands": ("true",),
        "engine": engine,
    }
    payload.update(overrides)
    spec = TaskSpec(**payload)
    spec.validate()
    return spec


def scripted_spec(repo: Path, *, engine: str = "langgraph", **overrides: Any) -> TaskSpec:
    """The cassette's spec: the scenario with one verify failure and one review rejection.

    The engine defaults to ``langgraph`` because a cassette is recorded for `TaskWorkflow`; a test
    that wants the plain engine passes it explicitly.
    """
    return task_spec(repo, engine=engine, **overrides)
