"""Record a replay cassette for `codna.durable.workflows.TaskWorkflow`.

Run this after a **`GRAPH_VERSION` bump** (or any change to the task graph's nodes or edges), and
commit the result:

    /tmp/codna-venv/bin/python -m tests.durable.record_workflow_history

Why it is a human-run step and not something pytest does: a cassette is a BUILD ARTIFACT that pins a
worker version. Re-recording it on every test run would make the replay suite assert "the code agrees
with itself", which is worth nothing — the whole point is that a change is caught, and the legitimate
response to a caught change is a version bump plus a new cassette.

The recording is of the REAL workflow (`TaskWorkflow`, real activities, real git, a fake sidecar), not
the D2.0 spike's toy graph, so the cassette covers the nodes, the conditional edges, the approval
interrupt and the checkpoint — every command a production run actually issues.

It also stamps ``codna_graph_version`` into the file. That key is not part of Temporal's JSON schema,
and the Replayer ignores unknown top-level keys; it exists so a stale cassette is detectable on its
own rather than only via the replay failing.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from codna.durable.activities import ACTIVITIES, activity_executor
from codna.durable.graph import build_task_graph
from codna.durable.version import GRAPH_VERSION
from codna.durable.workflows import TaskWorkflow
from tests.durable import fakes
from tests.durable.workflow_specs import TASK_QUEUE, scripted_spec

HISTORIES = Path(__file__).resolve().parent / "histories"
WORKFLOW_ID = "recorded-task-workflow"


async def record() -> dict:
    """Run the task workflow once, end to end, and return its history as JSON."""
    with tempfile.TemporaryDirectory(prefix="codna-cassette-") as scratch:
        repo = fakes.make_repo(Path(scratch) / "repo")
        runs_root = repo / "runs"
        runs_root.mkdir(exist_ok=True)
        harness = fakes.build_harness(repo, runs_root)
        # The scenario the cassette must cover: a verify failure and a review rejection, so the
        # history contains the loop's conditional edges as well as the straight path.
        harness.verify_passes = [False, True, True]
        harness.review_conclusions = ["neutral", "success"]
        spec = scripted_spec(repo)

        env = await WorkflowEnvironment.start_time_skipping()
        try:
            task_graph = build_task_graph()
            with activity_executor() as executor:
                async with Worker(
                    env.client,
                    task_queue=TASK_QUEUE,
                    workflows=[TaskWorkflow],
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                    plugins=[task_graph.plugin],
                ):
                    handle = await env.client.start_workflow(
                        TaskWorkflow.run,
                        args=[spec.to_payload(), None],
                        id=WORKFLOW_ID,
                        task_queue=TASK_QUEUE,
                        run_timeout=timedelta(minutes=5),
                    )
                    for _ in range(600):
                        status = await handle.query(TaskWorkflow.status)
                        if status.get("awaiting_approval"):
                            break
                        await asyncio.sleep(0.02)
                    else:
                        raise RuntimeError("the run never parked on the approval")
                    await handle.execute_update(
                        TaskWorkflow.approve, {"identity": "recorded@codna.ai"}
                    )
                    await handle.result()
                    history = await handle.fetch_history()
        finally:
            await env.shutdown()
    payload = json.loads(history.to_json())
    payload["codna_graph_version"] = GRAPH_VERSION
    return payload


def main(argv: list[str]) -> int:
    _ = argv
    HISTORIES.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(record())
    target = HISTORIES / f"task-workflow-g{GRAPH_VERSION}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {target} ({len(payload.get('events', []))} events, graph g{GRAPH_VERSION})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
