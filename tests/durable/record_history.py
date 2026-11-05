"""Record a replay history for the committed replay cassette.

Run this after a `GRAPH_VERSION` bump (or when the graph's shape changes) to regenerate
``cli/tests/durable/histories/task-graph-g<GRAPH_VERSION>.json``:

    /tmp/codna-venv/bin/python -m tests.durable.record_history

It needs the `durable` extra and a working Temporal test server (the same
``WorkflowEnvironment.start_time_skipping()`` the spike uses), so it is an explicit, human-run step
rather than something pytest does. The file it writes is what `test_spike_langgraph_temporal.py`
replays on every run — the Public-Preview safety net the plan asks for: if the graph's commands
change, the replay fails and the version bump is the only legitimate fix.

The history is written as the SDK's own JSON, which is stable across SDK patch releases and
readable in review — a binary protobuf would make "what changed in this cassette" unreviewable.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from codna.durable.graph import GRAPH_NAME, build_task_graph
from codna.durable.version import GRAPH_VERSION
from tests.durable.spike_workflows import (
    RUN_TIMEOUT,
    STUB_NODES,
    TASK_QUEUE,
    SpikeTaskWorkflow,
    initial_state,
)

HISTORIES_DIR = Path(__file__).resolve().parent / "histories"


def cassette_path(graph_version: int = GRAPH_VERSION) -> Path:
    return HISTORIES_DIR / f"task-graph-g{graph_version}.json"


async def record(workflow_id: str = "recorded-task-graph") -> dict:
    """Run the task graph once, end to end, and return the history as JSON."""
    env = await WorkflowEnvironment.start_time_skipping()
    try:
        client = env.client
        task_graph = build_task_graph(nodes=STUB_NODES)
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[SpikeTaskWorkflow],
            plugins=[task_graph.plugin],
        ):
            handle = await client.start_workflow(
                SpikeTaskWorkflow.run,
                initial_state(),
                id=workflow_id,
                task_queue=TASK_QUEUE,
                run_timeout=RUN_TIMEOUT,
            )
            for _ in range(300):
                status = await handle.query(SpikeTaskWorkflow.status)
                if status.get("interrupts"):
                    break
                await asyncio.sleep(0.02)
            else:
                raise RuntimeError("the run never parked on the approval node")
            await handle.execute_update(
                SpikeTaskWorkflow.approve, {"identity": "recorded@codna.ai", "approved": True}
            )
            await handle.result()
            history = await handle.fetch_history()
        return json.loads(history.to_json())
    finally:
        await env.shutdown()


def main(argv: list[str]) -> int:
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(record())
    target = cassette_path()
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(payload.get('events', []))} events, graph {GRAPH_NAME} g{GRAPH_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
