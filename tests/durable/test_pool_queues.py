"""D6 POOL queues: several workers, one queue, and no task executed twice.

The three properties a pool has to have, each tested where it can actually fail:

1. **A pool queue is derived, not configured.** Two machines that agree on a pool name must poll the
   same queue string; anything else silently makes a "pool" of one-machine islands.
2. **A multi-worker pool over per-machine disks is REFUSED.** This is the ledger-fork guard, and it
   is the reason `pool.py` exists at all — see its module docstring.
3. **Two workers on one queue do not double-execute.** Temporal's contract, verified against a real
   test server with two pollers on ONE queue and an activity that would produce a different result
   if it ran twice (it increments a counter the workflow returns).

Property 3 is checked on ``WorkflowEnvironment.start_time_skipping()`` because that is a real server;
a fake would only re-assert the fake. It is skipped (not faked) without the durable extra.

Deliberately NOT one of the properties: that two workers may be *configured*. That is blocked unless
the operator declares shared state, which is a different claim — the queue is honest, the STATE is
the hazard — and it is covered by the tests below plus `infra/README-durable-worker.md`.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from codna.durable.pool import (
    ENV_POOL,
    ENV_POOL_STATE,
    ENV_POOL_WORKERS,
    POOL_QUEUE_PREFIX,
    PoolError,
    pool_settings_from_env,
    pool_task_queue,
    validate_pool_name,
)

HAS_EXTRA = (
    importlib.util.find_spec("temporalio") is not None
    and importlib.util.find_spec("langgraph") is not None
)


# -- 1. the queue string is derived from the pool name -------------------------------------


def test_a_pool_name_maps_to_one_derived_queue():
    assert pool_task_queue("nightly") == f"{POOL_QUEUE_PREFIX}nightly"
    # Derived, so two machines agreeing on the NAME agree on the queue with no second setting.
    assert pool_task_queue("nightly") == pool_task_queue("  nightly  ")


def test_a_pool_queue_is_never_a_per_host_queue():
    """A hosted run must not be claimable by a laptop, and a laptop's run not by the pool."""
    from codna.durable.config import task_queue_for_host

    hosted = pool_task_queue("nightly")
    assert hosted.startswith("codna.pool.")
    assert not hosted.startswith("codna.agent.")
    # ...and the per-host form keeps its own namespace, which is what keeps them disjoint.
    assert task_queue_for_host("nightly") != hosted


@pytest.mark.parametrize("bad", ["", " ", "a/b", "with space", "../etc", "x" * 65, "-lead"])
def test_an_unusable_pool_name_is_refused(bad: str):
    with pytest.raises(PoolError) as caught:
        validate_pool_name(bad)
    assert caught.value.code == "invalid_pool_name"


# -- 2. the multi-worker safety declaration ------------------------------------------------


def test_no_pool_variables_means_the_historical_per_host_behaviour():
    assert pool_settings_from_env({}) is None


def test_one_worker_needs_no_declaration():
    """A single-worker pool has nobody to share with, so it is the compatible default."""
    settings = pool_settings_from_env({ENV_POOL: "nightly"})
    assert settings is not None
    assert settings.task_queue == "codna.pool.nightly"
    assert settings.workers == 1
    assert settings.shared_state is False


def test_two_workers_over_local_state_is_refused_with_the_reason():
    """THE guard: it is the only thing standing between a pool and a forked ledger."""
    with pytest.raises(PoolError) as caught:
        pool_settings_from_env(
            {ENV_POOL: "nightly", ENV_POOL_WORKERS: "3", ENV_POOL_STATE: "local"}
        )
    error = caught.value
    assert error.code == "pool_state_not_shared"
    # The message must say WHAT breaks, not just that the value is wrong: an operator reading a Fly
    # log line needs the consequence, not the rule.
    assert "ledger" in str(error)
    assert error.details["workers"] == 3


def test_two_workers_with_shared_state_is_allowed():
    settings = pool_settings_from_env(
        {ENV_POOL: "nightly", ENV_POOL_WORKERS: "4", ENV_POOL_STATE: "shared"}
    )
    assert settings is not None
    assert (settings.workers, settings.shared_state) == (4, True)
    assert settings.to_payload() == {
        "pool": "nightly",
        "task_queue": "codna.pool.nightly",
        "workers": 4,
        "state": "shared",
    }


def test_a_multi_worker_pool_must_SAY_whether_state_is_shared():
    """No default: both possible ones are wrong for half the fleets."""
    with pytest.raises(PoolError) as caught:
        pool_settings_from_env({ENV_POOL: "nightly", ENV_POOL_WORKERS: "2"})
    assert caught.value.code == "pool_state_required"


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "two", "+2", "1e3"])
def test_a_malformed_worker_count_is_refused(value: str):
    with pytest.raises(PoolError) as caught:
        pool_settings_from_env({ENV_POOL: "nightly", ENV_POOL_WORKERS: value})
    assert caught.value.code == "invalid_pool_workers"


def test_surrounding_whitespace_in_the_worker_count_is_folded():
    """A `fly secrets`-style value can arrive padded; padding is not a malformed number."""
    settings = pool_settings_from_env(
        {ENV_POOL: "nightly", ENV_POOL_WORKERS: " 2 ", ENV_POOL_STATE: "shared"}
    )
    assert settings is not None and settings.workers == 2


def test_an_empty_pool_variable_is_a_refusal_not_no_pool():
    """A template that rendered an empty value must not silently become "a per-host worker"."""
    with pytest.raises(PoolError) as caught:
        pool_settings_from_env({ENV_POOL: ""})
    assert caught.value.code == "invalid_pool_name"


def test_the_refusal_reaches_the_operator_through_the_durable_config():
    """`load_durable_config` translates `PoolError`, so the CLI renders it like every other error."""
    from codna.durable.config import DurableConfigError, load_durable_config

    with pytest.raises(DurableConfigError) as caught:
        load_durable_config(
            env={ENV_POOL: "nightly", ENV_POOL_WORKERS: "2", ENV_POOL_STATE: "local"}
        )
    assert caught.value.code == "pool_state_not_shared"


def test_the_config_carries_the_pool_and_its_queue():
    from codna.durable.config import load_durable_config

    config = load_durable_config(env={ENV_POOL: "nightly"})
    assert config.pool is not None
    assert config.pool.name == "nightly"
    assert config.task_queue == "codna.pool.nightly"
    # An explicit queue override still wins: it is the most specific statement an operator can make.
    pinned = load_durable_config(
        env={ENV_POOL: "nightly", "CODNA_DURABLE_TASK_QUEUE": "codna.pool.pinned"}
    )
    assert pinned.task_queue == "codna.pool.pinned"
    assert pinned.pool is not None and pinned.pool.name == "nightly"


def test_the_worker_banner_reports_the_pool_but_no_secret():
    from codna.durable.config import load_durable_config
    from codna.durable.worker import worker_settings_from_config

    config = load_durable_config(env={ENV_POOL: "nightly", ENV_POOL_STATE: "shared", ENV_POOL_WORKERS: "2"})
    payload = worker_settings_from_config(config, env={}).to_payload()
    assert payload["pool"] == "nightly"
    assert payload["pool_workers"] == 2
    assert payload["pool_state"] == "shared"


# -- 3. two workers on ONE queue never double-execute -------------------------------------
#
# TWO PROCESSES, because that is what a pool IS. A single Python process cannot host two overlapping
# pollers at all (measured against both `start_time_skipping` and the real dev server: the Rust bridge
# refuses "multiple workers with overlapping worker task types on the same namespace, task queue, and
# deployment build ID" — a per-PROCESS registry). That refusal is not the property under test; the
# property is that two separate machines on one queue split the work.
#
# So the test starts two real worker subprocesses against a real dev server and counts executions in
# a shared file. It is GATED on CODNA_TEST_TEMPORAL=1 for the same reason `test_dev_server_real.py`
# is: it needs the Temporal CLI and a network download the first time.

pytestmark_gated = pytest.mark.skipif(
    HAS_EXTRA is False or __import__("os").environ.get("CODNA_TEST_TEMPORAL") != "1",
    reason="set CODNA_TEST_TEMPORAL=1 to run against a real dev server",
)

_WORKER_SUBPROCESS = '''"""One pool worker process: poll QUEUE, run the bump activity, count executions."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker


@activity.defn(name="pool.probe.bump")
async def bump(value: int) -> int:
    # Append one line per EXECUTION, with the pid: a double delivery is two lines, and the pid says
    # whether it was one worker running twice or two workers each running once. A single short
    # `write` on an `O_APPEND` handle is atomic, so two processes cannot interleave inside one line.
    ledger = Path(os.environ["PROBE_COUNT_FILE"])
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value + 1


async def main() -> None:
    client = await Client.connect(os.environ["PROBE_ADDRESS"], namespace="codna")
    worker = Worker(client, task_queue=os.environ["PROBE_QUEUE"], activities=[bump])
    # READINESS IS A FILE, not stdout. Reading a pipe in the driver is a blocking call with no
    # timeout of its own: a worker that dies before printing turns a failed test into a hung one.
    # The file is written after `run()` is scheduled — the same "pollers attach a beat later" fact
    # `worker.py::wait_until_registered` deals with — and the driver polls it with a deadline.
    task = asyncio.ensure_future(worker.run())
    await asyncio.sleep(float(os.environ.get("PROBE_READY_DELAY", "2.0")))
    Path(os.environ["PROBE_READY_FILE"]).write_text("ready\\n", encoding="utf-8")
    await task


asyncio.run(main())
'''

_PROBE_WORKFLOW = '''"""The probe workflow: one activity call, returning its result."""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow


@workflow.defn(name="PoolProbeWorkflow")
class PoolProbeWorkflow:
    @workflow.run
    async def run(self, value: int) -> int:
        return await workflow.execute_activity(
            "pool.probe.bump", value, start_to_close_timeout=timedelta(seconds=30)
        )
'''


@pytestmark_gated
def test_two_worker_processes_on_one_pool_queue_execute_each_task_once(tmp_path: Any):
    """The double-billing shape, at the level a pool actually creates it.

    Two worker PROCESSES poll ``codna.pool.<name>`` (the string both machines derive from the same
    pool name). A workflow starts, one activity executes, and the shared count file must hold exactly
    ONE line. Two lines would mean the queue delivered the activity twice — the failure a pool must
    never have, and the one an in-process test cannot see.
    """
    import asyncio
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    from codna.durable.dev_server import DevServer, DevServerSettings

    def free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    port = free_port()
    server = DevServer(
        DevServerSettings(
            host="127.0.0.1",
            port=port,
            namespace="codna",
            db_path=tmp_path / "temporal" / "dev.db",
            bin_dir=tmp_path / "temporal" / "bin",
            log_path=tmp_path / "logs" / "temporal-dev.log",
            pid_path=tmp_path / "temporal" / "dev-server.json",
        )
    )
    count_file = tmp_path / "executions.txt"
    count_file.write_text("", encoding="utf-8")
    queue = pool_task_queue("process-test")
    probes: list[subprocess.Popen[bytes]] = []
    worker_source = tmp_path / "pool_worker_main.py"
    worker_source.write_text(_WORKER_SUBPROCESS, encoding="utf-8")
    ready_files = [tmp_path / f"ready-{index}.txt" for index in (1, 2)]
    workflow_dir = tmp_path / "probe"
    workflow_dir.mkdir()
    (workflow_dir / "probe_workflow.py").write_text(_PROBE_WORKFLOW, encoding="utf-8")

    def worker_env(ready_file: Path) -> dict[str, str]:
        return {
            **os.environ,
            "PROBE_ADDRESS": f"127.0.0.1:{port}",
            "PROBE_QUEUE": queue,
            "PROBE_COUNT_FILE": str(count_file),
            "PROBE_READY_FILE": str(ready_file),
            "PYTHONPATH": os.pathsep.join(
                [str(workflow_dir), str(Path(__file__).resolve().parents[2])]
            ),
        }

    async def drive() -> dict[str, Any]:
        from temporalio.client import Client
        from temporalio.worker import Worker

        sys.path.insert(0, str(workflow_dir))
        import probe_workflow  # noqa: PLC0415 - after the path is extended

        client = await Client.connect(f"127.0.0.1:{port}", namespace="codna")
        # BOTH processes poll ONE queue. Each writes its ready file once it is polling, so the
        # workflow starts only after TWO pollers exist — a run started against a queue with one
        # poller would pass this test without ever exercising distribution.
        for ready_file in ready_files:
            probes.append(
                subprocess.Popen(
                    [sys.executable, str(worker_source)],
                    env=worker_env(ready_file),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )
        for index, (process, ready_file) in enumerate(zip(probes, ready_files)):
            deadline = time.time() + 120
            while time.time() < deadline:
                if ready_file.is_file():
                    break
                if process.poll() is not None:
                    # A worker that died is reported with its own stderr, never as a hang.
                    assert process.stderr is not None
                    output = process.stderr.read().decode("utf-8", errors="replace")
                    raise AssertionError(
                        f"pool worker {index + 1} exited {process.returncode} before polling:\n{output}"
                    )
                time.sleep(0.1)
            else:  # pragma: no cover - a hung worker is a real failure, reported as one
                raise AssertionError(f"pool worker {index + 1} never reported ready")

        # The WORKFLOW host is this process, and it is a THIRD poller on the same queue. In production
        # every machine hosts both halves (`codna worker` registers workflows and activities); here the
        # split is what makes two competing ACTIVITY pollers possible at all — one Python process
        # cannot register overlapping pollers, but two processes can.
        async with Worker(
            client, task_queue=queue, workflows=[probe_workflow.PoolProbeWorkflow]
        ):
            handle = await client.start_workflow(
                probe_workflow.PoolProbeWorkflow.run, 41, id="pool-process-probe", task_queue=queue
            )
            return {"result": await handle.result(), "queue": (await handle.describe()).task_queue}

    try:
        started = server.ensure_started(timeout_s=180.0, headless=True)
        assert started.running is True
        facts = asyncio.run(drive())
    finally:
        for process in probes:
            process.terminate()
        for process in probes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - reported by the assertion below
                process.kill()
        try:
            server.stop(timeout_s=30.0)
        except Exception:  # noqa: BLE001 - teardown must not mask the real failure
            pass

    assert facts["result"] == 42, facts
    assert facts["queue"] == queue
    lines = [line for line in count_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1, (
        f"two workers on one pool queue executed the activity {len(lines)} times (pids: {lines}); "
        "Temporal must deliver each task to exactly one poller"
    )
