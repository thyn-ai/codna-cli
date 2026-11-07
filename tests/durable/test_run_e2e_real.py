"""End to end, against a REAL Temporal dev server: does `codna run` actually drive a run?

Every other durable test uses ``WorkflowEnvironment.start_time_skipping()`` — a test server with no
real clock and, as it turns out, no ``DescribeTaskQueue``. That is right for control flow and wrong
for this question: "does the shipped `codna run` + `codna worker` pair move a run from start to a
parked approval against the server a user actually gets?" A test server answering a simplified wire
cannot prove the argv, the versioning registration, or the visibility queries work on real Temporal.

So this file starts the real thing — the same `DevServer` the CLI starts, with the same
``--dynamic-config-value matching.enableDeploymentVersions=true`` — and drives the real commands.

What it proves, and why each one needs a real server:

* ``codna run --no-worker`` starts a workflow the server has, on the configured queue, and the
  ``status`` query reads it back.
* The worker registers a PINNED deployment version, and the run it starts is PINNED to that build
  (the visibility record says so) — the property that stops a build bump migrating an in-flight run.
* The task child, started by the run workflow with no override of its own, inherits the parent's
  version and reaches the same worker. Measured: it does.
* ``approve`` with an identity completes the task, and the LEDGER (not the CLI's own output) records
  the route — so a green result cannot come from a CLI printing a plausible dict.

The activities are the REAL ones against the harness's fake sidecar (`tests/durable/fakes.py`): what
is under test here is Temporal + the CLI, not the engine.

Gated on ``CODNA_TEST_TEMPORAL=1``: it downloads the Temporal CLI on first use and needs network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

HAS_EXTRA = (
    importlib.util.find_spec("temporalio") is not None
    and importlib.util.find_spec("langgraph") is not None
)
GATED = os.environ.get("CODNA_TEST_TEMPORAL") == "1"

pytestmark = [
    pytest.mark.skipif(not GATED, reason="set CODNA_TEST_TEMPORAL=1 to run against a real dev server"),
    pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed"),
]

CLI_DIR = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _fakes() -> Any:
    """The harness module, loaded by PATH (a site-packages `tests` package can win `sys.path`)."""
    path = Path(__file__).resolve().parent / "fakes.py"
    spec = importlib.util.spec_from_file_location("durable_e2e_fakes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    # E3: `codna run` refuses to start without a valid `.codna/policy.yaml` — an unattended run with no
    # policy would execute under whatever the approval profile allows. A real governed repository
    # carries one, so the fixture does too.
    (path / ".codna").mkdir(exist_ok=True)
    (path / ".codna" / "policy.yaml").write_text(
        "version: 1\n"
        "tools:\n"
        "  allow: [read_files, search_codebase, editor, apply_patch, run_commands]\n"
        "write_scope:\n"
        "  paths: ['**']\n"
        "command_allow_list:\n"
        "  commands: [pytest, git]\n"
        "budget:\n"
        "  max_cost_usd: 100.0\n",
        encoding="utf-8",
    )
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "test@codna.ai"],
        ["git", "config", "user.name", "test"],
        # A REAL origin, because the durable run's last step publishes. Without one the step correctly
        # SKIPS (`publish.skipped` / `approved_unpublished`) — which is the contract its own tests
        # pin — and this test is about Temporal + the CLI reaching the PR, not about publishing being
        # unavailable. The push itself is the harness's scripted seam, so no network is used.
        ["git", "remote", "add", "origin", "https://github.com/thyn-ai/codna.git"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        completed = subprocess.run(
            [argv[0], "-C", str(path), *argv[1:]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    return path


class _RealServer:
    """The real `DevServer` from `codna.durable.dev_server`, plus a connected client loop.

    The Temporal dev server is a separate PROCESS (started detached), so this class only owns the
    client loop and the stop call. That is the difference from the test-server helper in
    `test_run_cli.py`, which owns the server inside its own thread.
    """

    def __init__(self, config: Any, settings: Any) -> None:
        from codna.durable.dev_server import DevServer

        self._config = config
        self._dev = DevServer(settings)
        self._loop: Any = None
        self._thread: Any = None
        self._client: Any = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None

    def start(self) -> None:
        import asyncio

        started = self._dev.ensure_started(timeout_s=120.0, headless=True)
        assert started.running, started.to_payload()
        self._loop = asyncio.new_event_loop()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            try:
                from temporalio.client import Client

                self._client = self._loop.run_until_complete(
                    Client.connect(self._config.address, namespace=self._config.namespace)
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced by `connect`
                self._failure = exc
            finally:
                self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, name="durable-e2e-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=60) or self._failure is not None:
            raise AssertionError(f"the client never connected: {self._failure!r}")

    def connect(self, _config: Any) -> Any:
        """The CLI's `_connect` seam: hand back the client living on the helper's loop."""

        async def resolve() -> Any:
            return self._client

        return resolve()

    def call(self, coro: Any, *, timeout: float = 120.0) -> Any:
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def wait_parked(self, workflow_id: str, *, attempts: int = 1500) -> dict[str, Any]:
        import asyncio

        async def poll() -> dict[str, Any]:
            handle = self._client.get_workflow_handle(workflow_id)
            for _ in range(attempts):
                try:
                    status = await handle.query("status")
                except Exception:  # noqa: BLE001 - not created yet
                    await asyncio.sleep(0.02)
                    continue
                if status.get("awaiting_approval"):
                    return status
                await asyncio.sleep(0.02)
            raise AssertionError(f"{workflow_id} never parked")

        return self.call(poll())

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=20)
        if self._loop is not None:
            # temporalio's Client exposes no close/shutdown in 1.34; dropping the reference on a
            # stopped loop is the documented way to disconnect.
            self._client = None
            self._loop.close()
        try:
            self._dev.stop(timeout_s=40.0)
        except Exception:  # noqa: BLE001 - teardown must not mask the real failure
            pass


_ACTIVE: list[_RealServer] = []


@pytest.fixture(autouse=True)
def _stop_servers() -> Any:
    yield
    while _ACTIVE:
        _ACTIVE.pop().stop()


def _dev_settings(tmp_path: Path, port: int) -> Any:
    """Dev-server settings with the database UNDER the test.

    Sharing the developer's real ``~/.codna/temporal/dev.db`` would make this test depend on whatever
    promotion history that server already has — and, worse, would write the test's own deployment
    versions into a database belonging to the user. Each test gets its own SQLite file and pid.
    """
    from codna.durable.dev_server import DevServerSettings, default_lock_path

    root = tmp_path / "temporal"
    return DevServerSettings(
        host="127.0.0.1",
        port=port,
        namespace="codna",
        db_path=root / "dev.db",
        bin_dir=root / "bin",
        log_path=tmp_path / "logs" / "temporal-dev.log",
        pid_path=root / "dev-server.json",
        lock_path=default_lock_path(),
    )


def _durable_config(tmp_path: Path, port: int, *, engine: str = "plain") -> Any:
    """A `DurableConfig` on the test's free port, with runs under the test's own root."""
    from codna.durable.config import load_durable_config
    from codna.runtime.config import RuntimeConfig, build_runtime_paths

    paths = build_runtime_paths()
    runtime = RuntimeConfig(
        codna_home=tmp_path,
        port_base=18600,
        engine_port=18600,
        sidecar_port=18601,
        engine_url="http://127.0.0.1:18600",
        sidecar_url="http://127.0.0.1:18601",
        remote_engine_url=None,
        ignored_engine_url=None,
        engine_url_override=None,
        paths=paths,
        engine_dir=tmp_path / "engine",
        engine_python="python3",
        engine_env_file=tmp_path / ".env",
        sidecar_dir=tmp_path / "agent-core",
        sidecar_binary=None,
        cline_data_dir=tmp_path / "cline-data",
        codna_cli_version="0.0.0-test",
        engine_build_id="engine",
        sidecar_build_id="sidecar",
        runtime_config_hash="hash",
        allow_loopback_override=False,
    )
    env = {
        "CODNA_TEMPORAL_ADDRESS": f"127.0.0.1:{port}",
        "CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "runs"),
        "CODNA_DURABLE_ENGINE": engine,
    }
    return load_durable_config(env=env, runtime=runtime, hostname="e2e-host")


def test_run_starts_a_pinned_run_and_a_worker_executes_it_to_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The whole shipped path: real server, real worker registration, real `codna run`, real ledger."""
    from codna.durable import cli as durable_cli
    from codna.durable import run_cli
    from codna.durable.worker import registered_build_ids, worker_settings_from_config

    repo = _git_repo(tmp_path / "repo")
    fakes = _fakes()
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.verify_passes = [False, True, True]
    harness.review_conclusions = ["success"]

    port = _free_port()
    config = _durable_config(tmp_path, port)
    server = _RealServer(config, _dev_settings(tmp_path, port))
    server.start()
    _ACTIVE.append(server)

    monkeypatch.setattr(durable_cli, "_connect", server.connect)
    monkeypatch.setattr(durable_cli, "_config", lambda _args: config)

    # 1) The run's queue must have a worker before a pin can be satisfied — this is the ordering a
    #    user gets from `codna run` (which starts its own worker); here the worker is explicit so the
    #    pin path is what `--no-worker` + `--pin` exercises.
    settings = worker_settings_from_config(config, ensure_sidecar=False)
    builds = server.call(registered_build_ids(server._client, settings))
    assert builds == set(), "nothing should be registered before a worker starts"

    # 2) Start a worker the way `codna worker` does, in-process for the duration of the test.
    import asyncio

    from codna.durable.worker import run_worker

    stop = asyncio.Event()
    started = threading.Event()
    worker_failure: list[BaseException] = []

    async def worker_main() -> None:
        try:
            await run_worker(
                settings,
                stop=stop,
                install_signal_handlers=False,
                runtime=harness.runtime,
                client=server._client,
                on_started=lambda _b: started.set(),
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            worker_failure.append(exc)

    worker_task = asyncio.run_coroutine_threadsafe(worker_main(), server._loop)
    assert started.wait(timeout=120), "the worker never registered"
    assert not worker_failure, worker_failure

    # 3) NOW the build is registered, so the default (pin-when-available) pins the run to it.
    builds = server.call(registered_build_ids(server._client, settings))
    assert settings.build_id in builds, builds

    args = durable_cli.cmd_durable.__globals__  # noqa: F841 - keep the module referenced
    from codna import cli as top_cli

    run_args = top_cli.build_parser().parse_args(
        [
            "durable",
            "run",
            "--repo",
            str(repo),
            "--brief",
            "Fix the flaky retry",
            "--base-branch",
            "main",
            "--run-id",
            "run-e2e",
            "--no-start-server",
            "--pin",
        ]
    )
    assert run_cli.cmd_run(run_args) == 0
    banner = json.loads(capsys.readouterr().out)
    assert banner["started"]["run_id"] == "run-e2e"
    assert banner["started"]["versioning"]["pinned"] is True, banner
    assert banner["started"]["versioning"]["build_id"] == settings.build_id

    # 4) The task child inherits the parent's version and reaches the same worker: it parks.
    child = "task-run-e2e-task-1"
    parked = server.wait_parked(child)
    assert parked["task_id"] == "task-1"

    # 5) NOW the server has actually dispatched it: the version stamp names this build. Asserted after
    #    the park, not at start, because `most_recent_worker_version_stamp` is written when a workflow
    #    TASK runs — describing a just-started run reports an empty stamp on a healthy pinned run.
    async def describe() -> tuple[str, str]:
        handle = server._client.get_workflow_handle(config.run_workflow_id("run-e2e"))
        described = await handle.describe()
        raw = described.raw_description.workflow_execution_info
        return raw.worker_deployment_name, raw.most_recent_worker_version_stamp.build_id

    deployment, build = server.call(describe())
    assert deployment == "codna", deployment
    assert build == settings.build_id, (build, settings.build_id)

    # 6) Approving completes it, and the LEDGER — not the CLI's output — records the route.
    assert run_cli.cmd_run  # the CLI's own module is what ran
    approve_args = top_cli.build_parser().parse_args(
        ["durable", "approve", "--run", "run-e2e", "--task", "task-1", "--as", "e2e@codna.ai"]
    )
    assert durable_cli.cmd_durable(approve_args) == 0
    capsys.readouterr()
    route = harness.route("run-e2e", "task-1")
    assert route[0] == "plan" and route[-1] == "open_pr", route
    assert harness.checkpoints("run-e2e"), "no checkpoint was recorded"

    # 7) Draining stops the worker, and the stop is what the next `run` depends on.
    server._loop.call_soon_threadsafe(stop.set)
    worker_task.result(timeout=120)
    assert not worker_failure, worker_failure
