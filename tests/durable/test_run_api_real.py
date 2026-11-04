"""The Run API against a REAL Temporal server: does `TemporalRunBackend` actually drive one?

Every other test in `test_run_api.py` uses a scripted `RunBackend`, which is right for the transport
(the HTTP surface is what those tests are about) and useless for the question here: does the real
backend — the one that imports `run_cli._start_run`, `query_workflow`, `cli._send_update` and
`cli._signal` — actually start a workflow, read its status, and answer a parked task's Update?

So this file starts a real time-skipping Temporal server, a real worker with the REAL activities (over
the harness's fake sidecar), the real `TemporalRunBackend`, and the REAL HTTP server. The four steps
are the four things a client does, in order:

1. ``POST /runs`` -> 202, and the server really has the workflow (a `describe()` says so);
2. ``GET /runs/:id`` -> the status query's answer, plus the ledger the activities wrote;
3. ``GET /runs/:id/events`` -> the ledger as SSE, in the frames the IDE's parser reads;
4. ``POST /runs/:id/steer`` and ``/approve`` -> the workflow's own Updates, with the identity the
   request carried landing in the task's approval list.

Gated on ``CODNA_TEST_TEMPORAL=1`` (like `test_run_e2e_real.py`): the time-skipping server is a
downloaded binary, so this test is not part of the base suite.
"""
from __future__ import annotations

import http.client
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


def _fakes() -> Any:
    """The harness module, loaded by PATH (a site-packages `tests` package can win `sys.path`)."""
    path = Path(__file__).resolve().parent / "fakes.py"
    spec = importlib.util.spec_from_file_location("run_api_real_fakes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "test@codna.ai"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        completed = subprocess.run(
            [argv[0], "-C", str(path), *argv[1:]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    return path


class _TestServer:
    """A time-skipping Temporal server plus a worker, on their own loop in a thread.

    A `TemporalRunBackend` opens ITS OWN loop thread and connects to this server's address, so this
    class must expose the address rather than a client: that is what makes the backend the real thing
    under test instead of a client handed to it.
    """

    def __init__(self, config: Any, repo: Path) -> None:
        self._config = config
        self._repo = repo
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._env: Any = None
        self._stopping: Any = None
        self._harness_instance: Any = None
        self.address: str = ""

    def start(self) -> None:
        import asyncio

        started = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve(started))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                failure.append(exc)
                started.set()

        self._thread = threading.Thread(target=run, name="run-api-test-temporal", daemon=True)
        self._thread.start()
        assert started.wait(timeout=120), "the test server never started"
        assert not failure, failure

    async def _serve(self, started: threading.Event) -> None:
        import asyncio

        from temporalio.testing import WorkflowEnvironment
        from temporalio.worker import Worker

        from codna.durable.activities import ACTIVITIES, activity_executor
        from codna.durable.workflows import registered_workflows

        self._stopping = asyncio.Event()
        self._env = await WorkflowEnvironment.start_time_skipping()
        try:
            # The address the SERVER listens on: `TemporalRunBackend` must find it from the config.
            self.address = self._env.client.service_client.config.target_host
            self._harness_instance = self._harness()
            with activity_executor() as executor:
                async with Worker(
                    self._env.client,
                    task_queue=self._config.task_queue,
                    workflows=list(registered_workflows()),
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                ):
                    started.set()
                    # Time skipping OFF: a parked task's `run_timeout` is a timer, and skipping past
                    # it would end the run while the test is still asking it questions.
                    with self._env.auto_time_skipping_disabled():
                        await self._stopping.wait()
        finally:
            await self._env.shutdown()

    def _harness(self) -> Any:
        """The harness the WORKER uses: a fake sidecar and scripted verify/review.

        The activities read a process-wide runtime, so the harness is created here (once the server
        exists) rather than injected: the test's contract is the HTTP surface, not the harness.

        Built here rather than injected because the activities read a process-wide runtime; the test
        only needs the API's view of the run, so the harness is a fixture of the environment.
        """
        fakes = _fakes()
        return fakes.build_harness(self._repo, self._config.runs_root, cost_per_turn=0.02)

    @property
    def repo(self) -> Path:
        return self._repo

    def attach_repo(self, repo: Path) -> None:
        self._repo = repo

    @property
    def harness(self) -> Any:
        return self._harness_instance

    def stop(self) -> None:
        if self._loop is not None and self._stopping is not None:
            self._loop.call_soon_threadsafe(self._stopping.set)
        if self._thread is not None:
            self._thread.join(timeout=60)


_ACTIVE: list[_TestServer] = []


@pytest.fixture(autouse=True)
def _stop_servers() -> Any:
    yield
    while _ACTIVE:
        _ACTIVE.pop().stop()


def _config_with(tmp_path: Path, address: str) -> Any:
    """A `DurableConfig` pointed at the test server, with the runs under the test's own root."""
    from codna.runtime.config import RuntimeConfig, build_runtime_paths

    from codna.durable.config import load_durable_config

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
        "CODNA_TEMPORAL_ADDRESS": address,
        # `default`, not the product's `codna`: the time-skipping test server serves `default` only,
        # and a worker in a different namespace than the client would never see the run's tasks —
        # which is exactly the failure this test caught the first time it ran.
        "CODNA_TEMPORAL_NAMESPACE": "default",
        "CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "runs"),
        "CODNA_DURABLE_ENGINE": "plain",
    }
    return load_durable_config(env=env, runtime=runtime, hostname="run-api-host")


def _wait_until(predicate: Any, *, timeout: float = 120.0, interval: float = 0.05) -> Any:
    """Poll ``predicate`` until it returns something truthy. The one wait primitive here."""
    import time

    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition never became true (last: {last!r})")


class _Client:
    """A tiny HTTP client for one running server (no third-party deps)."""

    def __init__(self, server: Any) -> None:
        self._server = server

    def call(self, method: str, path: str, body: Any = None, *, timeout: float = 60.0) -> tuple[int, Any]:
        connection = http.client.HTTPConnection(
            self._server.host, self._server.port, timeout=timeout
        )
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "authorization": f"Bearer {self._server.token}",
            "accept": "application/json",
        }
        if payload is not None:
            headers["content-type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
        finally:
            connection.close()

    def stream(self, path: str, *, timeout: float = 120.0) -> bytes:
        connection = http.client.HTTPConnection(
            self._server.host, self._server.port, timeout=timeout
        )
        try:
            connection.request(
                "GET", path, headers={"authorization": f"Bearer {self._server.token}"}
            )
            return connection.getresponse().read()
        finally:
            connection.close()


def test_the_real_backend_starts_statuses_steers_and_streams_a_run(tmp_path: Path) -> None:
    """The whole surface, over the real Temporal calls — no scripted backend anywhere in this test."""
    from codna.durable.identity import AUTH_MODE_TOKEN, IdentityConfig, IdentityResolver
    from codna.durable.run_api import RunApiServer
    from codna.durable.run_api_backend import TemporalRunBackend
    from codna.durable.run_api_support import RunApiSettings, load_or_create_token

    repo = _git_repo(tmp_path / "repo")

    # The server's address is only known after it starts, but the WORKER needs the config first, so
    # the environment is started with a throwaway config and the real one is built from its address.
    bootstrap = _config_with(tmp_path, "127.0.0.1:7233")
    environment = _TestServer(bootstrap, repo)
    environment.attach_repo(repo)
    environment.start()
    _ACTIVE.append(environment)

    config = _config_with(tmp_path, environment.address)
    # The worker was started against `bootstrap`'s queue; both resolve the same hostname, so the
    # queue matches. Asserted rather than assumed below.
    assert config.task_queue == bootstrap.task_queue

    backend = TemporalRunBackend(config)
    settings = RunApiSettings(
        port=0,
        token_path=config.paths.root / "api.token",
        heartbeat_s=0.2,
        allowed_repo_roots=(tmp_path,),
    )
    # E2: the body's `identity` may only CONFIRM the authenticated principal, so the server is
    # started with the identity this scenario approves as. The alternative — a server that trusts the
    # body — is the vulnerability the role model exists to close.
    server = RunApiServer(
        config=config,
        backend=backend,
        settings=settings,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_TOKEN,
                local_token=load_or_create_token(settings.token_path),
                local_identity="approver@codna.ai",
            )
        ),
    ).start()
    client = _Client(server)
    try:
        # 1) Start it. 202 and the workflow exists on the server.
        status, body = client.call(
            "POST",
            "/runs",
            {
                "run_id": "run-api-real",
                "repo_dir": str(repo),
                "briefs": ["Fix the flaky retry"],
                "base_branch": "main",
                "engine": "plain",
            },
        )
        assert status == 202, body
        assert body["run"]["run_id"] == "run-api-real"
        assert body["run"]["tasks"] == 1

        # The run parks on its approval gate; that is the state a person acts on, so it is the state
        # the surface must be able to report.
        def parked() -> Any:
            status, body = client.call("GET", "/runs/run-api-real")
            assert status in (200, 404), body
            if status != 200:
                return None
            status_body = body.get("status") or {}
            return body if status_body.get("tasks_total") else None

        body = _wait_until(parked)
        assert body["status"]["run_id"] == "run-api-real"
        assert body["status"]["tasks_total"] == 1

        # Wait for the LEDGER to show the child's turn work — proof the real activities ran, not just
        # that a workflow exists.
        def has_events() -> Any:
            status, body = client.call("GET", "/runs/run-api-real")
            if status != 200:
                return None
            return body if body["ledger"]["records"] > 0 else None

        body = _wait_until(has_events)
        assert body["ledger"]["valid"] is True

        # 2) Wait until the task is genuinely PARKED before acting on it.
        #
        #    A steer is accepted at any time, but an approve is refused by the workflow's validator
        #    unless the task is waiting on a decision — so approving too early would test the refusal
        #    path, not the happy one. The wait reads the child's own `status` query (the same query
        #    `codna durable status --task` uses), because the Run API deliberately has no route that
        #    exposes a task's internal state: a client decides from the run's view, not from this.
        _wait_until(lambda: _task_awaiting_approval(backend, "run-api-real", "task-1"))

        # 3) Steer the parked task: the REAL Update, through the CLI's own sender.
        status, body = client.call(
            "POST",
            "/runs/run-api-real/steer",
            {"identity": "approver@codna.ai", "task_id": "task-1", "note": "also cover the retry"},
        )
        assert status == 200, body
        assert body["identity"] == "approver@codna.ai"
        assert body["result"]["applied"] is True
        assert body["result"]["identity"] == "approver@codna.ai"

        # 4) Approve it: the identity lands in the task's own approval list, which is the audit
        #    record — read back from the workflow itself, not from the HTTP response.
        status, body = client.call(
            "POST",
            "/runs/run-api-real/approve",
            {"identity": "approver@codna.ai", "task_id": "task-1"},
        )
        assert status == 200, body
        assert body["applied"] is True
        approvals = _task_approvals(backend, "run-api-real", "task-1")
        # Both identities are in the workflow's own record, in the order they acted.
        assert [item["kind"] for item in approvals] == ["steer", "approve"], approvals
        # ONE identity for both acts: the approver's. E2 removed the old second name — the body's
        # `identity` can no longer name somebody else, so a steer by the same person records the same
        # actor, which is what an audit trail should say.
        assert [item["identity"] for item in approvals] == [
            "approver@codna.ai",
            "approver@codna.ai",
        ], approvals

        # A second approval is never accepted. If it reaches the live workflow, the validator returns
        # 409; if the approved task has already completed, Temporal returns the same 404 boundary the
        # real CLI test accepts. Which one occurs is a timing race, not a choice the client controls.
        status, body = client.call(
            "POST",
            "/runs/run-api-real/approve",
            {"identity": "approver@codna.ai", "task_id": "task-1"},
        )
        assert status in (404, 409), body
        if status == 409:
            assert body["error"]["code"] in ("durable_not_parked", "durable_update_refused")
        else:
            assert body["error"]["code"] == "run_not_found", body
        assert body["error"]["details"]["action"] == "approve", body

        # 5) The stream: real ledger records, framed for the IDE, ending at the run's terminal.
        raw = client.stream("/runs/run-api-real/events?follow=1", timeout=180.0)
        text = raw.decode("utf-8")
        frames = [line for line in text.split("\n") if line.startswith("data: ")]
        assert frames, text[:600]
        decoded = [json.loads(frame[len("data: ") :]) for frame in frames]
        assert all(isinstance(item.get("event_type"), str) for item in decoded)
        event_types = [item["event_type"] for item in decoded]
        # The real activities notified; the run's own terminal closed the stream.
        assert "run.started" in event_types, event_types
        assert event_types[-1] == "run.finished", event_types
        # And the whole ledger is intact: the frames are the records, contiguous from 1.
        assert [item["event_index"] for item in decoded] == list(range(1, len(decoded) + 1))
    finally:
        server.stop()


def _task_status(backend: Any, run_id: str, task_id: str) -> dict[str, Any]:
    """The child task's status, through the same query `codna durable status --task` reads.

    The surface has no task-status route on purpose (a client acts on the RUN), so test
    introspection goes straight to the workflow. Both engines declare `status`; asking by name is
    what the CLI does for the same reason.
    """

    async def query() -> dict[str, Any]:
        handle = backend._client.get_workflow_handle(f"task-{run_id}-{task_id}")
        return await handle.query("status")

    try:
        return backend._call(query(), timeout_s=30.0)
    except Exception:  # noqa: BLE001 - the child may not exist yet, which is "not parked"
        return {}


def _task_awaiting_approval(backend: Any, run_id: str, task_id: str) -> bool:
    return bool(_task_status(backend, run_id, task_id).get("awaiting_approval"))


def _task_approvals(backend: Any, run_id: str, task_id: str) -> list[dict[str, Any]]:
    async def query() -> list[dict[str, Any]]:
        handle = backend._client.get_workflow_handle(f"task-{run_id}-{task_id}")
        return await handle.query("approvals")

    return list(backend._call(query(), timeout_s=30.0))


def test_the_real_backend_reports_an_unknown_run_as_404(tmp_path: Path) -> None:
    """A run id Temporal has never seen is the surface's 404, not a 502 from an RPC error."""
    from codna.durable.identity import AUTH_MODE_TOKEN, IdentityConfig, IdentityResolver
    from codna.durable.run_api import RunApiServer
    from codna.durable.run_api_backend import TemporalRunBackend
    from codna.durable.run_api_support import RunApiSettings, load_or_create_token

    repo = _git_repo(tmp_path / "repo")
    environment = _TestServer(_config_with(tmp_path, "127.0.0.1:7233"), repo)
    environment.attach_repo(repo)
    environment.start()
    _ACTIVE.append(environment)

    config = _config_with(tmp_path, environment.address)
    backend = TemporalRunBackend(config)
    settings = RunApiSettings(port=0, token_path=config.paths.root / "api.token")
    server = RunApiServer(
        config=config,
        backend=backend,
        settings=settings,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_TOKEN,
                local_token=load_or_create_token(settings.token_path),
                local_identity="a@b.c",
            )
        ),
    ).start()
    try:
        status, body = _Client(server).call("GET", "/runs/never-started")
        assert status == 404, body
        assert body["error"]["code"] == "run_not_found"

        status, body = _Client(server).call(
            "POST", "/runs/never-started/pause", {"identity": "a@b.c"}
        )
        assert status == 404, body
        assert body["error"]["code"] == "run_not_found"
    finally:
        server.stop()
