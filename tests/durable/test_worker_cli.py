"""The durable worker: settings, the injected runtime, registration and clean shutdown.

Two halves, matching the two halves of `worker.py`:

* **Settings and the runtime** run in the BASE suite (no `temporalio`), because those are the parts
  that decide what a worker connects to and what an activity spends money against. A bug there is a
  wrong-queue or wrong-token bug, and it must be catchable without the extra installed.
* **Registration and shutdown** are gated with `pytest.importorskip` and run against
  ``WorkflowEnvironment.start_time_skipping()``: the thing worth proving is that the REAL
  `Worker(...)` accepts the workflow/activity/plugin sets `run_worker` assembles. A mock would only
  prove the mock accepts them — and a rejected registration is exactly the startup failure this
  module exists to surface.

Nothing here starts a real Temporal CLI, a real sidecar or a real activity: the runtime and the
client are injected, and the stop event ends the poll loop deterministically.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from codna.durable.config import load_durable_config
from codna.durable.worker import (
    DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    WorkerError,
    WorkerSettings,
    build_activity_runtime,
    github_token_provider,
    plugins_for,
    worker_settings_from_config,
)
from codna.runtime.config import RuntimeConfig, build_runtime_paths

HAS_EXTRA = (
    __import__("importlib").util.find_spec("temporalio") is not None
    and __import__("importlib").util.find_spec("langgraph") is not None
)


@pytest.fixture
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    """The shared runtime config, injected so the sidecar port is not a developer's real one."""
    paths = build_runtime_paths()
    return RuntimeConfig(
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


@pytest.fixture
def config(runtime_config: RuntimeConfig, tmp_path: Path):
    env = {"CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "runs")}
    return load_durable_config(env=env, runtime=runtime_config, hostname="build-host")


@pytest.fixture
def settings(config: Any) -> WorkerSettings:
    return worker_settings_from_config(config, ensure_sidecar=False)


# -- settings ------------------------------------------------------------------------------


def test_settings_carry_the_configs_coordinates(config: Any, settings: WorkerSettings):
    """One source of truth: the worker polls exactly the queue the CLI starts runs on."""
    assert settings.address == config.address
    assert settings.namespace == config.namespace
    assert settings.task_queue == config.task_queue == "codna.agent.build-host"
    assert settings.engine == config.graph_engine
    assert settings.build_id == config.build_id
    assert settings.runs_root == config.runs_root
    assert settings.sidecar_url == config.sidecar_url


def test_the_default_concurrency_cap_is_the_plans_two(settings: WorkerSettings):
    """Each slot can hold an open sidecar stream; unbounded is how a worker exhausts a laptop."""
    assert settings.max_concurrent_activities == DEFAULT_MAX_CONCURRENT_ACTIVITIES == 2


def test_a_cap_of_zero_is_refused_rather_than_clamped(config: Any):
    """A typo'd cap that polls and never executes looks like a server problem for hours."""
    with pytest.raises(WorkerError) as caught:
        worker_settings_from_config(config, max_concurrent_activities=0)
    assert caught.value.code == "worker_settings_invalid"
    assert caught.value.details["field"] == "max_concurrent_activities"


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_a_non_integer_cap_is_refused(config: Any, bad: Any):
    with pytest.raises(WorkerError) as caught:
        worker_settings_from_config(config, max_concurrent_activities=bad)
    assert caught.value.code == "worker_settings_invalid"


def test_a_negative_grace_period_is_refused(config: Any):
    with pytest.raises(WorkerError) as caught:
        worker_settings_from_config(config, grace_period_s=-1.0)
    assert caught.value.details["field"] == "grace_period_s"


def test_the_sidecar_token_is_read_from_the_environment_and_never_defaulted(config: Any):
    assert worker_settings_from_config(config, env={}).sidecar_token is None
    with_token = worker_settings_from_config(config, env={"AGENT_CORE_AUTH_TOKEN": "s3cret"})
    assert with_token.sidecar_token == "s3cret"


def test_the_banner_never_contains_the_sidecar_token(settings: WorkerSettings):
    """The banner is printed and may be pasted into a bug report; a token there is a leak."""
    import dataclasses

    with_token = dataclasses.replace(settings, sidecar_token="s3cret")
    banner = with_token.to_payload()
    assert "s3cret" not in json.dumps(banner)
    assert banner["sidecar_auth"] is True
    assert settings.to_payload()["sidecar_auth"] is False


# -- the temporal profile ------------------------------------------------------------------


def test_a_local_config_yields_a_local_worker_with_no_credential(config: Any):
    """The historical default: no TLS, no key, and the banner says so."""
    local = worker_settings_from_config(config, ensure_sidecar=False)
    assert local.profile == "local"
    assert local.is_cloud is False
    assert local.api_key_env is None
    banner = local.to_payload()
    assert banner["temporal_profile"] == "local"
    assert banner["tls"] is False
    assert banner["api_key_env"] is None


def test_a_cloud_config_carries_the_profile_and_the_key_variable_name_only(runtime_config: Any):
    """A worker polling Cloud must say so, and must never put the key in its own banner."""
    from codna.durable.config import load_durable_config

    cloud = load_durable_config(
        env={
            "CODNA_DURABLE_RUNS_ROOT": str(runtime_config.codna_home / "runs"),
            "TEMPORAL_ENDPOINT": "codna.example.tmprl.cloud:7233",
            "TEMPORAL_NAMESPACE": "codna.example",
            "TEMPORAL_API_KEY": "super-secret-value",
        },
        runtime=runtime_config,
        hostname="build-host",
    )
    settings = worker_settings_from_config(cloud, ensure_sidecar=False)
    assert settings.is_cloud is True
    assert settings.profile == "cloud"
    assert settings.api_key_env == "TEMPORAL_API_KEY"
    banner = settings.to_payload()
    assert banner["temporal_profile"] == "cloud"
    assert banner["tls"] is True
    # THE assertion: the variable is named, the value is absent.
    assert banner["api_key_env"] == "TEMPORAL_API_KEY"
    assert "super-secret-value" not in json.dumps(banner)
    assert "super-secret-value" not in repr(settings)


# -- the injected runtime ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_runtime_uses_the_configured_sidecar_and_runs_root(settings: WorkerSettings):
    seen: list[Any] = []
    runtime = build_activity_runtime(
        settings, sidecar_factory=lambda _owner: seen.append("client") or object()
    )
    assert runtime.runs_root == settings.runs_root
    runtime.sidecar_factory("codna-run-run-1")
    assert seen == ["client"]


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_runtime_ledger_lands_in_the_configured_runs_root(settings: WorkerSettings):
    runtime = build_activity_runtime(settings, ledger_clock=lambda: 1_700_000_000.0)
    ledger = runtime.ledger("run-1")
    assert ledger.path == Path(settings.runs_root) / "run-1" / "ledger.ndjson"
    assert ledger.run_id == "run-1"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_runtime_refuses_a_path_traversing_run_id(settings: WorkerSettings):
    """The ledger is the audit trail; a run id that escapes the runs root is not a run id."""
    runtime = build_activity_runtime(settings)
    with pytest.raises(Exception) as caught:
        runtime.ledger("../escape")
    assert getattr(caught.value, "code", "") == "invalid_run_id"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_default_sidecar_client_carries_the_bearer(settings: WorkerSettings):
    """agent-core fails closed when a token is configured, so the worker must send it."""
    import dataclasses

    with_token = dataclasses.replace(settings, sidecar_token="s3cret", sidecar_url="http://127.0.0.1:1")
    runtime = build_activity_runtime(with_token)
    client = runtime.sidecar_factory("codna-run-run-1")
    try:
        assert client.base_url == "http://127.0.0.1:1"
        assert client._headers()["authorization"] == "Bearer s3cret"
        assert "s3cret" not in repr(client)
    finally:
        client.close()


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_default_sidecar_client_carries_the_runs_owner(settings: WorkerSettings):
    """REGRESSION: a client without the run's owner is refused on its OWN session's turn.

    The engine stamps a session with its owner at CREATE time and returns
    ``403 session_owner_mismatch`` for any later request whose ``x-session-owner`` disagrees. A
    factory that ignored its argument would build an owner-less client, send no header, be read as
    ``anonymous``, and fail every turn — with the fake sidecar still green, because it does not
    enforce ownership. That is exactly how this shipped broken; the assertion is on the HEADER the
    real engine reads, not on an internal field.
    """
    runtime = build_activity_runtime(settings)
    client = runtime.sidecar_factory("codna-run-run-1")
    try:
        assert client._headers()["x-session-owner"] == "codna-run-run-1"
    finally:
        client.close()


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_runtime_defaults_its_clock_to_wall_time(settings: WorkerSettings):
    """The ledger's `ts` outlives the process, so it is wall time — not `monotonic`."""
    import time as _time

    runtime = build_activity_runtime(settings, ledger_clock=None)
    before = _time.time()
    assert before - 5 <= runtime.clock() <= _time.time() + 5


# -- the GitHub token ----------------------------------------------------------------------


def test_the_github_token_prefers_the_codna_variable():
    """A personal `GITHUB_TOKEN` for other tools must not win over an explicit CODNA one."""
    provider = github_token_provider({"CODNA_GITHUB_TOKEN": "codna-token", "GITHUB_TOKEN": "other"})
    assert provider() == "codna-token"


def test_the_github_token_falls_back_to_the_plain_variable():
    assert github_token_provider({"GITHUB_TOKEN": "gh"})() == "gh"


def test_a_blank_github_token_is_not_a_token():
    """An empty variable is a misconfiguration; returning "" would send `Bearer ` to GitHub."""
    assert github_token_provider({"CODNA_GITHUB_TOKEN": "   ", "GITHUB_TOKEN": ""})() is None


def test_a_missing_github_token_is_reported_later_not_at_startup():
    """Planning needs no token; only `open_pr` does, and it already fails with its own code."""
    provider = github_token_provider({})
    assert provider() is None


# -- the engine plugin ---------------------------------------------------------------------


def test_the_plain_engine_registers_no_plugin():
    """Registering the graph plugin for the plain engine is a duplicate-activity error at startup."""
    assert plugins_for("plain") == []


def test_an_unknown_engine_is_refused(settings: WorkerSettings):
    with pytest.raises(WorkerError) as caught:
        plugins_for("langraph")  # the typo this exists for
    assert caught.value.code == "worker_engine_invalid"
    assert caught.value.details["allowed"] == ["langgraph", "plain"]


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_graph_engine_yields_a_fresh_plugin_each_time():
    """`LangGraphPlugin` consumes the graph's node metadata: a shared one raises on the second use."""
    first = plugins_for("langgraph")
    second = plugins_for("langgraph")
    assert len(first) == len(second) == 1
    assert first[0] is not second[0]


# -- the import contract -------------------------------------------------------------------


def test_the_worker_module_imports_without_the_extra(tmp_path: Path):
    """`cli.py` imports this module to parse `codna durable worker`; the base CLI must stay working."""
    import subprocess

    script = (
        "import sys\n"
        "import codna.durable.worker\n"
        "assert 'temporalio' not in sys.modules, sorted(sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


# -- registration and shutdown, against a real server ---------------------------------------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _stop_after_start() -> asyncio.Event:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.2, stop.set)
    return stop


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_worker_registers_and_stops_cleanly_on_a_real_server(tmp_path: Path, settings: WorkerSettings):
    """The whole startup contract: connect, register both engines, run, shut down on the event."""
    from temporalio.testing import WorkflowEnvironment

    import codna.durable.worker as worker_module
    from codna.durable.activities import ACTIVITIES, configure_default_runtime
    from codna.durable.workflows import registered_workflows

    started: list[dict[str, object]] = []
    queues: list[str] = []

    async def scenario() -> None:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            runtime = build_activity_runtime(settings, sidecar_factory=lambda _owner: object())
            local = _replace_queue(settings)
            queues.append(local.task_queue)
            await worker_module.run_worker(
                local,
                stop=await _stop_after_start(),
                on_started=started.append,
                install_signal_handlers=False,
                runtime=runtime,
                client=env.client,
            )
        finally:
            await env.shutdown()

    _run(scenario())
    assert started, "the worker never reported it had started"
    banner = started[0]
    # The test server has no queue of ours, so the scenario polls a per-test one; the banner must
    # report THAT queue, which is what proves the banner describes the worker that actually ran.
    assert banner["task_queue"] == queues[0]
    assert banner["task_queue"].endswith(".worker-test")
    assert banner["workflows"] == [cls.__name__ for cls in registered_workflows()]
    assert banner["activities"] == [activity.__name__ for activity in ACTIVITIES]
    _ = configure_default_runtime  # the runtime is installed as the process default by run_worker


def _replace_queue(settings: WorkerSettings) -> WorkerSettings:
    """The test server listens on no task queue of ours; poll a queue unique to this test."""
    import dataclasses

    return dataclasses.replace(settings, task_queue=f"{settings.task_queue}.worker-test")


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_an_unreachable_server_is_a_structured_startup_error(settings: WorkerSettings):
    """A wrong address must name the address and the fix, not raise a grpc traceback."""
    import dataclasses

    dead = dataclasses.replace(settings, address="127.0.0.1:1", task_queue="codna.agent.never")
    with pytest.raises(WorkerError) as caught:
        _run(_run_worker_without_client(dead))
    assert caught.value.code == "worker_server_unreachable"
    assert caught.value.details["address"] == "127.0.0.1:1"
    assert "dev-server start" in caught.value.hint


async def _run_worker_without_client(settings: WorkerSettings) -> None:
    from codna.durable.worker import run_worker

    stop = asyncio.Event()
    await run_worker(settings, stop=stop, install_signal_handlers=False)


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_worker_whose_sidecar_cannot_start_fails_before_it_polls(settings: WorkerSettings, monkeypatch: pytest.MonkeyPatch):
    """Activity work must never be delivered to a worker with no engine to run it."""
    import codna.agent_core_runtime as runtime_module

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("sidecar exploded")

    monkeypatch.setattr(runtime_module, "ensure_agent_core_running", explode)
    with pytest.raises(WorkerError) as caught:
        _run(_run_worker_ensuring_sidecar(settings))
    assert caught.value.code == "worker_sidecar_unavailable"
    assert "codna doctor" in caught.value.hint


async def _run_worker_ensuring_sidecar(settings: WorkerSettings) -> None:
    import dataclasses

    from codna.durable.worker import run_worker

    with_sidecar = dataclasses.replace(settings, ensure_sidecar=True)
    await run_worker(with_sidecar, stop=asyncio.Event(), install_signal_handlers=False)


# -- the deployment version (the registration identity) ------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_build_is_registered_as_a_pinned_deployment_version(settings: WorkerSettings):
    """A bare `build_id=` registers NOTHING to protect; a deployment version is what pins runs.

    Measured: a worker constructed with ``build_id=`` alone leaves ``describe_worker_deployment``
    reporting "no Worker Deployment found", so no run is ever pinned to a build and the plan's
    "old runs finish on the old build" mitigation is dead code.
    """
    from temporalio.common import VersioningBehavior

    from codna.durable.worker import DEPLOYMENT_NAME, deployment_version, deployment_version_config

    version = deployment_version(settings)
    assert version.deployment_name == DEPLOYMENT_NAME
    assert version.build_id == settings.build_id

    config = deployment_version_config(settings)
    assert config.version.build_id == settings.build_id
    assert config.use_worker_versioning is True
    assert config.default_versioning_behavior is VersioningBehavior.PINNED


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_graph_version_bump_is_a_new_deployment_version(settings: WorkerSettings, config: Any):
    """The pinning guarantee: a new graph ⇒ a new build id ⇒ a new version, never a silent upgrade."""
    import dataclasses

    from codna.durable.version import build_id_for
    from codna.durable.worker import deployment_version

    bumped = dataclasses.replace(settings, build_id=build_id_for("0.2.94", 2))
    assert deployment_version(settings).build_id != deployment_version(bumped).build_id
    assert deployment_version(bumped).deployment_name == deployment_version(settings).deployment_name
    _ = config


def test_the_banner_names_the_deployment_version_and_promotion_choice(settings: WorkerSettings):
    """An operator must be able to read which build a worker claimed, from its own output."""
    banner = settings.to_payload()
    assert banner["deployment"] == "codna"
    assert banner["version"] == f"codna.{settings.build_id}"
    assert banner["promote_current"] is True


def test_promotion_can_be_turned_off(config: Any):
    """"Do not move cluster traffic" is a real host policy, not a temporary flag."""
    off = worker_settings_from_config(config, promote_current=False)
    assert off.promote_current is False
    assert off.to_payload()["promote_current"] is False


def test_an_explicit_engine_overrides_the_configured_one(config: Any):
    """One host may run the plain fallback without editing the config every other command reads."""
    assert worker_settings_from_config(config, engine="plain").engine == "plain"
    assert worker_settings_from_config(config).engine == config.graph_engine


# -- graceful drain ------------------------------------------------------------------------


class _DrainingWorker:
    """A worker double that records the ORDER of shutdown vs. awaiting the run loop."""

    def __init__(self, *, finish_after: float = 0.0) -> None:
        self.task_queue = "codna.agent.test"
        self.events: list[str] = []
        self.shutdown_called = 0
        self._finish_after = finish_after

    async def shutdown(self) -> None:
        self.shutdown_called += 1
        self.events.append("shutdown:start")

    async def run(self) -> None:
        self.events.append("run:start")
        import asyncio

        await asyncio.sleep(self._finish_after)
        self.events.append("run:end")


def _draining_run_task(worker: _DrainingWorker):
    import asyncio

    return asyncio.ensure_future(worker.run())


def test_the_drain_asks_the_worker_to_stop_and_then_awaits_it():
    """The order is the guarantee: ``shutdown`` first (stop accepting), then wait for in-flight work.

    A worker that awaited ``run()`` and only THEN shut down would accept new tasks while draining; one
    that shut down and did not await would abandon an in-flight activity, which Temporal then
    re-delivers — and a paid turn would be billed twice.
    """
    from codna.durable.worker import _drain

    worker = _DrainingWorker(finish_after=0.01)

    async def scenario() -> None:
        task = _draining_run_task(worker)
        await asyncio.sleep(0)  # let `run` begin
        await _drain(worker, task, grace_period_s=5.0)
        assert task.done()

    asyncio.run(scenario())
    assert worker.shutdown_called == 1
    assert worker.events[:2] == ["run:start", "shutdown:start"]
    assert worker.events[-1] == "run:end", "the drain returned before the run loop finished"


def test_the_drain_surfaces_a_failed_run_instead_of_reporting_a_clean_stop():
    """A fatal poller error must not be swallowed into "worker stopped" — the exit code depends on it."""
    from codna.durable.worker import _drain

    class _Failing(_DrainingWorker):
        async def run(self) -> None:
            self.events.append("run:start")
            raise RuntimeError("poller died")

    worker = _Failing()

    async def scenario() -> None:
        task = _draining_run_task(worker)
        await asyncio.sleep(0)
        await _drain(worker, task, grace_period_s=1.0)

    with pytest.raises(RuntimeError, match="poller died"):
        asyncio.run(scenario())
    assert worker.shutdown_called == 1, "the worker must be stopped before the error propagates"


def test_the_drain_on_an_already_finished_run_still_stops_the_worker():
    """A run loop that ended on its own must STILL be shut down: activities live on another pool.

    A `run()` that returned or raised says nothing about the activities — they execute on the worker's
    thread pool, and one may still be mid-turn. Skipping `shutdown` because the run loop is done would
    abandon a paid turn, which Temporal then re-delivers.
    """
    from codna.durable.worker import _drain

    worker = _DrainingWorker()

    async def scenario() -> None:
        task = asyncio.ensure_future(worker.run())
        await task
        await _drain(worker, task, grace_period_s=1.0)

    asyncio.run(scenario())
    assert worker.shutdown_called == 1

@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_worker_registers_a_non_zero_graceful_shutdown_timeout(settings: WorkerSettings):
    """Temporal's default graceful timeout is ZERO, which cancels in-flight activities immediately.

    That default would abandon a paid turn mid-stream the moment the worker stops, and Temporal would
    then re-deliver it — the double-billing the heartbeat/turn_key design exists to avoid. This drives
    the REAL ``run_worker`` registration (with an injected client) and reads back the timeout the
    constructed Worker actually has, so a dropped kwarg cannot pass.
    """
    import asyncio
    import dataclasses

    from temporalio.testing import WorkflowEnvironment

    import codna.durable.worker as worker_module

    captured: list[object] = []
    real_worker = worker_module.Worker if hasattr(worker_module, "Worker") else None
    assert real_worker is None, "worker.py must not import temporalio at module scope"

    async def scenario() -> None:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            local = dataclasses.replace(
                settings,
                task_queue=f"{settings.task_queue}.grace-test",
                grace_period_s=17.0,
                promote_current=False,
            )
            runtime = build_activity_runtime(local, sidecar_factory=lambda _owner: object())
            original_register = worker_module._register

            def spy(worker_cls: Any, client: Any, *, settings: Any, **kwargs: Any) -> Any:
                captured.append(kwargs.get("graceful_shutdown_timeout"))
                return original_register(worker_cls, client, settings=settings, **kwargs)

            worker_module._register = spy
            try:
                stop = asyncio.Event()
                loop = asyncio.get_running_loop()
                loop.call_later(0.3, stop.set)
                await worker_module.run_worker(
                    local,
                    stop=stop,
                    install_signal_handlers=False,
                    runtime=runtime,
                    client=env.client,
                )
            finally:
                worker_module._register = original_register
        finally:
            await env.shutdown()

    asyncio.run(scenario())
    assert captured, "the registration path never ran"
    configured = captured[0]
    assert configured is not None, "no graceful_shutdown_timeout was passed to Worker(...)"
    assert configured.total_seconds() == 17.0, configured
