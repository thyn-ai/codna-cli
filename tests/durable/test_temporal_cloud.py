"""Temporal Cloud: does the connect path actually reach a real tenant and run a workflow?

Everything else in this suite proves behaviour against ``WorkflowEnvironment.start_time_skipping()``
or the local dev server. Neither can answer the questions a Cloud connection raises: does the
``tls=True, api_key=...`` path carry a real credential to a real endpoint, does the namespace name a
tenant the key is authorised for, and does a workflow started there actually execute? Those are
properties of the SERVER, so only the server can answer them.

GATED, and gated on THREE variables together:

    TEMPORAL_ENDPOINT=codna.<account>.tmprl.cloud:7233
    TEMPORAL_NAMESPACE=<namespace>
    TEMPORAL_API_KEY=<key — read from the environment, never written anywhere>

`pytest.mark.skipif` is what makes this file safe in the base suite and in CI: with none of the
variables set it collects one skipped test and touches no network. **The key value is never printed,
never asserted on, never included in a failure message** — every assertion below is about the
connection and the workflow, and the one place the key could leak (an exception's text) is scrubbed
by `_redact` before it reaches an assertion message.

NO WORKER DEPLOYMENT, NO SIDECAR, NO codna ACTIVITIES. The workflow is a two-line function whose only
job is to prove the round trip (start → execute → complete → history readable). Registering the real
durable worker here would drag in the agent-core sidecar and a model provider, which are not what
this test is about and are exactly the dependencies that make a Cloud test flaky.
"""
from __future__ import annotations

import importlib.util
import os
import threading
import uuid
from typing import Any

import pytest

HAS_EXTRA = importlib.util.find_spec("temporalio") is not None

# The three coordinates, in the order they are reported when one is missing.
ENDPOINT_ENV = "TEMPORAL_ENDPOINT"
NAMESPACE_ENV = "TEMPORAL_NAMESPACE"
API_KEY_ENV = "TEMPORAL_API_KEY"

_ENDPOINT = (os.environ.get(ENDPOINT_ENV) or "").strip()
_NAMESPACE = (os.environ.get(NAMESPACE_ENV) or "").strip()
_API_KEY = (os.environ.get(API_KEY_ENV) or "").strip()

_MISSING = [
    name
    for name, value in (
        (ENDPOINT_ENV, _ENDPOINT),
        (NAMESPACE_ENV, _NAMESPACE),
        (API_KEY_ENV, _API_KEY),
    )
    if not value
]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING),
        reason=(
            "Temporal Cloud is not configured (missing "
            f"{', '.join(_MISSING or ['nothing'])}); set {ENDPOINT_ENV}, {NAMESPACE_ENV} and "
            f"{API_KEY_ENV} to run the cloud connection test"
        ),
    ),
    pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed"),
]

# A Cloud namespace is a SHARED tenant: the workflow id and task queue are unique per run so two
# concurrent test runs (or two developers) cannot collide, and both are short so the id stays inside
# Temporal's limit.
_RUN_TOKEN = uuid.uuid4().hex[:8]
TASK_QUEUE = f"codna-cloud-test-{_RUN_TOKEN}"
WORKFLOW_ID = f"codna-cloud-test-{_RUN_TOKEN}"
# The whole test must stay "under a few seconds" per the D-2 acceptance note. Everything below is
# bounded by this, so a hung server fails the test instead of hanging the suite.
TIMEOUT_S = 30.0


def _redact(text: str) -> str:
    """Scrub the API key out of text that may reach an assertion message or the terminal.

    A failed connection's exception text can quote request metadata on some transports; a test that
    prints it verbatim would put the credential in CI logs. This is the only sanitisation point, so
    every message that includes server text goes through it.
    """
    if _API_KEY and _API_KEY in text:
        return text.replace(_API_KEY, "<redacted>")
    return text


def _run_in_thread(coro_factory: Any) -> Any:
    """Run one coroutine on a dedicated loop and return its result or re-raise its failure.

    A dedicated thread (rather than ``asyncio.run``) because ``temporalio``'s worker wants a loop that
    outlives the ``start_workflow`` call: the workflow is started INSIDE the worker's ``async with``
    so the worker is definitely polling before the start, and that whole block must be one coroutine
    that owns its loop. A failure is carried back on the result slot instead of being swallowed.
    """
    import asyncio

    outcome: dict[str, Any] = {}

    def target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            outcome["value"] = loop.run_until_complete(asyncio.wait_for(coro_factory(), TIMEOUT_S))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread
            outcome["error"] = exc
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=target, name="codna-cloud-test", daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT_S + 5.0)
    assert not thread.is_alive(), f"the cloud test did not finish within {TIMEOUT_S:g}s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


try:  # module-level workflow definitions; `@workflow.run` refuses a locally-defined class
    from temporalio import workflow as _workflow

    @_workflow.defn(name="CodnaCloudProbeWorkflow")
    class CloudProbeWorkflow:
        """The trivial workflow: mutate its argument so a replay cannot fake the answer."""

        @_workflow.run
        async def run(self, value: int) -> int:
            return value + 1

except Exception:  # pragma: no cover - the extra is absent; pytestmark skips the test anyway
    CloudProbeWorkflow = None  # type: ignore[assignment,misc]


# -- configuration --------------------------------------------------------------------------------


def _config() -> Any:
    """The durable config this environment resolves, with the Cloud profile forced explicitly.

    Built through the REAL loader (not a hand-made object) so the test exercises the same validation
    a `codna worker --temporal-profile cloud` does — including the refusal of a partial setup. The
    runs root is redirected into a temp dir by `CODNA_DURABLE_RUNS_ROOT` so nothing is written to the
    developer's `~/.codna`.
    """
    from codna.durable.config import load_durable_config

    return load_durable_config(env=dict(os.environ), profile="cloud")


# -- the connection -------------------------------------------------------------------------------


def test_the_cloud_profile_resolves_tls_and_a_key_without_storing_it():
    """The config must NAME its credential, never carry it.

    This runs before any network call: if the config were to hold the key as a field, its ``repr``
    would print it — and every failure message that formats a config would leak it. Asserted here
    rather than trusted, because "we were careful" is not a property a future edit preserves.
    """
    config = _config()
    assert config.profile == "cloud"
    assert config.requires_tls is True
    assert config.requires_api_key is True
    assert config.owns_local_server is False
    assert config.api_key_env == API_KEY_ENV
    assert _API_KEY not in repr(config)

    # `connect_options()` DOES carry the value — that is its job — so the guarantee that matters is
    # about FORMATTING: repr/str of the options must never contain the key, because those are what a
    # failing assertion, a debugger and a log line use. (Measured: a plain dict here printed a live
    # key into a terminal, which is why the type is `ConnectOptions`.)
    options = config.connect_options(env={API_KEY_ENV: _API_KEY})
    assert options["tls"] is True
    assert options["api_key"] == _API_KEY
    for formatted in (repr(options), str(options), f"{options}", "%s" % (options,)):
        assert _API_KEY not in formatted, "the credential must never be formatted into text"
    assert "<redacted>" in repr(options)

    # ...and the VALUE is still reachable from the environment when a connection needs it.
    assert config.temporal_api_key(env={API_KEY_ENV: _API_KEY}) == _API_KEY


def test_a_partial_cloud_configuration_is_refused_naming_the_missing_variable():
    """No silent fallback: a namespace-less Cloud setup must not become a loopback dev server."""
    from codna.durable.config import DurableConfigError, load_durable_config

    with pytest.raises(DurableConfigError) as caught:
        load_durable_config(env={ENDPOINT_ENV: _ENDPOINT, API_KEY_ENV: _API_KEY}, profile="cloud")
    assert caught.value.code == "temporal_cloud_config_incomplete"
    assert NAMESPACE_ENV in caught.value.details["missing"]
    # The detail map is printed by the CLI; it must name variables, never values.
    assert _API_KEY not in str(caught.value.details)


def test_the_namespace_from_the_environment_is_the_one_used():
    """The namespace is not inferred here — but it must be the one the connection is made with."""
    assert _config().namespace == _NAMESPACE


# -- a real workflow round trip -------------------------------------------------------------------


def test_a_workflow_started_on_temporal_cloud_completes():
    """Connect over TLS with the API key, start a workflow, and wait for a real result.

    The worker is created INSIDE the same coroutine as the start, so the task queue has a poller
    before the workflow exists — otherwise the test would race the poller and fail for a reason that
    has nothing to do with the credential. Everything is torn down in the same block, so a failure
    leaves no workflow running on the shared namespace.
    """
    assert CloudProbeWorkflow is not None  # pytestmark guarantees the extra is present

    async def scenario() -> dict[str, Any]:
        from temporalio.client import Client
        from temporalio.worker import Worker

        config = _config()
        client = await Client.connect(
            config.address, namespace=config.namespace, **config.connect_options()
        )
        handle: Any = None
        try:
            # A dedicated worker for this ONE probe workflow: the codna activities need a sidecar and
            # a provider, neither of which this test is about.
            async with Worker(client, task_queue=TASK_QUEUE, workflows=[CloudProbeWorkflow]):
                handle = await client.start_workflow(
                    CloudProbeWorkflow.run,
                    41,
                    id=WORKFLOW_ID,
                    task_queue=TASK_QUEUE,
                )
                result = await handle.result()
                description = await handle.describe()
            return {
                "result": result,
                "run_id": description.run_id,
                "status": description.status.name if description.status is not None else None,
                "history_length": description.history_length,
                "namespace": description.namespace,
                "task_queue": description.task_queue,
            }
        finally:
            # TERMINATE, not cancel, when the workflow is somehow still open: a cancel needs a live
            # worker and this namespace is shared, so an orphan is not an acceptable test artifact.
            try:
                target = handle if handle is not None else client.get_workflow_handle(WORKFLOW_ID)
                described = await target.describe()
                if described.status is not None and described.status.name == "RUNNING":
                    await target.terminate(reason="codna temporal-cloud test cleanup")
            except Exception:  # noqa: BLE001 - absent/already-closed is the goal state
                pass
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    try:
        facts = _run_in_thread(scenario)
    except Exception as exc:  # noqa: BLE001 - re-raised with the key scrubbed
        pytest.fail(_redact(f"the Temporal Cloud round trip failed: {type(exc).__name__}: {exc}"))

    assert facts["result"] == 42, facts
    assert facts["status"] == "COMPLETED", facts
    assert facts["namespace"] == _NAMESPACE, facts
    assert facts["task_queue"] == TASK_QUEUE, facts
    # A completed workflow has a real history. Asserted as "> 0" rather than an exact event count:
    # the server's event stream is its own business, and pinning a number here would fail on an
    # upgrade that changed nothing about durability.
    assert facts["history_length"] > 0, facts
