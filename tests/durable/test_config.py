"""`DurableConfig` — the coordinates of a durable run, and every way they can be wrong.

The config is boring until it is wrong: a typo'd engine silently selecting the other workflow
implementation, a task queue that does not match the host, or a run id that escapes the runs root
are each a run that never starts or writes somewhere it should not. So the validation branches are
tested individually, and the defaults are pinned to the plan's exact values.

`codna.runtime.config` is injectable: `load_durable_config(runtime=...)` lets this file assert the
sidecar hand-off without computing real fingerprints (and without a developer's checkout affecting
the result). The one test that does NOT inject it asserts the real integration.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from codna.durable.config import (
    DEFAULT_NAMESPACE,
    DEFAULT_TEMPORAL_ADDRESS,
    DurableConfigError,
    DurablePaths,
    GRAPH_VERSION,
    build_id_for,
    load_durable_config,
    parse_profile,
    sanitize_hostname,
    split_temporal_endpoint,
    task_queue_for_host,
    validate_run_id,
)
from codna import __version__
from codna.runtime.config import RuntimeConfig, build_runtime_paths


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeConfig:
    """A stand-in for the shared runtime config: only the sidecar coordinates are read."""
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
        codna_cli_version="0.2.94",
        engine_build_id="engine",
        sidecar_build_id="sidecar",
        runtime_config_hash="hash",
        allow_loopback_override=False,
    )


def _load(runtime: RuntimeConfig, env: dict[str, str] | None = None, hostname: str = "build-host"):
    return load_durable_config(env=env or {}, runtime=runtime, hostname=hostname)


# -- the plan's documented defaults -----------------------------------------------------


def test_defaults_match_the_plan(runtime: RuntimeConfig):
    config = _load(runtime)
    assert config.address == "127.0.0.1:7233"
    assert DEFAULT_TEMPORAL_ADDRESS == "127.0.0.1:7233"
    assert config.temporal_host == "127.0.0.1"
    assert config.temporal_port == 7233
    assert config.namespace == "codna"
    assert DEFAULT_NAMESPACE == "codna"
    assert config.task_queue == "codna.agent.build-host"
    assert config.graph_engine == "langgraph"
    # The sidecar comes from runtime/config.py, not from a second hardcoded port.
    assert config.sidecar_port == 18601
    assert config.sidecar_url == "http://127.0.0.1:18601"


def test_build_id_is_version_plus_graph_version(runtime: RuntimeConfig):
    # NO version literal here on purpose. A hardcoded "0.2.94" made this test a release tripwire that
    # fails the CLI suite on any version bump (it did: CI run 40408563638, pyproject 0.2.95 vs the
    # literal) while proving nothing extra — the real contract is "the build id is THE package
    # version plus THE graph version", so both sides are derived from their single source of truth.
    config = _load(runtime)
    assert config.build_id == build_id_for(__version__, GRAPH_VERSION)
    assert config.build_id == f"codna-{__version__}-g{GRAPH_VERSION}"
    assert config.graph_version == GRAPH_VERSION


def test_paths_land_under_the_codna_runtime_root(runtime: RuntimeConfig):
    config = _load(runtime)
    root = build_runtime_paths().root
    assert config.runs_root == root / "runs"
    assert config.dev_db_path == root / "temporal" / "dev.db"
    assert config.paths.dev_server_bin_dir == root / "temporal" / "bin"
    assert config.paths.logs_dir == root / "logs"


def test_the_real_runtime_is_used_when_none_is_injected():
    """The integration the injected fixture skips: the actual sidecar port from runtime/config."""
    config = load_durable_config(env={}, hostname="integration-host")
    assert config.sidecar_port >= 1024
    assert config.sidecar_url == f"http://127.0.0.1:{config.sidecar_port}"


def test_the_default_hostname_comes_from_the_machine():
    config = load_durable_config(env={})
    assert config.hostname == sanitize_hostname(socket.gethostname())
    assert config.task_queue.startswith("codna.agent.")


# -- overrides --------------------------------------------------------------------------


def test_environment_overrides_are_honoured(runtime: RuntimeConfig, tmp_path: Path):
    config = _load(
        runtime,
        env={
            "CODNA_TEMPORAL_ADDRESS": "127.0.0.1:8233",
            "CODNA_TEMPORAL_NAMESPACE": "codna-dev",
            "CODNA_DURABLE_TASK_QUEUE": "codna.agent.pool",
            "CODNA_DURABLE_ENGINE": "plain",
            "CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "elsewhere"),
        },
    )
    assert config.address == "127.0.0.1:8233"
    assert config.temporal_port == 8233
    assert config.namespace == "codna-dev"
    assert config.task_queue == "codna.agent.pool"
    assert config.graph_engine == "plain"
    assert config.needs_langgraph is False
    assert config.runs_root == tmp_path / "elsewhere"


def test_plain_engine_means_the_extra_is_not_required(runtime: RuntimeConfig):
    # The plan's fallback: with a broken LangGraph surface, `graph_engine=plain` must still run.
    assert _load(runtime).needs_langgraph is True
    assert _load(runtime, env={"CODNA_DURABLE_ENGINE": "plain"}).needs_langgraph is False


# -- validation failures ----------------------------------------------------------------


@pytest.mark.parametrize("address", ["", "127.0.0.1", "127.0.0.1:abc", "127.0.0.1:0", "127.0.0.1:70000", ":7233"])
def test_a_malformed_temporal_address_is_refused(runtime: RuntimeConfig, address: str):
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"CODNA_TEMPORAL_ADDRESS": address})
    assert excinfo.value.code == "invalid_temporal_address"
    assert excinfo.value.details["value"] == address


def test_a_typo_in_the_engine_is_refused_rather_than_defaulted(runtime: RuntimeConfig):
    """The dangerous silent fallback: `langgraphh` must not quietly mean `plain`.

    The two engines produce different workflow history, so a run started under the wrong one cannot
    be replayed by the worker that picks it up.
    """
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"CODNA_DURABLE_ENGINE": "langgraphh"})
    assert excinfo.value.code == "invalid_graph_engine"
    assert excinfo.value.details["allowed"] == ["langgraph", "plain"]


@pytest.mark.parametrize("value", ["with space", "", "queue;drop", "-leading", "x" * 100])
def test_an_invalid_task_queue_override_is_refused(runtime: RuntimeConfig, value: str):
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"CODNA_DURABLE_TASK_QUEUE": value})
    assert excinfo.value.code == "invalid_task_queue"


def test_an_invalid_namespace_is_refused(runtime: RuntimeConfig):
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"CODNA_TEMPORAL_NAMESPACE": "codna/other"})
    assert excinfo.value.code == "invalid_namespace"


def test_an_empty_hostname_is_refused_rather_than_guessed():
    with pytest.raises(DurableConfigError) as excinfo:
        sanitize_hostname("   ")
    assert excinfo.value.code == "invalid_hostname"


# -- hostname sanitization --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("angels-mbp.lan", "angels-mbp.lan"),
        ("Angel's MacBook.local", "Angel-s-MacBook.local"),
        ("Build Host 01", "Build-Host-01"),
        ("--edge--", "edge"),
        ("a" * 100, "a" * 63),
    ],
)
def test_hostnames_are_reduced_to_a_task_queue_safe_token(raw: str, expected: str):
    assert sanitize_hostname(raw) == expected


def test_the_task_queue_namespace_is_stable():
    assert task_queue_for_host("h") == "codna.agent.h"


# -- workflow ids -----------------------------------------------------------------------


def test_workflow_ids_are_derived_and_stable(runtime: RuntimeConfig):
    config = _load(runtime)
    assert config.workflow_id("run-1") == "task:run-1"
    assert config.run_workflow_id("run-1") == "run:run-1"
    # Derivation must be a pure function: the same run always maps to the same workflow id, or a
    # retry would start a second workflow.
    assert config.workflow_id("run-1") == config.workflow_id("run-1")


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", ".dot", "a..b", "x" * 200, "sp ace"])
def test_workflow_ids_refuse_an_unsafe_run_id(runtime: RuntimeConfig, run_id: str):
    config = _load(runtime)
    with pytest.raises(DurableConfigError) as excinfo:
        config.workflow_id(run_id)
    assert excinfo.value.code == "invalid_run_id"


def test_a_valid_run_id_passes_through():
    assert validate_run_id("run-2026-10-26_abc.1") == "run-2026-10-26_abc.1"


# -- paths ------------------------------------------------------------------------------


def test_run_paths_are_derived_from_the_runs_root(tmp_path: Path):
    paths = DurablePaths(
        root=tmp_path,
        temporal_dir=tmp_path / "temporal",
        dev_db_path=tmp_path / "temporal" / "dev.db",
        dev_server_bin_dir=tmp_path / "temporal" / "bin",
        runs_root=tmp_path / "runs",
        logs_dir=tmp_path / "logs",
    )
    assert paths.run_dir("run-1") == tmp_path / "runs" / "run-1"
    assert paths.ledger_path("run-1") == tmp_path / "runs" / "run-1" / "ledger.ndjson"
    assert paths.parquet_path("run-1") == tmp_path / "runs" / "run-1" / "ledger.parquet"


# -- temporal profiles: local (unchanged) vs cloud ---------------------------------------

_CLOUD_ENV = {
    "TEMPORAL_ENDPOINT": "codna.cihxb.tmprl.cloud:7233",
    "TEMPORAL_NAMESPACE": "codna.cihxb",
    "TEMPORAL_API_KEY": "a-fake-key-for-tests",
}


def test_with_no_cloud_variables_the_profile_is_local_and_nothing_else_changed(
    runtime: RuntimeConfig,
):
    """The default path must be the historical one, byte for byte.

    This is the regression guard for "cloud support" as a change: if adding it moved the local
    address, namespace, profile or connect options, every existing dev-server user would be
    pointed somewhere else.
    """
    config = _load(runtime)
    assert config.profile == "local"
    assert config.address == DEFAULT_TEMPORAL_ADDRESS
    assert config.namespace == DEFAULT_NAMESPACE
    assert config.api_key_env is None
    assert config.requires_tls is False
    assert config.requires_api_key is False
    assert config.owns_local_server is True
    # THE contract that keeps `Client.connect(address, namespace=...)` unchanged: no options at all.
    assert config.connect_options() == {}
    assert config.temporal_api_key() is None


def test_cloud_variables_alone_select_the_cloud_profile(runtime: RuntimeConfig):
    """Inference, not a flag: exporting the three variables is the whole configuration step."""
    config = _load(runtime, env=dict(_CLOUD_ENV))
    assert config.profile == "cloud"
    assert config.address == "codna.cihxb.tmprl.cloud:7233"
    assert config.temporal_host == "codna.cihxb.tmprl.cloud"
    assert config.temporal_port == 7233
    assert config.namespace == "codna.cihxb"
    assert config.api_key_env == "TEMPORAL_API_KEY"
    assert config.requires_tls is True
    assert config.owns_local_server is False


def test_the_explicit_profile_parameter_outranks_the_environment(runtime: RuntimeConfig):
    """`--temporal-profile local` on a shell that exports TEMPORAL_* is a deliberate opt-out."""
    config = load_durable_config(env=dict(_CLOUD_ENV), runtime=runtime, profile="local")
    assert config.profile == "local"
    assert config.address == DEFAULT_TEMPORAL_ADDRESS
    assert config.connect_options() == {}


def test_the_profile_environment_variable_is_honoured(runtime: RuntimeConfig):
    config = _load(runtime, env={"CODNA_TEMPORAL_PROFILE": "cloud", **_CLOUD_ENV})
    assert config.profile == "cloud"
    # ...and local is a real choice even with the cloud variables present.
    local = _load(runtime, env={"CODNA_TEMPORAL_PROFILE": "LOCAL", **_CLOUD_ENV})
    assert local.profile == "local"


@pytest.mark.parametrize("value", ["clod", "", "aws", "cloudy"])
def test_a_typo_in_the_profile_is_refused_rather_than_inferred(
    runtime: RuntimeConfig, value: str
):
    """A typo must not fall through to an inference that reads unrelated variables."""
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"CODNA_TEMPORAL_PROFILE": value})
    assert excinfo.value.code == "invalid_temporal_profile"
    assert excinfo.value.details["allowed"] == ["local", "cloud"]


@pytest.mark.parametrize("value,expected", [("Cloud", "cloud"), ("  cloud  ", "cloud"), ("LOCAL", "local")])
def test_the_profile_spelling_is_case_and_whitespace_folded(
    value: str, expected: str
):
    """What a person types should work; nothing ELSE is silently reinterpreted."""
    assert parse_profile(value) == expected
    assert parse_profile(None) is None


def test_cloud_takes_its_api_key_from_the_environment_and_never_stores_it(
    runtime: RuntimeConfig,
):
    """A frozen dataclass with a generated ``repr`` is the last place a credential belongs."""
    config = _load(runtime, env=dict(_CLOUD_ENV))
    assert _CLOUD_ENV["TEMPORAL_API_KEY"] not in repr(config)
    assert config.temporal_api_key(env=dict(_CLOUD_ENV)) == _CLOUD_ENV["TEMPORAL_API_KEY"]
    # The options carry it (that is their job) but never format it.
    options = config.connect_options(env=dict(_CLOUD_ENV))
    assert options["tls"] is True
    assert options["api_key"] == _CLOUD_ENV["TEMPORAL_API_KEY"]
    assert _CLOUD_ENV["TEMPORAL_API_KEY"] not in repr(options)
    assert _CLOUD_ENV["TEMPORAL_API_KEY"] not in str(options)


def test_a_key_that_disappears_before_connecting_is_refused_with_a_named_error(
    runtime: RuntimeConfig,
):
    """A blank credential at connect time is a cause, not an "unauthorised" from the server."""
    config = _load(runtime, env=dict(_CLOUD_ENV))
    for empty in ({}, {"TEMPORAL_API_KEY": ""}, {"TEMPORAL_API_KEY": "   "}):
        with pytest.raises(DurableConfigError) as excinfo:
            config.connect_options(env=empty)
        assert excinfo.value.code == "temporal_cloud_config_incomplete"
        assert excinfo.value.details["missing"] == ["TEMPORAL_API_KEY"]


@pytest.mark.parametrize(
    "missing,expected",
    [
        ("TEMPORAL_ENDPOINT", "TEMPORAL_ENDPOINT (or TEMPORAL_ADDRESS)"),
        ("TEMPORAL_NAMESPACE", "TEMPORAL_NAMESPACE"),
        ("TEMPORAL_API_KEY", "TEMPORAL_API_KEY"),
    ],
)
def test_a_partial_cloud_configuration_is_refused_naming_the_missing_variable(
    runtime: RuntimeConfig, missing: str, expected: str
):
    """No silent fallback to loopback: a half-configured cloud sends work nowhere."""
    env = {key: value for key, value in _CLOUD_ENV.items() if key != missing}
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env=env)
    assert excinfo.value.code == "temporal_cloud_config_incomplete"
    assert expected in excinfo.value.details["missing"]
    # The detail map is printed; it must name variables, never values.
    assert "a-fake-key-for-tests" not in str(excinfo.value.details)


def test_a_blank_cloud_variable_counts_as_missing_not_as_unset(runtime: RuntimeConfig):
    """A template that rendered an empty secret is a misconfiguration, not silence."""
    env = dict(_CLOUD_ENV)
    env["TEMPORAL_NAMESPACE"] = ""
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env=env)
    assert excinfo.value.code == "temporal_cloud_config_incomplete"
    assert "TEMPORAL_NAMESPACE" in excinfo.value.details["missing"]


def test_temporal_address_is_accepted_as_a_cloud_endpoint_source(runtime: RuntimeConfig):
    """An operator who exports one address should not have to export it again under a second name."""
    env = {key: value for key, value in _CLOUD_ENV.items() if key != "TEMPORAL_ENDPOINT"}
    env["TEMPORAL_ADDRESS"] = "codna.cihxb.tmprl.cloud:7233"
    assert _load(runtime, env=env).address == "codna.cihxb.tmprl.cloud:7233"


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("codna.cihxb.tmprl.cloud:7233", ("codna.cihxb.tmprl.cloud", 7233)),
        ("codna.cihxb.tmprl.cloud", ("codna.cihxb.tmprl.cloud", 7233)),
        ("https://codna.cihxb.tmprl.cloud", ("codna.cihxb.tmprl.cloud", 7233)),
        ("grpcs://codna.cihxb.tmprl.cloud:7233", ("codna.cihxb.tmprl.cloud", 7233)),
        ("tls://codna.cihxb.tmprl.cloud:7233", ("codna.cihxb.tmprl.cloud", 7233)),
    ],
)
def test_cloud_endpoints_are_accepted_in_the_forms_operators_copy(
    endpoint: str, expected: tuple[str, int]
):
    assert split_temporal_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://codna.cihxb.tmprl.cloud:7233",  # a cleartext downgrade: the key would travel in the clear
        "codna.cihxb.tmprl.cloud:7233/path",
        "codna.cihxb.tmprl.cloud:7233?x=1",
        "codna.cihxb.tmprl.cloud:7233#f",
        "user@codna.cihxb.tmprl.cloud:7233",
        "codna.cihxb.tmprl.cloud:abc",
        "codna.cihxb.tmprl.cloud:0",
        "codna.cihxb.tmprl.cloud:70000",
        "",
        "localhost:7233",
        "codna:7233",
    ],
)
def test_cloud_endpoints_that_cannot_be_a_cloud_server_are_refused(endpoint: str):
    with pytest.raises(DurableConfigError) as excinfo:
        split_temporal_endpoint(endpoint)
    assert excinfo.value.code == "invalid_temporal_endpoint"


# -- a loopback TEMPORAL_ADDRESS means a LOCAL server, never a Cloud attempt --------------


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1:7233", "localhost:7233", "[::1]:7233", "http://127.0.0.1:7233"],
)
def test_a_loopback_temporal_address_is_the_local_profile(runtime: RuntimeConfig, address: str):
    """``TEMPORAL_ADDRESS`` is the Temporal SDK's own variable.

    A developer who exports it for ``temporal``/the SDK — pointing at a local dev server — used to be
    inferred as a *Cloud* attempt, and every durable command then refused with
    ``temporal_cloud_config_incomplete`` before reading anything. Measured on this host with
    ``TEMPORAL_ADDRESS=127.0.0.1:7233`` exported in the shell: ``codna durable dev-server status``
    exited nonzero naming ``TEMPORAL_API_KEY`` and ``TEMPORAL_NAMESPACE``. A loopback address can only
    be a local server; the Cloud profile cannot connect to one.
    """
    config = _load(runtime, env={"TEMPORAL_ADDRESS": address})
    assert config.profile == "local"
    assert config.address == DEFAULT_TEMPORAL_ADDRESS


def test_a_loopback_address_beside_a_credential_stays_cloud(runtime: RuntimeConfig):
    """A key means the operator IS configuring Cloud, however the address looks: never downgrade it.

    Reclassifying this to ``local`` would silently ignore a credential the operator deliberately
    exported. The profile stays ``cloud``; whatever the Cloud profile then says about that address is
    the Cloud profile's own rule, not this inference's.
    """
    env = {"TEMPORAL_ADDRESS": "127.0.0.1:7233", "TEMPORAL_NAMESPACE": "ns", "TEMPORAL_API_KEY": "k"}
    assert _load(runtime, env=env).profile == "cloud"


def test_an_explicit_cloud_profile_is_never_reclassified(runtime: RuntimeConfig):
    """ ``CODNA_TEMPORAL_PROFILE=cloud`` is a statement; it outranks the loopback inference."""
    with pytest.raises(DurableConfigError) as excinfo:
        _load(
            runtime,
            env={"CODNA_TEMPORAL_PROFILE": "cloud", "TEMPORAL_ADDRESS": "127.0.0.1:7233"},
        )
    assert excinfo.value.code == "temporal_cloud_config_incomplete"


def test_a_remote_address_with_no_credential_still_names_what_is_missing(
    runtime: RuntimeConfig,
):
    """The partial-Cloud refusal this inference exists for is unchanged."""
    with pytest.raises(DurableConfigError) as excinfo:
        _load(runtime, env={"TEMPORAL_ADDRESS": "codna.cihxb.tmprl.cloud:7233"})
    assert excinfo.value.code == "temporal_cloud_config_incomplete"
    assert "TEMPORAL_API_KEY" in excinfo.value.details["missing"]
