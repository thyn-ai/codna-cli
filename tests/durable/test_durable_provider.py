"""Provider selection: a durable run can finally say WHICH model provider runs its turns.

The gap this closes: `TaskSpec` had no provider field at all, so `_ensure_session` built a session
spec with only `owner`. The engine therefore fell back to its own default (`anthropic`), which made
every durable turn on a machine holding only `OPENAI_API_KEY` die with

    ProviderCredentialError: no API key in env for 'anthropic'

and made a keyless local provider (Ollama / LM Studio / an OpenAI-compatible endpoint) unusable for a
durable run. The consequences were real and not theoretical: the nightly soak forwarded
`OPENAI_API_KEY` and the durable path never read it, so the soak exercised the ledger and never
reached a checkpoint, and D2's 24-hour criterion stayed `not_established`.

What is asserted here, in the order the plumbing runs:

* **The contract refuses bad coordinates** — blank/oversized/bad-charset provider, a non-http base
  URL, a base URL that needs a provider it has not got, and (the important one) a provider KEY VALUE
  pasted into any of these fields, which must be a refusal because a `TaskSpec` enters workflow
  history and is mirrored into the ledger.
* **The coordinates reach `create_session`** — asserted as the EXACT body the fake sidecar received,
  not as a call count, because "the provider was passed" is the whole claim.
* **The credential is read from the environment at the session, never carried in the spec** — with the
  spec's own fields asserted to be key-free even when a key is present in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

from codna.durable.contracts import (
    KEYLESS_PROVIDERS,
    PROVIDER_ENV_KEYS,
    ContractError,
    TaskContext,
    TaskSpec,
    provider_env_key_names,
    provider_session_spec,
    resolve_provider_env_key,
)
from tests.durable import fakes

# -- builders ------------------------------------------------------------------------------------


def _spec(**overrides: Any) -> TaskSpec:
    """A valid spec, so each test changes exactly the field it is about."""
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the flaky retry in the client",
        "repo_dir": "/tmp/repo",
        "verify_commands": ("true",),
        "engine": "plain",
    }
    payload.update(overrides)
    return TaskSpec(**payload)


def _validated(**overrides: Any) -> TaskSpec:
    spec = _spec(**overrides)
    spec.validate()
    return spec


# -- the fields exist and are explicit -----------------------------------------------------------


def test_task_spec_declares_the_four_provider_coordinates():
    """An explicit field set, so a canonical-or-formatting regression is visible here."""
    import dataclasses

    names = {field.name for field in dataclasses.fields(TaskSpec)}
    assert {"provider", "model", "base_url", "provider_key_source"} <= names


def test_absent_coordinates_mean_the_providers_own_defaults():
    """The pre-change behaviour is preserved: no provider means the engine resolves one."""
    spec = _validated()
    assert (spec.provider, spec.model, spec.base_url, spec.provider_key_source) == (
        None,
        None,
        None,
        None,
    )


def test_the_payload_round_trips_every_provider_coordinate():
    spec = _validated(
        provider="openai-compatible",
        model="qwen3.6:latest",
        base_url="http://127.0.0.1:11434/v1",
        provider_key_source="environment",
    )
    payload = spec.to_payload()
    assert payload["provider"] == "openai-compatible"
    assert payload["model"] == "qwen3.6:latest"
    assert payload["base_url"] == "http://127.0.0.1:11434/v1"
    assert payload["provider_key_source"] == "environment"
    # ...and the decoded spec is the same value, so a workflow round trip is lossless.
    assert TaskSpec(**{**payload, "verify_commands": tuple(payload["verify_commands"])}) == spec


def test_the_task_context_carries_the_coordinates_to_the_activity():
    """A request is what history hands a retried activity, so the context must carry them too."""
    spec = _validated(
        provider="openai",
        model="gpt-4.1",
        base_url="http://127.0.0.1:8080",
        provider_key_source="environment",
    )
    context = TaskContext.from_spec(spec)
    assert context.provider_fields() == {
        "provider": "openai",
        "model": "gpt-4.1",
        "base_url": "http://127.0.0.1:8080",
        "provider_key_source": "environment",
    }
    context.validate()  # the activity-side re-validation accepts it


# -- validation: the provider id ------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        "",  # empty
        "   ",  # blank
        "\t\n",  # whitespace only
        "OpenAI",  # not lowercase (the engine's aliases are)
        "open ai",  # whitespace inside
        "openai;rm -rf /",  # a shell-looking value must never reach a command
        "a" * 65,  # over the 64-character limit
        "prov/ider",  # a path separator
        "-openai",  # must start alphanumeric
        "ope\nai",  # an INTERNAL newline — `strip()` cannot remove it, and it is not an alias
        "openai\u0000",  # a NUL byte is not a provider id
    ],
)
def test_a_blank_oversized_or_bad_charset_provider_is_refused(provider: str):
    with pytest.raises(ContractError) as caught:
        _validated(provider=provider)
    assert caught.value.code in ("contract_invalid", "contract_contains_secret")


@pytest.mark.parametrize(
    "provider", ["anthropic", "openai", "openai-compatible", "ollama", "lmstudio", "litellm", "xai"]
)
def test_a_real_provider_id_is_accepted(provider: str):
    assert _validated(provider=provider).provider == provider


def test_a_provider_is_trimmed_before_it_is_recorded():
    """" openai " is the id `openai`: the recorded value must be canonical, not the caller's spacing."""
    assert _validated(provider="  openai  ").provider == "openai"


# -- validation: the model id ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["", "  ", "gpt 4.1", "model\nwith-newline", "m" * 201, "-leading-dash"],
)
def test_a_blank_or_malformed_model_is_refused(model: str):
    with pytest.raises(ContractError) as caught:
        _validated(provider="openai", model=model)
    assert caught.value.code in ("contract_invalid", "contract_contains_secret")


@pytest.mark.parametrize(
    "model", ["gpt-4.1", "claude-sonnet-4", "qwen3.6:latest", "anthropic/claude-sonnet-4", "llama3.1:70b"]
)
def test_a_real_model_id_is_accepted(model: str):
    assert _validated(provider="openai", model=model).model == model


# -- validation: the base URL ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",  # not http(s)
        "file:///etc/passwd",  # a file URL must never become a fetch target
        "example.com:11434",  # no scheme at all
        "http://",  # no host
        "http://127.0.0.1:11434/v1?key=abc",  # a query would be silently dropped
        "http://127.0.0.1:11434/v1#frag",  # a fragment likewise
        "http://user:pass@127.0.0.1:11434",  # embedded credentials
        "http://127.0.0.1:99999",  # an out-of-range port
        "http://127.0.0.1:not-a-port",  # a non-numeric port
        "http://127.0.0.1:11434 /v1",  # whitespace
    ],
)
def test_a_non_http_or_unsafe_base_url_is_refused(base_url: str):
    with pytest.raises(ContractError) as caught:
        _validated(provider="ollama", base_url=base_url)
    assert caught.value.code in ("contract_invalid", "contract_contains_secret")


def test_a_cleartext_remote_base_url_is_refused():
    """A provider key sent over plain http to another host is exposed in transit."""
    with pytest.raises(ContractError) as caught:
        _validated(provider="openai-compatible", base_url="http://inference.example.com/v1")
    assert caught.value.code == "contract_invalid"
    assert "loopback" in str(caught.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434",
        "http://127.0.0.1:11434/v1",
        "http://localhost:1234/v1",
        "http://localhost:4000",
        "http://[::1]:11434",
        "http://127.5.5.5:8080/v1",  # the whole 127/8 range is loopback
        "https://api.example.com/v1",  # a remote HTTPS endpoint is fine
    ],
)
def test_a_loopback_or_https_base_url_is_accepted(base_url: str):
    assert _validated(provider="ollama", base_url=base_url).base_url == base_url


def test_a_base_url_without_a_provider_is_refused():
    """An endpoint override with nothing to point it at would be a silent no-op at the engine."""
    with pytest.raises(ContractError) as caught:
        _validated(base_url="http://127.0.0.1:11434")
    assert caught.value.code == "contract_invalid"


# -- validation: the key source -------------------------------------------------------------------


def test_an_unknown_key_source_is_refused():
    with pytest.raises(ContractError) as caught:
        _validated(provider="openai", provider_key_source="inline")
    assert caught.value.code == "contract_invalid"


def test_a_key_source_without_a_provider_is_refused():
    """There is no env var to read without a provider, so the combination is incoherent."""
    with pytest.raises(ContractError) as caught:
        _validated(provider_key_source="environment")
    assert caught.value.code == "contract_invalid"


def test_the_environment_key_source_is_accepted():
    spec = _validated(provider="openai", provider_key_source="environment")
    assert spec.provider_key_source == "environment"


# -- validation: a KEY VALUE must never be representable ------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        # Every provider-key shape `state.find_secret_like_strings` knows about, in each of the four
        # fields. A `TaskSpec` enters Temporal history AND is mirrored into the run's ledger, which is
        # the artifact people paste into tickets — so a pasted key has to be a refusal, not a record.
        ("provider", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"),
        ("provider", "sk-proj-abcdefghijklmnopqrstuvwxyz012345"),
        ("model", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"),
        ("model", "sk-proj-abcdefghijklmnopqrstuvwxyz012345"),
        ("model", "AIzaSyA1234567890abcdefghijklmnopqrstuv"),
        ("model", "xai-abcdefghijklmnopqrstuvwx"),
        ("model", "hf_abcdefghijklmnopqrstuvwxyz"),
        ("model", "glpat-abcdefghijklmnopqrstuv"),
        ("model", "AKIAIOSFODNN7EXAMPLE"),
        ("base_url", "http://127.0.0.1:11434/v1/sk-ant-api03-abcdefghijklmnopqrstuvwxyz01"),
    ],
)
def test_a_key_value_in_a_spec_field_is_refused(field: str, value: str):
    """The secret-shaped validator is what makes "no credential in history" enforceable."""
    overrides: dict[str, Any] = {"provider": "openai"}
    overrides[field] = value
    with pytest.raises(ContractError) as caught:
        _validated(**overrides)
    assert caught.value.code == "contract_contains_secret"
    # The finding names the PATH, never the value — the error must be safe to log.
    findings = caught.value.details["findings"]
    assert any(finding["path"] == field for finding in findings)
    assert value not in str(caught.value.as_dict())


def test_a_key_in_a_bearer_header_shape_is_refused_in_the_model():
    with pytest.raises(ContractError) as caught:
        _validated(provider="openai", model="Bearer abcdefghijklmnopqrstuvwxyz0123456789")
    assert caught.value.code == "contract_contains_secret"


def test_a_key_on_a_task_context_is_refused_by_the_activity_side_revalidation():
    """A request decoded from history is re-validated, so the two halves cannot disagree."""
    context = TaskContext(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        repo_dir="/tmp/repo",
        base_branch="main",
        provider="openai",
        model="sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    with pytest.raises(ContractError) as caught:
        context.validate()
    assert caught.value.code == "contract_contains_secret"


# -- the env-key resolution (pure) ----------------------------------------------------------------


def test_the_key_env_map_agrees_with_the_keystores_aliases():
    """A drift between the two would be "the run cannot find its key", hours into a soak.

    `codna.keystore` is the CLI's keystore-facing spelling. The durable contract keeps its own copy
    because importing keystore pulls `codna.runtime.config` → httpx, which the workflow sandbox
    refuses. This test is what keeps the copy honest.
    """
    from codna.keystore import PROVIDER_ALIASES

    for provider, alias in PROVIDER_ALIASES.items():
        if provider in ("codna", "engine"):
            continue  # the Codna engine key is not a model provider
        assert provider in PROVIDER_ENV_KEYS, provider
        assert alias in PROVIDER_ENV_KEYS[provider], (provider, alias)


def test_every_keyless_provider_is_known_to_the_env_map():
    for provider in KEYLESS_PROVIDERS:
        assert provider in PROVIDER_ENV_KEYS, provider


def test_resolve_provider_env_key_reads_the_first_set_variable():
    assert resolve_provider_env_key("openai", {"OPENAI_API_KEY": "k-1"}) == "k-1"
    assert resolve_provider_env_key("gemini", {"GEMINI_API_KEY": "g", "GOOGLE_API_KEY": "x"}) == "g"
    # The fallback order matters: Gemini's second var is used only when its first is absent.
    assert resolve_provider_env_key("gemini", {"GOOGLE_API_KEY": "x"}) == "x"


def test_resolve_provider_env_key_refuses_a_blank_value():
    """An empty or whitespace var is "no key", not a key — sending "" would be a 401 at the engine."""
    assert resolve_provider_env_key("openai", {"OPENAI_API_KEY": ""}) is None
    assert resolve_provider_env_key("openai", {"OPENAI_API_KEY": "   "}) is None
    assert resolve_provider_env_key("openai", {}) is None


def test_an_unmapped_provider_has_no_env_names():
    """Not a refusal — the engine may still know it — but the durable path forwards nothing."""
    assert provider_env_key_names("some-future-provider") == ()
    assert resolve_provider_env_key("some-future-provider", {"ANYTHING": "x"}) is None


# -- the session spec sent to create_session ------------------------------------------------------


def _context(**overrides: Any) -> TaskContext:
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_number": 1,
        "repo_dir": "/tmp/repo",
        "base_branch": "main",
    }
    payload.update(overrides)
    return TaskContext(**payload)


def test_provider_session_spec_is_the_exact_body_for_a_hosted_provider():
    """The EXACT dict: every key is a real `SessionRecord` field, and absent values are omitted."""
    context = _context(provider="openai", model="gpt-4.1", provider_key_source="environment")
    assert provider_session_spec(context, key="sk-test") == {
        "provider": "openai",
        "model": "gpt-4.1",
        "provider_api_key": "sk-test",
        "require_provider_api_key": True,
    }


def test_provider_session_spec_omits_absent_coordinates_rather_than_sending_none():
    """The engine distinguishes "no provider" from "a provider named null"."""
    assert provider_session_spec(_context(), key=None) == {}


def test_provider_session_spec_for_a_keyless_local_provider():
    context = _context(
        provider="ollama", model="qwen3.6:latest", base_url="http://127.0.0.1:11434"
    )
    assert provider_session_spec(context, key=None) == {
        "provider": "ollama",
        "model": "qwen3.6:latest",
        "base_url": "http://127.0.0.1:11434",
    }


def test_a_keyless_provider_gets_no_credential_even_if_one_exists_in_the_environment():
    """`provider_key_source` is what decides, so a keyless run never forwards a stray env key."""
    context = _context(provider="ollama", provider_key_source=None)
    assert "provider_api_key" not in provider_session_spec(context, key=None)


# -- end to end: the real activity, the fake sidecar, the EXACT create body ------------------------


def _run_one_turn(tmp_path: Path, spec: TaskSpec) -> fakes.Harness:
    """Drive the REAL `run_cline_turn` body against the fake sidecar and return the harness."""
    from codna.durable.activities import run_cline_turn

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context = TaskContext.from_spec(spec)
    run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    return harness


def test_the_session_body_carries_the_chosen_provider_model_and_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """THE acceptance assertion: `create_session` receives the coordinates the spec chose.

    Asserted as the exact body the fake sidecar recorded, not as a mock's call count. A count would
    pass even if the coordinates were dropped on the way, which is precisely the bug this closes.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-value")

    repo_spec = _validated(
        provider="openai",
        model="gpt-4.1",
        base_url="http://127.0.0.1:8080",
        provider_key_source="environment",
    )
    # `repo_dir` must be the harness's real checkout, so the spec is rebuilt against it.
    spec = _validated(
        repo_dir=str(fakes.make_repo(tmp_path / "unused")),
        provider=repo_spec.provider,
        model=repo_spec.model,
        base_url=repo_spec.base_url,
        provider_key_source=repo_spec.provider_key_source,
    )
    harness = _run_one_turn(tmp_path, spec)

    assert harness.sidecar.sessions, "no session was created"
    body = harness.sidecar.sessions[0]["spec"]
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4.1"
    assert body["base_url"] == "http://127.0.0.1:8080"
    # The credential is attached by the WORKER from its environment...
    assert body["provider_api_key"] == "sk-test-key-value"
    assert body["require_provider_api_key"] is True
    # ...and the owner survives: the provider is an ADDITION to the spec, not a replacement.
    assert body["owner"] == "codna-run-run-1"


def test_the_session_body_never_carries_a_key_when_the_source_is_not_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A hosted provider with no source keeps the pre-change behaviour: the engine reads its own env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-forwarded")

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _validated(repo_dir=str(repo), provider="openai", model="gpt-4.1")
    context = TaskContext.from_spec(spec)

    from codna.durable.activities import run_cline_turn

    run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    body = harness.sidecar.sessions[0]["spec"]
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4.1"
    assert "provider_api_key" not in body
    assert "require_provider_api_key" not in body


def test_a_keyless_local_provider_reaches_create_session_with_no_credential(tmp_path: Path):
    """Track P's keyless path, now usable for a durable run."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _validated(
        repo_dir=str(repo),
        provider="ollama",
        model="qwen3.6:latest",
        base_url="http://127.0.0.1:11434",
        provider_key_source=None,
    )
    context = TaskContext.from_spec(spec)

    from codna.durable.activities import run_cline_turn

    run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    body = harness.sidecar.sessions[0]["spec"]
    # EXACT body, keyless: no credential and no fail-closed flag, so the engine keeps its own environment.
    # `multi_turn` is present because a durable task's session carries every one of its turns — without it
    # the SDK finalizes the session at the end of turn 1 and the next POST fails `session_not_found`.
    # `policy` is present because a durable (unattended) run is GOVERNED: the sidecar must refuse a
    # repository with no `.codna/policy.yaml` rather than run under the approval profile.
    assert body == {
        "owner": "codna-run-run-1",
        "provider": "ollama",
        "model": "qwen3.6:latest",
        "base_url": "http://127.0.0.1:11434",
        "multi_turn": True,
        "policy": {"required": True},
    }


def test_a_read_only_turn_forwards_the_provider_and_still_forbids_writing(tmp_path: Path):
    """The plan turn is a read-only turn; the provider must survive the empty write allow-list."""
    monkeypatch_key = "ANTHROPIC_API_KEY"
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _validated(
        repo_dir=str(repo), provider="anthropic", model="claude-sonnet-4", provider_key_source=None
    )
    context = TaskContext.from_spec(spec)

    from codna.durable.activities import plan_task

    plan_task(context.plan_request(spec.brief, spec.initial_state()))

    body = harness.sidecar.sessions[0]["spec"]
    assert body["allowed_write_paths"] == []
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-sonnet-4"
    assert monkeypatch_key not in body


def test_the_spec_a_workflow_decodes_keeps_the_provider_coordinates(tmp_path: Path):
    """The workflow-side decoder (`workflow_support.spec_from`) must not drop them.

    A field that reaches `to_payload` but is not read back is the exact shape of this bug: the CLI
    would accept `--provider`, record it, and the activity would still run on the engine default.
    """
    from codna.durable.workflow_support import spec_from

    spec = _validated(
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
        provider_key_source="environment",
    )
    decoded = spec_from(spec.to_payload())
    assert decoded.provider == "openrouter"
    assert decoded.model == "anthropic/claude-sonnet-4"
    assert decoded.base_url == "https://openrouter.ai/api/v1"
    assert decoded.provider_key_source == "environment"


def test_a_payload_recorded_before_the_provider_fields_still_decodes(tmp_path: Path):
    """A run already in flight has a payload with none of these keys; it must still replay."""
    from codna.durable.workflow_support import spec_from

    payload = {
        "schema_version": 1,
        "run_id": "run-old",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "an old run",
        "repo_dir": "/tmp/repo",
        "base_branch": "main",
        "verify_commands": [],
        "max_attempts": 3,
        "engine": "plain",
    }
    decoded = spec_from(payload)
    assert decoded.provider is None
    assert decoded.provider_key_source is None


def test_the_provider_does_not_leak_into_the_ledger_as_a_credential(tmp_path: Path):
    """The ledger records the provider, never the key — it is the artifact people paste into tickets."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _validated(repo_dir=str(repo), provider="openai", model="gpt-4.1")
    context = TaskContext.from_spec(spec)

    from codna.durable.activities import run_cline_turn

    run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    ledger_text = "\n".join(
        str(record.payload) for record in harness.ledger().records()
    )
    assert "provider_api_key" not in ledger_text
    # A key-shaped value would be caught by the state validator, but assert the shape directly too:
    # the record must not carry a credential even if some future field tried to.
    assert "sk-test-key-value" not in ledger_text
    assert "require_provider_api_key" not in ledger_text


# -- the CLI threads the flags through ------------------------------------------------------------


@pytest.fixture
def durable_config(tmp_path: Path) -> Any:
    """A real `DurableConfig` whose runs root is the test's directory.

    The runtime config is injected (as `test_run_cli.py` does) so this never reads or writes a
    developer's `~/.codna`; only `run_payload_for`'s spec validation is under test here.
    """
    from codna.durable.config import load_durable_config
    from codna.runtime.config import RuntimeConfig, build_runtime_paths

    runtime = RuntimeConfig(
        codna_home=tmp_path,
        port_base=18700,
        engine_port=18700,
        sidecar_port=18701,
        engine_url="http://127.0.0.1:18700",
        sidecar_url="http://127.0.0.1:18701",
        remote_engine_url=None,
        ignored_engine_url=None,
        engine_url_override=None,
        paths=build_runtime_paths(),
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
    return load_durable_config(
        env={"CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "runs"), "CODNA_DURABLE_ENGINE": "plain"},
        runtime=runtime,
        hostname="build-host",
    )


def test_run_payload_for_records_the_provider_in_every_task(tmp_path: Path, durable_config: Any):
    """`codna run --provider ...` must reach the START PAYLOAD, not just the parser."""
    from codna.durable import cli as durable_cli

    repo = tmp_path / "repo"
    repo.mkdir()
    payload = durable_cli.run_payload_for(
        config=durable_config,
        run_id="run-1",
        repo_dir=repo,
        briefs=["fix a", "fix b"],
        base_branch="main",
        verify_commands=["true"],
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=1,
        engine="plain",
        provider="ollama",
        model="qwen3.6:latest",
        base_url="http://127.0.0.1:11434",
        provider_key_source="environment",
    )
    for task in payload["tasks"]:
        assert task["provider"] == "ollama"
        assert task["model"] == "qwen3.6:latest"
        assert task["base_url"] == "http://127.0.0.1:11434"
        assert task["provider_key_source"] == "environment"


def test_run_payload_for_refuses_a_bad_provider_before_anything_starts(
    tmp_path: Path, durable_config: Any
):
    """A typo must be an exit code, not a workflow that fails inside Temporal's history."""
    from codna.durable import cli as durable_cli
    from codna.durable.cli import DurableCliError

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=["fix a"],
            base_branch="main",
            verify_commands=[],
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
            provider="not a provider",
        )
    assert caught.value.code == "contract_invalid"


def test_run_payload_for_refuses_a_key_pasted_into_a_provider_field(
    tmp_path: Path, durable_config: Any
):
    from codna.durable import cli as durable_cli
    from codna.durable.cli import DurableCliError

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=["fix a"],
            base_branch="main",
            verify_commands=[],
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
            provider="openai",
            model="sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
        )
    assert caught.value.code == "contract_contains_secret"


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["run", "--brief", "b", "--provider", "openai"], ("openai", None, None, None)),
        (
            ["run", "--brief", "b", "--provider", "ollama", "--model", "qwen3.6:latest"],
            ("ollama", "qwen3.6:latest", None, None),
        ),
        (
            [
                "run",
                "--brief",
                "b",
                "--provider",
                "openai",
                "--base-url",
                "http://127.0.0.1:8080",
                "--provider-key-source",
                "environment",
            ],
            ("openai", None, "http://127.0.0.1:8080", "environment"),
        ),
        # Both surfaces share the flag set: `codna durable run` must accept them too, or a documented
        # option would not exist on the path scripts use.
        (["durable", "run", "--brief", "b", "--provider", "openai"], ("openai", None, None, None)),
        (["durable", "run", "--brief", "b"], (None, None, None, None)),
        (["run", "--brief", "b"], (None, None, None, None)),
    ],
)
def test_both_run_surfaces_accept_the_provider_flags(argv: list[str], expected: tuple):
    """One flag set for `codna run` and `codna durable run` (`add_run_flags` owns both)."""
    from codna import cli as top_cli

    args = top_cli.build_parser().parse_args(argv)
    assert (args.provider, args.model, args.base_url, args.provider_key_source) == expected


def test_every_request_coordinate_has_a_cli_flag_and_no_key_source_does():
    """The third side of the triangle: `provider.py`'s vocabulary == the route's == the CLI's flags.

    `provider.py` owns the three coordinates, `POST /runs` accepts them, and `add_run_flags` must
    expose a flag for each — otherwise one entrance silently lacks a field the others have, which is
    the exact hole this change closed. The key SOURCE is the asymmetric one on purpose: the CLI accepts
    it (an operator typing `environment`), the HTTP surface does not (a caller does not decide where a
    credential comes from).
    """
    from codna import cli as top_cli
    from codna.durable.provider import PROVIDER_COORDINATE_FIELDS, PROVIDER_KEY_SOURCES

    # Parsed rather than introspected: the destination name IS the field name, so an accepted flag
    # that sets nothing (or a field with no flag) shows up here as a missing attribute.
    values = {"provider": "ollama", "model": "qwen3.6:latest", "base_url": "http://127.0.0.1:11434"}
    argv = ["run", "--brief", "b"]
    for field in PROVIDER_COORDINATE_FIELDS:
        argv.extend([f"--{field.replace('_', '-')}", values[field]])
    argv.extend(["--provider-key-source", PROVIDER_KEY_SOURCES[0]])
    args = top_cli.build_parser().parse_args(argv)
    for field in (*PROVIDER_COORDINATE_FIELDS, "provider_key_source"):
        assert getattr(args, field) == ("environment" if field == "provider_key_source" else values[field])
    assert PROVIDER_KEY_SOURCES == ("environment",)


def test_the_parser_refuses_an_unsupported_key_source():
    """The choices come from the contract, so a source the spec would refuse cannot be typed."""
    from codna import cli as top_cli

    with pytest.raises(SystemExit) as caught:
        top_cli.build_parser().parse_args(
            ["run", "--brief", "b", "--provider-key-source", "inline"]
        )
    assert caught.value.code == 2


def test_there_is_no_flag_that_puts_a_key_on_the_command_line():
    """A `--provider-api-key` would land in the process table and every shell history.

    The absence is the design: the only supported source is the worker's own environment, which is
    why `provider_key_source` is a closed set with one member.
    """
    from codna import cli as top_cli

    help_text = top_cli.build_parser().format_help()
    assert "--provider-api-key" not in help_text
    assert "--api-key" not in help_text


# -- the Run API's entrance: one validator, two surfaces -----------------------------------------
#
# `POST /runs` is what the IDE submits through, and it had no provider fields at all: an IDE run got
# the engine's default provider and died on any host whose key was for another one. The fix is
# `provider_coordinates_from_request`, and these tests pin that it is a CALL into the same validator
# rather than a second rulebook — same codes, same messages, same security rules.


def test_the_request_adapter_accepts_what_the_spec_accepts():
    """Same coordinates in, same coordinates out — including the derived key source."""
    from codna.durable.provider import provider_coordinates_from_request

    coordinates = provider_coordinates_from_request(
        {"provider": "  openai ", "model": " gpt-4.1 ", "base_url": "http://127.0.0.1:8080/v1"}
    )
    assert coordinates.provider == "openai"
    assert coordinates.model == "gpt-4.1"
    assert coordinates.base_url == "http://127.0.0.1:8080/v1"
    # Derived, never read from the request: naming a provider means "read the key from the worker's
    # own environment", which is the durable path's only source.
    assert coordinates.provider_key_source == "environment"


def test_an_absent_request_has_no_coordinates_at_all():
    """Backward compatibility is a value, not a comment: nothing supplied means nothing set."""
    from codna.durable.provider import provider_coordinates_from_request

    coordinates = provider_coordinates_from_request({"briefs": ["b"]})
    assert (coordinates.provider, coordinates.model, coordinates.base_url) == (None, None, None)
    # No provider means no env var to read, so no source either — not "environment" by habit.
    assert coordinates.provider_key_source is None


@pytest.mark.parametrize(
    "body",
    [
        {"provider": "OpenAI"},
        {"provider": ""},
        {"provider": "a" * 65},
        {"provider": "openai", "base_url": "http://api.example.com/v1"},
        {"provider": "openai", "base_url": "file:///etc/passwd"},
        {"provider": "openai", "base_url": "https://user:p@api.example.com"},
        {"base_url": "http://127.0.0.1:11434"},
    ],
)
def test_the_request_adapter_raises_the_contract_error(body: dict[str, Any]):
    """A refusal is the VALIDATOR's `ContractError` with the validator's code — nothing re-shaped."""
    from codna.durable.provider import provider_coordinates_from_request

    with pytest.raises(ContractError) as caught:
        provider_coordinates_from_request(body)
    assert caught.value.code in ("contract_invalid", "contract_contains_secret")


def test_the_request_adapter_refuses_a_key_in_any_coordinate():
    """The secret screen runs over all three, because all three enter history."""
    from codna.durable.provider import provider_coordinates_from_request

    key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    for body in ({"provider": key}, {"provider": "openai", "model": key}, {"provider": "openai", "base_url": key}):
        with pytest.raises(ContractError) as caught:
            provider_coordinates_from_request(body)
        assert caught.value.code == "contract_contains_secret"
        # The finding names the PATH only. Repeating the value would put it in a log.
        assert key not in str(caught.value.details)


@pytest.mark.parametrize("field", ["provider", "model", "base_url"])
def test_the_request_adapter_refuses_a_non_string_coordinate(field: str):
    """JSON is untrusted: ``3`` is not the provider "3", and an object is not a model id."""
    from codna.durable.provider import provider_coordinates_from_request

    body: dict[str, Any] = {"provider": "openai"}
    body[field] = {"id": "x"} if field != "model" else 7
    with pytest.raises(ContractError) as caught:
        provider_coordinates_from_request(body)
    assert caught.value.code == "contract_invalid"
    assert caught.value.details.get("field") == field


def test_the_request_coordinates_build_the_same_spec_the_cli_builds(tmp_path: Path, durable_config: Any):
    """THE ANTI-DIVERGENCE TEST: the API's payload and the CLI's must be the identical spec.

    Both entrances go through `run_payload_for`; this asserts that the coordinates the API derives from
    a JSON body produce a `TaskSpec` byte-identical to the one the CLI flags produce, so "the API
    accepts a provider" and "the CLI accepts a provider" cannot mean two different things.
    """
    from codna.durable.provider import provider_coordinates_from_request
    from codna.durable.run_cli import run_payload_for

    repo = tmp_path / "repo"
    repo.mkdir()
    coordinates = provider_coordinates_from_request(
        {"provider": "ollama", "model": "qwen3.6:latest", "base_url": "http://127.0.0.1:11434"}
    )
    from_api = run_payload_for(
        config=durable_config,
        run_id="run-same",
        repo_dir=repo,
        briefs=["b"],
        base_branch="main",
        verify_commands=(),
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=3,
        engine="plain",
        provider=coordinates.provider,
        model=coordinates.model,
        base_url=coordinates.base_url,
        provider_key_source=coordinates.provider_key_source,
    )
    from_cli = run_payload_for(
        config=durable_config,
        run_id="run-same",
        repo_dir=repo,
        briefs=["b"],
        base_branch="main",
        verify_commands=(),
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=3,
        engine="plain",
        provider="ollama",
        model="qwen3.6:latest",
        base_url="http://127.0.0.1:11434",
        provider_key_source="environment",
    )
    assert from_api == from_cli


# -- the workflow side: the TaskSpec reaches the graph's config -----------------------------------


def test_the_graphs_metadata_payload_carries_the_provider(tmp_path: Path):
    """`TaskWorkflowGraph` hands the spec over in ``config['metadata']``; the decoder must see it."""
    from codna.durable.contracts import CONFIG_TASK_SPEC, task_spec_from_config

    spec = _validated(
        repo_dir=str(tmp_path / "repo"),
        provider="lmstudio",
        base_url="http://127.0.0.1:1234/v1",
    )
    decoded = task_spec_from_config({"metadata": {CONFIG_TASK_SPEC: spec.to_payload()}})
    assert decoded.provider == "lmstudio"
    assert decoded.base_url == "http://127.0.0.1:1234/v1"
