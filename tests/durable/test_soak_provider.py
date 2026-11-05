"""The soak's provider selection: a PURE function over an environment mapping.

Why this is its own file (and its own contract): a soak that cannot select a provider does not test
durability. Before this existed the nightly workflow exported ``OPENAI_API_KEY`` unconditionally, the
durable path never read it, every turn died with ``ProviderCredentialError: no API key in env for
'anthropic'``, and the run ended without a single checkpoint — while the soak still went green having
"exercised the ledger". D2's 24-hour criterion stayed ``not_established`` because of it.

So the selection must be:

* **driven by what the environment actually has**, not by what a workflow hoped for;
* **keyless-first** — a local inference server that is listening beats spending money to test
  durability, and it is what makes Track P's keyless work reachable from a durable run at all;
* **loud when empty** — a NAMED failure (``soak_no_provider``), never a silent ledger-only run.

Every case below is table-driven against a plain dict plus an injected reachability probe, so nothing
here opens a socket, starts a process or reads the real ``os.environ``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

# The chaos script is a driver, not package code, so it is loaded by PATH (the same way
# `test_soak_harness.py` does it). Only its PURE helpers are used here.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "durable_soak.py"


def _soak() -> Any:
    spec = importlib.util.spec_from_file_location("codna_durable_soak_provider", _SCRIPT)
    assert spec is not None and spec.loader is not None, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _soak()


def _probe_up(*listening: str):
    """A probe that answers True only for the given base URLs (a fake listener)."""

    def probe(base_url: str) -> bool:
        return base_url in listening

    return probe


def _probe_down(_base_url: str) -> bool:
    return False


def _discover(models: tuple[str, ...] = ("qwen3.6:latest",)):
    """A fake local-server model list.

    A keyless provider needs an explicit model id (measured: ``model 'default' not found``), so every
    local-provider case must inject one. Tests that care about the model pass their own list.
    """

    def discoverer(_base_url: str) -> list[str]:
        return list(models)

    return discoverer


def _discover_none(_base_url: str) -> list[str]:
    """A server that is listening but has no model installed — unusable for a real turn."""
    return []


# -- picks a hosted provider when its key is the only thing available -----------------------------


def test_picks_openai_when_only_its_key_is_set(script: Any) -> None:
    """THE case from the report: a CI runner with exactly one secret must get a working provider."""
    choice = script.select_provider({"OPENAI_API_KEY": "sk-x"}, probe=_probe_down)
    assert not isinstance(choice, script.SoakError), choice
    assert choice.provider == "openai"
    assert choice.key_env == "OPENAI_API_KEY"
    assert choice.base_url is None  # the provider's own endpoint
    assert choice.keyless is False
    assert choice.reason == "openai"


@pytest.mark.parametrize(
    "env, provider, key_env",
    [
        ({"OPENAI_API_KEY": "k"}, "openai", "OPENAI_API_KEY"),
        ({"ANTHROPIC_API_KEY": "k"}, "anthropic", "ANTHROPIC_API_KEY"),
        ({"GEMINI_API_KEY": "k"}, "gemini", "GEMINI_API_KEY"),
        ({"OPENROUTER_API_KEY": "k"}, "openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_every_supported_hosted_provider_can_be_selected(
    script: Any, env: Mapping[str, str], provider: str, key_env: str
) -> None:
    choice = script.select_provider(env, probe=_probe_down)
    assert choice.provider == provider
    assert choice.key_env == key_env


def test_a_hosted_provider_wins_in_a_stable_order_when_several_keys_are_set(script: Any) -> None:
    """Determinism matters: two runs of the same environment must pick the same model."""
    choice = script.select_provider(
        {"OPENROUTER_API_KEY": "r", "ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"},
        probe=_probe_down,
    )
    assert choice.provider == "openai"
    assert script.select_provider({"ANTHROPIC_API_KEY": "a"}, probe=_probe_down).provider == "anthropic"


def test_a_blank_key_does_not_count_as_a_provider(script: Any) -> None:
    """An empty secret expands to "" in GitHub Actions; that is "absent", not "configured"."""
    choice = script.select_provider(
        {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "   "}, probe=_probe_down
    )
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"


# -- prefers a keyless local provider that is actually listening ---------------------------------


def test_a_listening_ollama_beats_a_hosted_key(script: Any) -> None:
    """A keyless local server is preferred: testing durability should not cost money.

    The base URL carries ``/v1`` because that is the provider's OWN manifest default
    (`llms/src/providers/builtins.ts`) and the engine appends ``/chat/completions`` to it. Omitting
    the segment produced a measured ``404 page not found`` on the first turn.
    """
    choice = script.select_provider(
        {"OPENAI_API_KEY": "sk-x"},
        probe=_probe_up("http://127.0.0.1:11434/v1"),
        discoverer=_discover(),
    )
    assert choice.provider == "ollama"
    assert choice.key_env is None
    assert choice.keyless is True
    assert choice.base_url == "http://127.0.0.1:11434/v1"
    assert choice.reason == "ollama"


def test_a_listening_lmstudio_is_selected_when_ollama_is_not(script: Any) -> None:
    choice = script.select_provider(
        {}, probe=_probe_up("http://127.0.0.1:1234/v1"), discoverer=_discover()
    )
    assert choice.provider == "lmstudio"
    assert choice.keyless is True
    assert choice.base_url == "http://127.0.0.1:1234/v1"


def test_an_absent_local_server_does_not_shadow_a_hosted_key(script: Any) -> None:
    """The probe is what separates "ollama is installed" from "ollama is running"."""
    choice = script.select_provider({"OPENAI_API_KEY": "sk-x"}, probe=_probe_down)
    assert choice.provider == "openai"


def test_an_ollama_host_without_a_scheme_is_normalised(script: Any) -> None:
    """``OLLAMA_HOST`` is a host:port, not a URL; the contract requires a scheme."""
    choice = script.select_provider(
        {"OLLAMA_HOST": "127.0.0.1:11434"},
        probe=_probe_up("http://127.0.0.1:11434/v1"),
        discoverer=_discover(),
    )
    assert choice.base_url == "http://127.0.0.1:11434/v1"


@pytest.mark.parametrize(
    "host_env, expected",
    [
        # A bare host gets the scheme AND the version segment.
        ({"OLLAMA_HOST": "127.0.0.1:11434"}, "http://127.0.0.1:11434/v1"),
        ({"OLLAMA_HOST": "http://127.0.0.1:11434"}, "http://127.0.0.1:11434/v1"),
        ({"OLLAMA_HOST": "http://127.0.0.1:11434/"}, "http://127.0.0.1:11434/v1"),
        # A path the operator chose is NEVER rewritten: a gateway may carry its own prefix.
        ({"OLLAMA_HOST": "http://127.0.0.1:11434/openai/v1"}, "http://127.0.0.1:11434/openai/v1"),
        ({"OLLAMA_HOST": "http://127.0.0.1:11434/api"}, "http://127.0.0.1:11434/api"),
    ],
)
def test_the_local_base_url_gets_the_version_segment_the_engine_expects(
    script: Any, host_env: Mapping[str, str], expected: str
) -> None:
    """Regression for a MEASURED 404: a local base URL without ``/v1`` made every turn fail.

    Measured on this repository's own soak: with ``http://127.0.0.1:11434`` the engine's first turn
    emitted ``{"error":{"name":"Error","message":"404 page not found"}}``, so the run reached a
    ``turn.started`` and no checkpoint. The fix is to use each provider's manifest default (which
    carries the version) rather than a hand-rolled URL.
    """
    choice = script.select_provider(host_env, probe=_probe_up(expected), discoverer=_discover())
    assert not isinstance(choice, script.SoakError), choice
    assert choice.base_url == expected


def test_a_remote_cleartext_local_endpoint_is_skipped_for_one_the_contract_accepts(script: Any) -> None:
    """The soak must not select a URL the contract refuses (a key in the clear), only to fail at start."""
    choice = script.select_provider(
        {"OLLAMA_HOST": "http://10.0.0.5:11434"},
        probe=_probe_up("http://10.0.0.5:11434/v1", "http://127.0.0.1:1234/v1"),
        discoverer=_discover(),
    )
    # Ollama's remote-cleartext URL is refused by the contract, so LM Studio's loopback default wins.
    assert choice.provider == "lmstudio"
    assert choice.base_url == "http://127.0.0.1:1234/v1"


def test_a_remote_https_local_endpoint_is_accepted(script: Any) -> None:
    """A remote endpoint is fine over https — that is the contract's rule, not a blanket loopback ban."""
    choice = script.select_provider(
        {"OLLAMA_HOST": "https://ollama.internal:11434"},
        probe=_probe_up("https://ollama.internal:11434/v1"),
        discoverer=_discover(),
    )
    assert choice.provider == "ollama"
    assert choice.base_url == "https://ollama.internal:11434/v1"


def test_a_local_provider_takes_the_first_model_the_server_reports(script: Any) -> None:
    """A keyless provider has no bundled default: measured, no model id fails ``model 'default' not found``.

    "First" rather than "best" on purpose — the soak tests durability, not model quality, and a
    ranking heuristic would make the choice vary between server versions.
    """
    choice = script.select_provider(
        {},
        probe=_probe_up("http://127.0.0.1:11434/v1"),
        discoverer=_discover(("qwen3.6:latest", "llama3.1:8b")),
    )
    assert choice.provider == "ollama"
    assert choice.model == "qwen3.6:latest"


def test_an_explicit_model_override_beats_the_discovered_one(script: Any) -> None:
    choice = script.select_provider(
        {"CODNA_SOAK_MODEL": "qwen3-coder:latest"},
        probe=_probe_up("http://127.0.0.1:11434/v1"),
        discoverer=_discover(("qwen3.6:latest",)),
    )
    assert choice.model == "qwen3-coder:latest"


def test_a_listening_local_server_with_no_model_is_not_selected(script: Any) -> None:
    """Listening but empty: a turn would fail, so this is not a usable provider."""
    choice = script.select_provider(
        {},
        probe=_probe_up("http://127.0.0.1:11434/v1"),
        discoverer=_discover_none,
    )
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"


def test_a_local_server_whose_model_list_raises_is_not_selected(script: Any) -> None:
    """An exception from the model list is "unusable", not a crash in the soak's setup."""

    def boom(_base_url: str) -> list[str]:
        raise OSError("connection refused")

    choice = script.select_provider(
        {}, probe=_probe_up("http://127.0.0.1:11434/v1"), discoverer=boom
    )
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"


def test_an_explicit_local_endpoint_wins_over_everything(script: Any) -> None:
    """An operator naming a gateway means it: it is not overridden by a well-known local default."""
    choice = script.select_provider(
        {
            "CODNA_SOAK_LOCAL_BASE_URL": "http://127.0.0.1:4000",
            "CODNA_SOAK_LOCAL_PROVIDER": "litellm",
            "OPENAI_API_KEY": "sk-x",
        },
        probe=_probe_up("http://127.0.0.1:11434"),
        discoverer=_discover(),
    )
    assert choice.provider == "litellm"
    assert choice.base_url == "http://127.0.0.1:4000"
    assert choice.reason == "explicit_local"
    # LiteLLM may or may not need a key; the one that is SET is forwarded so either deployment works.
    assert choice.key_env == "OPENAI_API_KEY"


def test_an_explicit_local_endpoint_without_a_provider_is_a_named_failure(script: Any) -> None:
    choice = script.select_provider(
        {"CODNA_SOAK_LOCAL_BASE_URL": "http://127.0.0.1:4000"}, probe=_probe_down
    )
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"
    assert "CODNA_SOAK_LOCAL_PROVIDER" in str(choice)


def test_an_explicit_hosted_provider_without_its_key_is_a_named_failure(script: Any) -> None:
    choice = script.select_provider(
        {"CODNA_SOAK_LOCAL_BASE_URL": "https://api.example.com/v1", "CODNA_SOAK_LOCAL_PROVIDER": "openai"},
        probe=_probe_down,
    )
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"
    assert choice.details["provider"] == "openai"
    assert "OPENAI_API_KEY" in choice.details["looked_for"]


# -- nothing available: a NAMED failure, never a silent ledger-only run --------------------------


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"CODNA_SOAK_MODEL": "gpt-4.1"},  # a model with no provider
        {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "  "},  # empty secrets
        {"PATH": "/usr/bin", "HOME": "/root"},  # an ordinary environment
    ],
)
def test_nothing_available_is_a_named_failure(script: Any, env: Mapping[str, str]) -> None:
    choice = script.select_provider(env, probe=_probe_down)
    assert isinstance(choice, script.SoakError)
    assert choice.code == "soak_no_provider"
    # The message must say what was looked for, so an operator can fix it without reading the source.
    assert choice.details["looked_for"]
    assert "OPENAI_API_KEY" in choice.details["looked_for"]
    assert choice.hint


def test_the_failure_names_every_route_that_was_checked(script: Any) -> None:
    choice = script.select_provider({}, probe=_probe_down)
    looked_for = set(choice.details["looked_for"])
    assert {"OLLAMA_HOST", "LMSTUDIO_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"} <= looked_for


# -- the choice carries NO key value -------------------------------------------------------------


def test_the_choice_reports_the_env_var_name_not_the_key(script: Any) -> None:
    """A report is an artifact people upload; it must never contain the credential."""
    choice = script.select_provider({"OPENAI_API_KEY": "sk-super-secret-value"}, probe=_probe_down)
    payload = choice.to_payload()
    assert payload["key_env"] == "OPENAI_API_KEY"
    assert "sk-super-secret-value" not in str(payload)


# -- the settings path: flag, selection, validation ------------------------------------------------


def _args(**overrides: Any) -> Any:
    """The argparse namespace `settings_from_args` reads, with sane defaults."""
    values: dict[str, Any] = {
        "kills": 1,
        "seed": 7,
        "duration": 60.0,
        "spread": 30.0,
        "engine": "plain",
        "run_id": "soak-test",
        "workdir": None,
        "python": "",
        "brief": "b",
        "verify": "true",
        "task_id": "task-1",
        "approve_as": "a@b.c",
        "no_sidecar": True,
        "keep_workdir": False,
        "report_out": None,
        "provider": None,
        "model": None,
        "base_url": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_settings_select_the_provider_from_the_environment(script: Any) -> None:
    settings = script.settings_from_args(
        _args(), env={"OPENAI_API_KEY": "k"}, probe=_probe_down
    )
    assert settings.provider == "openai"
    assert settings.provider_key_env == "OPENAI_API_KEY"
    assert settings.model is None


def test_settings_honour_an_explicit_provider_flag(script: Any) -> None:
    settings = script.settings_from_args(
        _args(provider="anthropic", model="claude-sonnet-4"),
        env={"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "o"},
        probe=_probe_up("http://127.0.0.1:11434"),  # a listening ollama must NOT win over the flag
    )
    assert settings.provider == "anthropic"
    assert settings.model == "claude-sonnet-4"
    assert settings.provider_key_env == "ANTHROPIC_API_KEY"


def test_settings_accept_an_explicit_keyless_provider_with_no_credential(script: Any) -> None:
    settings = script.settings_from_args(
        _args(provider="ollama", base_url="http://127.0.0.1:11434"),
        env={},
        probe=_probe_down,
    )
    assert settings.provider == "ollama"
    assert settings.provider_key_env is None
    assert settings.base_url == "http://127.0.0.1:11434"


def test_settings_refuse_an_explicit_hosted_provider_with_no_key(script: Any) -> None:
    with pytest.raises(script.SoakError) as caught:
        script.settings_from_args(_args(provider="openai"), env={}, probe=_probe_down)
    assert caught.value.code == "soak_no_provider"


def test_settings_fail_loudly_when_no_provider_is_available(script: Any) -> None:
    """The named error, not a settings object that would run a ledger-only soak."""
    with pytest.raises(script.SoakError) as caught:
        script.settings_from_args(_args(), env={}, probe=_probe_down)
    assert caught.value.code == "soak_no_provider"


# -- the exit code is distinct --------------------------------------------------------------------


def test_no_provider_has_its_own_exit_code(
    script: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare "setup failed" would hide that durability was NEVER tested.

    Both inputs to the decision are neutralised so the outcome does not depend on the machine this
    runs on: every provider var is cleared, and the loopback probe is replaced. Without that, a
    developer with Ollama running would get a *successful* selection — which is the intended
    behaviour, but makes this an environment test rather than a contract test.
    """
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_HOST",
        "LMSTUDIO_BASE_URL",
        "CODNA_SOAK_LOCAL_BASE_URL",
        "CODNA_SOAK_LOCAL_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(script, "_default_probe", lambda _url: False)

    code = script.main(["--no-sidecar"])
    assert code == script.EXIT_NO_PROVIDER
    assert script.EXIT_NO_PROVIDER != script.EXIT_SETUP_FAILED
    captured = capsys.readouterr()
    assert "soak_no_provider" in captured.err


def test_the_no_provider_code_is_documented_in_the_module(script: Any) -> None:
    """The exit code a CI reader sees must be explained where the others are."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "EXIT_NO_PROVIDER" in source
    assert "soak_no_provider" in source


# -- the run argv carries the provider, and NO key ------------------------------------------------


def _soak_instance(script: Any, **overrides: Any) -> Any:
    """A `Soak` built without touching the network: the constructor only resolves paths and ports."""
    settings = script.SoakSettings(
        kills=1,
        seed=1,
        duration_s=60.0,
        spread_s=30.0,
        provider="ollama",
        model="qwen3.6:latest",
        base_url="http://127.0.0.1:11434",
        provider_key_env=None,
        ensure_sidecar=False,
        **overrides,
    )
    return script.Soak(settings)


def test_the_run_argv_forwards_every_provider_coordinate(script: Any, tmp_path: Path) -> None:
    soak = _soak_instance(script, workdir=tmp_path / "work")
    argv = soak._run_argv()
    assert argv[0] == "run"
    assert argv[argv.index("--provider") + 1] == "ollama"
    assert argv[argv.index("--model") + 1] == "qwen3.6:latest"
    assert argv[argv.index("--base-url") + 1] == "http://127.0.0.1:11434"
    # The worker is told to read the credential from its own environment; a key VALUE is never here.
    assert argv[argv.index("--provider-key-source") + 1] == "environment"


def test_the_run_argv_omits_a_model_and_base_url_that_are_unset(script: Any, tmp_path: Path) -> None:
    """An empty flag would be refused by the contract; absent is "the provider's default"."""
    settings = script.SoakSettings(
        kills=1,
        seed=1,
        duration_s=60.0,
        spread_s=30.0,
        provider="openai",
        provider_key_env="OPENAI_API_KEY",
        ensure_sidecar=False,
    )
    soak = script.Soak(settings) if settings.workdir else None
    if soak is None:  # pragma: no cover - the constructor picks a temp dir when workdir is None
        return
    argv = soak._run_argv()
    assert "--model" not in argv
    assert "--base-url" not in argv
    assert argv[argv.index("--provider") + 1] == "openai"


def test_the_soak_env_inherits_the_provider_key_into_the_child_environment(
    script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker reads the key from its own process env — the durable path never carries it.

    Inheritance, not copying: the soak passes its whole environment down, so the variable the
    selection proved present is the one the child sees. This asserts the child really gets it, which
    is the half that "the provider was selected" does not cover.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-inherited")
    settings = script.SoakSettings(
        kills=1,
        seed=1,
        duration_s=60.0,
        spread_s=30.0,
        provider="openai",
        provider_key_env="OPENAI_API_KEY",
        ensure_sidecar=False,
    )
    env = script.build_soak_env(
        settings, workdir=tmp_path / "work", port_base=18800, temporal_port=18899
    )
    assert env["OPENAI_API_KEY"] == "sk-inherited"
    assert "CODNA_RUNTIME_ROOT" in env
    assert env["CODNA_DURABLE_ENGINE"] == settings.engine
    # ...and the selection's own env carries no provider variable of its own invention: the TaskSpec
    # names the provider, so a `CODNA_DURABLE_PROVIDER` would be a setting nothing reads.
    assert "CODNA_DURABLE_PROVIDER" not in env


def test_the_soak_env_refuses_a_key_that_was_selected_but_is_absent(
    script: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name without a value fails at START, not on the first turn minutes into the run."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = script.SoakSettings(
        kills=1,
        seed=1,
        duration_s=60.0,
        spread_s=30.0,
        provider="openai",
        provider_key_env="OPENAI_API_KEY",
        ensure_sidecar=False,
    )
    with pytest.raises(script.SoakError) as caught:
        script.build_soak_env(
            settings, workdir=tmp_path / "work", port_base=18810, temporal_port=18898
        )
    assert caught.value.code == "soak_provider_key_missing"


# -- the checkpoint is what "durability was exercised" means -------------------------------------


@pytest.mark.parametrize(
    "records, expected",
    [
        ([], False),
        ([{"kind": "turn.started"}, {"kind": "turn.finished"}], False),
        ([{"kind": "turn.started"}, {"kind": "checkpoint"}], True),
        ([{"kind": "checkpoint"}, {"kind": "verify"}], True),
    ],
)
def test_reached_checkpoint_reads_the_ledger(script: Any, records: list[dict[str, Any]], expected: bool) -> None:
    """A turn that died on credentials writes turn records and NO checkpoint: the distinction that
    makes "the invariants held" honest rather than vacuous."""
    assert script._reached_checkpoint(records) is expected


def test_the_report_shape_names_the_provider_and_the_checkpoint(script: Any) -> None:
    result = script.SoakResult(
        run_id="r",
        seed=1,
        schedule=[],
        kills_performed=0,
        worker_starts=1,
        approvals_sent=0,
        run_outcome="completed",
        run_reached_terminal=True,
        ledger_path="/tmp/ledger",
        report={"ok": True},
        duration_s=1.0,
        provider={"provider": "ollama", "key_env": None, "keyless": True},
        reached_checkpoint=True,
    )
    payload = result.to_payload()
    assert payload["provider"]["provider"] == "ollama"
    assert payload["reached_checkpoint"] is True


def test_the_provider_env_map_agrees_with_the_contracts(script: Any) -> None:
    """The soak keeps its own copy so `--print-schedule` works before the package imports.

    Two copies are only safe if something compares them, which is this test: a drift would be
    "the soak forwarded a provider the contract has no env var for".
    """
    from codna.durable.contracts import KEYLESS_PROVIDERS, PROVIDER_ENV_KEYS

    assert set(script._KEYLESS) == set(KEYLESS_PROVIDERS)
    for provider, names in PROVIDER_ENV_KEYS.items():
        assert provider in script._KEY_ENV_BY_PROVIDER, provider
        assert script._KEY_ENV_BY_PROVIDER[provider] == names, provider


def test_the_default_probe_treats_an_http_error_as_a_listener(script: Any) -> None:
    """A 404 from Ollama still proves something is listening; only a refusal is "absent"."""
    # A closed port on loopback: nothing is listening, so the probe must say so rather than raise.
    assert script._default_probe("http://127.0.0.1:1") is False
