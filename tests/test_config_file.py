from __future__ import annotations

import os

import pytest

from codna import config_file
from codna.config_file import ConfigError


@pytest.fixture(autouse=True)
def _restore_environ():
    """apply_config intentionally mutates the process env; snapshot+restore so tests never leak
    (e.g. CODNA_REQUIRE_EGRESS_DENY / provider keys) into later tests."""
    snap = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snap)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_discover_prefers_codna_yaml(tmp_path, monkeypatch):
    (tmp_path / "codna.yaml").write_text("model: {}\n", encoding="utf-8")
    (tmp_path / ".codna.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert os.path.basename(config_file.discover_config_path()) == "codna.yaml"


def test_discover_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config_file.discover_config_path() is None


def test_explicit_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError):
        config_file.discover_config_path(str(tmp_path / "nope.yaml"))


def test_model_env_ref_sets_provider_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODNA_AGENT_PROVIDER", raising=False)
    monkeypatch.setenv("CODNA_MODEL_KEY", "sk-openai-xyz")
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: openai\n  key: env:CODNA_MODEL_KEY\n")
    config_file.apply_config(str(p))
    assert os.environ["OPENAI_API_KEY"] == "sk-openai-xyz"
    assert os.environ["CODNA_AGENT_PROVIDER"] == "openai"


def test_unresolved_env_ref_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("CODNA_MODEL_KEY", raising=False)
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: openai\n  key: env:CODNA_MODEL_KEY\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "CODNA_MODEL_KEY" in str(exc.value)


def test_privacy_only_config_skips_unresolved_model_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CODNA_MODEL_KEY", raising=False)
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)
    p = _write(
        tmp_path,
        "codna.yaml",
        "model:\n  provider: openai\n  key: env:CODNA_MODEL_KEY\nprivacy:\n  egress: fail-closed\n",
    )

    config_file.apply_config(str(p), include_model=False)

    assert "OPENAI_API_KEY" not in os.environ
    assert "CODNA_AGENT_PROVIDER" not in os.environ
    assert os.environ["CODNA_REQUIRE_EGRESS_DENY"] == "1"


def test_unknown_provider_fails_closed(tmp_path):
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: nope\n  key: x\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "unknown" in str(exc.value)


def test_privacy_egress_sets_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("CODNA_REQUIRE_EGRESS_DENY", raising=False)
    p = _write(tmp_path, "codna.yaml", "privacy:\n  egress: fail-closed\n")
    config_file.apply_config(str(p))
    assert os.environ["CODNA_REQUIRE_EGRESS_DENY"] == "1"


def test_redact_false_warns_not_faked(tmp_path, capsys):
    p = _write(tmp_path, "codna.yaml", "privacy:\n  redact_secrets: false\n")
    config_file.apply_config(str(p))
    assert "redact_secrets=false is ignored" in capsys.readouterr().err


def test_json_body_parses_without_pyyaml(tmp_path, monkeypatch):
    # Simulate PyYAML absent: apply_config must still parse a JSON-content .yaml.
    import builtins

    real_import = builtins.__import__

    def no_yaml(name, *a, **k):
        if name == "yaml":
            raise ModuleNotFoundError("no yaml")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    monkeypatch.setenv("CODNA_MODEL_KEY", "sk-json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = _write(tmp_path, "codna.yaml", '{"model": {"provider": "openai", "key": "env:CODNA_MODEL_KEY"}}')
    config_file.apply_config(str(p))
    assert os.environ["OPENAI_API_KEY"] == "sk-json"


def test_malformed_yaml_raises_configerror_not_crash(tmp_path):
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: openai\n bad: : :\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "cannot parse" in str(exc.value)


# --- provider base URL (P1) -------------------------------------------------------------------------
#
# `model.base_url` decides WHERE the provider credential is sent, so it is validated at load time and it may
# only come from this file or the environment. Each rejection below is a distinct way an endpoint override
# could smuggle something past the consumer (a non-http scheme, embedded credentials, a dropped query).


def test_base_url_is_exported_for_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv(config_file.BASE_URL_ENV, raising=False)
    p = _write(
        tmp_path,
        "codna.yaml",
        "model:\n  provider: ollama\n  base_url: http://127.0.0.1:11434/v1/\n",
    )
    config_file.apply_config(str(p))
    # Trailing slash removed so the consumer's `${base}/v1` cannot become `//v1`.
    assert os.environ[config_file.BASE_URL_ENV] == "http://127.0.0.1:11434/v1"
    assert os.environ["CODNA_AGENT_PROVIDER"] == "ollama"


def test_base_url_accepts_an_env_ref_and_fails_closed_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("CODNA_PROVIDER_URL", "https://llm.example.com/v1")
    p = _write(
        tmp_path,
        "codna.yaml",
        "model:\n  provider: anthropic\n  base_url: env:CODNA_PROVIDER_URL\n",
    )
    config_file.apply_config(str(p))
    assert os.environ[config_file.BASE_URL_ENV] == "https://llm.example.com/v1"

    monkeypatch.delenv("CODNA_PROVIDER_URL")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "model.base_url: env:CODNA_PROVIDER_URL is not set" in str(exc.value)


@pytest.mark.parametrize(
    "bad, expected",
    [
        ("ftp://host/v1", "must be an http(s) URL"),
        ("file:///etc/passwd", "must be an http(s) URL"),
        ("https://user:pass@host/v1", "must not embed credentials"),
        ("https://:@host/v1", "must not embed credentials"),
        ("https://host/v1?key=abc", "must not carry a query string"),
        ("https://host/v1#frag", "must not carry a query string"),
        ("host-without-scheme/v1", "must be an http(s) URL"),
        ("   ", "must not be blank"),
        ("https://ho st/v1", "must not contain whitespace"),
    ],
)
def test_base_url_validation_is_fail_closed(tmp_path, bad, expected):
    p = _write(tmp_path, "codna.yaml", f"model:\n  provider: openai\n  base_url: {bad!r}\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert expected in str(exc.value)


def test_base_url_rejects_remote_plain_http(tmp_path):
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: openai\n  base_url: http://example.com/v1\n")
    with pytest.raises(ConfigError, match="plain http must point at loopback"):
        config_file.apply_config(str(p))


@pytest.mark.parametrize("value", [0, False, []])
def test_base_url_rejects_non_string_values(value):
    with pytest.raises(ConfigError, match="must be a string URL"):
        config_file._validated_base_url(value, field="model.base_url")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.2:11434/v1",
        "http://[::1]:11434/v1",
        "https://llm.example.com/v1",
    ],
)
def test_base_url_allows_https_and_loopback_http(url):
    assert config_file._validated_base_url(url, field="model.base_url") == url


def test_absent_base_url_leaves_the_provider_default_alone(tmp_path, monkeypatch):
    monkeypatch.delenv(config_file.BASE_URL_ENV, raising=False)
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: openai\n")
    config_file.apply_config(str(p))
    # Nothing is invented: no endpoint override means "the provider's own default".
    assert config_file.BASE_URL_ENV not in os.environ


def test_home_config_is_only_searched_when_model_config_is_consumed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".codna").mkdir(parents=True)
    (home / ".codna" / "config.yaml").write_text("model:\n  provider: ollama\n", encoding="utf-8")
    work = tmp_path / "repo"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)

    # Deterministic commands keep the historical project-only search.
    assert config_file.discover_config_path(include_home=False) is None
    # Model-consuming commands find the user's own file, which is the only scope a base URL may come from.
    assert config_file.discover_config_path(include_home=True) == str(home / ".codna" / "config.yaml")


def test_a_project_file_still_wins_over_the_home_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".codna").mkdir(parents=True)
    (home / ".codna" / "config.yaml").write_text("model:\n  provider: ollama\n", encoding="utf-8")
    work = tmp_path / "repo"
    work.mkdir()
    (work / "codna.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    assert config_file.discover_config_path(include_home=True) == "codna.yaml"


def test_a_keyless_provider_config_needs_no_key(tmp_path, monkeypatch):
    for var in ("CODNA_AGENT_PROVIDER", config_file.BASE_URL_ENV):
        monkeypatch.delenv(var, raising=False)
    for provider in sorted(config_file.KEYLESS_PROVIDERS):
        p = _write(
            tmp_path,
            "codna.yaml",
            f"model:\n  provider: {provider}\n  base_url: http://127.0.0.1:11434/v1\n",
        )
        config_file.apply_config(str(p), include_model=True)
        assert os.environ["CODNA_AGENT_PROVIDER"] == provider
        assert os.environ[config_file.BASE_URL_ENV] == "http://127.0.0.1:11434/v1"


def test_a_key_on_a_keyless_provider_is_refused_not_silently_dropped(tmp_path, monkeypatch):
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: ollama\n  key: sk-unused\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "not applicable to the keyless provider" in str(exc.value)


def test_an_unknown_provider_is_still_refused(tmp_path):
    p = _write(tmp_path, "codna.yaml", "model:\n  provider: totally-made-up\n")
    with pytest.raises(ConfigError) as exc:
        config_file.apply_config(str(p))
    assert "unknown" in str(exc.value)
