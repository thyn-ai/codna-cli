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
