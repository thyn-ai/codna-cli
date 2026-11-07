"""The provider ENDPOINT override's journey from the user's config to the sidecar request (P1).

A base URL decides where the provider credential is sent, so it may only come from the user's own config
(``~/.codna/config.yaml``, applied to ``CODNA_AGENT_BASE_URL`` by ``config_file``) or the environment — never
from the run request, which is assembled from repository-adjacent data. These tests pin both halves: the
runner forwards what the environment carries, and it invents nothing when the environment carries nothing.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from codna import config_file
from codna import packaged_agent_runner as par
from codna.packaged_repository_advanced import PackagedAgentRunRequest


@pytest.fixture(autouse=True)
def _restore_environ():
    """`apply_config` deliberately mutates the process env; restore it so a test that loads a config file
    cannot leak CODNA_AGENT_PROVIDER / CODNA_AGENT_BASE_URL into another file's assertions."""
    snap = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snap)


def _request(**kw) -> PackagedAgentRunRequest:
    base = dict(
        repository_id="r",
        snapshot_id="s",
        repo_root=Path("/no/such/repo"),
        issue_text="Fix the failing test.",
        model="m",
        signals={},
        evidence_bundle={},
        snapshot={"snapshot_id": "s", "snapshot_file_count": 0, "raw_repo_token_estimate": 100},
        task_kind="fix",
    )
    base.update(kw)
    return PackagedAgentRunRequest(**base)


def test_the_runner_forwards_the_configured_base_url(monkeypatch):
    monkeypatch.setenv(config_file.BASE_URL_ENV, "https://llm.example.com/v1")
    payload = par._sidecar_payload(_request(), Path("/tmp"))
    # The engine validates and normalizes it again, but it must ARRIVE — otherwise a configured endpoint is a
    # setting that silently does nothing for CLI runs.
    assert payload["base_url"] == "https://llm.example.com/v1"


def test_the_runner_invents_no_base_url_when_none_is_configured(monkeypatch):
    monkeypatch.delenv(config_file.BASE_URL_ENV, raising=False)
    payload = par._sidecar_payload(_request(), Path("/tmp"))
    # Absent means "the provider's own default". A placeholder would be a destination nobody chose.
    assert "base_url" not in payload


def test_a_blank_environment_value_is_not_forwarded(monkeypatch):
    monkeypatch.setenv(config_file.BASE_URL_ENV, "   ")
    payload = par._sidecar_payload(_request(), Path("/tmp"))
    assert "base_url" not in payload


def test_config_file_and_runner_agree_on_the_variable_name(tmp_path, monkeypatch):
    """Written by one module, read by the other: an agreement asserted instead of assumed."""
    monkeypatch.delenv(config_file.BASE_URL_ENV, raising=False)
    path = tmp_path / "codna.yaml"
    path.write_text("model:\n  provider: ollama\n  base_url: http://127.0.0.1:11434/v1\n", encoding="utf-8")
    config_file.apply_config(str(path))
    assert par._sidecar_payload(_request(), Path("/tmp"))["base_url"] == "http://127.0.0.1:11434/v1"


def test_the_provider_comes_from_the_same_place(monkeypatch):
    """A keyless provider must reach the sidecar as itself, not default to anthropic."""
    monkeypatch.setenv("CODNA_AGENT_PROVIDER", "ollama")
    payload = par._sidecar_payload(_request(), Path("/tmp"))
    assert payload["provider"] == "ollama"


def test_an_explicit_coordinate_beats_the_environment(monkeypatch):
    """A caller that names a provider has already decided (the durable run's TaskSpec).

    The environment is the fallback for the CLI's own commands, which have no spec to inherit from. If
    the environment won, a durable run started for Ollama could be redirected to whatever a developer's
    shell happened to export — and would bill the wrong provider for every review.
    """
    monkeypatch.setenv("ALGENTA_AGENT_PROVIDER", "anthropic")
    monkeypatch.setenv("CODNA_AGENT_MODEL", "claude-sonnet-4")
    monkeypatch.setenv(config_file.BASE_URL_ENV, "https://llm.example.com/v1")
    payload = par._sidecar_payload(
        _request(provider="ollama", model_id="qwen3-coder:latest", base_url="http://127.0.0.1:11434/v1"),
        Path("/tmp"),
    )
    assert payload["provider"] == "ollama"
    assert payload["model"] == "qwen3-coder:latest"
    assert payload["base_url"] == "http://127.0.0.1:11434/v1"


def test_an_absent_coordinate_leaves_the_environment_in_charge(monkeypatch):
    """`codna review` (no spec) keeps reading the developer's own configuration."""
    monkeypatch.setenv("ALGENTA_AGENT_PROVIDER", "ollama")
    payload = par._sidecar_payload(_request(), Path("/tmp"))
    assert payload["provider"] == "ollama"
