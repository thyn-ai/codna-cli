"""Shared runtime configuration contract tests."""
from __future__ import annotations

import pytest

from codna.runtime_config import (
    RuntimeConfigError,
    local_runtime_urls,
    resolve_api_key,
    resolve_engine_selection,
    resolve_local_urls_from_env,
    resolve_port_base,
)


def test_default_local_runtime_urls_are_fixed_loopback():
    assert resolve_port_base(None) == 18600
    assert local_runtime_urls() == ("http://127.0.0.1:18600", "http://127.0.0.1:18601")
    assert resolve_local_urls_from_env({}) == ("http://127.0.0.1:18600", "http://127.0.0.1:18601")


def test_port_base_override_validates_pair():
    assert resolve_port_base("20000") == 20000
    assert resolve_local_urls_from_env({"CODNA_PORT_BASE": "20000"}) == (
        "http://127.0.0.1:20000",
        "http://127.0.0.1:20001",
    )
    with pytest.raises(RuntimeConfigError, match="integer"):
        resolve_port_base("bad")
    with pytest.raises(RuntimeConfigError, match="between"):
        resolve_port_base("65535")


def test_api_key_resolution_uses_codna_then_legacy_keys():
    assert resolve_api_key({"CODNA_API_KEY": "codna", "ALGENTA_API_KEY": "legacy"}) == ("codna", "CODNA_API_KEY")
    assert resolve_api_key({"DE_API_KEY": "fallback"}) == ("fallback", "DE_API_KEY")
    assert resolve_api_key({}) == (None, None)


def test_explicit_engine_env_takes_precedence_over_local_stack_state():
    selection = resolve_engine_selection(
        env={"CODNA_ENGINE_URL": "https://api.codna.ai/", "CODNA_API_KEY": "codna_test_key"},
        state_reader=lambda: {"engine_url": "http://127.0.0.1:9999"},
    )

    assert selection.engine_url == "https://api.codna.ai"
    assert selection.api_key == "codna_test_key"
    assert selection.engine_url_source == "CODNA_ENGINE_URL"
    assert selection.api_key_source == "CODNA_API_KEY"


def test_local_stack_state_precedes_local_default():
    selection = resolve_engine_selection(
        env={"ALGENTA_API_KEY": "legacy_key"},
        state_reader=lambda: {"engine_url": "http://127.0.0.1:19000/"},
    )

    assert selection.engine_url == "http://127.0.0.1:19000"
    assert selection.api_key == "legacy_key"
    assert selection.engine_url_source == "local_stack_state"
    assert selection.api_key_source == "ALGENTA_API_KEY"


def test_local_default_uses_port_base_env_when_no_url_or_state():
    selection = resolve_engine_selection(
        env={"CODNA_PORT_BASE": "20000", "DE_API_KEY": "fallback_key"},
        state_reader=lambda: None,
    )

    assert selection.engine_url == "http://127.0.0.1:20000"
    assert selection.api_key == "fallback_key"
    assert selection.engine_url_source == "local_default"
    assert selection.api_key_source == "DE_API_KEY"
