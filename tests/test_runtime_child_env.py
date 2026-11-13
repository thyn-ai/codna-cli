"""Tests for the child-environment rules the local runtime spawns with.

These live beside the extracted module (``codna/runtime/child_env.py``) because the properties are the reason
it exists: WHICH keys cross the process boundary, and WHICH value wins when two sources disagree. Both were
previously asserted only indirectly through ``_sidecar_spawn``/``_engine_spawn``, which conflates "the
allow-list is right" with "the spawn wiring is right".
"""
from __future__ import annotations

from codna.runtime import child_env
from codna.runtime.config import ConfigValue


def test_child_env_from_keys_excludes_the_runtime_own_url_coordinates() -> None:
    """A keys-file entry must never retarget the child at a different engine.

    `ENGINE_URL_KEYS` are the runtime's OWN coordinates. If a `keys.txt` line could set one, a child process
    would be pointed at an engine the parent did not choose — the parent would then wait on a health check
    the child is not serving.
    """
    keys = {
        "CODNA_ENGINE_URL": ConfigValue("CODNA_ENGINE_URL", "http://127.0.0.1:9999", "keys_txt"),
        "ALGENTA_ENGINE_URL": ConfigValue("ALGENTA_ENGINE_URL", "http://127.0.0.1:9998", "keys_txt"),
        "ALGENTA_BASE_URL": ConfigValue("ALGENTA_BASE_URL", "http://127.0.0.1:9997", "keys_txt"),
        "OPENAI_API_KEY": ConfigValue("OPENAI_API_KEY", "from-keys", "keys_txt"),
    }

    assert child_env.child_env_from_keys(keys) == {"OPENAI_API_KEY": "from-keys"}


def test_child_env_from_keys_excludes_anything_outside_the_allow_list() -> None:
    """An arbitrary `keys.txt` line must not leak into a child process."""
    keys = {
        "OPENAI_API_KEY": ConfigValue("OPENAI_API_KEY", "allowed", "keys_txt"),
        "AWS_SECRET_ACCESS_KEY": ConfigValue("AWS_SECRET_ACCESS_KEY", "not-allowed", "keys_txt"),
        "DATABASE_URL": ConfigValue("DATABASE_URL", "postgres://nope", "keys_txt"),
    }

    assert child_env.child_env_from_keys(keys) == {"OPENAI_API_KEY": "allowed"}


def test_child_env_from_keys_drops_blank_values_and_handles_no_keys() -> None:
    keys = {"OPENAI_API_KEY": ConfigValue("OPENAI_API_KEY", "   ", "keys_txt")}

    assert child_env.child_env_from_keys(keys) == {}
    assert child_env.child_env_from_keys(None) == {}
    assert child_env.child_env_from_keys({}) == {}


def test_merged_child_env_keeps_a_shell_provider_key_over_the_keys_file(monkeypatch) -> None:
    """An operator's explicit export outranks a stored default for the SAME provider key."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-from-env")
    keys = {"OPENAI_API_KEY": ConfigValue("OPENAI_API_KEY", "openai-from-keys", "keys_txt")}

    merged = child_env.merged_child_env_from_keys(keys)

    assert merged["OPENAI_API_KEY"] == "openai-from-env"


def test_merged_child_env_adds_a_keys_file_value_the_shell_does_not_set(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    keys = {"ANTHROPIC_API_KEY": ConfigValue("ANTHROPIC_API_KEY", "from-keys", "keys_txt")}

    merged = child_env.merged_child_env_from_keys(keys)

    assert merged["ANTHROPIC_API_KEY"] == "from-keys"


def test_merged_child_env_inherits_the_parent_environment(monkeypatch) -> None:
    monkeypatch.setenv("CODNA_TEST_MARKER", "inherited")

    merged = child_env.merged_child_env_from_keys(None)

    assert merged["CODNA_TEST_MARKER"] == "inherited"


def test_every_allow_listed_api_key_is_also_a_provider_key() -> None:
    """The two sets must stay consistent: a provider key is by definition an allow-listed `_API_KEY`."""
    assert child_env.PROVIDER_CHILD_ENV_KEYS == frozenset(
        key for key in child_env.CHILD_ENV_KEY_ALLOWLIST if key.endswith("_API_KEY")
    )
    assert child_env.PROVIDER_CHILD_ENV_KEYS <= child_env.CHILD_ENV_KEY_ALLOWLIST
    # `TELYS_KERNEL` is allow-listed but is NOT a provider credential, so it must not be in the second set.
    assert "TELYS_KERNEL" in child_env.CHILD_ENV_KEY_ALLOWLIST
    assert "TELYS_KERNEL" not in child_env.PROVIDER_CHILD_ENV_KEYS


def test_remove_local_database_env_strips_both_coordinate_names(monkeypatch) -> None:
    env = {"DATABASE_URL": "postgres://nope", "ASYNC_DATABASE_URL": "postgres://also-nope", "KEEP": "1"}

    cleaned = child_env.remove_local_database_env(env)

    assert cleaned == {"KEEP": "1"}
