"""The environment a spawned local-runtime child inherits.

Split out of :mod:`codna.runtime.local_stack` to keep that module under the repository's file-size
guardrail (``cli/tests/test_modularity.py``). The concern here is narrow and independently testable: WHICH
keys cross the process boundary, and which values may not be overridden.

The allow-list is an ALLOW-list rather than a deny-list on purpose. A child process is handed the whole
parent environment minus everything not named here, so a variable added to the shell for some unrelated tool
cannot leak into the engine or the sidecar. ``ENGINE_URL_KEYS`` are blocked even when they appear in
``keys.txt`` because those are the runtime's OWN coordinates — a keys-file entry would otherwise retarget
the child at a different engine.
"""
from __future__ import annotations

import os
from typing import Mapping

from .config import ENGINE_URL_KEYS, LOCAL_DATABASE_ENV_KEYS, ConfigValue

CHILD_ENV_KEY_ALLOWLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "CODNA_API_KEY",
        "CURSOR_API_KEY",
        "DE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TELYS_KERNEL",
        "XAI_API_KEY",
    }
)
PROVIDER_CHILD_ENV_KEYS = frozenset(
    key for key in CHILD_ENV_KEY_ALLOWLIST if key.endswith("_API_KEY")
)


def child_env_from_keys(
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, str]:
    """The allow-listed values from ``keys``, minus the runtime's own URL coordinates."""
    if not keys:
        return {}
    blocked = set(ENGINE_URL_KEYS)
    output: dict[str, str] = {}
    for key, config_value in keys.items():
        if key in blocked or key not in CHILD_ENV_KEY_ALLOWLIST:
            continue
        value = config_value.value.strip()
        if value:
            output[key] = value
    return output


def merged_child_env_from_keys(
    keys: Mapping[str, ConfigValue] | None,
) -> dict[str, str]:
    """The parent environment with the allow-listed ``keys`` values merged in.

    A key that is ALREADY set in the shell WINS over the keys file: the operator's explicit export is a
    stronger intent than a stored default, and silently swapping the credential a user is watching in their
    terminal is the kind of surprise this ordering exists to avoid.
    """
    env = dict(os.environ)
    for key, value in child_env_from_keys(keys).items():
        if key in PROVIDER_CHILD_ENV_KEYS and os.environ.get(key):
            continue
        env[key] = value
    return env


def remove_local_database_env(env: dict[str, str]) -> dict[str, str]:
    """Drop the database coordinates: the local runtime refuses a DB-backed path by design."""
    for key in LOCAL_DATABASE_ENV_KEYS:
        env.pop(key, None)
    return env
