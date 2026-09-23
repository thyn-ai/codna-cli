"""A run that failed for a missing provider key must say so — and how to fix it.

agent-core's provider adapter aborts the run when no BYOK key is reachable; the final frame
carries "no API key in env for '<provider>' (tried: <ENV_NAMES>)" as its error. The MCP tool
surface renders only the exception message (`codna_fix error: <message>`), so the message
itself must name the fix. Verified end-to-end in a scrubbed environment (fresh HOME, keychain
disabled, no keys): the run terminates `runtime_invalid` with exactly that engine error while
the MCP surface showed only the generic line.
"""

from __future__ import annotations

import json

import pytest

from codna.packaged_agent_runner import _parse_final_frame  # noqa: SLF001
from codna.packaged_repository_advanced import PackagedRepositoryAdvancedError

_MISSING_KEY_FRAME = json.dumps(
    {
        "type": "final",
        "status": "failed",
        "terminal_state": "runtime_invalid",
        "error": "Error: no API key in env for 'anthropic' (tried: ANTHROPIC_API_KEY)",
    }
)


def test_a_missing_provider_key_names_the_key_and_the_fix():
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(_MISSING_KEY_FRAME, sidecar_url="http://127.0.0.1:1", task_kind="fix")
    assert excinfo.value.code == "agent_core_run_failed"
    assert str(excinfo.value) == (
        "codna fix needs a provider key — set ANTHROPIC_API_KEY (or store one with "
        "`codna key set anthropic`, or configure another provider). The local engine "
        "itself needs no login."
    )
    # The structured shape is preserved: same code, and the raw engine frame stays in details.
    assert excinfo.value.details["terminal_state"] == "runtime_invalid"
    assert (
        excinfo.value.details["error"]
        == "Error: no API key in env for 'anthropic' (tried: ANTHROPIC_API_KEY)"
    )


def test_other_failures_keep_the_generic_message():
    frame = json.dumps(
        {"type": "final", "status": "failed", "terminal_state": "failed", "error": "boom"}
    )
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(frame, sidecar_url="http://127.0.0.1:1", task_kind="fix")
    assert str(excinfo.value) == "Local agent-core did not complete the packaged fix run successfully."


def test_the_message_names_the_task_kind():
    with pytest.raises(PackagedRepositoryAdvancedError) as excinfo:
        _parse_final_frame(_MISSING_KEY_FRAME, sidecar_url="http://127.0.0.1:1", task_kind="review")
    assert str(excinfo.value).startswith("codna review needs a provider key")
