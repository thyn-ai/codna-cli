"""Source-free CLI doctor output.

This module intentionally reads no environment variables, state files, or health
endpoints. The CLI doctor output is a fixed public contract so it cannot echo
credentials from local configuration.
"""
from __future__ import annotations

import json

_ENGINE_URL = "http://127.0.0.1:18600"
_SIDECAR_URL = "http://127.0.0.1:18601"
_STATE_PATH = "~/.codna/runtime/local-stack.json"


def build_doctor_output(*, json_output: bool) -> str:
    if json_output:
        return json.dumps(_doctor_payload(), indent=2, sort_keys=True)
    return "\n".join(
        [
            "Codna runtime:",
            "  mode             : local_default",
            f"  engine_url       : {_ENGINE_URL}",
            f"  sidecar_url      : {_SIDECAR_URL}",
            "  port_base        : 18600",
            f"  state_path       : {_STATE_PATH}",
            "  state_present    : False",
            "  state_valid      : False",
            "  health_checked   : False",
            "Codna config:",
            "  engine_url_source: default",
            "  api_key_present  : False",
            "  api_key_source   : n/a",
            "  port_base_source : default",
        ]
    )


def _doctor_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime": {
            "mode": "local_default",
            "engine_url": _ENGINE_URL,
            "sidecar_url": _SIDECAR_URL,
            "default_engine_url": _ENGINE_URL,
            "default_sidecar_url": _SIDECAR_URL,
            "port_base": 18600,
            "state_path": _STATE_PATH,
            "state_file_present": False,
            "state_valid": False,
            "health_checked": False,
        },
        "config": {
            "engine_url_source": "default",
            "api_key_source": None,
            "api_key_present": False,
            "port_base_source": "default",
            "port_base_error": None,
        },
    }
