"""``codna.yaml`` loader — applies command-relevant config before dispatch.

Fail-closed on unresolved ``env:`` refs and unknown providers when a command actually consumes the
model configuration, so config is never a silent no-op for model-backed actions. Deterministic
commands still load privacy posture without requiring an LLM key. YAML via PyYAML when present, else
the file may be JSON (the same tolerant pattern as manifest loading).

Mapping to real controls:
  - ``model.provider`` + ``model.key`` → the provider's ``*_API_KEY`` env (via ``keystore.PROVIDER_ALIASES``),
    which ``runtime.local_stack`` forwards to the fixing engine. ``key: env:NAME`` reads ``$NAME`` (e.g.
    ``env:CODNA_MODEL_KEY``); a literal value is used as-is. The same provider is also written to
    ``CODNA_AGENT_PROVIDER`` so the packaged local Cline SDK sidecar does not silently default to another
    provider.
  - ``privacy.egress: fail-closed`` → sets ``CODNA_REQUIRE_EGRESS_DENY=1`` (the real enforcement is the
    sandbox network-deny policy + the PR manifest gate; there is no global egress env otherwise).
  - ``privacy.redact_secrets`` → accept-and-warn: ``true`` is the posture; ``false`` is *ignored with a
    warning* because there is no redaction toggle to honor. Never faked, never errors.
"""
from __future__ import annotations

import json
import os
import sys

from .keystore import PROVIDER_ALIASES

_SEARCH = ("codna.yaml", ".codna.yaml")


class ConfigError(Exception):
    """Invalid ``codna.yaml`` (surfaced to the user)."""


def _load(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None
    if yaml is not None:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # malformed YAML must fail closed, not crash the CLI
            raise ConfigError(f"cannot parse {path}: {exc}") from exc
    else:
        try:
            data = json.loads(raw or b"{}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConfigError(
                f"cannot parse {path}: PyYAML is not installed and the content is not JSON ({exc})"
            ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top-level config must be a mapping")
    return data


def _resolve_env_ref(value, *, field: str):
    """Resolve an ``env:NAME`` reference; fail closed if the referenced var is unset/empty."""
    if isinstance(value, str) and value.startswith("env:"):
        name = value[4:].strip()
        got = (os.environ.get(name) or "").strip()
        if not got:
            raise ConfigError(f"{field}: env:{name} is not set in the environment")
        return got
    return value


def discover_config_path(explicit: str | None = None) -> str | None:
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            raise ConfigError(f"--config {explicit}: file not found")
        return path
    for name in _SEARCH:
        if os.path.isfile(name):
            return name
    return None


def apply_config(path: str, *, include_model: bool = True) -> None:
    data = _load(path)

    model = data.get("model") or {}
    if include_model and model:
        provider = str(model.get("provider", "")).strip().lower()
        if not provider:
            raise ConfigError("model.provider is required when a model block is set")
        target = PROVIDER_ALIASES.get(provider)
        if not target:
            raise ConfigError(f"model.provider {provider!r} is unknown; valid: {sorted(PROVIDER_ALIASES)}")
        key = _resolve_env_ref(model.get("key"), field="model.key")
        if key:
            # Explicit config is authoritative — the user wrote codna.yaml deliberately.
            os.environ[target] = str(key)
        os.environ["CODNA_AGENT_PROVIDER"] = provider

    privacy = data.get("privacy") or {}
    egress = str(privacy.get("egress", "")).strip().lower()
    if egress in ("fail-closed", "deny", "none"):
        os.environ["CODNA_REQUIRE_EGRESS_DENY"] = "1"
    elif egress:
        print(f"codna: warning: privacy.egress={egress!r} unrecognized (use 'fail-closed')", file=sys.stderr)
    if privacy.get("redact_secrets") is False:
        print(
            "codna: warning: privacy.redact_secrets=false is ignored — secret redaction cannot be disabled",
            file=sys.stderr,
        )


def load_and_apply(explicit: str | None = None, *, include_model: bool = True) -> str | None:
    """Discover + apply a codna.yaml. Returns the path applied (or None). Raises ConfigError."""
    path = discover_config_path(explicit)
    if path:
        apply_config(path, include_model=include_model)
    return path
