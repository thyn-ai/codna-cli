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
  - ``model.base_url`` → ``CODNA_AGENT_BASE_URL``, the provider ENDPOINT override forwarded to the sidecar.
    A base URL decides where the provider credential is sent, so it is validated here (http/https, no
    embedded credentials, no query/fragment, plain HTTP only on loopback) and it may come from THIS
    file or the environment only — never from a repository-scoped settings file. ``env:NAME`` resolves
    the same way as ``model.key``.
  - ``privacy.egress: fail-closed`` → sets ``CODNA_REQUIRE_EGRESS_DENY=1`` (the real enforcement is the
    sandbox network-deny policy + the PR manifest gate; there is no global egress env otherwise).
  - ``privacy.redact_secrets`` → accept-and-warn: ``true`` is the posture; ``false`` is *ignored with a
    warning* because there is no redaction toggle to honor. Never faked, never errors.
"""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlsplit

from .keystore import PROVIDER_ALIASES
from .provider_url import is_loopback_host

_SEARCH = ("codna.yaml", ".codna.yaml")

# The env var the sidecar reads for a provider endpoint override. Kept in step with
# cli/codna/packaged_agent_runner.py (`_sidecar_payload`), which is the side that actually sends it.
BASE_URL_ENV = "CODNA_AGENT_BASE_URL"

# Providers that take NO credential. A model block for one of these is valid with no `key` at all: there is no
# keystore alias to write, and inventing one would imply a key is needed. Canonical list lives in three places
# that cannot import each other (this file, the IDE's `@codna/protocol`, and the engine's adapter.ts); each is
# pinned by its own test so they cannot drift apart silently.
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio", "openai-compatible", "litellm"})


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


def discover_config_path(explicit: str | None = None, *, include_home: bool = False) -> str | None:
    """Locate the config file: an explicit `--config`, `./codna.yaml`, `./.codna.yaml`, then — only when
    `include_home` — `~/.codna/config.yaml`.

    The HOME-relative file is opt-in (it is consulted exactly when a command consumes MODEL config) because a
    provider ENDPOINT is a property of the user's machine, not of whichever repository the CLI happens to run
    in: keys and endpoints set there must apply everywhere. Deterministic commands keep the historical
    project-only search, so they behave identically on every machine.
    """
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            raise ConfigError(f"--config {explicit}: file not found")
        return path
    for name in _SEARCH:
        if os.path.isfile(name):
            return name
    if include_home:
        home = os.path.join(os.path.expanduser("~"), ".codna", "config.yaml")
        if os.path.isfile(home):
            return home
    return None


def _validated_base_url(value: object, *, field: str) -> str:
    """Validate a provider endpoint override before it can become a fetch target.

    Fail-closed on every shape that could smuggle something past the consumer: a non-http(s) scheme, an
    embedded credential (``https://user:pass@host`` puts secrets in logs and in every request), a query or
    fragment (silently dropped by every consumer, so the config would lie about itself), and whitespace.
    Plain HTTP is restricted to loopback so a provider key cannot cross the network in cleartext. The
    trailing slash is removed so the consumer's ``${base}/v1`` cannot become ``//v1``.
    """
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"{field} must be a string URL")
    text = str(value or "").strip()
    if not text:
        raise ConfigError(f"{field} must not be blank")
    if any(ch.isspace() for ch in text):
        raise ConfigError(f"{field} must not contain whitespace")
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise ConfigError(f"{field} must be an http(s) URL, got {parts.scheme or 'no scheme'!r}")
    if not parts.hostname:
        raise ConfigError(f"{field} must name a host")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{field} must not embed credentials")
    if parts.query or parts.fragment:
        raise ConfigError(f"{field} must not carry a query string or fragment")
    if parts.scheme == "http" and not is_loopback_host(parts.hostname):
        raise ConfigError(
            f"{field} over plain http must point at loopback; a remote endpoint must use https"
        )
    path = parts.path.rstrip("/")
    netloc = parts.netloc
    return f"{parts.scheme}://{netloc}{path}"


def apply_config(path: str, *, include_model: bool = True) -> None:
    data = _load(path)

    model = data.get("model") or {}
    if include_model and model:
        provider = str(model.get("provider", "")).strip().lower()
        if not provider:
            raise ConfigError("model.provider is required when a model block is set")
        target = PROVIDER_ALIASES.get(provider)
        # A KEYLESS provider (local inference server / self-hosted proxy) legitimately has no key to alias.
        # An unknown provider is still an error: guessing would ship the user's key to a provider they did not
        # name, and accepting a typo silently is worse than refusing it.
        keyless = provider in KEYLESS_PROVIDERS
        if not target and not keyless:
            raise ConfigError(f"model.provider {provider!r} is unknown; valid: {sorted(PROVIDER_ALIASES)}")
        key = _resolve_env_ref(model.get("key"), field="model.key")
        if key:
            if not target:
                raise ConfigError(
                    f"model.key is not applicable to the keyless provider {provider!r} "
                    "(ollama/lmstudio/openai-compatible/litellm accept requests without a credential)"
                )
            # Explicit config is authoritative — the user wrote codna.yaml deliberately.
            os.environ[target] = str(key)
        os.environ["CODNA_AGENT_PROVIDER"] = provider
        # The endpoint override travels on its OWN variable so a provider switch cannot silently reuse the
        # previous provider's URL. `env:NAME` refs resolve (and fail closed) exactly like `model.key`.
        if "base_url" in model:
            raw = _resolve_env_ref(model.get("base_url"), field="model.base_url")
            os.environ[BASE_URL_ENV] = _validated_base_url(raw, field="model.base_url")

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
    path = discover_config_path(explicit, include_home=include_model)
    if path:
        apply_config(path, include_model=include_model)
    return path
