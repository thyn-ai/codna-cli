"""Provider selection for a durable run — a LEAF module the workflow side may import.

Why a durable run needs to name a provider at all: `TaskSpec` used to carry no provider field, so
`_ensure_session` built a session spec with only `owner` and the engine fell back to its own default
(`anthropic`). On a machine or CI runner holding only `OPENAI_API_KEY` every durable turn therefore
died with ``ProviderCredentialError: no API key in env for 'anthropic'``, and no keyless local
provider (Ollama / LM Studio / an OpenAI-compatible endpoint) could be used for a durable run at all.

This module owns the coordinates and their rules. Two are load-bearing:

**1. A provider KEY is never a workflow input.** A `TaskSpec` enters Temporal history and is mirrored
into the run's ledger — the artifact people paste into tickets. So a spec carries the *provider*,
never the credential: ``provider_key_source`` says only WHERE the key comes from (``environment``),
and the worker — which already holds the process environment — attaches it to the create-session body
at turn time. There is deliberately no ``"inline"`` member of `PROVIDER_KEY_SOURCES`: a key value in
an input is the thing this type makes unrepresentable. `find_secret_like_strings` is still run over
every field, so a key pasted into ``model`` is a refusal rather than a ledger record.

**2. ``base_url`` decides where the credential is SENT.** It is therefore validated as a security
boundary, not a format: http/https only, no userinfo, no query or fragment, and a CLEARTEXT URL must
be loopback. That is the same rule `codna.config_file._validated_base_url` and the IDE's
``@codna/protocol`` `decideProviderBaseUrl` apply, kept identical here so a durable run cannot
configure something the IDE refuses.

It lives in its own file because `contracts.py` crossed the repository's 1000-line ceiling
(`tests/test_modularity.py`), and because it is a genuine leaf: stdlib plus `contract_primitives` and
`state`, both of which the import-hygiene test already proves sandbox-clean. It must never import
`contracts` — that would close a cycle, since `contracts` imports this.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Mapping
from urllib.parse import urlsplit

from .contract_primitives import ContractError
from .state import find_secret_like_strings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .contracts import TaskContext

__all__ = [
    "KEYLESS_PROVIDERS",
    "MAX_MODEL_ID_CHARS",
    "MAX_PROVIDER_BASE_URL_CHARS",
    "MAX_PROVIDER_ID_CHARS",
    "PROVIDER_COORDINATE_FIELDS",
    "PROVIDER_ENV_KEYS",
    "PROVIDER_KEY_SOURCES",
    "ProviderCoordinates",
    "canonicalise_provider_fields",
    "provider_coordinates_from_request",
    "provider_env_key_names",
    "provider_session_spec",
    "resolve_provider_env_key",
    "validate_provider_fields",
]

# The providers that take NO credential. The same set the IDE's `@codna/protocol`
# (`KEYLESS_PROVIDERS`) and the engine's vendored adapter carry; each copy is pinned by its own test,
# so the three cannot drift apart without a red suite.
KEYLESS_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"ollama", "lmstudio", "openai-compatible", "litellm"}
)

# Where a provider credential comes from. A CLOSED set, with deliberately one member: ``environment``
# is the durable path (the worker reads the process env; nothing enters history). There is no
# ``inline`` — see rule 1 above.
PROVIDER_KEY_SOURCES: Final[tuple[str, ...]] = ("environment",)

# Provider ids and model ids. Both are forwarded to the engine, become part of a session record and are
# echoed in logs, so both are narrow. Provider ids are the engine's family aliases, lowercased
# (`anthropic`, `openai`, `openai-compatible`, `ollama`, `lmstudio`, `gemini`, …). A model id is a
# provider-scoped string that may carry `/`, `:`, `.`, `_`, `-` (`qwen3.6:latest`,
# `anthropic/claude-sonnet-4`).
MAX_PROVIDER_ID_CHARS: Final = 64
MAX_MODEL_ID_CHARS: Final = 200
MAX_PROVIDER_BASE_URL_CHARS: Final = 2048

_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")

# The env vars a provider's credential may live in, in priority order.
#
# A THIRD copy of this mapping, deliberately: `codna.keystore.PROVIDER_ALIASES` is the CLI's
# keystore-facing spelling and lives above this package (it imports `codna.runtime.config`, which
# imports httpx — so importing it here would drag the HTTP stack into the Temporal workflow sandbox and
# kill the worker at construction). The engine resolves its own collection `env` list. All three are
# pinned: `test_durable_provider.py` asserts this agrees with `keystore.PROVIDER_ALIASES` for every
# shared provider, so a drift is a red test rather than a run that cannot find its key.
PROVIDER_ENV_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai-native": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "cursor": ("CURSOR_API_KEY",),
    # A self-hosted proxy / local server usually needs no credential, but LiteLLM and an
    # OpenAI-compatible gateway often do. Named so a user who set one gets it forwarded.
    "litellm": ("LITELLM_API_KEY", "OPENAI_API_KEY"),
    "openai-compatible": ("OPENAI_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
    "lmstudio": ("LMSTUDIO_API_KEY",),
}


def provider_env_key_names(provider: str) -> tuple[str, ...]:
    """The env vars to look in for ``provider``, or ``()`` for an unmapped provider.

    An unmapped provider is not a refusal: the engine may still know it (and may read its own
    environment). It only means the durable path has no key to forward, which the caller checks.
    """
    return PROVIDER_ENV_KEYS.get(provider.strip().lower(), ())


def resolve_provider_env_key(provider: str, env: Mapping[str, str]) -> str | None:
    """The provider's credential from ``env``, or ``None`` when no named var holds a non-blank value.

    Pure over the mapping, so the worker's process environment and a test's are the same call. The
    value is never logged, never returned in an error's details, and never enters history.
    """
    for name in provider_env_key_names(provider):
        value = env.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def canonicalise_provider_fields(holder: Any) -> None:
    """Trim the string coordinates in place, leaving ``None`` alone.

    `TaskSpec`/`TaskContext` are frozen, which is why this uses ``object.__setattr__``: the value is
    being NORMALISED before anyone can observe it, which is not the mutation the frozen flag exists to
    prevent. Trimming on construction (not only in the validators) means ``to_payload`` records the
    canonical value, so a resumed session and a replayed activity see the same provider the first
    attempt used.

    A provider is NOT lowercased: ``OpenAI`` is a refusal (see `_require_provider_id`), and silently
    repairing it would hide a typo that means the engine cannot find the provider.
    """
    for name in ("provider", "model", "base_url", "provider_key_source"):
        value = getattr(holder, name)
        if isinstance(value, str):
            object.__setattr__(holder, name, value.strip())


def validate_provider_fields(holder: Any) -> None:
    """Validate the provider coordinates on a `TaskSpec` or a `TaskContext`.

    Shared by both types because a `TaskContext` is decoded from a request independently of the spec
    that produced it: an activity must refuse a bad coordinate the same way the workflow side does, or
    the two halves would disagree about what reached the engine.

    Every failure is a `ContractError` with a stable code — ``contract_invalid`` for a shape problem
    and ``contract_contains_secret`` for a credential smuggled into a field that enters history.
    """
    # SECRETS FIRST, and before the coherence checks: those checks report on a provider id, and
    # echoing a pasted key inside a "needs a provider" message would defeat screening it. The finding
    # names the PATH only, never the value.
    findings = find_secret_like_strings(
        {
            "provider": holder.provider,
            "model": holder.model,
            "base_url": holder.base_url,
            "provider_key_source": holder.provider_key_source,
        }
    )
    if findings:
        raise ContractError(
            "a provider coordinate contains secret-like data and may not enter workflow history",
            code="contract_contains_secret",
            details={"findings": findings},
        )
    provider = holder.provider
    if provider is not None:
        _require_provider_id(provider)
    model = holder.model
    if model is not None:
        _require_model_id(model)
    if holder.base_url is not None:
        _require_provider_base_url(holder.base_url)
    source = holder.provider_key_source
    if source is not None:
        if source not in PROVIDER_KEY_SOURCES:
            raise ContractError(
                "provider_key_source must be one of " + ", ".join(PROVIDER_KEY_SOURCES),
                code="contract_invalid",
                details={"field": "provider_key_source", "value": repr(source)},
            )
        # "Read the key from the environment" is meaningless without saying WHICH provider's env.
        if provider is None:
            raise ContractError(
                "provider_key_source needs a provider: there is no env var to read without one",
                code="contract_invalid",
                details={"field": "provider_key_source", "value": source},
            )
    # A base_url with no provider would be sent nowhere; the engine resolves a base_url against the
    # selected provider, so accepting it alone would be a silent no-op.
    if holder.base_url is not None and provider is None:
        raise ContractError(
            "base_url needs a provider: an endpoint override is meaningless without one",
            code="contract_invalid",
            details={"field": "base_url"},
        )


# The three coordinates a START REQUEST may name. Deliberately excludes ``provider_key_source``: the
# environment is the only source this durable path has, and a request that could name one is a request
# that could name an inline key. It is fixed to ``None`` (or, for a hosted provider, ``"environment"``)
# by `provider_coordinates_from_request`, never by the caller.
PROVIDER_COORDINATE_FIELDS: Final[tuple[str, ...]] = ("provider", "model", "base_url")


@dataclass(frozen=True)
class ProviderCoordinates:
    """What an external request (``POST /runs``) is allowed to say about the model provider.

    Every value is either ``None`` — "say nothing, the engine resolves its own default" — or a string
    that has passed `validate_provider_fields`. ``provider_key_source`` is derived HERE rather than
    accepted, which is what keeps a credential out of the wire: a caller names the provider, the worker
    reads the key from its own environment.
    """

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    provider_key_source: str | None = None


def provider_coordinates_from_request(body: Mapping[str, Any]) -> ProviderCoordinates:
    """Validate the provider fields of a start request, THROUGH the contract validator.

    This is the API's single entry point into `provider.py`, and it exists so the HTTP surface cannot
    grow a second set of rules: the coordinates are put into a `ProviderCoordinates` (which satisfies
    the same attribute contract as `TaskSpec`/`TaskContext`) and `validate_provider_fields` is called
    on it. A bad provider id, a non-loopback cleartext ``base_url``, a ``base_url`` with no provider
    and a secret-shaped value are therefore the SAME `ContractError` codes the CLI produces — one
    validator, two entrances.

    One refusal is specific to a REQUEST rather than to a spec: a value that is not a string at all
    (``provider: 3``, ``model: {...}``). The spec's own validators would refuse that too, but with a
    message about a "provider id", which is a confusing answer to "your JSON had an object there".

    ``provider_key_source`` is DERIVED, never read from the request: ``"environment"`` exactly when a
    provider is named, so the worker resolves the key from its own process env. There is no route field
    for it, and `PROVIDER_KEY_SOURCES` has one member, so "inline" is not expressible on this path.
    """
    coordinates = ProviderCoordinates(
        provider=_request_string(body, "provider"),
        model=_request_string(body, "model"),
        base_url=_request_string(body, "base_url"),
    )
    canonicalise_provider_fields(coordinates)
    validate_provider_fields(coordinates)
    return ProviderCoordinates(
        provider=coordinates.provider,
        model=coordinates.model,
        base_url=coordinates.base_url,
        provider_key_source="environment" if coordinates.provider is not None else None,
    )


def _request_string(body: Mapping[str, Any], field: str) -> str | None:
    """One coordinate as a ``str | None``, refusing a non-string outright.

    ``None`` and an absent key both mean "not supplied". A blank string is passed through on purpose:
    the validator below already refuses ``"   "`` with a message that names the field, and stripping it
    here would make the canonicalisation in `canonicalise_provider_fields` untestable through this path.
    """
    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError(
            f"{field} must be a string",
            code="contract_invalid",
            details={"field": field, "type": type(value).__name__},
        )
    return value


def _require_provider_id(value: Any) -> str:
    """A provider id: non-empty, bounded, and inside the engine's alias alphabet."""
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            "provider must be a non-empty string",
            code="contract_invalid",
            details={"field": "provider"},
        )
    text = value.strip()
    if len(text) > MAX_PROVIDER_ID_CHARS:
        raise ContractError(
            f"provider is {len(text)} characters, over the {MAX_PROVIDER_ID_CHARS}-character limit",
            code="contract_invalid",
            details={"field": "provider", "length": len(text)},
        )
    # Canonicalised on construction, so `" openai "` arrives here as `"openai"`. A value that still
    # fails the alphabet — including an INTERNAL newline, which `strip()` cannot remove — is a
    # refusal: the engine's provider aliases are a closed, lowercase set.
    if text != text.lower() or not _PROVIDER_ID.match(text):
        raise ContractError(
            "provider must be a lowercase id matching ^[a-z0-9][a-z0-9._-]{0,63}$ "
            "(e.g. anthropic, openai, ollama, openai-compatible)",
            code="contract_invalid",
            details={"field": "provider", "value": text[:MAX_PROVIDER_ID_CHARS]},
        )
    return text


def _require_model_id(value: Any) -> str:
    """A model id: non-empty, bounded, no whitespace, no control characters."""
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            "model must be a non-empty string", code="contract_invalid", details={"field": "model"}
        )
    text = value.strip()
    if len(text) > MAX_MODEL_ID_CHARS:
        raise ContractError(
            f"model is {len(text)} characters, over the {MAX_MODEL_ID_CHARS}-character limit",
            code="contract_invalid",
            details={"field": "model", "length": len(text)},
        )
    if not _MODEL_ID.match(text):
        raise ContractError(
            "model must match ^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$ (a provider model id, "
            "e.g. gpt-4.1, claude-sonnet-4, qwen3.6:latest)",
            code="contract_invalid",
            details={"field": "model", "value": text[:MAX_MODEL_ID_CHARS]},
        )
    return text


def _require_provider_base_url(value: Any) -> str:
    """A provider endpoint: http(s), no credentials/query/fragment, and not a remote cleartext host.

    A base URL decides WHERE the provider credential is sent, so this is a security check, not a
    formatting one:

    * only ``http``/``https`` — a ``file:``/``data:``/``javascript:`` URL must never become a fetch
      target;
    * no embedded userinfo — ``https://user:pass@host`` puts a secret in every request and log line;
    * no query or fragment — silently dropped by consumers, so the config would lie about itself;
    * a **cleartext** URL must be loopback. Sending a provider key over plain http to another host
      would expose it in transit, so a remote endpoint has to be https. This is the rule the IDE's
      ``decideProviderBaseUrl`` applies (``insecure_transport``), kept identical here so a durable run
      cannot configure something the IDE refuses.
    """
    if not isinstance(value, str) or not value.strip():
        raise ContractError(
            "base_url must be a non-empty string",
            code="contract_invalid",
            details={"field": "base_url"},
        )
    text = value.strip()
    if len(text) > MAX_PROVIDER_BASE_URL_CHARS:
        raise ContractError(
            f"base_url is over the {MAX_PROVIDER_BASE_URL_CHARS}-character limit",
            code="contract_invalid",
            details={"field": "base_url", "length": len(text)},
        )
    if any(character.isspace() for character in text):
        raise ContractError(
            "base_url must not contain whitespace",
            code="contract_invalid",
            details={"field": "base_url"},
        )
    try:
        parts = urlsplit(text)
        port = parts.port  # raises ValueError for a non-numeric / out-of-range port
    except ValueError as exc:
        raise ContractError(
            f"base_url is not a usable URL: {exc}",
            code="contract_invalid",
            details={"field": "base_url"},
        ) from exc
    if parts.scheme not in ("http", "https"):
        raise ContractError(
            f"base_url must be an http(s) URL, got {(parts.scheme or 'no scheme')!r}",
            code="contract_invalid",
            details={"field": "base_url"},
        )
    if not parts.hostname:
        raise ContractError(
            "base_url must name a host", code="contract_invalid", details={"field": "base_url"}
        )
    if parts.username or parts.password:
        raise ContractError(
            "base_url must not embed credentials",
            code="contract_invalid",
            details={"field": "base_url"},
        )
    if parts.query or parts.fragment:
        raise ContractError(
            "base_url must not carry a query string or fragment",
            code="contract_invalid",
            details={"field": "base_url"},
        )
    if parts.scheme == "http" and not _is_loopback_host(parts.hostname):
        raise ContractError(
            "base_url over plain http must point at loopback; a remote endpoint must use https "
            "(a provider key sent in cleartext is exposed in transit)",
            code="contract_invalid",
            details={"field": "base_url", "host": parts.hostname, "port": port},
        )
    return text


def _is_loopback_host(hostname: str) -> bool:
    """Loopback by name or by address, including the whole 127/8 range.

    Deliberately explicit rather than ``ipaddress.is_loopback`` alone: ``localhost`` is not an address,
    and a host must be recognised without a DNS lookup (this runs inside a contract validator, which
    must be deterministic and offline).
    """
    host = hostname.strip().lower().strip("[]")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host == "::1":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def provider_session_spec(context: "TaskContext", *, key: str | None) -> dict[str, Any]:
    """The provider part of a create-session body, for a task context.

    Absent coordinates are OMITTED rather than sent as ``None``: the engine distinguishes "no
    provider" (its own default) from "a provider named null", and every one of these keys is a real
    field on `SessionRecord`. ``key`` is attached only when the caller resolved one from the process
    environment; the workflow never sees it.
    """
    spec: dict[str, Any] = {}
    if context.provider is not None:
        spec["provider"] = context.provider
    if context.model is not None:
        spec["model"] = context.model
    if context.base_url is not None:
        spec["base_url"] = context.base_url
    if key:
        # `provider_api_key` + the fail-closed flag travel as a pair: a tenant-style deployment that
        # demands its own key must never fall back to the shared process env. The durable path has
        # exactly one such key (the env one), so it always sets both.
        spec["provider_api_key"] = key
        spec["require_provider_api_key"] = True
    return spec
