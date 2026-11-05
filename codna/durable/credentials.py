"""HOW a durable process authenticates to Temporal Cloud (D6): API key or mTLS, from the keystore.

`config.py` already decided *where* the server is and *that* Cloud needs TLS; it also names the
environment variable its API key would come from and reads that value at connect time. What it
deliberately does not do is decide between the two credential SHAPES Temporal Cloud accepts, because
that decision needs a keystore read and a fallback order:

* **API key** — one bearer value, the same thing the Cloud UI hands out.
* **mTLS** — a client certificate plus its private key (and optionally a private CA bundle).

Both are credentials, and in a HOSTED worker neither may be baked into the image, a build arg or a
committed file. So every value is resolved here, at connect time, from exactly two places:

1. the process environment (what a platform secret store — Fly secrets, a k8s Secret — injects into a
   running container; the image itself sets none of these), then
2. the OS keychain (`codna.keystore`), which is how a developer machine or the desktop app holds them.

Order matters and is stated rather than guessed: an environment value WINS, because a platform
secret is the operator's explicit statement about this container, while a keychain entry is whatever
happened to be on the machine. `lookup` and `env` are injectable so the order is testable without a
keychain.

**Never printed.** These values live only in local variables and in the returned dataclass, whose
``repr``/``str`` are redacted (same reason `ConnectOptions` exists: a failing assertion or a debugger
dump is where a key leaks). Errors name VARIABLES and keystore ENTRIES, never values.

`temporalio` is imported inside the one function that needs it (`TLSConfig`), so this module stays
importable by the base CLI — the same rule every other module in this package follows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

__all__ = [
    "AUTH_MODE_AUTO",
    "AUTH_MODES",
    "ENV_AUTH",
    "ENV_CLIENT_CERT",
    "ENV_CLIENT_KEY",
    "ENV_SERVER_CA",
    "TemporalAuthError",
    "TemporalCredentials",
    "fingerprint_of",
    "resolve_temporal_credentials",
]

#: `api_key` | `mtls` | `auto`. `auto` is the default: use mTLS when its material is present, else
#: the API key. An explicit mode turns a partial configuration into a named refusal instead of a
#: silent switch to the other shape — which is what an operator debugging "it authenticated as the
#: wrong tenant" needs.
ENV_AUTH: Final = "CODNA_TEMPORAL_AUTH"
AUTH_MODE_AUTO: Final = "auto"
AUTH_MODES: Final[tuple[str, ...]] = (AUTH_MODE_AUTO, "api_key", "mtls")

# The variable/keystore names. One spelling for both sources so an operator reading the docs, the
# `fly secrets set` line and the keystore entry sees the same string.
ENV_API_KEY: Final = "TEMPORAL_API_KEY"
ENV_CLIENT_CERT: Final = "CODNA_TEMPORAL_CLIENT_CERT_PEM"
ENV_CLIENT_KEY: Final = "CODNA_TEMPORAL_CLIENT_KEY_PEM"
ENV_SERVER_CA: Final = "CODNA_TEMPORAL_SERVER_CA_PEM"


class TemporalAuthError(RuntimeError):
    """The Cloud credential set is incomplete or contradictory. Always carries a stable ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


class ConnectOptions(dict):  # type: ignore[type-arg]
    """``Client.connect`` keyword arguments whose ``repr`` never prints a credential.

    A plain dict is a leak waiting to happen: ``repr(options)`` is what appears in a failing
    assertion, a debugger, a ``print`` during development and a log line that formats its arguments —
    and measured while building D2, an assertion that formatted these options printed a live Temporal
    Cloud API key into a terminal. Subclassing dict keeps ``**options`` working exactly as the SDK
    expects while making the accidental path safe: only ``options["api_key"]`` returns the value, and
    specifically asking for the key is a deliberate act.

    ``__str__`` is overridden too, because ``str()`` on a dict falls back to ``repr`` in most
    formatting and an f-string does not distinguish the two. mTLS material is nested inside a
    ``TLSConfig``; it never reaches this mapping as a string, so it cannot be formatted here.

    Lives in this module (not `config`) because the resolver below builds it, and `config` imports
    this module — re-exported from `config` so its historical import site keeps working.
    """

    _REDACTED_KEYS: Final = ("api_key",)

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return f"ConnectOptions({self._safe()})"

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.__repr__()

    def _safe(self) -> dict[str, object]:
        return {
            key: ("<redacted>" if key in self._REDACTED_KEYS and value else value)
            for key, value in self.items()
        }


@dataclass(frozen=True)
class TemporalCredentials:
    """The resolved credential material for one connection. Redacted when formatted.

    ``mode`` records which SHAPE was chosen ("api_key" or "mtls") so a startup banner can report it
    without reporting anything secret. ``sources`` records, per field, whether the value came from
    the environment or the keystore — provenance an operator needs and a value map would leak.
    """

    mode: str
    api_key: str | None = None
    client_cert_pem: bytes | None = None
    client_key_pem: bytes | None = None
    server_ca_pem: bytes | None = None
    sources: Mapping[str, str] | None = None

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return f"TemporalCredentials(mode={self.mode!r}, sources={dict(self.sources or {})})"

    __str__ = __repr__

    def to_payload(self) -> dict[str, object]:
        """The JSON-safe description a banner may print: shape + provenance, never material."""
        return {
            "mode": self.mode,
            "sources": dict(self.sources or {}),
            "mtls": self.mode == "mtls",
        }

    def connect_kwargs(self) -> dict[str, object]:
        """The ``Client.connect`` keywords for this credential shape.

        mTLS becomes a ``TLSConfig`` (Temporal Cloud still needs TLS; the client certificate is how
        the tenant knows who is calling). The API key path passes the key with ``tls=True`` exactly
        as `config.temporal_connect_options` does, so both paths are TLS and neither can downgrade.
        """
        if self.mode == "mtls":
            from temporalio.service import TLSConfig

            if self.client_cert_pem is None or self.client_key_pem is None:  # pragma: no cover
                raise TemporalAuthError(
                    "mTLS was selected without a certificate and key",
                    code="temporal_mtls_incomplete",
                    details={"mode": self.mode},
                )
            return ConnectOptions(
                tls=TLSConfig(
                    server_root_ca_cert=self.server_ca_pem,
                    client_cert=self.client_cert_pem,
                    client_private_key=self.client_key_pem,
                )
            )
        if not self.api_key:  # pragma: no cover - guarded by the resolver
            raise TemporalAuthError(
                "the api_key credential shape was selected without a key",
                code="temporal_cloud_config_incomplete",
                details={"mode": self.mode},
            )
        return ConnectOptions(tls=True, api_key=self.api_key)


def _read(name: str, env: Mapping[str, str], lookup: Any) -> tuple[str | None, str]:
    """Read one credential: environment first, then the keystore. Returns (value, source).

    An env var that is PRESENT BUT BLANK is treated as absent here on purpose: a platform that
    rendered an empty secret should fall through to the other source rather than become a value that
    fails to parse later. (The API-key *presence* check that must fail closed lives in
    `config._require_cloud_coordinates`, where "the operator said cloud" is known.)

    ``lookup`` is asked for EVERY name, not only the mTLS pair: a keychain is where a developer or the
    desktop app holds these, and `keystore.MANAGED_KEYS` deliberately does not include them, so a
    keystore lookup answers None and the resolver moves on. A keystore read is only reached for names
    the environment did not supply, so a platform-injected value is never shadowed by a local one.
    """
    raw = env.get(name)
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), "environment"
    if lookup is not None:
        try:
            value = lookup(name)
        except Exception:  # noqa: BLE001 - an unreadable keychain is "absent", not a crash
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip(), "keystore"
    return None, "missing"


def fingerprint_of(material: bytes) -> str:
    """A short, non-reversible id for a certificate, for logs and banners ("which cert is loaded?").

    A SHA-256 prefix: enough to tell two certificates apart, useless for recovering one, and stable
    across restarts so a banner comparison answers "did the rotation take effect".
    """
    import hashlib

    return hashlib.sha256(material).hexdigest()[:16]


def _pem_bytes(value: str, *, name: str) -> bytes:
    """Validate that a resolved value is PEM text before it reaches TLSConfig.

    A base64 body pasted without its header, or a path pasted instead of contents, otherwise fails
    deep inside gRPC with "could not load certificate", which says nothing about which variable is
    wrong. Refusing here names the variable.
    """
    text = value.strip()
    if "-----BEGIN" not in text:
        raise TemporalAuthError(
            f"{name} does not contain PEM data (a -----BEGIN line is missing).",
            code="temporal_mtls_material_invalid",
            details={"variable": name, "received_chars": len(text)},
        )
    return text.encode("utf-8")


def _resolve_mode(raw: str | None) -> str:
    value = (raw or AUTH_MODE_AUTO).strip().lower()
    if value not in AUTH_MODES:
        raise TemporalAuthError(
            f"{ENV_AUTH} must be one of {', '.join(AUTH_MODES)} (got {raw!r}).",
            code="invalid_temporal_auth",
            details={"variable": ENV_AUTH, "value": raw, "allowed": list(AUTH_MODES)},
        )
    return value


def resolve_temporal_credentials(
    env: Mapping[str, str] | None = None,
    *,
    lookup: Any | None = None,
) -> TemporalCredentials:
    """Resolve the credential shape for a Temporal Cloud connection, or refuse by name.

    ``lookup`` defaults to ``codna.keystore.get_key`` (returns None rather than raising when there is
    no keychain), and is only consulted for values the environment did not supply — so a test that
    injects a mapping never touches the real keychain.

    Every refusal names the VARIABLES or ENTRIES to set. None of them formats a value.
    """
    import os

    source = os.environ if env is None else env
    if lookup is None:
        try:
            from ..keystore import get_key as lookup  # type: ignore[assignment]
        except Exception:  # noqa: BLE001 - no keystore module means "environment only"
            lookup = None

    mode = _resolve_mode(source.get(ENV_AUTH))
    api_key, api_key_source = _read(ENV_API_KEY, source, lookup)
    cert, cert_source = _read(ENV_CLIENT_CERT, source, lookup)
    private_key, key_source = _read(ENV_CLIENT_KEY, source, lookup)
    server_ca, ca_source = _read(ENV_SERVER_CA, source, lookup)

    has_mtls = cert is not None and private_key is not None
    mtls_partial = (cert is None) != (private_key is None)
    if mode == "mtls" or (mode == AUTH_MODE_AUTO and has_mtls):
        if mtls_partial:
            missing = ENV_CLIENT_KEY if private_key is None else ENV_CLIENT_CERT
            raise TemporalAuthError(
                f"mTLS needs both {ENV_CLIENT_CERT} and {ENV_CLIENT_KEY}; {missing} is missing.",
                code="temporal_mtls_incomplete",
                details={"missing": [missing], "mode": mode},
            )
        if not has_mtls:
            raise TemporalAuthError(
                f"{ENV_AUTH}=mtls needs {ENV_CLIENT_CERT} and {ENV_CLIENT_KEY}.",
                code="temporal_mtls_incomplete",
                details={"missing": [ENV_CLIENT_CERT, ENV_CLIENT_KEY], "mode": mode},
            )
        assert cert is not None and private_key is not None
        return TemporalCredentials(
            mode="mtls",
            client_cert_pem=_pem_bytes(cert, name=ENV_CLIENT_CERT),
            client_key_pem=_pem_bytes(private_key, name=ENV_CLIENT_KEY),
            server_ca_pem=_pem_bytes(server_ca, name=ENV_SERVER_CA) if server_ca else None,
            sources={
                "client_cert": cert_source,
                "client_key": key_source,
                "server_ca": ca_source,
            },
        )

    # API-key shape. In `auto`, a PARTIAL mTLS setup with a usable key is still an error: the operator
    # clearly intended mTLS (they set half of it), and silently authenticating as the API key would
    # hide the half-configured certificate until the day the key is rotated away.
    if mode == AUTH_MODE_AUTO and mtls_partial:
        missing = ENV_CLIENT_KEY if private_key is None else ENV_CLIENT_CERT
        raise TemporalAuthError(
            f"{missing} is missing, so the mTLS credential set is incomplete. Set it, remove the "
            f"other half, or set {ENV_AUTH}=api_key to authenticate with the API key only.",
            code="temporal_mtls_incomplete",
            details={"missing": [missing], "mode": mode},
        )
    if api_key is None:
        raise TemporalAuthError(
            f"no Temporal Cloud credential is configured: set {ENV_API_KEY} (or "
            f"{ENV_CLIENT_CERT}+{ENV_CLIENT_KEY} for mTLS).",
            code="temporal_cloud_config_incomplete",
            details={
                "missing": [ENV_API_KEY],
                "alternatives": [ENV_CLIENT_CERT, ENV_CLIENT_KEY],
                "mode": mode,
            },
        )
    return TemporalCredentials(
        mode="api_key", api_key=api_key, sources={"api_key": api_key_source}
    )
