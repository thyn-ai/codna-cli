"""Turning an HTTP ``Authorization`` header into a `Principal` — the only thing that may do so.

Authorization is only worth anything if the identity it is decided against was verified HERE. So this
module is the single seam between "bytes on a socket" and "a subject with roles", and it has exactly
three outcomes: a `Principal`, an `AuthenticationError`, or a refusal to start.

**Three authentication modes, and why the third one is narrow.**

* ``token`` (the default, and what local development uses). The 0600 bearer token the server mints
  on its own disk. It is a *capability*, not an identity in the fleet sense: whoever can read
  ``~/.codna/runs/run-api.token`` is the local owner. The principal it produces is labelled
  ``source="local-token"`` so an auditor reading the ledger can tell a local act from a federated one.
* ``oidc``. A JWT from the existing device flow (`codna login`, RFC 8628). It is verified against the
  issuer's JWKS — signature, ``iss``, ``aud``, ``exp``/``nbf`` — and its roles come from a claim.
  Nothing about the request body or a client-set header is consulted: the only inputs are the token
  and the issuer metadata.
* ``local-trust``. No credential at all. This is the "local development keeps working" path, and
  making it *explicit* is the whole point: it requires an opt-in, it is refused unless the bind
  address is loopback (which `RunApiSettings` already enforces by construction), and it is refused
  outright when an OIDC issuer is configured — because a server that has an identity provider must
  not also have a hole that skips it. A principal from this mode is labelled
  ``source="local-unauthenticated"`` so "nobody proved who this was" is a permanent, readable fact.

**Failure is closed and uniform.** A missing, malformed, expired or wrongly-signed token is one
`AuthenticationError` with one code (`run_api_unauthorized`), and the message does not distinguish
"no token" from "a token that exists but is wrong": telling an unauthenticated caller whether their
guess was closer is a free oracle. The only exception is a provider problem (JWKS unreachable), which
is a 503 — that is this server's failure, not the caller's, and conflating them would send an operator
hunting a bad token when the identity provider is down.

**What is NOT proven here.** No live provider is reachable from this build's environment, so
`JwtVerifier` is exercised with a locally minted RSA key pair and an in-process JWKS — the SAME code
path the real verifier runs, with the network fetcher replaced by a scripted one. The half that is
unrun is the HTTPS fetch of ``/.well-known/openid-configuration`` and ``jwks_uri``; that is
`HttpJwksSource`, and it is a thin, separately-tested-by-construction adapter. `docs`-level honesty:
see the module `__all__` note below and the E2 report.
"""
from __future__ import annotations

import base64
import binascii
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Protocol

from .roles import Principal, Role, role_from_claim

__all__ = [
    "AUTH_MODE_LOCAL_TRUST",
    "AUTH_MODE_OIDC",
    "AUTH_MODE_TOKEN",
    "AUTH_MODES",
    "AuthenticationError",
    "IdentityConfig",
    "IdentityError",
    "IdentityResolver",
    "JwksSource",
    "JwtVerifier",
    "MemoryJwksSource",
    "ScriptedJwksSource",
    "VerifiedClaims",
    "bearer_from_header",
]

# The three modes as literals. Strings because they are configured from an environment variable and
# echoed into a banner; an Enum here would be converted at both ends for no safety.
AUTH_MODE_TOKEN: Final = "token"
AUTH_MODE_OIDC: Final = "oidc"
AUTH_MODE_LOCAL_TRUST: Final = "local-trust"
AUTH_MODES: Final[frozenset[str]] = frozenset({AUTH_MODE_TOKEN, AUTH_MODE_OIDC, AUTH_MODE_LOCAL_TRUST})

# The identity a local-trust caller is recorded as. It names the machine's owner rather than
# pretending to be a person: the `source` says "nobody proved this", and the identity itself is the
# OS user — the same actor the IDE (`os.userInfo().username`) and the durable CLI name, so an approval
# sent from a terminal and one sent from the IDE refer to the same person.
LOCAL_TRUST_SOURCE: Final = "local-unauthenticated"
LOCAL_TOKEN_SOURCE: Final = "local-token"
OIDC_SOURCE: Final = "oidc"


def local_owner_identity() -> str:
    """The OS user's name, or a marker — never empty, because the API refuses an empty identity.

    ``os.getlogin`` is deliberately avoided: it reads the controlling terminal and raises when there
    is none, which is exactly the case for a server started by an IDE. ``getpass.getuser`` reads the
    passwd database (via the environment, then the file), which is the OS identity proper.
    """
    import getpass

    try:
        name = getpass.getuser().strip()
    except (OSError, KeyError):  # pragma: no cover - a container with no passwd entry
        return "unknown-operator"
    return name or "unknown-operator"

# Tokens are bounded because a header is attacker-controlled and base64-decoding a 10 MB "JWT" is a
# denial of service this server can refuse for free.
MAX_TOKEN_CHARS: Final = 8192
# JWKS rotation means the cache may not be immortal, but a fetch per request is an outage amplifier.
DEFAULT_JWKS_TTL_S: Final = 300.0
# Clock skew allowed on exp/nbf. Small on purpose: a long leeway is a real extension of a token's
# life, and 60s covers the usual NTP drift without granting an extra minute of access.
DEFAULT_LEEWAY_S: Final = 60.0


class IdentityError(RuntimeError):
    """The identity layer is misconfigured. Raised at START time, never per request."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


class AuthenticationError(RuntimeError):
    """A caller could not be authenticated. One body for every cause (see the module docstring)."""

    def __init__(
        self,
        message: str = "this endpoint requires a valid bearer token.",
        *,
        code: str = "run_api_unauthorized",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


class ProviderUnavailable(RuntimeError):
    """The identity provider could not be reached. This server's problem, answered as 503."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = "run_api_identity_provider_unavailable"
        self.details: dict[str, object] = dict(details or {})


# -- the header --------------------------------------------------------------------------------


def bearer_from_header(header: str | None) -> str | None:
    """The token from ``Authorization: Bearer <token>``, or None.

    The scheme is matched case-insensitively (RFC 7235 does not require lowercase) and any other
    scheme is "absent" — an unauthenticated caller must not learn how its header parsed.
    """
    if not header or len(header) > MAX_TOKEN_CHARS + 64:
        return None
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.strip().lower() != "bearer":
        return None
    token = value.strip()
    if not token or len(token) > MAX_TOKEN_CHARS:
        return None
    return token


# -- configuration -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityConfig:
    """How this server decides who is calling. Validated at construction, so a bad mode cannot serve.

    Every optional field is validated only in the mode that uses it: requiring a JWKS path in token
    mode would make local development configure a provider it does not have, and permitting one in
    local-trust mode is the configuration mistake this type exists to refuse.
    """

    mode: str = AUTH_MODE_TOKEN
    # token mode
    local_token: str = ""
    # The roles the local owner holds. Full by default: the person at the keyboard may read, run and
    # approve. Restricting them is possible (a shared dev box), and an empty set is a deny-everything
    # server — which is a legitimate, explicit configuration, not an accident.
    local_roles: frozenset[str] = frozenset({"viewer", "operator", "approver"})
    local_identity: str = ""
    # oidc mode
    issuer: str = ""
    audience: str = ""
    role_claim: str = "roles"
    leeway_s: float = DEFAULT_LEEWAY_S
    jwks_ttl_s: float = DEFAULT_JWKS_TTL_S
    jwks_uri: str = ""
    jwks_json: Mapping[str, Any] | None = None
    # Injected in tests so no network is touched. `None` means "fetch https".
    jwks_fetcher: Callable[[str], Mapping[str, Any]] | None = None
    clock: Callable[[], float] = time.time
    # The identity used when a subject has no `sub` claim. Refusing is the default; a provider that
    # omits `sub` is broken, and inventing an identity for it would put a fabricated name in the
    # audit trail.
    allow_missing_subject: bool = False

    def __post_init__(self) -> None:
        if self.mode not in AUTH_MODES:
            raise IdentityError(
                f"unknown run API auth mode {self.mode!r}.",
                code="run_api_auth_mode_invalid",
                details={"mode": self.mode, "allowed": sorted(AUTH_MODES)},
            )
        unknown = sorted(set(self.local_roles) - {"viewer", "operator", "approver"})
        if unknown:
            raise IdentityError(
                f"unknown local roles: {', '.join(unknown)}.",
                code="run_api_local_role_invalid",
                details={"roles": unknown, "allowed": ["approver", "operator", "viewer"]},
            )
        if self.mode == AUTH_MODE_TOKEN and not self.local_token:
            raise IdentityError(
                "token mode needs the local bearer token.",
                code="run_api_token_missing",
                details={"mode": self.mode},
            )
        if self.mode == AUTH_MODE_OIDC:
            for name, value in (("issuer", self.issuer), ("audience", self.audience)):
                if not isinstance(value, str) or not value.strip():
                    raise IdentityError(
                        f"oidc mode needs {name}.",
                        code="run_api_oidc_incomplete",
                        details={"field": name},
                    )
            if not self.issuer.startswith("https://") and not self.issuer.startswith("http://127.0.0.1") \
                    and not self.issuer.startswith("http://localhost"):
                # An http:// issuer would let a network attacker swap the JWKS and mint their own
                # tokens. Loopback is exempt so a local mock provider is testable.
                raise IdentityError(
                    "the oidc issuer must be https (http is allowed for loopback mocks only).",
                    code="run_api_oidc_issuer_insecure",
                    details={"issuer": self.issuer},
                )
        if self.mode == AUTH_MODE_LOCAL_TRUST and self.issuer:
            # The hole that skips a configured provider. Refused rather than warned about, because a
            # warning is what an operator reads after the fact.
            raise IdentityError(
                "local-trust cannot be combined with an OIDC issuer: a server with an identity "
                "provider must not also accept unauthenticated callers.",
                code="run_api_local_trust_conflicts_with_oidc",
                details={"issuer": self.issuer},
            )
        if self.leeway_s < 0:
            raise IdentityError(
                "token clock leeway must not be negative.",
                code="run_api_oidc_incomplete",
                details={"leeway_s": self.leeway_s},
            )
        if self.jwks_ttl_s <= 0:
            raise IdentityError(
                "jwks_ttl_s must be positive.",
                code="run_api_oidc_incomplete",
                details={"jwks_ttl_s": self.jwks_ttl_s},
            )


# -- claims and verification -------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedClaims:
    """The claims a verifier vouches for. Only these reach `Principal`."""

    subject: str
    roles: frozenset[Role]
    issuer: str
    audience: tuple[str, ...]
    expires_at: float
    email: str = ""

    def identity(self) -> str:
        """The audit identity: the subject, plus the email when the provider supplies one.

        ``sub`` is authoritative (it is the stable, non-reassignable identifier); the email is
        appended because a ledger an auditor reads should name a person where the provider allows it.
        The subject is never *replaced* by the email — a mutable, human-chosen address is not an
        identity.
        """
        return f"{self.subject} ({self.email})" if self.email else self.subject


class JwksSource(Protocol):
    """Where a verification key set comes from."""

    def keys(self, *, force: bool = False) -> Mapping[str, Any]: ...


class ScriptedJwksSource:
    """A JWKS that never touches the network — for tests and for an on-disk deployment.

    ``payload`` may be a JWKS document (``{"keys": [...]}``) or a callable returning one, so a test
    can rotate keys between two verifications without rebuilding the verifier.
    """

    def __init__(self, payload: Mapping[str, Any] | Callable[[], Mapping[str, Any]]) -> None:
        self._payload = payload

    def keys(self, *, force: bool = False) -> Mapping[str, Any]:
        return self._payload() if callable(self._payload) else self._payload


MemoryJwksSource = ScriptedJwksSource


def _decode_segment(segment: str, *, name: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode((segment + padding).encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise AuthenticationError(details={"segment": name}) from exc


@dataclass
class _Claims:
    header: dict[str, Any]
    payload: dict[str, Any]
    signing_input: bytes
    signature: bytes


def _split(token: str) -> _Claims:
    """Decode (not verify) a compact JWS. Any malformation is one AuthenticationError."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError(details={"reason": "not_a_compact_jws"})
    header_raw = _decode_segment(parts[0], name="header")
    payload_raw = _decode_segment(parts[1], name="payload")
    try:
        header = json.loads(header_raw)
        payload = json.loads(payload_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthenticationError(details={"reason": "not_json"}) from exc
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise AuthenticationError(details={"reason": "not_an_object"})
    return _Claims(
        header=header,
        payload=payload,
        signing_input=f"{parts[0]}.{parts[1]}".encode("ascii"),
        signature=_decode_segment(parts[2], name="signature"),
    )


_SUPPORTED_ALGS: Final[frozenset[str]] = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"})


class JwtVerifier:
    """Verify a compact JWS against a JWKS and return the claims it vouches for.

    **The algorithm is chosen by US, not by the token.** ``alg: none`` and HMAC algorithms are
    refused outright, and the token's own ``alg`` must be one of `_SUPPORTED_ALGS`: the classic
    algorithm-confusion attack is a token claiming ``HS256`` and being "verified" with the public key
    as an HMAC secret, and the way to close it is to never accept a symmetric algorithm at all.

    PyJWT is used when present (it is a mature implementation and re-implementing RSA/ECDSA
    verification by hand is how signature checks acquire bugs). When it is absent the verifier
    REFUSES to run — it does not fall back to decoding-without-verifying, which would be the single
    worst failure this module could have.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        role_claim: str = "roles",
        leeway_s: float = DEFAULT_LEEWAY_S,
        clock: Callable[[], float] = time.time,
        allow_missing_subject: bool = False,
    ) -> None:
        try:
            import jwt as _jwt
        except ImportError as exc:  # pragma: no cover - the `durable` extra declares pyjwt
            raise IdentityError(
                "OIDC verification needs PyJWT, which the `durable` extra installs. Re-install "
                "with `pip install 'codna[durable]'`.",
                code="run_api_oidc_unavailable",
                details={"dependency": "pyjwt"},
            ) from exc
        self._jwt = _jwt
        self._issuer = issuer
        self._audience = audience
        self._role_claim = role_claim
        self._leeway = float(leeway_s)
        self._clock = clock
        self._allow_missing_subject = allow_missing_subject

    def verify(self, token: str, *, key_set: Mapping[str, Any]) -> VerifiedClaims:
        header = _split(token).header
        algorithm = header.get("alg")
        if not isinstance(algorithm, str) or algorithm not in _SUPPORTED_ALGS:
            raise AuthenticationError(details={"reason": "unsupported_alg"})
        key = self._select_key(key_set, header)
        try:
            payload = self._jwt.decode(
                token,
                key=key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud"]},
            )
        except self._jwt.ExpiredSignatureError as exc:
            raise AuthenticationError(details={"reason": "expired"}) from exc
        except self._jwt.InvalidAudienceError as exc:
            raise AuthenticationError(details={"reason": "audience"}) from exc
        except self._jwt.InvalidIssuerError as exc:
            raise AuthenticationError(details={"reason": "issuer"}) from exc
        except self._jwt.MissingRequiredClaimError as exc:
            raise AuthenticationError(details={"reason": "missing_claim"}) from exc
        except self._jwt.PyJWTError as exc:
            # One body for a bad signature, a malformed token and anything else PyJWT rejects: the
            # caller learns "not authenticated", never which part was close.
            raise AuthenticationError(details={"reason": "invalid"}) from exc
        return self._claims(payload)

    def _select_key(self, key_set: Mapping[str, Any], header: Mapping[str, Any]) -> Any:
        """Pick the JWKS entry the token's ``kid`` names, or the single entry when it names none.

        Matching on ``kid`` is not optional hardening: verifying a token with the wrong key of the
        same provider is how a token minted for another tenant gets accepted.
        """
        raw_keys = key_set.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise AuthenticationError(details={"reason": "jwks_empty"})
        kid = header.get("kid")
        candidates = [entry for entry in raw_keys if isinstance(entry, dict)]
        if isinstance(kid, str) and kid:
            matching = [entry for entry in candidates if entry.get("kid") == kid]
            if not matching:
                raise AuthenticationError(details={"reason": "unknown_kid"})
            candidates = matching
        if len(candidates) != 1:
            # Ambiguous: the token did not name a key and more than one is published. Guessing is
            # how a signature check becomes a "does any key verify this" check.
            raise AuthenticationError(details={"reason": "ambiguous_key"})
        entry = candidates[0]
        algorithm = self._jwt.algorithms.get_default_algorithms().get(str(entry.get("alg") or "RS256"))
        try:
            return algorithm.from_jwk(json.dumps(entry)) if algorithm is not None else entry
        except Exception as exc:  # noqa: BLE001 - a malformed JWKS entry is a provider problem
            raise AuthenticationError(details={"reason": "jwks_entry_invalid"}) from exc

    def _claims(self, payload: Mapping[str, Any]) -> VerifiedClaims:
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            if not self._allow_missing_subject:
                # An identity provider that omits `sub` cannot be used to attribute an action.
                raise AuthenticationError(details={"reason": "missing_subject"})
            subject = ""
        roles = role_from_claim(payload.get(self._role_claim))
        email = payload.get("email")
        audience = payload.get("aud")
        audiences = (audience,) if isinstance(audience, str) else tuple(
            value for value in (audience or ()) if isinstance(value, str)
        )
        return VerifiedClaims(
            subject=subject.strip(),
            roles=roles,
            issuer=str(payload.get("iss") or ""),
            audience=audiences,
            expires_at=float(payload.get("exp") or 0.0),
            email=email.strip() if isinstance(email, str) else "",
        )


# -- JWKS sources ------------------------------------------------------------------------------


class HttpJwksSource:
    """Fetch the provider's JWKS over HTTPS, caching it for ``ttl`` seconds.

    The one piece of this module that needs a live provider. It is deliberately small, has a hard
    timeout, and converts every transport failure into `ProviderUnavailable` (503) rather than into
    "not authenticated" (401) — an outage must not look like a bad password.
    """

    def __init__(
        self,
        *,
        issuer: str,
        jwks_uri: str = "",
        ttl_s: float = DEFAULT_JWKS_TTL_S,
        clock: Callable[[], float] = time.time,
        timeout_s: float = 10.0,
        opener: Callable[[str, float], Mapping[str, Any]] | None = None,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._jwks_uri = jwks_uri.strip()
        self._ttl = float(ttl_s)
        self._clock = clock
        self._timeout = float(timeout_s)
        self._opener = opener or _fetch_json
        self._lock = threading.Lock()
        self._cached: Mapping[str, Any] | None = None
        self._fetched_at = 0.0

    def keys(self, *, force: bool = False) -> Mapping[str, Any]:
        with self._lock:
            now = self._clock()
            if not force and self._cached is not None and now - self._fetched_at < self._ttl:
                return self._cached
            uri = self._jwks_uri or self._discover()
            try:
                document = self._opener(uri, self._timeout)
            except ProviderUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - every transport failure is one 503
                raise ProviderUnavailable(
                    "could not fetch the identity provider's signing keys.",
                    details={"jwks_uri": uri, "cause": type(exc).__name__},
                ) from exc
            if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
                raise ProviderUnavailable(
                    "the identity provider's JWKS is not a key set.",
                    details={"jwks_uri": uri},
                )
            self._cached = document
            self._fetched_at = now
            return document

    def _discover(self) -> str:
        """``/.well-known/openid-configuration`` → ``jwks_uri``. Only when none was configured."""
        uri = f"{self._issuer}/.well-known/openid-configuration"
        try:
            document = self._opener(uri, self._timeout)
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(
                "could not read the identity provider's discovery document.",
                details={"issuer": self._issuer, "cause": type(exc).__name__},
            ) from exc
        discovered = document.get("jwks_uri") if isinstance(document, dict) else None
        if not isinstance(discovered, str) or not discovered.startswith("https://"):
            raise ProviderUnavailable(
                "the discovery document has no https jwks_uri.",
                details={"issuer": self._issuer},
            )
        self._jwks_uri = discovered
        return discovered


def _fetch_json(url: str, timeout_s: float) -> Mapping[str, Any]:
    """A bounded HTTPS GET of a JSON document. Stdlib only (no new dependency for four lines)."""
    request = urllib.request.Request(  # noqa: S310 - the URL is a configured https issuer
        url, headers={"accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ProviderUnavailable("the identity provider's document is too large.", details={"url": url})
    document = json.loads(raw or b"{}")
    if not isinstance(document, dict):
        raise ProviderUnavailable("the identity provider's document is not an object.", details={"url": url})
    return document


# -- the resolver ------------------------------------------------------------------------------


@dataclass
class IdentityResolver:
    """Authenticate one request. Returns a `Principal` or raises. Never returns "anonymous"."""

    config: IdentityConfig
    _verifier: JwtVerifier | None = field(default=None, init=False, repr=False)
    _jwks: JwksSource | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.mode != AUTH_MODE_OIDC:
            return
        self._verifier = JwtVerifier(
            issuer=self.config.issuer,
            audience=self.config.audience,
            role_claim=self.config.role_claim,
            leeway_s=self.config.leeway_s,
            clock=self.config.clock,
            allow_missing_subject=self.config.allow_missing_subject,
        )
        if self.config.jwks_json is not None:
            self._jwks = ScriptedJwksSource(self.config.jwks_json)
        elif self.config.jwks_fetcher is not None:
            self._jwks = ScriptedJwksSource(lambda: self.config.jwks_fetcher and self.config.jwks_fetcher(
                self.config.jwks_uri or self.config.issuer
            ))
        else:
            self._jwks = HttpJwksSource(
                issuer=self.config.issuer,
                jwks_uri=self.config.jwks_uri,
                ttl_s=self.config.jwks_ttl_s,
                clock=self.config.clock,
            )

    def resolve(self, header: str | None) -> Principal:
        """The `Principal` for this request, or an `AuthenticationError`."""
        token = bearer_from_header(header)
        if self.config.mode == AUTH_MODE_LOCAL_TRUST:
            # Deliberately does not look at the header AT ALL: a local-trust server ignores
            # credentials rather than mixing two identity models on one socket.
            return self._local_trust()
        if token is None:
            raise AuthenticationError(details={"reason": "no_bearer_token"})
        if self.config.mode == AUTH_MODE_TOKEN:
            return self._local_token(token)
        return self._oidc(token)

    # -- per mode ------------------------------------------------------------

    def _local_trust(self) -> Principal:
        if self.config.issuer:
            # Belt and braces: `IdentityConfig` refuses this combination at construction, and the
            # resolver refuses it again because the config could be built by a caller that skipped
            # validation (a dataclass replace, a deserialized mapping).
            raise IdentityError(
                "local-trust cannot be used with an OIDC issuer.",
                code="run_api_local_trust_conflicts_with_oidc",
                details={"issuer": self.config.issuer},
            )
        return Principal(
            identity=self.config.local_identity or local_owner_identity(),
            roles=frozenset(self.config.local_roles),
            source=LOCAL_TRUST_SOURCE,
        )

    def _local_token(self, token: str) -> Principal:
        import hmac

        expected = self.config.local_token
        # Both sides encoded first: `compare_digest` accepts str only when both are ASCII, so a
        # non-ASCII header would raise instead of failing closed.
        if not hmac.compare_digest(expected.encode("utf-8"), token.encode("utf-8")):
            raise AuthenticationError(details={"reason": "token_mismatch"})
        return Principal(
            identity=self.config.local_identity or local_owner_identity(),
            roles=frozenset(self.config.local_roles),
            source=LOCAL_TOKEN_SOURCE,
        )

    def _oidc(self, token: str) -> Principal:
        assert self._verifier is not None and self._jwks is not None  # narrowed by __post_init__
        claims = self._verifier.verify(token, key_set=self._jwks.keys())
        return Principal(
            identity=claims.identity(),
            roles=claims.roles,
            source=OIDC_SOURCE,
        )

    @property
    def mode(self) -> str:
        return self.config.mode
