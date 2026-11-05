"""`DurableConfig` — the one place a durable run's coordinates are decided.

Every durable process (worker, Run API, CLI, the test harness) reads its address, namespace,
per-host task queue, on-disk roots, sidecar URL and graph engine from here, so a run started by
the CLI is findable by the worker that will execute it. Values come from a small, explicit set of
environment overrides plus defaults; anything malformed fails fast with a structured error rather
than being silently coerced into a default that would send work to the wrong queue.

Two deliberate constraints:

* **No `temporalio` import.** This module is part of the base CLI surface (``codna run`` must be
  able to *report* "durable support isn't installed"), so it may only touch stdlib plus
  ``codna.runtime``.
* **The task queue is per-host by design.** ``codna.agent.<hostname>`` means a run only executes
  on the machine that owns its checkout and its sidecar. A host being down stalls its runs; that
  is a property of the design (D2), not an accident, until hosted workers (D6) add pool queues.

``GRAPH_VERSION`` lives here rather than in ``graph.py`` so ``build_id`` can be computed without
importing ``langgraph``; ``codna.durable.graph`` re-exports it for the plan's documented location.
A change to the task graph must bump it, which produces a new ``build_id`` and therefore a new
pinned worker version — old runs finish on the old one.

TEMPORAL PROFILES — local dev server vs Temporal Cloud
======================================================
A durable run executes against one of two very different servers, and the difference is not only an
address: the local profile is ``temporal server start-dev`` on loopback (no TLS, no credential,
codna starts and stops it), while the cloud profile is Temporal Cloud (TLS, an API key, a namespace
that names the tenant, and no local process at all). So the profile is resolved EXPLICITLY, from
``CODNA_TEMPORAL_PROFILE`` or — with no override — from the presence of the cloud variables:

* **Nothing cloud-shaped is set** → ``local``. This is the historical path and it is unchanged: same
  loopback address, same ``codna`` namespace, same dev server, no TLS, no key.
* **Any of ``TEMPORAL_ADDRESS`` / ``TEMPORAL_ENDPOINT`` / ``TEMPORAL_API_KEY`` / ``TEMPORAL_NAMESPACE``
  is set** → ``cloud``, and ALL of an endpoint, a key and a namespace become required. A partially
  configured cloud is a structured ``temporal_cloud_config_incomplete`` refusal naming the missing
  VARIABLES — never a quiet fall back to loopback, which would send a run to a server that will
  never execute it while the operator believes it is running in the cloud.
* ``CODNA_TEMPORAL_PROFILE=local`` on a host whose shell exports ``TEMPORAL_*`` for the Temporal CLI
  is a deliberate opt-out, not a silent default.

**The API key is never a field of this dataclass.** ``DurableConfig`` is frozen and has a generated
``repr``, so a key stored here would be printed by any ``print(config)``, any exception that formats
a config, and any debugger dump. What the config carries is the NAME of the environment variable;
`DurableConfig.temporal_api_key` reads its value at connect time, and the caller keeps it in the one
local variable it hands to ``Client.connect`` — never in a log line, a banner or a detail map.

The endpoint is also validated against the profile's shape: Temporal Cloud serves TLS on 7233, so a
``http://`` scheme (a copy-paste from the Cloud UI's "connect" snippet), an embedded path, or a bare
hostname is refused here rather than becoming an opaque gRPC failure at the first RPC.
"""
from __future__ import annotations

import os
import re
import socket
from dataclasses import dataclass
from typing import Final, Literal, Mapping
from pathlib import Path

from ..runtime.config import RuntimeConfig, build_runtime_paths, resolve_runtime_config
from .credentials import (
    ENV_CLIENT_CERT,
    ENV_CLIENT_KEY,
    ConnectOptions as _CredentialConnectOptions,
    TemporalAuthError,
    TemporalCredentials,
    resolve_temporal_credentials,
)
from .pool import ENV_POOL, ENV_POOL_STATE, ENV_POOL_WORKERS, PoolError, PoolSettings
from .pool import pool_settings_from_env
from .version import GRAPH_VERSION, build_id_for

__all__ = [
    "ENV_CLIENT_CERT",
    "ENV_CLIENT_KEY",
    "ENV_POOL",
    "ENV_POOL_STATE",
    "ENV_POOL_WORKERS",
    "DEFAULT_TEMPORAL_ADDRESS",
    "DEFAULT_TEMPORAL_HOST",
    "DEFAULT_TEMPORAL_PORT",
    "DEFAULT_NAMESPACE",
    "ENV_CLOUD_API_KEY",
    "ENV_CLOUD_ENDPOINT",
    "ENV_CLOUD_NAMESPACE",
    "ENV_CLOUD_PROFILE",
    "GRAPH_VERSION",
    "TEMPORAL_PROFILES",
    "DurableConfig",
    "DurableConfigError",
    "DurablePaths",
    "GraphEngine",
    "PoolSettings",
    "TemporalProfile",
    "build_id_for",
    "load_durable_config",
    "parse_profile",
    "read_api_key",
    "sanitize_hostname",
    "split_temporal_endpoint",
    "task_queue_for_host",
    "temporal_connect_options",
]

GraphEngine = Literal["langgraph", "plain"]
GRAPH_ENGINES: Final[tuple[GraphEngine, ...]] = ("langgraph", "plain")

# WHICH SERVER a durable process talks to. `local` is the loopback dev server codna starts and stops
# itself; `cloud` is Temporal Cloud, which needs TLS, an API key and a tenant namespace and has no
# local process at all. The profile decides which of those are REQUIRED, so it is a resolved value on
# the config rather than a second boolean that could disagree with the address it accompanies.
TemporalProfile = Literal["local", "cloud"]
TEMPORAL_PROFILES: Final[tuple[TemporalProfile, ...]] = ("local", "cloud")

# The version of the task graph lives in `version.py` (a leaf module) and is re-exported above for
# callers that already have `config`. `graph.py` must import it from `version` directly: this module
# reaches `httpx` → `urllib.request`, which the Temporal workflow sandbox refuses.

DEFAULT_TEMPORAL_HOST: Final = "127.0.0.1"
DEFAULT_TEMPORAL_PORT: Final = 7233
DEFAULT_TEMPORAL_ADDRESS: Final = f"{DEFAULT_TEMPORAL_HOST}:{DEFAULT_TEMPORAL_PORT}"
DEFAULT_NAMESPACE: Final = "codna"

# A hostname becomes part of a task queue name, which Temporal accepts as a free-form string but
# which we also use as a filesystem/CLI token. Keep it to the same character class as session ids.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

ENV_ADDRESS: Final = "CODNA_TEMPORAL_ADDRESS"
ENV_NAMESPACE: Final = "CODNA_TEMPORAL_NAMESPACE"
ENV_TASK_QUEUE: Final = "CODNA_DURABLE_TASK_QUEUE"
ENV_ENGINE: Final = "CODNA_DURABLE_ENGINE"
ENV_RUNS_ROOT: Final = "CODNA_DURABLE_RUNS_ROOT"

# The Temporal CLI's own variable names for a Cloud connection, adopted deliberately: the soak and
# the gated test read the same names, so ONE export configures every entry point, and a machine that
# already has them set for `temporal` needs no codna-specific copies.
ENV_CLOUD_ENDPOINT: Final = "TEMPORAL_ENDPOINT"
# `TEMPORAL_ADDRESS` is the Temporal SDK's general "where is the server" variable. Accepted as an
# endpoint source for the cloud profile (an operator who exports one address should not have to
# export it again under a second name), but `TEMPORAL_ENDPOINT` wins when both are present.
ENV_CLOUD_ADDRESS: Final = "TEMPORAL_ADDRESS"
ENV_CLOUD_API_KEY: Final = "TEMPORAL_API_KEY"
ENV_CLOUD_NAMESPACE: Final = "TEMPORAL_NAMESPACE"
# `local` | `cloud`. Absent means "infer from the cloud variables": set ⇒ cloud, absent ⇒ local.
ENV_CLOUD_PROFILE: Final = "CODNA_TEMPORAL_PROFILE"

# Every variable whose presence means "this host is configured for, or at least attempting, Temporal
# Cloud". Used only to DECIDE whether cloud configuration is being attempted; the requirements
# themselves are checked one by one in `_resolve_profile`.
_CLOUD_VARIABLES: Final[tuple[str, ...]] = (
    ENV_CLOUD_ENDPOINT,
    ENV_CLOUD_ADDRESS,
    ENV_CLOUD_API_KEY,
    ENV_CLOUD_NAMESPACE,
    ENV_CLOUD_PROFILE,
    # D6: mTLS is the other Cloud credential shape, so configuring it is also an attempt at Cloud.
    # Without these, a certificate-only setup would be inferred as `local` and send a pool worker to
    # a loopback server nobody started.
    ENV_CLIENT_CERT,
    ENV_CLIENT_KEY,
)

# Temporal Cloud serves gRPC over TLS on 7233. A different port is not refused (Temporal Cloud has
# never published another one, but refusing a future port would be guessing); only the SCHEME and the
# shape are validated, because `https://...` is what the Cloud UI's snippets are copied as.
_CLOUD_DEFAULT_PORT: Final = 7233


class DurableConfigError(RuntimeError):
    """Invalid durable configuration. Always carries a machine-readable code."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})


@dataclass(frozen=True)
class DurablePaths:
    """On-disk layout for durable state, all under the shared ``~/.codna`` runtime root."""

    root: Path
    temporal_dir: Path
    dev_db_path: Path
    dev_server_bin_dir: Path
    runs_root: Path
    logs_dir: Path

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / validate_run_id(run_id)

    def ledger_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "ledger.ndjson"

    def parquet_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "ledger.parquet"


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    """Reject a run id that could escape the runs root or collide with another run's files.

    Run ids also become workflow ids and git ref components, so the accepted class is
    deliberately narrow.
    """
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise DurableConfigError(
            "invalid run id: must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
            code="invalid_run_id",
            details={"run_id": run_id if isinstance(run_id, str) else type(run_id).__name__},
        )
    if ".." in run_id:
        raise DurableConfigError(
            "invalid run id: '..' is not allowed",
            code="invalid_run_id",
            details={"run_id": run_id},
        )
    return run_id


def sanitize_hostname(raw: str) -> str:
    """Reduce a machine hostname to a task-queue-safe token.

    macOS reports ``Angel's MacBook.local`` or ``angels-mbp.lan``; neither can carry an apostrophe
    into a queue name that is also used as a filesystem token. Anything outside the safe class
    collapses to ``-``; an empty result is refused rather than replaced with a guess.
    """
    cleaned = _UNSAFE_CHARS.sub("-", (raw or "").strip()).strip("-.")
    if not cleaned:
        raise DurableConfigError(
            "cannot derive a task queue from an empty hostname.",
            code="invalid_hostname",
            details={"hostname": raw},
        )
    return cleaned[:63]


def task_queue_for_host(hostname: str) -> str:
    return f"codna.agent.{sanitize_hostname(hostname)}"


def _env(env: Mapping[str, str], name: str) -> str | None:
    """Read an override, distinguishing ABSENT from EMPTY.

    ``env.get(name) or default`` would treat ``CODNA_TEMPORAL_ADDRESS=""`` as "not set" and quietly
    use the loopback default — which is how a Kubernetes template that renders an empty variable
    sends a worker to the wrong server instead of failing. An empty override is a configuration
    error, so it is returned as ``""`` and the per-field validators refuse it.
    """
    if name not in env:
        return None
    return env[name]


def _split_address(raw: str) -> tuple[str, int]:
    value = (raw or "").strip()
    if not value:
        raise DurableConfigError(
            f"{ENV_ADDRESS} must be host:port.", code="invalid_temporal_address", details={"value": raw}
        )
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise DurableConfigError(
            f"{ENV_ADDRESS} must be host:port (got {value!r}).",
            code="invalid_temporal_address",
            details={"value": value},
        )
    try:
        port = int(port_text)
    except ValueError as exc:
        raise DurableConfigError(
            f"{ENV_ADDRESS} port must be an integer (got {port_text!r}).",
            code="invalid_temporal_address",
            details={"value": value},
        ) from exc
    if not 1 <= port <= 65535:
        raise DurableConfigError(
            f"{ENV_ADDRESS} port must be between 1 and 65535 (got {port}).",
            code="invalid_temporal_address",
            details={"value": value, "port": port},
        )
    return host.strip("[]"), port


def parse_profile(raw: str | None) -> TemporalProfile | None:
    """Parse a profile name, refusing anything unrecognized (or ``None`` for "not given").

    No silent default on a typo: ``CODNA_TEMPORAL_PROFILE=clod`` must not fall through to the local
    inference, because the inference would then decide from whatever ``TEMPORAL_*`` variables the
    shell happens to hold — which is exactly the ambiguity the explicit profile exists to remove.
    Case is folded (``Cloud`` is what a person types) but nothing else is.
    """
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value not in TEMPORAL_PROFILES:
        raise DurableConfigError(
            f"{ENV_CLOUD_PROFILE} must be one of {', '.join(TEMPORAL_PROFILES)} (got {raw!r}).",
            code="invalid_temporal_profile",
            details={"value": raw, "allowed": list(TEMPORAL_PROFILES)},
        )
    return value  # type: ignore[return-value]  # narrowed by the membership check above


def split_temporal_endpoint(raw: str, *, variable: str = ENV_CLOUD_ENDPOINT) -> tuple[str, int]:
    """``host[:port]`` (optionally ``scheme://host[:port]``) → ``(host, port)``, refusing junk.

    A Cloud endpoint arrives in whichever form the operator copied it: ``codna.cihxb.tmprl.cloud:7233``
    from the SDK snippet, ``https://codna.cihxb.tmprl.cloud`` from the Cloud UI. Both name the same
    server, so both are accepted — but only those two, and only for ``grpc``/``https``.

    What is REFUSED is everything that would otherwise turn into an opaque gRPC error at the first
    RPC: an empty value, an embedded path or query (a copied URL with ``/`` on the end), userinfo,
    and any other scheme (``http://`` is a downgrade a Cloud client must never silently perform — an
    API key would travel in the clear). A bare hostname is NOT refused: port 7233 is filled in,
    because that is the only port Temporal Cloud serves.
    """
    value = (raw or "").strip()
    if not value:
        raise DurableConfigError(
            f"{variable} must be host[:port] for Temporal Cloud.",
            code="invalid_temporal_endpoint",
            details={"variable": variable, "value": ""},
        )
    host_part = value
    if "://" in value:
        scheme, _, remainder = value.partition("://")
        scheme = scheme.strip().lower()
        if scheme not in ("https", "grpc", "grpcs", "tls"):
            raise DurableConfigError(
                f"{variable} uses the scheme {scheme!r}; Temporal Cloud requires TLS "
                f"(use https://, grpcs:// or a bare host:port).",
                code="invalid_temporal_endpoint",
                details={"variable": variable, "scheme": scheme},
            )
        host_part = remainder
    for forbidden, reason in (("/", "a path"), ("?", "a query"), ("#", "a fragment"), ("@", "userinfo")):
        if forbidden in host_part:
            raise DurableConfigError(
                f"{variable} must not carry {reason} (got {value!r}).",
                code="invalid_temporal_endpoint",
                details={"variable": variable, "value": value},
            )
    host = host_part
    port = _CLOUD_DEFAULT_PORT
    if ":" in host_part:
        host, _, port_text = host_part.rpartition(":")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise DurableConfigError(
                f"{variable} port must be an integer (got {port_text!r}).",
                code="invalid_temporal_endpoint",
                details={"variable": variable, "value": value},
            ) from exc
        if not 1 <= port <= 65535:
            raise DurableConfigError(
                f"{variable} port must be between 1 and 65535 (got {port}).",
                code="invalid_temporal_endpoint",
                details={"variable": variable, "port": port},
            )
    host = host.strip("[]").strip()
    if not host or "." not in host:
        # A bare `localhost`/`codna` here is almost always a local address pasted into the cloud
        # profile; it would resolve to nothing on the public internet.
        raise DurableConfigError(
            f"{variable} must name a server host, not {host!r}.",
            code="invalid_temporal_endpoint",
            details={"variable": variable, "host": host, "value": value},
        )
    return host, port


def read_api_key(
    name: str | None,
    *,
    env: Mapping[str, str] | None = None,
    profile: TemporalProfile = "cloud",
) -> str | None:
    """Read a named API key from ``env`` (default ``os.environ``), refusing a blank one.

    ``None`` means "this profile has no key" only when NO variable name is given. A named variable
    that is missing or blank is a refusal, not a ``None``: by the time a connection is being made,
    an empty credential means the process's environment changed after the config was resolved, and
    continuing would produce an opaque "unauthorised" from the server instead of a named cause.
    """
    if name is None:
        return None
    resolved_env = os.environ if env is None else env
    value = resolved_env.get(name)
    if not isinstance(value, str) or not value.strip():
        raise DurableConfigError(
            f"{name} is not set, so this {profile} Temporal connection has no credential.",
            code="temporal_cloud_config_incomplete",
            details={"missing": [name], "profile": profile},
        )
    return value.strip()


# RE-EXPORTED from `credentials` (D6 moved the definition there, because the resolver that builds
# these options lives in that module and `config` imports it). Kept as a module-level name here so
# every historical import site — including `config.ConnectOptions` in the D2 tests — keeps working;
# a test asserts the two names are the SAME object, not two classes that happen to agree.
ConnectOptions = _CredentialConnectOptions


def temporal_connect_options(
    *,
    tls: bool,
    api_key_env: str | None,
    env: Mapping[str, str] | None = None,
    profile: TemporalProfile = "cloud",
) -> dict[str, object]:
    """The ``Client.connect`` keyword arguments for an API-KEY connection.

    An EMPTY dict for the local profile is the contract, not an economy: the caller then calls
    ``Client.connect(address, namespace=...)`` exactly as before, so nothing about the dev-server
    path changed when cloud support was added. The key is read HERE (from the environment) and goes
    straight into the returned mapping, which the caller passes to ``Client.connect`` — never through
    a log line, a banner, a detail map or a dataclass field.

    ``api_key`` is passed with ``tls=True`` explicitly rather than relying on the SDK's inference
    ("TLS is enabled automatically when an api_key is provided"), because the SDK also lets
    ``rpc_metadata["authorization"]`` suppress the key — and an operator reading this code should
    not have to know that to be sure a Cloud connection is TLS.

    **Prefer `DurableConfig.connect_options`.** This is the API-KEY-ONLY shape, kept because it is the
    D2 contract many call sites and tests were written against; a connection that must honour mTLS
    goes through the config, which hands the decision to `credentials.resolve_temporal_credentials`.
    """
    if not tls and api_key_env is None:
        return ConnectOptions()
    options: dict[str, object] = ConnectOptions(tls=True)
    if api_key_env is not None:
        options["api_key"] = read_api_key(api_key_env, env=env, profile=profile)
    return options


def _resolve_profile(env: Mapping[str, str]) -> TemporalProfile:
    """Which server profile this environment configures, refusing a PARTIAL cloud setup.

    The inference (rather than defaulting to ``local``) is what makes "I exported a TEMPORAL_API_KEY
    but forgot the namespace" a named refusal instead of a run that talks to a loopback dev server
    nobody started. An explicit ``CODNA_TEMPORAL_PROFILE=local`` overrides the inference, which is
    the escape hatch for a host that exports the variables for the Temporal CLI.
    """
    explicit = parse_profile(_env(env, ENV_CLOUD_PROFILE))
    if explicit is not None:
        return explicit
    attempted = [name for name in _CLOUD_VARIABLES if _env(env, name) not in (None, "")]
    if attempted:
        return "cloud"
    return "local"


def _require_cloud_coordinates(env: Mapping[str, str]) -> tuple[str, int, str]:
    """The Cloud coordinates (host, port, namespace), all required TOGETHER.

    Called only for the cloud profile. The credential's PRESENCE is validated here, but its VALUE is
    deliberately not returned or stored — see `DurableConfig.temporal_api_key`, which reads it at
    connect time. The emptiness rule matches every other override: a variable that is PRESENT but
    blank is a misconfiguration (a template that rendered an empty secret), not "unset", so it
    counts as missing and is named in ``missing``.

    A CREDENTIAL is satisfied by an API key OR by a complete mTLS pair (D6): Temporal Cloud accepts
    either, so requiring the key specifically would refuse a correctly configured certificate setup.
    Half an mTLS pair is not a credential, and is named as the missing variable rather than silently
    falling back to "no credential" — the same rule the resolver applies at connect time.
    """
    endpoint_raw = _env(env, ENV_CLOUD_ENDPOINT)
    if endpoint_raw in (None, ""):
        endpoint_raw = _env(env, ENV_CLOUD_ADDRESS)
    api_key = _env(env, ENV_CLOUD_API_KEY)
    namespace = _env(env, ENV_CLOUD_NAMESPACE)
    client_cert = _env(env, ENV_CLIENT_CERT)
    client_key = _env(env, ENV_CLIENT_KEY)

    mtls_complete = bool(client_cert) and bool(client_key)
    credential_names = (ENV_CLOUD_API_KEY, ENV_CLIENT_CERT, ENV_CLIENT_KEY)

    missing: list[str] = []
    if endpoint_raw in (None, ""):
        missing.append(f"{ENV_CLOUD_ENDPOINT} (or {ENV_CLOUD_ADDRESS})")
    if not mtls_complete and api_key in (None, ""):
        # Name the half-configured certificate when that is what the operator clearly attempted;
        # otherwise name the key, which is the shape most setups use.
        if bool(client_cert) != bool(client_key):
            missing.append(ENV_CLIENT_KEY if not client_key else ENV_CLIENT_CERT)
        else:
            missing.append(ENV_CLOUD_API_KEY)
    if namespace in (None, ""):
        missing.append(ENV_CLOUD_NAMESPACE)
    if missing:
        raise DurableConfigError(
            f"Temporal Cloud is selected but {', '.join(missing)} is not set. Cloud needs an "
            "endpoint, a namespace and a credential (an API key, or a client certificate plus its "
            "key) together — there is no fallback to a local server.",
            code="temporal_cloud_config_incomplete",
            details={
                "missing": missing,
                "profile": "cloud",
                "profile_env": ENV_CLOUD_PROFILE,
                # Variable NAMES only. A detail map is printed by the CLI and pasted into tickets.
                "configured": sorted(
                    name
                    for name in (
                        ENV_CLOUD_ENDPOINT,
                        ENV_CLOUD_ADDRESS,
                        ENV_CLOUD_NAMESPACE,
                        *credential_names,
                    )
                    if _env(env, name) not in (None, "")
                ),
            },
        )
    assert endpoint_raw is not None and namespace is not None
    host, port = split_temporal_endpoint(endpoint_raw)
    resolved_namespace = _require_token(ENV_CLOUD_NAMESPACE, namespace, code="invalid_namespace")
    return host, port, resolved_namespace


def _require_token(label: str, raw: str, *, code: str) -> str:
    value = (raw or "").strip()
    if not _SAFE_TOKEN.match(value):
        raise DurableConfigError(
            f"{label} must match ^[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}$ (got {value!r}).",
            code=code,
            details={"value": value},
        )
    return value


def _parse_engine(raw: str | None) -> GraphEngine:
    """Parse the graph engine name, refusing anything unrecognized.

    No silent default on a typo: ``CODNA_DURABLE_ENGINE=langgraphh`` must not quietly select the
    plain engine, because the two engines produce different workflow history and a run started under
    the wrong one cannot be replayed by the worker that picks it up.
    """
    value = (raw or "langgraph").strip().lower()
    if value not in GRAPH_ENGINES:
        raise DurableConfigError(
            f"graph engine must be one of {', '.join(GRAPH_ENGINES)} (got {value!r}).",
            code="invalid_graph_engine",
            details={"value": value, "allowed": list(GRAPH_ENGINES)},
        )
    return value  # type: ignore[return-value]  # narrowed by the membership check above


@dataclass(frozen=True)
class DurableConfig:
    """Resolved durable coordinates. Immutable: a changed setting is a new config object.

    ``profile`` decides what the coordinates MEAN, so it is part of the value: ``local`` is a
    loopback dev server codna owns (no TLS, no credential), ``cloud`` is Temporal Cloud (TLS, API
    key, tenant namespace, no local process). The two ``..._requires_*`` properties below are the
    single answer to "does this config need a credential / a TLS connection", so no caller has to
    re-derive it from the address and get it wrong.

    ``api_key_env`` is the NAME of the variable holding the key — never the key. See the module
    docstring: a frozen dataclass with a generated ``repr`` is the last place a secret belongs.
    """

    address: str
    temporal_host: str
    temporal_port: int
    namespace: str
    task_queue: str
    hostname: str
    graph_engine: GraphEngine
    build_id: str
    graph_version: int
    paths: DurablePaths
    sidecar_url: str
    sidecar_port: int
    cli_version: str
    # `local` by default so every existing constructor of this dataclass (tests, callers) keeps the
    # historical dev-server semantics without naming a profile.
    profile: TemporalProfile = "local"
    # The env var a cloud connection reads its API key from, or None for the local profile.
    api_key_env: str | None = None
    # D6: the POOL this process belongs to, or None for the historical per-host queue. A pool makes
    # `task_queue` the shared `codna.pool.<name>` and carries the fleet size plus the shared-state
    # declaration that made more than one worker legal.
    pool: PoolSettings | None = None

    @property
    def is_cloud(self) -> bool:
        return self.profile == "cloud"

    @property
    def requires_tls(self) -> bool:
        """Temporal Cloud is TLS-only; the local dev server speaks cleartext on loopback."""
        return self.is_cloud

    @property
    def requires_api_key(self) -> bool:
        """Whether this profile needs a credential. Kept as a NAME for compatibility.

        Temporal Cloud accepts an API key OR an mTLS client certificate (E/D6), so "requires an API
        key" is only literally true for the api_key shape. What every caller actually needs to know —
        "may this connection be made with no credential at all?" — is answered here, and
        ``credential_mode`` says which shape will be used. Callers that branch on the SHAPE use
        ``credentials()``; nothing should branch on this to decide which value to read.
        """
        return self.is_cloud

    @property
    def owns_local_server(self) -> bool:
        """Whether codna may start/stop a Temporal server for this config.

        Only the local profile: there is no dev server to start for Cloud, and a command that tried
        would either download a CLI it will not use or mask a connection failure behind a successful
        local start.
        """
        return not self.is_cloud

    def temporal_api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        """The API key VALUE, read at call time — or None when this connection has no key.

        Returns ``None`` for the local profile AND for a cloud profile configured with mTLS: the key
        is one of two credential shapes (`credentials()` decides between them), so reporting a KEY
        for an mTLS connection would be a lie that a caller might act on.
        """
        resolved = os.environ if env is None else env
        if not self.is_cloud:
            return None
        if self.api_key_env is None:
            return None
        try:
            credentials = resolve_temporal_credentials(resolved)
        except TemporalAuthError:
            # An unusable credential set is reported by `connect_options`, which is where a caller
            # learns the cause; this accessor answers "is there a key" and has no place to fail.
            return None
        return credentials.api_key

    def credentials(self, env: Mapping[str, str] | None = None) -> TemporalCredentials | None:
        """The resolved credential SHAPE for this profile, or None for the local dev server.

        A named seam rather than inlining `resolve_temporal_credentials` at each call site: the
        worker, the CLI client and a deploy preflight must all agree about which shape is in use, and
        one function they share is the only way that stays true.
        """
        if not self.is_cloud:
            return None
        return resolve_temporal_credentials(os.environ if env is None else env)

    def connect_options(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        """The keyword arguments that make ``Client.connect`` honour this profile.

        The ONE place the TLS/credential decision is made, so ``codna worker`` and every
        ``codna durable`` command cannot disagree about whether they are talking to a TLS server with
        a credential. Empty for the local profile — ``Client.connect(address, namespace=...)`` is
        called exactly as it always was, which is what keeps the dev-server path unchanged.

        Cloud gets BOTH shapes through here (D6): an API key, or an mTLS ``TLSConfig``. A
        ``TemporalAuthError`` is translated into this module's error type so a caller sees one
        exception family with one code style.
        """
        if not self.is_cloud:
            return {}
        resolved = os.environ if env is None else env
        try:
            return resolve_temporal_credentials(resolved).connect_kwargs()
        except TemporalAuthError as exc:
            raise DurableConfigError(str(exc), code=exc.code, details=exc.details) from exc

    @property
    def dev_db_path(self) -> Path:
        return self.paths.dev_db_path

    @property
    def runs_root(self) -> Path:
        return self.paths.runs_root

    @property
    def needs_langgraph(self) -> bool:
        """True when the selected engine requires the LangGraph plugin (and thus the extra)."""
        return self.graph_engine == "langgraph"

    def run_dir(self, run_id: str) -> Path:
        return self.paths.run_dir(run_id)

    def ledger_path(self, run_id: str) -> Path:
        return self.paths.ledger_path(run_id)

    def workflow_id(self, run_id: str) -> str:
        """Stable workflow id for a task workflow (never reused, never regenerated)."""
        return f"task:{validate_run_id(run_id)}"

    def run_workflow_id(self, run_id: str) -> str:
        return f"run:{validate_run_id(run_id)}"


def _resolve_root(env: Mapping[str, str]) -> Path:
    """Resolve ``~/.codna`` (or the ``CODNA_RUNTIME_ROOT`` override).

    Reuses `codna.runtime.config.build_runtime_paths` rather than recomputing the root, so durable
    state lands beside the runtime the CLI already manages instead of inventing a second layout.
    """
    return build_runtime_paths().root


def load_durable_config(
    *,
    env: Mapping[str, str] | None = None,
    runtime: RuntimeConfig | None = None,
    hostname: str | None = None,
    profile: str | None = None,
) -> DurableConfig:
    """Resolve the durable config from the environment, defaulting to a loopback dev server.

    ``runtime`` is injectable so a caller that already resolved the shared runtime config (the
    CLI does, for the sidecar port) does not recompute its fingerprints.

    ``profile`` is the CLI's ``--temporal-profile``: an EXPLICIT choice that outranks both
    ``CODNA_TEMPORAL_PROFILE`` and the inference from the ``TEMPORAL_*`` variables. It is passed as
    a parameter rather than folded into ``env`` so a flag can never be mistaken for something the
    environment said (a test that injects a mapping must be able to prove the inference alone).

    With the ``local`` profile this function behaves EXACTLY as it always has: the same loopback
    address, ``codna`` namespace, dev-server paths and no credential. The cloud branch is the only
    new one, and it validates its four coordinates as a set before returning.
    """
    resolved_env = os.environ if env is None else env
    runtime_config = runtime if runtime is not None else resolve_runtime_config()

    effective_profile = parse_profile(profile) or _resolve_profile(resolved_env)
    api_key_env: str | None = None
    if effective_profile == "cloud":
        host, port, namespace = _require_cloud_coordinates(resolved_env)
        api_key_env = ENV_CLOUD_API_KEY
    else:
        address = _env(resolved_env, ENV_ADDRESS)
        host, port = _split_address(address if address is not None else DEFAULT_TEMPORAL_ADDRESS)
        namespace = _require_token(
            ENV_NAMESPACE,
            _env(resolved_env, ENV_NAMESPACE) or DEFAULT_NAMESPACE,
            code="invalid_namespace",
        )

    resolved_hostname = sanitize_hostname(hostname or socket.gethostname())
    # D6: a POOL replaces the per-host queue. Resolved before the queue override is read so
    # `CODNA_DURABLE_TASK_QUEUE` remains the most specific statement (it is the operator pinning the
    # exact queue) while `CODNA_DURABLE_POOL` is the fleet-level one. A pool refusal (`PoolError`)
    # becomes this module's error type: `pool_state_not_shared` in particular must reach an operator
    # as a structured startup failure, not a traceback.
    try:
        pool = pool_settings_from_env(resolved_env)
    except PoolError as exc:
        raise DurableConfigError(str(exc), code=exc.code, details=exc.details) from exc
    queue_override = _env(resolved_env, ENV_TASK_QUEUE)
    if queue_override is not None:
        task_queue = _require_token(ENV_TASK_QUEUE, queue_override, code="invalid_task_queue")
    elif pool is not None:
        task_queue = pool.task_queue
    else:
        task_queue = task_queue_for_host(resolved_hostname)

    engine = _parse_engine(_env(resolved_env, ENV_ENGINE))

    root = _resolve_root(resolved_env)
    runs_override = _env(resolved_env, ENV_RUNS_ROOT)
    runs_root = Path(runs_override).expanduser() if runs_override else root / "runs"
    temporal_dir = root / "temporal"
    paths = DurablePaths(
        root=root,
        temporal_dir=temporal_dir,
        dev_db_path=temporal_dir / "dev.db",
        dev_server_bin_dir=temporal_dir / "bin",
        runs_root=runs_root,
        logs_dir=root / "logs",
    )

    return DurableConfig(
        address=f"{host}:{port}",
        temporal_host=host,
        temporal_port=port,
        namespace=namespace,
        task_queue=task_queue,
        hostname=resolved_hostname,
        graph_engine=engine,
        build_id=build_id_for(),
        graph_version=GRAPH_VERSION,
        paths=paths,
        sidecar_url=runtime_config.sidecar_url,
        sidecar_port=runtime_config.sidecar_port,
        cli_version=runtime_config.codna_cli_version,
        profile=effective_profile,
        api_key_env=api_key_env,
        pool=pool,
    )
