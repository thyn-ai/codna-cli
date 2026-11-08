"""`codna serve`: the loopback Run API the IDE drives a run through.

Split out of `run_cli.py` at the repository's 1000-line ceiling (`test_modularity.py`), on a real
boundary: this is the one durable command that does not START work — it serves a surface over work
that already exists. A reviewer asking "what does the IDE talk to" reads this file and nothing else.

Everything the surface enforces (loopback only, the 0600 bearer token, an identity on every mutation)
lives in `run_api`; this is the translator that turns flags into `RunApiSettings` and a listen socket,
plus the `--no-temporal` backend that answers "this server cannot start runs" explicitly rather than
pretending, and states what it CAN do before a caller has to find out by failing.

Not imported at module scope by anything that must work without the durable extra: `run_api` is
imported inside `cmd_serve`.
"""
from __future__ import annotations

import sys
from typing import Any

from .run_api_backend import RunBackend, RunCapabilities, ledger_only_capabilities

__all__ = ["add_serve_flags", "cmd_serve"]


# -- serve (the Run API) -------------------------------------------------------------------


def _cli_module() -> Any:
    """The CLI module, resolved at CALL time.

    `serve_cli` is imported BY `run_cli` (for the registration), and `run_cli` is imported by `cli` —
    so a module-scope `from . import cli` here closes a cycle that only appears when `serve_cli` is
    the first of the three to be imported. Reaching the module object per call keeps one definition
    (a test that swaps `cli._config` still sees the change) without the import-order trap.
    """
    from . import cli as module

    return module


def _error(message: str, *, code: str, details: dict[str, object] | None = None, hint: str = "") -> Any:
    """The CLI's error type, reached through `run_cli` at CALL time.

    A module-scope import would close a cycle (`run_cli` imports this module for the registration), so
    it is resolved per call. Reaching the module object rather than the class also keeps ONE
    definition: a test that swaps `cli.DurableCliError` sees the change here.
    """
    from .run_cli import _error as _make

    return _make(message, code=code, details=details, hint=hint)


def cmd_serve(args: Any) -> int:
    """`codna durable serve` / `codna serve`: the loopback Run API the IDE drives a run through.

    A foreground command: it binds, prints ONE banner (the address and the token's PATH, never the
    token), and serves until interrupted. Everything the surface enforces — loopback only, the 0600
    bearer token, an identity on every mutation — lives in `run_api`; this is the translator that
    turns flags into `RunApiSettings` and a listen socket.

    ``--no-temporal`` is for a client that only needs the ledger (`GET /runs`,
    ``/runs/:id/events``): it starts the server with no Temporal client, so a read-only viewer does
    not require the durable extra or a running server. The routes that need Temporal answer 503 with
    a stable code rather than pretending.

    THE BANNER STATES THE MODE, not only the address. `capabilities` names which route keys this
    server serves and which it refuses, plus one sentence saying what still works; `temporal` stays
    because a caller keying on its absence should keep working. A mode an operator chose is described
    as a mode — never as a failure to reach something.
    """
    from . import require_durable
    from .identity import IdentityError, IdentityResolver
    from .run_api import RunApiServer, TemporalRunBackend
    from .run_api_support import RunApiError, RunApiSettings, load_or_create_token

    if not getattr(args, "no_temporal", False):
        require_durable()
    config = _cli_module()._config(args)
    # THE STACK, IN ONE COMMAND. `--start-dev-server` brings Temporal up (idempotently: `ensure_started`
    # adopts a server that is already answering) BEFORE the backend connects, so the very next thing this
    # process does — `TemporalRunBackend(config)` — finds the address live instead of refusing. Refused
    # beside `--no-temporal`, where there is no Temporal client to connect at all: accepting the pair would
    # silently do nothing, and a flag that does nothing is the class of defect this whole surface exists to
    # stop shipping.
    if getattr(args, "start_dev_server", False):
        if getattr(args, "no_temporal", False):
            raise _error(
                "--start-dev-server cannot be combined with --no-temporal.",
                code="serve_start_dev_server_conflict",
                details={"start_dev_server": True, "no_temporal": True},
                hint="drop --no-temporal to use durable mode, or drop --start-dev-server for ledger-only mode",
            )
        with _cli_module()._dev_server_errors():
            _cli_module()._dev_server(config).ensure_started(timeout_s=args.start_timeout)
    try:
        settings = RunApiSettings.for_config(
            config,
            host=args.host,
            port=args.port,
            heartbeat_s=args.heartbeat_interval,
            max_streams=args.max_streams,
        )
    except RunApiError as exc:
        raise _error(str(exc), code=exc.code, details=dict(exc.details), hint=exc.hint) from exc

    # The identity layer is built BEFORE the backend: a misconfigured auth mode must fail at start,
    # not on the first request that happens to arrive. The token is minted once here and handed to
    # the mode that uses it, so the banner can name its path in every mode.
    token = load_or_create_token(settings.token_path)
    try:
        identity = IdentityResolver(identity_config_for(args, token=token))
    except IdentityError as exc:
        raise _error(str(exc), code=exc.code, details=dict(exc.details)) from exc

    backend: RunBackend
    if getattr(args, "no_temporal", False):
        backend = _LedgerOnlyBackend()
    else:
        try:
            backend = TemporalRunBackend(config)
        except RunApiError as exc:
            raise _error(str(exc), code=exc.code, details=dict(exc.details), hint=exc.hint) from exc

    try:
        server = RunApiServer(
            config=config, backend=backend, settings=settings, identity=identity
        )
    except RunApiError as exc:
        backend.close()
        raise _error(str(exc), code=exc.code, details=dict(exc.details), hint=exc.hint) from exc
    try:
        server.bind()
    except RunApiError as exc:
        backend.close()
        raise _error(str(exc), code=exc.code, details=dict(exc.details), hint=exc.hint) from exc
    capabilities = backend.capabilities()
    _cli_module()._print(
        {
            "run_api": {
                "url": server.base_url,
                "host": settings.host,
                "port": server.port,
                "token_path": str(settings.token_path),
                "auth_mode": identity.mode,
                # Kept as the ADDRESS-OR-NULL field existing callers already read. `capabilities`
                # carries the rest; the two are derived from the same object, so they cannot disagree.
                "temporal": capabilities.temporal_address,
                # THE SAME COMPOSER THE ROUTES USE (`RunApi.capabilities_payload`), not a second
                # `as_payload()` plus a hand-added field. The banner is what an operator reads to
                # know what they started, and it must say the same thing `GET /capabilities` does —
                # including the stream ceiling, which is the one capability that comes from
                # `--max-streams` rather than from the backend.
                "capabilities": server.api.capabilities_payload(),
                "task_queue": config.task_queue,
            }
        }
    )
    # The banner is the ONLY way a caller learns the address and the token path, and the process then
    # serves forever. When stdout is a pipe (how an IDE spawns this) Python block-buffers it, so the
    # caller would see nothing until the server exits — i.e. never. Flush explicitly.
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - an interactive stop, not a failure
        pass
    finally:
        server.stop()
    return 0


def identity_config_for(args: Any, *, token: str) -> Any:
    """The `IdentityConfig` the serve flags describe. Pure, so a test asserts it without a socket.

    The precedence is deliberate and is the whole of E2's "local keeps working" requirement:

    * ``--auth-mode oidc`` (or ``--oidc-issuer``, which implies it) → OIDC only.
    * ``--auth-mode local-trust`` → the NARROW unauthenticated path. It is refused unless the bind is
      loopback (`RunApiSettings` already guarantees that) AND no issuer is configured; attempting
      both is an `IdentityError` at construction rather than a server that quietly has a hole.
    * otherwise → the local bearer token, which is what every existing caller uses.

    No flag reads an identity or a role from the environment's ambient state beyond the token file:
    the subject a request is attributed to always comes from a verified credential.
    """
    from .identity import (
        AUTH_MODE_LOCAL_TRUST,
        AUTH_MODE_OIDC,
        AUTH_MODE_TOKEN,
        IdentityConfig,
    )

    mode = str(getattr(args, "auth_mode", "") or "").strip().lower()
    issuer = str(getattr(args, "oidc_issuer", "") or "").strip()
    audience = str(getattr(args, "oidc_audience", "") or "").strip()
    role_claim = str(getattr(args, "oidc_role_claim", "") or "roles").strip() or "roles"
    jwks_uri = str(getattr(args, "oidc_jwks_uri", "") or "").strip()
    local_roles = getattr(args, "local_role", None)
    roles = frozenset(local_roles) if local_roles else frozenset({"viewer", "operator", "approver"})
    identity = str(getattr(args, "local_identity", "") or "").strip()

    if not mode:
        mode = AUTH_MODE_OIDC if issuer else AUTH_MODE_TOKEN
    if mode == AUTH_MODE_LOCAL_TRUST and issuer:
        # Named here as well as in `IdentityConfig` so the operator gets a message about the FLAGS
        # they typed rather than about a field they never saw.
        from .identity import IdentityError

        raise IdentityError(
            "--auth-mode local-trust cannot be combined with --oidc-issuer: a server with an "
            "identity provider must not also accept unauthenticated callers.",
            code="run_api_local_trust_conflicts_with_oidc",
            details={"issuer": issuer},
        )
    if mode == AUTH_MODE_OIDC and not audience:
        from .identity import IdentityError

        raise IdentityError(
            "OIDC mode needs --oidc-audience (the audience your provider issued tokens for).",
            code="run_api_oidc_incomplete",
            details={"field": "--oidc-audience"},
        )
    return IdentityConfig(
        mode=mode,
        local_token=token,
        local_roles=roles,
        local_identity=identity,
        issuer=issuer,
        audience=audience,
        role_claim=role_claim,
        jwks_uri=jwks_uri,
    )


class _LedgerOnlyBackend(RunBackend):
    """A backend for a server started with ``--no-temporal``: the ledger routes work, the rest say so.

    Explicit rather than a silent no-op: a caller that steers a run through a ledger-only server must
    be told the steering did not happen. ``run_api_temporal_disabled`` is that answer, and
    `require_run` still succeeds for a run whose ledger exists — which is exactly the read-only
    viewer's case.

    `capabilities` is the SAME fact stated before the fact, so a caller does not have to provoke a 503
    to discover the mode. Both come from `ledger_only_capabilities`, so the banner and the refusal
    cannot describe two different servers.
    """

    def __init__(self) -> None:
        self._capabilities = ledger_only_capabilities()

    def capabilities(self) -> RunCapabilities:
        return self._capabilities

    def _unavailable(self) -> Any:
        from .run_api_support import RunApiError as _Error

        return _Error(
            "this run API is in ledger-only mode (it was started with --no-temporal): it can list "
            "runs and read their history, but it cannot start or control a run.",
            code="run_api_temporal_disabled",
            status=503,
            details={
                "no_temporal": True,
                "mode": self._capabilities.mode,
                # The routes that DO work, named in the refusal itself: an operator reading a 503 is
                # owed the same statement the banner gives, not a hint to try something else.
                "available_routes": sorted(self._capabilities.available_routes),
            },
            hint=(
                "listing runs and reading their history work here; to start, steer or control a run, "
                "start Temporal (`codna durable dev-server start`) and restart `codna serve` without "
                "--no-temporal"
            ),
        )

    def start_run(self, *, payload: Any, run_id: str, run_timeout_s: int) -> dict[str, Any]:
        raise self._unavailable()

    def status(self, run_id: str) -> dict[str, Any] | None:
        return None

    def update(self, run_id: str, *, task_id: str, action: str, payload: Any) -> Any:
        raise self._unavailable()

    def signal(self, run_id: str, action: str) -> None:
        raise self._unavailable()

    def close(self) -> None:
        return None




def add_serve_flags(serve: Any) -> None:
    """The `serve` flags. One definition for both surfaces (`codna serve`, `durable serve`).

    ``--host`` exists so a typo is a clear refusal naming the loopback rule, not so it can be
    escaped: there is deliberately no ``--allow-remote``. The server carries a local bearer token,
    and a flag that turned that off is a flag somebody would eventually turn off.
    """
    from .run_api_support import (
        DEFAULT_HEARTBEAT_S,
        DEFAULT_HOST,
        DEFAULT_MAX_STREAMS,
        DEFAULT_PORT,
    )

    serve.add_argument("--host", default=DEFAULT_HOST,
                       help=f"bind address; loopback only (default: {DEFAULT_HOST})")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"bind port, 0 for an ephemeral one (default: {DEFAULT_PORT})")
    serve.add_argument("--heartbeat-interval", type=float, default=DEFAULT_HEARTBEAT_S,
                       help=f"seconds between SSE keepalives (default: {DEFAULT_HEARTBEAT_S:g})")
    serve.add_argument("--max-streams", type=int, default=DEFAULT_MAX_STREAMS,
                       help="concurrent SSE streams to serve before answering 503 "
                            f"(default: {DEFAULT_MAX_STREAMS})")
    serve.add_argument("--no-temporal", action="store_true",
                       help="LEDGER-ONLY mode: serve GET /runs and /runs/:id/events (the run list and "
                            "its history); starting or controlling a run is refused with "
                            "run_api_temporal_disabled. Needs no Temporal server and no durable extra.")
    # ONE COMMAND INSTEAD OF TWO, and the reason is the product, not convenience. The IDE cannot start a
    # task unless this API is up in DURABLE mode, which needs a Temporal server. Every message that told a
    # user to run `codna durable dev-server start` and THEN `codna serve` asked them to keep two processes
    # alive by hand, and a user who ran only the second got `run_api_temporal_unavailable`. This flag makes
    # the documented remedy a single command, and it is the same flag `codna worker` already carries, so the
    # two execution surfaces start their dependency the same way.
    #
    # It is OPT-IN, deliberately: starting a background server is a side effect on the user's machine, so it
    # happens when a person (or the IDE, with consent) asks for it — never as a silent default of `serve`.
    serve.add_argument("--start-dev-server", action="store_true",
                       help="start the local Temporal dev server first if it is not running, then serve in "
                            "durable mode (one command instead of `codna durable dev-server start` + `codna serve`)")
    serve.add_argument("--start-timeout", type=float, default=60.0,
                       help="how long --start-dev-server may take to bring Temporal up (default: 60)")
    # -- the identity layer ------------------------------------------------------------------
    #
    # `token` is the default because it is what every existing caller already has. `oidc` is the
    # hosted deployment's mode. `local-trust` is the ONE unauthenticated path, and it is a flag a
    # person has to type: an accident cannot produce it, and `IdentityConfig` refuses it beside an
    # issuer so a host with an identity provider cannot also have the hole.
    serve.add_argument(
        "--auth-mode",
        choices=("token", "oidc", "local-trust"),
        default="",
        dest="auth_mode",
        help="how callers are authenticated: the local bearer token (default), OIDC, or "
             "local-trust (NO credential — loopback only, refused beside an issuer; for local dev)",
    )
    serve.add_argument("--oidc-issuer", default="",
                       help="OIDC issuer URL (implies --auth-mode oidc)")
    serve.add_argument("--oidc-audience", default="", help="the token audience this server accepts")
    serve.add_argument("--oidc-role-claim", default="", dest="oidc_role_claim",
                       help="the claim carrying roles (default: roles)")
    serve.add_argument("--oidc-jwks-uri", default="", dest="oidc_jwks_uri",
                       help="the provider's JWKS URL (default: discovered from the issuer)")
    # The local owner's roles. Explicit rather than implicit so a shared dev box can be narrowed to
    # read-only without inventing a second auth mode.
    serve.add_argument(
        "--local-role",
        action="append",
        choices=("viewer", "operator", "approver"),
        default=None,
        dest="local_role",
        help="a role the local caller holds (repeatable; default: all three)",
    )
    serve.add_argument("--local-identity", default="", dest="local_identity",
                       help="the identity local callers are recorded as (default: local-owner)")
