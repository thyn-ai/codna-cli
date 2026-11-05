"""E2: the Run API's authorization model, its identity layer, and its audit export.

**What this file is about.** Nothing here asks "does the happy path work" — `test_run_api.py` owns
that. This file asks the four questions that decide whether the surface can be exposed to a second
person at all:

1. **Is authorization decided server-side, per route, and does it fail CLOSED?** The role matrix is
   asserted route by route, by authenticating as each role and observing 200/403. A route with no
   declared role must be a BUILD failure, so the test that matters most is the one that proves
   `Route` for an undeclared key raises — the router cannot be extended carelessly.
2. **Does an unauthenticated caller get NOTHING?** Including the case that used to be the hole: a
   bearer token that is valid but carries no roles.
3. **Is the identity verified, not asserted?** The body's `identity` may only CONFIRM the principal;
   a different name is a 403 and never reaches the backend.
4. **Does the OIDC seam work deterministically?** Tokens are minted locally with a real RSA key and
   verified against an in-process JWKS, so the SAME `JwtVerifier` code the hosted deployment runs is
   exercised — including the attacks (alg confusion, wrong audience, expired, unknown kid).

**The one thing not proven here, stated plainly.** No live identity provider is reachable from this
environment, so `HttpJwksSource`'s HTTPS fetch is exercised with a scripted opener rather than a
socket. The seam either side of it (verification, claim mapping, the 401/503 split) is real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("jwt", reason="OIDC verification needs PyJWT")

import http.client  # noqa: E402

from codna.durable.config import load_durable_config  # noqa: E402
from codna.durable.identity import (  # noqa: E402
    AUTH_MODE_LOCAL_TRUST,
    AUTH_MODE_OIDC,
    AUTH_MODE_TOKEN,
    AuthenticationError,
    IdentityConfig,
    IdentityError,
    IdentityResolver,
    ProviderUnavailable,
    bearer_from_header,
)
from codna.durable.ledger import RunLedger  # noqa: E402
from codna.durable.roles import (  # noqa: E402
    APPROVER,
    OPERATOR,
    ROUTE_ROLES,
    VIEWER,
    AuthorizationError,
    Principal,
    UnknownRouteError,
    required_role,
    role_from_claim,
)
from codna.durable.run_api import RunApiServer  # noqa: E402
from codna.durable.run_api_backend import (  # noqa: E402
    RunBackend,
    RunCapabilities,
    durable_capabilities,
)
from codna.durable.run_api_http import ALL_ROUTES, Route  # noqa: E402
from codna.durable.run_api_support import RunApiSettings  # noqa: E402
from codna.durable.siem import (  # noqa: E402
    FileSiemSink,
    NullSiemSink,
    RemoteSiemSink,
    RetentionPolicy,
    SiemDeliveryError,
    SiemEvent,
    build_siem_sink,
    retention_days_from_env,
)

IDENTITY = "ops@codna.ai"
REPO_ROOT = Path(__file__).resolve().parents[3]


# -- a scripted backend ------------------------------------------------------------------------


class ScriptedBackend(RunBackend):
    """Records what reached it. Nothing Temporal; the transport is what is under test."""

    def __init__(self, *, statuses: Mapping[str, dict[str, Any] | None] | None = None) -> None:
        self.statuses: dict[str, dict[str, Any] | None] = dict(statuses or {})
        self.updates: list[dict[str, Any]] = []
        self.signals: list[tuple[str, str]] = []
        self.started: list[dict[str, Any]] = []
        self.closed = False

    def capabilities(self) -> RunCapabilities:
        # A scripted backend with a Temporal-shaped answer: these tests exercise AUTH, and a
        # ledger-only answer would make every mutation a 503 before the role check under test.
        return durable_capabilities("127.0.0.1:7233")

    def start_run(self, *, payload: Mapping[str, Any], run_id: str, run_timeout_s: int) -> dict[str, Any]:
        self.started.append({"run_id": run_id, "payload": dict(payload)})
        return {"run_id": run_id, "workflow_id": f"run:{run_id}", "tasks": 0, "versioning": {}}

    def status(self, run_id: str) -> dict[str, Any] | None:
        return self.statuses.get(run_id)

    def update(self, run_id: str, *, task_id: str, action: str, payload: Mapping[str, Any]) -> Any:
        self.updates.append(
            {"run_id": run_id, "task_id": task_id, "action": action, "payload": dict(payload)}
        )
        return {"applied": True, "identity": payload.get("identity")}

    def signal(self, run_id: str, action: str) -> None:
        self.signals.append((run_id, action))

    def close(self) -> None:
        self.closed = True


# -- fixtures ----------------------------------------------------------------------------------


def _config(tmp_path: Path) -> Any:
    root = tmp_path / "runtime"
    os.environ["CODNA_RUNTIME_ROOT"] = str(root)
    env = dict(os.environ)
    env["CODNA_RUNTIME_ROOT"] = str(root)
    return load_durable_config(env=env, hostname="rbac-test-host")


class Live:
    """A running server plus the raw request helper the tests need."""

    def __init__(self, server: RunApiServer, backend: ScriptedBackend) -> None:
        self.server = server
        self.backend = backend
        self.host, self.port = server.address
        self.token = server.token

    def call(
        self,
        method: str,
        path: str,
        *,
        token: str | None = "__default__",
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10.0)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request_headers = {"accept": "application/json"}
        if payload is not None:
            request_headers["content-type"] = "application/json"
        resolved = self.token if token == "__default__" else token
        if resolved is not None:
            request_headers["authorization"] = f"Bearer {resolved}"
        request_headers.update(dict(headers or {}))
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            decoded: dict[str, Any] | None = None
            if raw and "json" in (response.getheader("content-type") or ""):
                decoded = json.loads(raw)
            return response.status, decoded
        finally:
            connection.close()


def _expected_status(entry: tuple[str, str, dict[str, Any] | None, str]) -> int:
    """The success status for a matrix row. A start is 202 (accepted, not created); a stream is 200."""
    method, path, _, _ = entry
    if method == "POST" and path == "/runs":
        return 202
    return 200


def _start(
    *,
    config: Any,
    identity: IdentityResolver,
    backend: ScriptedBackend | None = None,
    tmp_path: Path | None = None,
) -> Live:
    settings = RunApiSettings(
        port=0,
        token_path=config.paths.root / "run-api" / "run-api.token",
        heartbeat_s=0.2,
        poll_interval_s=0.02,
        request_timeout_s=5.0,
        write_timeout_s=2.0,
        allowed_repo_roots=(tmp_path,) if tmp_path is not None else (Path.cwd(),),
    )
    scripted = backend or ScriptedBackend()
    server = RunApiServer(
        config=config,
        backend=scripted,
        settings=settings,
        identity=identity,
        siem=NullSiemSink(),
    ).start()
    return Live(server, scripted)


def _seed(config: Any, run_id: str = "run-1") -> str:
    ledger = RunLedger.for_run(config.paths, run_id)
    ledger.append("run.started", payload={"event": "notification", "kind": "run.started"})
    return run_id


@pytest.fixture
def config(tmp_path: Path) -> Any:
    return _config(tmp_path)


def _resolver_for(config: Any, roles: frozenset[str]) -> IdentityResolver:
    """A local-token resolver holding EXACTLY ``roles`` — the role matrix's knob."""
    from codna.durable.run_api_support import load_or_create_token

    token_path = config.paths.root / "run-api" / "run-api.token"
    return IdentityResolver(
        IdentityConfig(
            mode=AUTH_MODE_TOKEN,
            local_token=load_or_create_token(token_path),
            local_identity=IDENTITY,
            local_roles=roles,
        )
    )


# -- 1. the role matrix, per route --------------------------------------------------------------
#
# Each entry is (method, path, body, required role). The role is the MINIMUM: a principal holding a
# stronger role must also pass, and that is asserted below. Adding a route without an entry here
# would leave it untested; the "every route is covered" test below is what makes this exhaustive.

ROUTE_MATRIX: tuple[tuple[str, str, dict[str, Any] | None, str], ...] = (
    ("GET", "/capabilities", None, VIEWER),
    ("GET", "/runs", None, VIEWER),
    ("GET", "/runs/run-1", None, VIEWER),
    ("GET", "/runs/run-1/events?follow=0", None, VIEWER),
    ("POST", "/runs", {"briefs": ["do a thing"]}, OPERATOR),
    ("POST", "/runs/run-1/steer", {"task_id": "task-1", "note": "n", "identity": IDENTITY}, OPERATOR),
    ("POST", "/runs/run-1/budget", {"task_id": "task-1", "max_cost_usd": 2, "identity": IDENTITY}, OPERATOR),
    ("POST", "/runs/run-1/pause", {"identity": IDENTITY}, OPERATOR),
    ("POST", "/runs/run-1/resume", {"identity": IDENTITY}, OPERATOR),
    ("POST", "/runs/run-1/cancel", {"identity": IDENTITY}, OPERATOR),
    ("POST", "/runs/run-1/approve", {"task_id": "task-1", "identity": IDENTITY}, APPROVER),
)


def _matrix_id(entry: tuple[str, str, dict[str, Any] | None, str]) -> str:
    method, path, _, role = entry
    return f"{method} {path.split('?')[0]} -> {role}"


def test_the_matrix_covers_every_declared_route() -> None:
    """The parametrization above must name EVERY route in the table, or a route is untested.

    Path shapes are mapped to keys here explicitly: the point is not to re-derive the router, it is
    to make "a route exists with no role matrix row" a red test.
    """
    covered = {
        "GET /capabilities": "runs.capabilities",
        "GET /runs": "runs.list",
        "GET /runs/run-1": "runs.get",
        "GET /runs/run-1/events": "runs.events",
        "POST /runs": "runs.start",
        "POST /runs/run-1/steer": "runs.steer",
        "POST /runs/run-1/budget": "runs.budget",
        "POST /runs/run-1/pause": "runs.pause",
        "POST /runs/run-1/resume": "runs.resume",
        "POST /runs/run-1/cancel": "runs.cancel",
        "POST /runs/run-1/approve": "runs.approve",
    }
    tested = {f"{method} {path.split('?')[0]}" for method, path, _, _ in ROUTE_MATRIX}
    assert tested == set(covered), "a route is missing from ROUTE_MATRIX"
    assert set(covered.values()) == set(ROUTE_ROLES), "a declared route key has no matrix row"


@pytest.mark.parametrize("entry", ROUTE_MATRIX, ids=_matrix_id)
def test_a_viewer_may_not_reach_a_privileged_route(config: Any, entry: Any) -> None:
    """The 403 half of the matrix: a viewer gets read routes and nothing else."""
    method, path, body, required = entry
    live = _start(config=config, identity=_resolver_for(config, frozenset({VIEWER})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call(method, path, body=body)
        if required == VIEWER:
            assert status == _expected_status(entry), payload
        else:
            assert status == 403, payload
            assert payload["error"]["code"] == "run_api_forbidden"
            assert payload["error"]["details"]["required_role"] == required
            assert payload["error"]["details"]["held_roles"] == ["viewer"]
    finally:
        live.server.stop()


@pytest.mark.parametrize("entry", ROUTE_MATRIX, ids=_matrix_id)
def test_an_operator_reaches_operator_routes_but_not_approval(config: Any, entry: Any) -> None:
    """Operator implies viewer; it does NOT imply approver. That separation is the point of E2."""
    method, path, body, required = entry
    live = _start(config=config, identity=_resolver_for(config, frozenset({OPERATOR})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call(method, path, body=body)
        if required == APPROVER:
            assert status == 403, payload
            assert payload["error"]["details"]["required_role"] == APPROVER
        else:
            assert status == _expected_status(entry), payload
    finally:
        live.server.stop()


@pytest.mark.parametrize("entry", ROUTE_MATRIX, ids=_matrix_id)
def test_an_approver_reaches_every_route(config: Any, entry: Any) -> None:
    """Approver implies operator implies viewer — the full expansion, proven on every route."""
    method, path, body, _ = entry
    live = _start(config=config, identity=_resolver_for(config, frozenset({APPROVER})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call(method, path, body=body)
        assert status == _expected_status(entry), payload
    finally:
        live.server.stop()


# -- 2. deny by default, structurally ----------------------------------------------------------


def test_an_undeclared_route_cannot_be_constructed() -> None:
    """THE structural guarantee: a route whose key has no role raises where it is constructed.

    Not "the handler checks a role" — the router's own type refuses to exist. A route added without a
    role requirement is an exception at import, which is the difference between "tested" and
    "cannot happen".
    """
    with pytest.raises(UnknownRouteError) as caught:
        Route("runs.teleport", "POST", "_create_run")
    assert "runs.teleport" in str(caught.value)


def test_every_declared_route_carries_a_role() -> None:
    for route in ALL_ROUTES:
        assert route.key in ROUTE_ROLES
        assert required_role(route.key) in {VIEWER, OPERATOR, APPROVER}


def test_every_handler_reachable_from_a_route_is_declared() -> None:
    """The closed set of handler names, so adding a handler is a test failure that asks for a role.

    `_serve` is the ONLY caller of a handler and it reaches one exclusively through a `Route`. If a
    new `_handler` appears on the class and is not in `ALL_ROUTES`, this fails — which is the prompt
    to add its key to `ROUTE_ROLES`, not a silent new endpoint.
    """
    from codna.durable.run_api_http import RunApiHandler

    declared = {route.handler for route in ALL_ROUTES}
    handlers = {
        name
        for name, value in vars(RunApiHandler).items()
        if name.startswith("_")
        and callable(value)
        and not name.startswith("__")
        # The infrastructure half: resolution, framing, the body, the error path, the stream pump.
        # These are NOT routes; they are what a route uses.
        and name
        not in {
            "_dispatch",
            "_route",
            "_resolve",
            "_serve",
            "_not_found",
            "_method_not_allowed",
            "_send_json",
            "_send_error",
            "_read_body",
            "_verified_identity",
            "_mutation_fields",
            "_send_sse_head",
            "_pump",
            "_write_frame",
            "_write_stream_error",
            "_write",
            "_peer_closed",
            "_route_by_key",
            # A SUB-handler, not a route: `_mutate` calls it after the role check, for the three
            # task-level Updates. It is named here (rather than in `ALL_ROUTES`) because it has no
            # path of its own — `POST /runs/:id/steer` resolves to `runs.steer` → `_mutate`.
            "_task_update",
        }
    }
    assert handlers == declared, (
        f"undeclared handler(s): {sorted(handlers - declared)}; "
        f"declared but missing: {sorted(declared - handlers)}"
    )


def test_the_transport_never_calls_a_handler_outside_serve() -> None:
    """Read from the source: `_dispatch`/`_route` must not shortcut past the role check.

    The role check lives between resolution and `_serve`, so a direct `self._create_run(...)` from the
    router would be an unauthenticated route *by construction*. AST rather than behaviour because a
    behaviour test can only exercise the paths somebody remembered to write.
    """
    import ast

    path = REPO_ROOT / "cli" / "codna" / "durable" / "run_api_http.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sources = ("_dispatch", "_route")
    forbidden = {
        "_create_run",
        "_list_runs",
        "_get_run",
        "_stream_events",
        "_mutate",
        "_serve",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in sources:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr in forbidden - {"_serve"}:
                raise AssertionError(
                    f"{node.name} calls {func.attr} directly; every handler must go through _serve"
                )


def test_required_role_refuses_an_unknown_key_instead_of_defaulting() -> None:
    """No default. A permissive fallback is the "if role: allow" bug wearing a dictionary."""
    with pytest.raises(UnknownRouteError):
        required_role("runs.nope")


def test_an_unauthenticated_caller_gets_nothing_on_every_route(config: Any) -> None:
    """No token at all: 401 everywhere, including the read routes."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({APPROVER})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        for method, path, body, _ in ROUTE_MATRIX:
            status, payload = live.call(method, path, body=body, token=None)
            assert status == 401, (method, path, payload)
            assert payload["error"]["code"] == "run_api_unauthorized"
        assert live.backend.started == [] and live.backend.updates == [] and live.backend.signals == []
    finally:
        live.server.stop()


def test_a_valid_token_with_no_roles_is_a_403_not_a_pass(config: Any) -> None:
    """Authenticated but unprivileged: the caller exists and may do nothing.

    This is the case a "if principal.roles: allow" check would let through, so it is asserted
    separately from the missing-token case: authentication and authorization are two decisions.
    """
    live = _start(config=config, identity=_resolver_for(config, frozenset()))
    try:
        _seed(config, "run-1")
        status, payload = live.call("GET", "/runs")
        assert status == 403, payload
        assert payload["error"]["code"] == "run_api_forbidden"
        assert payload["error"]["details"]["held_roles"] == []
    finally:
        live.server.stop()


def test_an_unrecognised_role_in_a_token_is_dropped_not_granted(config: Any) -> None:
    """A provider naming a role this build does not implement cannot create it."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({OPERATOR})))
    try:
        _seed(config, "run-1")
        # The resolver's config only ever holds known roles (validated), so the drop is asserted on
        # the claim mapping — the only place a token's role set is built.
        assert role_from_claim(["operator", "superuser", "ADMIN"]) == frozenset({"operator"})
        status, _ = live.call("GET", "/runs")
        assert status == 200
    finally:
        live.server.stop()


# -- 3. the identity is verified, never asserted ------------------------------------------------


def test_a_mutation_without_a_body_identity_is_400(config: Any) -> None:
    """The field stays required (the contract), but it can only CONFIRM the principal."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({APPROVER})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call("POST", "/runs/run-1/approve", body={"task_id": "task-1"})
        assert status == 400
        assert payload["error"]["code"] == "run_api_identity_required"
        assert live.backend.updates == []
    finally:
        live.server.stop()


def test_the_ledger_and_the_update_name_the_authenticated_identity(config: Any) -> None:
    """A signal's identity lands in the hash chain as the PRINCIPAL's, not the body's spelling."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({OPERATOR})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call(
            "POST", "/runs/run-1/pause", body={"identity": IDENTITY}
        )
        assert status == 200, payload
        assert payload["identity"] == IDENTITY
        tail = RunLedger.for_run(config.paths, "run-1").tail()
        assert tail.payload["identity"] == IDENTITY
    finally:
        live.server.stop()


def test_a_header_cannot_select_an_identity(config: Any) -> None:
    """A client-set header (X-Identity, X-User, …) is never consulted — proven by sending one."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({APPROVER})))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, payload = live.call(
            "POST",
            "/runs/run-1/approve",
            body={"task_id": "task-1", "identity": "cto@codna.ai"},
            headers={"x-identity": "cto@codna.ai", "x-forwarded-user": "cto@codna.ai"},
        )
        assert status == 403
        assert payload["error"]["code"] == "run_api_identity_mismatch"
        assert live.backend.updates == []
    finally:
        live.server.stop()


# -- 4. the narrow local unauthenticated path --------------------------------------------------


def test_local_trust_requires_an_explicit_mode() -> None:
    """The default is the token. An unauthenticated server exists only if somebody asked for one."""
    with pytest.raises(IdentityError):
        IdentityConfig(mode=AUTH_MODE_TOKEN, local_token="")
    config = IdentityConfig(mode=AUTH_MODE_LOCAL_TRUST)
    assert config.mode == AUTH_MODE_LOCAL_TRUST


def test_local_trust_is_refused_beside_an_oidc_issuer() -> None:
    """A server with an identity provider must not also have a hole that skips it."""
    with pytest.raises(IdentityError) as caught:
        IdentityConfig(mode=AUTH_MODE_LOCAL_TRUST, issuer="https://accounts.example")
    assert caught.value.code == "run_api_local_trust_conflicts_with_oidc"


def test_local_trust_refuses_an_issuer_at_resolution_too() -> None:
    """Belt and braces: a config built without validation is refused at the first request."""
    resolver = IdentityResolver(IdentityConfig(mode=AUTH_MODE_LOCAL_TRUST))
    with pytest.raises(IdentityError):
        # A dataclass replace is how a caller could bypass __post_init__ validation.
        from dataclasses import replace

        broken = replace(resolver.config, issuer="https://accounts.example")
        IdentityResolver(broken).resolve(None)


def test_local_trust_authenticates_nothing_and_says_so(config: Any) -> None:
    """The path works with NO header, and every act is labelled `local-unauthenticated`."""
    live = _start(
        config=config,
        identity=IdentityResolver(
            IdentityConfig(mode=AUTH_MODE_LOCAL_TRUST, local_identity=IDENTITY)
        ),
    )
    try:
        _seed(config, "run-1")
        status, body = live.call("GET", "/runs", token=None)
        assert status == 200, body
        status, body = live.call("GET", "/runs", token="any-junk-at-all")
        assert status == 200, "local-trust ignores credentials rather than mixing models"
        principal = live.server.identity.resolve(None)
        assert principal.source == "local-unauthenticated"
        assert principal.identity == IDENTITY
    finally:
        live.server.stop()


def test_local_trust_still_enforces_roles(config: Any) -> None:
    """Unauthenticated is not unprivileged-by-default: the local roles decide, and approval needs one."""
    live = _start(
        config=config,
        identity=IdentityResolver(
            IdentityConfig(
                mode=AUTH_MODE_LOCAL_TRUST,
                local_identity=IDENTITY,
                local_roles=frozenset({VIEWER}),
            )
        ),
    )
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        status, _ = live.call("GET", "/runs", token=None)
        assert status == 200
        status, body = live.call(
            "POST", "/runs/run-1/approve", body={"task_id": "task-1", "identity": IDENTITY}, token=None
        )
        assert status == 403
        assert body["error"]["details"]["required_role"] == APPROVER
    finally:
        live.server.stop()


# -- 5. the OIDC seam -------------------------------------------------------------------------


def _rsa_keypair() -> tuple[Any, dict[str, Any]]:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update({"kid": "test-key-1", "alg": "RS256", "use": "sig"})
    return private, {"keys": [jwk]}


def _token(private: Any, *, claims: dict[str, Any], kid: str = "test-key-1", algorithm: str = "RS256") -> str:
    import jwt

    pem = private.private_bytes(
        encoding=__import__("cryptography.hazmat.primitives.serialization", fromlist=["Encoding"]).Encoding.PEM,
        format=__import__(
            "cryptography.hazmat.primitives.serialization", fromlist=["PrivateFormat"]
        ).PrivateFormat.PKCS8,
        encryption_algorithm=__import__(
            "cryptography.hazmat.primitives.serialization", fromlist=["NoEncryption"]
        ).NoEncryption(),
    )
    return jwt.encode(claims, pem, algorithm=algorithm, headers={"kid": kid})


ISSUER = "https://accounts.example.test"
AUDIENCE = "codna-run-api"


def _oidc_resolver(*, jwks: Mapping[str, Any], clock: Any = None) -> IdentityResolver:
    return IdentityResolver(
        IdentityConfig(
            mode=AUTH_MODE_OIDC,
            local_token="unused-in-oidc-mode",
            issuer=ISSUER,
            audience=AUDIENCE,
            role_claim="roles",
            jwks_json=jwks,
            **({"clock": clock} if clock else {}),
        )
    )


def _claims(*, roles: list[str] | None = None, **overrides: Any) -> dict[str, Any]:
    import time as _time

    base: dict[str, Any] = {
        "sub": "user-123",
        "email": "dev@codna.ai",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(_time.time()) + 600,
        "iat": int(_time.time()),
        "roles": roles if roles is not None else ["viewer", "operator", "approver"],
    }
    base.update(overrides)
    return base


def test_oidc_config_needs_issuer_and_audience() -> None:
    with pytest.raises(IdentityError) as caught:
        IdentityConfig(mode=AUTH_MODE_OIDC, issuer="", audience=AUDIENCE)
    assert caught.value.code == "run_api_oidc_incomplete"
    with pytest.raises(IdentityError) as caught:
        IdentityConfig(mode=AUTH_MODE_OIDC, issuer=ISSUER, audience="")
    assert caught.value.code == "run_api_oidc_incomplete"


def test_oidc_refuses_a_plain_http_issuer() -> None:
    """An http issuer lets a network attacker swap the JWKS and mint their own tokens."""
    with pytest.raises(IdentityError) as caught:
        IdentityConfig(mode=AUTH_MODE_OIDC, issuer="http://accounts.example", audience=AUDIENCE)
    assert caught.value.code == "run_api_oidc_issuer_insecure"


def test_a_valid_token_yields_a_principal_from_its_verified_claims() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    principal = resolver.resolve(f"Bearer {_token(private, claims=_claims())}")
    assert principal.identity == "user-123 (dev@codna.ai)"
    assert principal.roles == frozenset({"viewer", "operator", "approver"})
    assert principal.source == "oidc"


def test_oidc_roles_come_from_the_claim_and_are_capped_to_known_roles() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    principal = resolver.resolve(
        f"Bearer {_token(private, claims=_claims(roles=['viewer', 'root']))}"
    )
    assert principal.roles == frozenset({"viewer"})
    with pytest.raises(AuthorizationError):
        principal.require("runs.approve")


def test_a_token_signed_by_another_key_is_refused() -> None:
    _, jwks = _rsa_keypair()
    other_private, _ = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(other_private, claims=_claims())}")


def test_an_alg_none_token_is_refused() -> None:
    """The algorithm is chosen by the verifier, never by the token."""
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    import jwt

    unsigned = jwt.encode(_claims(), key="", algorithm="none")
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {unsigned}")


def test_an_hs256_token_is_refused_even_with_the_public_key() -> None:
    """Algorithm confusion: a token claiming HS256 must never be HMAC-verified with a public key."""
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    import jwt

    confused = jwt.encode(_claims(), key="a-shared-secret-of-sufficient-length-000", algorithm="HS256",
                          headers={"kid": "test-key-1"})
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {confused}")


def test_an_expired_token_is_refused() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    expired = _claims(exp=1_000_000_000)  # 2001: long outside the 60s leeway
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(private, claims=expired)}")


def test_the_wrong_audience_is_refused() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(private, claims=_claims(aud='some-other-api'))}")


def test_the_wrong_issuer_is_refused() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(private, claims=_claims(iss='https://evil.example'))}")


def test_an_unknown_kid_is_refused() -> None:
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(private, claims=_claims(), kid='not-published')}")


def test_a_token_without_a_subject_is_refused() -> None:
    """A provider that omits `sub` cannot be used to attribute an action."""
    private, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    claims = _claims()
    claims.pop("sub")
    with pytest.raises(AuthenticationError):
        resolver.resolve(f"Bearer {_token(private, claims=claims)}")


def test_a_malformed_token_is_the_same_401_as_no_token() -> None:
    _, jwks = _rsa_keypair()
    resolver = _oidc_resolver(jwks=jwks)
    for header in ("Bearer not.a.jwt", "Bearer ", "Basic abc", None):
        with pytest.raises(AuthenticationError) as caught:
            resolver.resolve(header)
        assert caught.value.code == "run_api_unauthorized"


def test_a_jwks_failure_is_a_provider_error_not_a_bad_token() -> None:
    """An outage must not read as "your token is wrong" — it sends an operator to the wrong place."""
    from codna.durable.identity import HttpJwksSource

    def failing(url: str, timeout: float) -> Mapping[str, Any]:
        raise OSError("connection refused")

    source = HttpJwksSource(issuer=ISSUER, jwks_uri=f"{ISSUER}/jwks", opener=failing)
    with pytest.raises(ProviderUnavailable):
        source.keys()


def test_http_jwks_source_caches_within_its_ttl() -> None:
    from codna.durable.identity import HttpJwksSource

    fetches: list[str] = []

    def opener(url: str, timeout: float) -> Mapping[str, Any]:
        fetches.append(url)
        return {"keys": [{"kid": "k", "kty": "RSA"}]}

    now = [1000.0]
    source = HttpJwksSource(
        issuer=ISSUER, jwks_uri=f"{ISSUER}/jwks", ttl_s=60.0, clock=lambda: now[0], opener=opener
    )
    source.keys()
    source.keys()
    assert len(fetches) == 1
    now[0] += 61.0
    source.keys()
    assert len(fetches) == 2


def test_an_oidc_server_enforces_the_role_matrix_end_to_end(config: Any) -> None:
    """The seam, over a real socket: a viewer token reaches reads and is 403 on approval."""
    private, jwks = _rsa_keypair()
    live = _start(config=config, identity=_oidc_resolver(jwks=jwks))
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        viewer = _token(private, claims=_claims(roles=["viewer"]))
        status, body = live.call("GET", "/runs", token=viewer)
        assert status == 200, body
        status, body = live.call(
            "POST", "/runs/run-1/approve",
            token=viewer,
            body={"task_id": "task-1", "identity": "user-123 (dev@codna.ai)"},
        )
        assert status == 403, body
        assert body["error"]["details"]["required_role"] == APPROVER
        assert live.backend.updates == []
    finally:
        live.server.stop()


def test_bearer_parsing_is_case_insensitive_and_refuses_other_schemes() -> None:
    assert bearer_from_header("Bearer abc") == "abc"
    assert bearer_from_header("bearer abc") == "abc"
    assert bearer_from_header("BEARER  abc ") == "abc"
    assert bearer_from_header("Basic abc") is None
    assert bearer_from_header("Bearer") is None
    assert bearer_from_header(None) is None
    assert bearer_from_header("Bearer " + "x" * 100_000) is None


# -- 6. the SIEM export ------------------------------------------------------------------------


def test_a_file_sink_writes_one_json_object_per_line(tmp_path: Path) -> None:
    sink = FileSiemSink(tmp_path)
    principal = Principal(identity=IDENTITY, roles=frozenset({APPROVER}), source="local-token")
    sink.record(SiemEvent.action("run.approved", principal=principal, route="runs.approve",
                                run_id="run-1", task_id="task-1"))
    sink.record(SiemEvent.action("auth.denied", principal=principal, route="runs.start",
                                outcome="denied"))
    lines = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["run.approved", "auth.denied"]
    assert lines[0]["identity"] == IDENTITY
    assert lines[0]["roles"] == ["approver"]
    assert lines[0]["run_id"] == "run-1"
    assert lines[0]["iso_ts"].endswith("+00:00")


def test_a_siem_event_refuses_an_unknown_name_and_an_empty_identity() -> None:
    principal = Principal(identity=IDENTITY, roles=frozenset({VIEWER}), source="local-token")
    with pytest.raises(ValueError):
        SiemEvent.action("run.invented", principal=principal, route="runs.list")
    with pytest.raises(ValueError):
        SiemEvent(event="auth.succeeded", ts=0.0, identity="", roles=(), source="oidc")


def test_the_file_sink_rotates_at_its_size_bound(tmp_path: Path) -> None:
    sink = FileSiemSink(tmp_path, max_bytes=200)
    principal = Principal(identity=IDENTITY, roles=frozenset({VIEWER}), source="local-token")
    first = sink.path
    for _ in range(5):
        sink.record(SiemEvent.action("auth.succeeded", principal=principal, route="runs.list"))
    files = sorted(tmp_path.glob("run-api-*.ndjson"))
    assert len(files) > 1
    assert first != sink.path or len(files) > 1


def test_a_remote_sink_refuses_rather_than_silently_dropping() -> None:
    """No remote sink is configured in this build; it must SAY so at the call site."""
    principal = Principal(identity=IDENTITY, roles=frozenset({VIEWER}), source="local-token")
    with pytest.raises(SiemDeliveryError) as caught:
        RemoteSiemSink().record(SiemEvent.action("auth.succeeded", principal=principal, route="runs.list"))
    assert caught.value.code == "siem_remote_not_implemented"


def test_the_server_exports_an_approval_and_a_denial(config: Any, tmp_path: Path) -> None:
    """End to end: two requests, two SIEM rows, with the verified identity and the outcome."""
    live = _start(config=config, identity=_resolver_for(config, frozenset({OPERATOR})))
    sink = FileSiemSink(tmp_path / "siem")
    live.server.api.siem = sink
    try:
        _seed(config, "run-1")
        live.backend.statuses["run-1"] = {"run_id": "run-1"}
        live.call("GET", "/runs")
        live.call("POST", "/runs/run-1/approve", body={"task_id": "task-1", "identity": IDENTITY})
        rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
        names = [row["event"] for row in rows]
        assert "auth.succeeded" in names
        assert "auth.denied" in names
        denied = next(row for row in rows if row["event"] == "auth.denied")
        assert denied["route"] == "runs.approve"
        assert denied["detail"]["required"] == "approver"
        assert denied["identity"] == IDENTITY
    finally:
        live.server.stop()


def test_build_siem_sink_honours_the_directory_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODNA_SIEM_DIR", str(tmp_path / "from-env"))
    sink = build_siem_sink()
    assert isinstance(sink, FileSiemSink)
    assert sink.path.parent == tmp_path / "from-env"


# -- 7. retention beyond Temporal Cloud's window -----------------------------------------------


def test_a_retention_window_shorter_than_ninety_days_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODNA_SIEM_RETENTION_DAYS", "30")
    with pytest.raises(SiemDeliveryError) as caught:
        retention_days_from_env()
    assert caught.value.code == "siem_retention_too_short"


def test_a_malformed_retention_value_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODNA_SIEM_RETENTION_DAYS", "a-year")
    with pytest.raises(SiemDeliveryError) as caught:
        retention_days_from_env()
    assert caught.value.code == "siem_retention_invalid"


def test_the_default_retention_outlives_temporal_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODNA_SIEM_RETENTION_DAYS", raising=False)
    assert retention_days_from_env() > 90


def test_retention_prunes_only_expired_siem_files(tmp_path: Path) -> None:
    old = tmp_path / "run-api-20200101.ndjson"
    old.write_text("{}\n")
    os.utime(old, (0, 0))
    fresh = tmp_path / "run-api-29990101.ndjson"
    fresh.write_text("{}\n")
    policy = RetentionPolicy(days=365, siem_dir=tmp_path)
    removed = policy.apply()
    assert removed == [old.name]
    assert fresh.exists()


def test_retention_never_deletes_a_ledger(tmp_path: Path) -> None:
    """The evidence outlives the export: a retention job that erases a ledger is the worse bug."""
    runs = tmp_path / "runs" / "run-1"
    runs.mkdir(parents=True)
    ledger = runs / "ledger.ndjson"
    ledger.write_text("{}\n")
    os.utime(ledger, (0, 0))
    policy = RetentionPolicy(days=365, siem_dir=tmp_path / "siem")
    policy.apply()
    assert ledger.exists()


def test_retention_reports_a_broken_retained_ledger(tmp_path: Path) -> None:
    """`verify_ledgers` is the other half: retained evidence must still BE evidence."""
    root = tmp_path / "runs"
    good = root / "run-good"
    good.mkdir(parents=True)
    ledger = RunLedger(good / "ledger.ndjson", run_id="run-good")
    ledger.append("run.started", payload={"kind": "run.started"})

    broken = root / "run-broken"
    broken.mkdir(parents=True)
    (broken / "ledger.ndjson").write_text('{"seq": 1, "hash": "deadbeef"}\n')

    problems = RetentionPolicy(days=365, siem_dir=tmp_path / "siem").verify_ledgers(root)
    assert "run-broken" in problems
    assert "run-good" not in problems


# -- 8. the operational command ----------------------------------------------------------------


def test_the_audit_command_reports_retention_scope_and_the_export_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """`codna durable audit` answers "what is kept, where is the export, what did you prune"."""
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("CODNA_SIEM_DIR", str(tmp_path / "siem"))
    monkeypatch.delenv("CODNA_SIEM_RETENTION_DAYS", raising=False)
    from codna.durable.audit_cli import cmd_audit

    class Args:
        retention_days = 120
        apply_retention = True
        verify_ledgers = True
        temporal_profile = None

    assert cmd_audit(Args()) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["retention_days"] == 120
    assert body["export"]["sink"] == "file"
    assert "not implemented" in body["export"]["remote"]
    assert body["pruned"] == []
    assert body["broken_ledgers"] == {}


def test_the_audit_command_exits_two_when_a_ledger_is_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("CODNA_SIEM_DIR", str(tmp_path / "siem"))
    runs = tmp_path / "runtime" / "runs" / "run-bad"
    runs.mkdir(parents=True)
    (runs / "ledger.ndjson").write_text('{"seq": 1, "hash": "deadbeef"}\n')
    from codna.durable.audit_cli import cmd_audit

    class Args:
        retention_days = 365
        apply_retention = False
        verify_ledgers = True
        temporal_profile = None

    assert cmd_audit(Args()) == 2
    body = json.loads(capsys.readouterr().out)
    assert "run-bad" in body["broken_ledgers"]


def test_the_audit_command_refuses_a_short_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setenv("CODNA_RUNTIME_ROOT", str(tmp_path / "runtime"))
    from codna.durable.audit_cli import cmd_audit

    class Args:
        retention_days = 30
        apply_retention = False
        verify_ledgers = False
        temporal_profile = None

    assert cmd_audit(Args()) == 1
    body = json.loads(capsys.readouterr().out)
    assert body["error"]["code"] == "siem_retention_too_short"


# -- 9. the flags → IdentityConfig seam --------------------------------------------------------


class _ServeArgs:
    """The serve flags, as argparse would populate them."""

    def __init__(self, **overrides: Any) -> None:
        self.auth_mode = ""
        self.oidc_issuer = ""
        self.oidc_audience = ""
        self.oidc_role_claim = ""
        self.oidc_jwks_uri = ""
        self.local_role = None
        self.local_identity = ""
        for key, value in overrides.items():
            setattr(self, key, value)


def test_no_auth_flags_mean_the_local_bearer_token() -> None:
    """The default is what every existing caller already has, so nothing changes by omission."""
    from codna.durable.serve_cli import identity_config_for

    config = identity_config_for(_ServeArgs(), token="t")
    assert config.mode == AUTH_MODE_TOKEN
    assert config.local_roles == frozenset({"viewer", "operator", "approver"})


def test_local_trust_beside_an_issuer_is_refused_at_the_FLAG_layer() -> None:
    """The operator gets a message about the flags they typed, not a field they never saw."""
    from codna.durable.serve_cli import identity_config_for

    with pytest.raises(IdentityError) as caught:
        identity_config_for(
            _ServeArgs(auth_mode=AUTH_MODE_LOCAL_TRUST, oidc_issuer="https://a.example"), token="t"
        )
    assert caught.value.code == "run_api_local_trust_conflicts_with_oidc"


def test_an_issuer_implies_oidc_and_needs_an_audience() -> None:
    from codna.durable.serve_cli import identity_config_for

    with pytest.raises(IdentityError) as caught:
        identity_config_for(_ServeArgs(oidc_issuer="https://a.example"), token="t")
    assert caught.value.code == "run_api_oidc_incomplete"
    config = identity_config_for(
        _ServeArgs(oidc_issuer="https://a.example", oidc_audience="codna-run-api"), token="t"
    )
    assert config.mode == AUTH_MODE_OIDC
    assert config.audience == "codna-run-api"


def test_the_local_roles_can_be_narrowed_by_flag() -> None:
    from codna.durable.serve_cli import identity_config_for

    config = identity_config_for(_ServeArgs(local_role=["viewer"]), token="t")
    assert config.local_roles == frozenset({"viewer"})
    principal = IdentityResolver(config).resolve("Bearer t")
    assert principal.roles == frozenset({"viewer"})
    with pytest.raises(AuthorizationError):
        principal.require("runs.approve")


def test_local_trust_server_serves_with_no_credential_over_real_http(config: Any) -> None:
    """The narrow path, end to end: no header at all reaches a read route and is labelled as such."""
    live = _start(config=config, identity=IdentityResolver(IdentityConfig(mode=AUTH_MODE_LOCAL_TRUST)))
    try:
        _seed(config, "run-1")
        status, body = live.call("GET", "/runs", token=None)
        assert status == 200, body
        assert live.server.identity.mode == AUTH_MODE_LOCAL_TRUST
    finally:
        live.server.stop()
