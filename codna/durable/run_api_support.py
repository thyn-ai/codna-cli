"""Primitives the Run API is built from: its error type, its token, its SSE framing, its inputs.

Split out of `run_api.py` for the repository's 1000-line ceiling (`tests/test_modularity.py`), on
the seam that already exists elsewhere in this package (`contract_primitives.py`,
`activity_support.py`): the things whose CORRECTNESS is independent of HTTP live here, and the
transport lives in `run_api.py`. A reviewer asking "is the token really 0600 and compared in
constant time?" or "is this frame byte-identical to the sidecar's?" reads one file.

**The SSE constants are copies, not choices.** `agent-core/vendor/cline/algenta/server/http.ts`
(``streamSessionEventsLive``) is the framing the IDE's `streamClient.ts` parser has been proven
against, so the header set, the frame template and the heartbeat template are transcribed from it
verbatim. `tests/durable/test_run_api.py` asserts each one equals the literal in that TypeScript
file: a "reasonable" SSE stream is not the acceptance criterion — THIS shape is.

**Validation never coerces.** An external caller's JSON is untrusted data: ``True`` is not the
integer 1, ``"4.5"`` is not a number, and an unexpected field is an ERROR rather than something
quietly dropped (a caller whose field was ignored believes it took effect). Every refusal carries a
stable ``code`` and an HTTP status, and the body is the repository's documented
``{"error": {code, message, details}}`` shape.
"""
from __future__ import annotations

import json
import os
import secrets
import stat
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping

from .config import DurableConfigError, validate_run_id
from .ledger import GENESIS_HASH, LedgerError, LedgerRecord, compute_hash
from ..runtime.processes import ensure_directories

if TYPE_CHECKING:  # pragma: no cover - typing only; `config` reaches httpx, which a leaf must not
    from .config import DurableConfig

__all__ = [
    "DEFAULT_HEARTBEAT_S",
    "DEFAULT_HOST",
    "DEFAULT_LIST_LIMIT",
    "DEFAULT_MAX_STREAMS",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DEFAULT_WRITE_TIMEOUT_S",
    "LOOPBACK_HOSTS",
    "MAX_BODY_BYTES",
    "MAX_IDENTITY_CHARS",
    "MAX_LIST_LIMIT",
    "MUTATION_ROUTES",
    "RunApiError",
    "RunApiSettings",
    "SSE_EVENT_TEMPLATE",
    "SSE_HEADERS",
    "SSE_HEARTBEAT_TEMPLATE",
    "TERMINAL_EVENT",
    "TERMINAL_EVENT_TYPES",
    "TOKEN_FILENAME",
    "TURN_ABORTED_EVENT_TYPES",
    "TURN_COMPLETED_EVENT",
    "TURN_FAILED_EVENT_TYPES",
    "TURN_TERMINAL_EVENT_TYPES",
    "TURN_VERDICT_EVENT_TYPES",
    "INERT_EVENT_TYPES",
    "STALLED_AFTER_S",
    "LedgerTail",
    "constant_time_token_equals",
    "event_frame_payload",
    "event_type_of",
    "last_seq_of",
    "ledger_verdict",
    "load_or_create_token",
    "optional_positive_int",
    "optional_positive_number",
    "optional_string_list",
    "outcome_of",
    "outcome_of_turn_terminal",
    "parse_bearer",
    "reject_unknown_fields",
    "require_identity",
    "require_positive_number",
    "require_string_list",
    "require_task_id",
    "validated_run_id",
]

DEFAULT_HOST: Final = "127.0.0.1"
# A fixed default port so the IDE can be configured once. `--port 0` asks the OS for an ephemeral
# one, which is what the tests and any second server on one host need.
DEFAULT_PORT: Final = 8790
TOKEN_FILENAME: Final = "run-api.token"

# `localhost` is accepted as a spelling of loopback; anything else is refused outright. A hostname
# that HAPPENS to resolve to 127.0.0.1 today (or an empty string, meaning INADDR_ANY) is not enough:
# the bind must be loopback by construction, not by DNS resolution at start time.
LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "::1", "localhost"})

# One megabyte is far more than a run spec needs and far less than a client can make the server
# buffer. The sidecar's cap is 8 MiB because it carries chat payloads; this surface carries specs.
MAX_BODY_BYTES: Final = 1 * 1024 * 1024

# Every mutating route. Declared once so the router, the tests and the docs cannot disagree about
# how many there are or what they are called.
MUTATION_ROUTES: Final[frozenset[str]] = frozenset(
    {"steer", "approve", "pause", "resume", "budget", "cancel"}
)

DEFAULT_HEARTBEAT_S: Final = 15.0
DEFAULT_POLL_INTERVAL_S: Final = 0.25
DEFAULT_MAX_STREAMS: Final = 8
DEFAULT_REQUEST_TIMEOUT_S: Final = 30.0
DEFAULT_WRITE_TIMEOUT_S: Final = 10.0
DEFAULT_LIST_LIMIT: Final = 50
MAX_LIST_LIMIT: Final = 500
MAX_IDENTITY_CHARS: Final = 320
MAX_TASK_ID_CHARS: Final = 128

# -- SSE framing, transcribed from server/http.ts:44-158 ----------------------------------------
#
# The four headers `streamSessionEventsLive` writes, in its order. `x-accel-buffering: no` is not
# decoration: without it a proxy buffers the stream and the client sees nothing until it ends.
SSE_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("content-type", "text/event-stream"),
    ("cache-control", "no-cache"),
    ("connection", "keep-alive"),
    ("x-accel-buffering", "no"),
)
# One data frame: `id:` (so an EventSource can resume), `event:` (always session_event, the name the
# IDE's parser ignores), then the JSON. The blank line ends the frame.
SSE_EVENT_TEMPLATE: Final = "id: {event_index}\nevent: session_event\ndata: {data}\n\n"
# Keepalive: an SSE COMMENT (a line starting with `:`), which every SSE parser drops and which keeps
# a read timeout from firing on an idle run.
SSE_HEARTBEAT_TEMPLATE: Final = ": heartbeat {now_ms}\n\n"

# The event that ends a run's stream. Exactly one terminal ends the stream, and one already in the
# backlog ends the replay — the rule `streamSessionEventsLive` applies to a turn's terminal.
# `run.finished` is the run's own final notification (`RunWorkflow.run` emits it before returning).
TERMINAL_EVENT: Final = "run.finished"
TERMINAL_EVENT_TYPES: Final[frozenset[str]] = frozenset({TERMINAL_EVENT})

# -- turn terminals: not RUN terminals, but evidence that the run ended -------------------------
#
# The distinction the run list used to miss. `run.finished` is written only on the two `completed`
# exits of `RunWorkflow` (all tasks done, and goal met). Every other way a run stops — a task whose
# turn failed, a cancellation, a dead worker — leaves the ledger ending at a TURN terminal and
# nothing above it. Reading only `run.finished` therefore reported such a run as "not finished", and
# the sidebar animated it as Working indefinitely.
#
# The names and the membership are `sidecar_client.SSE_TERMINAL_TURN_EVENTS`, transcribed here (the
# same way the SSE constants above are) because a turn terminal is the SIDECAR's vocabulary, and
# `tests/durable/test_run_api.py` asserts the two sets are equal so they cannot drift.
TURN_COMPLETED_EVENT: Final = "session.turn_completed"
# The engine reported the turn as failed: a failure of the work, not of the harness.
TURN_FAILED_EVENT_TYPES: Final[frozenset[str]] = frozenset({"session.turn_failed"})
# The turn was stopped rather than lost. An INTERRUPTED turn is not a failure — the soak harness
# reports `turn_unpaired` as a warning for exactly this reason — so these resolve to `stopped`.
TURN_ABORTED_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"session.turn_aborted", "session.aborted", "session.stopped"}
)
# The turn terminals that are EVIDENCE about how the RUN ended.
#
# `session.turn_completed` is deliberately NOT here, and the omission is the point. A turn that
# completed is a task that did its job: it is work, not a verdict, and a run whose next task is still
# coming must not read as over because the previous turn ended cleanly. That is exactly why it is also
# absent from `INERT_EVENT_TYPES` — it is neither a verdict nor noise, it is progress.
#
# Listed in `TURN_TERMINAL_EVENT_TYPES` below only so `event_type_of`'s vocabulary is complete; nothing
# in the verdict path may treat it as an ending.
TURN_TERMINAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {TURN_COMPLETED_EVENT} | TURN_FAILED_EVENT_TYPES | TURN_ABORTED_EVENT_TYPES
)
# The subset a verdict may be DERIVED from: everything that says the turn did not finish its work.
TURN_VERDICT_EVENT_TYPES: Final[frozenset[str]] = TURN_FAILED_EVENT_TYPES | TURN_ABORTED_EVENT_TYPES

# Records that are INERT when looking for a verdict: they neither record work nor give one.
#
#   * `turn.resumed_from_ledger` / `session.resumed` narrate a RETRY. An aborted turn is followed by
#     a resume, and the resume is not evidence the run is alive again — the work after it is.
#   * `control.requested` is the audit record of a human's signal. A pause on a finished run is
#     allowed and harmless and appends a note PAST the terminal; "the last record is a control note"
#     must not be read as "this run never finished".
#
# Anything NOT in this set is work, and work after a turn terminal means it was superseded.
INERT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"turn.resumed_from_ledger", "session.resumed", "control.requested"}
)

# -- how long a quiet ledger is read as "stalled" rather than "working" -------------------------
#
# Twice `contracts.ACTIVITY_TURN_TIMEOUT_S` (60 min), kept as a literal here because this module is
# a LEAF that `run_api_http` imports at load time: importing `contracts` would pull the provider,
# state and prompt layers into the HTTP surface for one integer. The value is pinned by
# `test_stalled_threshold_is_twice_the_longest_activity`, so it cannot drift from what it mirrors.
#
# WHY TWICE THE LONGEST ACTIVITY. `ACTIVITY_TURN_TIMEOUT_S` is Temporal's own outer ceiling for one
# activity — the longest a paid turn may run before it is abandoned. A live run that has written
# NOTHING for longer than two such ceilings has outlived the whole timeout-and-retry machinery: its
# activity was abandoned, its retries ran and ended, and no worker is producing this ledger. Waiting
# longer only keeps the lie up for longer. Nothing is lost by being this generous, because the
# verdict is re-derived on every read: a run that starts writing again leaves `stalled` on its own,
# with no state to clear.
ACTIVITY_TURN_TIMEOUT_S: Final = 60 * 60
STALLED_AFTER_S: Final = 2 * ACTIVITY_TURN_TIMEOUT_S


class RunApiError(RuntimeError):
    """A request that cannot be served: an HTTP status plus a stable, machine-readable ``code``.

    The body is the repository's documented error shape, so a caller parses a Run API failure the
    same way it parses the CLI's — one error contract, two transports.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int = 400,
        details: dict[str, object] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = int(status)
        self.details: dict[str, object] = dict(details or {})
        self.hint = hint

    def as_body(self) -> dict[str, object]:
        error: dict[str, object] = {
            "code": self.code,
            "message": str(self),
            "details": self.details,
        }
        if self.hint:
            error["hint"] = self.hint
        return {"error": error}


class ClientGone(Exception):
    """The peer closed the connection mid-response. Not an error; never logged as one."""


# -- the token ---------------------------------------------------------------------------------


def load_or_create_token(path: Path) -> str:
    """Return the bearer token at ``path``, creating it 0600 the first time.

    Created with ``O_EXCL`` plus an explicit ``fchmod`` so the mode is 0600 regardless of the
    process umask: a token another local user can read is not a bearer token. An EXISTING file is
    restricted to 0600 as well (an operator's ``cp`` or ``install`` can widen it) and the result is
    VERIFIED — unable to secure it is a hard error, never a warning.

    Two servers on one host share one token: that is what lets the IDE hold a single credential.
    """
    target = Path(path).expanduser()
    ensure_directories(target.parent)
    if target.exists():
        _restrict_token_mode(target)
        return _read_token(target)
    token = secrets.token_urlsafe(48)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _restrict_token_mode(target)
        return _read_token(target)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, (token + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return token


def _read_token(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - an unreadable file on a working disk
        raise RunApiError(
            f"cannot read the run API token at {path}.",
            code="run_api_token_unreadable",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            details={"path": str(path), "cause": str(exc)},
        ) from exc
    if not value:
        raise RunApiError(
            f"the run API token at {path} is empty; delete it and restart to generate a new one.",
            code="run_api_token_empty",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            details={"path": str(path)},
        )
    return value


def _restrict_token_mode(path: Path) -> None:
    """Force 0600 and verify it. A wider mode is a local credential leak, not a preference."""
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if not mode & 0o077:
        return
    try:
        os.chmod(path, 0o600)
    except OSError as exc:  # pragma: no cover - a foreign-owned file
        raise RunApiError(
            f"the run API token at {path} is mode {mode:04o} and cannot be restricted.",
            code="run_api_token_insecure",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            details={"path": str(path), "mode": f"{mode:04o}", "cause": str(exc)},
            hint=f"chmod 600 {path}",
        ) from exc
    restricted = stat.S_IMODE(os.stat(path).st_mode)
    if restricted & 0o077:
        raise RunApiError(
            f"the run API token at {path} is still mode {restricted:04o}.",
            code="run_api_token_insecure",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            details={"path": str(path), "mode": f"{restricted:04o}"},
            hint=f"chmod 600 {path}",
        )


def parse_bearer(header: str | None) -> str | None:
    """The token from ``Authorization: Bearer <token>``, or None.

    The scheme is matched case-insensitively (RFC 7235 does not require lowercase) and a header with
    a different scheme is "absent" — an unauthenticated caller must not learn how its header parsed.
    """
    if not header:
        return None
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.strip().lower() != "bearer":
        return None
    token = value.strip()
    return token or None


def constant_time_token_equals(expected: str, provided: str) -> bool:
    """Compare two tokens with ``hmac.compare_digest``, never ``==``.

    Encoded to bytes first: ``compare_digest`` accepts ``str`` only when both are ASCII, so a
    non-ASCII header would raise instead of failing closed.
    """
    import hmac

    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


# -- settings ----------------------------------------------------------------------------------


ALLOWED_REPO_ROOTS_ENV: Final = "CODNA_DURABLE_REPO_ROOTS"


def allowed_repo_roots_from_env(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Operator-approved local workspace roots, defaulting to the server's current repository."""
    source = os.environ if env is None else env
    raw = (source.get(ALLOWED_REPO_ROOTS_ENV) or "").strip()
    if not raw:
        return (Path.cwd(),)
    roots = tuple(Path(part.strip()).expanduser().resolve() for part in raw.split(os.pathsep) if part.strip())
    return roots or (Path.cwd(),)


@dataclass(frozen=True)
class RunApiSettings:
    """Everything the server's behaviour depends on, validated at construction.

    Frozen and fully validated so a bad value fails where it is configured rather than on whichever
    request happens to reach the code that uses it. ``host`` is validated here, which is what makes
    "loopback only" a property of the type rather than a check somebody can forget.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token_path: Path = Path.home() / ".codna" / TOKEN_FILENAME
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    max_streams: int = DEFAULT_MAX_STREAMS
    max_body_bytes: int = MAX_BODY_BYTES
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    write_timeout_s: float = DEFAULT_WRITE_TIMEOUT_S
    allowed_repo_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.cwd(),))

    def __post_init__(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise RunApiError(
                f"the run API binds loopback only; refusing host {self.host!r}.",
                code="run_api_bind_not_loopback",
                status=HTTPStatus.BAD_REQUEST,
                details={"host": self.host, "allowed": sorted(LOOPBACK_HOSTS)},
                hint="this surface carries a local bearer token; put a proxy in front of it",
            )
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 0 <= self.port <= 65535:
            raise RunApiError(
                f"port must be 0..65535 (got {self.port!r}).",
                code="run_api_port_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"port": repr(self.port)},
            )
        for name, value in (
            ("heartbeat_s", self.heartbeat_s),
            ("poll_interval_s", self.poll_interval_s),
            ("request_timeout_s", self.request_timeout_s),
            ("write_timeout_s", self.write_timeout_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise RunApiError(
                    f"{name} must be a positive number (got {value!r}).",
                    code="run_api_settings_invalid",
                    status=HTTPStatus.BAD_REQUEST,
                    details={"field": name, "value": repr(value)},
                )
        for name, value in (
            ("max_streams", self.max_streams),
            ("max_body_bytes", self.max_body_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RunApiError(
                    f"{name} must be a positive integer (got {value!r}).",
                    code="run_api_settings_invalid",
                    status=HTTPStatus.BAD_REQUEST,
                    details={"field": name, "value": repr(value)},
                )
        if not self.allowed_repo_roots or any(not root.is_absolute() for root in self.allowed_repo_roots):
            raise RunApiError(
                "allowed repository roots must be absolute and non-empty.",
                code="run_api_settings_invalid",
                status=HTTPStatus.BAD_REQUEST,
                details={"field": "allowed_repo_roots"},
            )

    @classmethod
    def for_config(
        cls,
        config: DurableConfig,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        token_path: Path | None = None,
        allowed_repo_roots: tuple[Path, ...] | None = None,
        **overrides: Any,
    ) -> "RunApiSettings":
        """Defaults derived from the resolved durable config: the token lives beside the runs."""
        return cls(
            host=host,
            port=port,
            token_path=token_path or (config.paths.root / TOKEN_FILENAME),
            allowed_repo_roots=allowed_repo_roots_from_env() if allowed_repo_roots is None else allowed_repo_roots,
            **overrides,
        )


# -- request validation ------------------------------------------------------------------------
#
# Every function raises `RunApiError` with a stable code and the status that belongs to it. Nothing
# here coerces: see the module docstring.


def validated_run_id(raw: Any) -> str:
    try:
        return validate_run_id(raw)
    except DurableConfigError as exc:
        raise RunApiError(
            "run id must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$.",
            code="run_api_run_id_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"value": raw if isinstance(raw, str) else type(raw).__name__},
        ) from exc


def require_identity(body: Mapping[str, Any]) -> str:
    """Who is acting. Required on every mutation: an approval is an audit record, not a click."""
    raw = body.get("identity")
    if not isinstance(raw, str) or not raw.strip():
        raise RunApiError(
            "this route needs an identity: who is acting?",
            code="run_api_identity_required",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": "identity"},
            hint="approvals and steering notes are audit records; an anonymous one is refused",
        )
    identity = raw.strip()
    if len(identity) > MAX_IDENTITY_CHARS:
        raise RunApiError(
            f"identity must be at most {MAX_IDENTITY_CHARS} characters.",
            code="run_api_identity_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"length": len(identity), "max": MAX_IDENTITY_CHARS},
        )
    return identity


def require_task_id(body: Mapping[str, Any]) -> str:
    """The task an Update addresses. Bounded and path-free: it becomes part of a workflow id."""
    raw = body.get("task_id")
    if not isinstance(raw, str) or not raw.strip():
        raise RunApiError(
            "this route addresses a task: pass task_id (for example task-1).",
            code="run_api_task_required",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": "task_id"},
        )
    value = raw.strip()
    if "/" in value or "\\" in value or ".." in value or len(value) > MAX_TASK_ID_CHARS:
        raise RunApiError(
            "task_id must not contain a path separator.",
            code="run_api_task_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"task_id": value[:MAX_TASK_ID_CHARS]},
        )
    return value


def require_positive_number(body: Mapping[str, Any], field: str) -> float:
    raw = body.get(field)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not raw > 0:
        raise RunApiError(
            f"{field} must be a positive number.",
            code="run_api_number_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": field, "value": repr(raw)},
        )
    return float(raw)


def optional_positive_number(body: Mapping[str, Any], field: str) -> float | None:
    if body.get(field) is None:
        return None
    return require_positive_number(body, field)


def optional_positive_int(body: Mapping[str, Any], field: str, *, default: int) -> int:
    """A positive integer, ``default`` when absent. ``True``/``1.5``/``"2"`` are all refused."""
    if body.get(field) is None:
        return default
    raw = body[field]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise RunApiError(
            f"{field} must be a positive integer.",
            code="run_api_number_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": field, "value": repr(raw)},
        )
    return raw


def optional_positive_int_or_none(body: Mapping[str, Any], field: str) -> int | None:
    if body.get(field) is None:
        return None
    return optional_positive_int(body, field, default=0)


def optional_bool(body: Mapping[str, Any], field: str, *, default: bool) -> bool:
    """A JSON boolean, ``default`` when absent.

    Strict on the TYPE for the same reason the numeric helpers are: ``"true"``, ``1`` and ``0`` are
    each a plausible thing for a caller to send and each one means something different from the
    boolean it looks like. A flag whose value is silently coerced is a flag whose effect cannot be
    predicted from the request, and this particular flag decides whether a run creates one full
    checkout per task — the wrong guess leaves checkouts on a user's disk.
    """
    if body.get(field) is None:
        return default
    raw = body[field]
    if not isinstance(raw, bool):
        raise RunApiError(
            f"{field} must be a boolean.",
            code="run_api_field_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": field, "value": repr(raw), "type": type(raw).__name__},
        )
    return raw


def require_string_list(body: Mapping[str, Any], field: str) -> list[str]:
    if body.get(field) is None:
        raise RunApiError(
            f"{field} is required.",
            code="run_api_field_required",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": field},
        )
    return _string_list(body, field)


def optional_string_list(body: Mapping[str, Any], field: str) -> list[str]:
    if body.get(field) is None:
        return []
    return _string_list(body, field)


def _string_list(body: Mapping[str, Any], field: str) -> list[str]:
    raw = body[field]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise RunApiError(
            f"{field} must be a list of strings.",
            code="run_api_field_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"field": field},
        )
    return list(raw)


def reject_unknown_fields(body: Mapping[str, Any], allowed: frozenset[str], *, route: str) -> None:
    """Refuse a field this route does not consume.

    Dropping it silently is worse than refusing it: the caller would believe the value took effect.
    """
    unknown = sorted(set(body) - set(allowed))
    if unknown:
        raise RunApiError(
            f"{route} does not accept {', '.join(unknown)}.",
            code="run_api_field_unknown",
            status=HTTPStatus.BAD_REQUEST,
            details={"route": route, "unknown": unknown, "allowed": sorted(allowed)},
        )


# -- query parameters --------------------------------------------------------------------------


def after_param(query: Mapping[str, list[str]]) -> int:
    """``?after=`` — the last ledger seq the caller already has. Validated, never coerced.

    A mistyped resume point silently SKIPS records, and the records it would skip are exactly the
    ones the caller has not seen. So a bad value is a 400 on every run, not only on one whose ledger
    happens to exist.
    """
    raw = query.get("after")
    if raw is None:
        return 0
    value = raw[-1].strip() if raw else ""
    if not value.isdigit():
        raise RunApiError(
            "after must be a non-negative integer (the last seq you already have).",
            code="run_api_after_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"after": value[:32]},
        )
    return int(value)


def follow_param(query: Mapping[str, list[str]]) -> bool:
    raw = query.get("follow")
    if not raw:
        return True
    value = raw[-1].strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise RunApiError(
        "follow must be 0 or 1.",
        code="run_api_follow_invalid",
        status=HTTPStatus.BAD_REQUEST,
        details={"follow": value[:16]},
    )


def limit_param(query: Mapping[str, list[str]], *, default: int = DEFAULT_LIST_LIMIT) -> int:
    raw = query.get("limit")
    if not raw:
        return default
    value = raw[-1].strip()
    if not value.isdigit():
        raise RunApiError(
            "limit must be a positive integer.",
            code="run_api_limit_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"limit": value[:16]},
        )
    parsed = int(value)
    if not 1 <= parsed <= MAX_LIST_LIMIT:
        raise RunApiError(
            f"limit must be 1..{MAX_LIST_LIMIT}.",
            code="run_api_limit_invalid",
            status=HTTPStatus.BAD_REQUEST,
            details={"limit": parsed, "max": MAX_LIST_LIMIT},
        )
    return parsed


# -- the ledger as the event source ------------------------------------------------------------


class LedgerTail:
    """Incremental reader of an append-only NDJSON ledger.

    Reads only bytes appended since the last poll, keeping a trailing partial line buffered. Both
    properties matter:

    * re-reading the whole file every ``poll_interval_s`` is O(n²) over a long run;
    * parsing a half-written last line would report a confident "malformed ledger" for a record that
      is merely still being written — the one failure mode a tail must not have.

    A file that shrinks (truncated or replaced) restarts from zero rather than seeking past EOF.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._buffer = b""

    @property
    def path(self) -> Path:
        return self._path

    def poll(self) -> list[LedgerRecord]:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self._offset:
            self._offset = 0
            self._buffer = b""
        if size == self._offset:
            return []
        with self._path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read(size - self._offset)
        self._offset += len(chunk)
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        records: list[LedgerRecord] = []
        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerError(
                    "a ledger record is not valid JSON",
                    code="ledger_line_invalid",
                    details={"path": str(self._path), "error": str(exc)},
                ) from exc
            if not isinstance(decoded, dict):
                raise LedgerError(
                    "a ledger record is not an object",
                    code="ledger_line_invalid",
                    details={"path": str(self._path)},
                )
            records.append(LedgerRecord.from_payload(decoded))
        return records


def event_type_of(record: LedgerRecord) -> str:
    """The ``event_type`` a frame carries. The IDE's parser drops a frame without a string one.

    Three shapes reach a ledger:

    * ``turn.event`` wraps the engine's own event, so its type is the engine's;
    * ``note`` is a notification or a control record. A notification names itself in ``kind``
      (``run.started``, ``run.finished``); a control record names itself in ``event``
      (``control.requested``);
    * everything else is named by its ledger kind (``task.finished``, ``approval.granted``, ...).
    """
    payload = record.payload
    if record.kind == "turn.event":
        raw = payload.get("event_type")
        return raw if isinstance(raw, str) and raw else "turn.event"
    if record.kind == "note":
        if payload.get("event") in (None, "", "notification"):
            kind = payload.get("kind")
            if isinstance(kind, str) and kind:
                return kind
        raw = payload.get("event")
        if isinstance(raw, str) and raw:
            return raw
        return "note"
    return record.kind


def event_frame_payload(record: LedgerRecord) -> dict[str, Any]:
    """The JSON object inside one ``data:`` line.

    ``event_index`` is the ledger's ``seq`` — the number ``?after=`` resumes from — so a client that
    persists the last index it saw reattaches exactly where it left off, with no second cursor to
    keep in sync. ``hash``/``prev_hash`` ride along so a client can verify a record it cached.
    """
    return {
        "event_index": record.seq,
        "event_type": event_type_of(record),
        "run_id": record.run_id,
        "task_id": record.task_id,
        "session_id": record.session_id,
        "kind": record.kind,
        "ts": record.ts,
        "payload": record.payload,
        "prev_hash": record.prev_hash,
        "hash": record.hash,
    }


def outcome_of(path: Path) -> str | None:
    """The run's outcome, read from the tail of the ledger. ``None`` when the run has not ended.

    A bounded read: the answer is in the final records, so scanning a year of ledger to find it
    would be work with no answer at the end of it. ``None`` is the honest answer for a truncated
    tail as well — a record that did not commit is not an outcome.

    TWO SHAPES OF "FINISHED", because the workflow writes two. ``run.finished`` is the run's own
    terminal and carries the outcome the workflow decided. A run whose task never got that far ends
    on a TURN terminal and writes nothing above it (see ``TURN_TERMINAL_EVENT_TYPES``). Both are
    read here, so a run that plainly ended stops being reported as open.

    A turn terminal has no ``outcome`` field, so one is DERIVED from WHICH terminal it was, in the
    vocabulary run-level outcomes already use (``RunOutcome`` in ``run_contracts``, plus the
    ``failed`` the workflow already finalises with). The distinction that matters is kept: a turn
    the engine FAILED is ``failed``; a turn that was stopped or interrupted is ``stopped``. An
    interrupted turn is not a failure — the same rule the soak harness applies by reporting
    ``turn_unpaired`` as a warning rather than a violation.
    """
    ended, outcome = terminal_verdict(path)
    return outcome if ended else None


def outcome_of_turn_terminal(event_type: str) -> str:
    """The run outcome implied by a turn terminal. Total over ``TURN_VERDICT_EVENT_TYPES``.

    ``failed`` for a turn the engine reported as failed; ``stopped`` for one that was aborted or
    interrupted. Both are non-``completed``, so both reach the sidebar as a run the user has to look
    at rather than as one still in progress.

    ``session.turn_completed`` is REFUSED rather than mapped: a completed turn is success, and folding
    it into "stopped" would make a turn that finished its job indistinguishable from one that was
    killed. A caller holding a completed turn has no verdict to derive, and this raises instead of
    inventing one.
    """
    if event_type in TURN_FAILED_EVENT_TYPES:
        return "failed"
    if event_type in TURN_ABORTED_EVENT_TYPES:
        return "stopped"
    raise ValueError(
        f"{event_type!r} is not a turn verdict: a turn that completed is progress, not an ending"
    )


def _terminal_verdict(records: list[LedgerRecord]) -> tuple[bool, str | None]:
    """``(ended, outcome)`` from a tail, newest first. The ONE place the terminal rules are applied.

    **A ``run.finished`` is final.** It is the run's own completion and nothing is scheduled after
    it, so the whole window is searched for one FIRST. That is what keeps the property the
    control-note case depends on: a pause recorded AFTER the run finished (allowed and harmless)
    cannot un-finish it, and neither can a stray record.

    **A turn terminal is superseded by work that followed it.** An aborted turn is not always an
    ending: a worker restarts the turn under the same ``turn_key``, and the ledger then reads
    ``session.aborted``, ``turn.resumed_from_ledger``, ``session.resumed``, ... The abort records
    what happened; it is not a verdict that outlives the resume. So when the run wrote no terminal
    of its own, this returns the NEWEST turn verdict, looking past the INERT records above: they
    neither confirm nor revive one. Any other record is work, and work supersedes a turn terminal.

    Measured against the 97 ledgers of a 24h soak: 52 end at a ``review`` record and 23 at
    ``session.aborted``. Reading "the newest terminal anywhere" would call 75 of them finished;
    reading "the newest verdict, superseded by work" calls finished exactly the ones that stopped.
    """
    for record in records:
        if event_type_of(record) != TERMINAL_EVENT:
            continue
        raw = record.payload.get("outcome")
        return True, raw if isinstance(raw, str) and raw else None
    for record in records:
        event_type = event_type_of(record)
        if event_type in INERT_EVENT_TYPES:
            continue
        if event_type in TURN_VERDICT_EVENT_TYPES:
            return True, outcome_of_turn_terminal(event_type)
        # Any other record is work — a completed turn, a task starting, a tool call — that happened
        # AFTER an older turn verdict, so it supersedes it.
        return False, None
    return False, None


def terminal_verdict(path: Path) -> tuple[bool, str | None]:
    """``(ended, outcome)`` for one ledger, from a SINGLE bounded tail read.

    The one entry point a route should use when it reports both facts, because two separate reads can
    observe two different file states: a record appended between them makes ``terminal: False`` sit
    beside ``outcome: "completed"``, and a response that contradicts itself is worse than either
    answer alone. ``outcome_of`` remains for a caller that wants only the cause.
    """
    return _terminal_verdict(_tail_records(path, limit=_OUTCOME_SCAN_RECORDS))


def ledger_verdict(
    path: Path, *, updated_ts: float, now: float | None = None
) -> dict[str, object]:
    """Everything a LIST read needs about one ledger, from a single bounded tail pass.

    One pass rather than one per question: the answers would otherwise be able to disagree if a
    record were appended between them, and a row that said "stalled" while carrying an outcome would
    be worse than either answer alone.

    * ``ended`` — a terminal is present, so the run is over. This is what stops a spinner.
    * ``outcome`` — what the ledger recorded, or what its turn terminal implies. ``None`` when it
      ended without saying why, which is a fact rather than a gap to fill.
    * ``stalled`` — no terminal AND nothing appended for ``STALLED_AFTER_S``. The honest third
      state: not finished, and not being worked on either.

    ``updated_ts`` is the ledger's mtime — the same value the list reports as the row's age, so the
    row's "3d" and this verdict cannot disagree about how long it has been quiet.
    """
    ended, outcome = _terminal_verdict(_tail_records(path, limit=_OUTCOME_SCAN_RECORDS))
    current = time.time() if now is None else now
    quiet_for_s = max(0.0, current - updated_ts)
    return {
        "ended": ended,
        "outcome": outcome,
        "stalled": (not ended) and quiet_for_s >= STALLED_AFTER_S,
        "quiet_for_s": round(quiet_for_s, 3),
    }


def last_seq_of(path: Path) -> int:
    """The highest ``seq`` in the ledger, read from the tail. Zero when the file is unreadable.

    The tail read can land on a partial final line (a writer mid-append), so this takes the highest
    ``seq`` across the window rather than assuming the last complete line is the last record.
    """
    records = _tail_records(path, limit=_OUTCOME_SCAN_RECORDS)
    return max((record.seq for record in records), default=0)


_TAIL_WINDOW_BYTES: Final = 64 * 1024
# How far back the outcome scan looks. A terminal is followed by at most a handful of control notes,
# so a small bound finds it without turning a summary into a full scan.
_OUTCOME_SCAN_RECORDS: Final = 32


def _tail_records(path: Path, *, limit: int) -> list[LedgerRecord]:
    """Up to ``limit`` complete records from the end of the file, reading only the last 64 KiB."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - _TAIL_WINDOW_BYTES))
            tail = handle.read()
    except OSError:
        return []
    if not tail:
        return []
    # A window that starts mid-record leaves a fragment at the front; it is dropped rather than
    # parsed, because its beginning is outside the window.
    chunks = tail.split(b"\n")
    start = 1 if (size > _TAIL_WINDOW_BYTES and not tail.startswith(b"\n")) else 0
    found: list[LedgerRecord] = []
    for raw in reversed(chunks[start:]):
        text = raw.strip()
        if not text:
            continue
        try:
            decoded = json.loads(text.decode("utf-8"))
            if not isinstance(decoded, dict):
                continue
            found.append(LedgerRecord.from_payload(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError, LedgerError):
            continue
        if len(found) >= limit:
            break
    return found


def ledger_problems(records: list[LedgerRecord]) -> list[str]:
    """Hash-chain and contiguity problems across the records a caller has read."""
    problems: list[str] = []
    expected_prev = GENESIS_HASH
    expected_seq = 1
    for record in records:
        if record.seq != expected_seq:
            problems.append(f"seq {record.seq} at position {expected_seq}")
        if record.prev_hash != expected_prev:
            problems.append(f"seq {record.seq}: prev_hash mismatch")
        recomputed = compute_hash(record.to_payload())
        if recomputed != record.hash:
            problems.append(f"seq {record.seq}: hash mismatch")
        expected_prev = record.hash
        expected_seq = record.seq + 1
    return problems


def now_ms() -> int:
    """Wall-clock milliseconds for a heartbeat comment. Injectable by callers that care."""
    return int(time.time() * 1000)
