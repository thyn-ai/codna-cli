"""`TaskState` — the workflow's memory, and the reason a durable run stays durable.

Temporal persists every workflow input, activity input and event payload in history. A state that
carried transcripts, diffs or model output would grow history without bound and, worse, would put
arbitrary generated text (and whatever a repo's prompt injection talked the model into echoing)
into the event log. So `TaskState` carries **references only**: ids, git refs, an event index, a
verdict, a budget number. The payloads themselves live where they belong — the ledger for events,
git for trees, the sidecar for transcripts.

Two invariants are enforced here rather than trusted:

* **Size.** The serialized state must stay under 2 MB. Temporal's default payload limit is what
  actually bites in production (4 MB blob, 2 MB per-payload guidance), and a state that close to
  the ceiling means something big got in that should have been a reference.
* **No secrets.** A state that reaches history is, for practical purposes, published: it is
  readable by anyone with UI access to the namespace, it survives in backups, and it is exported
  by `temporal workflow show`. Provider API keys, GitHub tokens and bearer headers must never
  appear, so the same value patterns `codna.sandbox` uses to scrub a worker's environment are
  applied here to refuse one outright. This is a backstop, not the primary control: activities
  resolve keys from the keystore and never put them in inputs.

Both checks run on the way in (`TaskState.from_payload`, used when decoding a workflow input) and
on the way out (`TaskState.to_payload`, used before any activity call), so a field added later
cannot bypass them by following a different path.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Final, Literal, Mapping

from ..sandbox import WRITE_TOKEN_NAMES, WRITE_TOKEN_VALUE
from .publish import (
    PublishSkipReason,
    publish_skip_from_payload,
    validate_publish_skip,
)
from .state_coercion import Coercion

__all__ = [
    "MAX_STATE_BYTES",
    "AskRef",
    "Attempts",
    "Budget",
    "CheckpointRef",
    "PendingApproval",
    "PullRequestRef",
    "ReviewVerdict",
    "STATE_SCHEMA_VERSION",
    "TaskState",
    "TaskStateError",
    "VerifyRef",
    "assert_state_encodable",
    "find_secret_like_strings",
]

STATE_SCHEMA_VERSION: Final = 1
MAX_STATE_BYTES: Final = 2 * 1024 * 1024

ReviewVerdict = Literal["approved", "rejected", "changes_requested"]

# Beyond the GitHub token patterns codna.sandbox already owns, a state can also leak a provider
# key. These are the shapes that actually get pasted into prompts and env files:
#   sk-...            OpenAI/Anthropic-style (Anthropic uses sk-ant-..., OpenAI sk-proj-...)
#   sk-ant-api03-...  Anthropic
#   AIza...           Google
#   xai-...           xAI
#   hf_...            Hugging Face
#   glpat-...         GitLab
#   AKIA...           AWS access key id
_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    WRITE_TOKEN_VALUE,
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", re.IGNORECASE),
)

# Field NAMES that must never appear in a state payload at all — a caller passing one is not
# smuggling a value, it is telling us the design was violated. Kept separate from the value
# patterns because the name alone is the error, whatever the value looks like.
_FORBIDDEN_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    name.lower()
    for name in (
        *WRITE_TOKEN_NAMES,
        "api_key",
        "apikey",
        "provider_api_key",
        "authorization",
        "auth_token",
        "bearer",
        "password",
        "secret",
        "private_key",
        "client_secret",
    )
)

_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,254}$")
# The engine's turn id alphabet: `turn_<session id>_<seq>` with session ids restricted to the same
# character set. Bounded generously because it legitimately EXCEEDS `_REF`'s 255 chars.
_TURN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:-]{0,511}$")
_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_URL = re.compile(r"^https://[^\s]{1,300}$")


class TaskStateError(ValueError):
    """A state that may not enter workflow history."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


def find_secret_like_strings(payload: Any, *, path: str = "") -> list[dict[str, str]]:
    """Return every secret-looking finding in an already-decoded payload, with its JSON path.

    Walks dicts, lists and strings. Returning the path (not the value) keeps the report itself
    safe to log — the whole point is that this text is already too sensitive to repeat.
    """
    findings: list[dict[str, str]] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key)
                child = f"{where}.{key_text}" if where else key_text
                if key_text.lower() in _FORBIDDEN_FIELD_NAMES:
                    findings.append({"path": child, "reason": "forbidden_field_name"})
                walk(value, child)
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")
            return
        if isinstance(node, str):
            for pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(node):
                    findings.append({"path": where or "$", "reason": "secret_value_pattern"})
                    return

    walk(payload, path)
    return findings


def validated_state(raw: Any) -> dict[str, Any]:
    """Re-validate a state payload through `TaskState` and return its canonical form.

    Lives HERE (a leaf) rather than in `contracts.py` because both the request contracts and the
    separate bookkeeping module need it, and `contracts` imports the latter: a circular import of a
    pure helper would be a cycle for no reason. Round-tripping means an activity cannot be handed a
    payload the workflow side would refuse — the same size and secret checks run here, and the dict
    that comes back is exactly the field set `TaskState` owns (no smuggled keys).
    """
    return TaskState.from_payload(raw).to_payload()


def assert_state_encodable(payload: Mapping[str, Any]) -> int:
    """Enforce the size + secret invariants on a state payload. Returns the serialized size.

    Callers get one function for both rules because they must always hold together: a check that
    can be run alone is a check that will be run alone.
    """
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_reject_non_json)
    except TypeError as exc:
        raise TaskStateError(
            f"task state is not JSON-serializable: {exc}",
            code="state_not_serializable",
            details={"error": str(exc)},
        ) from exc
    size = len(encoded.encode("utf-8"))
    if size > MAX_STATE_BYTES:
        raise TaskStateError(
            f"task state is {size} bytes, over the {MAX_STATE_BYTES}-byte limit: store a reference, "
            "not the payload.",
            code="state_too_large",
            details={"bytes": size, "limit": MAX_STATE_BYTES},
        )
    findings = find_secret_like_strings(payload)
    if findings:
        raise TaskStateError(
            "task state contains secret-like data and may not enter workflow history: "
            + ", ".join(sorted({finding["path"] for finding in findings}))
            + ".",
            code="state_contains_secret",
            details={"findings": findings},
        )
    return size


def _reject_non_json(value: Any) -> Any:
    """`json.dumps` default hook: refuse rather than stringify an unexpected object.

    Stringifying would let a `Path`, a model object or a bytes blob slip into history in a form
    nobody can decode back, which is worse than a loud failure here.
    """
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


@dataclass(frozen=True)
class CheckpointRef:
    """A committed turn: the git ref and the commit it resolves to.

    Both halves are required. A ref alone is mutable (something can move it under us) and a SHA
    alone is not discoverable; together they are an auditable "this is the tree turn N produced".
    """

    ref: str
    sha: str

    def validate(self) -> None:
        if not _REF.match(self.ref):
            raise TaskStateError(
                f"invalid checkpoint ref {self.ref!r}",
                code="invalid_checkpoint_ref",
                details={"ref": self.ref},
            )
        if not _SHA.match(self.sha):
            raise TaskStateError(
                f"invalid checkpoint sha {self.sha!r}",
                code="invalid_checkpoint_sha",
                details={"sha": self.sha},
            )

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {"ref": self.ref, "sha": self.sha}

    @classmethod
    def from_payload(cls, raw: Any) -> "CheckpointRef | None":
        if raw is None:
            return None
        data = _as_mapping(raw, "checkpoint_ref")
        ref = _require_str(data, "ref", "checkpoint_ref.ref")
        sha = _require_str(data, "sha", "checkpoint_ref.sha")
        value = cls(ref=ref, sha=sha)
        value.validate()
        return value


@dataclass(frozen=True)
class VerifyRef:
    """The outcome of a verification run, as references: the command set's digest and the log."""

    ref: str
    passed: bool
    command_digest: str
    log_path: str | None = None

    def validate(self) -> None:
        if not _REF.match(self.ref):
            raise TaskStateError(
                f"invalid verify ref {self.ref!r}", code="invalid_verify_ref", details={"ref": self.ref}
            )
        if not _SHA.match(self.command_digest):
            raise TaskStateError(
                "invalid verify command_digest (expected a hex digest)",
                code="invalid_verify_digest",
                details={"command_digest": self.command_digest},
            )
        if self.log_path is not None and not _REF.match(self.log_path):
            raise TaskStateError(
                f"invalid verify log_path {self.log_path!r}",
                code="invalid_verify_log_path",
                details={"log_path": self.log_path},
            )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "ref": self.ref,
            "passed": self.passed,
            "command_digest": self.command_digest,
            "log_path": self.log_path,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "VerifyRef | None":
        if raw is None:
            return None
        data = _as_mapping(raw, "verify_ref")
        passed = data.get("passed")
        if not isinstance(passed, bool):
            raise TaskStateError(
                "verify_ref.passed must be a boolean",
                code="invalid_verify_ref",
                details={"passed": repr(passed)},
            )
        log_path = data.get("log_path")
        if log_path is not None and not isinstance(log_path, str):
            raise TaskStateError(
                "verify_ref.log_path must be a string or null",
                code="invalid_verify_log_path",
                details={"log_path": repr(log_path)},
            )
        value = cls(
            ref=_require_str(data, "ref", "verify_ref.ref"),
            passed=passed,
            command_digest=_require_str(data, "command_digest", "verify_ref.command_digest"),
            log_path=log_path,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class PullRequestRef:
    """An opened pull request, identified both by number and by the branch it came from.

    The branch is what makes `open_pr` idempotent: a retried activity finds the existing PR by
    branch instead of opening a second one.
    """

    number: int
    url: str
    branch: str
    head_sha: str | None = None

    def validate(self) -> None:
        if not isinstance(self.number, int) or isinstance(self.number, bool) or self.number <= 0:
            raise TaskStateError(
                f"invalid pull request number {self.number!r}",
                code="invalid_pr_number",
                details={"number": repr(self.number)},
            )
        if not _URL.match(self.url):
            raise TaskStateError(
                "invalid pull request url (https only)",
                code="invalid_pr_url",
                details={"url": self.url},
            )
        if not _REF.match(self.branch):
            raise TaskStateError(
                f"invalid pull request branch {self.branch!r}",
                code="invalid_pr_branch",
                details={"branch": self.branch},
            )
        if self.head_sha is not None and not _SHA.match(self.head_sha):
            raise TaskStateError(
                "invalid pull request head_sha (expected a hex sha)",
                code="invalid_pr_head_sha",
                details={"head_sha": self.head_sha},
            )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "number": self.number,
            "url": self.url,
            "branch": self.branch,
            "head_sha": self.head_sha,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "PullRequestRef | None":
        if raw is None:
            return None
        data = _as_mapping(raw, "pr")
        number = data.get("number")
        if not isinstance(number, int) or isinstance(number, bool):
            raise TaskStateError(
                "pr.number must be an integer",
                code="invalid_pr_number",
                details={"number": repr(number)},
            )
        head_sha = data.get("head_sha")
        if head_sha is not None and not isinstance(head_sha, str):
            raise TaskStateError(
                "pr.head_sha must be a string or null",
                code="invalid_pr_head_sha",
                details={"head_sha": repr(head_sha)},
            )
        value = cls(
            number=number,
            url=_require_str(data, "url", "pr.url"),
            branch=_require_str(data, "branch", "pr.branch"),
            head_sha=head_sha,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class Budget:
    """Spend caps and what has been spent. Money is a number here, never a string.

    ``unpriced_tokens`` is carried explicitly: a run whose provider reported no price must be
    able to say "this much spend is unknown" instead of quietly treating it as $0.
    """

    max_cost_usd: float | None
    spent_usd: float
    max_tokens: int | None
    spent_tokens: int
    unpriced_tokens: int = 0

    def validate(self) -> None:
        for name, value in (
            ("max_cost_usd", self.max_cost_usd),
            ("spent_usd", self.spent_usd),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            ):
                raise TaskStateError(
                    f"budget.{name} must be a non-negative number or null",
                    code="invalid_budget",
                    details={name: repr(value)},
                )
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("spent_tokens", self.spent_tokens),
            ("unpriced_tokens", self.unpriced_tokens),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise TaskStateError(
                    f"budget.{name} must be a non-negative integer or null",
                    code="invalid_budget",
                    details={name: repr(value)},
                )

    @property
    def exhausted(self) -> bool:
        """True when either cap is met or crossed. Deterministic: no clock, no I/O."""
        if self.max_cost_usd is not None and self.spent_usd >= self.max_cost_usd:
            return True
        if self.max_tokens is not None and self.spent_tokens >= self.max_tokens:
            return True
        return False

    def with_spend(self, *, cost_usd: float, tokens: int, priced: bool = True) -> "Budget":
        """Fold one turn's spend in. `priced=False` records the tokens but not a dollar figure."""
        if cost_usd < 0 or tokens < 0:
            raise TaskStateError(
                "spend deltas must be non-negative",
                code="invalid_budget",
                details={"cost_usd": cost_usd, "tokens": tokens},
            )
        return replace(
            self,
            spent_usd=self.spent_usd + (cost_usd if priced else 0.0),
            spent_tokens=self.spent_tokens + tokens,
            unpriced_tokens=self.unpriced_tokens + (0 if priced else tokens),
        )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "max_cost_usd": self.max_cost_usd,
            "spent_usd": self.spent_usd,
            "max_tokens": self.max_tokens,
            "spent_tokens": self.spent_tokens,
            "unpriced_tokens": self.unpriced_tokens,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "Budget":
        data = _as_mapping(raw, "budget")
        value = cls(
            max_cost_usd=_optional_number(data.get("max_cost_usd"), "budget.max_cost_usd"),
            spent_usd=_optional_number(data.get("spent_usd"), "budget.spent_usd") or 0.0,
            max_tokens=_optional_int(data.get("max_tokens"), "budget.max_tokens"),
            spent_tokens=_optional_int(data.get("spent_tokens"), "budget.spent_tokens") or 0,
            unpriced_tokens=_optional_int(data.get("unpriced_tokens"), "budget.unpriced_tokens") or 0,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class PendingApproval:
    """A human decision the run is parked on.

    ``identity`` is who must act; it is recorded when the approval is requested so an Update from
    somebody else can be refused instead of silently accepted. The resume value carries the same
    identity, which is how the workflow proves the approval it resumed on is the one it asked for.
    """

    kind: Literal["approve", "budget", "policy", "ask"]
    reason: str
    requested_at_index: int
    identity: str | None = None

    def validate(self) -> None:
        if self.kind not in ("approve", "budget", "policy", "ask"):
            raise TaskStateError(
                f"invalid approval kind {self.kind!r}",
                code="invalid_pending_approval",
                details={"kind": self.kind},
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TaskStateError(
                "approval reason must be a non-empty string",
                code="invalid_pending_approval",
                details={"reason": repr(self.reason)},
            )
        if not isinstance(self.requested_at_index, int) or isinstance(self.requested_at_index, bool):
            raise TaskStateError(
                "approval requested_at_index must be an integer",
                code="invalid_pending_approval",
                details={"requested_at_index": repr(self.requested_at_index)},
            )
        if self.requested_at_index < 0:
            raise TaskStateError(
                "approval requested_at_index must be >= 0",
                code="invalid_pending_approval",
                details={"requested_at_index": self.requested_at_index},
            )
        if self.identity is not None and (not isinstance(self.identity, str) or not self.identity.strip()):
            raise TaskStateError(
                "approval identity must be a non-empty string or null",
                code="invalid_pending_approval",
                details={"identity": repr(self.identity)},
            )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind,
            "reason": self.reason,
            "requested_at_index": self.requested_at_index,
            "identity": self.identity,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "PendingApproval | None":
        if raw is None:
            return None
        data = _as_mapping(raw, "pending_approval")
        identity = data.get("identity")
        if identity is not None and not isinstance(identity, str):
            raise TaskStateError(
                "pending_approval.identity must be a string or null",
                code="invalid_pending_approval",
                details={"identity": repr(identity)},
            )
        value = cls(
            kind=_require_str(data, "kind", "pending_approval.kind"),  # type: ignore[arg-type]
            reason=_require_str(data, "reason", "pending_approval.reason"),
            requested_at_index=_require_int(data, "requested_at_index", "pending_approval.requested_at_index"),
            identity=identity,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class Attempts:
    """Per-stage attempt counters. Bounded by the caller's policy, recorded here so replay sees them.

    `implement` is the one that matters for the plan's verify-fails-once / review-rejects-once
    loop; the others exist so a bounded retry in any stage is observable and testable, and so the
    budget/attempt ceiling can be evaluated from state alone.
    """

    plan: int = 0
    implement: int = 0
    verify: int = 0
    review: int = 0
    open_pr: int = 0

    def validate(self) -> None:
        for name in ("plan", "implement", "verify", "review", "open_pr"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TaskStateError(
                    f"attempts.{name} must be a non-negative integer",
                    code="invalid_attempts",
                    details={name: repr(value)},
                )

    def incremented(self, stage: Literal["plan", "implement", "verify", "review", "open_pr"]) -> "Attempts":
        return replace(self, **{stage: getattr(self, stage) + 1})

    @property
    def total(self) -> int:
        return self.plan + self.implement + self.verify + self.review + self.open_pr

    def to_payload(self) -> dict[str, int]:
        self.validate()
        return {
            "plan": self.plan,
            "implement": self.implement,
            "verify": self.verify,
            "review": self.review,
            "open_pr": self.open_pr,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "Attempts":
        if raw is None:
            return cls()
        data = _as_mapping(raw, "attempts")
        value = cls(
            plan=_optional_int(data.get("plan"), "attempts.plan") or 0,
            implement=_optional_int(data.get("implement"), "attempts.implement") or 0,
            verify=_optional_int(data.get("verify"), "attempts.verify") or 0,
            review=_optional_int(data.get("review"), "attempts.review") or 0,
            open_pr=_optional_int(data.get("open_pr"), "attempts.open_pr") or 0,
        )
        value.validate()
        return value


@dataclass(frozen=True)
class AskRef:
    """A decision the agent parked on, as a REFERENCE (D4).

    The engine's terminal event names the ask (`ask_id`, `kind`, `tool`, `prompt`). Only the reference
    belongs in workflow history: the tool payload it is about can be arbitrarily large and
    attacker-influenced, and history has a hard ceiling. `prompt` is kept because it is the one part a
    human needs in order to decide, and the engine has already bounded and redacted it.
    """

    # The engine's own kind ("command" | "tool" | "followup"); unknown values are refused rather than
    # coerced, because "followup" answers a question and "tool" releases a blocked call.
    kind: Literal["command", "tool", "followup"]
    ask_id: str
    tool: str
    prompt: str

    def validate(self) -> None:
        if self.kind not in ("command", "tool", "followup"):
            raise TaskStateError(
                f"invalid ask kind {self.kind!r}",
                code="invalid_ask_ref",
                details={"kind": self.kind},
            )
        for name, value in (("ask_id", self.ask_id), ("tool", self.tool)):
            if not isinstance(value, str) or not value.strip():
                raise TaskStateError(
                    f"ask.{name} must be a non-empty string",
                    code="invalid_ask_ref",
                    details={name: repr(value)},
                )
        if not _REF.match(self.ask_id):
            raise TaskStateError(
                f"invalid ask.ask_id {self.ask_id!r}",
                code="invalid_ask_ref",
                details={"ask_id": self.ask_id},
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise TaskStateError(
                "ask.prompt must be a non-empty string",
                code="invalid_ask_ref",
                details={"prompt": repr(self.prompt)[:120]},
            )

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {
            "kind": self.kind,
            "ask_id": self.ask_id,
            "tool": self.tool,
            "prompt": self.prompt,
        }

    @classmethod
    def from_payload(cls, raw: Any) -> "AskRef | None":
        if raw is None:
            return None
        data = _as_mapping(raw, "ask")
        # `kind` is read with the ASK's own error code: a caller decoding a malformed ask should be
        # told which reference is wrong, not that "the task state" is.
        kind = data.get("kind")
        if not isinstance(kind, str) or kind not in ("command", "tool", "followup"):
            raise TaskStateError(
                f"ask.kind must be one of command, tool, followup (got {kind!r})",
                code="invalid_ask_ref",
                details={"kind": repr(kind)},
            )
        try:
            value = cls(
                kind=kind,
                ask_id=_require_str(data, "ask_id", "ask.ask_id"),
                tool=_require_str(data, "tool", "ask.tool"),
                prompt=_require_str(data, "prompt", "ask.prompt"),
            )
        except TaskStateError as exc:
            # Re-label the shared coercion helpers' generic code: the failure IS a bad ask reference.
            if exc.code == "invalid_task_state":
                raise TaskStateError(str(exc), code="invalid_ask_ref", details=exc.details) from exc
            raise
        value.validate()
        return value


@dataclass(frozen=True)
class TaskState:
    """Everything one task workflow remembers between activities — and nothing more.

    All payload-bearing fields are references (`plan_ref`, `checkpoint_ref`, `verify_ref`,
    `session_id` + `last_event_index`). The `budget` and `attempts` are small bounded scalars; the
    verdict and the PR are identifiers. Nothing here can grow with the size of the repository, the
    conversation or the diff, which is what keeps a run's history under the 10,240-event ceiling.

    ``turn_seq`` is the durable-turn counter: every paid turn increments it exactly once and its
    value is the `turn_key` suffix (``{run}:{task}:{turn_seq}``). That is what makes a re-executed
    activity after a worker crash replay the *same* turn instead of buying a new one.

    ``last_turn_id`` is the engine's id for the turn ``last_turn_key`` refers to, recorded as soon as
    the engine accepts the POST (see `started_turn`). ``turn_key`` replay alone only protects a turn
    whose session still exists: the engine's turn-key index is per session, so a session re-created
    after a restart gives the same key a fresh, empty index and a second paid turn. Recording the turn
    id lets a retry attach to THAT turn instead of asking the engine again — correct whether or not
    the session survived. The returned state cannot by itself serve a retry (a retry is handed the
    pre-turn state from history), so the same pair is also written to the LEDGER as `turn.started`;
    this field carries it forward to the checkpoint, verify and review steps, which do read it.
    """

    run_id: str
    task_id: str
    turn_seq: int = 0
    session_id: str | None = None
    last_turn_key: str | None = None
    last_turn_id: str | None = None
    last_event_index: int = 0
    plan_ref: str | None = None
    checkpoint_ref: CheckpointRef | None = None
    verify_ref: VerifyRef | None = None
    review_verdict: ReviewVerdict | None = None
    pr: PullRequestRef | None = None
    budget: Budget = field(default_factory=lambda: Budget(None, 0.0, None, 0, 0))
    pending_approval: PendingApproval | None = None
    attempts: Attempts = field(default_factory=Attempts)
    schema_version: int = STATE_SCHEMA_VERSION
    # D4: the engine ASK this turn stopped on, when it stopped on one.
    #
    # A REFERENCE, not the conversation: the ask id, its kind and the one-line prompt the engine already
    # bounded. It is what makes "this task is parked because the agent needs an approval" visible in
    # workflow history and in `status` — without it, a turn that ended on a question would be
    # indistinguishable from a turn that finished, and the router would send it to `verify`, i.e. it
    # would try to validate work that was never done.
    ask: "AskRef | None" = None
    # WHY publishing was skipped: a closed set from `publish.py`. `None` is the normal case (a pull
    # request was opened, or the publish step has not run). Without this field a SKIP and "never ran"
    # would both be ``pr is None``. A REFUSAL is NOT recorded — it is an error the user must fix.
    publish_skip: "PublishSkipReason | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise TaskStateError(
                "run_id must be a non-empty string", code="invalid_task_state", details={"run_id": repr(self.run_id)}
            )
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TaskStateError(
                "task_id must be a non-empty string",
                code="invalid_task_state",
                details={"task_id": repr(self.task_id)},
            )

    # -- validation ---------------------------------------------------------------

    def validate(self) -> None:
        """Validate every reference. Called by both `to_payload` and `from_payload`."""
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise TaskStateError(
                f"unsupported task state schema version {self.schema_version} "
                f"(this build speaks {STATE_SCHEMA_VERSION})",
                code="unsupported_state_schema",
                details={"schema_version": self.schema_version, "supported": STATE_SCHEMA_VERSION},
            )
        if not isinstance(self.turn_seq, int) or isinstance(self.turn_seq, bool) or self.turn_seq < 0:
            raise TaskStateError(
                "turn_seq must be a non-negative integer",
                code="invalid_turn_seq",
                details={"turn_seq": repr(self.turn_seq)},
            )
        if (
            not isinstance(self.last_event_index, int)
            or isinstance(self.last_event_index, bool)
            or self.last_event_index < 0
        ):
            raise TaskStateError(
                "last_event_index must be a non-negative integer",
                code="invalid_last_event_index",
                details={"last_event_index": repr(self.last_event_index)},
            )
        for name in ("session_id", "last_turn_key", "last_turn_id", "plan_ref"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskStateError(
                    f"{name} must be a non-empty string or null",
                    code="invalid_task_state",
                    details={name: repr(value)},
                )
        if self.session_id is not None and not _REF.match(self.session_id):
            raise TaskStateError(
                f"invalid session_id {self.session_id!r}",
                code="invalid_session_id",
                details={"session_id": self.session_id},
            )
        # A turn id is ``turn_<session id>_<n>`` in the engine, so it is longer than the session id it
        # embeds: it may not be checked against `_REF`'s 255-char bound, only against its alphabet.
        if self.last_turn_id is not None and not _TURN_ID.match(self.last_turn_id):
            raise TaskStateError(
                f"invalid last_turn_id {self.last_turn_id!r}",
                code="invalid_turn_id",
                details={"last_turn_id": self.last_turn_id},
            )
        if self.last_turn_id is not None and (self.last_turn_key is None or self.session_id is None):
            # The pair is the record of "turn X started in session S": a turn id with neither the key
            # nor the session cannot be matched to the ``(run, task, turn_seq)`` a retry asks about.
            raise TaskStateError(
                "a state with last_turn_id must also carry last_turn_key and session_id",
                code="invalid_task_state",
                details={
                    "last_turn_id": self.last_turn_id,
                    "last_turn_key": self.last_turn_key,
                    "session_id": self.session_id,
                },
            )
        if self.plan_ref is not None and not _REF.match(self.plan_ref):
            raise TaskStateError(
                f"invalid plan_ref {self.plan_ref!r}",
                code="invalid_plan_ref",
                details={"plan_ref": self.plan_ref},
            )
        if self.review_verdict is not None and self.review_verdict not in (
            "approved",
            "rejected",
            "changes_requested",
        ):
            raise TaskStateError(
                f"invalid review_verdict {self.review_verdict!r}",
                code="invalid_review_verdict",
                details={"review_verdict": self.review_verdict},
            )
        if self.checkpoint_ref is not None:
            self.checkpoint_ref.validate()
        if self.verify_ref is not None:
            self.verify_ref.validate()
        if self.pr is not None:
            self.pr.validate()
        self.budget.validate()
        self.attempts.validate()
        if self.pending_approval is not None:
            self.pending_approval.validate()
        if self.ask is not None:
            self.ask.validate()
        # The publish-skip rules live in `publish.py` with the error type INJECTED, the same way
        # `state_coercion` does it: `TaskStateError` is defined HERE, so importing it back is a cycle.
        validate_publish_skip(self.publish_skip, pr=self.pr, fail=TaskStateError)

    # -- serialization ------------------------------------------------------------

    def to_payload(self) -> dict[str, object]:
        """Serialize for a workflow/activity boundary, refusing an oversized or secret-bearing state."""
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "turn_seq": self.turn_seq,
            "session_id": self.session_id,
            "last_turn_key": self.last_turn_key,
            "last_turn_id": self.last_turn_id,
            "last_event_index": self.last_event_index,
            "plan_ref": self.plan_ref,
            "checkpoint_ref": self.checkpoint_ref.to_payload() if self.checkpoint_ref else None,
            "verify_ref": self.verify_ref.to_payload() if self.verify_ref else None,
            "review_verdict": self.review_verdict,
            "pr": self.pr.to_payload() if self.pr else None,
            "budget": self.budget.to_payload(),
            "pending_approval": self.pending_approval.to_payload() if self.pending_approval else None,
            "attempts": self.attempts.to_payload(),
            "ask": self.ask.to_payload() if self.ask else None,
            "publish_skip": self.publish_skip,
        }
        assert_state_encodable(payload)
        return payload

    @classmethod
    def from_payload(cls, raw: Any) -> "TaskState":
        """Decode a state payload, validating it before it can be trusted by workflow code."""
        data = _as_mapping(raw, "task_state")
        size = assert_state_encodable(data)
        schema = data.get("schema_version", STATE_SCHEMA_VERSION)
        if not isinstance(schema, int) or isinstance(schema, bool):
            raise TaskStateError(
                "schema_version must be an integer",
                code="unsupported_state_schema",
                details={"schema_version": repr(schema)},
            )
        state = cls(
            schema_version=schema,
            run_id=_require_str(data, "run_id", "run_id"),
            task_id=_require_str(data, "task_id", "task_id"),
            turn_seq=_optional_int(data.get("turn_seq"), "turn_seq") or 0,
            session_id=_optional_str(data.get("session_id"), "session_id"),
            last_turn_key=_optional_str(data.get("last_turn_key"), "last_turn_key"),
            last_turn_id=_optional_str(data.get("last_turn_id"), "last_turn_id"),
            last_event_index=_optional_int(data.get("last_event_index"), "last_event_index") or 0,
            plan_ref=_optional_str(data.get("plan_ref"), "plan_ref"),
            checkpoint_ref=CheckpointRef.from_payload(data.get("checkpoint_ref")),
            verify_ref=VerifyRef.from_payload(data.get("verify_ref")),
            review_verdict=_optional_str(data.get("review_verdict"), "review_verdict"),  # type: ignore[arg-type]
            pr=PullRequestRef.from_payload(data.get("pr")),
            budget=Budget.from_payload(data.get("budget") or {}),
            pending_approval=PendingApproval.from_payload(data.get("pending_approval")),
            attempts=Attempts.from_payload(data.get("attempts")),
            ask=AskRef.from_payload(data.get("ask")),
            # Absent in a payload recorded before publishing could be skipped: `None` is what such a
            # history meant (its `open_pr` either opened a pull request or failed).
            publish_skip=publish_skip_from_payload(data.get("publish_skip"), fail=TaskStateError),
        )
        state.validate()
        _ = size  # already enforced; kept for the reader: the decode path checks the same bound
        return state

    # -- derived helpers ----------------------------------------------------------

    @property
    def turn_key(self) -> str:
        """The idempotency key for the NEXT turn. Deterministic in state, so a retry reuses it."""
        return f"{self.run_id}:{self.task_id}:{self.turn_seq + 1}"

    def started_turn(self, *, session_id: str, turn_id: str) -> "TaskState":
        """Advance to the next paid turn AND record the engine's id for it, in one transition.

        ``turn_id`` is recorded at the moment the engine accepts the POST — before the stream is read
        — so the state a completed activity returns carries the identity of the turn it bought. That
        is what a later reader (or a re-attach after a worker restart) matches a ``turn_key`` against
        instead of asking the engine, whose own replay index is per session and disappears with it.
        """
        if not isinstance(turn_id, str) or not turn_id:
            raise TaskStateError(
                "turn_id must be a non-empty string",
                code="invalid_turn_id",
                details={"turn_id": repr(turn_id)},
            )
        updated = replace(
            self,
            turn_seq=self.turn_seq + 1,
            session_id=session_id,
            last_turn_key=self.turn_key,
            last_turn_id=turn_id,
        )
        updated.validate()
        return updated

    def with_events(self, last_event_index: int) -> "TaskState":
        """Record how far the turn stream was consumed, so a reconnect asks for `?after=`."""
        if last_event_index < self.last_event_index:
            raise TaskStateError(
                f"event index went backwards ({last_event_index} < {self.last_event_index})",
                code="event_index_regressed",
                details={"new": last_event_index, "previous": self.last_event_index},
            )
        return replace(self, last_event_index=last_event_index)

    def with_spend(self, *, cost_usd: float, tokens: int, priced: bool = True) -> "TaskState":
        return replace(self, budget=self.budget.with_spend(cost_usd=cost_usd, tokens=tokens, priced=priced))

    def awaiting(self, approval: PendingApproval) -> "TaskState":
        return replace(self, pending_approval=approval)

    def resumed(self) -> "TaskState":
        """Clear the pending approval — the resume value has been applied."""
        return replace(self, pending_approval=None)

    def state_digest(self) -> str:
        """Stable digest of the reference set, for ledger/e2e correlation."""
        encoded = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# -- coercion helpers (see state_coercion.py; bound here so the error type stays defined once) ---
#
# The helpers live in their own module because `state.py` crossed the 1000-line ceiling when `AskRef`
# landed, and the boundary is genuine: they are private (no `from .state import` anywhere names them)
# and share one dependency, `TaskStateError`. It is BOUND rather than imported back, so `state_coercion`
# never imports `state` and the error class keeps exactly one definition.
_COERCION = Coercion(TaskStateError)

_as_mapping = _COERCION.as_mapping
_require_str = _COERCION.require_str
_optional_str = _COERCION.optional_str
_require_int = _COERCION.require_int
_optional_int = _COERCION.optional_int
_optional_number = _COERCION.optional_number


# Exposed for tests: the field list is the contract with the plan, so a drifted field is visible.
STATE_FIELDS: Final[tuple[str, ...]] = tuple(field.name for field in fields(TaskState))
