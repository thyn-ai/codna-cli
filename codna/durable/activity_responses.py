"""The activity RESPONSE contracts — what every activity hands back.

Split from `contracts.py` at the point where the module crossed the repository's 1000-line ceiling
(`test_modularity.py`). The split is by concern, not by size: a request says what to do, a response
says what happened, and the two are read by different code — requests are built by workflow steps,
responses are decoded by the workflow's budget arithmetic and the ledger.

Two shapes of response, deliberately different:

* `ActivityOutcome` is what a state-transforming activity returns: the new `TaskState` payload plus
  this call's spend as a DELTA, plus small diagnostics.
* The others (`CheckpointOutcome`, `PullRequestOutcome`, `EpisodeOutcome`, `NotifyOutcome`) are
  narrower because their callers need one specific fact — a commit sha, a PR number, a digest — and a
  wide response would let a caller read something that is not part of the contract.

Everything here is re-exported from `contracts`, so callers keep importing from one module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping

from .contract_primitives import ContractError, optional_cap

__all__ = [
    "ActivityOutcome",
    "CheckpointOutcome",
    "EpisodeOutcome",
    "NotifyOutcome",
    "PullRequestOutcome",
]


@dataclass(frozen=True)
class ActivityOutcome:
    """What every state-transforming activity returns.

    ``state`` is the new `TaskState` payload. Spend is returned as a DELTA rather than folded in,
    so the workflow's budget arithmetic is workflow code: replayable, auditable, and impossible to
    double count when an activity is retried (the delta is recomputed from the ledger's record of
    the previous turn, which a retry does not change).

    ``details`` is diagnostics for the ledger (a turn id, a terminal event name) — never state, and
    never anything a later step reads. It is size-bounded so a chatty activity cannot inflate
    history.
    """

    state: dict[str, Any]
    cost_usd: float = 0.0
    tokens: int = 0
    priced: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        from .contract_primitives import require_int
        from .contracts import validated_state

        state = validated_state(self.state)
        optional_cap(self.cost_usd, "cost_usd")
        require_int(self.tokens, "tokens", minimum=0)
        if not isinstance(self.details, Mapping):
            raise ContractError(
                "details must be a mapping",
                code="contract_invalid",
                details={"field": "details"},
            )
        if len(self.details) > 32:
            raise ContractError(
                "details carries more than 32 keys",
                code="contract_invalid",
                details={"keys": len(self.details)},
            )
        return state


@dataclass(frozen=True)
class CheckpointOutcome:
    """The commit a turn produced, and whether this call created it.

    ``created=False`` is the idempotent path: the ref already pointed at this tree (a retried
    activity, or a turn that changed nothing), so no second commit exists.
    """

    state: dict[str, Any]
    ref: str
    sha: str
    created: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        from .contract_primitives import require_ref
        from .contracts import validated_state

        state = validated_state(self.state)
        require_ref(self.ref, "ref")
        require_ref(self.sha, "sha")
        if not isinstance(self.created, bool):
            raise ContractError(
                "created must be a boolean", code="contract_invalid", details={"field": "created"}
            )
        return state


@dataclass(frozen=True)
class PullRequestOutcome:
    state: dict[str, Any]
    details: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        from .contracts import validated_state

        return validated_state(self.state)


@dataclass(frozen=True)
class EpisodeOutcome:
    path: str
    digest: str
    ledger_seq: int | None


@dataclass(frozen=True)
class NotifyOutcome:
    seq: int | None
    kind: str


# The five response types, for a caller that validates or decodes by name.
RESPONSE_TYPES: Final[dict[str, type]] = {
    "activity": ActivityOutcome,
    "checkpoint": CheckpointOutcome,
    "pull_request": PullRequestOutcome,
    "episode": EpisodeOutcome,
    "notify": NotifyOutcome,
}
