"""`TaskContext`: the part of a task an ACTIVITY needs, as a leaf type (D5).

Split out of `contracts.py` for two reinforcing reasons, and this is the one that made it necessary:

1. **Size.** `contracts.py` was at the repository's 1000-line ceiling (`test_modularity.py`).
2. **Temporal DECODES request types from their annotations.** A request field annotated with a forward
   reference — or with ``Any`` — is decoded as a plain dict by a worker replaying it, and then an
   activity that calls ``context.validate()`` dies with ``'dict' object has no attribute 'validate'``.
   Measured while splitting this file: a `TYPE_CHECKING` import produced exactly that on every
   ``episode`` activity. The type must therefore be a REAL, importable module attribute.

It is a LEAF: stdlib, `dataclasses`, `contract_primitives` and `provider`. It deliberately does not
import `contracts` — that module imports THIS one, and the cycle is what would bring the decode
problem back.

**Why the split is a real boundary, not a size dodge.** A context is what TRAVELS WITH A REQUEST: the
task's identity, the shared checkout, and the policy fields that must survive a retry (the provider
coordinates, whether the task works in its own worktree, whether its turns carry project rules). The
requests that CARRY it live in `contracts.py`. A reviewer asking "what does a retried activity
receive" reads this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contract_primitives import ContractError, GraphEngine, require_engine, require_ref
from .provider import canonicalise_provider_fields as _canonicalise_provider_fields
from .provider import validate_provider_fields as _validate_provider_fields

__all__ = ["TaskContext"]


@dataclass(frozen=True)
class TaskContext:
    """The part of a `TaskSpec` an activity needs, without the brief.

    Split out so a request does not repeat the brief once per activity: a task with ten turns would
    otherwise put the brief in history ten times, and history is the thing that has a hard ceiling.
    """

    run_id: str
    task_id: str
    task_number: int
    # The SHARED checkout the run was started against. It stays the repository every git command is
    # anchored to (a worktree's objects and refs live here), but it is not necessarily where a turn
    # EDITS: when ``worktree`` is set, the task's own checkout is derived from this plus the run id and
    # the task id, and `activity_support.working_dir` is what a step asks for. Keeping the base here
    # means a spec recorded before per-task worktrees existed still decodes to a usable context.
    repo_dir: str
    base_branch: str
    engine: GraphEngine = "langgraph"
    # The provider coordinates travel WITH the context, because they are what the turn activity turns
    # into a create-session body. They are here rather than only on the spec so a retried activity gets
    # them from its own request — a request is what history hands back.
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    provider_key_source: str | None = None
    # D5: work in this task's own worktree rather than in ``repo_dir``. Travelling WITH the context is
    # what makes a retried activity's answer independent of anything the worker remembers.
    worktree: bool = False
    # D5 goal mode: which checkout, when it is not this task's own (a "fix CI" attempt shares its
    # parent's). Travelling WITH the context for the same reason `worktree` does: a retried activity
    # must derive the same path from its own request, never from anything the worker remembers.
    worktree_task_id: str = ""
    # D5: attach the repository's own conventions to this task's turns. The POLICY, not the text —
    # see `TaskSpec.attach_project_rules`.
    attach_project_rules: bool = True
    # Whether this task's turns are given the repository's user-defined subagents. The POLICY, not the
    # roster — see `TaskSpec.attach_subagents`. Travelling WITH the context is what makes a retried
    # activity's roster independent of anything the worker remembers.
    attach_subagents: bool = True
    # D5/MCP: the servers this task's turns are given — validated CONFIGURATION, with `env` entries as
    # `{"from_env": "NAME"}` references rather than values (see `TaskSpec.mcp_servers`). It travels on
    # the context for the same reason the provider coordinates do: a retried activity must build the
    # same session from its OWN request, never from something the worker remembers.
    mcp_servers: tuple[dict[str, Any], ...] = ()
    # D7/SWARM: whether THIS task's turns may delegate. Travels on the context for the same reason the
    # provider coordinates do: a retried activity must build the same session from its OWN request.
    # Each flag is independent — see `TaskSpec.enable_spawn_agent`.
    enable_spawn_agent: bool = False
    enable_agent_teams: bool = False
    # D5/X8: the EXISTING branch this task works in (run-from-a-pull-request), or `""` for its own.
    # It travels on the context for the same reason `worktree` does: `ensure_checkout` and every
    # branch-aware step must derive the same answer from the request the activity was handed.
    worktree_branch: str = ""

    def __post_init__(self) -> None:
        # Same canonicalisation as `TaskSpec`, for the same reason: a context decoded from a request
        # must resolve to the value the spec recorded, or two attempts at one turn could disagree
        # about which provider they are running on.
        _canonicalise_provider_fields(self)

    @classmethod
    def from_spec(cls, spec: Any) -> "TaskContext":
        return cls(
            run_id=spec.run_id,
            task_id=spec.task_id,
            task_number=spec.task_number,
            repo_dir=spec.repo_dir,
            base_branch=spec.base_branch,
            engine=spec.engine,
            provider=spec.provider,
            model=spec.model,
            base_url=spec.base_url,
            provider_key_source=spec.provider_key_source,
            worktree=spec.worktree,
            worktree_task_id=spec.worktree_task_id,
            attach_project_rules=spec.attach_project_rules,
            attach_subagents=spec.attach_subagents,
            mcp_servers=tuple(spec.mcp_servers),
            enable_spawn_agent=bool(spec.enable_spawn_agent),
            enable_agent_teams=bool(spec.enable_agent_teams),
            worktree_branch=spec.worktree_branch,
        )

    def provider_fields(self) -> dict[str, str | None]:
        """The four provider coordinates, so a caller does not have to know their names."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "provider_key_source": self.provider_key_source,
        }

    def validate(self) -> None:
        require_ref(self.run_id, "context.run_id")
        require_ref(self.task_id, "context.task_id")
        if not isinstance(self.task_number, int) or isinstance(self.task_number, bool) or self.task_number < 1:
            raise ContractError(
                "context.task_number must be an integer >= 1",
                code="contract_invalid",
                details={"field": "task_number", "value": repr(self.task_number)},
            )
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "context.repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir"},
            )
        require_ref(self.base_branch, "context.base_branch")
        require_engine(self.engine)
        _validate_provider_fields(self)
        if not isinstance(self.worktree, bool):
            raise ContractError(
                "context.worktree must be a boolean",
                code="contract_invalid",
                details={"field": "worktree", "value": repr(self.worktree)},
            )
        if not isinstance(self.worktree_task_id, str):
            raise ContractError(
                "context.worktree_task_id must be a string",
                code="contract_invalid",
                details={"field": "worktree_task_id"},
            )
        if self.worktree_task_id:
            require_ref(self.worktree_task_id, "context.worktree_task_id")
        if not isinstance(self.attach_project_rules, bool):
            raise ContractError(
                "context.attach_project_rules must be a boolean",
                code="contract_invalid",
                details={"field": "attach_project_rules"},
            )
        if not isinstance(self.attach_subagents, bool):
            raise ContractError(
                "context.attach_subagents must be a boolean",
                code="contract_invalid",
                details={"field": "attach_subagents"},
            )
        # The MCP config is re-validated from the context too: a context is what a REQUEST carries, and
        # a request that crossed a channel is still untrusted. The same normalizer the spec uses, so
        # the two cannot disagree about what a valid MCP config is.
        from .mcp import McpConfigError, normalize_mcp_servers

        try:
            normalize_mcp_servers(self.mcp_servers)
        except McpConfigError as exc:
            raise ContractError(
                str(exc), code=exc.code, details=exc.details
            ) from exc
        # D7/SWARM: re-validated from the CONTEXT too, for the same reason the MCP config is: a context
        # is what a request carries, and a request that crossed a channel is still untrusted.
        from .swarm import SwarmConfigError, normalize_swarm

        try:
            normalize_swarm(
                enable_spawn_agent=self.enable_spawn_agent,
                enable_agent_teams=self.enable_agent_teams,
            )
        except SwarmConfigError as exc:
            raise ContractError(str(exc), code=exc.code, details=exc.details) from exc
        if not isinstance(self.worktree_branch, str):
            raise ContractError(
                "context.worktree_branch must be a string",
                code="contract_invalid",
                details={"field": "worktree_branch"},
            )

    @property
    def branch(self) -> str:
        if self.worktree_branch:
            return self.worktree_branch.strip()
        return f"codna/run-{self.run_id}/task-{self.task_number}"

    @property
    def checkout_task_id(self) -> str:
        """The task id whose worktree this context works in (its own, unless it is a repair attempt)."""
        return self.worktree_task_id or self.task_id

    # The REQUEST builders. Each imports its request class HERE rather than at module scope: the
    # classes live in `contracts`, which imports this module (it re-exports `TaskContext`), so a
    # module-scope import would close a cycle — and the cycle would surface as a workflow failure
    # (`NameError: PlanRequest is not defined`) only once a turn actually ran. Resolved per call,
    # after `contracts` is fully loaded, which is always true by the time a builder is used.
    #
    # These exist so a request's payload cannot disagree with its context: the context IS the field
    # value, so a caller cannot pair task A's provider coordinates with task B's worktree.

    def plan_request(
        self, brief: str, state: Mapping[str, Any], steer: Sequence[str] = (), goal_context: str = ""
    ) -> Any:
        from .contracts import PlanRequest

        return PlanRequest(
            state=dict(state),
            context=self,
            brief=brief,
            steer=tuple(steer),
            goal_context=goal_context,
        )

    def turn_request(
        self, prompt: str, state: Mapping[str, Any], steer: Sequence[str] = ()
    ) -> Any:
        from .contracts import TurnRequest

        return TurnRequest(state=dict(state), context=self, prompt=prompt, steer=tuple(steer))

    def checkpoint_request(self, state: Mapping[str, Any]) -> Any:
        from .contracts import CheckpointRequest

        return CheckpointRequest(state=dict(state), context=self)

    def verify_request(
        self, state: Mapping[str, Any], commands: Sequence[str] = ()
    ) -> Any:
        from .contracts import VerifyRequest

        return VerifyRequest(state=dict(state), context=self, commands=tuple(commands))

    def review_request(self, state: Mapping[str, Any], base_sha: str | None = None) -> Any:
        from .contracts import ReviewRequest

        return ReviewRequest(state=dict(state), context=self, base_sha=base_sha)

    def pull_request_request(
        self, title: str, state: Mapping[str, Any], body: str = ""
    ) -> Any:
        from .contracts import PullRequestRequest

        return PullRequestRequest(state=dict(state), context=self, title=title, body=body)


