"""The wire contract between the durable workflows and their activities — a LEAF module.

Every value that crosses a workflow boundary must be describable on both sides, and the workflow
side is a Temporal sandbox: it may not import `httpx`, `subprocess` or anything that reaches
`urllib.request` (`tests/durable/test_import_hygiene.py` is the guard). `activities.py` needs all
of those; `workflows.py` needs the shapes it passes over the wire. If the shapes lived in
`activities.py`, importing them from a workflow would drag the HTTP stack into the sandbox and kill
the workflow at worker construction — exactly the failure that produced `version.py`.

So the shapes live here: stdlib, `dataclasses`, and `typing` only (plus `state.py`, which the
hygiene test already proves sandbox-clean).

Two rules govern these types, both learned from `state.py`:

* **References, not payloads.** A `TaskSpec` carries a brief and a repo path; activity requests
  carry a `TaskState` *payload* (already a set of references) plus the small scalars an activity
  needs. Transcripts, diffs and logs stay in the ledger, git and the run directory. History is
  bounded, so a request that would grow with the size of the repository is a request that will
  eventually break a run.
* **Validate on the way in.** Every request/response validates itself (`validate()`), and every
  activity re-validates the `TaskState` payload it is handed rather than trusting the caller. A
  retried activity receives the same payload from history, so an invalid one is a code bug that
  must surface immediately, not a value to coerce.

`ACTIVITY_*` timeout constants live here too so the graph's default activity options and the plain
workflow's explicit `execute_activity` calls cannot drift: the D2.0 spike proved an activity
scheduled without a `start_to_close_timeout` fails the workflow task.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Final, Literal, Mapping

from .contract_primitives import (
    MAX_STEER_NOTE_CHARS,
    MAX_STEER_NOTES,
    ContractError,
    GraphEngine,
    optional_cap as _optional_cap,
    require_engine as _require_engine,
    require_int as _require_int,
    require_ref as _require_ref,
)
# The provider coordinates live in their own LEAF module (`provider.py`): the rules are long enough
# that keeping them here would push this file past the repository's 1000-line ceiling, and they are a
# genuine leaf — stdlib plus `contract_primitives`/`state`, both already proven sandbox-clean. They
# are re-exported because every caller imports this module for the contract types, and a caller should
# not have to know which file a rule landed in. See `provider.py` for the two security properties.
from .provider import (
    KEYLESS_PROVIDERS,
    MAX_MODEL_ID_CHARS,
    MAX_PROVIDER_BASE_URL_CHARS,
    MAX_PROVIDER_ID_CHARS,
    PROVIDER_ENV_KEYS,
    PROVIDER_KEY_SOURCES,
    canonicalise_provider_fields as _canonicalise_provider_fields,
    provider_env_key_names,
    provider_session_spec,
    resolve_provider_env_key,
    validate_provider_fields as _validate_provider_fields,
)
from .state import Budget, TaskState, find_secret_like_strings, validated_state

__all__ = [
    "ACTIVITY_HEARTBEAT_TIMEOUT_S",
    "ACTIVITY_TIMEOUT_S",
    "ACTIVITY_TURN_TIMEOUT_S",
    "ACTIVITY_VERIFY_TIMEOUT_S",
    "KEYLESS_PROVIDERS",
    "MAX_MODEL_ID_CHARS",
    "MAX_PROVIDER_BASE_URL_CHARS",
    "MAX_PROVIDER_ID_CHARS",
    "MAX_GOAL_CONTEXT_CHARS",
    "PROVIDER_ENV_KEYS",
    "PROVIDER_KEY_SOURCES",
    "ActivityOutcome",
    "CheckpointOutcome",
    "CheckpointRequest",
    "ContractError",
    "EpisodeOutcome",
    "EpisodeRequest",
    "GRAPH_ENGINES",
    "GraphEngine",
    "HEARTBEAT_INTERVAL_S",
    "MAX_STEER_NOTE_CHARS",
    "MAX_STEER_NOTES",
    "NAME_CHECKPOINT",
    "NAME_EPISODE",
    "NAME_NOTIFY",
    "NAME_OPEN_PR",
    "NAME_PLAN",
    "NAME_RECALL",
    "NAME_REVIEW",
    "NAME_TURN",
    "NAME_VERIFY",
    "NAME_WATCH_CI",
    "NAME_WORKTREE_CLEANUP",
    "NodeContext",
    "NotifyOutcome",
    "NotifyRequest",
    "PlanRequest",
    "PullRequestOutcome",
    "PullRequestRequest",
    "RecallOutcome",
    "RecallRequest",
    "RecalledEpisode",
    "ReviewRequest",
    "RunCompletion",
    "RunOutcome",
    "RunProgress",
    "RunSpec",
    "TASK_SPEC_FIELDS",
    "TURN_REQUEST_FIELDS",
    "TaskCompletion",
    "TaskContext",
    "TaskOutcome",
    "TaskSpec",
    "TaskSummary",
    "TurnRequest",
    "VerifyRequest",
    "WatchCiOutcome",
    "WatchCiRequest",
    "WorktreeCleanupOutcome",
    "WorktreeCleanupRequest",
    "implement_prompt",
    "plan_prompt",
    "provider_env_key_names",
    "provider_session_spec",
    "resolve_provider_env_key",
    "task_spec_from_config",
]

# Re-exported from the two modules this file was split into when it crossed the repository's
# 1000-line ceiling (`test_modularity.py`). Kept as re-exports so every caller keeps importing one
# module: the split is about file size and reviewability, not about making callers choose.
from .activity_responses import (  # noqa: E402 - re-export, after the leaf imports it depends on
    ActivityOutcome,
    CheckpointOutcome,
    EpisodeOutcome,
    NotifyOutcome,
    PullRequestOutcome,
)
# D5 goal mode's two activity shapes. Re-exported here for the same reason the responses above are:
# every caller imports one module. Both are leaf-safe (stdlib + `goal.py`/`goal_contracts.py`, which
# are pure), so this cannot drag I/O into a workflow. `goal_steps` — the IMPLEMENTATION — is NOT
# imported here: it reaches git and subprocess, which the workflow sandbox refuses.
from .goal_contracts import (  # noqa: E402
    NAME_RECALL,
    NAME_WATCH_CI,
    RecallOutcome,
    RecallRequest,
    RecalledEpisode,
    WatchCiOutcome,
    WatchCiRequest,
)
# The run-level BOOKKEEPING requests (episode / notify / worktree cleanup). Split out at the
# repository's 1000-line ceiling (`test_modularity.py`) on a real boundary — they record or report
# something ABOUT a run, and none of them can change what a turn does. Re-exported so every caller
# keeps importing one module.
from .contracts_bookkeeping import (  # noqa: E402
    EpisodeRequest,
    NotifyRequest,
    WorktreeCleanupOutcome,
    WorktreeCleanupRequest,
)
# The PROMPT builders — "what text does the model receive", as opposed to "what shape crosses the
# wire". Split out for the same ceiling and the same review reason: a prompt change is read as a
# prompt. Re-exported so a caller that imports `implement_prompt` from here still works.
from .prompts import implement_prompt, plan_prompt, pr_title  # noqa: E402
# The graph-config decoders. Re-exported here because `graph.py` and both engines import the ONE
# contracts module, and a caller should not have to know which file a decoder landed in.
from .task_config import (  # noqa: E402
    CONFIG_STEER,
    CONFIG_TASK_SPEC,
    steer_from_config,
    task_spec_from_config,
)
# The four names above are RE-EXPORTS, not local uses: `graph.py` and both engines import the one
# contracts module, and a caller should not have to know which file a decoder landed in.
_ = (CONFIG_STEER, CONFIG_TASK_SPEC, steer_from_config, task_spec_from_config)
# `TaskContext` — a LEAF module, not merely a moved class. Temporal decodes a request from its
# annotations, so this type has to be a real importable attribute; a forward reference is decoded as
# a plain dict and every activity that validates it then dies. Re-exported for the same reason as
# everything else here: one module for callers.
from .task_context import TaskContext  # noqa: E402
from .run_contracts import (  # noqa: E402
    RunCompletion,
    RunOutcome,
    RunProgress,
    RunSpec,
    TaskCompletion,
    TaskSummary,
)


# Registered activity type names. Explicit and stable, and they live HERE (a sandbox-clean leaf)
# because both the graph and the plain workflow name them: `activities.py` registers under these
# names, `graph.py` schedules them, and a module rename must not silently orphan in-flight runs.
#
# The `codna.step.` namespace is load-bearing. The LangGraph plugin derives a node activity's type as
# ``<graph>.<node>`` — ``codna.task.open_pr`` for the ``open_pr`` node — and a worker registers the
# graph's node activities AND these. Naming a step ``codna.task.open_pr`` would therefore collide with
# the node of the same name and make `Worker(...)` raise "More than one activity named ...". The
# namespaces are kept disjoint on purpose, and `tests/durable/test_task_workflow_time_skipping.py`
# asserts it, because the collision only appears when a worker registers both engines at once.
NAME_PLAN: Final = "codna.step.plan"
NAME_TURN: Final = "codna.step.turn"
NAME_CHECKPOINT: Final = "codna.step.checkpoint"
NAME_VERIFY: Final = "codna.step.verify"
NAME_REVIEW: Final = "codna.step.review"
NAME_OPEN_PR: Final = "codna.step.open_pr"
NAME_EPISODE: Final = "codna.step.episode"
NAME_NOTIFY: Final = "codna.step.notify"

# D5 goal mode: the ceiling on the plan-side goal block a `TaskSpec` may carry. Generous for an
# objective plus four episodes, small enough that a run cannot smuggle a transcript into history.
MAX_GOAL_CONTEXT_CHARS: Final = 16_000

# The RUN-level worktree sweep (D5). A run that gave each task its own checkout must reclaim that disk
# when it finishes, or every run leaks one full checkout per task. It is scheduled by `RunWorkflow` and
# ONLY for a run that opted into worktrees, so a run whose spec predates the flag produces exactly the
# history it always did — that is what makes this addition replay-safe for in-flight runs.
NAME_WORKTREE_CLEANUP: Final = "codna.step.worktree_cleanup"

# The metadata key a workflow uses to hand the `TaskSpec` to the graph. In the config rather than
# the state because the state is a validated `TaskState` payload with a closed field set: a `brief`
# in it would be a field nothing validates, growing with the request, in the one place that must not.
GRAPH_ENGINES: Final[tuple[GraphEngine, ...]] = ("langgraph", "plain")

TaskOutcome = Literal["approved", "rejected", "budget_exhausted", "stopped"]

# Ceilings, not detectors. The turn activity heartbeats every 10s (see `activities.run_cline_turn`),
# so a wedged worker is retried seconds after it dies; these are the outer limits for a turn that is
# genuinely long (a big plan, a long implement) and the point at which Temporal stops waiting.
ACTIVITY_TIMEOUT_S: Final = 30 * 60
ACTIVITY_TURN_TIMEOUT_S: Final = 60 * 60
ACTIVITY_VERIFY_TIMEOUT_S: Final = 45 * 60
# Larger than the activity's own heartbeat period by a wide margin: one missed beat must not retry a
# paid turn (the turn_key makes a retry safe, but it still costs a round trip and a replay).
ACTIVITY_HEARTBEAT_TIMEOUT_S: Final = 2 * 60
# How often a long activity beats. A heartbeat is a server round trip, so 10s: far below every
# timeout here, and the detail payload stays tiny. One value, shared by the graph's node wrapper and
# `activities.run_cline_turn` — it lives here so neither has to import the other.
HEARTBEAT_INTERVAL_S: Final = 10.0



def _canonical_mcp_servers(value: Any) -> tuple[dict[str, Any], ...]:
    """The canonical MCP config, or the caller's value when it is already the canonical form.

    Canonicalisation happens at CONSTRUCTION (like the provider fields) so history carries the shape
    the engine sees. It does not RAISE here: an invalid config must still construct so `validate()`
    is the one place that reports a coded error, and `to_payload` calls `validate()` first.
    """
    from .mcp import McpConfigError, normalize_mcp_servers

    try:
        return tuple(dict(server) for server in normalize_mcp_servers(value))
    except McpConfigError:
        return value if isinstance(value, tuple) else tuple(value or ())


def _mcp_servers_json(value: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [dict(server) for server in value]


@dataclass(frozen=True)
class TaskSpec:
    """Everything one task needs, and nothing that grows with the repository.

    ``brief`` and ``repo_dir`` are the only free-form values; both are bounded here because this
    object is a workflow input and therefore enters history. ``task_number`` is the ``<n>`` of the
    branch name ``codna/run-<run>/task-<n>``.

    ``provider``/``model``/``base_url``/``provider_key_source`` select the MODEL PROVIDER the task's
    turns run on. They are inputs, so every one is validated and secret-screened: the spec names the
    provider, NEVER its credential (see `provider.py`).
    """

    run_id: str
    task_id: str
    task_number: int
    brief: str
    repo_dir: str
    base_branch: str = "main"
    verify_commands: tuple[str, ...] = ()
    max_attempts: int = 3
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    engine: GraphEngine = "langgraph"
    schema_version: int = 1
    # Which provider runs this task's turns. ``None`` means "the engine's own default", which is the
    # historical behaviour and is preserved so an existing run spec still replays.
    provider: str | None = None
    # The provider's model id. ``None`` means "the provider's default model" (the engine resolves it).
    model: str | None = None
    # An endpoint override for the provider (a local inference server, a self-hosted proxy).
    base_url: str | None = None
    # WHERE the provider credential comes from. ``None`` = no credential is attached by the durable
    # path, which is correct for a keyless provider and is the pre-change behaviour for a hosted one
    # (the engine then reads its own process environment, as before).
    provider_key_source: str | None = None
    # D5: whether this task works in its OWN git worktree (``<runs_root>/<run>/wt-<task>``) instead of
    # the shared checkout at ``repo_dir``. ``False`` is the pre-D5 behaviour and the default, so a spec
    # recorded before this field existed still decodes to what it meant.
    #
    # Additive and replay-safe: Temporal compares a replayed workflow's COMMANDS, not the input
    # payloads of a scheduled activity (verified against `histories/task-workflow-g1.json`: changing a
    # request's fields replays clean, while changing the number of scheduled activities raises
    # `NondeterminismError`). So this flag can ride in the spec without a `GRAPH_VERSION` bump — but the
    # ACTIVITY SEQUENCE must not change, which is why the worktree is created by the existing
    # checkpoint and turn steps rather than by a new step.
    worktree: bool = False
    # D5 GOAL MODE: the run-level context this task's PLAN turn must see — the objective plus a recap
    # of the episodes earlier tasks in the same run recorded. Built by `RunWorkflow` (deterministically,
    # from the objective and an activity's episode references) and empty for a task-mode run.
    #
    # A pre-formatted block rather than two fields, because the two are always consumed together and
    # the plan prompt is the only reader: one bounded string cannot disagree with itself. Bounded by
    # `MAX_GOAL_CONTEXT_CHARS` below, and screened for secrets for the same reason the brief is — it
    # enters workflow history.
    goal_context: str = ""
    # D5 goal mode: WHICH checkout this task works in. Empty means "its own", derived from
    # ``task_id``. A "fix CI" attempt sets it to its PARENT's id, because the fix must commit in the
    # same worktree and on the same branch — while keeping its own ``task_id`` so its turn keys,
    # checkpoint messages and episode file are the attempt's own rather than the parent's.
    #
    # A separate field rather than reusing `task_id`, because the two answer different questions:
    # "which checkout" (a place, shared by a task and its repair attempts) and "which attempt" (an
    # identity that must not collide).
    worktree_task_id: str = ""
    # D5: whether this task's turns are given the repository's own conventions (AGENTS.md,
    # `.codna/rules`, `.cursorrules`, …) in their session context. The POLICY travels; the TEXT does
    # not — it is a file's contents (unbounded, repo-controlled), so the activity reads it from the
    # tree it is about to edit. ``True`` is the default because a run that ignores a repository's
    # written conventions is the bug this closes.
    attach_project_rules: bool = True
    # D5/MCP: the MCP servers this task's turns are given, as validated CONFIGURATION (see
    # `mcp.normalize_mcp_servers`): a name, a transport, and — for a stdio server — `env` entries that
    # are REFERENCES (`{"from_env": "NAME"}`), never secret values. A `TaskSpec` is workflow history and
    # therefore the ledger, so this is the same rule `provider_key_source` follows: the spec says WHERE
    # a credential comes from and the activity resolves it from the worker's own environment.
    #
    # `()` is "no servers", the answer for every spec recorded before this field existed, and it is
    # why the default is empty rather than None: a tuple is what `to_payload` round-trips.
    mcp_servers: tuple[dict[str, Any], ...] = ()
    # D7/SWARM: whether this task's turns may spawn a subagent and/or form a team. The engine supports
    # both but gates them behind these two create-session fields (`algenta/server/policy.ts`), and the
    # durable path used to send neither — so a Temporal/LangGraph run could not delegate at all.
    #
    # Two plain booleans because that is exactly what the engine takes, and they are gate-checked
    # against the run's approval profile in `validate()` (fail-closed, `swarm.normalize_swarm`). They
    # are NOT derived from each other: `runtime-builder.ts` enables the generic `spawn_agent` tool under
    # one flag and the team runtime under the other, so a team-only session is legal.
    #
    # `False` is "no delegation", which is exactly what every spec recorded before this field existed
    # sent, so such a history replays with the same request it recorded.
    enable_spawn_agent: bool = False
    enable_agent_teams: bool = False
    # D5/X8: work in an EXISTING branch instead of creating ``codna/run-<run>/task-<n>``. This is the
    # run-from-a-pull-request path: the PR card's "auto-fix"/"address-comments" start a run whose work
    # must land on the branch the PR is already open against.
    #
    # Empty is the normal case and means "my own branch", so a spec recorded before this field existed
    # decodes to exactly what it meant. Set, the branch must ALREADY EXIST — a missing one is a
    # refusal, never a new branch — and it is checked out, never reset (`worktree.create_task_worktree`).
    #
    # Page: `worktree` must be True for this to have any effect. A shared checkout cannot be on two
    # branches at once, and the run that uses this one must not move the developer's HEAD.
    worktree_branch: str = ""

    def __post_init__(self) -> None:
        # Canonicalise on construction, so the recorded payload (and therefore history, the ledger and
        # every resumed session) carries the value the engine will see — not the caller's spacing.
        _canonicalise_provider_fields(self)
        # Same rule for MCP: a `${NAME}` shorthand is stored as the `{"from_env": …}` reference it
        # means, so `to_payload` (history!) never carries the shorthand form and a decode/encode round
        # trip is byte-identical.
        object.__setattr__(self, "mcp_servers", _canonical_mcp_servers(self.mcp_servers))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError(
                "unsupported task spec schema version",
                code="unsupported_contract_schema",
                details={"schema_version": self.schema_version},
            )
        _require_ref(self.run_id, "run_id")
        _require_ref(self.task_id, "task_id")
        if not isinstance(self.task_number, int) or isinstance(self.task_number, bool) or self.task_number < 1:
            raise ContractError(
                "task_number must be an integer >= 1 (it names the task branch)",
                code="contract_invalid",
                details={"field": "task_number", "value": repr(self.task_number)},
            )
        if not isinstance(self.brief, str) or not self.brief.strip():
            raise ContractError("brief must be a non-empty string", code="contract_invalid", details={"field": "brief"})
        if len(self.brief) > 32_000:
            raise ContractError(
                "brief is over the 32,000-character budget for a workflow input",
                code="contract_invalid",
                details={"length": len(self.brief)},
            )
        if not isinstance(self.repo_dir, str) or not self.repo_dir.startswith("/"):
            raise ContractError(
                "repo_dir must be an absolute path",
                code="contract_invalid",
                details={"field": "repo_dir", "value": repr(self.repo_dir)},
            )
        _require_ref(self.base_branch, "base_branch")
        if not isinstance(self.verify_commands, tuple) or not all(
            isinstance(command, str) and command.strip() for command in self.verify_commands
        ):
            raise ContractError(
                "verify_commands must be a tuple of non-empty shell commands",
                code="contract_invalid",
                details={"field": "verify_commands"},
            )
        if any(len(command) > 2000 for command in self.verify_commands):
            raise ContractError(
                "a verify command is over the 2000-character limit",
                code="contract_invalid",
                details={"field": "verify_commands"},
            )
        _require_int(self.max_attempts, "max_attempts", minimum=1)
        _optional_cap(self.max_cost_usd, "max_cost_usd")
        if self.max_tokens is not None:
            _require_int(self.max_tokens, "max_tokens", minimum=1)
        _require_engine(self.engine)
        _validate_provider_fields(self)
        if not isinstance(self.worktree, bool):
            raise ContractError(
                "worktree must be a boolean",
                code="contract_invalid",
                details={"field": "worktree", "value": repr(self.worktree)},
            )
        if not isinstance(self.goal_context, str):
            raise ContractError(
                "goal_context must be a string",
                code="contract_invalid",
                details={"field": "goal_context", "value": type(self.goal_context).__name__},
            )
        if len(self.goal_context) > MAX_GOAL_CONTEXT_CHARS:
            raise ContractError(
                f"goal_context is over the {MAX_GOAL_CONTEXT_CHARS}-character budget for a workflow input",
                code="contract_invalid",
                details={"length": len(self.goal_context)},
            )
        findings = find_secret_like_strings(
            {
                "brief": self.brief,
                "verify_commands": list(self.verify_commands),
                "repo_dir": self.repo_dir,
                "goal_context": self.goal_context,
            }
        )
        if findings:
            raise ContractError(
                "the task brief contains secret-like data and may not enter workflow history",
                code="contract_contains_secret",
                details={"findings": findings},
            )
        if not isinstance(self.worktree_task_id, str):
            raise ContractError(
                "worktree_task_id must be a string",
                code="contract_invalid",
                details={"field": "worktree_task_id"},
            )
        if self.worktree_task_id:
            _require_ref(self.worktree_task_id, "worktree_task_id")
        if not isinstance(self.attach_project_rules, bool):
            raise ContractError(
                "attach_project_rules must be a boolean",
                code="contract_invalid",
                details={"field": "attach_project_rules", "value": repr(self.attach_project_rules)},
            )
        # Normalized through the SAME function the decode path uses, so a spec built in Python and a
        # spec decoded from a payload cannot disagree about what a valid MCP config is. That call
        # re-raises as a ContractError so the workflow boundary sees one error family.
        from .mcp import McpConfigError, normalize_mcp_servers

        try:
            normalize_mcp_servers(self.mcp_servers)
        except McpConfigError as exc:
            raise ContractError(
                str(exc), code=exc.code, details=exc.details
            ) from exc
        # D7/SWARM: the two delegation flags, refused here when the approval profile forbids them.
        # Fail-closed at the SPEC boundary — on the CLI, before a workflow exists — rather than at
        # session-create time three activities into a run. The engine refuses them independently
        # (`resolveSessionPolicy`), so this is a second gate in front of the first, never instead of it.
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
                "worktree_branch must be a string",
                code="contract_invalid",
                details={"field": "worktree_branch", "value": type(self.worktree_branch).__name__},
            )
        if self.worktree_branch:
            # A branch name becomes a git argv element, so it is validated as a ref rather than
            # trusted. It may not start with '-': that is the one shape git would read as an OPTION.
            name = self.worktree_branch.strip()
            if not name or name.startswith("-"):
                raise ContractError(
                    "worktree_branch must be a branch name that does not start with '-'",
                    code="contract_invalid",
                    details={"field": "worktree_branch", "value": repr(name)[:100]},
                )
            _require_ref(name, "worktree_branch")
            if not self.worktree:
                # Refused rather than ignored: adopting a branch in a SHARED checkout would move the
                # developer's HEAD out from under them, which is the one thing per-task worktrees exist
                # to prevent. A caller that set this meant to work in isolation.
                raise ContractError(
                    "worktree_branch requires worktree=True: an adopted branch needs its own checkout",
                    code="contract_invalid",
                    details={"field": "worktree_branch"},
                )

    @property
    def branch(self) -> str:
        """``codna/run-<run>/task-<n>`` — stable for the whole task, and the PR's head branch.

        An ADOPTED branch (``worktree_branch``) wins: that is the branch a run-from-a-PR task works on,
        and it must be the branch its commits are made on, its refs point from, and its pull request is
        opened against. Everything else about the task's identity is unchanged.
        """
        if self.worktree_branch:
            return self.worktree_branch.strip()
        return f"codna/run-{self.run_id}/task-{self.task_number}"

    @property
    def worktree_name(self) -> str:
        """``wt-<task>`` — the directory name, matching `codna.durable.worktree`'s prefix."""
        return f"wt-{self.checkout_task_id}"

    @property
    def checkout_task_id(self) -> str:
        """WHICH checkout this task works in: its own, or the task it is repairing (D5).

        Empty ``worktree_task_id`` is the normal case and means "my own". A "fix CI" attempt names its
        parent, so both attempts share one worktree on one branch — while each keeps a ``task_id`` of
        its own for turn keys, checkpoint messages and its episode.
        """
        return self.worktree_task_id or self.task_id

    def initial_state(self) -> dict[str, Any]:
        """The `TaskState` payload a task starts from. Fresh per call (LangGraph mutates it)."""
        self.validate()
        state = TaskState(
            run_id=self.run_id,
            task_id=self.task_id,
            budget=Budget(
                max_cost_usd=self.max_cost_usd,
                spent_usd=0.0,
                max_tokens=self.max_tokens,
                spent_tokens=0,
                unpriced_tokens=0,
            ),
        )
        return state.to_payload()

    def with_remaining_budget(self, remaining_usd: float | None) -> "TaskSpec":
        """This task, capped by what the RUN has left. Used by `RunWorkflow` per child."""
        if remaining_usd is None:
            return self
        cap = remaining_usd if self.max_cost_usd is None else min(self.max_cost_usd, remaining_usd)
        return replace(self, max_cost_usd=cap)

    def with_goal_context(self, goal_context: str) -> "TaskSpec":
        """This task, planning from a run-level objective and recap (D5 goal mode).

        A separate method rather than a caller passing the field, so the bound and the secret screen
        run on the value before it can enter history: `replace` skips `validate()`, so every path
        that mutates a spec goes through a method that re-validates.
        """
        if not isinstance(goal_context, str):
            raise ContractError(
                "goal context must be a string",
                code="contract_invalid",
                details={"field": "goal_context"},
            )
        updated = replace(self, goal_context=goal_context)
        updated.validate()
        return updated

    def with_ci_fix(self, *, task_id: str, brief: str, goal_context: str = "") -> "TaskSpec":
        """A "fix CI" task derived from this one: a NEW identity in the SAME worktree and branch.

        Two things must be true at once, and they are why this is not a plain `replace`:

        * **The work is the same work.** The fix has to land as a commit on the branch CI is already
          running against, in the same checkout, and update the same pull request — so
          ``task_number`` (which names the branch) is preserved and ``worktree_task_id`` points at the
          PARENT's directory.
        * **The attempt is a new attempt.** A distinct ``task_id`` is what gives the retry its own
          turn keys, its own checkpoint commit message, its own episode file and its own child
          workflow id. Reusing the parent's id would collide on all four: the checkpoint's
          ``_existing_commit`` looks up the commit by message and would find the previous attempt's,
          and the branch-exists guard would refuse the worktree.

        ``replace`` skips `validate()`, so the result is re-validated: an over-long brief or a
        secret-bearing one must fail here, before it can enter history.
        """
        updated = replace(
            self,
            task_id=task_id,
            brief=brief,
            goal_context=goal_context,
            worktree_task_id=self.task_id,
        )
        updated.validate()
        if updated.branch != self.branch:  # pragma: no cover - replace cannot change task_number
            raise ContractError(
                "a fix CI task must stay on its parent's branch",
                code="contract_invalid",
                details={"branch": updated.branch, "parent": self.branch},
            )
        return updated

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_number": self.task_number,
            "brief": self.brief,
            "repo_dir": self.repo_dir,
            "base_branch": self.base_branch,
            "verify_commands": list(self.verify_commands),
            "max_attempts": self.max_attempts,
            "max_cost_usd": self.max_cost_usd,
            "max_tokens": self.max_tokens,
            "engine": self.engine,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "provider_key_source": self.provider_key_source,
            "worktree": self.worktree,
            "goal_context": self.goal_context,
            "worktree_task_id": self.worktree_task_id,
            "attach_project_rules": self.attach_project_rules,
            "mcp_servers": _mcp_servers_json(self.mcp_servers),
            "enable_spawn_agent": self.enable_spawn_agent,
            "enable_agent_teams": self.enable_agent_teams,
            "worktree_branch": self.worktree_branch,
        }


@dataclass(frozen=True)
class NodeContext:
    """What a doing-step needs beyond the state: the task, and any steering the human sent.

    Built from the graph's config metadata, which the LangGraph plugin round-trips across the
    activity boundary intact (measured in the D2.2 probe: `config["metadata"]` reaches an
    activity-bound node unchanged). The request builders live here so the graph's nodes and the
    plain workflow construct byte-identical requests — the property that makes engine parity a fact
    rather than a coincidence.
    """

    spec: TaskSpec
    steer: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.spec.validate()
        _validate_steer(self.steer)

    @property
    def context(self) -> TaskContext:
        return TaskContext.from_spec(self.spec)

    def plan_request(self, state: Mapping[str, Any]) -> "PlanRequest":
        return self.context.plan_request(
            self.spec.brief, state, self.steer, self.spec.goal_context
        )

    def turn_request(self, state: Mapping[str, Any], prompt: str) -> "TurnRequest":
        return self.context.turn_request(prompt, state, self.steer)

    def checkpoint_request(self, state: Mapping[str, Any]) -> "CheckpointRequest":
        return self.context.checkpoint_request(state)

    def verify_request(self, state: Mapping[str, Any]) -> "VerifyRequest":
        return self.context.verify_request(state, self.spec.verify_commands)

    def review_request(self, state: Mapping[str, Any]) -> "ReviewRequest":
        return self.context.review_request(state)

    def pull_request_request(self, state: Mapping[str, Any]) -> "PullRequestRequest":
        return self.context.pull_request_request(pr_title(self.spec), state)


# -- requests ---------------------------------------------------------------------------
#
# Every request carries the state payload it is transforming plus `TaskContext`. A retried activity
# is handed the SAME payload from history, which is what makes "the activity is a pure function of
# (request, sidecar/ledger state)" a property a test can assert rather than a hope.


@dataclass(frozen=True)
class PlanRequest:
    """A read-only turn that produces a plan reference."""

    state: dict[str, Any]
    context: TaskContext
    brief: str
    steer: tuple[str, ...] = ()
    # D5 goal mode: the run's objective and the episodes of earlier tasks, appended to the plan
    # prompt. Empty for a task-mode run, which is what makes an existing history replay identically.
    goal_context: str = ""

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        if not isinstance(self.brief, str) or not self.brief.strip():
            raise ContractError("plan brief must be non-empty", code="contract_invalid", details={"field": "brief"})
        _validate_steer(self.steer)
        if not isinstance(self.goal_context, str) or len(self.goal_context) > MAX_GOAL_CONTEXT_CHARS:
            raise ContractError(
                "plan goal_context must be a string within the goal-context budget",
                code="contract_invalid",
                details={"field": "goal_context", "length": len(getattr(self, "goal_context", "") or "")},
            )
        _require_ref(self.context.run_id, "run_id")
        _require_ref(self.context.task_id, "task_id")
        return validated_state(self.state)

    def prompt(self) -> str:
        """The plan turn's prompt — the ONE place the goal block is appended.

        Every caller (both engines' plan steps) builds its prompt through this, so a goal run cannot
        plan on one engine with the objective and on the other without it.
        """
        return plan_prompt(self.brief, self.steer, self.goal_context)


@dataclass(frozen=True)
class TurnRequest:
    """The heartbeating paid turn: one prompt, one ``turn_key``, one stream."""

    state: dict[str, Any]
    context: TaskContext
    prompt: str
    read_only: bool = False
    steer: tuple[str, ...] = ()

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ContractError(
                "turn prompt must be non-empty", code="contract_invalid", details={"field": "prompt"}
            )
        if len(self.prompt) > 64_000:
            raise ContractError(
                "turn prompt is over the 64,000-character limit",
                code="contract_invalid",
                details={"length": len(self.prompt)},
            )
        _validate_steer(self.steer)
        state = validated_state(self.state)
        # A paid turn must know how far the previous one got, or its reconnect would replay (and be
        # billed for) events the ledger already has.
        if state["last_turn_key"] and state["session_id"] is None:
            raise ContractError(
                "a state with last_turn_key must also carry the session it belongs to",
                code="contract_invalid",
                details={"last_turn_key": state["last_turn_key"]},
            )
        return state


@dataclass(frozen=True)
class VerifyRequest:
    """Run the task's verification commands under `codna.sandbox`."""

    state: dict[str, Any]
    context: TaskContext
    commands: tuple[str, ...] = ()
    timeout_s: int = ACTIVITY_VERIFY_TIMEOUT_S

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        if not isinstance(self.commands, tuple) or not all(
            isinstance(command, str) and command.strip() for command in self.commands
        ):
            raise ContractError(
                "commands must be a tuple of non-empty shell commands",
                code="contract_invalid",
                details={"field": "commands"},
            )
        _require_int(self.timeout_s, "timeout_s", minimum=1)
        return validated_state(self.state)


@dataclass(frozen=True)
class CheckpointRequest:
    """Commit the working tree as turn ``turn_seq`` and move the run's git refs. Idempotent."""

    state: dict[str, Any]
    context: TaskContext

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        state = validated_state(self.state)
        if state["turn_seq"] < 1:
            raise ContractError(
                "a checkpoint needs turn_seq >= 1: nothing has been turned yet",
                code="contract_invalid",
                details={"turn_seq": state["turn_seq"]},
            )
        return state


@dataclass(frozen=True)
class ReviewRequest:
    """Review the checkpointed diff (reuses `codna.review_findings`)."""

    state: dict[str, Any]
    context: TaskContext
    base_sha: str | None = None

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        state = validated_state(self.state)
        if state["checkpoint_ref"] is None:
            raise ContractError(
                "a review needs a checkpoint to review",
                code="contract_invalid",
                details={"task_id": self.context.task_id},
            )
        return state


@dataclass(frozen=True)
class PullRequestRequest:
    """Push the checkpoint commit and open (or find) the pull request. Idempotent by branch."""

    state: dict[str, Any]
    context: TaskContext
    title: str
    body: str = ""

    def validate(self) -> dict[str, Any]:
        self.context.validate()
        if not isinstance(self.title, str) or not self.title.strip():
            raise ContractError(
                "a pull request needs a title", code="contract_invalid", details={"field": "title"}
            )
        if len(self.title) > 256:
            raise ContractError(
                "pull request title is over the 256-character limit",
                code="contract_invalid",
                details={"length": len(self.title)},
            )
        if not isinstance(self.body, str) or len(self.body) > 32_000:
            raise ContractError(
                "pull request body must be a string of at most 32,000 characters",
                code="contract_invalid",
                details={"field": "body"},
            )
        state = validated_state(self.state)
        if state["checkpoint_ref"] is None:
            raise ContractError(
                "a pull request needs a checkpoint commit to push",
                code="contract_invalid",
                details={"task_id": self.context.task_id},
            )
        return state


# -- helpers ----------------------------------------------------------------------------




def _validate_steer(steer: Any) -> None:
    if not isinstance(steer, tuple):
        raise ContractError(
            "steer notes must be a tuple of strings", code="contract_invalid", details={"field": "steer"}
        )
    if len(steer) > MAX_STEER_NOTES:
        raise ContractError(
            f"at most {MAX_STEER_NOTES} steer notes are carried into a turn",
            code="contract_invalid",
            details={"count": len(steer)},
        )
    for note in steer:
        if not isinstance(note, str) or not note.strip():
            raise ContractError(
                "a steer note must be a non-empty string",
                code="contract_invalid",
                details={"field": "steer"},
            )
        if len(note) > MAX_STEER_NOTE_CHARS:
            raise ContractError(
                f"a steer note is over the {MAX_STEER_NOTE_CHARS}-character limit",
                code="contract_invalid",
                details={"length": len(note)},
            )
    findings = find_secret_like_strings({"steer": list(steer)})
    if findings:
        raise ContractError(
            "a steer note contains secret-like data and may not enter workflow history",
            code="contract_contains_secret",
            details={"findings": findings},
        )


# Exposed for tests: the field lists are the contract, so a drifted field is visible.
TASK_SPEC_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(TaskSpec))
TURN_REQUEST_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(TurnRequest))
