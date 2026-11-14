"""Decoding a task from the LangGraph CONFIG the plugin round-trips (D2.2/D5).

Split out of `contracts.py` at the repository's 1000-line ceiling (`test_modularity.py`), on a real
boundary: everything here reads `config["metadata"]` — the channel the LangGraph plugin carries across
the activity boundary — and turns it into validated values. The rest of `contracts.py` describes what
crosses the wire; this describes how a node FINDS it.

**Explicit failure is the point.** Without the spec a node cannot know which repository to act on, and
defaulting to "some repository" would edit the wrong tree. So a missing or malformed config is a
`contract_task_spec_missing` refusal naming the metadata keys it saw — this is the ONLY place the
graph's config contract is read, so a caller that gets it wrong fails here rather than inside an
activity three steps later.

**Absent fields decode to what a recorded history MEANT.** A payload written before per-task
worktrees, goal mode or project rules existed carries none of those keys, and every one of them
defaults to the pre-feature behaviour so an in-flight run replays with the prompts and trees it
actually used. That is the rule for every additive field on `TaskSpec`, and it is why the defaults
here are not simply the dataclass defaults.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contract_primitives import ContractError
from .mcp import normalize_mcp_servers


def _mcp_servers(raw: Any) -> tuple[dict[str, Any], ...]:
    """A recorded ``mcp_servers`` list, re-normalized through the ONE validator.

    Round-tripping through `normalize_mcp_servers` means a history decoded here is subject to the same
    rules as a spec built in Python: a payload edited by hand (or written by an older, laxer build)
    cannot smuggle a literal secret into the field, and an invalid one is refused rather than sent.
    """
    return normalize_mcp_servers(raw)

__all__ = [
    "CONFIG_STEER",
    "CONFIG_TASK_SPEC",
    "steer_from_config",
    "task_spec_from_config",
]

# The metadata keys the workflow puts in the graph's config. They ride in the config rather than the
# state because the state is a validated `TaskState` payload with a closed field set: a `brief` in
# it would be a field nothing validates, growing with the request, in the one place that must not.
CONFIG_TASK_SPEC: str = "codna_task"
CONFIG_STEER: str = "codna_steer"


def task_spec_from_config(config: Mapping[str, Any] | None) -> Any:
    """Recover the `TaskSpec` from a LangGraph config, refusing a missing or malformed one.

    Explicit failure: without the spec a node cannot know which repository to act on, and defaulting
    to "some repository" would edit the wrong tree. This is the ONLY place the graph's config
    contract is read, so a caller that gets it wrong fails here rather than inside an activity.
    """
    # Imported HERE: `contracts` imports this module for the re-exports, so a module-scope import
    # back into it would close the cycle. Resolved at call time, which also means this function
    # works under a workflow sandbox that has already imported `contracts`.
    from .contracts import TaskSpec

    if not isinstance(config, Mapping):
        raise ContractError(
            "a task-graph node needs the LangGraph config it runs under",
            code="contract_task_spec_missing",
            details={},
        )
    metadata = config.get("metadata")
    if not isinstance(metadata, Mapping) or CONFIG_TASK_SPEC not in metadata:
        raise ContractError(
            f"task-graph nodes need config['metadata']['{CONFIG_TASK_SPEC}'] (the TaskSpec payload)",
            code="contract_task_spec_missing",
            details={"metadata_keys": sorted(str(key) for key in (metadata or {}))},
        )
    raw = metadata[CONFIG_TASK_SPEC]
    if isinstance(raw, TaskSpec):
        raw.validate()
        return raw
    if not isinstance(raw, Mapping):
        raise ContractError(
            f"config metadata {CONFIG_TASK_SPEC!r} must be a TaskSpec payload",
            code="contract_task_spec_missing",
            details={"type": type(raw).__name__},
        )
    spec = TaskSpec(
        schema_version=int(raw.get("schema_version", 1)),
        run_id=raw.get("run_id"),  # type: ignore[arg-type]
        task_id=raw.get("task_id"),  # type: ignore[arg-type]
        task_number=raw.get("task_number"),  # type: ignore[arg-type]
        brief=raw.get("brief"),  # type: ignore[arg-type]
        repo_dir=raw.get("repo_dir"),  # type: ignore[arg-type]
        base_branch=raw.get("base_branch", "main"),
        verify_commands=tuple(raw.get("verify_commands") or ()),
        max_attempts=int(raw.get("max_attempts", 3)),
        max_cost_usd=raw.get("max_cost_usd"),
        max_tokens=raw.get("max_tokens"),
        engine=raw.get("engine", "langgraph"),
        provider=raw.get("provider"),
        model=raw.get("model"),
        base_url=raw.get("base_url"),
        provider_key_source=raw.get("provider_key_source"),
        # Absent in a payload recorded before per-task worktrees existed, hence the default: a run
        # already in flight keeps working in its shared checkout, exactly as it was started to.
        worktree=bool(raw.get("worktree", False)),
        # Absent in a payload recorded before goal mode existed: empty is what a task-mode plan turn
        # has always carried, so such a history replays with the same prompt it recorded.
        goal_context=str(raw.get("goal_context") or ""),
        worktree_task_id=str(raw.get("worktree_task_id") or ""),
        # Absent in a payload recorded before project rules were attached: `False` is what that run
        # MEANT (it sent no guidance), so decoding it as `True` would change its prompts on replay.
        attach_project_rules=bool(raw.get("attach_project_rules", False)),
        # Absent in a payload recorded before user-defined subagents could be attached: `False` is what
        # that run MEANT (it sent no roster), so decoding it as the new default would change its
        # sessions on replay.
        attach_subagents=bool(raw.get("attach_subagents", False)),
        # Absent before MCP servers could be carried: `()` is exactly what such a run sent, so its
        # replay creates the same sessions it recorded rather than gaining an MCP field.
        mcp_servers=_mcp_servers(raw.get("mcp_servers")),
        # D7/SWARM: absent in a payload recorded before delegation could be carried, and `False` is
        # exactly what such a run sent, so it replays creating the same sessions it recorded.
        enable_spawn_agent=bool(raw.get("enable_spawn_agent", False)),
        enable_agent_teams=bool(raw.get("enable_agent_teams", False)),
        # Absent before run-from-a-branch existed: `""` is "my own branch", which is exactly what such
        # a run worked on, so its replay adopts nothing.
        worktree_branch=str(raw.get("worktree_branch") or ""),
        auto_approve_local_only=raw.get("auto_approve_local_only", False),
        approval_policy_digest=raw.get("approval_policy_digest", ""),
    )
    spec.validate()
    return spec


def steer_from_config(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The steering notes a workflow put in the graph's config metadata, if any."""
    if not isinstance(config, Mapping):
        return ()
    metadata = config.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get(CONFIG_STEER)
    if not isinstance(raw, (list, tuple)):
        return ()
    notes = tuple(str(note) for note in raw if str(note).strip())
    # The SAME bound and secret screen the contract applies, reached from the leaf that owns the
    # constants. Duplicating them here would let a steer note pass the config path and be refused at
    # the contract boundary, i.e. a task that fails on its first turn instead of on its input.
    from .contract_primitives import MAX_STEER_NOTE_CHARS, MAX_STEER_NOTES, ContractError
    from .state import find_secret_like_strings

    if len(notes) > MAX_STEER_NOTES:
        raise ContractError(
            f"at most {MAX_STEER_NOTES} steer notes are carried into a turn",
            code="contract_invalid",
            details={"count": len(notes)},
        )
    for note in notes:
        if len(note) > MAX_STEER_NOTE_CHARS:
            raise ContractError(
                f"a steer note is over the {MAX_STEER_NOTE_CHARS}-character limit",
                code="contract_invalid",
                details={"length": len(note)},
            )
    findings = find_secret_like_strings({"steer": list(notes)})
    if findings:
        raise ContractError(
            "a steer note contains secret-like data and may not enter workflow history",
            code="contract_contains_secret",
            details={"findings": findings},
        )
    return notes


