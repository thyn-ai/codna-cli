"""Building the create-session request: the ONE place a durable session is described (D7).

Split out of `activity_support.py` at the repository's 1000-line ceiling (`test_modularity.py`) on a
real boundary: everything here answers "what does this turn's session look like", and nothing else in
that module does. Both call sites (`ensure_session` and the resume path) already imported these by
name, so moving them changed no caller.

**Why the request is assembled in one function.** A session is created once and then carries every turn
of a task, so a field omitted here is omitted for the whole task. Each block below exists because it was
once missing and the failure was silent rather than loud: the provider coordinates because the engine
otherwise falls back to `anthropic`; `multi_turn` because the SDK finalises a one-shot session at the
end of turn 1 (`SessionNotFoundError` on turn 2); `policy` because an unattended run must not fall back
to "whatever the profile allowed"; MCP and the delegation flags because a capability the run asked for
must not be dropped at the boundary.

**What is resolved here and what is not.** A `TaskSpec` is workflow history, so it may name WHERE a
credential comes from but never the credential. The provider key and an MCP server's `env` values are
resolved from the WORKER's process environment at this moment, and an unset variable is a refusal
naming it rather than a session that starts without it.
"""
from __future__ import annotations

import os
from typing import Any

from .contracts import provider_session_spec, resolve_provider_env_key
from .mcp import mcp_session_spec
from .swarm import swarm_session_spec

# NOTE: `owner_for` lives in `activity_support`, which imports THIS module, so it is resolved at call
# time (inside `_session_spec`) rather than at module scope. A module-scope import here would close the
# cycle and make `activity_support` unimportable.

__all__ = ["_project_rules", "_provider_key", "_session_spec"]


def _session_spec(context: Any, *, read_only: bool, working_tree: str = "") -> dict[str, Any]:
    """The create-session spec. Read-only is an EMPTY WRITE ALLOW-LIST, not a prompt instruction.

    The PROVIDER coordinates are merged in here — this is the one place a session is created, so it
    is the one place they can be forgotten. Without them the engine silently falls back to its own
    default provider (`anthropic`), which is why a host holding only `OPENAI_API_KEY` used to fail
    every durable turn with `ProviderCredentialError`, and why no keyless local provider could be
    used for a durable run at all.

    ``injected_context`` carries the repository's own conventions (D5). It is built HERE rather than
    transported because it is file contents: the engine folds it into the system prompt, and reading it
    from ``working_tree`` is what makes a task in its own worktree see the rules on ITS branch.

    The provider credential is resolved from the PROCESS ENVIRONMENT at this moment and never travels:
    a `TaskSpec` is workflow history, so it names the provider and its key SOURCE. See `provider.py`.
    """
    # `owner_for` is resolved HERE: it lives in `activity_support`, which imports this module, so a
    # module-scope import would close the cycle.
    from .activity_support import owner_for

    spec: dict[str, Any] = {"owner": owner_for(context)}
    if read_only:
        spec["allowed_write_paths"] = []
    # A durable task's session carries EVERY turn of that task (plan, implement, re-implement), so the
    # engine must not treat the first one as the whole conversation. Without this the SDK finalizes the
    # session at the end of turn 1 (`finalizeSingleRun` → `sessions.delete`) and the SECOND POST answers
    # `SessionNotFoundError: session not found` — measured against the real sidecar, with no restart
    # involved. Declared here, at the one place a session is created, so no step can forget it.
    spec["multi_turn"] = True
    rules = _project_rules(context, working_tree)
    if rules:
        spec["injected_context"] = rules
    # E3 policy-as-code: a durable run is governed. `required: true` makes a repository with NO
    # `.codna/policy.yaml` a refusal (`policy_missing`) instead of a silent fall-back to the
    # approval profile — an unattended multi-hour run is exactly the case where "no policy" must
    # not mean "whatever the profile allowed". The sidecar reads the file from `working_tree`, so
    # the policy in force is the one on the branch the turn edits, not the shared checkout's.
    spec["policy"] = {"required": True}
    # D5/MCP: the servers this session is given, credential REFERENCES expanded from this worker's own
    # environment (a missing variable raises, naming it — see `mcp.mcp_session_spec`).
    spec.update(mcp_session_spec(context, os.environ))
    # D7/SWARM: the two delegation fields, when (and only when) this task asked for them. Absent for
    # every run that did not, so the pre-feature request stays byte-identical — see
    # `swarm.swarm_session_spec`. This is the ONLY place a session is created, which is why the flags
    # are added here rather than at each turn: no step can forget what it never has to remember.
    spec.update(swarm_session_spec(context))
    spec.update(provider_session_spec(context, key=_provider_key(context)))
    return spec


def _project_rules(context: Any, working_tree: str) -> str:
    """The repository's conventions AND skills for this turn, or ``""`` (D5).

    Read from ``working_tree`` — the tree the turn is about to edit — because a rules file or a
    ``SKILL.md`` can differ between a task's branch and the shared checkout. ``""`` when the task opted
    out; the read itself is bounded in ``rules.project_context_for``.

    **Skills travel here rather than as a second field.** A durable turn's create-session body has one
    ``injected_context`` string, and the skills CATALOGUE belongs in the same prompt as the rules: it is
    the same kind of content, it is spent from the same budget, and a second field would need a second
    agreement with the engine for no gain. The rules are rendered first, the catalogue takes what is
    left — see ``rules.load_context``.
    """
    if not getattr(context, "attach_project_rules", False):
        return ""
    if not working_tree:
        # No tree named means the caller did not say where the turn works; falling back to
        # `repo_dir` would read the SHARED checkout, which may be another task's branch.
        return ""
    from .rules import project_context_for

    return project_context_for(working_tree)


def _provider_key(context: Any) -> str | None:
    """The provider credential this turn should carry, or `None`.

    Only resolved when the task ASKED for an env-sourced key (`provider_key_source == "environment"`).
    A keyless local provider needs none, and a hosted provider with no source set keeps the pre-change
    behaviour: the engine reads its own process environment, exactly as before.
    """
    if context.provider_key_source != "environment" or context.provider is None:
        return None
    return resolve_provider_env_key(context.provider, os.environ)
