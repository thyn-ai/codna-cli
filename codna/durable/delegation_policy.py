"""The delegation policy answer and the sentence that reports it — WITHOUT the workflow runtime.

Both functions here are about the repository's ``.codna/policy.yaml``: the activity body that reads it
(`delegation_policy`) and the ledger sentence built from its answer (`delegation_policy_note`). They
live in their own module because neither needs a workflow, an activity decorator or the HTTP stack, and
because the OPTIONAL ``durable`` extra must not be a prerequisite for reading either one.

**Why that matters, not just "it is tidy".** `capability_report` is the module that USES the note, and
it schedules a real activity, so it imports `workflow_support` → ``temporalio``. A test that imported
the note from there turned "the extra is not installed" into `ModuleNotFoundError: No module named
'temporalio'` for a pure string builder — the exact convention `tests/durable/conftest.py` documents
("nothing here may import ``temporalio``/``langgraph`` at module scope"). The same holds for the
activity BODY: the body is a function of its request plus one file, so asserting its answer must not
require installing the runtime it is scheduled on.

So this module imports stdlib, ``typing`` and the contract dataclasses only. The two modules that DO
need the runtime keep ownership of their half: `activity_steps` re-binds `delegation_policy` as the
registered activity body, and `capability_report` re-exports `delegation_policy_note` for its callers
and schedules the activity through `workflow_activity`.

**The read is an activity, and that is load-bearing.** A workflow may not touch the filesystem —
measured as ``RestrictedWorkflowAccessError: Cannot access os.path from inside a workflow`` when this
logic was first written inline in ``RunWorkflow._report_capabilities``. Putting the answer in an
activity result also puts it in history, so a replay derives the report instead of re-reading a file
that may have changed since.
"""
from __future__ import annotations

from typing import Final

from .contracts import DelegationPolicyOutcome, DelegationPolicyRequest

__all__ = [
    "delegation_policy",
    "delegation_policy_note",
]

#: The one policy path an operator edits. Named in the note so the fix is obvious; the activity
#: reports the real path it read, so this is only the fallback for a malformed answer.
_FALLBACK_POLICY_PATH: Final = ".codna/policy.yaml"


def delegation_policy(request: DelegationPolicyRequest) -> DelegationPolicyOutcome:
    """Which prompt-named team tools this run's repository policy strips, and the file that decided it.

    Empty for the ordinary run, so the caller reports nothing: a repository with no policy (the
    approval profile decides alone, and every profile the durable path resolves grants the tools), or
    a policy that names them all.

    PURE, so it needs neither the extra nor a running activity to be asserted — and it is the object
    `activity_steps` re-binds as the registered body, so a test calling this calls what the worker runs.
    """
    request.validate()
    # Imported inside the function: `policy_document` reaches the YAML reader and the findings digest,
    # and `swarm` is a pure-data module that should stay cheap for the spec-validation callers.
    from .swarm import policy_stripped_team_tools

    from ..policy_document import policy_path_for

    stripped = policy_stripped_team_tools(
        request.repo_dir, enable_agent_teams=request.enable_agent_teams
    )
    # `str(...)`: `policy_path_for` returns a `Path`, and the contract declares a `str` — a Temporal
    # result is decoded from its annotations, so handing back a Path would make the outcome a shape
    # the workflow did not declare (and `str`-typed callers would break on `.endswith`).
    return DelegationPolicyOutcome(
        stripped=tuple(stripped),
        policy_path=str(policy_path_for(request.repo_dir)),
    )


def delegation_policy_note(run_id: str, stripped: tuple[str, ...], policy_path: str) -> str | None:
    """The "the flags are on but the policy forbids the tools" row, or ``None`` when there is none.

    ``None`` is the answer for every run that delegates normally, which is why the ordinary case
    records nothing beyond the attachment note it already had.
    """
    if not stripped:
        return None
    return (
        f"run {run_id}: the delegation flags are attached but "
        f"{policy_path or _FALLBACK_POLICY_PATH} does not permit the team tools the prompt names, so "
        f"the engine will advertise no team tool to this run: {', '.join(stripped)}. Add them to "
        f"`tools.allow`, or the run cannot form a team whatever `enable_agent_teams` says."
    )
