"""The eight durable step implementations — what a durable run actually does.

Split from `activities.py` when that module crossed the repository's 1000-line ceiling
(`test_modularity.py`). `activity_support.py` owns the runtime, the error type and git;
`activities.py` owns registration; this module owns the steps, and is the one file a reviewer reads
to answer "what does a turn DO".

The PAID turn's own machinery — the retry contract, the ledger lookup that decides whether to buy a
turn or re-attach to one, the session handling and the stream loop — lives in `turn_runner.py`. It
was split out on the same line ceiling, and because "at most one paid turn per `turn_key`" is the
property worth reading end to end rather than in fragments between other steps.

Graph node → step function:

============================  ==================================================
node                          step
============================  ==================================================
``plan``                      `plan_task`      (a read-only turn)
``implement``                 `run_cline_turn` (the heartbeating paid turn)
                              + `checkpoint_turn` (one commit per ``turn_seq``)
``verify``                    `verify_task`    (under `codna.sandbox`)
``review``                    `review_task`    (reuses `codna.review_findings`)
``open_pr``                   `open_pr`        (reuses `codna.gitpush` + `review_github`)
after ``open_pr``             `remember_episode`
on a decision                 `notify`
============================  ==================================================

**Why `implement` is two activities.** A checkpoint is only meaningful once the turn's edits exist,
and the graph cannot express "and then commit" without adding a node — which changes history and
therefore `GRAPH_VERSION`. The implement step calls `checkpoint_turn` right after the turn returns,
so there is exactly one checkpoint per ``turn_seq`` (asserted by counting the ledger's ``checkpoint``
records, not by trusting the call graph).

**Two deliberate signatures.** A registered activity takes exactly its request (Temporal inspects the
signature, and a second parameter would have to be decodable from the payload). The runtime is
therefore injected by `activities.configure_default_runtime`, and `runtime_for(...)` hands a worker-
thread caller a runtime whose cancellation probe is bound to the workflow thread — Temporal's activity
context is a contextvar and does not survive the hop into a thread.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Sequence

from . import publish_steps as _publish_steps
from .activity_support import (
    ActivityError,
    ActivityRuntime,
    decode_request,
    drive_turn,
    ensure_checkout,
    git,
    git_ok,
)
from .contracts import (
    ActivityOutcome,
    CheckpointOutcome,
    CheckpointRequest,
    EpisodeOutcome,
    EpisodeRequest,
    NotifyOutcome,
    NotifyRequest,
    PlanRequest,
    ReviewRequest,
    TaskContext,
    TurnRequest,
    VerifyRequest,
    WorktreeCleanupOutcome,
    WorktreeCleanupRequest,
)
from .ledger import RunLedger
from .state import AskRef, CheckpointRef, TaskState, TaskStateError, VerifyRef
from .worktree import remove_run_worktrees

__all__ = [
    "checkpoint_turn",
    "notify",
    "open_pr",
    "plan_task",
    "remember_episode",
    "review_task",
    "run_cline_turn",
    "verify_task",
    "worktree_cleanup",
]

# Which notification kinds are worth a log line (a run-level event, never a per-token one).
_CONSOLE_KINDS: Final[frozenset[str]] = frozenset(
    {"run.started", "task.started", "task.finished", "approval.granted", "run.finished"}
)

# The lazy runtime accessor is patched in by `activities.py` (it owns the process-wide instance). A
# module-level import of `activities` here would be a cycle: `activities` imports this module to
# register the steps.
_runtime_provider: Any = None


def _run_dir(runs_root, run_id: str):
    """The run's directory, created if needed. Local so this module does not import `activities`."""
    from .config import validate_run_id

    directory = Path(runs_root) / validate_run_id(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _runtime() -> ActivityRuntime:
    """The runtime this step runs with, resolved through `activities` (which owns the default)."""
    if _runtime_provider is None:  # pragma: no cover - `activities` always installs it
        raise ActivityError(
            "no ActivityRuntime provider is installed; import codna.durable.activities first",
            code="activity_runtime_missing",
            details={},
        )
    return _runtime_provider()



# -- the heartbeating turn ---------------------------------------------------------------
#
# `drive_turn` (in `activity_support`) owns the money guarantee: one paid turn per
# ``(run, task, turn_seq)``, across a `kill -9` mid-turn and any number of activity retries. A step
# body must never POST to the sidecar itself — that is how the invariant gets bypassed.


def plan_task(request: PlanRequest) -> "ActivityOutcome":  # noqa: F821 - see contracts
    """A read-only turn that produces `plan_ref`.

    A read-only turn is still a paid turn: its own ``turn_seq``, its own ``turn_key``, its own ledger
    records — so the cost of planning is visible rather than folded into the first implement turn.
    """
    from .contracts import ActivityOutcome

    active = _runtime()
    state = decode_request(request, "plan_task")
    # The plan turn is read-only, but it still EDITS nothing only because of the engine's write
    # allow-list — so it must be given the same tree the implement turn will be given. Otherwise the
    # plan is produced from one checkout and applied in another.
    tree = ensure_checkout(active, request.context)
    active.beat({"task_id": request.context.task_id, "phase": "plan", "turn_seq": state.turn_seq})
    outcome = drive_turn(
        active,
        request.context,
        state,
        # D5 goal mode: the ONE place the objective + episode recap is appended. Empty for a
        # task-mode run, so such a turn sends exactly the prompt it always did.
        prompt=request.prompt(),
        read_only=True,
        phase="plan",
        notes=request.steer,
        working_tree=tree,
    )
    # D4: a plan turn that stopped on a decision produced NO plan. Deriving a `plan_ref` for it would
    # claim a plan exists (that ref is all `after_plan` checks), so the task would march on to implement
    # with nothing to implement from. The ask is recorded instead and the router parks.
    ask = _ask_ref(outcome)
    plan_ref = (
        None
        if ask is not None
        else _plan_ref_for(request, session_id=outcome.session_id, turn_id=outcome.turn_id)
    )
    updated = state.with_events(outcome.event_index).started_turn(
        session_id=outcome.session_id, turn_id=outcome.turn_id
    )
    if ask is not None:
        updated = replace(updated, ask=ask)
    updated = replace(updated, plan_ref=plan_ref, attempts=updated.attempts.incremented("plan"))
    updated = updated.with_spend(
        cost_usd=outcome.cost_usd, tokens=outcome.tokens, priced=outcome.priced
    )
    return ActivityOutcome(
        state=updated.to_payload(),
        cost_usd=outcome.cost_usd,
        tokens=outcome.tokens,
        priced=outcome.priced,
        details={
            "turn_id": outcome.turn_id,
            "plan_ref": plan_ref,
            "session_id": outcome.session_id,
            "resumed": outcome.resumed,
            # D4: `ask` is reported here as well as on the state, because a reader of the ledger wants
            # to know WHY a plan turn produced no reference without decoding the whole state payload.
            "finish_reason": outcome.finish_reason,
            **({"ask_id": ask.ask_id} if ask is not None else {}),
        },
    )


def _ask_ref(outcome: Any) -> AskRef | None:
    """The engine's ask as a history-safe reference, or None when the turn did not park.

    Strict by construction: a turn whose `finish_reason` is `ask` but which carries no usable ask is a
    PROTOCOL violation, and the honest response is to fail the activity (`activity_ask_malformed`)
    rather than treat it as completed. That distinction is the whole point — a silently "completed"
    turn that never ran is the failure this work exists to prevent.
    """
    payload = getattr(outcome, "ask", None)
    if payload is None:
        if getattr(outcome, "finish_reason", None) == "ask":
            raise ActivityError(
                f"turn {outcome.turn_id} reported finish_reason=ask but carried no ask payload",
                code="activity_ask_malformed",
                details={"turn_id": outcome.turn_id, "session_id": outcome.session_id},
            )
        return None
    try:
        return AskRef.from_payload(payload)
    except TaskStateError as exc:
        raise ActivityError(
            f"turn {outcome.turn_id} returned an unusable ask: {exc}",
            code="activity_ask_malformed",
            details={"turn_id": outcome.turn_id, "cause": exc.as_dict()["error"]},
        ) from exc


def _plan_ref_for(request: PlanRequest, *, session_id: str, turn_id: str) -> str:
    """A plan REFERENCE derived from the turn that produced it.

    Not the plan text: `TaskState` may not carry model output (unbounded, attacker-influenced). The
    text is in the ledger under that turn's event indices; this points at them, and is derived so a
    replayed turn yields the same reference.
    """
    digest = hashlib.sha256(f"{session_id}:{turn_id}".encode()).hexdigest()[:16]
    return f"refs/plans/{request.context.task_id}-{digest}"


def run_cline_turn(request: TurnRequest) -> "ActivityOutcome":  # noqa: F821 - see contracts
    """The heartbeating paid turn. See `activity_support.drive_turn` for the sequence and why."""
    from .contracts import ActivityOutcome

    active = _runtime()
    state = decode_request(request, "run_cline_turn")
    # Created HERE rather than once per task: a worker may pick up a turn whose task was started
    # elsewhere (or whose worktree an operator reclaimed), and the step that runs is the only one that
    # knows it is running. Idempotent, so the usual case is one `git worktree list`.
    tree = ensure_checkout(active, request.context)
    outcome = drive_turn(
        active,
        request.context,
        state,
        prompt=request.prompt,
        read_only=request.read_only,
        phase="implement",
        notes=request.steer,
        working_tree=tree,
    )
    updated = state.with_events(outcome.event_index).started_turn(
        session_id=outcome.session_id, turn_id=outcome.turn_id
    )
    updated = replace(updated, attempts=updated.attempts.incremented("implement"))
    updated = updated.with_spend(
        cost_usd=outcome.cost_usd, tokens=outcome.tokens, priced=outcome.priced
    )
    # D4: the turn stopped on a decision. Recording it on the state is what makes the ROUTER park the
    # task instead of sending an unwritten change to verify — and it is the only place the ask id
    # survives the activity boundary, so a human's answer can still be posted against it.
    ask = _ask_ref(outcome)
    if ask is not None:
        updated = replace(updated, ask=ask)
    return ActivityOutcome(
        state=updated.to_payload(),
        cost_usd=outcome.cost_usd,
        tokens=outcome.tokens,
        priced=outcome.priced,
        details={
            "turn_id": outcome.turn_id,
            "session_id": outcome.session_id,
            "event_index": outcome.event_index,
            "resumed": outcome.resumed,
            "finish_reason": outcome.finish_reason,
            **({"ask_id": ask.ask_id} if ask is not None else {}),
        },
    )


# -- checkpoint --------------------------------------------------------------------------


def turn_ref(context: TaskContext, turn_seq: int) -> str:
    return f"refs/codna/runs/{context.run_id}/turn-{turn_seq}"


def checkpoint_turn(request: CheckpointRequest) -> CheckpointOutcome:
    """Commit the working tree as ``turn_seq`` and move the run's audit refs. Idempotent.

    One commit and one ref per ``turn_seq`` — the plan's exit criterion, and what lets an auditor
    answer "what did turn 3 change". Idempotency is structural, not a flag: the commit message
    carries the turn key, so a retried checkpoint finds the commit it already made and re-points the
    refs at it instead of creating a second. A turn that changed nothing still writes its ref,
    pointing at the previous commit: "turn 3 produced exactly the tree turn 2 did" is a fact worth
    recording, and a missing ref would be indistinguishable from a lost turn.
    """
    active = _runtime()
    state = decode_request(request, "checkpoint_turn")
    context = request.context
    turn_seq = state.turn_seq
    message = f"codna: turn {context.run_id}:{context.task_id}:{turn_seq}"
    ref = turn_ref(context, turn_seq)
    # WHERE THE COMMIT IS MADE. The worktree for a worktree-backed task, the shared checkout otherwise.
    # This is the requirement the whole D5 slice turns on: a commit made in the repo root would leave
    # the task's own checkout dirty and its branch pointing at a commit the worktree never saw.
    tree = ensure_checkout(active, context)

    existing = _existing_commit(active, tree, branch=context.branch, message=message)
    if existing is not None:
        sha, created = existing, False
    else:
        _ensure_on_branch(active, tree, context, state)
        sha = _commit(active, tree, message=message)
        created = True

    # Refs are updated in the REPOSITORY, not in the worktree: a worktree shares its parent's refs and
    # object store but is not itself the place refs are authoritative. Using `context.repo_dir` here
    # (as before) is still correct for a worktree-backed task — the branch and the audit ref then simply
    # point at a commit that is reachable from the task's branch, which is what `test_durable_worktree.py`
    # asserts. Updating them from inside the worktree would work too but would break for a LINKED
    # worktree whose git dir moved, and it reads as if the worktree owned the repository.
    git(active, context.repo_dir, ["update-ref", f"refs/heads/{context.branch}", sha])
    git(active, context.repo_dir, ["update-ref", ref, sha])
    active.beat({"task_id": context.task_id, "phase": "checkpoint", "turn_seq": turn_seq})

    ledger = active.ledger(context.run_id)
    if not _ledger_has_checkpoint(ledger, context.task_id, turn_seq):
        ledger.append(
            "checkpoint",
            task_id=context.task_id,
            payload={
                "turn_seq": turn_seq,
                "ref": ref,
                "sha": sha,
                "created": created,
                "branch": context.branch,
            },
        )
    updated = replace(state, checkpoint_ref=CheckpointRef(ref=ref, sha=sha))
    return CheckpointOutcome(
        state=updated.to_payload(),
        ref=ref,
        sha=sha,
        created=created,
        details={"branch": context.branch, "turn_seq": turn_seq},
    )


def _ledger_has_checkpoint(ledger: RunLedger, task_id: str, turn_seq: int) -> bool:
    """Whether this turn's checkpoint is already recorded — the ledger half of idempotency.

    Needed because the two halves can disagree after a crash: the commit exists but the ledger
    append did not, or the other way round. Checking the ledger keeps a retry from writing a second
    ``checkpoint`` record, which would make "exactly one per turn_seq" false.
    """
    for record in ledger.records():
        if (
            record.kind == "checkpoint"
            and record.task_id == task_id
            and record.payload.get("turn_seq") == turn_seq
        ):
            return True
    return False


def _existing_commit(runtime: ActivityRuntime, tree: str, *, branch: str, message: str) -> str | None:
    """The commit a previous attempt of THIS turn already made, matched by its exact message.

    Searched only on this task's branch in the tree the commit would be made in: a worktree shares
    the parent's object store, so a previous attempt's branch tip is visible from here. Restricting
    the walk to that branch avoids scanning every ref on each cold checkpoint and cannot be satisfied
    by a same-message commit on an unrelated branch.
    """
    out = git_ok(
        runtime,
        tree,
        ["log", f"refs/heads/{branch}", "--fixed-strings", "--format=%H", f"--grep={message}", "-n", "1"],
    )
    if not out or not out.strip():
        return None
    return out.strip().splitlines()[0].strip() or None


def _ensure_on_branch(runtime: ActivityRuntime, tree: str, context: TaskContext, state: TaskState) -> None:
    """Make sure the task branch exists and is checked out in ``tree`` before a commit is made on it.

    The first checkpoint creates the branch at the tree's current HEAD — for a worktree-backed task
    that is the commit the worktree was created from, so the branch, the worktree and the base branch
    start as one commit (no empty diff, no phantom "turn 0"). Later checkpoints continue on the ref the
    previous one left; a branch that exists while this task has no checkpoint is a collision, not
    something to force over.

    ``tree`` is a parameter because the git commands have to run WHERE THE COMMIT WILL BE MADE. A
    ``checkout`` issued in the shared repository would silently move the repo root's HEAD instead of
    the task's worktree — a check that then reports "wrong branch" forever, and a shared tree left on
    another task's branch.
    """
    current = git_ok(runtime, tree, ["rev-parse", "--abbrev-ref", "HEAD"])
    if current is not None and current.strip() == context.branch:
        return
    if state.checkpoint_ref is not None:
        git(runtime, tree, ["checkout", context.branch])
        return
    exists = git_ok(runtime, tree, ["rev-parse", "--verify", f"refs/heads/{context.branch}"])
    if exists is None:
        git(runtime, tree, ["checkout", "-B", context.branch])
        return
    if getattr(context, "worktree_branch", ""):
        # RUN FROM A PULL REQUEST: the branch existed before this run and its commits ARE the work being
        # continued. Checking it out is the whole point; `-B` would reset it to the tree's HEAD and
        # discard the pull request's history, so the adopt path checks out and never resets. Safe by
        # construction: `create_task_worktree` only ever produced this worktree ON this branch, so the
        # checkout is a no-op there; on a resumed tree it is the branch the checkpoint must commit to.
        git(runtime, tree, ["checkout", context.branch])
        return
    raise ActivityError(
        f"branch {context.branch} already exists but this task has no checkpoint to continue from",
        code="activity_branch_conflict",
        details={"branch": context.branch, "turn_seq": state.turn_seq, "tree": tree},
    )


def _commit(runtime: ActivityRuntime, tree: str, *, message: str) -> str:
    """Stage everything in ``tree`` and commit it. Returns the sha (HEAD when nothing changed)."""
    git(runtime, tree, ["add", "-A"])
    staged = git(runtime, tree, ["diff", "--cached", "--name-only"])
    if staged.strip():
        git(
            runtime,
            tree,
            [
                "-c",
                "user.email=codna@codna.ai",
                "-c",
                "user.name=codna",
                "commit",
                "--no-verify",
                "-m",
                message,
            ],
        )
    return git(runtime, tree, ["rev-parse", "HEAD"]).strip()


# -- verify ------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyPlan:
    """The commands to run and the digest that identifies the set."""

    commands: tuple[str, ...]
    digest: str


def _verify_plan(request: VerifyRequest, runtime: ActivityRuntime, *, tree: str) -> VerifyPlan:
    """The request's commands, else the repo's own detected runner (via `codna.testrun`).

    A repo where no runner can be detected is a FAILED verification with a clear code, not a pass:
    "we could not check" must never read as "it is fine".

    Detection runs in ``tree`` — the task's own checkout — because that is the tree about to be tested.
    A monorepo whose runner config only exists on the task's branch (or a worktree with a local
    ``pyproject.toml``) would otherwise be probed in the wrong directory and reported as undetectable.
    """
    if runtime.run_tests is not None:
        return VerifyPlan(("injected",), _digest_commands(("injected",)))
    if request.commands:
        return VerifyPlan(request.commands, _digest_commands(request.commands))
    from ..testrun import resolve_test_command

    resolved = resolve_test_command(tree)
    command = resolved[0] if resolved else None
    if not command:
        raise ActivityError(
            "no verification command was supplied and none could be detected for this repository",
            code="activity_verify_undetectable",
            details={"repo_dir": tree},
        )
    return VerifyPlan((command,), _digest_commands((command,)))


def _digest_commands(commands: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(commands).encode()).hexdigest()


def verify_task(request: VerifyRequest) -> "ActivityOutcome":  # noqa: F821 - see contracts
    """Run the task's verification under `codna.sandbox`.

    The sandbox is the enforcement boundary, not a convenience: it strips write credentials, so a
    verification run — untrusted repository code — can never see the GitHub token the same worker
    uses later to open the pull request.

    A non-zero exit is a RESULT, not an exception: `passed=False` routes the graph back to
    `implement`, whereas raising would retry a command that is deterministically red and spend the
    attempt budget on the same failure.
    """
    from .contracts import ActivityOutcome

    active = _runtime()
    state = decode_request(request, "verify_task")
    # The tests must run against the tree the turn edited. In the shared checkout they would run against
    # whatever another task left there, which is a green build for code nobody wrote.
    tree = ensure_checkout(active, request.context)
    plan = _verify_plan(request, active, tree=tree)
    active.beat({"task_id": request.context.task_id, "phase": "verify", "command_digest": plan.digest})

    if active.run_tests is not None:
        outcome = active.run_tests(request.context, plan.commands)
        passed = bool(getattr(outcome, "passed", False))
        returncode = getattr(outcome, "returncode", None)
        log_path = getattr(outcome, "log_path", None) or _write_verify_log(
            active, request, plan, outcome
        )
    else:
        result = _run_in_sandbox(request, plan, tree=tree)
        passed = bool(result.accepted())
        returncode = result.returncode
        log_path = _write_verify_log(active, request, plan, result)

    ref = f"verify/{request.context.task_id}/{plan.digest[:12]}"
    updated = replace(
        state,
        verify_ref=VerifyRef(ref=ref, passed=passed, command_digest=plan.digest, log_path=log_path),
        attempts=state.attempts.incremented("verify"),
    )
    active.ledger(request.context.run_id).append(
        "verify",
        task_id=request.context.task_id,
        payload={
            "ref": ref,
            "passed": passed,
            "command_digest": plan.digest,
            "returncode": returncode,
            "commands": list(plan.commands),
            "log_path": log_path,
        },
    )
    return ActivityOutcome(
        state=updated.to_payload(),
        details={"verify_ref": ref, "passed": passed, "returncode": returncode},
    )


def _run_in_sandbox(request: VerifyRequest, plan: VerifyPlan, *, tree: str) -> Any:
    """Execute each command under the sandbox, in ``tree``, stopping at the first failure."""
    from ..sandbox import Sandbox, scrub_env

    sandbox = Sandbox(
        network="deny",
        timeout_seconds=int(request.timeout_s),
        env=scrub_env(dict(os.environ)),
    )
    last: Any = None
    for command in plan.commands:
        last = sandbox.run(["/bin/sh", "-lc", command], cwd=tree)
        if not last.accepted():
            return last
    if last is None:  # pragma: no cover - the contract guarantees at least one command
        raise ActivityError(
            "the verification plan had no commands", code="activity_verify_undetectable", details={}
        )
    return last


def _write_verify_log(
    runtime: ActivityRuntime, request: VerifyRequest, plan: VerifyPlan, result: Any
) -> str:
    """Persist the output beside the ledger; return a path RELATIVE to the run directory.

    Relative because `VerifyRef.log_path` is validated against a reference pattern that refuses
    absolute paths — and because a run directory that moves must not invalidate its own history.
    """
    logs = _run_dir(runtime.runs_root, request.context.run_id) / "verify"
    logs.mkdir(parents=True, exist_ok=True)
    turn_seq = int(request.state.get("turn_seq") or 0)
    path = logs / f"turn-{turn_seq}-{plan.digest[:12]}.log"
    body = "\n".join(f"$ {command}" for command in plan.commands)
    body += f"\n\nreturncode: {getattr(result, 'returncode', '?')}"
    body += f"\ntimed_out: {bool(getattr(result, 'timed_out', False))}\n"
    body += "\n-- stdout --\n" + str(getattr(result, "stdout", "")) + "\n"
    body += "-- stderr --\n" + str(getattr(result, "stderr", "")) + "\n"
    path.write_text(body, encoding="utf-8")
    return posixpath.join("verify", path.name)


# -- review ------------------------------------------------------------------------------


def review_task(request: ReviewRequest) -> "ActivityOutcome":  # noqa: F821 - see contracts
    """Review the checkpointed diff through `codna.review_findings`.

    The verdict is a closed set, and the mapping is deliberately unforgiving: only a clean review is
    ``approved``. ``neutral`` means findings that do not block — the reviewer found something and did
    not say yes, so it is ``changes_requested``, which routes back to `implement`. The one wrong
    default that matters is opening a pull request because we did not understand the verdict.
    """
    from .contracts import ActivityOutcome

    active = _runtime()
    state = decode_request(request, "review_task")
    checkpoint = state.checkpoint_ref
    if checkpoint is None:  # pragma: no cover - the contract already refuses this
        raise ActivityError(
            "a review needs a checkpoint",
            code="activity_review_without_checkpoint",
            details={"task_id": request.context.task_id},
        )
    active.beat({"task_id": request.context.task_id, "phase": "review", "sha": checkpoint.sha})
    # The diff being reviewed is the one in the tree that produced the checkpoint, so the range is
    # resolved there. In a shared checkout the range would still resolve (the objects are shared), but
    # the reviewer's own repository reader would see the wrong working tree.
    tree = ensure_checkout(active, request.context)
    report = (
        active.run_review(request, checkpoint.sha)
        if active.run_review is not None
        else _review_via_findings(request, checkpoint.sha, tree=tree)
    )
    verdict = _verdict_from(report)
    findings = _findings_summary(report)
    updated = replace(
        state,
        review_verdict=verdict,
        attempts=state.attempts.incremented("review"),
    )
    active.ledger(request.context.run_id).append(
        "review",
        task_id=request.context.task_id,
        payload={
            "verdict": verdict,
            "sha": checkpoint.sha,
            "conclusion": getattr(report, "conclusion", None),
            "findings": findings,
        },
    )
    return ActivityOutcome(
        state=updated.to_payload(),
        details={"verdict": verdict, "findings": findings},
    )


def _review_via_findings(request: ReviewRequest, sha: str, *, tree: str) -> Any:
    """The real review: the shared findings pipeline over the commit range, read in ``tree``."""
    from ..review_findings import load_review_config, run_diff_review

    # The run's OWN provider, not the engine's default. `codna run --provider ollama` is a decision
    # for the whole task, and a review is part of it: without these the review step read only the
    # process environment, so a durable run on a keyless local provider planned, implemented,
    # verified and checkpointed on it and then failed its review with "no API key in env for
    # 'anthropic'" (measured: run-20261107T165622Z, ledger `verify passed` → review error).
    #
    # `provider_key_source` is deliberately NOT forwarded: it tells a SESSION where a credential
    # comes from, and a review request carries no credential at all (the runner resolves one below
    # the model). Passing it would be a field `run_diff_review` does not have.
    context = request.context
    coordinates = (
        {"provider": context.provider, "model": context.model, "base_url": context.base_url}
        if context.provider is not None
        else {}
    )
    return run_diff_review(
        tree,
        repository=tree,
        diff_range=f"{request.base_sha or context.base_branch}...{sha}",
        config=load_review_config(tree),
        head_sha=sha,
        **coordinates,
    )


def _verdict_from(report: Any) -> str:
    conclusion = getattr(report, "conclusion", None)
    if conclusion == "success":
        return "approved"
    if conclusion == "failure":
        return "rejected"
    return "changes_requested"


def _findings_summary(report: Any) -> dict[str, int]:
    """Findings counted by severity — small enough for the ledger, never the findings themselves."""
    summary: dict[str, int] = {}
    for finding in getattr(report, "findings", None) or ():
        severity = str(getattr(finding, "severity", "unknown"))
        summary[severity] = summary.get(severity, 0) + 1
    return summary


# -- open_pr -----------------------------------------------------------------------------
#
# The publish decision — open a pull request, record a SKIP when the repository has nothing to
# publish to, or refuse with a named code when a GitHub remote has no write token — lives in
# `publish_steps.py`. It moved there because this module is at the repository's 1000-line ceiling and
# because the reasoning is worth reading as one piece: see that module's docstring for the contract.
# It is RE-EXPORTED here on the same rule as every other step, so `activities.py` registers one object
# and a test calling `activity_steps.open_pr` calls what a worker runs.

open_pr = _publish_steps.open_pr


# -- episode + notify --------------------------------------------------------------------


def remember_episode(request: EpisodeRequest) -> EpisodeOutcome:
    """Record what the task learned — reference-only, beside the run's ledger.

    A deterministic file plus a ledger note, not a Telys write: long-term memory is the ledger +
    episode path, and the Telys index is the worker's concern (D2.4 owns the license and kernel).
    What this guarantees is that the episode EXISTS with a digest recorded, so a later indexer can
    be re-run from the file without a transcript that no longer exists — and so re-running this
    activity produces the same file and the same digest.
    """
    active = _runtime()
    state = decode_request(request, "remember_episode")
    active.beat({"task_id": request.context.task_id, "phase": "episode"})
    body: dict[str, Any] = {
        "schema_version": 1,
        "run_id": request.context.run_id,
        "task_id": request.context.task_id,
        "outcome": request.outcome,
        # D5 goal mode: the plan-side recap reads these two, so "what did this task do, and what does
        # it point at" is answerable from the episode alone.
        "brief": request.brief,
        "plan_ref": state.plan_ref,
        "turn_seq": state.turn_seq,
        "attempts": state.attempts.to_payload(),
        "budget": state.budget.to_payload(),
        "verify_ref": state.verify_ref.to_payload() if state.verify_ref else None,
        "review_verdict": state.review_verdict,
        "checkpoint_ref": state.checkpoint_ref.to_payload() if state.checkpoint_ref else None,
        "pr": state.pr.to_payload() if state.pr else None,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    episodes = _run_dir(active.runs_root, request.context.run_id) / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    path = episodes / f"{request.context.task_id}.json"
    path.write_text(
        json.dumps({**body, "digest": digest}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = active.ledger(request.context.run_id).append(
        "note",
        task_id=request.context.task_id,
        payload={
            "event": "episode.remembered",
            "outcome": request.outcome,
            "digest": digest,
            "path": f"episodes/{path.name}",
        },
    )
    return EpisodeOutcome(
        path=f"episodes/{path.name}",
        digest=digest,
        ledger_seq=record.seq if record is not None else None,
    )


def worktree_cleanup(request: WorktreeCleanupRequest) -> WorktreeCleanupOutcome:
    """Reclaim (or deliberately keep) a finished run's per-task worktrees.

    THE POLICY, in one place: a run that gave each task its own checkout removes those checkouts when
    it reaches a terminal outcome, and records why in the ledger. A leaked worktree is a whole extra
    checkout on disk, so "leave it, someone may want it" is not a policy — but nor is deleting it
    silently: a run KEPT for debugging passes ``keep=True`` and says so, so "no worktrees" and "we chose
    to keep them" are distinguishable from "the sweep never ran".

    Safe by construction:

    * It removes only worktrees git reports UNDER ``<runs_root>/<run>/``. A task's commits live on its
      branch in the shared repository, so nothing a checkpoint recorded is lost with the checkout.
    * It is idempotent — a second call finds nothing and returns an empty tuple.
    * Clean-up FAILURE does not fail the run. The run already finished; a sweep that cannot delete a
      directory must not turn a completed run into a failed one, so the failure is recorded as a ledger
      note with its code and the caller sees ``removed`` for what it DID manage.
    """
    active = _runtime()
    request.validate()
    if request.keep:
        _note_cleanup(active, request, removed=(), kept=True)
        return WorktreeCleanupOutcome(removed=(), kept=True, reason=request.reason)
    try:
        removed = remove_run_worktrees(
            repo_dir=request.repo_dir, runs_root=active.runs_root, run_id=request.run_id
        )
    except Exception as exc:  # noqa: BLE001 - a finished run must not be un-finished by its own tidying
        code = str(getattr(exc, "code", "activity_worktree_cleanup_failed"))
        _note_cleanup(active, request, removed=(), kept=True, failed=code)
        return WorktreeCleanupOutcome(removed=(), kept=True, reason=f"{request.reason}: {code}")
    _note_cleanup(active, request, removed=[str(path) for path in removed], kept=False)
    return WorktreeCleanupOutcome(
        removed=tuple(str(path) for path in removed), kept=False, reason=request.reason
    )


def _note_cleanup(
    runtime: ActivityRuntime,
    request: WorktreeCleanupRequest,
    *,
    removed: Sequence[str],
    kept: bool,
    failed: str | None = None,
) -> None:
    """Record the sweep's outcome. One ledger note, so the decision has an audit trail."""
    payload: dict[str, Any] = {
        "event": "worktrees.cleaned",
        "reason": request.reason,
        "kept": kept,
        "removed": list(removed),
        "count": len(removed),
    }
    if failed is not None:
        payload["failed"] = failed
    runtime.ledger(request.run_id).append("note", task_id=None, payload=payload)


def notify(request: NotifyRequest) -> NotifyOutcome:
    """Append a structured notification to the run's ledger and log it once.

    A notification IS a ledger record: the Run API renders the ledger, so there is no second
    channel and nothing to fan out. That is what makes `notify` unable to fail because nobody is
    listening — the property a webhook would not have.
    """
    active = _runtime()
    request.validate()
    payload: dict[str, Any] = {
        "event": "notification",
        "kind": request.kind,
        "message": request.message,
        # A record-shaped notification carries its evidence in the same ledger row, so an auditor
        # reading `goal.verdict` sees the checks it was taken over. Always present (empty for a
        # message-only notification), so a reader needs no branch.
        "evidence": dict(request.evidence),
    }
    if request.actor:
        # Present only when a person caused this row, so an approval's ledger record names WHO
        # without a second record and without a branch a reader has to know about.
        payload["identity"] = request.actor
    if request.repo_dir:
        # THE CHECKOUT THIS RUN WORKS IN, recorded once by the run's opening notification. `GET /runs`
        # reads it back so a sidebar row names the repository the run belongs to rather than the folder
        # the reading window happens to have open. Absent for every other notification, so a
        # message-only row keeps the payload shape it had — a reader needs no branch to tell them apart,
        # because the ABSENCE of the key is the honest answer for a row that is not about a checkout.
        payload["repo_dir"] = request.repo_dir
    record = active.ledger(request.run_id).append(
        "note",
        task_id=request.task_id,
        payload=payload,
    )
    if request.kind in _CONSOLE_KINDS:
        active.emit(
            {
                "event": request.kind,
                "run_id": request.run_id,
                "task_id": request.task_id,
                "message": request.message,
            }
        )
    return NotifyOutcome(seq=record.seq if record is not None else None, kind=request.kind)


