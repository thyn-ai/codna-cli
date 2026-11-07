"""The publish step: open a pull request, or record honestly why it could not be opened.

`open_pr` is the unconditional last doing-node of every durable run, so "publishing is impossible"
had nowhere to land: the activity raised, the node failed, and a run whose work was committed,
verified and approved was reported as a failed run. This module is that decision, moved out of
`activity_steps.py` (which is at the repository's 1000-line ceiling) and made explicit.

THE CONTRACT, and why the two cases are treated differently:

* **No remote, or a remote that is not GitHub → SKIP.** Nothing here has a pull request to open, and
  no credential would change that. The commits exist on the task's branch, verification ran and the
  reviewer approved: the work is done. The step records ``publish.skipped`` with a named reason and
  finishes without a ``pr``; the task's outcome becomes ``approved_unpublished``. This is the
  "make me a small script" case — a local-only repository — and calling it a failure is the defect
  this change exists to remove.
* **A GitHub remote but no write token → REFUSE.** A GitHub origin is an explicit statement that
  publishing is wanted, so a missing token is a one-command fix, not a property of the repository.
  A skip here would leave a user believing a pull request exists when none does, so the activity
  raises ``activity_pr_token_missing`` (unchanged, and the code callers already handle) AND writes a
  ledger note naming the variables and the remote, so the failure is actionable from the run's own
  audit trail rather than only from a workflow history someone has to fetch.

Why detecting "no remote" HERE is not a new kind of I/O: this module already runs `git rev-parse`
and `git config --get remote.origin.url` through the injected runtime, and a *transient* git problem
must not be mistaken for "this repository has no remote". So the probe asks three questions and
distinguishes the answers: the origin URL (present / absent), and whether the repository itself is
readable (a `git rev-parse --git-dir` failure keeps the hard failure, `activity_git_failed`).

Nothing here writes a credential anywhere, and the remote URL is REDACTED before it reaches a ledger
note: an origin can legitimately embed a token (``https://x-access-token:...@github.com/...``) and a
ledger is the run's permanent, exportable record.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Final, Mapping

from .activity_support import (
    ActivityError,
    ActivityRuntime,
    decode_request,
    ensure_checkout,
    git_ok,
    redact,
)
from .contracts import PullRequestRequest
from .publish import (
    GITHUB_TOKEN_ENV,
    PUBLISH_SKIP_NO_REMOTE,
    PUBLISH_SKIP_REMOTE_NOT_GITHUB,
)
from .state import PullRequestRef, TaskState, TaskStateError

__all__ = ["open_pr", "publish_probe"]

#: The token variables, from the ONE definition in `publish` — see its docstring for why it is not
#: repeated here.
_TOKEN_ENV: Final[tuple[str, ...]] = GITHUB_TOKEN_ENV


def publish_probe(runtime: ActivityRuntime, tree: str) -> tuple[str, str | None]:
    """``(origin_url, skip_reason)`` for a checkout — the reason is None when publishing is possible.

    The three outcomes are distinguished deliberately:

    * an origin that parses as GitHub → ``(url, None)``: publishing is possible (a token may still
      be missing, which the caller decides about);
    * no origin → ``("", "no_remote")``; an origin that is not GitHub → ``(url, "remote_not_github")``.
      Both mean nothing to publish to, and the URL is still returned for the REDACTED ledger note,
      because "your origin is a GitLab URL" is the answer a user needs;
    * git itself cannot run in the tree → it RAISES. A broken checkout is not "no remote", and
      reporting it as a skip would turn a real failure into a silent one.
    """
    from .. import review_github

    if git_ok(runtime, tree, ["rev-parse", "--git-dir"]) is None:
        # A tree that is not a git repository (or a git binary that cannot run) is a hard failure:
        # "no remote" is a statement ABOUT a repository, and this is the absence of one.
        raise ActivityError(
            "the task's checkout does not resolve as a git repository, so publishing cannot be "
            "attempted and the state of the work cannot be established.",
            code="activity_git_failed",
            details={"repo_dir": tree},
        )
    url = (git_ok(runtime, tree, ["config", "--get", "remote.origin.url"]) or "").strip()
    if not url:
        return "", PUBLISH_SKIP_NO_REMOTE
    if review_github.repo_slug_from_url(url) is None:
        # Present but not GitHub-shaped: the user's next step is different from "add a remote"
        # (they may be on GitLab deliberately), so it is its own reason rather than no_remote.
        return url, PUBLISH_SKIP_REMOTE_NOT_GITHUB
    return url, None


def open_pr(request: PullRequestRequest) -> Any:
    """Push the checkpoint commit and open (or find) the pull request. Idempotent by branch.

    Idempotency is the branch name: the PR's head is ``codna/run-<run>/task-<n>``, so a retried
    activity finds the open PR instead of opening a second one. The lookup runs FIRST — a retry
    whose first attempt already pushed and opened must neither push again nor fail because there is
    nothing new to send.

    The two unpublishable cases are described in the module docstring; the skip path returns a state
    with ``publish_skip`` set and NO ``pr``, which is what routes the task to
    ``approved_unpublished`` instead of ``approved``.
    """
    from .activity_responses import PullRequestOutcome

    active = _runtime()
    state = decode_request(request, "open_pr")
    checkpoint = state.checkpoint_ref
    if checkpoint is None:  # pragma: no cover - the contract already refuses this
        raise ActivityError(
            "a pull request needs a checkpoint commit",
            code="activity_pr_without_checkpoint",
            details={"task_id": request.context.task_id},
        )
    active.beat({"task_id": request.context.task_id, "phase": "open_pr", "sha": checkpoint.sha})
    # `push_branch` verifies the sha exists in the repository it is given; a worktree shares that
    # object store, so pushing from here finds the commit the checkpoint made there. The ORIGIN URL is
    # read from the task's tree too (a worktree inherits the parent's remotes).
    tree = ensure_checkout(active, request.context)

    origin, skip_reason = publish_probe(active, tree)
    if skip_reason is not None:
        return _skipped(active, request, state, skip_reason, origin=origin)

    token = _github_token(active)
    if not token:
        _record_token_refusal(active, request, origin)
        raise ActivityError(
            "opening a pull request needs a GitHub write token, and none was configured. This "
            "repository HAS a GitHub origin, so publishing was requested: set CODNA_GITHUB_TOKEN "
            "(or GITHUB_TOKEN) in the worker's environment, or store it in the keystore.",
            code="activity_pr_token_missing",
            details={
                "task_id": request.context.task_id,
                "branch": request.context.branch,
                "variable_names": list(_TOKEN_ENV),
            },
        )

    opened = (
        active.push_and_open(request, checkpoint.sha)
        if active.push_and_open is not None
        else _push_and_open(active, request, checkpoint.sha, tree=tree, origin=origin, token=token)
    )
    pr = _pr_ref_from(opened, branch=request.context.branch, sha=checkpoint.sha)
    updated = replace(state, pr=pr, publish_skip=None, attempts=state.attempts.incremented("open_pr"))
    # At most ONE record per task, for the same reason as the skip note: a retried activity must not
    # append a second `pull_request.opened` for the same pull request. The ledger's dedupe key is
    # `(session_id, event_index)` and a note carries neither, so the check is explicit.
    _append_note_once(
        active,
        request,
        event="pull_request.opened",
        payload={
            "event": "pull_request.opened",
            "number": pr.number,
            "url": pr.url,
            "branch": pr.branch,
            "head_sha": pr.head_sha,
        },
    )
    return PullRequestOutcome(
        state=updated.to_payload(),
        details={"number": pr.number, "url": pr.url, "branch": pr.branch},
    )


def _append_note_once(
    active: ActivityRuntime,
    request: PullRequestRequest,
    *,
    event: str,
    payload: dict[str, Any],
) -> None:
    """Append one publish note per (task, event), however many times the activity is retried.

    The ledger's own dedupe key is `(session_id, event_index)`, which a bookkeeping note does not
    carry — so an activity that raises is retried (`ACTIVITY_RETRIES` attempts) and would write the
    same note once per attempt. MEASURED: the refusal path wrote three identical `publish.blocked`
    records. "One record per fact" is a property this ledger is read for, so the check is explicit:
    the same pattern `checkpoint_turn` uses for its own idempotency.

    The scan is over this run's ledger only, and it is bounded by one publish step per task — the
    same cost `_ledger_has_checkpoint` already pays for the same reason.
    """
    ledger = active.ledger(request.context.run_id)
    for record in ledger.records():
        if record.kind == "note" and record.task_id == request.context.task_id:
            recorded = record.payload
            if recorded.get("event") == event and recorded.get("reason") == payload.get("reason"):
                return
    ledger.append("note", task_id=request.context.task_id, payload=payload)


def _skipped(
    active: ActivityRuntime,
    request: PullRequestRequest,
    state: TaskState,
    reason: str,
    *,
    origin: str,
) -> Any:
    """Record the skip and finish WITHOUT a pull request. The work is done; publishing is not.

    The note is the audit trail a user reads, so it says what was skipped, why, and what already
    exists (the branch and the commit) — "nothing happened" is not the same fact as "the branch is
    there and there is nowhere to send it". Written at most ONCE per task (see `_append_note_once`).
    """
    from .activity_responses import PullRequestOutcome

    updated = replace(
        state,
        pr=None,
        publish_skip=reason,
        attempts=state.attempts.incremented("open_pr"),
    )
    _append_note_once(
        active,
        request,
        event="publish.skipped",
        payload={
            "event": "publish.skipped",
            "reason": reason,
            "branch": request.context.branch,
            "head_sha": state.checkpoint_ref.sha if state.checkpoint_ref else None,
            # Redacted: an origin can embed a credential, and a ledger is exported. Absent for a
            # repository with no remote at all, which is its own (truthful) answer.
            "origin": redact(origin) if origin else None,
            "message": _skip_message(reason),
        },
    )
    return PullRequestOutcome(
        state=updated.to_payload(),
        details={"skipped": True, "reason": reason, "branch": request.context.branch},
    )


def _skip_message(reason: str) -> str:
    if reason == PUBLISH_SKIP_NO_REMOTE:
        return (
            "the repository has no 'origin' remote, so there is nothing to open a pull request "
            "against; the work is committed on this task's branch and verification and review "
            "completed."
        )
    if reason == PUBLISH_SKIP_REMOTE_NOT_GITHUB:
        return (
            "the repository's 'origin' remote is not a GitHub repository, so this worker cannot open "
            "a pull request; the work is committed on this task's branch."
        )  # pragma: no cover - the closed set is checked by `TaskState`
    raise ValueError(f"{reason!r} is not a publish-skip reason")


def _record_token_refusal(
    active: ActivityRuntime, request: PullRequestRequest, origin: str
) -> None:
    """The ledger half of the refusal: WHAT to set, and WHICH remote asked for it.

    Written before the error is raised, because the error goes to the workflow (and from there to a
    Temporal history an operator may not open), while the LEDGER is what the Run API, `codna durable
    ledger` and the IDE all read. Recording it also makes the refusal auditable: "the run stopped
    because no token was configured" is a fact about the run, not only about one activity attempt.

    Written at most ONCE per task: the activity is retried up to `ACTIVITY_RETRIES` times on the same
    refusal, and three identical records for one fact would misrepresent the run.
    """
    _append_note_once(
        active,
        request,
        event="publish.blocked",
        payload={
            "event": "publish.blocked",
            "reason": "no_credential",
            "branch": request.context.branch,
            "origin": redact(origin),
            "variable_names": list(_TOKEN_ENV),
            "message": (
                "this repository has a GitHub origin, so a pull request was expected, but no write "
                f"token was configured ({', '.join(_TOKEN_ENV)}); it was not skipped."
            ),
        },
    )


def _push_and_open(
    runtime: ActivityRuntime,
    request: PullRequestRequest,
    sha: str,
    *,
    tree: str,
    origin: str,
    token: str,
) -> Mapping[str, Any]:
    """Push the branch, then create (or find) the PR through the shared GitHub helpers.

    The origin and the token are PARAMETERS, not read here again: the caller has already resolved
    both (and refuses a missing remote or token with its own named code), so a second read would be
    a second chance for the two to disagree.
    """
    from .. import review_github
    from ..gitpush import GitPushError, push_branch

    slug = review_github.repo_slug_from_url(origin)
    if not slug:  # pragma: no cover - the caller probed the origin before calling this
        raise ActivityError(
            "the repository has no parseable GitHub origin, so a pull request cannot be opened",
            code="activity_pr_no_origin",
            details={"repo_dir": tree},
        )
    existing = review_github.find_open_pr_for_branch(slug, request.context.branch, token)
    if existing is not None:
        return existing
    try:
        push_branch(
            tree,
            branch=request.context.branch,
            sha=sha,
            push_url=f"https://x-access-token:{token}@github.com/{slug}.git",
        )
    except GitPushError as exc:
        raise ActivityError(
            f"pushing {request.context.branch} failed: {exc}",
            code="activity_push_failed",
            details={"branch": request.context.branch},
        ) from exc
    return review_github.create_pull_request(
        slug,
        token=token,
        title=request.title,
        head=request.context.branch,
        base=request.context.base_branch,
        body=request.body,
    )


def _github_token(runtime: ActivityRuntime) -> str | None:
    """The token the worker injected, else the process environment (the historical fallback).

    The runtime's provider is the real resolution (environment, then the OS keystore) and it is
    resolved LAZILY, per call — see `worker.github_token_provider`. The environment fallback keeps a
    directly-invoked activity (the tests, an operator's REPL) behaving as it always did.

    ``push_and_open`` is the seam a harness scripts (the GitHub API is the one genuinely external
    boundary), and it is called BEFORE this resolution only when it is set: a harness that scripts the
    push must still see the SAME credential refusal a real worker would produce for a GitHub origin
    with no token, so the token check runs first and the seam is used only once it passes.
    """
    if runtime.github_token is not None:
        return runtime.github_token()
    for name in _TOKEN_ENV:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _pr_ref_from(opened: Mapping[str, Any], *, branch: str, sha: str) -> PullRequestRef:
    number = opened.get("number")
    url = opened.get("url")
    if not isinstance(number, int) or isinstance(number, bool) or not isinstance(url, str):
        raise ActivityError(
            "the pull request result is missing an integer number or a url",
            code="activity_pr_response_invalid",
            details={"number": repr(number), "url": repr(url)},
        )
    try:
        return PullRequestRef(number=number, url=url, branch=branch, head_sha=sha)
    except TaskStateError as exc:
        raise ActivityError(
            f"the pull request result is not representable in task state: {exc}",
            code="activity_pr_response_invalid",
            details={"cause": exc.as_dict()["error"]},
        ) from exc


# The step layer's runtime accessor, imported LAZILY: `activity_steps` imports this module to
# register the step, and a module-level import back would be a cycle. The step bodies it wraps are
# the same objects the activity registry holds, so this indirection changes only the import order.
def _runtime() -> ActivityRuntime:
    from . import activity_steps

    return activity_steps._runtime()