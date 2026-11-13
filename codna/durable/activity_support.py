"""The runtime, the error and the paid-turn machinery every durable activity shares.

Split from `activities.py` when that module crossed the repository's 1000-line ceiling
(`test_modularity.py`). The boundary is "what an activity body needs" vs. "what a body does":

* this module owns the INJECTED DEPENDENCIES (`ActivityRuntime`), the error type the whole step
  layer raises (`ActivityError`), request decoding/validation, the git runner, and the checkout a
  step works in;
* `turn_driver.py` owns the one piece of machinery that must never be re-implemented per step —
  `drive_turn`, which buys at most one paid turn per ``turn_key``;
* `activity_steps.py` owns the eight step implementations;
* `activities.py` owns registration and the process-wide runtime, and re-exports both.

`drive_turn` is imported at the END of this module and re-exported, for two reasons that pull in the
same direction: every step already imports THIS module, so the money contract stays reachable from
one place; and `turn_driver` needs this module's runtime types, so the import has to come after they
exist. Keeping it out of `activity_steps.py` is what makes the contract impossible to bypass by
adding a new step that posts to the sidecar itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from temporalio import activity

from .contracts import HEARTBEAT_INTERVAL_S
# `_project_rules` is imported for RE-EXPORT: `test_goal_rules.py` exercises it directly, and it
# has always been reachable from this module. `_session_spec` is what the two call sites below use.
from .session_spec import _project_rules, _session_spec  # noqa: F401
from .state import TaskState, TaskStateError
from .worktree import GIT_ENV_DROP

__all__ = [
    "ActivityError",
    "ActivityRuntime",
    "HEARTBEAT_INTERVAL_S",
    "RESUMED_EVENT",
    "TurnOutcome",
    "beat_now",
    "decode_request",
    "default_heartbeat",
    "drive_turn",
    "ensure_checkout",
    "git",
    "git_ok",
    "owner_for",
    "redact",
    "working_dir",
]


class ActivityError(RuntimeError):
    """A durable step failed in a way a retry cannot fix. Always carries a stable code."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass
class ActivityRuntime:
    """Every external dependency an activity has, injected.

    This is what lets the activities run in a test with no sidecar, no network and no git remote —
    and it is how the worker injects the keystore-resolved token (D2.4), so no activity reads a
    provider key itself. Fields are explicit rather than a bag of kwargs: a new dependency is a
    visible change to this contract, not a hidden `os.environ` read buried in an activity.
    """

    runs_root: Path
    # Takes the OWNER of the run whose turn is about to run. The engine stamps a session with its
    # owner at CREATE time and refuses any later request whose `x-session-owner` disagrees, so the
    # client that starts a turn MUST be built with the same owner the session was created with.
    # A no-argument factory cannot express that: it would default to "anonymous" and be refused
    # with `session_owner_mismatch` on its own session's first turn. The owner is the run id.
    sidecar_factory: Callable[[str], Any]
    ledger_factory: Callable[[str], Any]
    clock: Callable[[], float] = time.time
    heartbeat: Callable[[Mapping[str, Any]], None] | None = None
    log: Callable[[str], None] | None = None
    # Resolved lazily so importing this module never needs credentials. Returns a token or None.
    github_token: Callable[[], str | None] | None = None
    # Cancellation, as a plain callable: a body runs on a worker thread, where Temporal's activity
    # context (a contextvar) is absent, so the workflow thread's probe is what the body sees.
    cancelled: Callable[[], bool] | None = None
    # Test seams for the three steps that would otherwise shell out or call a model:
    run_tests: Callable[[Any, tuple[str, ...]], Any] | None = None
    run_review: Callable[[Any, str], Any] | None = None
    push_and_open: Callable[[Any, str], Mapping[str, Any]] | None = None
    git: Callable[[str, Sequence[str]], str] | None = None
    # D5 goal mode: the CI gate's seam. A callable taking a `WatchCiRequest` and returning a
    # `WatchCiOutcome`, so a test (and a host with another CI provider) drives the loop without a
    # GitHub token — while the REAL body still runs, writing its ledger note and classification.
    watch_ci: Callable[[Any], Any] | None = None

    def emit(self, event: Mapping[str, Any]) -> None:
        if self.log is not None:
            self.log(json.dumps(dict(event), sort_keys=True, default=str))

    def beat(self, details: Mapping[str, Any]) -> None:
        (self.heartbeat or default_heartbeat)(details)

    def is_cancelled(self) -> bool:
        """Whether this activity was cancelled. On a worker thread the probe is the bound one."""
        if self.cancelled is not None:
            return bool(self.cancelled())
        return _temporal_cancelled()

    def ledger(self, run_id: str) -> Any:
        """The run's ledger. One indirection so every step opens it the same way."""
        return self.ledger_factory(run_id)


def default_heartbeat(details: Mapping[str, Any]) -> None:
    """Heartbeat through Temporal when there is an activity context; no-op otherwise.

    These functions also run in tests, where there may be no activity context at all.
    `activity.heartbeat` raises there, and a heartbeat that raises would fail a step that is
    otherwise fine, so the absence of a context is detected rather than assumed.
    """
    try:
        activity.heartbeat(dict(details))
    except RuntimeError:
        return


def beat_now(details: Mapping[str, Any]) -> None:
    """Heartbeat from a workflow thread while a body runs on a worker thread (the graph's nodes)."""
    default_heartbeat(details)


def _temporal_cancelled() -> bool:
    try:
        return bool(activity.is_cancelled())
    except RuntimeError:
        return False


def decode_request(request: Any, label: str) -> TaskState:
    """Validate a request's contract checks and return its decoded `TaskState`.

    Runs on EVERY attempt, including a replayed one: a retry that receives a payload this build
    cannot validate must fail here rather than three steps later.
    """
    try:
        payload = request.validate()
    except (TypeError, ValueError) as exc:
        raise ActivityError(
            f"{label}: invalid request: {exc}",
            code=str(getattr(exc, "code", "activity_request_invalid")),
            details={"cause": getattr(exc, "details", None) or {}},
        ) from exc
    try:
        return TaskState.from_payload(payload)
    except TaskStateError as exc:
        raise ActivityError(
            f"{label}: invalid task state: {exc}",
            code="activity_state_invalid",
            details={"cause": exc.as_dict()["error"]},
        ) from exc


# -- git --------------------------------------------------------------------------------

# Environment variables that would redirect git away from the checkout we were given. ONE list, in
# `worktree.py` (which needs it for the same reason), so the two git runners cannot drift apart.
_GIT_ENV_DROP: Final[tuple[str, ...]] = GIT_ENV_DROP


def redact(text: str) -> str:
    """Strip an embedded credential from a URL: ``https://x-access-token:<t>@host`` → ``https://host``."""
    marker = "x-access-token:"
    index = text.find(marker)
    if index < 0:
        return text
    at = text.find("@", index)
    if at < 0:
        return text
    scheme_end = text.find("://")
    prefix = text[: scheme_end + 3] if scheme_end >= 0 else ""
    return f"{prefix}{text[at + 1 :]}"


def _default_git(repo_dir: str, args: Sequence[str]) -> str:
    """Run one git command in ``repo_dir`` and return stdout, or raise with git's own message.

    A cleaned environment on purpose: a worker inherits its launcher's environment, and an inherited
    ``GIT_DIR`` would silently point these commits at an unrelated repository. Arguments are
    redacted before they reach an error or a log: a caller may pass an authenticated push URL.
    """
    env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}
    try:
        completed = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ActivityError(
            f"git could not run in {repo_dir}: {exc}",
            code="activity_git_unavailable",
            details={"repo_dir": repo_dir, "error": str(exc)},
        ) from exc
    if completed.returncode != 0:
        raise ActivityError(
            f"git {redact(' '.join(args[:2]))} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[:400]}",
            code="activity_git_failed",
            details={
                "repo_dir": repo_dir,
                "args": [redact(arg) for arg in args],
                "returncode": completed.returncode,
            },
        )
    return completed.stdout


def git(runtime: ActivityRuntime, repo_dir: str, args: Sequence[str]) -> str:
    return (runtime.git or _default_git)(repo_dir, args)


def owner_for(context: Any) -> str:
    """The sidecar owner scope: a durable run is its own owner, so it cannot re-attach to a chat."""
    return f"codna-run-{context.run_id}"


def git_ok(runtime: ActivityRuntime, repo_dir: str, args: Sequence[str]) -> str | None:
    """Run git where a non-zero exit is an ANSWER ("does this ref exist?"), not a failure."""
    try:
        return git(runtime, repo_dir, args)
    except ActivityError:
        return None


# -- the task's working directory (D5) ----------------------------------------------------
#
# EVERY repo-touching step asks these two functions where to work, and nothing else. That is the whole
# threading story: `TaskContext.repo_dir` stays the SHARED checkout (the repository a worktree's objects
# and refs belong to), `working_dir(context)` is the tree a step should edit, and `ensure_checkout`
# makes the latter exist. A step that used `context.repo_dir` directly would silently edit the shared
# tree for a worktree-backed task — which is the bug this indirection exists to make impossible.


def _checkout_id(context: Any) -> str:
    """The task id whose CHECKOUT this context works in (D5 goal mode's fix-CI task shares its parent's).

    Read off the context, never remembered: a retried activity derives the same path from its own
    request. The fallback keeps every pre-D5 context (and every context built without the field)
    meaning "my own worktree", which is what it always did.
    """
    return getattr(context, "worktree_task_id", "") or context.task_id


def working_dir(runtime: ActivityRuntime, context: Any) -> str:
    """The tree this task's steps work in: its own worktree, or the shared checkout.

    Derived, never remembered, so a retried activity computes the same answer from its own request.
    ``context.worktree`` False is the pre-D5 behaviour and returns ``repo_dir`` unchanged.
    """
    if not getattr(context, "worktree", False):
        return context.repo_dir
    from .worktree import worktree_path_for

    return str(
        worktree_path_for(runtime.runs_root, context.run_id, _checkout_id(context))
    )


def ensure_checkout(runtime: ActivityRuntime, context: Any) -> str:
    """``working_dir`` for this task, CREATED if the task asked for a worktree.

    Called at the START of every repo-touching step rather than once, because a worker may pick up a
    task whose activity was already started on another worker (or whose worktree was reclaimed by an
    operator): the step that runs must be able to rely on the tree existing, and only the step that
    runs knows it is running. Idempotent, so the second call in a turn costs one ``git worktree list``.

    Refuse-with-a-code on every failure: a task that cannot get its own checkout must not quietly fall
    back to the shared one, where its commits would land on another task's tree.
    """
    from .worktree import WorktreeError, create_task_worktree

    base = context.repo_dir
    if not getattr(context, "worktree", False):
        return base
    try:
        made = create_task_worktree(
            repo_dir=base,
            runs_root=runtime.runs_root,
            run_id=context.run_id,
            task_id=_checkout_id(context),
            task_number=context.task_number,
            base_branch=context.base_branch,
            adopt_branch=getattr(context, "worktree_branch", ""),
        )
    except WorktreeError as exc:
        raise ActivityError(
            f"could not prepare the worktree for task {context.task_id}: {exc}",
            code="activity_worktree_unavailable",
            details={"cause_code": exc.code, **exc.details},
        ) from exc
    return str(made.path)



# The paid-turn machinery lives in its own module (see `turn_driver`): the ledger-first guarantee is the
# one piece a new step must not be able to re-implement, and it is re-exported here because every step
# already imports THIS module. Moving the name, not the boundary.
# `_scan_ledger` is re-exported for the same reason as the other three: it is a name this module
# published before the split, and a caller (the resume-floor tests) reads it from here. `F401` is
# silenced rather than the import dropped, because an unused RELATIVE import is the point.
from .turn_driver import (  # noqa: E402,F401  (last: it imports THIS module's runtime types)
    RESUMED_EVENT,
    TurnOutcome,
    _scan_ledger,
    drive_turn,
)
