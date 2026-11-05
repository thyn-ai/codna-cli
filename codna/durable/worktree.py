"""Per-task git worktrees — one checkout per task, at the plan's exact path (D5).

A durable run's tasks used to share one checkout, so task 2 began in whatever tree task 1 left behind
and two tasks could never be worked on at once. The plan's model is one worktree per task::

    git worktree add ~/.codna/runs/<run>/wt-<task> -b codna/run-<run>/task-<n>

This module owns that lifecycle (create / remove / list) and the four safety properties that make it
safe to call from a retried activity:

**1. The path is derived, never remembered.** ``runs_root / <run> / wt-<task>`` is a pure function of
the run id and the task id, both of which are already validated tokens. A retry, a replay or a worker
restart therefore computes the same path, with nothing to look up and nothing to disagree about. The
runs root is the one `DurableConfig` resolves (``CODNA_DURABLE_RUNS_ROOT`` or ``~/.codna/runs``), so
the default is the plan's literal path and an override moves the whole run directory together.

**2. Nothing is clobbered.** An existing path that is not this task's worktree is a refusal
(``worktree_path_exists``), and so is an existing branch (``worktree_branch_exists``). Being asked to
create something that already exists is NOT an error when it is exactly what was asked for: an
already-registered worktree of this repository, on this task's branch, at this task's path is returned
with ``created=False``. That is what makes the call idempotent under Temporal's at-least-once retries.

**3. Nothing outside the runs root is ever created or deleted.** Both the path and its parent are
resolved and checked to be strictly INSIDE the runs root before a single git command runs, so a
misconfigured run id or a symlinked directory cannot make ``remove`` delete a user's checkout.

**4. A failure leaves no half state — and that is why the ORDER is worktree-then-branch.**

`git worktree add -b <branch> <path>` is NOT atomic, measured (git 2.54.0): it creates the branch
FIRST, then materialises the worktree, so a refused path leaves an orphan branch behind::

    $ git worktree add -b codna/run-r1/task-2 /tmp/exists main
    Preparing worktree (new branch 'codna/run-r1/task-2')
    fatal: '/tmp/exists' already exists
    $ git rev-parse --verify refs/heads/codna/run-r1/task-2     # ... and the branch is there
    fa99a8e...

An orphan branch is the worst possible leftover for THIS module: "the branch exists while the task has
no checkpoint" is the exact collision `checkpoint_turn` refuses to force over, so one failed create
would wedge every later attempt at that task. So the order here is:

    git worktree add --detach <path> <base>     # creates ONLY the worktree
    git -C <path> checkout --quiet -b <branch>  # creates the branch INSIDE that worktree

The only reachable half state is a registered worktree on a detached HEAD, which is both detectable
(``_registered``) and removable (``remove_task_worktree``), and the ``except`` below removes it before
re-raising. A create that died between the two steps is COMPLETED, not refused: the branch does not
exist yet, so ``checkout -b`` clobbers nothing.

Git is run here rather than through `activity_support` because this module is imported BY it (for
``ensure_checkout``), and because remove has to tell "git does not know this path" (an answer) apart
from "git failed" (an error) — which needs the exit code, not an exception.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

__all__ = [
    "GIT_ENV_DROP",
    "WORKTREE_DIR_PREFIX",
    "Worktree",
    "WorktreeError",
    "branch_for_task",
    "create_task_worktree",
    "list_run_worktrees",
    "remove_run_worktrees",
    "remove_task_worktree",
    "worktree_path_for",
]

# The plan's directory name: ``wt-<task>`` beside the run's ledger, episodes and verify logs.
WORKTREE_DIR_PREFIX: Final = "wt-"

# Environment variables that would redirect git away from the checkout it was given. A worker inherits
# its launcher's environment, and an inherited ``GIT_DIR`` silently points `worktree add` (and every
# later ``-C`` command) at an unrelated repository. The same list guards `activity_support._default_git`,
# which imports THIS tuple, so the two cannot drift.
GIT_ENV_DROP: Final[tuple[str, ...]] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)

# A path token. Deliberately the same class as run ids: these values become directory names AND git
# ref components, so nothing outside ``[A-Za-z0-9_.-]`` may reach the filesystem.
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_GIT_TIMEOUT_S: Final = 300


class WorktreeError(RuntimeError):
    """A worktree operation that must not be silently tolerated. Always carries a stable ``code``."""

    def __init__(self, message: str, *, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, object] = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": str(self), "details": self.details}}


@dataclass(frozen=True)
class Worktree:
    """A task's checkout: where it is, which branch it is on, and whether this call made it."""

    path: Path
    branch: str
    created: bool
    repo_dir: Path


# -- naming ----------------------------------------------------------------------------------


def branch_for_task(run_id: str, task_number: int) -> str:
    """``codna/run-<run>/task-<n>`` — the task's branch, identical to `TaskSpec.branch`.

    One definition would be nicer than two, but the contracts module may not import this one (it is a
    filesystem/git module, and a workflow imports the contracts). `test_durable_worktree.py` pins the two
    together, so a change to either is a red test rather than two branches for one task.
    """
    run = _require_token(run_id, "run_id")
    if not isinstance(task_number, int) or isinstance(task_number, bool) or task_number < 1:
        raise WorktreeError(
            "task_number must be an integer >= 1 (it names the task branch)",
            code="worktree_input_invalid",
            details={"field": "task_number", "value": repr(task_number)},
        )
    return f"codna/run-{run}/task-{task_number}"


def worktree_path_for(runs_root: str | Path, run_id: str, task_id: str) -> Path:
    """``<runs_root>/<run>/wt-<task>``, resolved. Pure: same inputs, same path, always."""
    run = _require_token(run_id, "run_id")
    task = _require_token(task_id, "task_id")
    return _runs_root(runs_root) / run / f"{WORKTREE_DIR_PREFIX}{task}"


# -- create ----------------------------------------------------------------------------------


def create_task_worktree(
    *,
    repo_dir: str | Path,
    runs_root: str | Path,
    run_id: str,
    task_id: str,
    task_number: int,
    base_branch: str,
    adopt_branch: str = "",
) -> Worktree:
    """Create (or find) this task's worktree. Idempotent; never clobbers; never leaves half state.

    Order, and why — see the module docstring: ``--detach`` first so the only possible half state is a
    worktree the ``except`` can remove, then ``checkout -b`` inside it so no orphan branch is ever
    created. A worktree that already exists on a DETACHED head has its branch created here (a create
    that died between the two steps), which is why the branch-exists check below runs before the
    ``checkout`` and not before the worktree.

    ``adopt_branch`` is the RUN-FROM-A-PULL-REQUEST mode: work in an EXISTING branch instead of
    creating ``codna/run-<run>/task-<n>``. Three properties are deliberate and are what keep it from
    being a force-push:

    * The branch must ALREADY EXIST (locally, or as ``refs/remotes/origin/<branch>``). A missing one is
      a refusal — never a new branch, because "fix this PR" pointing at a branch that is not there is a
      caller mistake, not something to invent.
    * The branch is CHECKED OUT, never reset. ``checkout -b`` would move it to the base commit and
      silently discard the PR's work.
    * ``task_number`` is still required and still validated, so the path and the ref guards are the same
      ones as ever; only the BRANCH NAME differs.
    """
    repo = _require_repo(repo_dir)
    root = _runs_root(runs_root)
    path = _guarded_path(root, run_id, task_id)
    branch = _adopted_branch(adopt_branch) if adopt_branch else branch_for_task(run_id, task_number)

    registered = _registered(repo, path)
    if registered is not None and not registered.path.exists():
        # A stale registration whose directory is gone: a manual `rm -rf`, or a create killed between
        # `worktree add` and the branch checkout. The REGISTRATION ITSELF is the evidence — it names
        # this path and this branch — so the state is recoverable rather than ambiguous:
        #
        # * on THIS task's branch: restore the checkout onto it, which resumes the task;
        # * detached (an interrupted create): prune and fall through, where the branch-exists guard
        #   still applies and the normal path completes the create;
        # * any OTHER branch: prune and fall through — that is somebody else's ref, so the guards
        #   below refuse it exactly as they would without the stale entry.
        stale_branch = registered.branch
        _git_ok(repo, ["worktree", "prune"])
        registered = None
        if stale_branch == branch:
            path.parent.mkdir(parents=True, exist_ok=True)
            _git(repo, ["worktree", "add", str(path), branch])
            return Worktree(path=path, branch=branch, created=True, repo_dir=repo)
    if registered is not None:
        if registered.branch == branch:
            return Worktree(path=path, branch=branch, created=False, repo_dir=repo)
        if registered.branch is None and not adopt_branch:
            # A create that died between `worktree add` and `checkout -b`. Complete it below.
            if _branch_exists(repo, branch):
                raise WorktreeError(
                    f"{path} is a detached worktree, but branch {branch} already exists",
                    code="worktree_branch_exists",
                    details={"branch": branch, "path": str(path)},
                )
            _checkout_branch(path, branch)
            return Worktree(path=path, branch=branch, created=False, repo_dir=repo)
        raise WorktreeError(
            f"{path} is already a worktree of {repo} on branch {registered.branch}, not {branch}",
            code="worktree_path_conflict",
            details={"path": str(path), "found": registered.branch, "expected": branch},
        )

    if path.exists():
        raise WorktreeError(
            f"refusing to create a worktree at {path}: the path already exists",
            code="worktree_path_exists",
            details={"path": str(path)},
        )

    if adopt_branch:
        return _adopt_existing_branch(repo=repo, path=path, branch=branch, adopt_branch=adopt_branch)

    if _branch_exists(repo, branch):
        raise WorktreeError(
            f"refusing to create branch {branch}: it already exists",
            code="worktree_branch_exists",
            details={"branch": branch, "path": str(path)},
        )
    base = _resolve_base(repo, base_branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(repo, ["worktree", "add", "--detach", str(path), base])
        _checkout_branch(path, branch)
    except BaseException:
        # Not `except WorktreeError`: a KeyboardInterrupt mid-create must not leave the half state
        # either. The rollback itself is best-effort — it must not mask the original failure, which is
        # the one worth reporting and the one a retry will hit again.
        _rollback(repo, path)
        raise
    return Worktree(path=path, branch=branch, created=True, repo_dir=repo)


def _adopted_branch(value: str) -> str:
    """The name of a branch to ADOPT, validated as a ref component before it reaches git.

    Uses git's own check rather than a regex: ref rules are subtle (no leading ``-``, no ``..``, no
    ``@{``), and a name that git would interpret as an OPTION is the one case that must never reach a
    command line — ``git worktree add <path> -b`` would read ``-b`` as a flag.
    """
    name = (value or "").strip()
    if not name:
        raise WorktreeError(
            "an adopted branch needs a name", code="worktree_input_invalid", details={"field": "adopt_branch"}
        )
    if name.startswith("-"):
        raise WorktreeError(
            f"refusing to adopt {name!r}: a branch name may not start with '-'",
            code="worktree_input_invalid",
            details={"field": "adopt_branch", "value": name},
        )
    return name


def _adopt_existing_branch(
    *, repo: Path, path: Path, branch: str, adopt_branch: str
) -> Worktree:
    """Check an EXISTING branch out into a new worktree. Never creates, never resets.

    Two sources, in order: a local ``refs/heads/<branch>`` (the common case — a previous checkout), then
    ``refs/remotes/origin/<branch>`` (a fresh clone looking at a pull request's head). A local branch
    named ``origin/x`` is deliberately not reachable, which is why the remote ref is spelled out rather
    than resolved by ``_resolve_base``'s generic lookup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _branch_exists(repo, branch):
        try:
            _git(repo, ["worktree", "add", str(path), branch])
        except BaseException:
            _rollback(repo, path)
            raise
        return Worktree(path=path, branch=branch, created=True, repo_dir=repo)

    remote_ref = f"refs/remotes/origin/{branch}"
    code, out, _err = _git_ok(repo, ["rev-parse", "--verify", f"{remote_ref}^{{commit}}"])
    if code != 0 or not out.strip():
        raise WorktreeError(
            f"cannot adopt branch {adopt_branch!r}: it does not exist in {repo} (looked for "
            f"refs/heads/{branch} and {remote_ref}). Fetch the pull request's head first.",
            code="worktree_branch_missing",
            details={"branch": branch, "repo_dir": str(repo)},
        )
    try:
        _git(repo, ["worktree", "add", "-b", branch, str(path), remote_ref])
    except BaseException:
        _rollback(repo, path)
        raise
    return Worktree(path=path, branch=branch, created=True, repo_dir=repo)


# -- remove ----------------------------------------------------------------------------------


def remove_task_worktree(
    *, repo_dir: str | Path, runs_root: str | Path, run_id: str, task_id: str
) -> bool:
    """Remove this task's worktree, returning whether there was one. Idempotent.

    ``--force`` because a turn legitimately leaves untracked files behind (a build artefact, a log):
    the work to keep is COMMITTED, and a commit lives in the repository's object store, not in the
    worktree. Removing the checkout therefore loses nothing a checkpoint recorded.

    "git does not know this path" is an ANSWER, not a failure: it is what the second call of an
    idempotent removal sees, and what a partially-cleaned crash sees. A path that is not a worktree but
    is nonetheless occupied by files is a refusal — deleting somebody's directory because we were
    asked to "remove a worktree" would be the worst possible interpretation of this function.
    """
    repo = _require_repo(repo_dir)
    root = _runs_root(runs_root)
    path = _guarded_path(root, run_id, task_id)

    registered = _registered(repo, path)
    if registered is None:
        _remove_empty_dir(path)
        return False
    _git_ok(repo, ["worktree", "remove", "--force", str(path)])
    _git_ok(repo, ["worktree", "prune"])
    if path.exists():
        _remove_empty_dir(path)
    if path.exists():
        raise WorktreeError(
            f"git no longer lists {path} as a worktree, but the directory is not empty",
            code="worktree_path_occupied",
            details={"path": str(path)},
        )
    return True


def remove_run_worktrees(*, repo_dir: str | Path, runs_root: str | Path, run_id: str) -> tuple[Path, ...]:
    """Remove every worktree of a run. Returns the paths removed; idempotent.

    The operator-facing sweep (and the activity's body): after a run reaches a terminal outcome, or
    after a run was terminated so no activity could finalise it, this is what reclaims the disk.
    """
    repo = _require_repo(repo_dir)
    root = _runs_root(runs_root)
    removed: list[Path] = []
    for path in list_run_worktrees(repo_dir=repo, runs_root=root, run_id=run_id):
        task_id = path.name[len(WORKTREE_DIR_PREFIX) :]
        if remove_task_worktree(repo_dir=repo, runs_root=root, run_id=run_id, task_id=task_id):
            removed.append(path)
    _git_ok(repo, ["worktree", "prune"])
    return tuple(removed)


# -- list ------------------------------------------------------------------------------------


def list_run_worktrees(*, repo_dir: str | Path, runs_root: str | Path, run_id: str) -> tuple[Path, ...]:
    """The run's worktrees, as GIT reports them.

    From git rather than from the filesystem on purpose: a directory that git no longer knows is not a
    worktree (it is litter), and reporting it here would make a leak look like a healthy checkout.
    """
    repo = _require_repo(repo_dir)
    run_dir = _runs_root(runs_root) / _require_token(run_id, "run_id")
    return tuple(
        sorted(entry.path for entry in _worktrees(repo) if _is_inside(entry.path, run_dir))
    )


# -- git -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """One ``git worktree list --porcelain`` stanza: where it is and what it is on."""

    path: Path
    branch: str | None


def _worktrees(repo: Path) -> tuple[_Entry, ...]:
    out = _git(repo, ["worktree", "list", "--porcelain"])
    entries: list[_Entry] = []
    current: dict[str, str] = {}
    for line in out.splitlines() + [""]:
        if not line.strip():
            if current.get("worktree"):
                entries.append(_Entry(path=Path(current["worktree"]), branch=_branch_of(current)))
            current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return tuple(entries)


def _branch_of(stanza: Mapping[str, str]) -> str | None:
    ref = stanza.get("branch", "")
    prefix = "refs/heads/"
    return ref[len(prefix) :] if ref.startswith(prefix) else None


def _registered(repo: Path, path: Path) -> _Entry | None:
    """The entry for ``path``, matched on RESOLVED paths (git reports symlink-free absolute paths)."""
    target = _resolve(path)
    for entry in _worktrees(repo):
        if _resolve(entry.path) == target:
            return entry
    return None


def _branch_exists(repo: Path, branch: str) -> bool:
    return _git_ok(repo, ["rev-parse", "--verify", f"refs/heads/{branch}"])[0] == 0


def _resolve_base(repo: Path, base_branch: str) -> str:
    """The commit a new task branch starts from: the local branch, else its origin ref.

    Explicit, and a refusal when neither resolves. Today's shared-checkout path silently used whatever
    HEAD happened to be, so a mistyped ``--base-branch`` produced a task based on the wrong tree with
    nothing saying so. A per-task worktree has to name its base, so it names it or refuses.
    """
    name = (base_branch or "").strip()
    if not name:
        raise WorktreeError(
            "a worktree needs a base branch", code="worktree_base_missing", details={"branch": ""}
        )
    for candidate in (f"refs/heads/{name}", f"refs/remotes/origin/{name}"):
        code, out, _err = _git_ok(repo, ["rev-parse", "--verify", f"{candidate}^{{commit}}"])
        if code == 0 and out.strip():
            return out.strip()
    raise WorktreeError(
        f"base branch {name!r} does not exist in {repo} (looked for refs/heads/{name} and "
        f"refs/remotes/origin/{name})",
        code="worktree_base_missing",
        details={"branch": name, "repo_dir": str(repo)},
    )


def _checkout_branch(path: Path, branch: str) -> None:
    _git(path, ["checkout", "--quiet", "-b", branch])


def _rollback(repo: Path, path: Path) -> None:
    """Undo a failed create: drop the worktree and its admin files, then the (empty) directory."""
    _git_ok(repo, ["worktree", "remove", "--force", str(path)])
    _git_ok(repo, ["worktree", "prune"])
    try:
        _remove_empty_dir(path)
    except WorktreeError:
        # A directory that will not go away is reported by the ORIGINAL failure's caller, and by the
        # next create (which refuses an existing path). Raising here would replace the real cause.
        return


def _remove_empty_dir(path: Path) -> None:
    if not path.is_dir():
        return
    try:
        path.rmdir()
    except OSError as exc:
        raise WorktreeError(
            f"{path} exists and is not empty, so it is not a worktree this module may delete",
            code="worktree_path_occupied",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def _git(cwd: Path, args: Sequence[str]) -> str:
    code, out, err = _git_ok(cwd, args)
    if code != 0:
        raise WorktreeError(
            f"git {args[0]} failed in {cwd}: {(err or out).strip()[:400]}",
            code="worktree_git_failed",
            details={"repo_dir": str(cwd), "args": list(args[:3]), "returncode": code},
        )
    return out


def _git_ok(cwd: Path, args: Sequence[str]) -> tuple[int, str, str]:
    env = {key: value for key, value in os.environ.items() if key not in GIT_ENV_DROP}
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(
            f"git could not run in {cwd}: {exc}",
            code="worktree_git_unavailable",
            details={"repo_dir": str(cwd), "error": str(exc)},
        ) from exc
    return completed.returncode, completed.stdout, completed.stderr


# -- path guards -----------------------------------------------------------------------------


def _runs_root(runs_root: str | Path) -> Path:
    return _resolve(Path(runs_root).expanduser())


def _resolve(path: Path) -> Path:
    """Resolve symlinks where possible, WITHOUT requiring the path to exist."""
    return Path(os.path.realpath(str(path)))


def _is_inside(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return str(relative) not in ("", ".")


def _guarded_path(root: Path, run_id: str, task_id: str) -> Path:
    """The task's path, refused unless it is strictly inside the runs root.

    Belt and braces on top of `_require_token`: the tokens cannot contain ``/`` or ``..``, so the
    joined path cannot escape — but this is the check that would still hold if a future change widened
    the token class, and it is what `remove` relies on before deleting anything.
    """
    path = _resolve(root / _require_token(run_id, "run_id") / f"{WORKTREE_DIR_PREFIX}{_require_token(task_id, 'task_id')}")
    if not _is_inside(path, root):
        raise WorktreeError(
            f"{path} is outside the runs root {root}; refusing to touch it",
            code="worktree_outside_runs_root",
            details={"path": str(path), "runs_root": str(root)},
        )
    return path


def _require_repo(repo_dir: str | Path) -> Path:
    repo = _resolve(Path(repo_dir).expanduser())
    code, out, err = _git_ok(repo, ["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out.strip() != "true":
        raise WorktreeError(
            f"{repo} is not a git work tree: {(err or out).strip()[:400]}",
            code="worktree_repo_not_git",
            details={"repo_dir": str(repo)},
        )
    return repo


def _require_token(value: str, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN.match(value):
        raise WorktreeError(
            f"{field} must match ^[A-Za-z0-9][A-Za-z0-9_.-]{{0,127}}$ (it names a directory and a ref)",
            code="worktree_input_invalid",
            details={"field": field, "value": value if isinstance(value, str) else type(value).__name__},
        )
    if ".." in value:
        raise WorktreeError(
            f"{field} must not contain '..'",
            code="worktree_input_invalid",
            details={"field": field, "value": value},
        )
    return value
