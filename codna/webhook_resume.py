"""Resume, or honestly close, the jobs a webhook restart interrupted.

The hosted webhook is one Fly machine that every merge to ``main`` redeploys. A job running at
that moment is killed: ``WorkerPool.stop`` drains for the grace window, the process exits, and
the queue row stays ``running`` with its Check Run ``in_progress``. On the next boot
``WebhookQueue.recover_stale`` requeues the row -- but the Check Run's id lived only in a local
variable of ``process_job``, so the ONLY thing that could ever complete it was the retry reaching
``complete_stale_check_runs`` after token minting and the account bridge. When the retry died
before that point (or the row was already on its last attempt) the run spun forever, and since
``codna review`` is a required check the pull request could not merge until someone pushed again.
Live: thyn-ai/algenta#1056 head 13b07d1b, check run 105937057683 started 2026-09-19T17:27:51Z,
machine replaced 17:28:49Z-17:29:27Z, still ``in_progress`` at 17:44Z with one row gone
``failed`` and no second run.

Two pieces, both stdlib, both unit-tested with fakes:

* :class:`RunningJobRegistry` -- one small JSON file per claimed job, written on claim, updated
  with the Check Run id the moment it exists, removed on finish, on the volume BESIDE the queue
  (``<queue dir>/running/``) so it survives SIGTERM and SIGKILL alike. ``stop()`` marks whatever
  is still registered as interrupted by ``sigterm`` so the next boot can say why.
* :func:`reconcile_interrupted` -- at startup, after ``recover_stale``, every registered job is
  resolved exactly once: resumed (its stale run completed as ``cancelled`` and a fresh attempt
  replaces it), superseded (the PR head moved: the stale head is not re-run), or cancelled with
  the re-trigger comment (out of attempts, or already resumed once). Nothing this App opened is
  left ``in_progress``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import webhook_github
from .webhook import WebhookJob, check_run_name
from .webhook_queue import QueuedJob, WebhookQueue

# A job is brought back automatically after ONE restart. A second interruption of the same job
# (deploys 6 minutes apart on 2026-09-19 restarted the machine twice) is not re-run again: the
# check is completed as cancelled with the comment that re-triggers it, so a job that cannot
# finish between deploys never loops, and the queue's own attempt cap still bounds everything.
_MAX_RESUMES = 1
REGISTRY_DIRNAME = "running"

_RETRIGGER = {
    "review": "Comment `@codna review` on the pull request to run it again.",
    "fix": "Reply `@codna fix` on the finding (or re-apply the `codna-fix` label) to run it again.",
    "secure": "Re-apply the `codna-secure` label to run it again.",
    "queue": "Re-queue the pull request to run it again.",
}


def registry_dir_for(queue_path: str | Path) -> Path:
    """Where the registry lives: beside the durable queue (``/data/running`` on the hosted app)."""
    return Path(queue_path).parent / REGISTRY_DIRNAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunningJob:
    """What the next process needs to know about a job this one was running."""

    row_id: int
    kind: str
    repo: str
    ref: str | None
    pr_number: int | None
    installation_id: int | None
    issue_number: int | None
    reason: str
    delivery_id: str | None
    attempt: int
    started_at: str
    check_run_id: int | None = None
    resumes: int = 0
    interrupted_by: str | None = None
    interrupted_at: str | None = None


class RunningJobRegistry:
    """Per-job JSON files in one directory. Every method is best-effort and never raises: the
    registry exists to make restarts honest, it must never take a live job down.

    Three threads write it: the worker thread (``start`` on claim, ``record_check_run`` right
    after ``create_check_run``, ``finish`` when the job ends), the main thread (``mark_interrupted``
    from ``WorkerPool.stop`` once the drain grace runs out) and the reconcile thread
    (``mark_resumed``). Every read-modify-write holds one lock, so a stamp landing while the run
    id is being recorded cannot make either update disappear: the surviving file always carries
    both. Each write stages through its own unique temp file, so two writers can never share a
    staging path even outside the lock."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()  # serializes every read-modify-write of an entry
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _log("registry_unavailable", error=f"{type(exc).__name__}: {exc}")

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, row_id: int) -> Path:
        return self._root / f"{int(row_id)}.json"

    def _write(self, entry: RunningJob) -> None:
        """Caller holds ``self._lock``. The staging file is unique per write (``mkstemp``): a
        fixed ``{row_id}.json.tmp`` let two writers truncate and rename the same file."""
        path = self._path(entry.row_id)
        tmp: str | None = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._root, prefix=f"{int(entry.row_id)}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), sort_keys=True))
            os.replace(tmp, path)  # atomic: a crash mid-write leaves the old entry, never half a file
        except OSError as exc:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            _log("registry_write_failed", row_id=entry.row_id, error=f"{type(exc).__name__}: {exc}")

    def get(self, row_id: int) -> RunningJob | None:
        return self._read(self._path(row_id))

    @staticmethod
    def _read(path: Path) -> RunningJob | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RunningJob(**data)
        except (OSError, ValueError, TypeError):
            return None  # missing, being written, or from an incompatible version: skip, don't crash

    def entries(self) -> list[RunningJob]:
        try:
            paths = sorted(p for p in self._root.glob("*.json") if p.suffix == ".json")
        except OSError:
            return []
        found = [self._read(p) for p in paths]
        return sorted((e for e in found if e is not None), key=lambda e: e.row_id)

    def start(self, qjob: QueuedJob) -> RunningJob:
        """Record a claimed job. The resume count of an earlier entry for the same row (the
        attempt this one replaces) is carried forward, so the cap counts restarts, not attempts."""
        job = qjob.job
        with self._lock:
            prior = self.get(qjob.row_id)
            entry = RunningJob(
                row_id=qjob.row_id, kind=job.kind, repo=job.repo_full_name, ref=job.ref,
                pr_number=job.pr_number, installation_id=job.installation_id,
                issue_number=job.issue_number, reason=job.reason, delivery_id=qjob.delivery_id,
                attempt=qjob.attempts, started_at=_utc_now(),
                resumes=prior.resumes if prior is not None else 0,
            )
            self._write(entry)
        return entry

    def record_check_run(self, row_id: int, check_run_id: int | None) -> None:
        """The job's Check Run exists now: from here on a restart must complete it."""
        if not check_run_id:
            return
        with self._lock:
            entry = self.get(row_id)
            if entry is not None:
                self._write(replace(entry, check_run_id=int(check_run_id)))

    def finish(self, row_id: int) -> None:
        with self._lock:
            try:
                self._path(row_id).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                _log("registry_clear_failed", row_id=row_id, error=f"{type(exc).__name__}: {exc}")

    def mark_resumed(self, row_id: int) -> None:
        with self._lock:
            entry = self.get(row_id)
            if entry is not None:
                self._write(replace(entry, resumes=entry.resumes + 1, check_run_id=None,
                                    interrupted_by=None, interrupted_at=None))

    def mark_interrupted(self, reason: str) -> int:
        """Stamp every job still registered (the drain grace ran out on it) with why it is about
        to die. Returns how many. Called from ``WorkerPool.stop`` on SIGTERM. Holds the lock for
        the whole pass: a worker recording its run id, or finishing, in the middle of it lands
        strictly before or strictly after, never inside a stale snapshot."""
        stamped = 0
        with self._lock:
            for entry in self.entries():
                self._write(replace(entry, interrupted_by=reason, interrupted_at=_utc_now()))
                stamped += 1
        return stamped


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook-worker", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _token_for(entry: RunningJob, *, app_id: str | None, private_key: str | None, github: Any,
               environ: Any) -> str | None:
    """The same scope the job itself ran with (it created the run, so it may complete it)."""
    if entry.installation_id and app_id and private_key:
        try:
            return github.installation_token(app_id, private_key, entry.installation_id,
                                             repo_full_name=entry.repo, kind=entry.kind)
        except Exception as exc:  # noqa: BLE001 -- minting may be exactly what a fresh boot cannot do yet
            _log("job_resume_token_failed", row_id=entry.row_id, error=type(exc).__name__)
            return None
    return environ.get("GITHUB_TOKEN") or None


def _close(entry: RunningJob, token: str, github: Any, *, conclusion: str, summary: str) -> int | None:
    """Complete the job's own Check Run if it is still open. Returns the id of the run this call
    completed, or None when there was nothing to close (no run recorded, already completed by the
    job's own CLI -- its findings stay -- or a transport error). The residual race between the
    read and the write lives in ``webhook_github.complete_check_run_if_open``."""
    if not entry.check_run_id:
        return None
    try:
        closed = bool(github.complete_check_run_if_open(
            entry.repo, token, entry.check_run_id, conclusion=conclusion, summary=summary,
            name=check_run_name(entry.kind),
        ))
    except Exception as exc:  # noqa: BLE001 -- a transport error must not stop the other entries
        _log("job_resume_close_failed", row_id=entry.row_id, error=type(exc).__name__)
        return None
    return int(entry.check_run_id) if closed else None


def _requeue_current_head(entry: RunningJob, queue: WebhookQueue, head: str) -> bool:
    """Queue this pull request's CURRENT head for the same kind of job. True when a row was added.

    Only for the merge-gating kinds: a `codna review` (or the merge-group check that inherits its
    verdict) missing on the current head blocks every merge, which is the damage worth repairing
    automatically. A `fix` is deliberately NOT re-queued against a head it never examined -- that
    would spend a metered run, and open a pull request, off the back of a restart nobody asked for.

    Best-effort: a queue that rejects the row (a same-head job already waiting, a delivery id
    collision) just means the head is already covered.
    """
    if entry.kind not in ("review", "queue") or not entry.pr_number:
        return False
    job = WebhookJob(entry.kind, entry.repo, ref=head, pr_number=entry.pr_number,
                     installation_id=entry.installation_id,
                     reason=f"restart_requeue_current_head_after_{entry.reason or 'interrupt'}"[:120])
    try:
        # A delivery id derived from the row + head makes this idempotent across repeated boots:
        # two restarts in a row cannot queue the same head twice.
        return bool(queue.enqueue(job, delivery_id=f"restart-{entry.row_id}-{head}", priority=1))
    except Exception as exc:  # noqa: BLE001 -- reconciliation must finish even if this cannot
        _log("job_requeue_failed", row_id=entry.row_id, repo=entry.repo, head=head[:8],
             error=type(exc).__name__)
        return False


def _reconcile_one(entry: RunningJob, queue: WebhookQueue, registry: RunningJobRegistry, *,
                   app_id: str | None, private_key: str | None, github: Any,
                   environ: Any) -> tuple[str, int | None]:
    """One entry's verdict and the id of the Check Run this boot actually completed (None when
    none was): ``mark_resumed`` clears the id from the entry, so it is captured here, not read
    back later."""
    state = queue.row_state(entry.row_id)
    if state is None or state[0] == "done":
        registry.finish(entry.row_id)  # finished right before the kill, or the row is gone
        return "finished", None
    status, attempts = state
    token = _token_for(entry, app_id=app_id, private_key=private_key, github=github, environ=environ)
    if token is None:
        return "skipped", None  # entry kept: the next boot tries again; the retry's own hygiene still applies
    name = check_run_name(entry.kind)
    cause = "a redeploy" if entry.interrupted_by == "sigterm" else "a service restart"
    hint = _RETRIGGER.get(entry.kind, "Trigger it again to run it.")
    if status == "failed":
        # recover_stale found no attempts left (or the last attempt failed on its own): terminal.
        closed = _close(entry, token, github, conclusion="cancelled",
                        summary=f"{name}: interrupted by {cause} with no attempts left. {hint}")
        registry.finish(entry.row_id)
        return "cancelled", closed
    if entry.resumes >= _MAX_RESUMES:
        queue.complete(entry.row_id, status="failed", retry=False,
                       result={"error": "interrupted_after_resume", "attempt": attempts})
        closed = _close(entry, token, github, conclusion="cancelled",
                        summary=f"{name}: interrupted by {cause} again after one automatic resume; "
                                f"not re-run automatically. {hint}")
        registry.finish(entry.row_id)
        return "cancelled", closed
    if entry.pr_number and entry.ref:
        head = github.pull_request_head_sha(entry.repo, token, entry.pr_number)
        if head and head != entry.ref:
            queue.complete(entry.row_id, status="done", retry=False,
                           result={"summary": f"superseded: PR head moved to {head[:8]} across a restart"})
            # "which gets its own run" was an assumption, and on 2026-09-19 it was wrong: the push
            # that moved the head was delivered while the machine was being replaced, so no job for
            # the CURRENT head existed anywhere and nothing re-drove it. Reviews for heads pushed
            # 02:00-02:06 only appeared after 02:17, one at a time, because each was waiting on a
            # delivery that had already been consumed. Queue the current head HERE instead of
            # hoping for a redelivery GitHub will never send, ahead of rows queued while it waited
            # (priority=1): that head is already blocking a merge on a required check.
            requeued = _requeue_current_head(entry, queue, head)
            closed = _close(entry, token, github, conclusion="cancelled",
                            summary=f"{name}: interrupted by {cause}; the pull request head has since "
                                    f"moved to {head[:8]}"
                                    + (f", which has been queued for its own {name} run."
                                       if requeued else ", which gets its own run.")
                                    + " This commit was not re-run.")
            registry.finish(entry.row_id)
            return "superseded", closed
    closed = _close(entry, token, github, conclusion="cancelled",
                    summary=f"{name}: interrupted by {cause}; re-queued automatically as attempt {attempts + 1}. "
                            f"A new `{name}` run replaces this one. If none appears: {hint}")
    registry.mark_resumed(entry.row_id)  # the row is already 'retry' (recover_stale) -- it runs again
    return "resumed", closed


def reconcile_interrupted(queue: WebhookQueue, registry: RunningJobRegistry, *,
                          app_id: str | None, private_key: str | None,
                          github: Any = None, environ: Any = None) -> dict[str, int]:
    """Resolve every job the previous process left registered. Call AFTER ``queue.recover_stale``
    and BEFORE any worker claims: the decisions here (run again / do not) are made on rows that
    recover_stale has just put back in ``retry``. Returns counts per verdict. Never raises."""
    github = github or webhook_github
    environ = os.environ if environ is None else environ
    outcome = {"resumed": 0, "superseded": 0, "cancelled": 0, "finished": 0, "skipped": 0}
    for entry in registry.entries():
        closed: int | None = None
        try:
            verdict, closed = _reconcile_one(entry, queue, registry, app_id=app_id,
                                             private_key=private_key, github=github, environ=environ)
        except Exception as exc:  # noqa: BLE001 -- one bad entry must not strand the others
            _log("job_resume_error", row_id=entry.row_id, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            verdict = "skipped"
        outcome[verdict] += 1
        # check_run_id: the run the job had opened. closed_check_run_id: the one THIS boot completed
        # (None: none recorded, the job's own CLI had already completed it, or the close failed).
        _log("job_resume", row_id=entry.row_id, kind=entry.kind, repo=entry.repo, ref=entry.ref,
             check_run_id=entry.check_run_id, closed_check_run_id=closed,
             interrupted_by=entry.interrupted_by, resumes=entry.resumes, verdict=verdict)
    return outcome


def reconcile_in_background(queue: WebhookQueue, registry: RunningJobRegistry, *,
                            app_id: str | None, private_key: str | None,
                            done: threading.Event) -> threading.Thread:
    """Run :func:`reconcile_interrupted` on its own thread and set ``done`` when it has finished
    (or failed). Off the startup path because each GitHub call may take up to its 30 s timeout
    and ``/health`` must answer at once after a deploy; the pool holds every claim on ``done``."""

    def _run() -> None:
        try:
            reconcile_interrupted(queue, registry, app_id=app_id, private_key=private_key)
        except Exception as exc:  # noqa: BLE001 -- reconciliation must never keep the pool from starting
            _log("job_resume_error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            done.set()

    thread = threading.Thread(target=_run, name="codna-webhook-reconcile", daemon=True)
    thread.start()
    return thread
