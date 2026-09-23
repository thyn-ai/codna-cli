"""Live control plane for the jobs this webhook process is running right now.

The durable queue (``webhook_queue``) knows what is *waiting*; this module knows what is
*running*, and is the only thing that can stop it. Two callers need that:

  * **supersede on a new head** — a force-push or a new commit makes an in-flight ``codna review``
    of the old head worthless (its findings point at lines that no longer exist) and an in-flight
    ``codna fix`` of the old head actively harmful (it would open a PR against a stale base). The
    ingress cancels them the moment the new head's job is enqueued, rather than letting them run to
    completion and post a verdict nobody asked for. Observed 2026-09-19 on thyn-ai/mojo-kernels#32:
    reviews started 01:58:30 (ba252b20) and 02:01:57 (04e3c8a5) both ran to completion while the
    head had already moved to 21e3610.
  * **the job deadline** — ``webhook_procs._run_job_process`` bounds only the CLI *child*. A job's
    thread-hold is that bound PLUS every pre/post phase (installation token, the account bridge, CI
    triage's job-log fetches, the Check Run and comment posts): each of those has its own HTTP
    timeout, but nothing bounds their sum, so a thread could be held well past
    ``CODNA_WEBHOOK_JOB_TIMEOUT_S`` with a required check sitting ``in_progress``. The pool's
    watchdog cancels through here instead.

Cancellation is **cooperative and honest about its reach**. It does three things, in this order:
kills the job's process group (which is what unblocks the worker thread in every observed case,
since the thread is inside ``communicate()``), tears down the detached runtime the job started, and
completes the job's Check Run as ``neutral`` — never ``failure``, because a superseded or timed-out
job says nothing about the pull request's code. The Check Run is completed by the *canceller*, not
by the worker thread, so a thread wedged somewhere the kill cannot reach still stops blocking
merges. What it cannot do is force a Python thread out of a blocking syscall; that residual is
documented on ``JobHandle.cancel`` and lands as a row ``recover_stale()`` resolves on the next boot.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# Kinds that gate a merge: the org rulesets require `codna review`, and a merge group's inherited
# verdict is that same check reporting on the queue's temporary commit.
PRIORITY_KINDS = frozenset({"review", "queue"})
# Kinds a moving PR head invalidates. A merge-group job is deliberately excluded: its `ref` is the
# queue's own group commit, which has nothing to do with the PR head and must never be superseded
# by one (GitHub would kick the PR out of the queue for a missing required check).
SUPERSEDABLE_KINDS = frozenset({"review", "fix"})


def _log(event: str, **fields: Any) -> None:
    """One stderr JSON line, same shape as the worker's, so one grep covers both."""
    payload = {"service": "codna-webhook-control", "event": event, **fields}
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


@dataclass
class JobHandle:
    """One running job, and the means to stop it.

    Written by the worker thread (``attach_*`` as each resource comes into existence) and read by
    the ingress thread and the watchdog, so every mutation holds ``_lock``.
    """

    row_id: int
    kind: str
    repo: str
    pr_number: int | None
    ref: str | None
    deadline_s: float
    started_monotonic: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _pgids: set[int] = field(default_factory=set, repr=False)
    _runtime_tmp: str | None = field(default=None, repr=False)
    # The job's own scoped installation token and Check Run, recorded so a CANCELLER can complete
    # that run without minting anything. Never included in snapshot() -- /healthz is public.
    _token: str | None = field(default=None, repr=False)
    _check_run_id: int | None = field(default=None, repr=False)
    _finalized: bool = field(default=False, repr=False)
    # Whether this job's Check Run is CONFIRMED completed. The worker thread reads it as it
    # unwinds: if no canceller managed to close the run, the worker closes it itself rather than
    # trusting that someone else did.
    _check_closed: bool = field(default=False, repr=False)
    cancel_reason: str | None = None
    cancel_detail: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.started_monotonic

    @property
    def check_run_id(self) -> int | None:
        with self._lock:
            return self._check_run_id

    @property
    def check_closed(self) -> bool:
        with self._lock:
            return self._check_closed

    def mark_check_closed(self) -> None:
        with self._lock:
            self._check_closed = True

    @property
    def token(self) -> str | None:
        """The job's own scoped installation token, for a canceller closing its Check Run. Never
        rendered anywhere public -- ``snapshot()`` is what /healthz sees."""
        with self._lock:
            return self._token

    def attach_process(self, proc: Any) -> None:
        """Record a job subprocess started with ``start_new_session=True`` (pid == pgid).

        A cancellation that lands BETWEEN Popen and this call would otherwise be lost, so the
        process is killed immediately if the job is already cancelled.
        """
        pid = getattr(proc, "pid", None)
        if not isinstance(pid, int) or pid <= 1:
            return
        with self._lock:
            self._pgids.add(pid)
            already = self._cancelled.is_set()
        if already:
            _kill_group(pid)

    def attach_runtime(self, tmp: str | None) -> None:
        with self._lock:
            self._runtime_tmp = tmp

    def attach_check_run(self, token: str | None, check_run_id: int | None) -> None:
        with self._lock:
            self._token = token or self._token
            self._check_run_id = check_run_id if check_run_id is not None else self._check_run_id

    def retarget(self, ref: str) -> None:
        """Record the head this job actually works on, once the worker has resolved it.

        A review job is registered with the head its EVENT carried, but it reviews the pull
        request's head as of its start (``review._materialize_pr`` fetches ``pull/N/head``), and
        the worker anchors its Check Run there (thyn-ai/codna#569). Superseding compares against
        this ref, so it has to be the reviewed head: otherwise the late-delivered event for that
        very head would cancel the job already reviewing it, and ``/healthz`` would name a commit
        the job is not looking at.
        """
        with self._lock:
            self.ref = ref

    def finalize(self) -> bool:
        """True for exactly one caller: whoever gets it owns writing this job's terminal queue row.

        The canceller and the worker thread both race to finish a cancelled job, and a second
        ``complete()`` would overwrite the cancellation's own result with a late 'done'.
        """
        with self._lock:
            if self._finalized:
                return False
            self._finalized = True
            return True

    def cancel(self, *, reason: str, detail: str | None = None) -> None:
        """Signal the job and kill its process group. FAST: no I/O, no waiting. Never raises.

        Deliberately does only the part that must happen immediately and cannot block, because one
        caller is the HTTP ingress thread, which has to acknowledge GitHub within 10 seconds --
        the settling afterwards (runtime teardown, which waits on processes; the Check Run update,
        a 30 s-timeout API call) belongs to :func:`settle_cancelled`, off that path.

        Reach: the process group dies, so a worker thread blocked in ``communicate()`` returns
        promptly. A thread blocked in something else (a socket read inside a GitHub call) is NOT
        interrupted -- it finishes that call and then sees ``cancelled``. Its queue row is written
        by the canceller either way, so nothing depends on the thread noticing.
        """
        with self._lock:
            first = not self._cancelled.is_set()
            if first:
                self.cancel_reason = reason
                self.cancel_detail = detail
            pgids = set(self._pgids)
        self._cancelled.set()
        for pgid in pgids:
            _kill_group(pgid)      # signal only: killpg never blocks
        if first:
            _log("job_cancelled", row_id=self.row_id, kind=self.kind, repo=self.repo,
                 pr_number=self.pr_number, ref=self.ref, reason=reason, detail=detail,
                 age_s=round(self.age_s, 1))

    def teardown_runtime(self) -> None:
        """Stop the detached sidecar + engine this job started. Waits on processes -- never call it
        from the ingress thread. Belt and braces: ``run_codna_job`` tears the same runtime down in
        its ``finally`` as soon as the killed subprocess returns."""
        with self._lock:
            tmp = self._runtime_tmp
        if not tmp:
            return
        try:
            from .webhook_procs import _teardown_job_runtime

            _teardown_job_runtime(tmp)
        except Exception:  # noqa: BLE001 -- teardown is hygiene; it must never break a cancel
            pass

    def snapshot(self) -> dict[str, Any]:
        """Public-safe view: no token, no check id, no job content -- see webhook._health_payload."""
        return {
            "kind": self.kind,
            "repo": self.repo,
            "pr": self.pr_number,
            "head": self.ref[:8] if self.ref else None,
            "age_s": round(self.age_s, 1),
            "cancelled": self.cancelled,
        }


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)  # start_new_session=True in _run_job_process: pid == pgid
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


class JobControl:
    """Registry of the jobs running in THIS process. One instance per process (``get_control``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[int, JobHandle] = {}

    def register(self, *, row_id: int, kind: str, repo: str, pr_number: int | None,
                 ref: str | None, deadline_s: float) -> JobHandle:
        handle = JobHandle(row_id=row_id, kind=kind, repo=repo, pr_number=pr_number, ref=ref,
                           deadline_s=deadline_s)
        with self._lock:
            self._jobs[row_id] = handle
        return handle

    def release(self, row_id: int) -> None:
        with self._lock:
            self._jobs.pop(row_id, None)

    def get(self, row_id: int) -> JobHandle | None:
        with self._lock:
            return self._jobs.get(row_id)

    def running(self) -> list[JobHandle]:
        with self._lock:
            return list(self._jobs.values())

    def snapshot(self) -> list[dict[str, Any]]:
        """Oldest first: a saturation report is read top-down, and the oldest job is the story."""
        return [h.snapshot() for h in sorted(self.running(), key=lambda h: h.started_monotonic)]

    def superseded_by(self, *, repo: str, pr_number: int | None, new_ref: str | None) -> list[JobHandle]:
        """Running review/fix jobs on the same pull request whose head is no longer the current one.

        Cross-kind on purpose: a new head invalidates an in-flight ``codna fix`` exactly as it
        invalidates an in-flight ``codna review`` -- a fix computed against a stale head must not
        open a pull request. Jobs with no ``ref`` (a `@codna review` comment resolves the head when
        it runs) are left alone, and so are merge-group jobs.
        """
        if not repo or pr_number is None or not new_ref:
            return []
        return [
            h for h in self.running()
            if h.repo == repo and h.pr_number == pr_number and h.kind in SUPERSEDABLE_KINDS
            and h.ref and h.ref != new_ref and not h.cancelled
        ]

    def overdue(self) -> list[JobHandle]:
        return [h for h in self.running() if not h.cancelled and h.age_s > h.deadline_s]


_CONTROL: JobControl | None = None
_CONTROL_LOCK = threading.Lock()


def get_control() -> JobControl:
    """The process-wide registry. Shared by the ingress thread, the worker threads and the watchdog."""
    global _CONTROL
    if _CONTROL is None:
        with _CONTROL_LOCK:
            if _CONTROL is None:
                _CONTROL = JobControl()
    return _CONTROL


def cancel_jobs(handles: Iterable[JobHandle], *, reason: str, summary: str,
                github: Any | None = None, queue: Any | None = None) -> int:
    """Cancel each handle and settle it inline. Returns how many were cancelled.

    For callers that are already off the request path (the pool's watchdog). The ingress uses
    :func:`supersede_running`, which signals inline and settles on its own thread.
    """
    handles = list(handles)
    for handle in handles:
        handle.cancel(reason=reason, detail=summary)
    settle_cancelled(handles, reason=reason, summary=summary, github=github, queue=queue)
    return len(handles)


def settle_cancelled(handles: Iterable[JobHandle], *, reason: str, summary: str,
                     github: Any | None = None, queue: Any | None = None) -> None:
    """The slow half of a cancellation: tear down the runtime, close the Check Run, write the row.

    ``neutral``, never ``failure``: a superseded or timed-out job made no finding about the code.
    The Check Run is closed here rather than on the worker thread so that a thread the kill cannot
    reach still stops blocking merges, and the queue row is written here for the same reason --
    ``JobHandle.finalize`` makes sure the worker thread does not write a second one.

    **The row is written only once the run is confirmed closed**, and that ordering is the whole
    safety argument. A terminal row is also what tells ``recover_stale``/``webhook_resume`` there
    is nothing to resolve on the next boot, so writing one after a FAILED close (a GitHub blip, or
    this daemon thread dying with the process) would strand the run ``in_progress`` with no path
    left to close it -- on a repo where `codna review` is required, permanently unmergeable. When
    the close does not confirm, this leaves the row alone: the worker thread closes the run as it
    unwinds (``process_job``), and if even that does not happen the row stays ``running`` for the
    next boot's reconciler, which has the run's id in the registry.

    Blocks on process waits and a GitHub call, so it never runs on the ingress thread.
    """
    for handle in handles:
        handle.teardown_runtime()
        closed = close_check_run(handle, summary=summary, github=github)
        if not closed:
            continue  # deliberately no terminal row -- see the docstring
        if queue is not None and handle.finalize():
            try:
                queue.complete(handle.row_id, status="done", retry=False,
                               result={"summary": summary, "cancelled": reason})
            except Exception as exc:  # noqa: BLE001 -- a queue hiccup must not abort the cancel
                _log("cancel_row_complete_failed", row_id=handle.row_id, error=type(exc).__name__)


def close_check_run(handle: JobHandle, *, summary: str, github: Any | None = None) -> bool:
    """Complete a cancelled job's Check Run as ``neutral``. True when it is confirmed closed.

    Prefers ``complete_check_run_if_open``, which reads the run first and leaves it alone once it
    is ``completed``: a review whose CLI had already posted its findings must not have them
    replaced by "superseded by <sha>". That helper cannot distinguish "already completed" from "the
    call failed", and both come back False, so False here means "not confirmed" rather than
    "failed" -- callers treat it as a reason to leave the job's row for another path to resolve,
    never as a reason to give up on the run.
    """
    check_run_id = handle.check_run_id
    token = handle.token
    if not check_run_id or not token:
        handle.mark_check_closed()
        return True  # cancelled before it opened a run: nothing to close, nothing to strand
    client = github
    if client is None:
        from . import webhook_github  # local import keeps this module import-light

        client = webhook_github
    from .webhook import check_run_name

    name = check_run_name(handle.kind)
    closer = getattr(client, "complete_check_run_if_open", None)
    try:
        if callable(closer):
            closed = bool(closer(handle.repo, token, check_run_id, conclusion="neutral",
                                 summary=summary, name=name))
        else:  # a client without the guarded helper (an older embedder, a test double)
            client.update_check_run(handle.repo, token, check_run_id, conclusion="neutral",
                                    summary=summary, name=name)
            closed = True
    except Exception as exc:  # noqa: BLE001 -- never let GitHub take a cancellation down
        _log("cancel_check_run_failed", row_id=handle.row_id, error=type(exc).__name__)
        return False
    if closed:
        handle.mark_check_closed()
    return closed


def supersede_running(job: Any, *, github: Any | None = None, queue: Any | None = None,
                      background: bool = True) -> int:
    """Cancel every running job on this pull request that a newly-arrived head has superseded.

    Called from the HTTP ingress, which must acknowledge GitHub inside its 10 s delivery timeout
    or the delivery is retried. So the stale job is STOPPED inline -- ``cancel`` is a killpg and an
    event, microseconds -- while the settling (runtime teardown, which waits on processes, and the
    Check Run update, a 30 s-timeout API call) is handed to a short-lived daemon thread. Pass
    ``background=False`` off the request path, or in a test that wants the settling to have
    happened when this returns.
    """
    handles = get_control().superseded_by(repo=job.repo_full_name, pr_number=job.pr_number,
                                          new_ref=job.ref)
    if not handles:
        return 0
    summary = f"superseded by {(job.ref or '')[:8]}"
    for handle in handles:
        handle.cancel(reason="superseded", detail=summary)
    if not background:
        settle_cancelled(handles, reason="superseded", summary=summary, github=github, queue=queue)
        return len(handles)

    def _settle() -> None:
        settle_cancelled(handles, reason="superseded", summary=summary, github=github, queue=queue)

    threading.Thread(target=_settle, name="codna-webhook-supersede", daemon=True).start()
    return len(handles)


# --- how big the pool should be, and what the service says about its own saturation ------------
# What one job's isolated runtime actually costs in memory. MEASURED, not guessed (2026-09-19,
# this tree): one agent-core sidecar -- the Bun process the CLI starts per job under the job's own
# CODNA_RUNTIME_ROOT -- peaks at 731 MB resident, and the codna CLI process that drives it at
# 43 MB, plus the git clone/checkout and test subprocesses it spawns. 900 MB is that sum with a
# little room. The same number sizes the machine in infra/fly/codna-webhook.fly.toml; keep them
# together. A review's wall clock is dominated by the model turn, not by local CPU, so memory --
# not cores -- is what bounds this pool.
_JOB_MEMORY_MB = 900
# Never derive more than this, whatever the machine reports: past the sidecar's own concurrency the
# runtime sheds work (admission_control.sidecar_ceiling), so more claimers would only queue.
_MAX_DERIVED_CONCURRENCY = 6
# Headroom for the ingress, the SQLite queue, page cache and the kernel. A machine that is exactly
# full is a machine that OOM-kills a job mid-review.
_RESERVED_MEMORY_MB = 700


# cgroup v2, then v1. A container's OWN bound, which is what the pool has to fit inside.
_CGROUP_MEMORY_LIMIT_PATHS = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")


def _machine_memory_mb() -> int | None:
    """Total RAM visible to this process, or None where it cannot be read.

    Reads the cgroup limit first, because a container on a large host sees the HOST's RAM through
    sysconf and would derive a pool the machine cannot hold. Unknown never guesses: the caller
    falls back to a fixed default.
    """
    for path in _CGROUP_MEMORY_LIMIT_PATHS:
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            break  # unconstrained cgroup: fall through to the host's own total
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < (1 << 62):  # v1 writes a sentinel near 2^63 to mean "no limit"
            return value // (1024 * 1024)
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        return None


def default_concurrency(environ: Mapping[str, str] | None = None) -> int:
    """How many worker threads to run: derived from the machine's memory, overridable by env.

    A hardcoded default silently mis-sizes the pool every time the machine changes -- it is why a
    4x-larger machine would still have run two threads. ``CODNA_WEBHOOK_CONCURRENCY`` still wins
    when set, so an operator can pin it; otherwise the pool tracks the machine it was given.
    """
    source = os.environ if environ is None else environ
    override = str(source.get("CODNA_WEBHOOK_CONCURRENCY", "")).strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass  # a typo must not take the webhook down: fall through to the derivation
    total_mb = _machine_memory_mb()
    if not total_mb:
        return 2  # unknown machine: the long-standing default, unchanged
    usable = total_mb - _RESERVED_MEMORY_MB
    return max(1, min(_MAX_DERIVED_CONCURRENCY, usable // _JOB_MEMORY_MB))


def health_payload(server: Any, *, ok: bool) -> dict[str, Any]:
    """``/healthz``: liveness PLUS enough scheduling state to see saturation without a token.

    Tonight's outage was invisible from outside: seven pull-request heads across four repositories
    waited 15+ minutes for `codna review` -- a required check, so every one of them was blocking a
    merge -- and the only endpoint that said anything about the queue was `/debug/queue`, which
    needs the webhook secret. `/ready` reported depth by STATUS, which cannot distinguish a backlog
    of reviews (all merge-blocking) from a backlog of fixes (blocking nothing).

    Public-safe by construction: counts keyed by kind, and for running jobs only the repository,
    pull-request number, the first 8 characters of the head SHA and an age. No tokens, no summaries,
    no findings, no delivery ids, no installation ids, no job content of any kind.
    """
    payload: dict[str, Any] = {"service": "codna-webhook", "ok": ok}
    queue = getattr(server, "queue", None)
    if queue is not None:
        try:
            payload["queue_depth"] = queue.depth_by_kind()
        except Exception:  # noqa: BLE001 -- liveness must never fail because a diagnostic did
            payload["queue_depth"] = None
    pool = getattr(server, "worker_pool", None)
    if pool is not None:
        try:
            diag = pool.diagnostics()
            payload["workers"] = {k: diag.get(k) for k in (
                "configured_threads", "busy_threads", "priority_thread_reserved", "wedged")}
        except Exception:  # noqa: BLE001
            payload["workers"] = None
    try:
        payload["running"] = get_control().snapshot()
    except Exception:  # noqa: BLE001
        payload["running"] = None
    return payload
