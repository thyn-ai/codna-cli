"""The worker pool: which job runs next, on which thread, and for how long.

Split out of ``webhook_worker`` (the 1000-line modularity ceiling), which keeps the *content* of a
job -- credentials, Check Runs, the CLI invocation -- and hands this module the *scheduling* of it:

  * **class before arrival.** ``review`` and the merge-group ``queue`` job gate merges (the org
    rulesets make `codna review` a required check); ``fix`` / ``secure`` do not. The queue orders by
    that class, and the pool additionally keeps its LAST free thread for the gating classes while
    other threads run fixes -- on 2026-09-19/20 one 30-minute `codna fix` plus one serial reviewer
    was enough to leave seven heads across four repositories without a required check for 15+
    minutes.
  * **a deadline for the whole job.** ``webhook_procs._run_job_process`` bounds only the CLI child;
    the watchdog here bounds everything the thread does, so no job can hold a thread (and a
    required check) past ``CODNA_WEBHOOK_JOB_TIMEOUT_S`` whatever it is blocked on.

Names the tests and callers patch (``_JOB_TIMEOUT_S``, ``process_job``) are
reached THROUGH :mod:`codna.webhook_worker` rather than imported as values, so patching that module
still governs this one -- and so this module can import it without a cycle (``webhook_worker``
exposes ``WorkerPool`` lazily, via ``__getattr__``).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from . import webhook_control, webhook_github, webhook_worker
from .admission_control import duration_to_learn, get_admitter
from .webhook import WebhookError, check_run_name
from .webhook_control import PRIORITY_KINDS, JobHandle, get_control
from .webhook_queue import _MAX_ATTEMPTS, QueuedJob, WebhookQueue
from .webhook_resume import RunningJobRegistry, reconcile_in_background
from .webhook_worker import JobResult

def _accepts_kwarg(func: Any, name: str) -> bool:
    """Whether ``func`` takes this keyword (or **kwargs). False for anything unintrospectable."""
    if not callable(func):
        return False
    try:
        import inspect

        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def _left_for_the_next_boot(handle: JobHandle | None) -> bool:
    """True when a cancelled job's Check Run is NOT confirmed closed, by any path.

    Both closers (the canceller's settle, and ``process_job`` as the worker unwinds) record their
    outcome on the handle, so this one predicate decides what the job leaves behind: with the run
    still open, the queue row and the registry entry must both survive for the next boot to
    resolve -- see ``webhook_control.settle_cancelled``.
    """
    return handle is not None and handle.cancelled and not handle.check_closed


class WorkerPool:
    def __init__(self, queue: WebhookQueue, *, concurrency: int = 2, poll_interval: float = 1.0,
                 app_id: str | None = None, private_key: str | None = None,
                 drain_timeout: float | None = None,
                 process: Callable[..., JobResult] | None = None,
                 registry: RunningJobRegistry | None = None) -> None:
        self._queue = queue
        self._registry = registry  # None: no restart bookkeeping (bare pools in tests / embedders)
        self._reconciled = threading.Event()  # start(): claims wait until interrupted jobs are resolved
        self._concurrency = max(1, concurrency)
        self._poll = poll_interval
        self._app_id = app_id
        self._private_key = private_key
        self._drain_timeout = drain_timeout if drain_timeout is not None else float(
            os.environ.get("CODNA_WEBHOOK_DRAIN_S", "25"))
        self._process = process or self._default_process
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._busy_lock = threading.Lock()
        self._busy_since: dict[str, float] = {}  # thread name -> monotonic start of its current job
        self._busy_kinds: dict[str, str] = {}    # thread name -> kind of its current job
        self._control = get_control()
        self._watchdog: threading.Thread | None = None
        # Queues that predate the reserved-thread rule (an embedder's own, a test double) keep the
        # plain claim(); asked by signature rather than by catching TypeError, which would also
        # swallow a real one raised INSIDE claim().
        self._claim_takes_reserve = _accepts_kwarg(getattr(queue, "claim", None), "reserve_priority")
        # The whole job's wall clock, not just its CLI child's. _run_job_process bounds the
        # subprocess; the phases around it (installation token, the account bridge, CI triage's
        # job-log fetches, the Check Run and comment posts) each have their own HTTP timeout but
        # nothing bounds their SUM -- so the thread-hold this pool actually promises has to be
        # enforced here. The slack lets the subprocess path report its own, more specific timeout
        # first whenever the child is where the time went.
        self._job_deadline_s = float(webhook_worker._JOB_TIMEOUT_S + webhook_worker._KILL_DRAIN_S + 30)

    def _default_process(self, qjob: QueuedJob) -> JobResult:
        record = (lambda cid: self._registry.record_check_run(qjob.row_id, cid)) if self._registry else None
        return webhook_worker.process_job(qjob, app_id=self._app_id, private_key=self._private_key, on_check_run=record,
                           control=self._control.get(qjob.row_id))

    def _loop(self) -> None:
        # MC-predictive admission (admission_control): thread count is only the ceiling on CLAIMERS;
        # how many fixes actually run at once is decided per-fix against the observed duration
        # distribution, so a fix that would breach the SLA under current contention waits instead.
        admitter = get_admitter()
        while not self._stop.is_set() and not self._reconciled.wait(self._poll):
            pass  # start() is still deciding which requeued rows run again: claim nothing yet
        while not self._stop.is_set():
            # ADMIT BEFORE CLAIMING. A claimed row is status='running', which claim() excludes
            # (webhook_queue.py:122, inside BEGIN IMMEDIATE) — so parking a claimed job while waiting
            # for capacity cannot be double-processed. The real damage is subtler: claim() already
            # incremented `attempts`, and recover_stale() requeues anything left 'running' at startup,
            # so a restart during a wait burns a retry for work that never began — enough restarts and
            # the job dies 'orphaned_after_max_attempts' having never run once. Holding capacity
            # instead of a queue row also keeps counts()/recent() honest about what is really running.
            verdict = admitter.try_admit()   # atomic check-and-reserve
            if not verdict.admit:
                self._stop.wait(self._poll)
                continue
            # From here the slot is RESERVED and must be released on every path (finally below).
            # Deliberately try_admit() + poll rather than admitter.gate(): gate() blocks up to
            # max_wait_s=1800 without observing self._stop, which would ignore shutdown and defeat
            # CODNA_WEBHOOK_DRAIN_S draining. This loop re-checks _stop every poll interval.
            slot_started = time.monotonic()
            qjob: QueuedJob | None = None      # bound BEFORE the try: claim() itself can raise,
            #                                    and the finally below must still release the slot
            try:
                try:
                    qjob = self._claim()
                except Exception as exc:  # noqa: BLE001 — a transient queue error must not kill the thread
                    webhook_worker._log_worker_loop_error(exc)
                    self._stop.wait(self._poll)
                    continue
                if qjob is None:
                    self._stop.wait(self._poll)
                    continue
                try:
                    self._run_claimed(qjob)
                except Exception as exc:  # noqa: BLE001 — complete() itself can raise inside a handler
                    webhook_worker._log_worker_loop_error(exc, qjob)
            finally:
                # Release the slot on every path; LEARN only from a fix. finish() records positive
                # durations only, so 0.0 frees the slot without touching the window: an idle poll has
                # no duration, and a review's wall-clock belongs to its own series (review_budget), not
                # to the fix distribution this admitter forecasts against (duration_to_learn).
                admitter.finish(duration_to_learn(qjob.job.kind, time.monotonic() - slot_started) if qjob is not None else 0.0)

    def _claim(self) -> QueuedJob | None:
        """One claim, with the reserved-thread rule applied when this pool's queue supports it."""
        if not self._claim_takes_reserve:
            return self._queue.claim()  # an embedder's own queue / a test double: plain claim
        return self._queue.claim(reserve_priority=self._reserve_priority_thread())

    def _reserve_priority_thread(self) -> bool:
        """True when this claim must prefer merge-gating work: taking a fix now would leave the pool
        with no thread for a review.

        Reviews are a REQUIRED check on the public rulesets, so every queued review is blocking a
        merge, while no fix ever is. Tonight's saturation is the case: one `codna fix` held a thread
        for ~30 minutes while the other worked through reviews serially, and seven heads across four
        repositories waited 15+ minutes for a check that gates every merge. With the last free
        thread reserved, a fix waits for a review instead of the other way round -- and when no
        review is waiting the queue hands that thread a fix anyway, so nothing idles (claim()).
        """
        with self._busy_lock:
            non_priority = sum(1 for kind in self._busy_kinds.values() if kind not in PRIORITY_KINDS)
        return non_priority >= max(1, self._concurrency) - 1

    def _run_claimed(self, qjob: QueuedJob) -> None:
        me = threading.current_thread().name
        with self._busy_lock:
            self._busy_since[me] = time.monotonic()
            self._busy_kinds[me] = qjob.job.kind
        handle = self._control.register(
            row_id=qjob.row_id, kind=qjob.job.kind, repo=qjob.job.repo_full_name,
            pr_number=qjob.job.pr_number, ref=qjob.job.ref, deadline_s=self._job_deadline_s,
        )
        if self._registry:
            self._registry.start(qjob)  # on the volume: a restart from here on finds this job
        try:
            self._run_claimed_inner(qjob, handle)
        finally:
            with self._busy_lock:
                self._busy_since.pop(me, None)
                self._busy_kinds.pop(me, None)
            self._control.release(qjob.row_id)
            # Read at the point of use, not earlier: a settle thread finishing in between would
            # make an older snapshot keep an entry whose row is already terminal. The window is
            # now a single call, and what remains self-heals -- the next boot's reconciler clears
            # any entry whose row it finds already 'done'.
            if self._registry and not _left_for_the_next_boot(handle):
                # An unresolved cancellation KEEPS its registry entry: that entry (with the Check
                # Run id in it) is what lets the next boot close the run and queue the current
                # head. Clearing it here would throw away the only remaining way to resolve it.
                self._registry.finish(qjob.row_id)

    def _run_claimed_inner(self, qjob: QueuedJob, handle: JobHandle | None = None) -> None:
        """Process one claimed job and mark its queue row terminal. A worker must never die on one
        bad job, so every outcome ends in complete() -- unless a canceller already wrote this row's
        terminal state (JobHandle.finalize decides that exactly once, for exactly one writer)."""
        try:
            webhook_worker._log_worker_phase(qjob, "claimed")
            result = self._process(qjob)
            webhook_worker._log_worker_phase(qjob, "cancelled" if result.cancelled else ("done" if result.ok else "failed"))
            if not result.ok:
                webhook_worker._log_worker_failure(qjob, code="job_failed", message=result.summary)
            if result.cancelled and _left_for_the_next_boot(handle):
                # The invariant from webhook_control.settle_cancelled, which this path has to honour
                # too: NO terminal row until the Check Run is confirmed closed. A terminal row is
                # what tells recover_stale/webhook_resume there is nothing left to resolve, so
                # writing one now would strand the run `in_progress` with no path back -- and where
                # `codna review` is required, that pull request never merges. Left 'running' (with
                # its registry entry), the next boot closes the run and queues the current head.
                #
                # Consume the finalize token FIRST, before anything that could raise: the handlers
                # below would otherwise still be able to write that forbidden row on their way out
                # (a broken stderr pipe in the phase log is enough). Nothing else needs the token --
                # a settle that later closes the run skips its own write when finalize() is gone,
                # which leaves the row exactly where this path wants it.
                self._may_finalize(handle)
                webhook_worker._log_worker_phase(qjob, "cancel_close_unconfirmed")
                return
            if not self._may_finalize(handle):
                return  # the canceller owns this row; a second complete() would overwrite it
            self._queue.complete(qjob.row_id, status="done" if result.ok else "failed",
                                 result={"summary": result.summary, "attempt": qjob.attempts},
                                 retry=result.retryable, retry_after_s=result.retry_after_s)
            # A review whose pull request head moved while it ran: the head it did NOT review is
            # queued now, after this row is terminal, so nothing about it can hold the row open.
            webhook_worker.requeue_moved_head(qjob, result, self._queue)
        except WebhookError as exc:
            webhook_worker._log_worker_failure(qjob, code=exc.code, message=str(exc))
            if self._may_finalize(handle):
                self._queue.complete(qjob.row_id, status="failed", result={"error": exc.code, "message": str(exc)})
            self._notify_terminal_comment_fix_failure(qjob, exc.code)
            self._close_unreported_check_run(qjob, exc.code)
        except Exception as exc:  # noqa: BLE001 — a worker must never die on one bad job
            webhook_worker._log_worker_failure(qjob, code="job_crashed", message=str(exc))
            if self._may_finalize(handle):
                self._queue.complete(qjob.row_id, status="failed",
                                     result={"error": "job_crashed", "message": str(exc)[:500]})
            self._notify_terminal_comment_fix_failure(qjob, "job_crashed")
            self._close_unreported_check_run(qjob, "job_crashed")

    @staticmethod
    def _may_finalize(handle: JobHandle | None) -> bool:
        """Whether THIS thread should write the job's terminal row. Always true without a handle."""
        return True if handle is None else handle.finalize()

    def _close_unreported_check_run(self, qjob: QueuedJob, code: str) -> None:
        """A job that RAISED out of process_job may never have touched the Check Run the ingress
        opened for its row (token minting is the usual point). When the queue has now ended the row
        as ``failed`` -- deterministic, no further attempt -- that run is completed here so a
        required check does not sit `queued` forever; a retry keeps the run for the next attempt,
        and a `dead` row is told by the reaper's dead-letter notice. Best-effort (the failure may be
        exactly that no token mints); the reaper's retired-run sweep is the backstop. Never raises.
        """
        if not qjob.check_run_id:
            return
        try:
            state = self._queue.row_state(qjob.row_id)
        except Exception:  # noqa: BLE001
            return
        if state is None or state[0] != "failed":
            return
        job = qjob.job
        token: str | None = None
        try:
            if job.installation_id and self._app_id and self._private_key:
                token = webhook_github.installation_token(self._app_id, self._private_key, job.installation_id,
                                                          repo_full_name=job.repo_full_name, kind=job.kind)
        except Exception:  # noqa: BLE001 — minting may be the very failure being reported
            token = None
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            return
        name = check_run_name(job.kind)
        try:
            webhook_github.complete_check_run_if_open(
                job.repo_full_name, token, qjob.check_run_id, conclusion="failure", name=name,
                summary=f"{name} failed before it could report ({code}) after {qjob.attempts} attempt(s). "
                        f"The worker logged the details; trigger it again to re-run it.")
        except Exception:  # noqa: BLE001 — never let the close take the worker down
            return

    def _notify_terminal_comment_fix_failure(self, qjob: QueuedJob, code: str) -> None:
        """Keep the "a comment trigger never goes silent" promise when process_job itself RAISES.

        process_job replies in-thread on every outcome it can reach -- but a WebhookError raised
        before its first reply (token minting is the usual one) unwinds straight to _run_claimed,
        and the retry loop then burns the remaining attempts the same way, in seconds, with the
        only trace a stderr JSON line. From the requester's side that is exactly the silence
        observed on 2026-09-17: `@codna fix` replies with no ack, no check, no PR, no error. On
        the LAST attempt only, tell the thread what happened. Best-effort: a token is minted if
        possible (the failure may be precisely that), else GITHUB_TOKEN is tried; never raises.
        """
        job = qjob.job
        ctx = job.context or {}
        if job.kind != "fix" or not isinstance(ctx.get("in_reply_to_id"), int) or not job.pr_number:
            return
        if qjob.attempts < _MAX_ATTEMPTS:
            return
        token: str | None = None
        try:
            if job.installation_id and self._app_id and self._private_key:
                token = webhook_github.installation_token(
                    self._app_id, self._private_key, job.installation_id,
                    repo_full_name=job.repo_full_name, kind="review",  # read+comment scope suffices
                )
        except Exception:  # noqa: BLE001 — minting may be the very failure being reported
            token = None
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            return
        body = (f"⚠️ Codna couldn't process this `@codna fix` request ({code}) after "
                f"{qjob.attempts} attempt(s). The worker logged the details.")
        try:
            webhook_github.post_review_comment_reply(
                job.repo_full_name, token, job.pr_number, ctx["in_reply_to_id"], body
            )
        except Exception:  # noqa: BLE001 — never let the notification take the worker down
            return

    def _watchdog_loop(self) -> None:
        """Enforce the job deadline the pool advertises, for the WHOLE job rather than its child.

        Until this existed, `CODNA_WEBHOOK_JOB_TIMEOUT_S` bounded only the CLI subprocess; a job
        could hold its thread well past it in the phases around that call, with a required check
        left `in_progress` the entire time and nothing to do but redeploy. An overdue job is
        cancelled the same way a superseded one is: process group killed, runtime torn down, Check
        Run completed `neutral` (never `failure` -- the job made no finding), row written terminal.
        """
        interval = max(1.0, min(self._poll * 5, 15.0))
        while not self._stop.wait(interval):
            try:
                overdue = self._control.overdue()
            except Exception as exc:  # noqa: BLE001 — the watchdog must outlive any one bad scan
                webhook_worker._log_worker_loop_error(exc)
                continue
            for handle in overdue:
                bound = int(handle.deadline_s)
                webhook_control.cancel_jobs(
                    [handle], reason="job_deadline", queue=self._queue,
                    summary=(f"{check_run_name(handle.kind)}: stopped after {bound}s — this job passed the "
                             f"webhook's per-job deadline (CODNA_WEBHOOK_JOB_TIMEOUT_S) and its worker "
                             f"thread was released. Nothing was concluded about this commit; trigger it "
                             f"again to re-run it."),
                )

    def start(self) -> None:
        self._queue.recover_stale()
        if self._registry is None:
            self._reconciled.set()
        else:  # resolve every job the previous process left registered; _loop claims only after this
            reconcile_in_background(self._queue, self._registry, app_id=self._app_id,
                                    private_key=self._private_key, done=self._reconciled)
        for i in range(self._concurrency):
            t = threading.Thread(target=self._loop, name=f"codna-webhook-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="codna-webhook-watchdog",
                                          daemon=True)
        self._watchdog.start()

    def diagnostics(self) -> dict[str, Any]:
        alive = sum(1 for thread in self._threads if thread.is_alive())
        now = time.monotonic()
        with self._busy_lock:
            running_for = [now - since for since in self._busy_since.values()]
        longest = max(running_for, default=0.0)
        # An alive thread is not a working thread: one sitting on a single job past the bound
        # _run_job_process enforces (plus slack) is wedged, and /ready must say so rather than
        # report a healthy pool that processes nothing.
        wedged = longest > webhook_worker._JOB_TIMEOUT_S + webhook_worker._KILL_DRAIN_S + 60
        diag: dict[str, Any] = {
            "configured_threads": self._concurrency,
            "started_threads": len(self._threads),
            "alive_threads": alive,
            "busy_threads": len(running_for),
            "longest_running_s": round(longest, 1),
            "wedged": wedged,
            "ready": alive >= self._concurrency and not wedged,
            # Is the last free thread currently held for review/merge-gating work?
            "priority_thread_reserved": self._reserve_priority_thread(),
            "job_deadline_s": int(self._job_deadline_s),
        }
        # Admission state: threads are only the claimer ceiling, so "alive_threads" alone no longer
        # describes capacity. Surface what the admitter actually decided on (in-flight, how many
        # duration samples it has learned from, p50) — best-effort, never break diagnostics.
        try:
            admitter = get_admitter()
            diag["admission"] = {
                **admitter.snapshot(),
                "ceiling": admitter.cfg.ceiling,
                "det_cap": admitter._det_cap(None),
            }
        except Exception:  # noqa: BLE001 — diagnostics must not fail on the admitter
            diag["admission"] = None
        return diag

    def stop(self, timeout: float | None = None) -> None:
        """Signal stop and DRAIN: wait (up to the drain grace) for the in-flight job to complete
        and mark itself done, so a routine redeploy doesn't abandon + re-run it (duplicate PRs).
        Jobs still running past the grace are left for recover_stale on the next boot."""
        grace = self._drain_timeout if timeout is None else timeout
        self._stop.set()
        deadline = time.monotonic() + max(0.0, grace)
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._watchdog is not None:
            self._watchdog.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._registry:  # whatever is still registered outlived the grace: say so for the next boot
            self._registry.mark_interrupted("sigterm")
