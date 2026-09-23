"""The review Check Run is opened ``queued`` by the INGRESS, the moment a delivery is accepted.

Until now the worker created the run when it claimed the job. With the horizontal queue (ingress +
worker pool) a review that waits for a slot is INVISIBLE to the tenant until then: the 2026-09-20
rollout measured delivery -> check-created p50 ~2 s but p95 up to 23 s during a 21-review fan-out on
6 slots, and nothing on the pull request said a review was even coming. Here the ingress:

  1. mints the job's own scoped token and creates the ``codna review`` run ``queued`` on the event
     head (:func:`precreate`), inside a time budget that keeps GitHub's 10 s delivery window;
  2. hands the run's id to ``enqueue``, so the row is BORN owning it (``check_run_id``,
     ``check_started_at``); the worker that claims the row flips the run to ``in_progress`` and
     completes it with the verdict (:func:`open_check_run`, below) -- one run per commit;
  3. right after the enqueue, checks that the run IS bound to a row (:func:`bind_or_close`): an
     insert that collapsed onto a live row for the same head hands the run to that row when it has
     none, and anything still unowned is completed at once -- a ``queued`` run nobody owns would
     otherwise sit on the commit forever, because only this App can complete it;
  4. completes the ``queued`` runs of the rows its enqueue retired (:func:`settle_retired`): a newer
     head supersedes the WAITING older heads in the same transaction as its insert, and their runs
     end ``neutral`` "superseded by <sha>" exactly as a RUNNING row's does. The reaper runs the same
     sweep every tick over every retired row (superseded, cancelled by an operator, exported by a
     rollback, failed before the worker could report), so no path leaves a run open.

Only ``review`` deliveries that carry a head are pre-created: the review is the merge-gating check
whose visibility this exists for, and it is the kind a second delivery for one head collapses onto
(``jobs_one_live_head``). A ``fix`` from a red check suite is one of several deliveries per head
(the 24 h rule collapses the rest), so pre-creating it would orphan a run per collapsed delivery.
Postgres backend only: the SQLite queue has no reaper to sweep a stranded run and runs its jobs in
the same process, where the claim follows the enqueue by one poll interval -- it keeps create-at-
claim (``ShadowQueue.enqueue`` / ``SpoolingQueue.enqueue`` drop the id on that path).

Every function here is duck-typed over ``github`` and ``queue`` and unit-tested with fakes.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .webhook import WebhookJob, check_run_name
from .webhook_control import JobHandle
from .webhook_queue import QueuedJob

PRECREATE_BUDGET_ENV = "CODNA_WEBHOOK_CHECK_PRECREATE_BUDGET_S"
DEFAULT_PRECREATE_BUDGET_S = 5.0  # GitHub retries a delivery not acknowledged within 10 s; the enqueue has 2 s
_GENERIC_RETRIGGER = "Trigger it again to run it."


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PreCreated:
    """A Check Run the ingress opened for a delivery, and the token that may complete it."""

    check_run_id: int
    token: str
    head_sha: str


def wants_queued_check(job: WebhookJob) -> bool:
    """Whether the ingress opens this job's Check Run at enqueue: a review of a known head."""
    return job.kind == "review" and bool(job.ref) and job.pr_number is not None


def precreate_budget_s(environ: Mapping[str, str] | None = None) -> float:
    source = os.environ if environ is None else environ
    try:
        return max(0.5, float(source.get(PRECREATE_BUDGET_ENV) or DEFAULT_PRECREATE_BUDGET_S))
    except ValueError:
        return DEFAULT_PRECREATE_BUDGET_S


def token_for(*, github: Any, app_id: str | None, private_key: str | None, environ: Mapping[str, str] | None = None,
              ) -> Callable[[Any], str | None]:
    """A ``row -> token`` resolver: the App's installation token scoped to the row's repo and kind
    (the scope the job itself runs with), else ``GITHUB_TOKEN`` for a single-repo self-host. Rows
    are dicts (the queue's) or WebhookJobs. None when nothing can be minted; never raises."""
    source = os.environ if environ is None else environ
    cache: dict[tuple[Any, str, str], str | None] = {}

    def _resolve(row: Any) -> str | None:
        if isinstance(row, WebhookJob):
            installation, repo, kind = row.installation_id, row.repo_full_name, row.kind
        else:
            installation, repo, kind = row.get("installation_id"), row.get("repo"), row.get("kind") or "review"
        if installation and app_id and private_key:
            key = (installation, str(repo), str(kind))
            if key not in cache:
                try:
                    cache[key] = github.installation_token(app_id, private_key, int(installation),
                                                           repo_full_name=repo, kind=kind)
                except Exception as exc:  # noqa: BLE001 -- the caller degrades; the log says why
                    _log("check_run_token_failed", repo=repo, kind=kind, error=type(exc).__name__)
                    cache[key] = None
            return cache[key]
        return source.get("GITHUB_TOKEN") or None

    return _resolve


# --- 1 + 2: open the run before the row exists ------------------------------------------------------
def precreate(job: WebhookJob, *, queue: Any, delivery_id: str | None, github: Any, app_id: str | None,
              private_key: str | None, environ: Mapping[str, str] | None = None, budget_s: float | None = None,
              late_gate: threading.Event | None = None) -> PreCreated | None:
    """Open the job's ``codna review`` run ``queued`` on its event head. None means "create at claim"
    (today's path): not a review with a head, a delivery the queue has already seen, a head that
    already has a live job, a queue that cannot answer (Postgres unreachable -- the row is about to
    be spooled), no token, or GitHub not answering inside the budget.

    The dedup questions are asked FIRST because ``enqueue`` decides them inside its own transaction
    and only reports a bool: a run opened for a delivery that then collapses would be a second run on
    the commit. ``bind_or_close`` still catches what slips through the window between the question
    and the insert. A create that answers AFTER the budget is not lost: once ``late_gate`` is set
    (the caller sets it when its enqueue has returned) the run is handed to the delivery's own
    waiting row, or completed at once when a worker already claimed it (:func:`_late_precreated`).
    """
    if not wants_queued_check(job):
        return None
    name = check_run_name(job.kind)
    try:
        if queue.has_delivery(delivery_id):
            _log("check_run_precreate_skipped", repo=job.repo_full_name, pr_number=job.pr_number,
                 delivery_id=delivery_id, reason="delivery_seen")
            return None
        if queue.has_pending(repo=job.repo_full_name, pr_number=job.pr_number, ref=job.ref, kind=job.kind):
            _log("check_run_precreate_skipped", repo=job.repo_full_name, pr_number=job.pr_number,
                 delivery_id=delivery_id, reason="head_already_pending")
            return None
    except Exception as exc:  # noqa: BLE001 -- the queue cannot answer: the delivery is about to be spooled
        _log("check_run_precreate_skipped", repo=job.repo_full_name, pr_number=job.pr_number,
             delivery_id=delivery_id, reason="queue_unavailable", error=type(exc).__name__)
        return None
    resolve = token_for(github=github, app_id=app_id, private_key=private_key, environ=environ)

    def _create() -> PreCreated | None:
        token = resolve(job)
        if not token:
            return None
        cid = github.create_check_run(job.repo_full_name, token, name=name, head_sha=job.ref,
                                      summary=f"{name} queued ({job.reason}); waiting for a worker", status="queued")
        return PreCreated(check_run_id=int(cid), token=token, head_sha=str(job.ref)) if cid else None

    from .webhook_backend import _call_with_timeout  # the spool's bounded call, reused

    def _late(pre: PreCreated | None) -> None:
        if pre is not None:
            _late_precreated(pre, job=job, queue=queue, delivery_id=delivery_id, github=github)

    try:
        made = _call_with_timeout(_create, budget_s if budget_s is not None else precreate_budget_s(environ),
                                  on_late=_late, late_gate=late_gate)
    except TimeoutError:
        _log("check_run_precreate_timeout", repo=job.repo_full_name, pr_number=job.pr_number, delivery_id=delivery_id)
        return None
    except Exception as exc:  # noqa: BLE001 -- GitHub refused or the transport failed: today's path
        _log("check_run_precreate_failed", repo=job.repo_full_name, pr_number=job.pr_number,
             delivery_id=delivery_id, error=type(exc).__name__)
        return None
    if made is not None:
        _log("check_run_precreated", repo=job.repo_full_name, pr_number=job.pr_number, head=str(job.ref)[:8],
             delivery_id=delivery_id, check_run_id=made.check_run_id)
    return made


def _late_precreated(pre: PreCreated, *, job: WebhookJob, queue: Any, delivery_id: str | None, github: Any) -> None:
    """A create that outran the budget: the delivery's row exists by now (the gate), without a run.
    Adopt onto it while it is still waiting; otherwise a worker has it and opened its own run."""
    try:
        adopted = bool(queue.adopt_check_run(job, pre.check_run_id, delivery_id=delivery_id))
    except Exception as exc:  # noqa: BLE001
        _log("check_run_late_adopt_error", delivery_id=delivery_id, error=type(exc).__name__)
        adopted = False
    if adopted:
        _log("check_run_late_adopted", repo=job.repo_full_name, delivery_id=delivery_id, check_run_id=pre.check_run_id)
        return
    _close_orphan(pre, job=job, github=github,
                  summary=f"{check_run_name(job.kind)}: this run was opened after its job had already started "
                          f"on another run; that run carries the verdict.")
    _log("check_run_late_closed", repo=job.repo_full_name, delivery_id=delivery_id, check_run_id=pre.check_run_id)


def _in_spool(queue: Any, delivery_id: str | None) -> bool | None:
    """Whether this delivery's row is WAITING in the SQLite spool (``SpoolingQueue.spool``): the
    Postgres write failed and the file took it. False for a queue WITHOUT a spool (nothing could
    have spooled it); None when a spool exists but cannot say (no ``find_in_flight``, or the file
    itself raised) -- the caller treats "cannot tell" as unsafe, never as "not spooled"."""
    spool = getattr(queue, "spool", None)
    if spool is None:
        return False
    finder = getattr(spool, "find_in_flight", None)
    if not callable(finder) or not delivery_id:
        return None
    try:
        return finder(delivery_id) is not None
    except Exception:  # noqa: BLE001
        return None


def _close_orphan(pre: PreCreated, *, job: WebhookJob, github: Any, summary: str) -> bool:
    """Complete a run no row owns. ``cancelled``, never ``neutral``: this run is on the CURRENT head
    and GitHub shows the newest run of a name, so a passing conclusion here could wave the head
    through a required check before the run that actually reviews it has reported. Never raises."""
    try:
        github.update_check_run(job.repo_full_name, pre.token, pre.check_run_id, conclusion="cancelled",
                                summary=summary, name=check_run_name(job.kind))
        return True
    except Exception as exc:  # noqa: BLE001 -- the reaper's sweep cannot see an unowned run; say so loudly
        _log("check_run_orphan_close_failed", repo=job.repo_full_name, check_run_id=pre.check_run_id,
             error=type(exc).__name__)
        return False


# --- 3: after the enqueue, the run is owned by a row or gone --------------------------------------
def bind_or_close(pre: PreCreated, *, job: WebhookJob, queue: Any, delivery_id: str | None, fresh: bool,
                  github: Any) -> str:
    """Returns ``bound`` (the delivery's row carries this run), ``adopted`` (the insert collapsed
    onto a live row for the same head that had no run: it owns this one now) or ``closed`` (no
    row owns it: completed at once -- the enqueue collapsed onto a row that already has its run,
    or Postgres failed and the delivery went to the spool, whose row the worker gives a fresh run).

    The bind read is a VERIFICATION of what a fresh enqueue already did (the row is inserted with
    the id). When that read raises, the run is closed only if the delivery demonstrably went to
    the spool (``queue.spool.find_in_flight``); a Postgres blip right after a successful insert
    otherwise leaves a bound run alone (``bound`` reported unverified), and a worker never starts
    a run it did not first read as open (``open_check_run``)."""
    bound = None
    unreadable = False
    try:
        bound = queue.check_run_bound(delivery_id)
    except Exception as exc:  # noqa: BLE001 -- Postgres did not answer; the spool says where the row went
        unreadable = True
        _log("check_run_bind_unreadable", delivery_id=delivery_id, error=type(exc).__name__)
    if bound == pre.check_run_id:
        return "bound"
    if unreadable and fresh and _in_spool(queue, delivery_id) is False:
        # The insert succeeded (fresh) and the delivery is demonstrably NOT in the file, so it is a
        # Postgres row born with this id; only the read-back failed. Closing here would cancel the
        # run the worker is about to run on (codna review of thyn-ai/codna#596). "Cannot tell"
        # (None) falls through to the close: a wrongly closed bound run is repaired by the worker,
        # which reads the run before reusing it and replaces a completed one; a wrongly kept
        # spooled run has no owner and would stay `queued` forever.
        _log("check_run_bind_unverified", repo=job.repo_full_name, delivery_id=delivery_id,
             check_run_id=pre.check_run_id)
        return "bound"
    if not fresh:
        try:
            if queue.adopt_check_run(job, pre.check_run_id):
                _log("check_run_adopted", repo=job.repo_full_name, pr_number=job.pr_number, head=pre.head_sha[:8],
                     delivery_id=delivery_id, check_run_id=pre.check_run_id)
                return "adopted"
        except Exception as exc:  # noqa: BLE001
            _log("check_run_adopt_error", delivery_id=delivery_id, error=type(exc).__name__)
    name = check_run_name(job.kind)
    why = ("a second delivery for this commit collapsed onto the job already queued for it; that job's run "
           "carries the verdict" if not fresh else
           "the queue could not be reached when this delivery was accepted; the job was spooled and the worker "
           "that runs it opens a new run")
    _close_orphan(pre, job=job, github=github, summary=f"{name}: {why}. Comment `@codna review` to re-run if this "
                                                           f"stays the newest check on the commit.")
    _log("check_run_orphan_closed", repo=job.repo_full_name, pr_number=job.pr_number, head=pre.head_sha[:8],
         delivery_id=delivery_id, check_run_id=pre.check_run_id, fresh=fresh)
    return "closed"


# --- the worker's side: the run the row carries is UPDATED, never created twice -----------------
def open_check_run(qjob: QueuedJob, token: str, github: Any, *, head_sha: str,
                   on_check_run: Callable[[int], None] | None, control: JobHandle | None,
                   log: Callable[..., None] | None = None) -> int | None:
    """The Check Run this job reports on, anchored to ``head_sha`` and now ``in_progress``.

    The row's OWN run when it carries one for exactly this commit (the ingress opened it ``queued``
    at enqueue, above) and it is still open: flipped to ``in_progress`` here, never created twice,
    so the pull request shows ONE `codna review` per commit from the moment the delivery was
    accepted. Otherwise a fresh run, as the worker always did: rows an older ingress or a requeue
    wrote without one, a review re-anchored to a head that moved before it started
    (``webhook_worker._review_head_at_start`` completed the event head's run), and a run that is
    no longer open -- a retry whose previous attempt completed it, an operator's ``retry-now`` of
    a row whose run the sweep or the dead-letter notice already closed. Whether the run is open is
    ASKED of GitHub (``check_run_status``) on every claim rather than inferred from the attempt
    count: a completed run is never reopened, and a run this job does NOT reuse is closed if it is
    still open before the fresh one is created -- an unreadable answer (a 404, a bad body, a client
    without the helper) can therefore never strand the run the ingress opened. ``log`` is the
    worker's phase logger, bound to the job.
    """
    job = qjob.job
    name = check_run_name(job.kind)
    owned = qjob.check_run_id if (qjob.check_run_id and job.ref and head_sha == job.ref) else None
    if owned:
        probe = getattr(github, "check_run_status", None)
        try:
            status = probe(job.repo_full_name, token, owned) if callable(probe) else None
        except Exception:  # noqa: BLE001 -- unreadable: never started blind; replaced, and closed if open
            status = None
        if status not in ("queued", "in_progress"):
            closed = None
            if status != "completed":
                # Unknown state: the run may well still be `queued` (the ingress opened it seconds
                # ago). Nothing else ever completes the run of a row this worker holds, so it is
                # closed here, read-then-write, before its replacement exists. A run already
                # completed needs nothing.
                closed = _close_abandoned(job, token, github, owned, name=name, reason=job.reason)
            if log is not None:
                log("check_run_not_reusable", check_run_id=owned, status=status, closed=closed)
            owned = None
    if owned:
        github.start_check_run(job.repo_full_name, token, owned, name=name, summary=f"{name} started ({job.reason})")
        if log is not None:
            log("check_run_started", check_run_id=owned, head=head_sha)
        check_id: int | None = owned
    else:
        # A retry (recover_stale after a restart, or attempt 2/3 after a crash) must supersede the
        # run its predecessor left open on this commit -- only this App can complete it, so
        # otherwise it spins forever next to the new one (thyn-ai/algenta@cc26ddd6 `codna fix`,
        # in_progress from 2026-09-17 04:49 onward). Opportunistic: a client without the helper
        # (older fakes, other implementations) just skips it.
        complete_stale = getattr(github, "complete_stale_check_runs", None)
        if callable(complete_stale) and qjob.attempts > 1:
            # Only a retry has a predecessor to supersede. On a first attempt the only open run of
            # this name on the commit would be a concurrent sibling job's LIVE one.
            try:
                complete_stale(
                    job.repo_full_name, token, head_sha=head_sha, name=name,
                    summary=f"{name}: an earlier attempt was interrupted (worker restart); "
                            f"superseded by a new run ({job.reason}).",
                )
            except Exception:  # noqa: BLE001 -- hygiene must never block creating this job's own run
                pass
        check_id = github.create_check_run(
            job.repo_full_name, token, name=name, head_sha=head_sha,
            summary=f"{name} started ({job.reason})",
        )
    if check_id and on_check_run:
        on_check_run(check_id)  # the row / restart registry: a killed job's run is completed on the next boot
    if check_id and control is not None:
        control.attach_check_run(token, check_id)  # a cancellation can now close THIS run
    return check_id


def _close_abandoned(job: WebhookJob, token: str, github: Any, check_run_id: int, *, name: str, reason: str) -> bool | None:
    """Complete a row-carried run this job will not reuse, if it is still open. ``cancelled``: a
    fresh run of the same name replaces it on this commit and is the newest. True when this call
    closed it, False when it was not open (or the client cannot say), None without a closer."""
    closer = getattr(github, "complete_check_run_if_open", None)
    if not callable(closer):
        return None
    try:
        return bool(closer(job.repo_full_name, token, check_run_id, conclusion="cancelled", name=name,
                           summary=f"{name}: replaced by a fresh run ({reason}); its state could not be read."))
    except Exception:  # noqa: BLE001 -- best-effort hygiene; the fresh run is what the tenant sees
        return False


def report_without_running(qjob: QueuedJob, token: str, github: Any, *, opening: str, conclusion: str,
                           summary: str) -> None:
    """Complete the job's Check Run with a verdict reached before anything ran (the account bridge
    unreachable, the org not linked, automation switched off): the row's own run when the ingress
    opened one, else a fresh one -- one `codna review` per commit either way."""
    job = qjob.job
    name = check_run_name(job.kind)
    cid = qjob.check_run_id or github.create_check_run(job.repo_full_name, token, name=name, head_sha=job.ref,
                                                       summary=opening)
    if cid:
        github.update_check_run(job.repo_full_name, token, cid, conclusion=conclusion, summary=summary, name=name)


# --- 4: the runs of retired rows ---------------------------------------------------------------------
def retired_verdict(row: Mapping[str, Any], *, retrigger: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(conclusion, summary) for the still-open run of a row that ended without a worker closing it."""
    name = check_run_name(str(row.get("kind") or "review"))
    hint = (retrigger or {}).get(str(row.get("kind") or ""), _GENERIC_RETRIGGER)
    status = str(row.get("status") or "")
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    last_error = str(row.get("last_error") or "")
    if status == "superseded":
        # A superseded row examined nothing and its commit is no longer the head: neutral, the same
        # conclusion a RUNNING row gets when a newer head cancels it (webhook_control).
        detail = str(result.get("summary") or last_error or "superseded")
        return "neutral", f"{name}: {detail} -- this commit is no longer the pull request head; see the check on the current head."
    if status == "exported":
        return "cancelled", f"{name}: the queue was moved to the SQLite file before this job ran; a new run replaces this one."
    if status == "failed":
        # The worker's error code names the cause (`result.error`); last_error carries its message.
        code, message = str(result.get("error") or ""), last_error or str(result.get("summary") or "")
        detail = f"{code}: {message}" if code and message and code not in message else (code or message or "failed")
        return "failure", f"{name}: failed before it could report ({detail}). {hint}"
    reason = (last_error or str(result.get("summary") or status)).removeprefix("cancelled:")
    return "cancelled", f"{name}: cancelled before it ran ({reason}). {hint}"


def settle_retired(queue: Any, *, github: Any, token_for: Callable[[Any], str | None],
                   repo: str | None = None, pr_number: int | None = None,
                   retrigger: Mapping[str, str] | None = None, limit: int = 100) -> int:
    """Complete the Check Runs of retired rows that still carry one (``webhook_pg_ops.
    retired_check_runs``), exactly once per row (``posted.check_closed`` is set only after the run is
    confirmed completed, so a GitHub hiccup is retried next time). Returns how many rows were settled.
    A run already completed by someone else -- the job's own CLI, a canceller -- is left as it is.
    No-op on a queue without Postgres. Never raises."""
    from . import webhook_pg_ops
    from .webhook_ops import _pg

    pg = _pg(queue)
    if pg is None:
        return 0
    try:
        rows = webhook_pg_ops.retired_check_runs(pg, repo=repo, pr_number=pr_number, limit=limit)
    except Exception as exc:  # noqa: BLE001
        _log("retired_check_runs_unreadable", error=type(exc).__name__)
        return 0
    settled = 0
    for row in rows:
        rid, cid = int(row["id"]), int(row["check_run_id"])
        token = token_for(row)
        if not token:
            continue
        conclusion, summary = retired_verdict(row, retrigger=retrigger)
        name = check_run_name(str(row.get("kind") or "review"))
        try:
            closed = bool(github.complete_check_run_if_open(row["repo"], token, cid, conclusion=conclusion,
                                                            summary=summary, name=name))
        except Exception as exc:  # noqa: BLE001
            _log("retired_check_run_close_failed", row_id=rid, check_run_id=cid, error=type(exc).__name__)
            continue
        if not closed:
            # Already completed (the job's CLI, a canceller, the dead-letter notice) -- or the call
            # failed; only a read tells. Without one, leave the row for the next pass.
            probe = getattr(github, "check_run_status", None)
            try:
                already = callable(probe) and probe(row["repo"], token, cid) == "completed"
            except Exception:  # noqa: BLE001
                already = False
            if not already:
                _log("retired_check_run_unconfirmed", row_id=rid, check_run_id=cid, status=row.get("status"))
                continue
        try:
            pg.mark_posted(rid, "check_closed")
        except Exception as exc:  # noqa: BLE001 -- the run IS closed; a second pass finds it completed
            _log("retired_check_run_mark_failed", row_id=rid, error=type(exc).__name__)
        _log("retired_check_run_closed", row_id=rid, kind=row.get("kind"), repo=row.get("repo"),
             status=row.get("status"), check_run_id=cid, conclusion=conclusion, by_sweep=closed)
        settled += 1
    return settled


def settle_retired_in_background(queue: Any, *, github: Any, token_for: Callable[[Any], str | None],
                                 repo: str, pr_number: int | None) -> threading.Thread:
    """The ingress's fast path: off the request thread (a GitHub call per retired row), for the
    pull request whose newer head just retired its older ones. The reaper is the backstop."""
    thread = threading.Thread(
        target=lambda: settle_retired(queue, github=github, token_for=token_for, repo=repo, pr_number=pr_number),
        name="codna-webhook-settle-retired", daemon=True)
    thread.start()
    return thread
