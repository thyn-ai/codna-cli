"""Worker liveness and the reaper for the Postgres queue.

Two threads, both stdlib, both unit-tested with fakes:

* :class:`WorkerRegistration` -- registers this process in ``workers`` and heartbeats every
  ``interval_s`` with its live slot count; when the scaler (or an operator) sets ``draining`` it
  acknowledges, stops the queue from claiming, and reports ``drained`` once nothing is running so
  the process can exit 0 and the machine can stop.
* :class:`Reaper` -- runs on whichever process holds the ``codna_webhook:reaper`` leader lock
  (the ingress in production; in ``role=all`` the single machine). Each tick it (1) resolves
  running rows whose lease lapsed with the verdicts ``webhook_resume._reconcile_one`` uses today
  -- superseded head: cancelled and the CURRENT head queued; interrupted before: dead with the
  re-trigger hint; else its stale Check Run is completed ``cancelled`` and the row re-queued
  with ``resumes + 1``; (2) tells tenants about dead-lettered jobs exactly once (Check Run
  completed with the dead-letter conclusion, comment-fix thread replied to), never a silent drop
  and never a red check from our own outage; (3) once Postgres has been healthy again for
  ``outage_retry_after_s``, runs dead rows whose failure text names an outage one more time;
  (4) forgets workers whose heartbeat is long gone; (5) completes the Check Runs of rows that were
  retired before any worker touched them -- the ``queued`` run the ingress opens at enqueue, when
  a newer head superseded the row, an operator cancelled it, a rollback exported it, or it failed
  before the worker could report (``webhook_queued_check.settle_retired``; the ingress does the
  same for the pull request it just retired rows on, this is the backstop).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Callable

from . import webhook_github, webhook_pg_ops, webhook_queued_check, webhook_retry
from .webhook import WebhookJob, check_run_name
from .webhook_pg_queue import MAX_RESUMES, PostgresQueue

RETRIGGER = {
    "review": "Comment `@codna review` on the pull request to run it again.",
    "fix": "Reply `@codna fix` on the finding (or re-apply the `codna-fix` label) to run it again.",
    "secure": "Re-apply the `codna-secure` label to run it again.",
    "queue": "Re-queue the pull request to run it again.",
}
DEAD_LETTER_CONCLUSION_ENV = "CODNA_WEBHOOK_DEAD_LETTER_CONCLUSION"


def _log(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"service": "codna-webhook-reaper", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True, default=str), file=sys.stderr, flush=True)


def dead_letter_conclusion(tenant_override: str | None, environ: Any = None) -> str:
    """``neutral`` (never a red check from our outage; GitHub treats it as passing a required
    check) or ``action_required`` (fail closed). Per-tenant override first, then the env, then
    ``neutral``. The owner's decision -- see infra/README-webhook-deploy.md."""
    source = os.environ if environ is None else environ
    value = tenant_override or source.get(DEAD_LETTER_CONCLUSION_ENV) or "neutral"
    return value if value in ("neutral", "action_required") else "neutral"


class WorkerRegistration:
    """See the module docstring. ``busy`` is read from the queue's live lease table."""

    def __init__(self, queue: PostgresQueue, *, role: str, slots: int, interval_s: float = 10.0,
                 machine_id: str | None = None, region: str | None = None, image: str | None = None,
                 on_drained: Callable[[], None] | None = None) -> None:
        self._queue = queue
        self._role = role
        self._slots = int(slots)
        self._interval = float(interval_s)
        self.interval_s = self._interval
        self._machine_id = machine_id or os.environ.get("FLY_MACHINE_ID")
        self._region = region or os.environ.get("FLY_REGION")
        self._image = image or os.environ.get("FLY_IMAGE_REF")
        self._on_drained = on_drained
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last: dict[str, Any] | None = None
        self.drained = False

    def busy(self) -> int:
        return len(self._queue.active_leases())  # the live lease table is the truth about busy slots

    def register(self) -> None:
        webhook_pg_ops.register_worker(self._queue, machine_id=self._machine_id, region=self._region,
                                       image=self._image, role=self._role, slots=self._slots)

    def beat(self) -> dict[str, Any] | None:
        row = webhook_pg_ops.worker_heartbeat(self._queue, busy=self.busy())
        if row is None:  # pruned while we were away (a long partition): come back
            self.register()
            row = webhook_pg_ops.worker_heartbeat(self._queue, busy=self.busy())
        self.last = row
        draining = bool(row and row.get("draining"))
        self._queue.draining = draining
        if draining and self.busy() == 0 and not self.drained:
            self.drained = True
            _log("worker_drained", owner=self._queue.owner)
            if self._on_drained is not None:
                self._on_drained()
        return row

    def start(self) -> None:
        self.register()
        self._thread = threading.Thread(target=self._loop, name="codna-webhook-worker-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.beat()
            except Exception as exc:  # noqa: BLE001 -- a DB blip must not kill liveness reporting
                _log("worker_heartbeat_error", error=type(exc).__name__)

    def stop(self) -> None:
        self._stop.set()
        try:
            webhook_pg_ops.deregister_worker(self._queue)
        except Exception:  # noqa: BLE001
            pass


class Reaper:
    """See the module docstring. ``github`` is duck-typed (tests pass a fake)."""

    def __init__(self, queue: PostgresQueue, *, app_id: str | None, private_key: str | None,
                 github: Any = webhook_github, environ: Any = None, interval_s: float = 15.0,
                 outage_retry_after_s: float = 300.0, leader: Any = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._queue = queue
        self._app_id = app_id
        self._private_key = private_key
        self._github = github
        self._environ = os.environ if environ is None else environ
        self._interval = float(interval_s)
        self._outage_retry_after_s = float(outage_retry_after_s)
        self._leader = leader if leader is not None else webhook_pg_ops.LeaderLock(queue, "reaper")
        self._clock = clock
        self._healthy_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        self.last_outcome: dict[str, int] = {}

    # -- tokens ----------------------------------------------------------------------------------
    def _token(self, row: dict[str, Any]) -> str | None:
        if row.get("installation_id") and self._app_id and self._private_key:
            try:
                return self._github.installation_token(self._app_id, self._private_key, int(row["installation_id"]),
                                                       repo_full_name=row["repo"], kind=row["kind"])
            except Exception as exc:  # noqa: BLE001
                _log("reaper_token_failed", row_id=row["id"], error=type(exc).__name__)
                return None
        return self._environ.get("GITHUB_TOKEN") or None

    def _close(self, row: dict[str, Any], token: str, *, conclusion: str, summary: str) -> bool:
        cid = row.get("check_run_id")
        if not cid:
            return False
        try:
            return bool(self._github.complete_check_run_if_open(row["repo"], token, int(cid), conclusion=conclusion,
                                                                summary=summary, name=check_run_name(row["kind"])))
        except Exception as exc:  # noqa: BLE001
            _log("reaper_close_failed", row_id=row["id"], error=type(exc).__name__)
            return False

    # -- (1) expired leases -----------------------------------------------------------------------
    def reap_expired(self) -> dict[str, int]:
        outcome = {"resumed": 0, "superseded": 0, "dead": 0, "cancelled": 0, "skipped": 0}
        for row in webhook_pg_ops.expired_leases(self._queue):
            try:
                outcome[self._reap_one(row)] += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not strand the others
                _log("reaper_error", row_id=row.get("id"), error=f"{type(exc).__name__}: {str(exc)[:200]}")
                outcome["skipped"] += 1
        return outcome

    def _reap_one(self, row: dict[str, Any]) -> str:
        rid, kind = int(row["id"]), row["kind"]
        name = check_run_name(kind)
        hint = RETRIGGER.get(kind, "Trigger it again to run it.")
        token = self._token(row)
        actor = f"reaper@{self._queue.owner}"
        if row.get("cancel_requested"):
            # An operator or a superseding head asked; the worker's heartbeat already failed on it.
            why = str(row["cancel_requested"])
            if webhook_pg_ops.finish_row(self._queue, rid, status="cancelled", error=f"cancelled:{why}", actor=actor,
                                         summary=f"{name}: cancelled ({why})."):
                if token:
                    self._close(row, token, conclusion="cancelled", summary=f"{name}: cancelled -- {why}. {hint}")
                return "cancelled"
            return "skipped"
        if int(row["resumes"]) >= MAX_RESUMES:
            text = (f"{name}: interrupted again (its worker's lease lapsed) after one automatic resume; not re-run "
                    f"automatically. {hint}")
            if webhook_pg_ops.finish_row(self._queue, rid, status="dead", error="interrupted_after_resume", actor=actor, summary=text):
                if token:
                    self._close(row, token, conclusion="cancelled", summary=text)
                return "dead"
            return "skipped"
        if token and row.get("pr_number") and row.get("ref") and kind in ("review", "fix"):
            head = None
            try:
                head = self._github.pull_request_head_sha(row["repo"], token, int(row["pr_number"]))
            except Exception as exc:  # noqa: BLE001
                _log("reaper_head_lookup_failed", row_id=rid, error=type(exc).__name__)
            if head and head != row["ref"]:
                text = (f"{name}: its worker's lease lapsed and the pull request head has since moved to {head[:8]}, "
                        f"which gets its own run. This commit was not re-run.")
                if webhook_pg_ops.finish_row(self._queue, rid, status="cancelled", error="superseded", actor=actor, summary=text):
                    self._close(row, token, conclusion="cancelled", summary=text)
                    if kind == "review":
                        self._queue.enqueue(
                            WebhookJob(kind, row["repo"], ref=head, pr_number=int(row["pr_number"]),
                                       installation_id=row.get("installation_id"),
                                       reason=f"reaper_requeue_current_head_after_{row.get('reason') or 'interrupt'}"[:120]),
                            delivery_id=f"reaper-{rid}-{head}", priority=1, unless_pending=True)
                    return "superseded"
                return "skipped"
        hold = webhook_retry.backoff_s(kind, 1)
        if webhook_pg_ops.requeue_interrupted(self._queue, rid, cause="lease_expired", hold_s=hold, actor=actor):
            if token:
                self._close(row, token, conclusion="cancelled",
                            summary=f"{name}: its worker stopped responding; re-queued automatically as attempt "
                                    f"{int(row['attempts']) + 1}. A new `{name}` run replaces this one. If none appears: {hint}")
            return "resumed"
        return "skipped"

    # -- (2) dead letters, told once -----------------------------------------------------------------
    def notify_dead_letters(self) -> int:
        told = 0
        for row in webhook_pg_ops.unposted_dead(self._queue):
            try:
                if self._notify_one(row):
                    told += 1
            except Exception as exc:  # noqa: BLE001
                _log("dead_letter_notify_error", row_id=row.get("id"), error=type(exc).__name__)
        return told

    def _notify_one(self, row: dict[str, Any]) -> bool:
        rid, kind = int(row["id"]), row["kind"]
        name = check_run_name(kind)
        if not self._queue.mark_posted(rid, "dead_letter"):
            return False  # another reaper got there first
        text = webhook_retry.dead_letter_summary(name, int(row["attempts"]), rid, retrigger=RETRIGGER.get(kind, ""))
        token = self._token(row)
        if not token:
            _log("dead_letter_unnotified", row_id=rid, reason="no_token")
            return True
        conclusion = dead_letter_conclusion(row.get("dead_letter_conclusion"), self._environ)
        posted = False
        if row.get("check_run_id"):
            try:
                self._github.update_check_run(row["repo"], token, int(row["check_run_id"]), conclusion=conclusion,
                                              summary=text, name=name)
                posted = True
            except Exception as exc:  # noqa: BLE001
                _log("dead_letter_check_update_failed", row_id=rid, error=type(exc).__name__)
        elif row.get("ref"):
            try:
                cid = self._github.create_check_run(row["repo"], token, name=name, head_sha=row["ref"], summary=f"{name}: could not complete")
                if cid:
                    self._github.update_check_run(row["repo"], token, int(cid), conclusion=conclusion, summary=text, name=name)
                    self._queue.record_check_run(rid, int(cid))
                    posted = True
            except Exception as exc:  # noqa: BLE001
                _log("dead_letter_check_create_failed", row_id=rid, error=type(exc).__name__)
        ctx = row.get("context") if isinstance(row.get("context"), dict) else {}
        if kind == "fix" and isinstance(ctx.get("in_reply_to_id"), int) and row.get("pr_number"):
            try:
                self._github.post_review_comment_reply(row["repo"], token, int(row["pr_number"]), int(ctx["in_reply_to_id"]),
                                                       "⚠️ " + text)
                posted = True
            except Exception as exc:  # noqa: BLE001
                _log("dead_letter_reply_failed", row_id=rid, error=type(exc).__name__)
        elif row.get("issue_number") and not row.get("ref"):
            try:
                self._github.post_issue_comment(row["repo"], token, int(row["issue_number"]), "⚠️ " + text)
                posted = True
            except Exception as exc:  # noqa: BLE001
                _log("dead_letter_comment_failed", row_id=rid, error=type(exc).__name__)
        _log("dead_letter_notified", row_id=rid, kind=kind, repo=row["repo"], conclusion=conclusion, posted=posted)
        return True

    # -- (3) run outage victims again once the outage is over ----------------------------------------
    def retry_after_outage(self) -> int:
        """Once Postgres has answered for ``outage_retry_after_s``, dead rows whose failure text names
        an outage get one more run (``retry_now``), each exactly once (``posted.outage_retry``)."""
        if not self._queue.ping():
            self._healthy_since = None
            return 0
        now = self._clock()
        if self._healthy_since is None:
            self._healthy_since = now
        if now - self._healthy_since < self._outage_retry_after_s:
            return 0
        retried = 0
        for row in webhook_pg_ops.dead_after_outage(self._queue):
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            if not webhook_retry.looks_like_outage(row.get("last_error") or "", error_code=str(result.get("error") or "")):
                continue
            if not self._queue.mark_posted(int(row["id"]), "outage_retry"):
                continue
            out = webhook_pg_ops.retry_now(self._queue, int(row["id"]), actor=f"reaper-outage@{self._queue.owner}")
            if out.get("outcome") == "queued":
                retried += 1
        return retried

    # -- (5) the runs of rows retired before a worker saw them ----------------------------------------
    def close_retired_check_runs(self) -> int:
        """A ``queued`` Check Run the ingress opened belongs to a row; when that row ends without a
        worker (superseded while waiting, cancelled by an operator, exported by a rollback, failed
        before reporting) nothing else completes the run. Exactly once per row (``posted.check_closed``)."""
        return webhook_queued_check.settle_retired(self._queue, github=self._github, token_for=self._token,
                                                   retrigger=RETRIGGER)

    # -- the loop --------------------------------------------------------------------------------------
    def tick(self) -> dict[str, int]:
        self.ticks += 1
        out = self.reap_expired()
        out["dead_letters_notified"] = self.notify_dead_letters()
        out["outage_retries"] = self.retry_after_outage()
        out["retired_checks_closed"] = self.close_retired_check_runs()
        out["workers_pruned"] = webhook_pg_ops.prune_dead_workers(self._queue)
        self.last_outcome = out
        if any(out.values()):
            _log("reaper_tick", **out)
        return out

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="codna-webhook-reaper", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if not self._leader.try_acquire():
                    continue
                self.tick()
            except Exception as exc:  # noqa: BLE001 -- the reaper must outlive any one error
                _log("reaper_loop_error", error=f"{type(exc).__name__}: {str(exc)[:200]}")

    def is_leader(self) -> bool:
        return bool(getattr(self._leader, "held", False))

    def stop(self) -> None:
        self._stop.set()
        try:
            self._leader.release()
        except Exception:  # noqa: BLE001
            pass
