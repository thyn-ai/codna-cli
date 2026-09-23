"""Worker pool for Codna's GitHub App webhook - decoupled from the ingress.

Codna is local; the App is the only hosted surface — so workers are in-process threads in
that single app (not a machine fleet). The webhook is KEYLESS: it runs the packaged Codna CLI
with the encapsulated local runtime and a per-installation Codna account credential when the
install is linked. Each job:

  1. resolves the org's Codna run credential for the job's installation (webhook_metering);
     with NONE it FAILS CLOSED — posts a "link your account" Check Run when the event has a
     commit SHA, or an issue comment when the trigger is issue-label based, and does NOT spend;
  2. mints a short-lived, repo-scoped GitHub installation token (for the branch/PR/SARIF,
     NOT for metering);
  3. opens a Check Run, runs the ``codna`` CLI in a FRESH scrubbed temp workspace
     (only the scoped GitHub token + the per-org Codna key are injected — no raw
     provider key, and the host's other secrets never leak);
  4. reports on the Check Run and marks the queue row done/failed.

``process_job`` takes its GitHub client + runner as parameters, so it is unit-tested with
fakes and no network.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from . import webhook_ci_triage, webhook_control, webhook_github, webhook_metering, webhook_queued_check
from .agent_core_runtime import PROVIDER_RUNTIME_ENV_KEYS
from .webhook_control import JobHandle
from .webhook import (
    check_run_name,
    _FIX_WRITE_ASSOCIATIONS,
    WebhookJob,
    codna_command,
    webhook_marker,
)
from .webhook_queue import _MAX_ATTEMPTS, QueuedJob
from .webhook_summaries import (  # noqa: F401 -- re-exported: callers and tests address these via this module
    _CLI_SUMMARY,
    _NON_RETRYABLE_CODES,
    _SUMMARY_CODE,
    _cli_completed_check,
    _error_json_summary,
    _neutral_check_summary,
    _review_job_summary,
    _summary_is_neutral,
    _summary_is_retryable,
    _terse_cli_failure,
)
from .webhook_procs import (  # noqa: F401 -- re-exported: tests and callers address them via this module
    _KILL_DRAIN_S,
    _kill_process_group,
    _pid_alive,
    _pids_from_state_files,
    _pids_with_runtime_root,
    _run_job_process,
    _teardown_job_runtime,
)

_JOB_TIMEOUT_S = int(os.environ.get("CODNA_WEBHOOK_JOB_TIMEOUT_S", "3600"))


# Raw LLM provider keys are NEVER INHERITED from the webhook's own ambient host env — the
# webhook is keyless. Exclude them from the allowlist below; the one job-specific provider key
# (the linked org's own BYOK key, resolved fresh per job — see webhook_metering.
# resolve_provider_credentials) is INJECTED explicitly by _job_env instead, exactly like the
# per-org CODNA_API_KEY already is, so no house/other-org/ambient secret can leak into a job.
_PROVIDER_API_KEYS = frozenset({
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
})
# codna's own provider-name -> raw-key env var mapping (cli/codna/keystore.py's PROVIDER_TO_ENV).
# Google is dual-set (both GEMINI_API_KEY and GOOGLE_API_KEY) since the packaged agent-core
# sidecar's own SDKs vary in which one they read; the key value is identical either way.
_PROVIDER_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
# Safe, non-secret system + codna config the job may inherit.
_SAFE_CONFIG = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "PWD", "TZ",
    "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "NODE_ENV",
    "CODNA_SIDECAR_DIR", "CODNA_AGENT_CORE_DIR", "CODNA_RUNTIME_ROOT", "CODNA_PORT_BASE",
    "ALGENTA_ENGINE_DIR", "TELYS_HOME",
    "CODNA_AGENT_MODEL", "CODNA_FIX_MODEL", "ALGENTA_AGENT_MODEL", "ALGENTA_AGENT_PROVIDER",
    # Admission tuning (admission_control.get_admitter). SLA and ceiling were already here;
    # MODEL_RATE_BUDGET is the deterministic floor and was missing, so a hosted job could see two of
    # the three knobs and silently fall back to the default floor for the third.
    "CODNA_FIX_SLA_S", "CODNA_FIX_CEILING", "CODNA_MODEL_RATE_BUDGET", "CODNA_ADMISSION_MC",
    "CODNA_TIMEOUT_S",
    # The review turn budget (review_budget): where observed review turns persist across jobs (the
    # webhook points it at its volume) and the forecast's accepted breach risk.
    "CODNA_REVIEW_HISTORY_DIR", "CODNA_REVIEW_BUDGET_EPSILON",
})
# ALLOWLIST (not denylist): the job runs untrusted repo code, so it inherits ONLY safe config
# + Telys pointers/tuning. It does NOT inherit engine URL overrides, raw LLM provider keys, or
# an ambient API key. The per-ORG Codna credential is INJECTED explicitly per job, so each fix
# meters to the right account and no house/other-org secret can leak.
_ENV_ALLOWLIST = (
    _SAFE_CONFIG
    | (set(PROVIDER_RUNTIME_ENV_KEYS) - _PROVIDER_API_KEYS)  # Telys pointers/tuning, NOT provider keys
)


@dataclass(frozen=True)
class JobResult:
    ok: bool
    summary: str
    retryable: bool = True  # False = deterministic failure (bad inputs); the queue must not retry it
    retry_after_s: float | None = None  # hold the retry back in the queue this long (no thread sleeps)
    check_completed: bool = False  # the CLI already completed the job's Check Run (a review's findings)
    neutral: bool = False  # settled non-run (the sandbox cannot run the repo's tests): ok, check ends neutral
    cancelled: bool = False  # stopped from outside (head moved, or the job deadline): never a failure
    # A review found the pull request head had moved while it ran: the check on the reviewed
    # commit stands, and this head needs a review of its own. The pool queues it (requeue_moved_head).
    requeue_head: str | None = None

    @property
    def conclusion(self) -> str:
        if self.neutral or self.cancelled:
            # A superseded or timed-out job examined nothing, so it must not report `failure` on a
            # commit: on a repo where `codna review` is required that would block the merge it was
            # supposed to unblock.
            return "neutral"
        return "success" if self.ok else "failure"


_COMMENT_FIX_ACK = ("🔧 On it — analyzing this finding and opening a verified fix PR if the patch "
                    "passes tests and clears the risk gate.")


def _cancelled_result(job: WebhookJob, control: JobHandle) -> JobResult:
    """The one JobResult shape for a job stopped from outside.

    ``ok`` because nothing went wrong with the code, ``cancelled`` so the Check Run ends neutral,
    and NOT retryable: re-running a review of a head that has already been replaced would spend a
    metered run to publish a verdict about a commit nobody is merging. ``check_completed`` keeps
    ``process_job``'s generic ``update_check_run`` off this run -- a cancelled job's run is closed
    through ``webhook_control.close_check_run``, which never clobbers findings a CLI already
    posted, by whichever of the canceller and the worker gets there first.
    """
    detail = control.cancel_detail or control.cancel_reason or "cancelled"
    return JobResult(True, f"{check_run_name(job.kind)}: {detail}", retryable=False,
                     cancelled=True, check_completed=True)


def _prepare_comment_fix(job: WebhookJob, *, token: str, app_id: str | None, github: Any) -> str | None:
    """Validate + enrich a `@codna fix` review-comment job in place. Returns a user-facing SKIP reason
    (to reply with, do-not-run) or None to proceed. Mutates job.context with the finding fields so
    codna_command can render a precise --issue.

    Security: only act on a finding COMMENT that this GitHub App actually authored AND that parses as
    a Codna finding — a human can copy the marker text but cannot forge the App authorship."""
    from .review_github import parse_codna_finding

    ctx = job.context or {}
    if ctx.get("is_fork"):
        return ("`@codna fix` can't auto-fix a forked-PR finding yet (Codna can't push to the fork's "
                "branch). Apply the suggested change manually for now.")
    parent = github.get_pull_review_comment(job.repo_full_name, token, ctx.get("in_reply_to_id"))
    if not parent or not github.comment_authored_by_app(parent, app_id):
        return "`@codna fix` only works as a reply to a codna review finding."
    finding = parse_codna_finding(parent.get("body") or "")
    if not finding:
        return "Couldn't identify a codna review finding on this thread — reply `@codna fix` on a codna review comment."
    if not ctx.get("head_ref"):
        return "Couldn't determine the pull-request branch to target for the fix."
    ctx.update({
        "fp": finding.get("fp"),
        "path": ctx.get("path") or finding.get("path"),
        "line": ctx.get("line") or finding.get("line"),
        "severity": finding.get("severity"),
        "category": finding.get("category"),
        "title": finding.get("title"),
        "explanation": finding.get("explanation"),
    })
    return None


_WRITE_PERMISSIONS = frozenset({"admin", "maintain", "write"})

_WORKFLOWS_REMEDY = (
    "⚠️ Codna can't push a fix to this branch yet: its `.github/workflows` files differ from the default "
    "branch, and GitHub only lets an App push such a branch when the App has the **Workflows** permission. "
    "An admin can grant it once — GitHub App settings → Permissions & events → Repository permissions → "
    "Workflows: Read and write → save, then accept the new permission on the installation — or update this "
    "branch from the default branch. Then reply `@codna fix` again; nothing was spent."
)


def _workflows_permission_preflight(job: WebhookJob, token: str, github: Any) -> str | None:
    """The remedy text when the push is known to be refused, else None (proceed)."""
    has = getattr(github, "installation_has_workflows", None)
    granted = has(job.installation_id) if callable(has) else None
    if granted is not False:
        return None  # granted, or unknown (no fix token minted yet in this process) -> let it run
    differs = getattr(github, "workflows_differ_from_default", None)
    if not callable(differs) or not differs(job.repo_full_name, token, job.ref):
        return None
    return _WORKFLOWS_REMEDY


def _comment_fix_authorized(job: WebhookJob, token: str, github: Any) -> str | None:
    """None when the commenter may request a fix; otherwise a short reason for the job summary."""
    ctx = job.context or {}
    login = ctx.get("commenter")
    association = ctx.get("author_association")
    lookup = getattr(github, "collaborator_permission", None)
    permission = lookup(job.repo_full_name, token, login) if (callable(lookup) and login) else None
    if permission is not None:
        if permission in _WRITE_PERMISSIONS:
            return None
        return f"skipped: @{login} has {permission!r} access, `@codna fix` needs write"
    # No authoritative answer: fall back to what the payload said. A context without any association
    # at all predates this check (or is a synthetic job) and is allowed through unchanged.
    if association is None or association in _FIX_WRITE_ASSOCIATIONS:
        return None
    return f"skipped: @{login or 'unknown'} is {association} and the permission lookup was unavailable"


def _comment_fix_result_msg(job: WebhookJob, token: str, result: "JobResult", github: Any) -> str:
    """The reply Codna leaves on the finding thread after a comment-fix run."""
    if result.ok:
        url = github.find_open_pr_by_marker(job.repo_full_name, token, webhook_marker(job))
        return f"✅ Opened a verified fix PR: {url}" if url else "✅ Fix verified and pushed."
    if result.summary.lower().startswith("codna fix failed ("):
        # A structured CLI error (clone timeout, bad inputs, ...) is not a rejected patch: say
        # what actually happened instead of blaming the test + risk gate.
        return f"⚠️ Codna couldn't complete this fix — {_terse_cli_failure(result.summary)}. Nothing was pushed."
    return ("⚠️ Codna couldn't open a verified fix for this finding — the patch didn't clear the "
            "test + risk gate, so nothing was pushed (fail-closed). You can apply the fix manually.")


def _free_port_pair() -> tuple[int, int] | None:
    """Two ADJACENT free ports (engine, engine+1 for the sidecar -- runtime_config.
    local_runtime_urls' own contract) via OS-assigned ephemeral ports, not a hash of job
    identity: a hash risks two DIFFERENT jobs deriving the SAME port by coincidence, which is
    exactly the collision this exists to eliminate, not reintroduce more quietly. Binding port 0
    and reading back what the kernel assigned, then releasing it immediately, is the standard way
    to get a free port without a race against every other consumer of "free port" on the box --
    it isn't airtight (something else could grab it in the gap before the sidecar itself binds),
    but that residual is the same one every ephemeral-port scheme accepts, not a new one.
    Returns None (never raises) if binding just doesn't work; the caller falls back to sharing
    the default ports rather than failing the job over a nice-to-have."""
    for _attempt in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                base = probe.getsockname()[1]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", base + 1))
        except OSError:
            continue  # base+1 was taken; try another base rather than fail the job over this
        return base, base + 1
    return None


def _isolated_runtime_env(tmp: str) -> dict[str, str]:
    """CODNA_RUNTIME_ROOT + CODNA_PORT_BASE unique to THIS job, so concurrent jobs never share a
    sidecar in the first place -- the root cause of the P0 this closes: two jobs sharing one
    sidecar by design (same runtime_config_hash) tore each other's down mid-turn whenever their
    provider keys differed, which is the routine case on a multi-tenant webhook, not an edge case.
    CODNA_RUNTIME_ROOT nests inside the job's own scratch dir (`tmp`, already a fresh
    TemporaryDirectory per job), so it's already guaranteed unique and is cleaned up for free when
    that directory is removed -- no separate bookkeeping, nothing to leak across jobs.
    """
    env = {"CODNA_RUNTIME_ROOT": str(Path(tmp) / ".codna-runtime")}
    ports = _free_port_pair()
    if ports is not None:
        env["CODNA_PORT_BASE"] = str(ports[0])
    # No free-port fallback: CODNA_RUNTIME_ROOT alone still isolates each job's runtime STATE
    # (they no longer read/write the same local-stack.json), even if two jobs end up sharing the
    # default ports on a probe failure -- degraded, not broken, and never worth failing the job.
    return env


def _job_env(
    token: str | None, engine_key: str | None,
    provider: str | None = None, provider_key: str | None = None,
    *, tmp: str | None = None, git_identity: tuple[str, str] | None = None,
    check_run_id: int | None = None, check_head_sha: str | None = None,
) -> dict[str, str]:
    """Minimal allowlisted env + the scoped GitHub token, per-org Codna key, and (when the
    linked org has configured one) their own BYOK provider key — set consistently regardless of
    which of Anthropic/OpenAI/Google the org picked, so the local agent-core sidecar always finds
    a usable key instead of only ever checking for Anthropic's."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    if tmp is not None:
        # Job-specific isolation OVERRIDES whatever the host process's own env carries for these
        # two keys -- inheriting the host's shared values here would silently defeat the whole
        # point (every job would still resolve to the one shared runtime this exists to avoid).
        env.update(_isolated_runtime_env(tmp))
        # Every scratch dir the job's tools create (the review clone, secure-pr checkouts, junit
        # files, agent-core temp) lands INSIDE the job dir and is deleted with it. The review
        # clone used to land in the machine's shared /tmp and was never removed -- a ~0.5 GB
        # leak per algenta review that filled the root fs (ENOSPC, thyn-ai/algenta#1023).
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = tmp
    if token:
        env["GITHUB_TOKEN"] = token
    if engine_key:
        env["CODNA_API_KEY"] = engine_key  # per-installation org credential meters THIS org
    if provider and provider_key:
        for env_var in _PROVIDER_ENV_VARS.get(provider, ()):
            env[env_var] = provider_key
        # ALGENTA_AGENT_PROVIDER/CODNA_AGENT_PROVIDER take precedence over the sidecar's own
        # default (anthropic) in packaged_agent_runner.py -- without this, an org's OpenAI/Google
        # key would be present but never selected, and the sidecar would still look for Anthropic.
        env["ALGENTA_AGENT_PROVIDER"] = provider
    if git_identity:
        # the App's bot account -- fix commits then resolve to a real GitHub identity
        env["CODNA_GIT_USER_NAME"], env["CODNA_GIT_USER_EMAIL"] = git_identity
    if check_run_id is not None:
        # The Check Run this job opened. `codna review --post` completes THAT run with its findings
        # instead of opening a second one, so a review shows as ONE "codna review" check.
        env["CODNA_CHECK_RUN_ID"] = str(check_run_id)
    if check_head_sha:
        # The commit that run is anchored to, so the CLI's `--json` output can say whether the
        # check, the reviewed head and the pull request's head agree (review_github.post_review).
        env["CODNA_CHECK_RUN_HEAD_SHA"] = check_head_sha
    from .webhook_backend import current_job_id  # the queue row this thread runs: the review-body run stamp
    if current_job_id() is not None:  # (set by webhook_service's pool wrapper; see review_github.post_review)
        env["CODNA_REVIEW_RUN_ID"] = str(current_job_id())
    return env


def run_codna_job(job: WebhookJob, *, token: str | None, engine_key: str | None,
                  provider: str | None = None, provider_key: str | None = None,
                  codna_bin: str = "codna", git_identity: tuple[str, str] | None = None,
    check_run_id: int | None = None, control: JobHandle | None = None,
    check_head_sha: str | None = None,
) -> JobResult:
    """Default runner: run packaged codna for the job in an isolated, scrubbed workspace.

    ``control`` is this job's cancellation handle: the CLI subprocess is registered on it so a
    superseding head or the deadline watchdog can kill the process group, and every return path
    below reports the cancellation rather than whatever partial output the kill produced.
    """
    ctx = job.context or {}
    if job.kind == "fix" and ctx.get("routing_error"):
        # A report on the feedback repo with no product route: this is a correctly-handled
        # non-run, not a failure to retry (same convention as the "org not linked" / "automation
        # disabled" branches in process_job) -- report=True carries the explanation through to the
        # comment posted on the public issue, and spends nothing running codna for nowhere to go.
        return JobResult(True, ctx["routing_error"])
    # ignore_cleanup_errors: on a drain past its grace the interpreter exits while the job's CLI is
    # still writing here (it lives in its own session), and the finalizer's rmtree raced it into an
    # "[Errno 39] Directory not empty" traceback on every redeploy (2026-09-19). prepare_scratch_root
    # sweeps whatever survives on the next boot.
    with tempfile.TemporaryDirectory(prefix="codna-webhook-job-", ignore_cleanup_errors=True) as tmp:
        if control is not None:
            # A cancellation arriving before (or instead of) the subprocess still has to reach the
            # detached sidecar + engine this job is about to start under CODNA_RUNTIME_ROOT.
            control.attach_runtime(tmp)
        if job.kind == "secure":
            if not token:
                return JobResult(False, "secure job needs an installation token to fetch SARIF")
            sarif_path = webhook_github.fetch_code_scanning_sarif(job.repo_full_name, token, dest_dir=tmp)
            argv = codna_command(job, sarif_path=sarif_path)
        elif job.kind == "fix" and ctx.get("feedback_issue") and not ctx.get("in_reply_to_id"):
            # Feedback-routed fix: the issue lives on the FEEDBACK repo, not job.repo_full_name (the
            # routed TARGET repo this token is scoped to) -- so there is nothing this function can
            # fetch itself. process_job mints a second, feedback-repo-scoped token and fetches the
            # text into ctx["issue_text"] before calling this runner; a missing value here means
            # that fetch failed, and must be reported, not silently swapped for the wrong text.
            issue_text = ctx.get("issue_text")
            if not issue_text:
                return JobResult(False, "feedback-routed fix is missing its report text (fetch failed)")
            argv = codna_command(job, issue_text=issue_text)
        elif job.kind == "fix" and job.issue_number and not ctx:
            # Label-triggered fix (no ref, no comment context): `codna fix` needs the issue's own
            # text to know what to fix -- fetch it now, the one thing codna_command can't do itself
            # (it has no token). A fetch failure/empty issue must not silently no-op: report it the
            # same way process_job reports every other fix outcome for an issue-label trigger.
            if not token:
                return JobResult(False, "issue-label fix needs an installation token to fetch the issue")
            issue_text = webhook_github.fetch_issue_text(job.repo_full_name, token, job.issue_number)
            if not issue_text:
                return JobResult(False, f"could not fetch issue #{job.issue_number} text to describe the fix")
            argv = codna_command(job, issue_text=issue_text)
        else:
            argv = codna_command(job)
        argv[0] = codna_bin
        try:
            try:
                proc = _run_job_process(
                    argv, env=_job_env(token, engine_key, provider, provider_key, tmp=tmp, git_identity=git_identity,
                                   check_run_id=check_run_id, check_head_sha=check_head_sha),
                    cwd=tmp, timeout=_JOB_TIMEOUT_S,
                    on_process=control.attach_process if control is not None else None,
                )
            except subprocess.TimeoutExpired:
                if control is not None and control.cancelled:
                    return _cancelled_result(job, control)
                return JobResult(False, f"{check_run_name(job.kind)} timed out after {_JOB_TIMEOUT_S}s")
        finally:
            _teardown_job_runtime(tmp)  # on success, failure and timeout alike
        if control is not None and control.cancelled:
            # The process group was killed from outside. Whatever it managed to print describes a
            # run that no longer matters -- report the cancellation, not a half-finished verdict.
            return _cancelled_result(job, control)
        # Did `codna review --post` complete the job's own Check Run with its findings? Carried on
        # EVERY return path below, so an unparsable or unexpected stdout can never make process_job
        # overwrite those findings with a one-line wrapper summary.
        completed = _cli_completed_check(proc.stdout, check_run_id)
        if job.kind == "review":
            # `codna review ... --json` (codna_command's only caller of --json) prints ONE pure
            # JSON object to stdout for the caller (review_github.py's own posting logic) to
            # consume -- capture_output means it's also sitting right here in proc.stdout. The
            # generic tail-of-stdout fallback below would dump that raw JSON as the Check Run
            # summary verbatim: an unreadable wall of braces where a human expects a sentence. The
            # detailed findings already live on the "codna review" run this same subprocess
            # completes (review_github.py::summary_body) and as inline PR comments; this summary is
            # only used when the CLI did not complete the run itself.
            summary = _review_job_summary(proc.stdout)
            if summary is not None:
                return JobResult(proc.returncode == 0, summary, check_completed=completed)
        if proc.returncode != 0:
            # Every CodnaError leaves the CLI's structured `{"error": {...}}` on stderr; say its
            # message as a sentence instead of pasting the braces into the Check Run summary.
            summary = _error_json_summary(job.kind, proc.stdout, proc.stderr)
            if summary is not None:
                if _summary_is_neutral(summary):  # this sandbox cannot run the repo's tests: not a defect
                    return JobResult(True, _neutral_check_summary(summary), neutral=True, check_completed=completed)
                return JobResult(False, summary, retryable=_summary_is_retryable(summary), check_completed=completed)
        tail = (proc.stdout or "")[-1500:] + (("\n" + proc.stderr[-500:]) if proc.stderr else "")
        return JobResult(proc.returncode == 0, tail.strip() or f"{check_run_name(job.kind)} exit {proc.returncode}",
                         check_completed=completed)


def _current_pr_head(job: WebhookJob, token: str, github: Any) -> str | None:
    """The pull request's head right now, or None when the client cannot say (no helper, an API
    error). Never raises: the caller falls back to the head the event carried."""
    lookup = getattr(github, "pull_request_head_sha", None)
    if not callable(lookup) or not job.pr_number:
        return None
    try:
        head = lookup(job.repo_full_name, token, job.pr_number)
    except Exception:  # noqa: BLE001 -- a lookup hiccup must not fail the job; the event head still works
        return None
    return str(head) if head else None


def _review_head_at_start(qjob: QueuedJob, token: str, github: Any, control: JobHandle | None) -> str | None:
    """Resolve the head a review job is about to review, so its Check Run is anchored THERE.

    The review reads the pull request's CURRENT head (``review._materialize_pr`` fetches
    ``pull/N/head``), not the SHA the queued event carried; when the head moved between the event
    and this job starting -- a rebase while the queue was saturated, or a late delivery -- a check
    created on the event SHA lands on a commit that is no longer the head. The branch ruleset then
    sees the required `codna review` as ABSENT and an approved pull request stays BLOCKED
    (thyn-ai/algenta#1085: event head 7679ecbe, reviewed head 5a3ba5ff, check on 7679ecbe;
    thyn-ai/codna#569). Superseding covers the case where the NEW head's event arrives; this
    covers the case where it does not.

    When the head moved, any run this App still has ``in_progress`` on the event SHA is completed
    ``neutral`` -- left open it would strand automation waiting for "all checks complete" on a
    commit nobody is merging -- and the job's cancellation handle is retargeted so the late event
    for the resolved head does not supersede the job already reviewing it. Returns None when the
    head cannot be read: the caller keeps the event head, which is today's behaviour.
    """
    job = qjob.job
    head = _current_pr_head(job, token, github)
    if not head:
        return None
    name = check_run_name(job.kind)
    if job.ref and head != job.ref:
        _log_worker_phase(qjob, "head_moved_before_start", event_head=job.ref, head=head,
                          check_run_id=qjob.check_run_id)
        moved = (f"{name}: head moved to {head[:8]} before the review started; "
                 f"see the check on that commit.")
        if qjob.check_run_id:
            # The run the ingress opened `queued` on the event head is this row's own: complete it
            # by id (the listing sweep below is the net for anything else), and anchor a fresh run on
            # the head that is actually reviewed -- webhook_queued_check.open_check_run creates it because the row's run
            # is for another commit, and record_check_run re-points the row at the new one.
            try:
                github.update_check_run(job.repo_full_name, token, qjob.check_run_id, conclusion="neutral",
                                        summary=moved, name=name)
            except Exception:  # noqa: BLE001 -- the sweep below and the reaper's retired-run pass still see it
                pass
        complete_stale = getattr(github, "complete_stale_check_runs", None)
        if callable(complete_stale):
            try:
                complete_stale(
                    job.repo_full_name, token, head_sha=job.ref, name=name, conclusion="neutral",
                    summary=moved,
                )
            except Exception:  # noqa: BLE001 -- hygiene on the stale commit must never block this job's own run
                pass
    if control is not None:
        control.retarget(head)
    return head


def _review_head_at_end(qjob: QueuedJob, token: str, github: Any, reviewed: str) -> str | None:
    """The pull request's head if it moved away from ``reviewed`` during the turn, else None.

    The review just posted is valid evidence for ``reviewed`` and its check stays there; the new
    head needs a review of its own, which the pool queues from the returned value. Only a head the
    lookup could read counts: an unreadable head is not a moved one.
    """
    head = _current_pr_head(qjob.job, token, github)
    if not head or head == reviewed:
        return None
    _log_worker_phase(qjob, "head_moved_during_turn", reviewed_head=reviewed, head=head)
    return head


def requeue_moved_head(qjob: QueuedJob, result: JobResult, queue: Any) -> bool:
    """Queue a review of the head a review job found had moved under it. True when a row was added.

    Priority 1 within the merge-gating class, like the boot reconciler's requeue: that head is
    already waiting on a check the ruleset requires. Skipped when a job for exactly that head is
    already queued or running -- the new head's own event may have arrived meanwhile -- decided by
    the queue inside the insert's own transaction (``enqueue(unless_pending=True)``), so a delivery
    landing in between cannot make it two; the enqueue's supersede retires whatever older heads are
    still waiting. Best-effort: never raises.
    """
    head = result.requeue_head
    job = qjob.job
    if not head or job.kind != "review" or not job.pr_number or queue is None:
        return False
    enqueue = getattr(queue, "enqueue", None)
    if not callable(enqueue):
        return False
    try:
        added = bool(enqueue(
            WebhookJob("review", job.repo_full_name, ref=head, pr_number=job.pr_number,
                       installation_id=job.installation_id, reason="head_moved_during_review"),
            delivery_id=f"head-moved-{qjob.row_id}-{head}", priority=1, unless_pending=True,
        ))
    except Exception as exc:  # noqa: BLE001 -- the review that just posted must not fail over this
        _log_worker_phase(qjob, "head_moved_requeue_failed", head=head, error=type(exc).__name__)
        return False
    _log_worker_phase(qjob, "head_moved_requeued" if added else "head_moved_requeue_skipped", head=head)
    return added


def _inherit_review_for_merge_group(job: WebhookJob, token: str | None, github: Any) -> JobResult:
    """Post the PR head's completed ``codna review`` verdict as the merge group's own check.

    Fails CLOSED: when the PR head carries no completed review (a rename window, a repo where the
    check was never required), the group check is a ``failure`` that says how to get one -- a
    ``neutral`` would let an unreviewed PR through a queue that requires the review."""
    name = check_run_name("review")
    if not (token and job.ref and job.pr_number):
        return JobResult(False, f"{name} (merge group): missing token, group sha or PR number", retryable=False)
    head = github.pull_request_head_sha(job.repo_full_name, token, job.pr_number)
    prior = github.latest_completed_check_run(job.repo_full_name, token, head_sha=head, name=name) if head else None
    cid = github.create_check_run(job.repo_full_name, token, name=name, head_sha=job.ref,
                                  summary=f"{name}: inheriting the verdict of PR #{job.pr_number}")
    if prior is None:
        text = (f"{name}: no completed review found on PR #{job.pr_number}'s head"
                f"{' ' + head[:8] if head else ''}. Comment `@codna review` on the PR, then re-queue it.")
        if cid:
            github.update_check_run(job.repo_full_name, token, cid, conclusion="failure", summary=text, name=name)
        return JobResult(False, text, retryable=False)
    first_line = (prior.get("summary") or "").split("\n", 1)[0][:300]
    text = (f"{name}: inherited from PR #{job.pr_number} at {head[:8]} ({prior['conclusion']}). "
            f"Merge groups are not re-reviewed; the queued PR head is what was reviewed. {first_line}").strip()
    if cid:
        github.update_check_run(job.repo_full_name, token, cid, conclusion=prior["conclusion"], summary=text, name=name)
    return JobResult(True, text)


def process_job(
    qjob: QueuedJob,
    *,
    app_id: str | None,
    private_key: str | None,
    github: Any = webhook_github,
    runner: Callable[..., JobResult] = run_codna_job,
    resolve_engine_key: Callable[[int | None], str | None] = webhook_metering.resolve_engine_key,
    resolve_provider_credentials: Callable[
        [int | None], tuple[str | None, str | None]
    ] = webhook_metering.resolve_provider_credentials,
    resolve_fix_enabled: Callable[[int | None], bool] = webhook_metering.resolve_fix_enabled,
    on_check_run: Callable[[int], None] | None = None,
    control: JobHandle | None = None,
) -> JobResult:
    """Resolve the org credential (fail closed), mint a GitHub token, open a Check Run, run, report."""
    job = qjob.job
    token: str | None = os.environ.get("GITHUB_TOKEN")
    if job.installation_id and app_id and private_key:
        token = github.installation_token(
            app_id, private_key, job.installation_id, repo_full_name=job.repo_full_name, kind=job.kind
        )
    if control is not None:
        # Hand the cancellation handle this job's scoped token now: a canceller needs it to close
        # the Check Run, and it must not have to mint one of its own on a thread that may be wedged.
        control.attach_check_run(token, None)

    if job.kind == "queue":
        # Merge-queue group: copy the PR head's `codna review` verdict onto the group commit. No
        # metering, no CLI, no review of the group itself -- see classify_event's merge_group branch.
        return _inherit_review_for_merge_group(job, token, github)

    # A `@codna fix` review-comment reply is answered IN-THREAD on every outcome (never silent).
    ctx = job.context or {}
    reply_to = ctx.get("in_reply_to_id") if (job.kind == "fix" and isinstance(ctx.get("in_reply_to_id"), int)) else None

    # FAIL CLOSED: no per-installation org credential -> do NOT run a cloud fix on a house key.
    # Prompt the user to link their Codna account (managed allowance) or add BYOK. Never spend.
    try:
        engine_key = resolve_engine_key(job.installation_id)
    except webhook_metering.BridgeUnavailable as exc:
        # NO answer from the bridge (control plane mid-redeploy) is not "not linked": retry, and
        # on the last attempt say so honestly instead of posting the link prompt to a linked org.
        summary = f"account bridge unavailable ({exc})"
        if qjob.attempts < _MAX_ATTEMPTS:
            return JobResult(False, summary + " -- will retry", retryable=True,
                             retry_after_s=webhook_metering.BRIDGE_RETRY_WAIT_S)
        text = webhook_metering.unreachable_summary(job.kind, str(exc))
        if token and job.ref:
            webhook_queued_check.report_without_running(qjob, token, github, opening="codna: account service unreachable",
                                    conclusion="neutral", summary=text)
        if token and job.issue_number and not job.ref:
            github.post_issue_comment(job.repo_full_name, token, job.issue_number, "⚠️ " + text)
        return JobResult(False, summary + " -- gave up after retries", retryable=False)
    if not engine_key:
        if token and job.ref:
            webhook_queued_check.report_without_running(qjob, token, github, opening="codna: account not linked",
                                    conclusion="neutral", summary=webhook_metering.unlinked_summary())
        if token and job.issue_number and not job.ref:
            issue_comment_url = github.post_issue_comment(
                job.repo_full_name,
                token,
                job.issue_number,
                "⚠️ Codna can't run this request yet — this org isn't linked to a metered Codna "
                "account. " + webhook_metering.unlinked_summary(),
            )
            if not issue_comment_url:
                return JobResult(
                    False,
                    "not linked to a metered Codna account, but failed to post the link prompt issue comment",
                )
        # A comment trigger must never go silent: tell the user in-thread why nothing happened.
        if reply_to and token:
            github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to,
                                             "⚠️ Codna can't open a fix yet — this org isn't linked to a "
                                             "metered Codna account. " + webhook_metering.unlinked_summary())
        return JobResult(True, "not linked to a metered Codna account — prompted to link/authorize")

    # A convenience kill switch (an admin's "pause Codna" toggle), NOT a security boundary -- so
    # this FAILS OPEN (see webhook_metering.resolve_fix_enabled): only an explicit "off" from the
    # live bridge lands here, never a bridge hiccup or a missing field. Checked only now that we
    # know the org IS linked, so an unlinked org still gets the link prompt above, not this one.
    if not resolve_fix_enabled(job.installation_id):
        fix_disabled_summary = (
            "Codna's automatic fixes are turned off for this organization. An admin can turn "
            "them back on from the account page (Security -> Codna automation)."
        )
        if token and job.ref:
            webhook_queued_check.report_without_running(qjob, token, github, opening="codna: automation disabled",
                                    conclusion="neutral", summary=fix_disabled_summary)
        if token and job.issue_number and not job.ref:
            issue_comment_url = github.post_issue_comment(
                job.repo_full_name,
                token,
                job.issue_number,
                "⚠️ " + fix_disabled_summary,
            )
            if not issue_comment_url:
                return JobResult(
                    False,
                    "fix automation disabled for this org, but failed to post the disabled-notice issue comment",
                )
        # A comment trigger must never go silent: tell the user in-thread why nothing happened.
        if reply_to and token:
            github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to,
                                             "⚠️ " + fix_disabled_summary)
        return JobResult(True, "fix automation disabled for this org — skipped")

    # The org's own BYOK key (whichever of Anthropic/OpenAI/Google they configured), if any --
    # a resolution failure here must never block the fix: it just means no local provider key
    # gets set, same as before this existed (house-metered fixes still worked without one).
    try:
        provider, provider_key = resolve_provider_credentials(job.installation_id)
    except Exception:  # noqa: BLE001 — a BYOK-lookup hiccup must never fail the whole job
        provider, provider_key = None, None

    # Comment-triggered verified fix of a review finding: validate + enrich BEFORE spending. A skip
    # (fork PR / not a Codna finding / no target branch) replies + returns.
    if reply_to and token:
        skip = _prepare_comment_fix(job, token=token, app_id=app_id, github=github)
        if skip:
            github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to, skip)
            return JobResult(True, f"comment-fix not run: {skip[:120]}")

    # Feedback-routed fix: job.repo_full_name is the ROUTED TARGET repo (that's what the token
    # minted above is scoped to), but the report text lives on the FEEDBACK repo -- a different
    # repo under the same installation. Mint a second, feedback-scoped token to read it. A mint or
    # fetch failure here must not crash the job: run_codna_job's own guard (ctx["issue_text"]
    # missing) turns it into a clean, reportable JobResult instead.
    feedback_repo_name = ctx.get("feedback_repo")
    feedback_issue = ctx.get("feedback_issue")
    feedback_token: str | None = None
    if (job.kind == "fix" and feedback_repo_name and feedback_issue and not ctx.get("routing_error")
            and app_id and private_key and job.installation_id):
        try:
            feedback_token = github.installation_token(
                app_id, private_key, job.installation_id, repo_full_name=feedback_repo_name, kind="fix",
            )
        except Exception:  # noqa: BLE001 — a mint failure degrades to a reported fetch failure, not a crash
            feedback_token = None
        if feedback_token:
            issue_text = github.fetch_issue_text(feedback_repo_name, feedback_token, feedback_issue)
            if issue_text:
                ctx["issue_text"] = issue_text

    # Authorize a `@codna fix` reply BEFORE anything is spent or posted: the repository's collaborator
    # permission for the commenter is authoritative; the payload's author_association only decides
    # when that lookup itself is unavailable. Unauthorized requests get one polite reply and end as a
    # non-failure (nothing to retry).
    if reply_to and token:
        verdict = _comment_fix_authorized(job, token, github)
        if verdict is not None:
            if qjob.attempts <= 1:
                github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to,
                                                 "⚠️ `@codna fix` is limited to people with write access to this repository.")
            return JobResult(True, verdict, retryable=False)

    # Pre-flight the one GitHub rule that fails AFTER all the work is done: without the `workflows`
    # permission the App cannot push to a branch whose .github/workflows differ from the default
    # branch. Say so now, with the remedy, instead of analyzing, patching and then being refused.
    if reply_to and token and job.ref:
        blocked = _workflows_permission_preflight(job, token, github)
        if blocked is not None:
            if qjob.attempts <= 1:
                github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to, blocked)
            return JobResult(True, "skipped: workflows permission missing for a branch whose workflows differ from the default branch",
                             retryable=False)

    # Idempotency: reuse a PR a prior attempt already opened. Checked BEFORE the ack so a redelivery
    # doesn't post a spurious "analyzing…" then "already opened". For a comment-fix only an OPEN PR
    # counts as a live duplicate — a closed prior fix PR must not block re-requesting a fresh one.
    if job.kind == "fix" and token:
        existing = github.find_open_pr_by_marker(job.repo_full_name, token, webhook_marker(job),
                                                 open_only=bool(reply_to))
        if existing:
            if reply_to:
                github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to,
                                                 f"✅ Already opened a verified fix PR: {existing}")
            return JobResult(True, f"already opened (idempotent reuse): {existing}")

    # Only NOW (about to actually run) acknowledge in-thread — but ONLY on the first attempt, so a
    # retried job doesn't post duplicate "On it…" acks on the same finding thread.
    if reply_to and token and qjob.attempts <= 1:
        github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to, _COMMENT_FIX_ACK)

    ci_note: str | None = None
    check_id = None
    # The commit this job's Check Run is anchored to: the head the event carried -- except for a
    # review, which reads the pull request's CURRENT head and must report there (#569).
    check_sha = job.ref
    if token and job.kind == "review" and job.pr_number:
        check_sha = _review_head_at_start(qjob, token, github, control) or job.ref
    if token and check_sha:
        # The row's own run (opened `queued` by the ingress) flipped to in_progress, or a fresh one.
        check_id = webhook_queued_check.open_check_run(
            qjob, token, github, head_sha=check_sha, on_check_run=on_check_run, control=control,
            log=lambda phase, **fields: _log_worker_phase(qjob, phase, **fields))
    if job.kind == "fix" and job.reason == "check_suite_failure" and token and job.ref:
        # Before spending a metered run on a red check suite: is the failure still current, and is
        # it a code defect at all? A `bun install` tarball error or a runner that died is fixed by a
        # re-run, not a patch -- chasing one, the agent ran the repo's tests in its sandbox and
        # produced a refused test-only patch (thyn-ai/algenta#1040, 2026-09-19). webhook_ci_triage.
        verdict = webhook_ci_triage.triage(job, token, github, app_id=app_id, private_key=private_key)
        _log_worker_phase(qjob, f"ci_triage_{verdict.verdict}")
        if verdict.verdict in webhook_ci_triage.TERMINAL:
            if check_id:
                github.update_check_run(job.repo_full_name, token, check_id, conclusion="neutral",
                                        summary=verdict.summary, name=check_run_name(job.kind))
            return JobResult(True, f"ci triage: {verdict.verdict} -- nothing to fix on this commit")
        if verdict.issue_text and isinstance(job.context, dict):
            job.context["ci_failure"] = verdict.issue_text  # codna_command hands it to the agent as --issue
        ci_note = verdict.note
    _log_worker_phase(qjob, "runner_start")  # everything before this is token/metering/prepare/ack
    try:
        identity_fn = getattr(github, "app_bot_identity", None)
        git_identity = identity_fn(app_id, private_key) if callable(identity_fn) else None
        result = runner(job, token=token, engine_key=engine_key, provider=provider, provider_key=provider_key,
                        git_identity=git_identity, check_run_id=check_id, control=control,
                        check_head_sha=check_sha if check_id else None)
    except Exception as exc:  # noqa: BLE001 — a check-run this function opened must always be
        # completed, even on a crash the runner itself didn't turn into a JobResult (run_codna_job
        # only catches subprocess.TimeoutExpired). Without this, check_id is created above but
        # never reaches update_check_run below, leaving it permanently "queued" on the PR/commit —
        # the exact bug behind codna-app-smoke's check-suite, stuck since 2026-07-08. Never surface
        # the raw exception text here (could embed argv/path/token content); the class name is safe.
        result = JobResult(False, f"{check_run_name(job.kind)} crashed unexpectedly ({type(exc).__name__})")
    _log_worker_phase(qjob, "runner_done")
    if result.cancelled and control is not None and check_id and not control.check_closed:
        # No canceller has CONFIRMED this run closed -- its settle thread may not have run yet, or
        # GitHub refused it and the failure was swallowed. Close it here, through the same
        # never-clobber helper, so a cancelled job can never leave a required check `in_progress`.
        webhook_control.close_check_run(control, summary=result.summary, github=github)
    moved_note: str | None = None
    if job.kind == "review" and token and check_sha and not result.cancelled:
        # Did the head move while the review ran? The check on the reviewed commit stands -- the
        # review is evidence for THAT commit -- and the new head gets a review of its own, queued
        # by the pool from ``requeue_head``. Never an approval carried onto a head nobody reviewed:
        # the CLI's own post-time head check downgrades one (review_github.post_review). A cancelled
        # job is left alone: whatever superseded it is already driving the new head.
        moved_to = _review_head_at_end(qjob, token, github, check_sha)
        if moved_to:
            result = replace(result, requeue_head=moved_to)
            moved_note = (f"The pull request head has since moved to {moved_to[:8]}; that head is queued "
                          f"for its own {check_run_name(job.kind)} run.")
    if check_id and token and not result.check_completed:
        # A review's CLI already completed this run with its findings (see _job_env): overwriting it
        # here would replace the findings with a one-line wrapper summary.
        github.update_check_run(
            job.repo_full_name, token, check_id,
            conclusion=result.conclusion,
            summary=result.summary + "".join(f"\n\n{note}" for note in (ci_note, moved_note) if note),
            name=check_run_name(job.kind),
        )
    # An issue-label fix has neither a commit (no check_id: job.ref is never set for this trigger)
    # nor a comment thread (no reply_to) to report through, so the issue itself is the ONLY place it
    # can speak. Report BOTH outcomes here.
    #
    # Success used to be silent, on the reasoning that "a success speaks for itself via the opened
    # PR". It does not: the PR body carried only the `codna-webhook-id:` marker, which is not a
    # GitHub reference, so nothing ever cross-linked the PR to the issue. Observed live -- codna
    # opened a correct fix PR and the issue's newest comment was still an OLD failure, so the issue
    # looked broken while the work was done. Now the issue always gets the outcome, with a link.
    if not check_id and not reply_to and token and job.issue_number:
        if result.cancelled:
            # ok=True (nothing went wrong with the code) but NOTHING was opened: the success text
            # below would send the user hunting for a pull request that does not exist.
            github.post_issue_comment(job.repo_full_name, token, job.issue_number,
                                      f"⚠️ Codna stopped this {job.kind}: {result.summary}")
        elif ctx.get("routing_error"):
            # An unrouted report on the feedback repo itself: job.repo_full_name IS the feedback
            # repo here (there was no target to route to), so this posts correctly with the SAME
            # token already minted above -- no feedback-specific plumbing needed for this branch.
            github.post_issue_comment(
                job.repo_full_name, token, job.issue_number,
                f"⚠️ Codna couldn't route this report automatically: {ctx['routing_error']}. "
                "Add the right `product:<name>` label and re-apply `codna-fix` to retry.",
            )
        # `job.kind == "fix"` is required for the SUCCESS half: the `codna-secure` label produces an
        # issue-triggered job with no ref and no thread too, but it is a READ-ONLY classification pass
        # that never opens a PR (codna_command builds `codna secure` without --open-pr). Without this
        # term, a clean secure run posted "✅ ... opened a pull request for it." and sent the user
        # hunting for a PR that does not exist -- the same kind of false statement this whole change
        # exists to remove.
        elif result.ok and job.kind == "fix":
            pr_url = None
            try:
                pr_url = github.find_open_pr_by_marker(job.repo_full_name, token, webhook_marker(job))
            except Exception:  # noqa: BLE001 — never turn a successful fix into a failure over a lookup
                pr_url = None
            github.post_issue_comment(
                job.repo_full_name, token, job.issue_number,
                (f"✅ Codna opened a fix for this issue: {pr_url}" if pr_url
                 else "✅ Codna finished this fix and opened a pull request for it."),
            )
        elif result.ok:
            # Non-fix issue trigger (today: `codna-secure`). Say what actually happened.
            github.post_issue_comment(
                job.repo_full_name, token, job.issue_number,
                f"✅ Codna finished the {job.kind} pass for this issue.",
            )
        else:
            github.post_issue_comment(
                job.repo_full_name, token, job.issue_number,
                f"⚠️ Codna couldn't finish this fix: {result.summary}",
            )
    # Feedback-routed fix that DID have a target: job.issue_number is None (there is no issue in the
    # TARGET repo), so the generic per-issue block above never fires for it -- report back to the
    # PUBLIC issue instead, using the feedback-scoped token (the job's own `token` is scoped only to
    # the target repo and cannot post there).
    if feedback_repo_name and feedback_issue and not ctx.get("routing_error"):
        comment_token = feedback_token or token
        if comment_token:
            if result.ok:
                pr_url = None
                try:
                    # The PR, if any, is in the TARGET repo -- look it up with the token scoped
                    # there, never with the feedback-scoped token (which cannot see it).
                    pr_url = github.find_open_pr_by_marker(job.repo_full_name, token, webhook_marker(job))
                except Exception:  # noqa: BLE001 — never turn a successful fix into a failure over a lookup
                    pr_url = None
                msg = (f"✅ Codna opened a fix in {job.repo_full_name}: {pr_url}" if pr_url
                       else f"✅ Codna finished this fix in {job.repo_full_name} and opened a pull request for it.")
            else:
                msg = f"⚠️ Codna couldn't finish this fix in {job.repo_full_name}: {result.summary}"
            github.post_issue_comment(feedback_repo_name, comment_token, feedback_issue, msg)
    # Reply the outcome ONCE: on success immediately, on failure only when the job is out of retries
    # (else a transient-then-succeeding fix would leave a spurious fail-closed reply, and every failed
    # attempt would spam the thread).
    # ... or when the failure is deterministic (retryable=False): there will be no further attempt,
    # so this IS the final one -- the 2026-09-17 clone-timeout fix ended silent because it was
    # terminal on attempt 1 of 3 and this condition only looked at the attempt counter.
    if reply_to and token and (result.ok or qjob.attempts >= _MAX_ATTEMPTS or not result.retryable):
        body = (f"⚠️ Codna stopped this fix: {result.summary}. Nothing was pushed."
                if result.cancelled else _comment_fix_result_msg(job, token, result, github))
        github.post_review_comment_reply(job.repo_full_name, token, job.pr_number, reply_to, body)
    return result


def _job_log_context(qjob: QueuedJob) -> dict[str, object]:
    job = qjob.job
    return {
        "service": "codna-webhook-worker",
        "event": "job_failed",
        "row_id": qjob.row_id,
        "delivery_id": qjob.delivery_id,
        "attempt": qjob.attempts,
        "kind": job.kind,
        "repo": job.repo_full_name,
        "ref": job.ref,
        "installation_id": job.installation_id,
        "reason": job.reason,
    }


def _log_worker_failure(qjob: QueuedJob, *, code: str, message: str) -> None:
    payload = _job_log_context(qjob)
    payload["code"] = code
    payload["message"] = message[:2000]
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _log_worker_phase(qjob: QueuedJob, phase: str, **fields: object) -> None:
    """One stderr JSON line per job phase, so a stalled job is locatable from `fly logs`.

    Until now a job only ever logged its FAILURE. A job that wedged (or was killed) between claim
    and its first visible action left no trace at all: on 2026-09-17 two `@codna fix` jobs sat
    silent for hours with `/ready` green, and there was nothing to grep for. `event` differs from
    the failure line's so the two never get confused in a filter. ``fields`` are extra keys a
    phase wants on record (the two heads of a move, for one).
    """
    payload = _job_log_context(qjob)
    payload.update(fields)
    payload["event"] = "job_phase"
    payload["phase"] = phase
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _log_worker_loop_error(exc: BaseException, qjob: QueuedJob | None = None) -> None:
    """A worker thread that dies takes half the pool with it and nothing restarts it: log and go on."""
    payload = _job_log_context(qjob) if qjob is not None else {"service": "codna-webhook-worker"}
    payload["event"] = "worker_loop_error"
    payload["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def __getattr__(name: str) -> Any:
    """Expose ``WorkerPool`` here although it now lives in :mod:`codna.webhook_pool`.

    The pool reads this module's patchable names (``get_admitter``, ``_JOB_TIMEOUT_S``,
    ``process_job``) at call time, so it must import this module -- which means this module cannot
    import it at module level. Resolving the attribute on first access instead keeps every existing
    ``from codna.webhook_worker import WorkerPool`` working with no cycle.
    """
    if name == "WorkerPool":
        from .webhook_pool import WorkerPool

        return WorkerPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
