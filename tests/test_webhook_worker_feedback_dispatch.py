"""process_job / run_codna_job for a feedback-routed fix: a second, feedback-repo-scoped token
mint + issue-text fetch before running, and outcome reporting back to the PUBLIC issue (not
job.issue_number, which is None for a routed job -- the issue lives on a different repo)."""
from __future__ import annotations

import codna.webhook_worker as worker_module
from codna.webhook import WebhookJob
from codna.webhook_queue import QueuedJob
from codna.webhook_worker import JobResult, process_job, run_codna_job


def _feedback_ctx(*, issue: int = 5, routed: bool = True) -> dict:
    ctx = {"feedback_repo": "thyn-ai/feedback", "feedback_issue": issue, "fp": f"feedback#{issue}"}
    if not routed:
        ctx["routing_error"] = "no `product:<name>` label was found on this report"
    return ctx


def _qjob(job: WebhookJob, attempts: int = 1) -> QueuedJob:
    return QueuedJob(row_id=1, delivery_id="d1", attempts=attempts, job=job)


class _FakeGitHub:
    """Records which repo each token/comment/fetch call targeted -- the property that matters
    here is that the TARGET-repo operations and the FEEDBACK-repo operations never cross tokens."""

    def __init__(self, *, existing_pr=None, issue_text="steps to reproduce", token_mint_fails=False,
                pr_opens_on_run=False):
        self.calls: list[tuple] = []
        self._existing_pr = existing_pr
        self._issue_text = issue_text
        self._token_mint_fails = token_mint_fails
        # Models reality: before the runner has actually run, the idempotency pre-check finds
        # nothing; only once the (fake) run has "opened" a PR does the post-run outcome lookup
        # find it. Both call sites use the SAME find_open_pr_by_marker, so a single always-on
        # existing_pr would trip the pre-run idempotency short-circuit and the runner would never
        # be called at all -- this flag lets a test model "the PR appears only after running".
        self._pr_opens_on_run = pr_opens_on_run
        self._find_pr_calls = 0

    def installation_token(self, app_id, private_key, installation_id, *, repo_full_name, kind):
        self.calls.append(("token", repo_full_name, kind))
        if self._token_mint_fails and repo_full_name == "thyn-ai/feedback":
            from codna.webhook import WebhookError

            raise WebhookError("installation_token_failed", "mint failed")
        return f"scoped-token-for-{repo_full_name}"

    def fetch_issue_text(self, repo_full_name, token, issue_number):
        self.calls.append(("fetch_issue_text", repo_full_name, issue_number, token))
        return self._issue_text

    def find_open_pr_by_marker(self, repo_full_name, token, marker, *, open_only=False):
        self.calls.append(("find_pr", repo_full_name, token))
        self._find_pr_calls += 1
        if self._pr_opens_on_run:
            return self._existing_pr if self._find_pr_calls > 1 else None
        return self._existing_pr

    def create_check_run(self, repo, token, *, name, head_sha, summary):
        self.calls.append(("create_check", repo))
        return None  # feedback jobs have no ref -> never actually reached

    def update_check_run(self, *a, **k):
        self.calls.append(("update_check",))

    def post_issue_comment(self, repo, token, issue_number, body):
        self.calls.append(("issue_comment", repo, issue_number, token, body[:80]))
        return "https://github.com/x/issues/1#issuecomment-1"


def _resolve_engine_key(_installation_id):
    return "org-codna-key"


def _resolve_provider_credentials(_installation_id):
    return (None, None)


def _resolve_fix_enabled(_installation_id):
    return True


def _process(job, *, github, runner, attempts=1):
    return process_job(
        _qjob(job, attempts=attempts), app_id="app-1", private_key="pem", github=github, runner=runner,
        resolve_engine_key=_resolve_engine_key, resolve_provider_credentials=_resolve_provider_credentials,
        resolve_fix_enabled=_resolve_fix_enabled,
    )


# ── run_codna_job: the routing-error short-circuit ─────────────────────────────────────────────
def test_run_codna_job_routing_error_returns_ok_true_without_running_codna(monkeypatch):
    """A correctly-handled non-run (same convention as 'org not linked' / 'automation disabled'),
    not a retryable failure -- it must not shell out to codna at all."""
    def _boom(*a, **k):
        raise AssertionError("must not invoke codna for a routing error")

    monkeypatch.setattr(worker_module, "_run_job_process", _boom)
    job = WebhookJob("fix", "thyn-ai/feedback", context=_feedback_ctx(routed=False))
    result = run_codna_job(job, token="t", engine_key="k")
    assert result.ok is True
    assert "product:<name>" in result.summary


# ── run_codna_job: the feedback issue-text guard ────────────────────────────────────────────────
def test_run_codna_job_feedback_fix_fails_closed_when_issue_text_is_missing():
    """ctx["issue_text"] is set by process_job BEFORE calling the runner; if that fetch failed,
    run_codna_job must report it, not silently build a doomed `codna fix` with no --issue."""
    job = WebhookJob("fix", "thyn-ai/codna", context=_feedback_ctx())  # no ctx["issue_text"]
    result = run_codna_job(job, token="scoped-token-for-thyn-ai/codna", engine_key="k")
    assert result.ok is False
    assert "report text" in result.summary


def test_run_codna_job_feedback_fix_uses_the_pre_fetched_issue_text(monkeypatch):
    captured = {}

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return _P()

    monkeypatch.setattr(worker_module, "_run_job_process", _fake_run)
    ctx = _feedback_ctx()
    ctx["issue_text"] = "the report says the CLI hangs on a monorepo"
    job = WebhookJob("fix", "thyn-ai/codna", context=ctx)
    result = run_codna_job(job, token="t", engine_key="k")

    assert result.ok is True
    assert "the report says the CLI hangs on a monorepo" in captured["argv"]


# ── process_job: the second, feedback-scoped token mint + issue-text fetch ─────────────────────
def test_process_job_mints_a_second_token_scoped_to_the_feedback_repo():
    github = _FakeGitHub()
    job = WebhookJob("fix", "thyn-ai/codna", installation_id=7, context=_feedback_ctx())

    def _runner(job, **kw):
        # By the time the runner is called, process_job must have already fetched the text.
        assert job.context.get("issue_text") == "steps to reproduce"
        return JobResult(True, "fixed")

    _process(job, github=github, runner=_runner)

    token_calls = [c for c in github.calls if c[0] == "token"]
    assert ("token", "thyn-ai/codna", "fix") in token_calls   # target-repo token (for the fix/PR)
    assert ("token", "thyn-ai/feedback", "fix") in token_calls  # feedback-repo token (to read the report)
    fetch_calls = [c for c in github.calls if c[0] == "fetch_issue_text"]
    assert fetch_calls == [("fetch_issue_text", "thyn-ai/feedback", 5, "scoped-token-for-thyn-ai/feedback")]


def test_process_job_survives_a_feedback_token_mint_failure(monkeypatch):
    """A mint failure for the SECOND token must not crash the job -- run_codna_job's own guard
    turns the resulting missing issue_text into a clean, reportable JobResult instead."""
    github = _FakeGitHub(token_mint_fails=True)
    job = WebhookJob("fix", "thyn-ai/codna", installation_id=7, context=_feedback_ctx())
    result = _process(job, github=github, runner=run_codna_job)
    assert result.ok is False
    assert "report text" in result.summary


# ── process_job: outcome reporting goes to the PUBLIC issue, not job.issue_number ───────────────
def test_process_job_success_comments_on_the_feedback_issue_with_the_pr_link():
    github = _FakeGitHub(existing_pr="https://github.com/thyn-ai/codna/pull/99", pr_opens_on_run=True)
    job = WebhookJob("fix", "thyn-ai/codna", installation_id=7, context=_feedback_ctx())

    result = _process(job, github=github, runner=lambda job, **kw: JobResult(True, "fixed"))
    assert result.ok is True

    comments = [c for c in github.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    _, repo, issue_number, token, body = comments[0]
    assert repo == "thyn-ai/feedback"          # NOT job.repo_full_name ("thyn-ai/codna")
    assert issue_number == 5                    # the FEEDBACK issue, not job.issue_number (None)
    assert token == "scoped-token-for-thyn-ai/feedback"   # feedback-scoped, not the target-repo one
    assert "thyn-ai/codna" in body and "pull/99" in body

    # Both the idempotency pre-check and the outcome lookup use the TARGET-repo token -- the
    # feedback token can't see PRs on the target repo.
    find_calls = [c for c in github.calls if c[0] == "find_pr"]
    assert find_calls == [("find_pr", "thyn-ai/codna", "scoped-token-for-thyn-ai/codna")] * 2


def test_process_job_failure_comments_the_failure_reason_on_the_feedback_issue():
    github = _FakeGitHub()
    job = WebhookJob("fix", "thyn-ai/codna", installation_id=7, context=_feedback_ctx())

    result = _process(job, github=github, runner=lambda job, **kw: JobResult(False, "tests failed"))
    assert result.ok is False

    comments = [c for c in github.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    _, repo, issue_number, _token, body = comments[0]
    assert repo == "thyn-ai/feedback" and issue_number == 5
    assert "tests failed" in body


def test_unrouted_report_is_reported_on_the_feedback_repo_via_the_existing_generic_path():
    """The unrouted case sets job.repo_full_name = the feedback repo itself and job.issue_number =
    the feedback issue, so the PRE-EXISTING generic per-issue comment block handles it -- this
    confirms that reuse actually works, and that the message names the reason."""
    github = _FakeGitHub()
    job = WebhookJob("fix", "thyn-ai/feedback", installation_id=7, issue_number=5,
                     context=_feedback_ctx(routed=False))

    result = _process(job, github=github, runner=run_codna_job)
    assert result.ok is True

    comments = [c for c in github.calls if c[0] == "issue_comment"]
    assert len(comments) == 1
    _, repo, issue_number, _token, body = comments[0]
    assert repo == "thyn-ai/feedback" and issue_number == 5
    assert "couldn't route this report automatically" in body
    assert "product:<name>" in body

    # No feedback-repo second mint for the unrouted case -- job.repo_full_name IS already the
    # feedback repo, so the single token minted for it is enough.
    token_calls = [c for c in github.calls if c[0] == "token"]
    assert len(token_calls) == 1


def test_routed_job_never_touches_the_generic_issue_number_comment_path():
    """job.issue_number is None for a routed job -- confirms the generic per-issue block (which
    would otherwise try to comment on issue #None in the TARGET repo) never fires for it."""
    github = _FakeGitHub()
    job = WebhookJob("fix", "thyn-ai/codna", installation_id=7, context=_feedback_ctx())
    assert job.issue_number is None

    _process(job, github=github, runner=lambda job, **kw: JobResult(True, "fixed"))

    comments = [c for c in github.calls if c[0] == "issue_comment"]
    # Exactly one comment (on the feedback issue) -- not two, which would happen if BOTH the
    # generic block and the feedback-specific block fired.
    assert len(comments) == 1
    assert comments[0][1] == "thyn-ai/feedback"
