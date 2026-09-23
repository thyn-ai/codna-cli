"""Cross-repo dispatch: a `codna-fix` label on the public feedback repo routes by the issue's own
`product:<name>` label to that product's real (often private) repo.

Pure-core tests only (classify_event / codna_command) — no network, matching test_webhook.py's
conventions. process_job/run_codna_job's two-token-mint + outcome-comment behavior is covered
separately in test_webhook_worker_feedback_dispatch.py, since that needs the fake GitHub client.
"""
from __future__ import annotations

import pytest

from codna.webhook import FIX_LABEL, WebhookError, WebhookJob, classify_event, codna_command, webhook_marker


def _feedback_env(monkeypatch, routes: dict[str, str] | None = None) -> None:
    monkeypatch.setenv("CODNA_FEEDBACK_REPO", "thyn-ai/feedback")
    import json

    monkeypatch.setenv("CODNA_FEEDBACK_ROUTES", json.dumps(routes if routes is not None else {
        "algenta": "thyn-ai/algenta", "codna": "thyn-ai/codna",
    }))


def _labeled_payload(*, number: int, product_labels: list[str], installation: int = 7) -> dict:
    return {
        "action": "labeled",
        "repository": {"full_name": "thyn-ai/feedback"},
        "installation": {"id": installation},
        "label": {"name": FIX_LABEL},
        "issue": {"number": number, "labels": [{"name": n} for n in product_labels]},
    }


# ── routing: the successful case ────────────────────────────────────────────────────────────────
def test_routes_to_the_target_repo_named_by_the_product_label(monkeypatch):
    _feedback_env(monkeypatch)
    job = classify_event("issues", _labeled_payload(number=42, product_labels=["needs-triage", "product:codna"]))

    assert job.kind == "fix"
    assert job.repo_full_name == "thyn-ai/codna"     # the ROUTED target, not the feedback repo
    assert job.installation_id == 7
    assert job.issue_number is None                  # no issue in the target repo -- context carries it
    assert job.context["feedback_repo"] == "thyn-ai/feedback"
    assert job.context["feedback_issue"] == 42
    assert job.context["product"] == "codna"
    assert "routing_error" not in job.context
    assert job.reason == "feedback_routed_fix"


def test_the_marker_is_unique_per_feedback_issue():
    """REGRESSION-shaped: a ref-less, issue-number-less job would otherwise collapse to the literal
    marker "default" (webhook_marker's fallback), and a second unrelated feedback report would be
    silently treated as "already opened (idempotent reuse)" of the FIRST one's PR."""
    j1 = WebhookJob("fix", "thyn-ai/codna", context={"feedback_repo": "thyn-ai/feedback",
                                                     "feedback_issue": 1, "fp": "feedback#1"})
    j2 = WebhookJob("fix", "thyn-ai/codna", context={"feedback_repo": "thyn-ai/feedback",
                                                     "feedback_issue": 2, "fp": "feedback#2"})
    assert webhook_marker(j1) != webhook_marker(j2)
    assert "feedback#1" in webhook_marker(j1)
    assert "feedback#2" in webhook_marker(j2)


# ── routing: the unrouted / negative cases ──────────────────────────────────────────────────────
def test_unrouted_when_no_product_label_is_present(monkeypatch):
    _feedback_env(monkeypatch)
    job = classify_event("issues", _labeled_payload(number=9, product_labels=["needs-triage"]))

    assert job.repo_full_name == "thyn-ai/feedback"   # stays on the feedback repo -- there's no target
    assert job.issue_number == 9                      # so the generic issue-comment path can report it
    assert "no `product:<name>` label" in job.context["routing_error"]
    assert job.reason == "feedback_unrouted"


def test_unrouted_when_the_products_route_is_not_configured(monkeypatch):
    _feedback_env(monkeypatch, routes={"algenta": "thyn-ai/algenta"})  # no "sqai" entry
    job = classify_event("issues", _labeled_payload(number=9, product_labels=["product:sqai"]))

    assert job.repo_full_name == "thyn-ai/feedback"
    assert "no route is configured for product `sqai`" in job.context["routing_error"]


def test_a_malformed_routes_env_degrades_to_unrouted_not_a_crash(monkeypatch):
    monkeypatch.setenv("CODNA_FEEDBACK_REPO", "thyn-ai/feedback")
    monkeypatch.setenv("CODNA_FEEDBACK_ROUTES", "{not valid json")
    job = classify_event("issues", _labeled_payload(number=1, product_labels=["product:codna"]))
    assert job.context["routing_error"]


def test_feedback_dispatch_is_off_when_codna_feedback_repo_is_unset(monkeypatch):
    """No CODNA_FEEDBACK_REPO configured -> this is just an ordinary codna-fix label on whatever
    repo it was applied to, exactly as it worked before this feature existed."""
    monkeypatch.delenv("CODNA_FEEDBACK_REPO", raising=False)
    payload = _labeled_payload(number=9, product_labels=["product:codna"])
    payload["repository"]["full_name"] = "thyn-ai/feedback"  # even ON what WOULD be the feedback repo
    job = classify_event("issues", payload)
    assert job.reason == "labeled_codna_fix"          # ordinary label path, not feedback dispatch
    assert job.context is None


def test_a_report_on_a_DIFFERENT_repo_never_triggers_feedback_dispatch(monkeypatch):
    """CODNA_FEEDBACK_REPO is configured, but this label event is on some OTHER repo -- must take
    the ordinary label path, not be mistaken for a feedback report just because routing is on."""
    _feedback_env(monkeypatch)
    payload = _labeled_payload(number=9, product_labels=["product:codna"])
    payload["repository"]["full_name"] = "acme/unrelated-repo"
    job = classify_event("issues", payload)
    assert job.repo_full_name == "acme/unrelated-repo"
    assert job.reason == "labeled_codna_fix"
    assert job.context is None


# ── codna_command: the discriminator between comment-fix and feedback-routed shapes ─────────────
def test_feedback_routed_job_uses_the_plain_issue_shape_not_comment_fix():
    """REGRESSION-shaped: `if ctx:` alone would force-fit a feedback-context job into the
    comment-fix branch (which requires head_ref -- a PR branch that doesn't exist here) and raise
    fix_requires_head_ref instead of building a normal --issue command."""
    job = WebhookJob("fix", "thyn-ai/codna",
                     context={"feedback_repo": "thyn-ai/feedback", "feedback_issue": 5, "fp": "feedback#5"})
    cmd = codna_command(job, issue_text="codna report: something is broken")
    assert cmd[:4] == ["codna", "fix", "https://github.com/thyn-ai/codna.git", "--open-pr"]
    assert cmd[cmd.index("--issue") + 1] == "codna report: something is broken"
    assert "--base-branch" not in cmd and "--ref" not in cmd


def test_feedback_routed_job_without_issue_text_raises_a_catchable_error():
    job = WebhookJob("fix", "thyn-ai/codna",
                     context={"feedback_repo": "thyn-ai/feedback", "feedback_issue": 5, "fp": "feedback#5"})
    with pytest.raises(WebhookError) as exc:
        codna_command(job)  # no issue_text supplied
    assert exc.value.code == "fix_requires_issue_text"


def test_feedback_routed_pr_body_cross_links_the_public_issue_not_a_local_one():
    job = WebhookJob("fix", "thyn-ai/codna",
                     context={"feedback_repo": "thyn-ai/feedback", "feedback_issue": 5, "fp": "feedback#5"})
    cmd = codna_command(job, issue_text="x")
    body = cmd[cmd.index("--pr-body") + 1]
    assert "thyn-ai/feedback#5" in body
    assert webhook_marker(job) in body


def test_comment_fix_shape_is_unaffected_by_the_new_discriminator():
    """The existing comment-fix path (identified by in_reply_to_id) must still work exactly as
    before -- the feedback-routing addition must not have narrowed what counts as a comment-fix."""
    ctx = {"in_reply_to_id": 111, "head_sha": "deadbeef", "head_ref": "feature", "title": "SQL injection"}
    job = WebhookJob("fix", "acme/app", ref="deadbeef", pr_number=7, context=ctx)
    cmd = codna_command(job)
    assert cmd[:3] == ["codna", "fix", "https://github.com/acme/app.git"]
    assert "--base-branch" in cmd and cmd[cmd.index("--base-branch") + 1] == "feature"


def test_comment_fix_without_head_ref_still_raises_fix_requires_head_ref():
    """REGRESSION-shaped: confirms the new `ctx.get("in_reply_to_id")` discriminator still reaches
    the head_ref check for a genuine (if malformed) comment-fix context."""
    job = WebhookJob("fix", "acme/app", context={"in_reply_to_id": 111})  # no head_ref
    with pytest.raises(WebhookError) as exc:
        codna_command(job)
    assert exc.value.code == "fix_requires_head_ref"


# ── E2E signed ingress: full HTTP → classify → route path ───────────────────────────────────────
def test_signed_ingress_routes_feedback_codna_fix_label_to_thyn_ai_codna(monkeypatch, tmp_path):
    """E2E: a signed `issues.labeled` delivery on thyn-ai/feedback with `codna-fix` and
    `product:codna` must be accepted (202), enqueued, and the queued job must target
    thyn-ai/codna — not the feedback repo itself.

    This covers the real live scenario described in the Phase 3 cross-repo dispatch verification
    report: label a report on the public intake repo, the webhook routes it to thyn-ai/codna.
    """
    import hashlib
    import hmac
    import json
    import threading
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer

    from codna.webhook import FIX_LABEL, _Handler
    from codna.webhook_queue import WebhookQueue
    from codna.webhook_worker import JobResult, WorkerPool

    _feedback_env(monkeypatch)
    secret = "e2e-test-secret"
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)

    queue = WebhookQueue(tmp_path / "q.db")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.queue = queue  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    payload = json.dumps({
        "action": "labeled",
        "repository": {"full_name": "thyn-ai/feedback"},
        "installation": {"id": 7},
        "label": {"name": FIX_LABEL},
        "issue": {"number": 42, "labels": [{"name": "needs-triage"}, {"name": "product:codna"}]},
    }).encode()
    sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhooks/github", data=payload, method="POST",
            headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig,
                     "X-GitHub-Delivery": "e2e-feedback-codna-1",
                     "Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=5)
        body = json.loads(resp.read())

        assert resp.status == 202
        assert body["status"] == "accepted"
        assert body["kind"] == "fix"
        assert body["repo"] == "thyn-ai/codna"   # routed to the product repo, NOT feedback
        assert body["reason"] == "feedback_routed_fix"
        assert queue.counts().get("queued") == 1

        # Drain the queue and confirm the job targets thyn-ai/codna with the right context.
        processed_jobs = []
        pool = WorkerPool(
            queue, concurrency=1, poll_interval=0.02,
            process=lambda qj: (processed_jobs.append(qj.job) or JobResult(True, "ok")),
        )
        pool.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and queue.counts().get("done", 0) < 1:
            time.sleep(0.05)
        pool.stop()

        assert len(processed_jobs) == 1
        job = processed_jobs[0]
        assert job.kind == "fix"
        assert job.repo_full_name == "thyn-ai/codna"
        assert job.issue_number is None                      # issue lives on the feedback repo
        assert (job.context or {}).get("feedback_repo") == "thyn-ai/feedback"
        assert (job.context or {}).get("feedback_issue") == 42
        assert (job.context or {}).get("product") == "codna"
        assert "routing_error" not in (job.context or {})
        assert queue.counts().get("done") == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
