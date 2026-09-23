"""Codna GitHub App webhook — the App channel, owned in Codna (not the Algenta SDK/engine).

Unit-tests the stdlib-only pure core: signature verification, event classification into
Codna jobs (fix + security-autofix), and codna-argv building.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from codna.webhook import (
    FIX_LABEL,
    SECURE_LABEL,
    WebhookError,
    WebhookJob,
    classify_event,
    codna_command,
    verify_signature,
)


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_signature_accepts_valid_and_rejects_tampered():
    secret, body = "s3cr3t", json.dumps({"a": 1}).encode()
    assert verify_signature(secret, body, _sign(secret, body)) is True
    # tampered body
    assert verify_signature(secret, body + b"x", _sign(secret, body)) is False
    # wrong secret
    assert verify_signature("other", body, _sign(secret, body)) is False


def test_verify_signature_rejects_missing_or_malformed_header():
    secret, body = "s3cr3t", b"{}"
    assert verify_signature(secret, body, None) is False
    assert verify_signature(secret, body, "") is False
    assert verify_signature(secret, body, "sha1=deadbeef") is False  # wrong algo prefix
    assert verify_signature("", body, _sign("", body)) is False       # no secret configured


def _check_suite_payload(conclusion="failure", pull_requests=None, app=None, check_runs=3):
    return {
        "action": "completed",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 42},
        "check_suite": {
            "conclusion": conclusion,
            "head_sha": "abc123",
            "pull_requests": pull_requests if pull_requests is not None else [],
            "app": app if app is not None else {"slug": "github-actions", "id": 15368},
            "latest_check_runs_count": check_runs,
        },
    }


def _pr(number=7, ref="feature/login", sha="abc123"):
    return {"number": number, "head": {"ref": ref, "sha": sha}}


def test_classify_check_suite_on_a_push_is_never_a_trigger():
    """The original objection, still enforced: a red check with no associated pull request is a
    statement about the repo's CI health, not about a change someone proposed. It catches
    infra/outage flakes that resolve on their own retry, so it must stay ignored."""
    assert classify_event("check_suite", _check_suite_payload()) is None
    assert classify_event("check_suite", _check_suite_payload(conclusion="success")) is None


@pytest.mark.parametrize("conclusion", ["success", "cancelled", "timed_out", "stale", "neutral", "skipped", ""])
def test_classify_check_suite_only_reacts_to_an_actual_failure(conclusion):
    """Anything other than `failure` is not evidence the code is broken. Measured on one repo in a
    single day: of 25 suites, 19 succeeded, 5 were CANCELLED by a concurrency group and 1 really
    failed -- so treating `cancelled` as fixable would burn metered runs on scheduling noise."""
    assert classify_event("check_suite", _check_suite_payload(conclusion, [_pr()])) is None


def test_classify_check_suite_failure_on_a_pull_request_is_a_fix_job():
    """The case the objection does not cover: the PR is the explicit human action, and the failure
    is scoped to that person's proposed change."""
    job = classify_event("check_suite", _check_suite_payload("failure", [_pr()]))
    assert job is not None
    assert job.kind == "fix"
    assert job.repo_full_name == "acme/app"
    assert job.pr_number == 7
    assert job.ref == "abc123"
    assert job.installation_id == 42
    assert job.reason == "check_suite_failure"
    # head_ref must travel so the fix stacks onto the PR's own branch.
    assert (job.context or {}).get("head_ref") == "feature/login"


def test_classify_check_suite_ignores_codnas_own_check_suite():
    """REGRESSION (2026-09-18, thyn-ai/telys#123): GitHub completed the App's OWN check suite as
    `failure` with zero check runs, two seconds before the review's first run existed, and the App
    enqueued a fix for it -- a metered run to fix a Codna job, not the PR's code. A failed
    `codna fix` run had the same effect on other PRs. Our own suite is never a CI failure."""
    payload = _check_suite_payload("failure", [_pr()], app={"slug": "codna-ai", "id": 4061960})
    assert classify_event("check_suite", payload) is None


def test_classify_check_suite_ignores_a_self_hosted_installs_own_app_by_id(monkeypatch):
    """A self-hosted App runs under another slug; GITHUB_APP_ID identifies it instead."""
    monkeypatch.setenv("GITHUB_APP_ID", "777")
    own = _check_suite_payload("failure", [_pr()], app={"slug": "acme-codna", "id": 777})
    other = _check_suite_payload("failure", [_pr()], app={"slug": "acme-ci", "id": 778})
    assert classify_event("check_suite", own) is None
    assert classify_event("check_suite", other) is not None


def test_classify_check_suite_with_no_check_runs_is_not_fixable():
    """A suite that 'failed' without a single check run has nothing to fix."""
    payload = _check_suite_payload("failure", [_pr()], check_runs=0)
    assert classify_event("check_suite", payload) is None


def test_classify_check_suite_ignores_codnas_own_fix_prs():
    """Codna pushes fix branches as `codna/<plan>`. Their CI runs like any other, so without this
    a red fix PR triggers a fix OF THE FIX, and so on -- an unbounded loop that spends a metered
    run and opens a PR every cycle."""
    payload = _check_suite_payload("failure", [_pr(ref="codna/abc123def456")])
    assert classify_event("check_suite", payload) is None


def test_classify_check_suite_skips_codna_branch_but_still_serves_a_human_pr():
    """A suite can list several PRs. Skipping Codna's own must not skip the human one beside it."""
    payload = _check_suite_payload(
        "failure", [_pr(number=1, ref="codna/aaa"), _pr(number=2, ref="fix/real-bug", sha="deadbee")]
    )
    job = classify_event("check_suite", payload)
    assert job is not None and job.pr_number == 2 and job.ref == "deadbee"


def test_classify_labels_route_to_fix_and_secure():
    base = {"action": "labeled", "repository": {"full_name": "acme/app"},
            "installation": {"id": 7}, "issue": {"number": 12}}
    fix = classify_event("issues", {**base, "label": {"name": FIX_LABEL}})
    assert fix == WebhookJob("fix", "acme/app", installation_id=7, issue_number=12, reason="labeled_codna_fix")
    secure = classify_event("issues", {**base, "label": {"name": SECURE_LABEL}})
    assert secure == WebhookJob("secure", "acme/app", installation_id=7, issue_number=12, reason="labeled_codna_secure")
    # an unrelated label is ignored
    assert classify_event("issues", {**base, "label": {"name": "wontfix"}}) is None


def test_classify_code_scanning_alert_is_a_secure_job():
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 9},
        "ref": "refs/heads/main",
    }
    job = classify_event("code_scanning_alert", payload)
    assert job == WebhookJob("secure", "acme/app", ref="refs/heads/main", installation_id=9, reason="code_scanning_alert")


def test_classify_ignores_unknown_events_and_missing_repo():
    assert classify_event("push", {"repository": {"full_name": "acme/app"}}) is None
    assert classify_event("issues", {"action": "labeled", "label": {"name": FIX_LABEL}}) is None  # no repository


# ---- review triggers (the proactive PR-review App surface) ------------------------------------

def test_classify_pull_request_opened_is_a_review_job():
    for action in ("opened", "synchronize", "reopened", "ready_for_review"):
        payload = {
            "action": action,
            "repository": {"full_name": "acme/app"},
            "installation": {"id": 5},
            "pull_request": {"number": 17, "head": {"sha": "deadbeef"}},
        }
        job = classify_event("pull_request", payload)
        assert job == WebhookJob("review", "acme/app", ref="deadbeef", pr_number=17,
                                 installation_id=5, reason=f"pull_request_{action}")


def test_classify_pull_request_draft_and_closed_are_ignored():
    base = {"repository": {"full_name": "acme/app"}, "pull_request": {"number": 1, "draft": True,
            "head": {"sha": "x"}}}
    assert classify_event("pull_request", {**base, "action": "opened"}) is None  # draft
    assert classify_event("pull_request", {**base, "action": "closed",
                                           "pull_request": {"number": 1, "head": {"sha": "x"}}}) is None


def test_classify_comment_codna_review_on_a_pr():
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 3},
        "issue": {"number": 21, "pull_request": {"url": "..."}},
        "comment": {"body": "please @codna review this"},
    }
    # only a line STARTING with @codna review triggers (mid-sentence mention does not)
    assert classify_event("issue_comment", payload) is None
    payload["comment"]["body"] = "@codna review\nthanks"
    job = classify_event("issue_comment", payload)
    assert job == WebhookJob("review", "acme/app", pr_number=21, installation_id=3, reason="comment_codna_review")


def test_classify_comment_on_plain_issue_is_ignored():
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "issue": {"number": 21},  # no pull_request key → a plain issue
        "comment": {"body": "@codna review"},
    }
    assert classify_event("issue_comment", payload) is None


def test_classify_inline_review_comment_codna_review():
    payload = {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 8},
        "pull_request": {"number": 30},
        "comment": {"body": "@codna review"},
    }
    job = classify_event("pull_request_review_comment", payload)
    assert job == WebhookJob("review", "acme/app", pr_number=30, installation_id=8,
                             reason="review_comment_codna_review")


def test_codna_command_review_posts_and_is_read_only():
    job = WebhookJob("review", "acme/app", ref="abc123", pr_number=42)
    cmd = codna_command(job)
    assert cmd == ["codna", "review", "https://github.com/acme/app.git", "--pr", "42",
                   "--post", "--json", "--ref", "abc123"]
    assert "--open-pr" not in cmd and "contents" not in cmd
    # a review job with no PR fails closed
    with pytest.raises(WebhookError) as exc:
        codna_command(WebhookJob("review", "acme/app"))
    assert exc.value.code == "review_requires_pr"


def test_review_token_scope_has_no_contents_write():
    from codna.webhook_github import token_permissions_for

    perms = token_permissions_for("review")
    assert perms == {"contents": "read", "pull_requests": "write", "checks": "write"}
    assert perms["contents"] == "read"  # review NEVER pushes code
    # with Commit statuses granted the review token may also READ commit statuses (a failed Vercel
    # deployment is one); nothing else changes, and nothing becomes writable
    with_statuses = token_permissions_for("review", statuses=True)
    assert with_statuses == {**perms, "statuses": "read"}
    assert [v for k, v in with_statuses.items() if k != "pull_requests" and k != "checks"] == ["read", "read"]


# ---- @codna fix wedge: route a review finding into a verified fix PR ---------------------------

def _review_comment_payload(body="@codna fix", *, in_reply_to=555, head_repo="acme/app",
                            author_association="MEMBER"):
    return {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 5},
        "pull_request": {
            "number": 7,
            "head": {"sha": "deadbeef", "ref": "feature", "repo": {"full_name": head_repo}},
            "base": {"ref": "main", "repo": {"full_name": "acme/app"}},
        },
        "comment": {"body": body, "in_reply_to_id": in_reply_to, "path": "src/a.py", "line": 10,
                    "author_association": author_association},
    }


def test_classify_review_comment_codna_fix_builds_a_context_job():
    job = classify_event("pull_request_review_comment", _review_comment_payload())
    assert job.kind == "fix" and job.pr_number == 7 and job.reason == "review_comment_codna_fix"
    assert job.ref == "deadbeef"  # analyze the PR head
    ctx = job.context
    assert ctx["in_reply_to_id"] == 555 and ctx["path"] == "src/a.py" and ctx["line"] == 10
    assert ctx["head_ref"] == "feature" and ctx["base_ref"] == "main" and ctx["is_fork"] is False


def test_classify_review_comment_fix_requires_write_level_commenter():
    # @codna fix SPENDS + PUSHES → only write-level commenters may trigger it.
    for assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert classify_event("pull_request_review_comment", _review_comment_payload(author_association=assoc)) is not None
    # Since 2026-09-17 the payload association no longer decides: GitHub sent CONTRIBUTOR for an org
    # admin, so every association classifies and the WORKER authorizes against the repository's
    # collaborator permission. The association still travels in the context for the fallback.
    for assoc in ("CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "NONE", None):
        job = classify_event("pull_request_review_comment", _review_comment_payload(author_association=assoc))
        assert job is not None and job.context["author_association"] == assoc


def test_classify_review_comment_fix_requires_a_reply_to_a_finding():
    # A `@codna fix` review comment that is NOT a reply (no in_reply_to_id) is ignored.
    payload = _review_comment_payload()
    payload["comment"].pop("in_reply_to_id")
    assert classify_event("pull_request_review_comment", payload) is None


def test_classify_review_comment_fix_detects_fork():
    job = classify_event("pull_request_review_comment", _review_comment_payload(head_repo="fork/app"))
    assert job.context["is_fork"] is True


def test_codna_command_comment_fix_targets_the_pr_branch_and_localizes():
    ctx = {"in_reply_to_id": 555, "path": "src/a.py", "line": 10, "head_sha": "deadbeef",
           "head_ref": "feature", "base_ref": "main", "is_fork": False,
           "severity": "high", "category": "security", "title": "SQLi", "explanation": "unescaped"}
    job = WebhookJob("fix", "acme/app", ref="deadbeef", pr_number=7, context=ctx)
    cmd = codna_command(job)
    assert cmd[:5] == ["codna", "fix", "https://github.com/acme/app.git", "--open-pr", "--issue"]
    issue = cmd[5]
    assert "src/a.py:10" in issue and "SQLi" in issue and "unescaped" in issue
    assert cmd[cmd.index("--ref") + 1] == "deadbeef"
    assert cmd[cmd.index("--base-branch") + 1] == "feature"  # stack the fix onto the PR's branch


def test_codna_command_comment_fix_fails_closed_without_head_ref():
    job = WebhookJob("fix", "acme/app", pr_number=7, context={"in_reply_to_id": 555, "head_ref": None})
    with pytest.raises(WebhookError) as exc:
        codna_command(job)
    assert exc.value.code == "fix_requires_head_ref"


def test_plain_fix_command_unchanged_by_the_wedge():
    # A check_suite fix job (no context, no issue_number) still builds the same shape of argv --
    # the comment-fix "wedge" (context branch) doesn't leak into it. It DOES get --tests (see
    # test_check_suite_fix_auto_discovers_failing_tests below) since there's no comment to localize.
    cmd = codna_command(WebhookJob("fix", "acme/app", ref="abc123"))
    assert cmd[:6] == ["codna", "fix", "https://github.com/acme/app.git", "--open-pr", "--ref", "abc123"]
    assert "--issue" not in cmd and "--base-branch" not in cmd


def test_check_suite_fix_auto_discovers_failing_tests():
    # ci_check_suite_failed has a ref but no issue text -- codna must discover what's broken itself.
    cmd = codna_command(WebhookJob("fix", "acme/app", ref="abc123", reason="ci_check_suite_failed"))
    assert "--tests" in cmd
    assert "--issue" not in cmd


def test_check_suite_fix_on_a_pr_stacks_onto_that_pr_branch():
    """The fix has to target the PR's OWN branch. Opening it against the default branch would try
    to repair code that does not exist there yet -- the failing change lives only on the PR."""
    job = WebhookJob("fix", "acme/app", ref="abc123", pr_number=7,
                     context={"head_ref": "feature/login"}, reason="check_suite_failure")
    cmd = codna_command(job)
    assert "--tests" in cmd
    assert cmd[cmd.index("--base-branch") + 1] == "feature/login"
    # A head_ref-only context must NOT be mistaken for the comment-fix wedge, which needs a
    # finding to localize and would otherwise demand --issue text this job has none of.
    assert "--issue" not in cmd


def test_two_failing_suites_on_one_commit_reuse_a_single_fix_pr():
    """A PR usually has several workflows. Each red suite delivers its own check_suite event, so
    without a shared idempotency marker one push would open N fix PRs for the same commit."""
    from codna.webhook import webhook_marker

    a = WebhookJob("fix", "acme/app", ref="abc123", pr_number=7,
                   context={"head_ref": "feature/login"}, reason="check_suite_failure")
    b = WebhookJob("fix", "acme/app", ref="abc123", pr_number=7,
                   context={"head_ref": "feature/login"}, reason="check_suite_failure")
    assert webhook_marker(a) == webhook_marker(b)
    # A later push to the same PR is a different commit and legitimately gets its own fix PR.
    c = WebhookJob("fix", "acme/app", ref="def456", pr_number=7,
                   context={"head_ref": "feature/login"}, reason="check_suite_failure")
    assert webhook_marker(c) != webhook_marker(a)


def test_issue_label_fix_requires_issue_text():
    # labeled_codna_fix has neither a ref nor tests to run -- without the fetched issue text,
    # `codna fix` would immediately die with "needs --issue ..."; codna_command must refuse first
    # with a clear, catchable error rather than silently building a command that's doomed to fail.
    job = WebhookJob("fix", "acme/app", issue_number=12, reason="labeled_codna_fix")
    with pytest.raises(WebhookError) as exc:
        codna_command(job)
    assert exc.value.code == "fix_requires_issue_text"


def test_issue_label_fix_uses_the_fetched_issue_text():
    job = WebhookJob("fix", "acme/app", issue_number=12, reason="labeled_codna_fix")
    cmd = codna_command(job, issue_text="Login crashes on empty password")
    assert cmd[:4] == ["codna", "fix", "https://github.com/acme/app.git", "--open-pr"]
    assert cmd[cmd.index("--issue") + 1] == "Login crashes on empty password"
    assert "--ref" not in cmd and "--tests" not in cmd


def test_webhook_marker_distinct_per_finding_on_same_head():
    from codna.webhook import webhook_marker

    base = {"head_sha": "deadbeef", "head_ref": "feature"}
    j1 = WebhookJob("fix", "acme/app", ref="deadbeef", pr_number=7, context={**base, "in_reply_to_id": 111})
    j2 = WebhookJob("fix", "acme/app", ref="deadbeef", pr_number=7, context={**base, "in_reply_to_id": 222})
    assert webhook_marker(j1) != webhook_marker(j2)  # two findings on one head -> two distinct PRs
    assert "111" in webhook_marker(j1) and "222" in webhook_marker(j2)


def test_codna_command_fix_opens_pr_with_idempotency_marker():
    from codna.webhook import webhook_marker

    job = WebhookJob("fix", "acme/app", ref="abc123")
    cmd = codna_command(job)
    assert cmd[:6] == ["codna", "fix", "https://github.com/acme/app.git", "--open-pr", "--ref", "abc123"]
    # a deterministic marker is embedded in the PR body so a re-run can find + reuse the PR
    body = cmd[cmd.index("--pr-body") + 1]
    assert webhook_marker(job) in body
    assert "codna-webhook-id: acme/app#fix#abc123" in body


def test_webhook_marker_is_deterministic_per_trigger_not_per_delivery():
    from codna.webhook import webhook_marker

    a = WebhookJob("fix", "acme/app", ref="abc123", installation_id=1)
    b = WebhookJob("fix", "acme/app", ref="abc123", installation_id=999)  # different delivery/install
    assert webhook_marker(a) == webhook_marker(b)  # same trigger -> same marker -> dedupes re-runs
    c = WebhookJob("fix", "acme/app", ref="def456")
    assert webhook_marker(c) != webhook_marker(a)  # different commit -> different marker


def test_codna_command_secure_requires_sarif_and_is_read_only():
    job = WebhookJob("secure", "acme/app", ref="main")
    # a single server process must not open security PRs — secure is read-only classification here
    cmd = codna_command(job, sarif_path="/tmp/x.sarif")
    assert cmd == ["codna", "secure", "https://github.com/acme/app.git", "--from-sarif", "/tmp/x.sarif", "--ref", "main"]
    assert "--open-pr" not in cmd
    # secure without a SARIF fails closed
    with pytest.raises(WebhookError) as exc:
        codna_command(job)
    assert exc.value.code == "secure_requires_sarif"


def test_webhook_subcommand_is_registered_in_the_cli():
    import codna.cli as cli_module

    parser = cli_module.build_parser()
    ns = parser.parse_args(["webhook", "serve", "--port", "9000"])
    assert ns.func is cli_module.cmd_webhook
    assert ns.action == "serve"
    assert ns.port == 9000


def test_webhook_missing_secret_fails_before_listening_banner(monkeypatch, capsys):
    from argparse import Namespace

    from codna.webhook_cli import cmd_webhook

    monkeypatch.delenv("CODNA_GITHUB_WEBHOOK_SECRET", raising=False)
    with pytest.raises(SystemExit) as exc:
        cmd_webhook(Namespace(action="serve", host="127.0.0.1", port=19091))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "CODNA_GITHUB_WEBHOOK_SECRET" in err
    assert "listening on" not in err


def test_webhook_serve_error_fails_before_listening_banner(monkeypatch, capsys):
    from argparse import Namespace

    import codna.webhook_service as webhook_service
    from codna.webhook import WebhookError
    from codna.webhook_cli import cmd_webhook

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")

    def fail_start(*, host, port):
        raise WebhookError("queue_open_failed", f"unable to open queue on {host}:{port}")

    # `codna webhook serve` enters through webhook_service.serve (the role/backend-aware entrypoint).
    monkeypatch.setattr(webhook_service, "serve", fail_start)
    with pytest.raises(SystemExit) as exc:
        cmd_webhook(Namespace(action="serve", host="127.0.0.1", port=19091))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "unable to open queue" in err
    assert "listening on" not in err


def test_webhook_queue_override_creates_parent(tmp_path, monkeypatch):
    from codna.webhook import default_queue_path

    queue_path = tmp_path / "missing" / "nested" / "queue.db"
    monkeypatch.setenv("CODNA_WEBHOOK_QUEUE", str(queue_path))

    assert default_queue_path() == str(queue_path)
    assert queue_path.parent.is_dir()


def _serve_ephemeral(monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer

    from codna.webhook import _Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


def test_health_endpoint_is_live_when_the_secret_is_configured(monkeypatch):
    import json as _json
    import urllib.request

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")
    httpd, port = _serve_ephemeral(monkeypatch)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            assert resp.status == 200
            assert _json.loads(resp.read())["ok"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ready_endpoint_requires_webhook_secret_and_app_auth(monkeypatch):
    import json as _json
    import urllib.error
    import urllib.request

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_APP_PRIVATE_KEY", raising=False)
    httpd, port = _serve_ephemeral(monkeypatch)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5)
        assert exc.value.code == 503
        failed = _json.loads(exc.value.read())
        assert failed["checks"] == {"webhook_secret": True, "github_app_auth": False, "worker_pool": True}

        monkeypatch.setenv("GITHUB_TOKEN", "single-repo-token")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as resp:
            assert resp.status == 200
            ready = _json.loads(resp.read())
        assert ready["ok"] is True
        assert ready["checks"] == {"webhook_secret": True, "github_app_auth": True, "worker_pool": True}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ready_reports_queue_depth_by_status(tmp_path, monkeypatch):
    """17 PRs opened within a minute sat behind the 2-thread pool for ~15 min on 2026-09-19 and
    nothing public said why their checks were missing: /ready now carries the queue depth."""
    import json as _json
    import urllib.request

    from codna.webhook_queue import WebhookQueue

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")
    monkeypatch.setenv("GITHUB_TOKEN", "single-repo-token")
    queue = WebhookQueue(tmp_path / "q.db")
    for i in range(3):
        queue.enqueue(WebhookJob("review", "acme/app", ref=f"sha{i}", installation_id=7, pr_number=i, reason="test"),
                      delivery_id=f"d{i}")
    assert queue.claim() is not None                      # one running, two still queued
    httpd, port = _serve_with_queue(queue)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5) as resp:
            ready = _json.loads(resp.read())
        assert ready["queue"] == {"queued": 2, "running": 1}   # counts only, never job content
        assert "recent" not in ready and "jobs" not in ready
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_debug_queue_requires_token_and_redacts_recent_results(tmp_path, monkeypatch):
    import json as _json
    import urllib.error
    import urllib.request

    from codna.webhook_queue import WebhookQueue

    secret = "s3cr3t"
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)
    queue = WebhookQueue(tmp_path / "q.db")
    queue.enqueue(
        WebhookJob("fix", "acme/app", ref="abc123", installation_id=7, reason="test"),
        delivery_id="delivery-1",
    )
    qjob = queue.claim()
    assert qjob is not None
    # A credential-shaped value the redactor must catch. Assembled at runtime so the source never
    # carries a literal that secret scanning reads as a real token (codna secret-scanning alert #1).
    fake_token = "ghp_" + "1234567890" * 3 + "123456"  # ghp_ + 36 chars, the real PAT shape
    assert len(fake_token) == 40
    queue.complete(qjob.row_id, status="failed", result={"error": "boom", "message": f"token {fake_token}"})
    httpd, port = _serve_with_queue(queue)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/debug/queue", timeout=5)
        assert exc.value.code == 401

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/debug/queue",
            headers={"X-Codna-Debug-Token": secret},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            payload = _json.loads(resp.read())
        assert payload["queue"]["counts"]["retry"] == 1
        assert payload["queue"]["recent"][0]["result"]["error"] == "boom"
        assert "ghp_" not in payload["queue"]["recent"][0]["result"]["message"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unsigned_delivery_is_rejected_401(monkeypatch):
    import urllib.error
    import urllib.request

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")
    httpd, port = _serve_ephemeral(monkeypatch)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhooks/github",
            data=b"{}",
            headers={"X-GitHub-Event": "issues", "Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 401  # fail-closed: no valid signature
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_end_to_end_signed_ingress_dedups_enqueues_and_worker_drains(tmp_path, monkeypatch):
    """The whole pipeline: signed POST -> verify -> dedup -> durable queue -> worker runs once."""
    import json as _json
    import threading
    import time
    import urllib.request
    from http.server import ThreadingHTTPServer

    from codna.webhook import _Handler
    from codna.webhook_queue import WebhookQueue
    from codna.webhook_worker import JobResult, WorkerPool

    secret = "hook-secret"
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)
    queue = WebhookQueue(tmp_path / "q.db")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.queue = queue  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    payload = _json.dumps({
        "action": "labeled",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 7},
        "issue": {"number": 12},
        "label": {"name": FIX_LABEL},
    }).encode()
    sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    def post(delivery_id):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhooks/github", data=payload, method="POST",
            headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig,
                     "X-GitHub-Delivery": delivery_id, "Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=5)

    try:
        assert post("delivery-1").status == 202              # accepted + enqueued
        assert post("delivery-1").status == 200              # redelivery -> duplicate, not re-queued
        assert queue.counts().get("queued") == 1

        processed = []
        pool = WorkerPool(queue, concurrency=1, poll_interval=0.02,
                          process=lambda qj: (processed.append(qj.job.kind) or JobResult(True, "ok")))
        pool.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and queue.counts().get("done", 0) < 1:
            time.sleep(0.05)
        pool.stop()

        assert processed == ["fix"]                          # ran exactly once despite the duplicate
        assert queue.counts().get("done") == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def _serve_with_queue(queue):
    import threading
    from http.server import ThreadingHTTPServer

    from codna.webhook import _Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.queue = queue  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def test_missing_delivery_header_still_dedups_via_synthesized_key(tmp_path, monkeypatch):
    """No X-GitHub-Delivery (proxy stripped / replay) must NOT defeat the once-only guarantee."""
    import urllib.request

    from codna.webhook_queue import WebhookQueue

    secret = "s3cr3t"
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)
    queue = WebhookQueue(tmp_path / "q.db")
    httpd, port = _serve_with_queue(queue)
    body = json.dumps({"action": "labeled", "repository": {"full_name": "acme/app"},
                       "issue": {"number": 12}, "label": {"name": FIX_LABEL}}).encode()
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        for _ in range(3):  # identical signed deliveries, NO X-GitHub-Delivery header
            req = urllib.request.Request(f"http://127.0.0.1:{port}/webhooks/github", data=body, method="POST",
                                         headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig})
            urllib.request.urlopen(req, timeout=5).read()
        assert queue.counts().get("queued") == 1  # synthesized key collapsed them to one
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_non_object_json_body_returns_400_not_a_crash(monkeypatch):
    import urllib.error
    import urllib.request

    secret = "s3cr3t"
    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", secret)
    httpd, port = _serve_ephemeral(monkeypatch)
    try:
        for raw in (b"[]", b"null", b"123"):  # valid JSON, not an object
            sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/webhooks/github", data=raw, method="POST",
                                         headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": sig})
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req, timeout=5)
            assert exc.value.code == 400  # clean 400, handler thread did not die
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_malformed_content_length_returns_400(monkeypatch):
    import socket

    monkeypatch.setenv("CODNA_GITHUB_WEBHOOK_SECRET", "s3cr3t")
    httpd, port = _serve_ephemeral(monkeypatch)
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(
            b"POST /webhooks/github HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n"
            b"X-GitHub-Event: issues\r\n\r\n"
        )
        status_line = s.recv(256).split(b"\r\n", 1)[0]
        s.close()
        assert b"400" in status_line  # clean 400, not a dropped connection
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- PR title + issue cross-reference for a label-triggered fix -------------------------------
def test_issue_label_fix_names_the_pr_after_the_issue():
    """fetch_issue_text formats an issue as "<title>\\n\\n<body>", so line 1 is the issue title.

    Regression: the label path passed NO --pr-title, so fix_run fell back to the model's free-text
    `root_cause` and a live fix PR shipped titled "codna: fix Here's a summary of every change
    made:". The comment-fix path always passed an explicit title; this closes that asymmetry."""
    job = WebhookJob("fix", "acme/app", issue_number=8, reason="labeled_codna_fix")
    text = "Naive token comparison in src/auth.py allows timing attack\n\nverify_token() uses `==`."
    cmd = codna_command(job, issue_text=text)
    assert cmd[cmd.index("--pr-title") + 1] == (
        "codna: fix Naive token comparison in src/auth.py allows timing attack"
    )


def test_issue_label_fix_pr_body_references_the_issue_without_closing_it():
    """The codna-webhook-id marker is NOT a GitHub reference, so nothing cross-linked the fix PR to
    the issue. A plain `#N` reference is used deliberately -- never a closing keyword, because a bot
    should surface the link, not decide to close a human's issue."""
    job = WebhookJob("fix", "acme/app", issue_number=8, reason="labeled_codna_fix")
    cmd = codna_command(job, issue_text="Something is broken\n\ndetails")
    from codna.webhook import webhook_marker

    body = cmd[cmd.index("--pr-body") + 1]
    marker = webhook_marker(job)
    assert marker in body  # idempotency marker still present
    # Assert on the PROSE half with the marker stripped out. Checking `"#8" in body` alone passes
    # even with this fix fully reverted, because the marker itself ends in "#8" -- a test that
    # cannot fail is worse than no test.
    prose = body.replace(marker, "")
    assert "#8" in prose, "the issue must be referenced OUTSIDE the idempotency marker to cross-link"
    for closing in ("closes #8", "fixes #8", "resolves #8"):
        assert closing not in body.lower(), f"must not auto-close the issue ({closing!r})"


def test_issue_label_fix_omits_pr_title_when_the_issue_has_no_usable_first_line():
    """No title is better than a garbage one -- fix_run's own fallback then applies."""
    job = WebhookJob("fix", "acme/app", issue_number=8, reason="labeled_codna_fix")
    cmd = codna_command(job, issue_text="Here's a summary of every change made:")
    assert "--pr-title" not in cmd


def test_pr_title_from_issue_text_rejects_model_prose_and_cleans_decoration():
    from codna.webhook import pr_title_from_issue_text as t

    assert t("## Bug: crash on empty input\n\nbody") == "Bug: crash on empty input"
    assert t("- fix the thing:\n\nbody") == "fix the thing"
    assert t("Here is what I did:") is None
    assert t("Here's a summary of every change made:") is None  # the prose that shipped live
    assert t("Summary of changes:") is None                     # section header, not a subject
    assert t("") is None and t(None) is None
    assert len(t("word " * 60)) <= 72


def test_pr_title_keeps_legitimate_titles_that_merely_start_like_prose():
    """The prose filter must stay NARROW. A first cut matched a bare leading `summary`/`the
    following`/`i have` and threw away perfectly good human issue titles."""
    from codna.webhook import pr_title_from_issue_text as t

    for title in (
        "Summary tab shows stale totals after a refund",
        "The following endpoints 500 on empty payload",
        "I have no access to the audit log as an org owner",
        "Details panel truncates long names",
    ):
        assert t(f"{title}\n\nSteps to reproduce: ...") == title


def test_pr_title_never_falls_through_into_the_issue_body():
    """First line only. Walking into the body just swapped one nonsense title for another -- a repro
    step, a quoted log, an @-mention -- which is the very defect this helper exists to prevent.
    None is safe: codna_command then omits --pr-title and fix_run's fallback applies."""
    from codna.webhook import pr_title_from_issue_text as t

    assert t("Here's a summary:\n\nSteps to reproduce: open Reports, reload.") is None
    assert t("\n\n") is None


def test_pr_title_defangs_smuggled_github_closing_keywords():
    """Issue text is attacker-influenced and the PR title becomes the branch's COMMIT MESSAGE, where
    GitHub honours closing keywords on merge -- so a crafted issue title could close an unrelated
    issue (a maintainer's security tracker, say). The words stay readable; the reference is inert."""
    from codna.webhook import pr_title_from_issue_text as t

    for hostile, banned in (
        ("Login crash on empty password. Closes #1", "#1"),
        ("Broken thing, fixes #42", "#42"),
        ("Whatever, resolves GH-7", "GH-7"),
        ("Bug, closes https://github.com/acme/app/issues/99", "#99"),
    ):
        out = t(hostile)
        assert banned not in out, f"{banned!r} still live in {out!r}"
        assert "issues/" not in out


def test_fallback_pr_title_never_uses_model_prose():
    """fix_run's safety net for callers that pass no --pr-title (a hand-run `codna fix`, the Action)."""
    from codna.fix_run import _fallback_pr_title

    assert _fallback_pr_title("Login crashes on empty password", {}) == (
        "codna: fix Login crashes on empty password"
    )
    # The exact prose that shipped a broken title, with no issue text to fall back on.
    assert _fallback_pr_title(None, {"root_cause": "Here's a summary of every change made:"}) == (
        "codna: fix bug"
    )
    assert _fallback_pr_title(None, {}) == "codna: fix bug"



def test_delivery_log_line_names_event_outcome_and_job(capsys):
    import json as _json

    from codna.webhook import WebhookJob, _log_delivery

    payload = {"action": "created", "repository": {"full_name": "acme/app"}}
    _log_delivery("pull_request_review_comment", payload, "d-1", "ignored")
    job = WebhookJob("fix", "acme/app", ref="sha1", pr_number=7, reason="review_comment_codna_fix")
    _log_delivery("pull_request_review_comment", payload, "d-2", "accepted", job)
    lines = [_json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    assert [x["outcome"] for x in lines] == ["ignored", "accepted"]
    assert lines[0]["github_event"] == "pull_request_review_comment" and lines[0]["repo"] == "acme/app"
    assert lines[1]["kind"] == "fix" and lines[1]["pr_number"] == 7 and lines[1]["delivery_id"] == "d-2"
    assert all(x["service"] == "codna-webhook" and x["event"] == "delivery" for x in lines)



def test_fix_reply_is_classified_for_any_association_and_carries_the_commenter():
    from codna.webhook import classify_event

    payload = {
        "action": "created",
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 42},
        "pull_request": {"number": 63, "draft": False,
                         "head": {"sha": "a" * 40, "ref": "dependabot/x", "repo": {"full_name": "acme/app"}},
                         "base": {"ref": "main", "repo": {"full_name": "acme/app"}}},
        "comment": {"id": 2, "in_reply_to_id": 1, "body": "@codna fix", "author_association": "CONTRIBUTOR",
                    "path": "python/pyproject.toml", "line": 69, "user": {"login": "0xamlab"}},
        "sender": {"login": "0xamlab", "type": "User"},
    }
    job = classify_event("pull_request_review_comment", payload)
    assert job is not None and job.kind == "fix" and job.pr_number == 63
    assert job.context["commenter"] == "0xamlab" and job.context["author_association"] == "CONTRIBUTOR"


# ---- job scratch root (TMPDIR on the volume) ----------------------------------------------------

def test_prepare_scratch_root_creates_a_missing_dir_and_makes_it_the_default_temp_dir(monkeypatch, tmp_path):
    """A fresh volume has no /data/tmp yet; the server must create it and every tempfile call in
    this process (the worker's per-job TemporaryDirectory first of all) must land inside it."""
    import tempfile
    from pathlib import Path

    from codna.webhook_procs import prepare_scratch_root

    scratch = tmp_path / "data" / "tmp"
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)  # drop any cached default; restored after the test
    assert prepare_scratch_root() == scratch
    assert scratch.is_dir()
    assert Path(tempfile.gettempdir()) == scratch
    with tempfile.TemporaryDirectory(prefix="codna-webhook-job-") as job_dir:
        assert Path(job_dir).parent == scratch


def test_prepare_scratch_root_sweeps_only_stale_codna_dirs(monkeypatch, tmp_path):
    """REGRESSION: 61 codna-* dirs sat in the machine's /tmp on 2026-09-18 -- review clones an
    older CLI never removed plus per-job dirs of processes killed mid-job. At startup nothing of
    ours can be live, so every codna-* directory is stale. Anything else on the volume is not
    ours to delete."""
    import tempfile

    from codna.webhook_procs import prepare_scratch_root

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "codna-webhook-job-old").mkdir()
    (scratch / "codna-webhook-job-old" / "clone.txt").write_text("x", encoding="utf-8")
    (scratch / "codna-review-old").mkdir()
    (scratch / "keep-me").mkdir()
    (scratch / "codna-notes.txt").write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", None)
    prepare_scratch_root()
    assert not (scratch / "codna-webhook-job-old").exists()
    assert not (scratch / "codna-review-old").exists()
    assert (scratch / "keep-me").is_dir()
    assert (scratch / "codna-notes.txt").is_file()


def test_prepare_scratch_root_without_tmpdir_changes_nothing(monkeypatch):
    import tempfile

    from codna.webhook_procs import prepare_scratch_root

    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert prepare_scratch_root() is None
    assert tempfile.tempdir is None



# ---- merge queues: the group commit inherits the PR head's review --------------------------------

def _merge_group_payload(action="checks_requested", pr=48, base="main"):
    return {
        "action": action,
        "repository": {"full_name": "acme/app"},
        "installation": {"id": 42},
        "merge_group": {
            "head_sha": "feedfacefeedfacefeedfacefeedfacefeedface",
            "head_ref": f"refs/heads/gh-readonly-queue/{base}/pr-{pr}-0123456789abcdef0123456789abcdef01234567",
            "base_ref": f"refs/heads/{base}",
        },
    }


def test_merge_group_pr_number_parses_the_queue_ref():
    from codna.webhook import merge_group_pr_number

    assert merge_group_pr_number("refs/heads/gh-readonly-queue/main/pr-48-0123456789abcdef0123456789abcdef01234567") == 48
    assert merge_group_pr_number("refs/heads/gh-readonly-queue/release/1.x/pr-7-0123456789abcdef0123456789abcdef01234567") == 7
    assert merge_group_pr_number("refs/heads/feature/pr-48-not-a-queue") is None
    assert merge_group_pr_number("") is None


def test_classify_merge_group_checks_requested_is_a_queue_job():
    job = classify_event("merge_group", _merge_group_payload())
    assert job is not None and job.kind == "queue"
    assert job.repo_full_name == "acme/app" and job.pr_number == 48
    assert job.ref == "feedfacefeedfacefeedfacefeedfacefeedface"
    assert job.reason == "merge_group_checks_requested"


def test_classify_merge_group_destroyed_or_unparseable_is_ignored():
    assert classify_event("merge_group", _merge_group_payload(action="destroyed")) is None
    p = _merge_group_payload()
    p["merge_group"]["head_ref"] = "refs/heads/something-else"
    assert classify_event("merge_group", p) is None


def test_classify_check_suite_failure_carries_the_suite_id_for_triage():
    payload = _check_suite_payload("failure", [_pr()])
    payload["check_suite"]["id"] = 555
    job = classify_event("check_suite", payload)
    assert job is not None and job.context["check_suite_id"] == 555
    assert job.context["head_ref"] == "feature/login"


def test_check_suite_fix_with_ci_evidence_passes_it_as_the_issue():
    job = WebhookJob("fix", "acme/app", ref="abc123", pr_number=7,
                     context={"head_ref": "feature/login", "check_suite_id": 555,
                              "ci_failure": "CI failed at step X"},
                     reason="check_suite_failure")
    cmd = codna_command(job)
    assert "--tests" in cmd
    assert cmd[cmd.index("--issue") + 1] == "CI failed at step X"
    assert cmd[cmd.index("--base-branch") + 1] == "feature/login"
