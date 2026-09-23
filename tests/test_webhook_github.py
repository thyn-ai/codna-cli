"""Pure builders for the webhook's GitHub API layer: least-privilege token scoping + Check Runs."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from codna.webhook import WebhookError
from codna.webhook_github import (
    _app_jwt,
    app_auth_config_ready,
    check_run_payload,
    create_check_run,
    fetch_issue_text,
    normalize_private_key_pem,
    post_issue_comment,
    scoped_token_request,
    token_permissions_for,
)


def test_fix_token_can_write_secure_token_only_reads():
    fix = token_permissions_for("fix")
    assert fix["contents"] == "write" and fix["pull_requests"] == "write"
    assert fix["issues"] == "write"  # issue-label triggers need a visible fail-closed prompt
    secure = token_permissions_for("secure")
    assert secure["contents"] == "read" and secure["security_events"] == "read"
    assert secure["issues"] == "write"  # issue-label triggers need a visible fail-closed prompt
    assert "pull_requests" not in secure  # secure classification never needs write


def test_unknown_kind_has_no_token_scope():
    with pytest.raises(WebhookError) as exc:
        token_permissions_for("delete-everything")
    assert exc.value.code == "unknown_job_kind"


def test_scoped_token_request_is_limited_to_the_one_repo():
    body = scoped_token_request("acme/app", "fix")
    assert body["repositories"] == ["app"]  # repo NAME, scoped to just this repo
    assert body["permissions"]["contents"] == "write"


def test_check_run_payload_omits_conclusion_until_completed():
    inprog = check_run_payload("codna fix", "sha1", status="in_progress", conclusion=None, summary="started")
    assert "conclusion" not in inprog
    assert inprog["status"] == "in_progress"
    done = check_run_payload("codna fix", "sha1", status="completed", conclusion="success", summary="ok")
    assert done["conclusion"] == "success"
    assert done["output"]["summary"] == "ok"


def test_app_auth_accepts_secret_store_escaped_pem_newlines():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    escaped = pem.replace("\n", "\\n")
    assert normalize_private_key_pem(escaped) == pem.strip()
    assert app_auth_config_ready("12345", escaped) is True
    assert _app_jwt("12345", escaped, now=1_700_000_000).count(".") == 2


def test_check_run_create_failure_raises_structured_error(monkeypatch):
    class Response:
        status_code = 403
        text = "checks permission denied"

        @staticmethod
        def json():
            return {}

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=lambda *args, **kwargs: Response()))
    with pytest.raises(WebhookError) as exc:
        create_check_run("acme/app", "token", name="codna fix", head_sha="sha1", summary="started")
    assert exc.value.code == "check_run_create_failed"
    assert "403" in str(exc.value)


def test_post_issue_comment_returns_url(monkeypatch):
    class Response:
        status_code = 201

        @staticmethod
        def json():
            return {"html_url": "https://github.com/acme/app/issues/12#issuecomment-1"}

    seen = {}

    def post(url, *, headers, json, timeout, follow_redirects):
        seen["url"] = url
        seen["body"] = json["body"]
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))
    url = post_issue_comment("acme/app", "token", 12, "link account")
    assert url.endswith("#issuecomment-1")
    assert seen["url"].endswith("/repos/acme/app/issues/12/comments")
    assert seen["body"] == "link account"
    assert seen["timeout"] == 30.0


def test_fetch_issue_text_joins_title_and_body(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"title": "Login crashes", "body": "Empty password throws a 500."}

    seen = {}

    def get(url, *, headers, timeout, follow_redirects):
        seen["url"] = url
        return Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=get))
    text = fetch_issue_text("acme/app", "token", 12)
    assert text == "Login crashes\n\nEmpty password throws a 500."
    assert seen["url"].endswith("/repos/acme/app/issues/12")


def test_fetch_issue_text_falls_back_to_title_only_when_body_is_empty(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"title": "Login crashes", "body": None}

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=lambda *a, **k: Response()))
    assert fetch_issue_text("acme/app", "token", 12) == "Login crashes"


# --- complete_stale_check_runs: a retry supersedes a dead in_progress run --------------------------
# Only the App that created a Check Run can complete it; a run abandoned by a redeploy therefore spins
# forever (thyn-ai/algenta@cc26ddd6 `codna fix`, in_progress from 2026-09-17 04:49 onward) unless the
# next attempt closes it first.
def _stale_httpx(get_status=200, runs=({"id": 11}, {"id": 12}), patch_fail_ids=(), queued_runs=()):
    """``runs`` are the commit's in_progress runs of the name, ``queued_runs`` its queued ones: the
    listing endpoint filters on ONE status per request, so the sweep asks twice and the fake answers
    per ``params["status"]``. ``seen["get"]`` is the last listing, ``seen["listed"]`` every status asked."""
    seen = {"get": None, "listed": [], "patched": []}

    def get(url, *, headers, params, timeout, follow_redirects):
        seen["get"] = {"url": url, "params": params}
        seen["listed"].append(params["status"])
        page = list(runs) if params["status"] == "in_progress" else list(queued_runs)

        class GetResp:
            status_code = get_status

            @staticmethod
            def json():
                return {"check_runs": page}

        return GetResp()

    def patch(url, *, headers, json, timeout, follow_redirects):
        run_id = int(url.rsplit("/", 1)[-1])
        seen["patched"].append((run_id, json["conclusion"], json["output"]["summary"]))
        return SimpleNamespace(status_code=403 if run_id in patch_fail_ids else 200, text="nope")

    return SimpleNamespace(get=get, patch=patch), seen


def test_complete_stale_check_runs_cancels_every_in_progress_run_of_that_name(monkeypatch):
    from codna.webhook_github import complete_stale_check_runs

    fake, seen = _stale_httpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    n = complete_stale_check_runs("acme/app", "token", head_sha="sha1", name="codna fix", summary="superseded")
    assert n == 2
    assert seen["get"]["url"].endswith("/repos/acme/app/commits/sha1/check-runs")
    # One listing per open status: GitHub filters on a single `status` value per request, and a run
    # the ingress opened `queued` for a row that never reached a worker is as stranded as an
    # in_progress one.
    assert seen["listed"] == ["in_progress", "queued"]
    assert seen["get"]["params"] == {"check_name": "codna fix", "status": "queued", "per_page": 50}
    assert seen["patched"] == [(11, "cancelled", "superseded"), (12, "cancelled", "superseded")]


def test_complete_stale_check_runs_sweeps_queued_runs_too(monkeypatch):
    """The ingress opens a review's run `queued` (webhook_queued_check); when the head moved before the
    job started, that run is on a commit nobody merges and is closed with the in_progress ones."""
    from codna.webhook_github import complete_stale_check_runs

    fake, seen = _stale_httpx(runs=(), queued_runs=({"id": 77},))
    monkeypatch.setitem(sys.modules, "httpx", fake)
    n = complete_stale_check_runs("acme/app", "token", head_sha="sha1", name="codna review",
                                  summary="head moved", conclusion="neutral")
    assert n == 1
    assert seen["patched"] == [(77, "neutral", "head moved")]


def test_create_check_run_can_open_queued_and_start_flips_it_in_progress(monkeypatch):
    """The ingress creates the run `queued`; the worker that claims the row PATCHes it `in_progress`
    with a fresh started_at (the job's own clock, not the queue wait), never a second run."""
    from codna.webhook_github import start_check_run

    sent = _fake_post(monkeypatch, [_Resp(201, {"id": 4242})])
    assert create_check_run("acme/app", "t", name="codna review", head_sha="abc", summary="queued", status="queued") == 4242
    assert sent[0]["status"] == "queued" and "conclusion" not in sent[0] and sent[0]["head_sha"] == "abc"
    # default unchanged: a worker creating at claim still opens in_progress
    sent = _fake_post(monkeypatch, [_Resp(201, {"id": 4243})])
    create_check_run("acme/app", "t", name="codna review", head_sha="abc", summary="started")
    assert sent[0]["status"] == "in_progress"

    import httpx

    patched = []

    def patch(url, *, headers, json, timeout, follow_redirects):
        patched.append((url, json))
        return _Resp(200, {"id": 4242, "status": "in_progress"})

    monkeypatch.setattr(httpx, "patch", patch)
    start_check_run("acme/app", "t", 4242, name="codna review", summary="codna review started (pull_request_opened)")
    url, body = patched[0]
    assert url.endswith("/repos/acme/app/check-runs/4242")
    assert body["status"] == "in_progress" and "conclusion" not in body and "head_sha" not in body
    assert body["started_at"].endswith("Z") and body["output"]["summary"].startswith("codna review started")
    monkeypatch.setattr(httpx, "patch", lambda *a, **k: _Resp(422, text="Validation Failed"))
    with pytest.raises(WebhookError) as exc:
        start_check_run("acme/app", "t", 4242, name="codna review", summary="s")
    assert exc.value.code == "check_run_update_failed"


def test_check_run_status_reads_the_run_and_is_none_on_any_failure(monkeypatch):
    """A retry asks whether the run its row carries is still open before reusing it; an unreadable run
    answers None so the worker opens a fresh one rather than reopening a completed one."""
    from codna.webhook_github import check_run_status

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200, {"id": 1, "status": "queued"}))
    assert check_run_status("acme/app", "t", 1) == "queued"
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(200, {"id": 1, "status": "completed"}))
    assert check_run_status("acme/app", "t", 1) == "completed"
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(404, {}))
    assert check_run_status("acme/app", "t", 1) is None

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert check_run_status("acme/app", "t", 1) is None


def test_complete_stale_check_runs_is_best_effort_never_raises(monkeypatch):
    from codna.webhook_github import complete_stale_check_runs

    # listing fails -> nothing to do, no exception
    fake, _ = _stale_httpx(get_status=500)
    monkeypatch.setitem(sys.modules, "httpx", fake)
    assert complete_stale_check_runs("acme/app", "token", head_sha="sha1", name="codna fix", summary="s") == 0
    # one PATCH is refused (someone else's run of the same name): skip it, still close the other
    fake, seen = _stale_httpx(patch_fail_ids=(11,))
    monkeypatch.setitem(sys.modules, "httpx", fake)
    assert complete_stale_check_runs("acme/app", "token", head_sha="sha1", name="codna fix", summary="s") == 1
    assert [p[0] for p in seen["patched"]] == [11, 12]


def test_fetch_issue_text_none_on_api_failure(monkeypatch):
    class Response:
        status_code = 404

        @staticmethod
        def json():
            return {}

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=lambda *a, **k: Response()))
    assert fetch_issue_text("acme/app", "token", 12) is None


def test_fetch_issue_text_none_when_title_and_body_both_blank(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"title": "  ", "body": ""}

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=lambda *a, **k: Response()))
    assert fetch_issue_text("acme/app", "token", 12) is None


def test_comment_authored_by_app_gate():
    from codna.webhook_github import comment_authored_by_app

    # Strong signal: performed_via_github_app.id matches this App id (a user cannot forge this).
    assert comment_authored_by_app({"performed_via_github_app": {"id": 4061960}}, 4061960) is True
    assert comment_authored_by_app({"performed_via_github_app": {"id": 4061960}}, "4061960") is True  # str/int
    # A different App id → not ours.
    assert comment_authored_by_app({"performed_via_github_app": {"id": 999}}, 4061960) is False
    # A human who merely pasted the marker (no App attribution, User type) → rejected.
    assert comment_authored_by_app({"user": {"type": "User"}}, 4061960) is False
    # Fallback when the App id isn't configured: accept a Bot-type author only.
    assert comment_authored_by_app({"user": {"type": "Bot"}}, None) is True
    assert comment_authored_by_app({"user": {"type": "User"}}, None) is False



# --- fix tokens ask for `workflows: write` when the App has it, and learn when it does not -----
from codna import webhook_github  # noqa: E402

class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload


def _fake_post(monkeypatch, responses):
    import httpx

    sent = []

    def post(url, *, headers, json, timeout, follow_redirects):
        sent.append(json)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(webhook_github, "_app_jwt", lambda app_id, pem, *, now: "jwt")
    monkeypatch.setattr(webhook_github, "normalize_private_key_pem", lambda pem: pem, raising=False)
    return sent


def test_fix_token_requests_workflows_and_remembers_it_is_granted(monkeypatch):
    webhook_github._WORKFLOWS_GRANTED.clear()
    sent = _fake_post(monkeypatch, [_Resp(201, {"token": "t1"})])
    assert webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="fix") == "t1"
    assert sent[0]["permissions"]["workflows"] == "write"
    assert webhook_github._WORKFLOWS_GRANTED["42"] is True


def test_fix_token_falls_back_without_workflows_when_github_says_not_granted(monkeypatch):
    webhook_github._WORKFLOWS_GRANTED.clear()
    sent = _fake_post(monkeypatch, [
        _Resp(422, text='{"message":"The permissions requested are not granted to this installation."}'),
        _Resp(201, {"token": "t2"}),
    ])
    assert webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="fix") == "t2"
    assert "workflows" in sent[0]["permissions"] and "workflows" not in sent[1]["permissions"]
    assert webhook_github._WORKFLOWS_GRANTED["42"] is False
    # the next mint for that installation does not ask again
    sent2 = _fake_post(monkeypatch, [_Resp(201, {"token": "t3"})])
    webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="fix")
    assert len(sent2) == 1 and "workflows" not in sent2[0]["permissions"]


def test_review_tokens_never_ask_for_workflows(monkeypatch):
    webhook_github._WORKFLOWS_GRANTED.clear()
    webhook_github._STATUSES_GRANTED.clear()
    sent = _fake_post(monkeypatch, [_Resp(201, {"token": "r"})])
    webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="review")
    assert "workflows" not in sent[0]["permissions"] and "contents" in sent[0]["permissions"]


# --- review tokens ask for `statuses: read` (Commit statuses) so the PR head's failed commit statuses
# --- -- a Vercel deployment reports as one -- are visible to failing_required_checks; an installation
# --- that never granted it makes GitHub refuse the whole mint (422), so the token is minted once more
# --- without it, the answer is remembered, and the operator log says so (thyn-ai/codna-site#57).

def test_review_token_requests_statuses_read_and_remembers_it_is_granted(monkeypatch, capsys):
    webhook_github._WORKFLOWS_GRANTED.clear()
    webhook_github._STATUSES_GRANTED.clear()
    sent = _fake_post(monkeypatch, [_Resp(201, {"token": "r1"})])
    assert webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="review") == "r1"
    assert len(sent) == 1  # one mint, no fallback round-trip
    assert sent[0]["permissions"] == {"contents": "read", "pull_requests": "write", "checks": "write", "statuses": "read"}
    assert sent[0]["repositories"] == ["app"]
    assert webhook_github._STATUSES_GRANTED["42"] is True
    assert "statuses_permission_missing" not in capsys.readouterr().err


def test_review_token_falls_back_without_statuses_when_github_says_not_granted(monkeypatch, capsys):
    webhook_github._WORKFLOWS_GRANTED.clear()
    webhook_github._STATUSES_GRANTED.clear()
    sent = _fake_post(monkeypatch, [
        _Resp(422, text='{"message":"The permissions requested are not granted to this installation."}'),
        _Resp(201, {"token": "r2"}),
    ])
    assert webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="review") == "r2"
    assert sent[0]["permissions"]["statuses"] == "read" and "statuses" not in sent[1]["permissions"]
    # every other permission is identical on the retry: only the ungranted one is dropped
    assert {k: v for k, v in sent[0]["permissions"].items() if k != "statuses"} == sent[1]["permissions"]
    assert sent[1]["permissions"] == {"contents": "read", "pull_requests": "write", "checks": "write"}
    assert webhook_github._STATUSES_GRANTED["42"] is False
    err = capsys.readouterr().err
    import json as _json
    logged = _json.loads(next(ln for ln in err.splitlines() if "statuses_permission_missing" in ln))
    assert logged["event"] == "statuses_permission_missing" and logged["level"] == "warning"
    assert logged["installation_id"] == "42" and logged["repo"] == "acme/app" and logged["kind"] == "review"
    assert "Commit statuses: read" in logged["remedy"]
    assert "r2" not in err and "jwt" not in err  # never a token in the log
    # the next mint for that installation does not ask again, and does not log again
    sent2 = _fake_post(monkeypatch, [_Resp(201, {"token": "r3"})])
    assert webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind="review") == "r3"
    assert len(sent2) == 1 and "statuses" not in sent2[0]["permissions"]
    assert "statuses_permission_missing" not in capsys.readouterr().err
    # a DIFFERENT installation is asked on its own account (multi-tenant: grants are per installation)
    sent3 = _fake_post(monkeypatch, [_Resp(201, {"token": "r4"})])
    webhook_github.installation_token("1", "pem", 43, repo_full_name="other/app", kind="review")
    assert sent3[0]["permissions"]["statuses"] == "read" and webhook_github._STATUSES_GRANTED["43"] is True


def test_review_mint_422_for_another_reason_is_still_an_error_not_a_silent_retry(monkeypatch):
    """A 422 that is not about permissions (a repository the installation cannot see) is raised as
    before: the fallback exists for the one ungranted permission, not to mask every refusal."""
    webhook_github._STATUSES_GRANTED.clear()
    sent = _fake_post(monkeypatch, [
        _Resp(422, text='{"message":"There is at least one repository that does not exist or is not accessible to the parent installation."}'),
    ])
    with pytest.raises(WebhookError) as exc:
        webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/gone", kind="review")
    assert exc.value.code == "installation_token_failed" and len(sent) == 1
    assert "42" not in webhook_github._STATUSES_GRANTED  # nothing learned from an unrelated refusal


def test_fix_and_other_tokens_never_ask_for_statuses(monkeypatch):
    webhook_github._WORKFLOWS_GRANTED.clear()
    webhook_github._STATUSES_GRANTED.clear()
    for kind in ("fix", "secure", "queue", "ci_triage", "ci_rerun"):
        sent = _fake_post(monkeypatch, [_Resp(201, {"token": "t"})])
        webhook_github.installation_token("1", "pem", 42, repo_full_name="acme/app", kind=kind)
        assert "statuses" not in sent[0]["permissions"], kind
    assert token_permissions_for("fix", statuses=True) == token_permissions_for("fix")  # the flag is review-only



def test_workflows_differ_compares_path_and_blob_sha_sets(monkeypatch):
    import httpx

    files = {
        "main": [{"type": "file", "path": ".github/workflows/security.yml", "sha": "aaa"}],
        "abc": [{"type": "file", "path": ".github/workflows/security.yml", "sha": "bbb"}],
        "same": [{"type": "file", "path": ".github/workflows/security.yml", "sha": "aaa"}],
    }

    def get(url, *, headers, timeout, follow_redirects, params=None):
        if url.endswith("/repos/acme/app"):
            return _Resp(200, {"default_branch": "main"})
        ref = (params or {}).get("ref")
        if ref == "none":
            return _Resp(404, {})
        return _Resp(200, files[ref])

    monkeypatch.setattr(httpx, "get", get)
    assert webhook_github.workflows_differ_from_default("acme/app", "t", "abc") is True
    assert webhook_github.workflows_differ_from_default("acme/app", "t", "same") is False
    assert webhook_github.workflows_differ_from_default("acme/app", "t", "none") is True  # no dir vs one file
    webhook_github._WORKFLOWS_GRANTED.clear()
    assert webhook_github.installation_has_workflows(42) is None
    webhook_github._WORKFLOWS_GRANTED["42"] = False
    assert webhook_github.installation_has_workflows(42) is False



def test_app_bot_identity_is_resolved_from_the_app_slug_and_bot_user_id(monkeypatch):
    import httpx

    webhook_github._BOT_IDENTITY.clear()
    monkeypatch.setattr(webhook_github, "_app_jwt", lambda app_id, pem, *, now: "jwt")
    hits = []

    def get(url, *, headers, timeout, follow_redirects, params=None):
        hits.append(url)
        if url.endswith("/app"):
            return _Resp(200, {"slug": "codna-ai"})
        if url.endswith("/users/codna-ai%5Bbot%5D"):
            return _Resp(200, {"id": 293953567, "login": "codna-ai[bot]"})
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", get)
    assert webhook_github.app_bot_identity("1", "pem") == ("codna-ai[bot]", "293953567+codna-ai[bot]@users.noreply.github.com")
    assert webhook_github.app_bot_identity("1", "pem") == ("codna-ai[bot]", "293953567+codna-ai[bot]@users.noreply.github.com")
    assert len(hits) == 2  # memoized after the first resolution
    assert webhook_github.app_bot_identity(None, None) is None



def test_latest_completed_check_run_skips_non_verdict_conclusions(monkeypatch):
    """A merge group must inherit a VERDICT: a newer cancelled/stale/skipped run (a superseded retry,
    a stale head) says nothing about the code and must not block -- or wave through -- the queue."""
    from codna.webhook_github import latest_completed_check_run

    class _Resp:
        status_code = 200

        def json(self):
            return {"check_runs": [
                {"conclusion": "cancelled", "completed_at": "2026-09-19T03:00:00Z", "output": {"summary": "superseded"}, "html_url": "c"},
                {"conclusion": "skipped", "completed_at": "2026-09-19T02:59:00Z", "output": {"summary": "skip"}, "html_url": "s"},
                {"conclusion": "neutral", "completed_at": "2026-09-19T02:50:00Z", "output": {"summary": "**codna review** — 1 finding"}, "html_url": "n"},
                {"conclusion": "success", "completed_at": "2026-09-19T02:40:00Z", "output": {"summary": "older"}, "html_url": "o"},
            ]}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp())
    run = latest_completed_check_run("acme/app", "t", head_sha="h" * 40, name="codna review")
    assert run == {"conclusion": "neutral", "summary": "**codna review** — 1 finding", "html_url": "n"}

    class _NoVerdict(_Resp):
        def json(self):
            return {"check_runs": [{"conclusion": "stale", "completed_at": "2026-09-19T03:00:00Z", "output": {}, "html_url": ""}]}

    monkeypatch.setattr("httpx.get", lambda *a, **k: _NoVerdict())
    assert latest_completed_check_run("acme/app", "t", head_sha="h" * 40, name="codna review") is None


def test_complete_stale_check_runs_honours_the_conclusion_it_is_given(monkeypatch):
    """The event SHA of a review whose head moved before the job started is closed NEUTRAL (codna#569):
    `cancelled` is for a retry superseding its own dead predecessor."""
    from codna.webhook_github import complete_stale_check_runs

    fake, seen = _stale_httpx(runs=({"id": 31},))
    monkeypatch.setitem(sys.modules, "httpx", fake)
    n = complete_stale_check_runs("acme/app", "token", head_sha="7679ecbe", name="codna review",
                                  summary="head moved to 5a3ba5ff before the review started", conclusion="neutral")
    assert n == 1
    assert seen["get"]["url"].endswith("/repos/acme/app/commits/7679ecbe/check-runs")
    assert seen["patched"] == [(31, "neutral", "head moved to 5a3ba5ff before the review started")]
