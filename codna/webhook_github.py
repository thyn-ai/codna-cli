"""GitHub App API glue for Codna's webhook: scoped tokens, Check Runs, SARIF fetch.

Lazily imports httpx / cryptography (not base Codna deps) so the pure webhook core stays
stdlib-only. The token/check-run *payload builders* are pure and unit-tested; only the HTTP
calls need the network.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Mapping

from .webhook import WebhookError

_API = "https://api.github.com"


# installation id -> whether the App has the `workflows` permission there. Learned from GitHub's
# answer to the first fix-token request per process (422 when it is not granted), so the extra
# round-trip happens once, not per job.
_WORKFLOWS_GRANTED: dict[str, bool] = {}

# installation id -> whether the App has the `statuses` permission there (Commit statuses: read),
# learned the same way from the first review-token request per process. Requesting a permission the
# installation never granted makes GitHub refuse the WHOLE mint (422), so an installation without it
# gets a review token without `statuses` -- the review then sees check runs but no commit statuses
# (a Vercel deployment reports as one), and the operator log says so once.
_STATUSES_GRANTED: dict[str, bool] = {}


def token_permissions_for(kind: str, *, workflows: bool = False, statuses: bool = False) -> dict[str, str]:
    """Least-privilege installation-token permissions for a job kind.

    fix needs to push a branch + open a PR; secure only reads code + code-scanning results.
    Both may post a Check Run for status. ``workflows=True`` adds `workflows: write` to a fix
    token: GitHub refuses ANY push from an App without it when the new branch's
    .github/workflows files differ from the default branch -- which is every PR branch that
    predates a workflow change on main (thyn-ai/algenta#1003, 2026-09-17), even though the fix
    commit itself never touches a workflow.

    ``statuses=True`` adds `statuses: read` to a review token: the PR head's ``statusCheckRollup``
    (review_github.failing_required_checks) lists commit statuses -- a Vercel deployment reports as
    one -- only to a token that may read them; without it the review sees check runs alone and
    approved thyn-ai/codna-site#57 with no word about its FAILED required `Vercel` status.
    """
    if kind == "fix":
        perms = {"contents": "write", "pull_requests": "write", "checks": "write", "issues": "write"}
        if workflows:
            perms["workflows"] = "write"
        return perms
    if kind == "secure":
        return {"contents": "read", "security_events": "read", "checks": "write", "issues": "write"}
    if kind == "review":
        # Read code, write PR review comments + a check. No contents:write — review never pushes.
        perms = {"contents": "read", "pull_requests": "write", "checks": "write"}
        if statuses:
            perms["statuses"] = "read"
        return perms
    if kind == "queue":
        # Read the PR head + its check runs, write ONE check on the merge-group commit.
        return {"pull_requests": "read", "checks": "write"}
    if kind == "ci_triage":
        # Read the evidence behind a failed check suite: the Actions job's steps and log
        # (webhook_ci_triage). Minted separately from the fix token because an installation that
        # never granted `actions` answers 422 to the whole request, and the fix must still run then.
        return {"actions": "read", "checks": "read"}
    if kind == "ci_rerun":
        # Same, plus re-running the failed jobs of a run that died on infrastructure.
        return {"actions": "write", "checks": "read"}
    raise WebhookError("unknown_job_kind", f"no token scope for job kind {kind!r}")


def scoped_token_request(repo_full_name: str, kind: str, *, workflows: bool = False,
                         statuses: bool = False) -> dict[str, Any]:
    """Body for POST /app/installations/{id}/access_tokens — scoped to the one repo + minimal perms."""
    repo_name = repo_full_name.split("/", 1)[-1]
    return {"repositories": [repo_name],
            "permissions": token_permissions_for(kind, workflows=workflows, statuses=statuses)}


def normalize_private_key_pem(private_key_pem: str) -> str:
    """Normalize PEM values from secret stores.

    Some secret import paths preserve multiline PEMs, while dotenv-style import paths store
    escaped ``\n`` sequences. GitHub App auth must accept both without weakening validation.
    """
    value = private_key_pem.strip().strip('"').strip("'")
    if "\\n" in value and "\n" not in value:
        value = value.replace("\\n", "\n")
    return value.strip()


def _app_jwt(app_id: str, private_key_pem: str, *, now: int) -> str:
    import base64

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    def _b64url(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64url(json.dumps({"iat": now - 60, "exp": now + 540, "iss": str(app_id)}).encode())
    signing_input = f"{header}.{claims}".encode()
    key = serialization.load_pem_private_key(normalize_private_key_pem(private_key_pem).encode(), password=None)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{claims}.{_b64url(signature)}"


def app_auth_config_ready(app_id: str | None, private_key_pem: str | None) -> bool:
    """Return True only when App auth config is present and can sign a JWT locally."""
    if not (app_id and private_key_pem):
        return False
    try:
        _app_jwt(app_id, private_key_pem, now=1_700_000_000)
    except Exception:  # noqa: BLE001 - readiness reports false, not secret-bearing details
        return False
    return True


def installation_token(
    app_id: str,
    private_key_pem: str,
    installation_id: int,
    *,
    repo_full_name: str,
    kind: str,
) -> str:
    """Mint a short-lived installation token scoped to one repo with least-privilege perms."""
    if not (app_id and private_key_pem and installation_id):
        raise WebhookError("github_app_not_configured", "App id / private key / installation id required.")
    try:
        import time as _time

        import httpx
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("github_app_auth_unavailable", f"App auth needs cryptography+httpx: {exc}") from exc
    try:
        jwt = _app_jwt(app_id, private_key_pem, now=int(_time.time()))
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("github_app_auth_unavailable", f"cannot sign App JWT: {exc}") from exc

    def _mint(*, with_workflows: bool, with_statuses: bool):
        return httpx.post(
            f"{_API}/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {jwt}", "Accept": "application/vnd.github+json"},
            json=scoped_token_request(repo_full_name, kind, workflows=with_workflows, statuses=with_statuses),
            timeout=30.0, follow_redirects=True,
        )

    # The one OPTIONAL permission a kind asks for -- `workflows` (fix) or `statuses` (review) -- is
    # requested until this installation says it never granted it; GitHub answers 422 for the whole
    # mint then, so the token is minted once more without it and the answer is remembered per process.
    inst = str(installation_id)
    want_workflows = kind == "fix" and _WORKFLOWS_GRANTED.get(inst, True)
    want_statuses = kind == "review" and _STATUSES_GRANTED.get(inst, True)
    resp = _mint(with_workflows=want_workflows, with_statuses=want_statuses)
    if (want_workflows or want_statuses) and resp.status_code == 422 and "permission" in resp.text.lower():
        if want_workflows:
            # The App has not been granted `workflows` on this installation: remember, mint without it.
            _WORKFLOWS_GRANTED[inst] = False
        if want_statuses:
            # Not granted `Commit statuses: read` here (multi-tenant: each installation grants the
            # App's new permissions on its own schedule). The review runs on check runs alone; a
            # failed commit status such as Vercel cannot be named until the installation grants it.
            _STATUSES_GRANTED[inst] = False
            _warn("statuses_permission_missing", installation_id=inst, repo=repo_full_name, kind=kind,
                  effect="commit statuses (e.g. Vercel) are invisible to the review's red-head check",
                  remedy="accept the App's 'Commit statuses: read' permission on this installation")
        resp = _mint(with_workflows=False, with_statuses=False)
    elif (want_workflows or want_statuses) and resp.status_code < 300:
        if want_workflows:
            _WORKFLOWS_GRANTED[inst] = True
        if want_statuses:
            _STATUSES_GRANTED[inst] = True
    if resp.status_code >= 300:
        raise WebhookError("installation_token_failed", f"installation token: {resp.status_code} {resp.text[:200]}")
    token = resp.json().get("token")
    if not token:
        raise WebhookError("installation_token_missing", "GitHub returned no installation token.")
    return str(token)


def check_run_payload(name: str, head_sha: str, *, status: str, conclusion: str | None, summary: str) -> dict[str, Any]:
    """Pure builder for a Check Run create/update body (queued -> in_progress -> completed)."""
    body: dict[str, Any] = {"name": name, "head_sha": head_sha, "status": status,
                            "output": {"title": name, "summary": summary}}
    if status == "completed":
        body["conclusion"] = conclusion or "neutral"
    return body


def create_check_run(repo_full_name: str, token: str, *, name: str, head_sha: str, summary: str,
                     status: str = "in_progress") -> int | None:
    """Create a Check Run so status shows in the PR UI. Returns its id.

    ``in_progress`` by default (a worker that is about to run the job). The ingress creates a
    review's run ``queued`` the moment the delivery is accepted, so the check is visible while
    the job waits for a worker slot; ``start_check_run`` flips it when a worker claims the row.
    """
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("check_run_unavailable", f"Check Run creation needs httpx: {exc}") from exc
    resp = httpx.post(
        f"{_API}/repos/{repo_full_name}/check-runs",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json=check_run_payload(name, head_sha, status=status, conclusion=None, summary=summary),
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        raise WebhookError("check_run_create_failed", f"create check run: {resp.status_code} {resp.text[:200]}")
    check_run_id = resp.json().get("id")
    if not check_run_id:
        raise WebhookError("check_run_create_failed", "create check run: GitHub returned no check_run id")
    return check_run_id


def start_check_run(repo_full_name: str, token: str, check_run_id: int, *, name: str, summary: str) -> None:
    """Flip a ``queued`` Check Run to ``in_progress``: the worker that claimed its job is running it.

    ``started_at`` is set to now, so the run's duration in the UI is the job's own, not the time
    it waited in the queue (GitHub Actions stamps its job runs the same way)."""
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("check_run_unavailable", f"Check Run update needs httpx: {exc}") from exc
    import time as _time

    body = check_run_payload(name, "", status="in_progress", conclusion=None, summary=summary)
    body.pop("head_sha", None)  # a PATCH never re-anchors the run
    body["started_at"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
    resp = httpx.patch(
        f"{_API}/repos/{repo_full_name}/check-runs/{check_run_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json=body, timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        raise WebhookError("check_run_update_failed", f"start check run: {resp.status_code} {resp.text[:200]}")


def check_run_status(repo_full_name: str, token: str, check_run_id: int) -> str | None:
    """``queued`` / ``in_progress`` / ``completed`` for one Check Run, or None when it cannot be read
    (not ours, gone, or a transport error). A retry asks this before reusing the run its row
    carries: a run the previous attempt already completed is not reopened, a fresh one replaces it."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(
            f"{_API}/repos/{repo_full_name}/check-runs/{int(check_run_id)}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=20.0, follow_redirects=True,
        )
    except Exception:  # noqa: BLE001
        return None
    data = _check_run_json(resp)
    status = data.get("status") if data else None
    return str(status) if status else None


def update_check_run(repo_full_name: str, token: str, check_run_id: int, *, conclusion: str, summary: str,
                     name: str = "Codna") -> None:
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("check_run_unavailable", f"Check Run update needs httpx: {exc}") from exc
    resp = httpx.patch(
        f"{_API}/repos/{repo_full_name}/check-runs/{check_run_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json=check_run_payload(name, "", status="completed", conclusion=conclusion, summary=summary),
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        raise WebhookError("check_run_update_failed", f"update check run: {resp.status_code} {resp.text[:200]}")


def _warn(event: str, **fields: Any) -> None:
    """One JSON line on stderr, the worker's log shape, for a condition an operator must see."""
    payload: dict[str, Any] = {"service": "codna-webhook-worker", "level": "warning", "event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _check_run_json(resp: Any) -> dict[str, Any] | None:
    """The run a 2xx answer carries, else None (error status, or a body that is not the run)."""
    if resp.status_code >= 300:
        return None
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def complete_check_run_if_open(repo_full_name: str, token: str, check_run_id: int, *, conclusion: str,
                               summary: str, name: str) -> bool:
    """Complete ONE known Check Run, but only while it is still open. Returns True if it did.

    The restart reconciler (webhook_resume) knows the exact id of the run an interrupted job
    opened. `codna review --post` completes that same run with its findings, and the kill can
    land right after -- so the run is read first and left alone once it is ``completed``.

    What that guarantee is worth, honestly. GitHub has no conditional update for Check Runs (no
    If-Match, no status-guarded PATCH), and a PATCH by the owning App on an already-completed run
    answers 200 and replaces its output. So a completion that lands between the read and the
    write here is overwritten, and afterwards nothing on the API tells: the run simply carries
    this summary. The window is kept as small as the API allows -- one request on an already-open
    connection: read, then write on the same client, nothing in between. A completion that lands
    right AFTER the write is visible: the run is read back once more and, when its output is not
    what was written, a ``check_run_clobbered`` warning names the run and its suite so the race is
    at least on record. Who could be writing at that moment: only a ``codna`` child of the
    previous process that outlived it (``WorkerPool.stop`` joins its threads but never signals
    the job's process group), and on the hosted app the machine replacement kills that child
    before this process boots. Never raises (best-effort hygiene, same contract as
    complete_stale_check_runs)."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return False
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"{_API}/repos/{repo_full_name}/check-runs/{check_run_id}"
    body = check_run_payload(name, "", status="completed", conclusion=conclusion, summary=summary)
    try:
        with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
            before = _check_run_json(client.get(url))
            if before is None or before.get("status") == "completed":
                return False  # missing, not ours to read, or already completed: its findings stay
            if client.patch(url, json=body).status_code >= 300:
                return False  # completed by someone else in between (a 422), or not ours
            try:
                after = _check_run_json(client.get(url))
            except Exception:  # noqa: BLE001 -- the read-back is diagnostics only
                after = None
    except Exception:  # noqa: BLE001 -- transport error at any step
        return False
    if after is not None and after.get("status") == "completed":
        output = after.get("output") or {}
        if (output.get("title"), output.get("summary")) != (name, summary):
            # Someone completed this run right after this write (its content won, this
            # cancellation note is gone): the interrupted job's own CLI, still alive somewhere.
            _warn("check_run_clobbered", repo=repo_full_name, check_run_id=int(check_run_id),
                  check_suite_id=(after.get("check_suite") or {}).get("id"),
                  head_sha=after.get("head_sha"), name=name,
                  observed_title=str(output.get("title") or "")[:200],
                  observed_summary=str(output.get("summary") or "")[:200],
                  observed_conclusion=after.get("conclusion"), written_summary=summary[:200])
    return True


def complete_stale_check_runs(
    repo_full_name: str, token: str, *, head_sha: str, name: str, summary: str,
    conclusion: str = "cancelled",
) -> int:
    """Complete (as ``conclusion``, default ``cancelled``) every still-open (``queued`` or
    ``in_progress``) Check Run named ``name`` on ``head_sha``. Returns how many were closed.
    Best-effort hygiene: never raises.

    A run this worker opened and then lost -- killed mid-job by a redeploy, or a crash before
    update_check_run -- stays "in progress" on that commit forever, because only the App that
    created a Check Run may complete it and nothing else ever will. Observed live:
    thyn-ai/algenta@cc26ddd6 `codna fix` in_progress from 2026-09-17 04:49 onward. Called right
    before a fresh run is created for the same (sha, name), so a retry supersedes its dead
    predecessor instead of stacking a second spinner next to it -- and, as ``neutral``, on the
    event SHA of a review whose pull request head had already moved when the job started
    (thyn-ai/codna#569): a run left ``in_progress`` on a commit nobody is merging still strands
    automation that waits for "all checks complete". ``queued`` runs are swept too: the ingress
    opens a review's run in that state, and one whose row never reached a worker is as stranded
    as an ``in_progress`` one. The listing endpoint filters on one status per request.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return 0
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    closed = 0
    for status in ("in_progress", "queued"):
        try:
            resp = httpx.get(
                f"{_API}/repos/{repo_full_name}/commits/{head_sha}/check-runs",
                headers=headers, params={"check_name": name, "status": status, "per_page": 50},
                timeout=30.0, follow_redirects=True,
            )
        except Exception:  # noqa: BLE001
            continue
        if resp.status_code >= 300:
            continue
        try:
            runs = (resp.json() or {}).get("check_runs") or []
        except Exception:  # noqa: BLE001
            continue
        for run in runs:
            run_id = run.get("id")
            if not run_id:
                continue
            try:
                update_check_run(
                    repo_full_name, token, run_id, conclusion=conclusion, summary=summary, name=name
                )
                closed += 1
            except Exception:  # noqa: BLE001 — someone else's run, already completed, or a transport error
                continue
    return closed


def find_open_pr_by_marker(repo_full_name: str, token: str, marker: str, *, max_pages: int = 3,
                           open_only: bool = False) -> str | None:
    """Return the html_url of an existing PR whose body carries ``marker``, else None.

    Used for idempotency: before a fix job opens a PR, the worker checks whether a prior
    (possibly crashed-then-recovered) attempt already opened one. Default lists PRs newest-first
    (state=all), so a just-opened PR is found immediately. ``open_only=True`` restricts to OPEN PRs
    — for the comment-fix path, a CLOSED prior fix PR must NOT block re-requesting a fresh fix.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    for page in range(1, max_pages + 1):
        resp = httpx.get(
            f"{_API}/repos/{repo_full_name}/pulls",
            headers=headers,
            params={"state": "open" if open_only else "all", "sort": "created",
                    "direction": "desc", "per_page": 100, "page": page},
            timeout=30.0, follow_redirects=True,
        )
        if resp.status_code >= 300:
            return None
        items = resp.json()
        if not items:
            break
        for pr in items:
            if marker in (pr.get("body") or ""):
                return pr.get("html_url")
    return None


_BOT_IDENTITY: dict[str, tuple[str, str]] = {}


def app_bot_identity(app_id: str | None, private_key_pem: str | None) -> tuple[str, str] | None:
    """(git user.name, git user.email) for this App's bot account, e.g.
    ("codna-ai[bot]", "293953567+codna-ai[bot]@users.noreply.github.com") -- the identity GitHub
    links to the App, so its commits resolve to a real account and CLA bots can allowlist it.
    Memoized per process; None when it cannot be resolved (caller keeps the default identity)."""
    if not (app_id and private_key_pem):
        return None
    if str(app_id) in _BOT_IDENTITY:
        return _BOT_IDENTITY[str(app_id)]
    try:
        import time as _time

        import httpx

        jwt = _app_jwt(str(app_id), private_key_pem, now=int(_time.time()))
        app = httpx.get(f"{_API}/app", headers={"Authorization": f"Bearer {jwt}", "Accept": "application/vnd.github+json"},
                        timeout=30.0, follow_redirects=True)
        slug = (app.json() or {}).get("slug") if app.status_code < 300 else None
        if not slug:
            return None
        user = httpx.get(f"{_API}/users/{slug}%5Bbot%5D", headers={"Accept": "application/vnd.github+json"},
                         timeout=30.0, follow_redirects=True)
        bot_id = (user.json() or {}).get("id") if user.status_code < 300 else None
        if not bot_id:
            return None
    except Exception:  # noqa: BLE001 — identity is a nicety; never fail a job over it
        return None
    identity = (f"{slug}[bot]", f"{bot_id}+{slug}[bot]@users.noreply.github.com")
    _BOT_IDENTITY[str(app_id)] = identity
    return identity


def installation_has_workflows(installation_id: int | str | None) -> bool | None:
    """What the last fix-token mint learned about the `workflows` permission on this installation:
    True / False, or None when no fix token has been minted in this process yet."""
    return _WORKFLOWS_GRANTED.get(str(installation_id)) if installation_id is not None else None


def _workflow_files(repo_full_name: str, token: str, ref: str) -> set[tuple[str, str]] | None:
    """(path, blob sha) of every file under .github/workflows at *ref*; empty set when the directory
    does not exist; None on any other failure."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(
            f"{_API}/repos/{repo_full_name}/contents/.github/workflows",
            params={"ref": ref},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=30.0, follow_redirects=True,
        )
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code == 404:
        return set()
    if resp.status_code >= 300:
        return None
    try:
        entries = resp.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(entries, list):
        return None
    return {(str(e.get("path")), str(e.get("sha"))) for e in entries if isinstance(e, dict) and e.get("type") == "file"}


def workflows_differ_from_default(repo_full_name: str, token: str, head_sha: str) -> bool | None:
    """True when the branch at *head_sha* carries different .github/workflows files than the default
    branch -- the condition under which GitHub refuses ANY push by an App without the `workflows`
    permission, even one that touches no workflow. None when it cannot be determined."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        repo = httpx.get(
            f"{_API}/repos/{repo_full_name}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=30.0, follow_redirects=True,
        )
        default_branch = (repo.json() or {}).get("default_branch") if repo.status_code < 300 else None
    except Exception:  # noqa: BLE001
        default_branch = None
    if not default_branch:
        return None
    at_head = _workflow_files(repo_full_name, token, head_sha)
    at_default = _workflow_files(repo_full_name, token, default_branch)
    if at_head is None or at_default is None:
        return None
    return at_head != at_default


def collaborator_permission(repo_full_name: str, token: str, username: str) -> str | None:
    """The repository permission GitHub grants *username*: admin | maintain | write | triage | read |
    none. None on any failure (the caller decides how to degrade). This is the authoritative answer
    the webhook payload's author_association is not."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(
            f"{_API}/repos/{repo_full_name}/collaborators/{username}/permission",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=30.0, follow_redirects=True,
        )
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code >= 300:
        return None
    try:
        permission = (resp.json() or {}).get("permission")
    except Exception:  # noqa: BLE001
        return None
    return str(permission) if permission else None


def get_pull_review_comment(repo_full_name: str, token: str, comment_id: int) -> dict | None:
    """GET one PR review comment (the parent a user replied `@codna fix` to). None on any failure."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    resp = httpx.get(
        f"{_API}/repos/{repo_full_name}/pulls/comments/{comment_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        return None
    body = resp.json()
    return body if isinstance(body, dict) else None


def comment_authored_by_app(comment: Mapping[str, Any], app_id: str | int | None) -> bool:
    """True iff a review comment was authored by THIS GitHub App (not a human who pasted a marker).

    ``performed_via_github_app.id`` is set by GitHub and cannot be forged by a user, so it is the
    strong signal; fall back to a Bot-type author only when the App id is unknown."""
    via = comment.get("performed_via_github_app")
    if app_id and isinstance(via, dict) and via.get("id") is not None:
        return str(via.get("id")) == str(app_id)
    user = comment.get("user") or {}
    return user.get("type") == "Bot"  # weaker fallback when the App id isn't configured


def post_review_comment_reply(repo_full_name: str, token: str, pr_number: int,
                              in_reply_to_id: int, body: str) -> str | None:
    """Reply into an existing PR review-comment thread. Returns the reply's html_url (or None)."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    resp = httpx.post(
        f"{_API}/repos/{repo_full_name}/pulls/{pr_number}/comments/{in_reply_to_id}/replies",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"body": body},
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        return None
    data = resp.json()
    return data.get("html_url") if isinstance(data, dict) else None


def fetch_issue_text(repo_full_name: str, token: str, issue_number: int) -> str | None:
    """GET an issue's title + body, formatted for `codna fix --issue`. None on any failure or an
    empty/whitespace-only body+title (nothing to describe the fix with)."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    resp = httpx.get(
        f"{_API}/repos/{repo_full_name}/issues/{issue_number}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        return None
    data = resp.json()
    if not isinstance(data, dict):
        return None
    title = str(data.get("title") or "").strip()
    body = str(data.get("body") or "").strip()
    text = f"{title}\n\n{body}".strip() if body else title
    return text or None


def post_issue_comment(repo_full_name: str, token: str, issue_number: int, body: str) -> str | None:
    """Post a normal issue/PR conversation comment. Returns html_url, or None on API failure."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    resp = httpx.post(
        f"{_API}/repos/{repo_full_name}/issues/{issue_number}/comments",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"body": body},
        timeout=30.0, follow_redirects=True,
    )
    if resp.status_code >= 300:
        return None
    data = resp.json()
    return data.get("html_url") if isinstance(data, dict) else None


def fetch_code_scanning_sarif(repo_full_name: str, token: str, *, dest_dir: str) -> str:
    """Download the latest code-scanning analysis SARIF to dest_dir; returns the path."""
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise WebhookError("sarif_fetch_unavailable", f"SARIF fetch needs httpx: {exc}") from exc
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    listing = httpx.get(
        f"{_API}/repos/{repo_full_name}/code-scanning/analyses",
        headers=headers, params={"per_page": 1}, timeout=30.0, follow_redirects=True,
    )
    if listing.status_code >= 300:
        raise WebhookError("code_scanning_list_failed", f"list analyses: {listing.status_code} {listing.text[:200]}")
    analyses = listing.json()
    if not analyses:
        raise WebhookError("no_code_scanning_analysis", f"no code-scanning analysis for {repo_full_name}.")
    sarif = httpx.get(
        f"{_API}/repos/{repo_full_name}/code-scanning/analyses/{analyses[0]['id']}",
        headers={**headers, "Accept": "application/sarif+json"}, timeout=60.0, follow_redirects=True,
    )
    if sarif.status_code >= 300:
        raise WebhookError("code_scanning_sarif_failed", f"fetch SARIF: {sarif.status_code} {sarif.text[:200]}")
    path = os.path.join(dest_dir, "code-scanning.sarif")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(sarif.text)
    return path


def pull_request_head_sha(repo_full_name: str, token: str, number: int) -> str | None:
    """The current head SHA of a pull request, or None when it cannot be read."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(f"{_API}/repos/{repo_full_name}/pulls/{number}", headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                         timeout=20.0, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    try:
        return str(resp.json()["head"]["sha"])
    except (ValueError, KeyError, TypeError):
        return None


# Check-run conclusions that are statements about the code (what a merge group may inherit).
_VERDICT_CONCLUSIONS = frozenset({"success", "neutral", "failure", "action_required"})


def latest_completed_check_run(repo_full_name: str, token: str, *, head_sha: str, name: str) -> dict | None:
    """The newest COMPLETED check run named ``name`` on ``head_sha`` (``conclusion``, ``summary``,
    ``html_url``), or None. Only this App's own runs are considered."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(f"{_API}/repos/{repo_full_name}/commits/{head_sha}/check-runs",
                         headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, params={"check_name": name, "status": "completed", "per_page": 20},
                         timeout=20.0, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    try:
        runs = [r for r in resp.json().get("check_runs", []) if isinstance(r, dict)]
    except ValueError:
        return None
    runs.sort(key=lambda r: str(r.get("completed_at") or ""), reverse=True)
    for run in runs:
        # Only a VERDICT may be inherited by a merge group. `cancelled` (a retry superseded this run),
        # `stale`, `timed_out` and `skipped` say nothing about the code; propagating one would block
        # a queue on a superseded run or, for `skipped`, wave an unreviewed PR through.
        if str(run.get("conclusion") or "") in _VERDICT_CONCLUSIONS:
            return {"conclusion": str(run["conclusion"]), "summary": str((run.get("output") or {}).get("summary") or ""),
                    "html_url": str(run.get("html_url") or "")}
    return None


# --- CI-failure triage (webhook_ci_triage) ------------------------------------------------------
_FAILED_CONCLUSIONS = frozenset({"failure", "timed_out"})


_LINK_NEXT_RE = re.compile(r'<([^<>]+)>\s*;[^,<>]*\brel="next"')


def _next_page(link_header: str) -> str | None:
    """The URL of ``rel="next"`` in a GitHub ``Link`` header, or None on the last page. Matched
    on the ``<url>; rel="next"`` shape itself rather than by splitting on commas, which a URL may
    contain."""
    m = _LINK_NEXT_RE.search(link_header or "")
    return m.group(1) if m else None


_SUITE_PAGES_MAX = 5  # 500 check runs; a suite beyond that is reported unknown, never falsely green


def failing_check_runs_in_suite(repo_full_name: str, token: str, suite_id: int) -> list[dict] | None:
    """The check runs of ``suite_id`` that are still failing, newest per name (``filter=latest``):
    ``id`` (== the Actions job id for github-actions runs), ``name``, ``html_url``, ``app_slug``.
    An empty list means the suite has no failing job any more (re-run); None means unknown.

    Walks every page (``Link: rel="next"``): an empty FIRST page of a larger suite would otherwise
    read as "green" and skip a real code failure. When the listing still comes up short of
    ``total_count`` the answer is None -- unknown runs the fix as before; a false green skips it.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url: str | None = f"{_API}/repos/{repo_full_name}/check-suites/{int(suite_id)}/check-runs"
    params: dict[str, Any] | None = {"per_page": 100, "filter": "latest"}
    runs: list = []
    total: Any = None
    for _ in range(_SUITE_PAGES_MAX):
        if not url:
            break
        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=20.0, follow_redirects=True)
        except Exception:  # noqa: BLE001
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        page = data.get("check_runs") if isinstance(data, dict) else None
        if not isinstance(page, list):
            return None
        runs.extend(page)
        if total is None:
            total = data.get("total_count")
        url, params = _next_page(resp.headers.get("link", "")), None
    if isinstance(total, int) and total > len(runs):
        return None
    out: list[dict] = []
    for run in runs:
        if not isinstance(run, dict) or run.get("status") != "completed":
            continue
        if str(run.get("conclusion") or "") not in _FAILED_CONCLUSIONS:
            continue
        out.append({"id": run.get("id"), "name": str(run.get("name") or ""),
                    "html_url": str(run.get("html_url") or ""),
                    "app_slug": str(((run.get("app") or {}).get("slug")) or "")})
    return out


def actions_job(repo_full_name: str, token: str, job_id: int) -> dict | None:
    """An Actions job: ``run_id``, ``run_attempt``, ``name``, ``steps`` (name + conclusion). Needs
    ``actions: read``; None on any failure (including the permission being absent)."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    try:
        resp = httpx.get(f"{_API}/repos/{repo_full_name}/actions/jobs/{int(job_id)}",
                         headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                         timeout=20.0, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {"run_id": data.get("run_id"), "run_attempt": data.get("run_attempt"),
            "name": str(data.get("name") or ""), "html_url": str(data.get("html_url") or ""),
            "steps": [{"name": str(s.get("name") or ""), "conclusion": s.get("conclusion")}
                      for s in (data.get("steps") or []) if isinstance(s, dict)]}


_LOG_TAIL_BYTES = 512 * 1024


def actions_job_log_tail(repo_full_name: str, token: str, job_id: int, *, max_lines: int = 400) -> str | None:
    """The last ``max_lines`` lines of an Actions job's raw log (the endpoint redirects to a signed
    blob URL; httpx drops the Authorization header across that origin change). Streams and keeps
    only the last 512 KiB so a multi-megabyte log never sits in memory on the 2 GB machine."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    buf = bytearray()
    try:
        with httpx.stream("GET", f"{_API}/repos/{repo_full_name}/actions/jobs/{int(job_id)}/logs",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          timeout=30.0, follow_redirects=True) as resp:
            if resp.status_code != 200:
                return None  # the `with` closes the response; nothing to drain
            for chunk in resp.iter_bytes():
                buf += chunk
                if len(buf) > _LOG_TAIL_BYTES:
                    del buf[:-_LOG_TAIL_BYTES]
    except Exception:  # noqa: BLE001
        return None
    text = buf.decode("utf-8", errors="replace")
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines) if lines else None


def rerun_failed_jobs(repo_full_name: str, token: str, run_id: int) -> bool:
    """Re-run only the failed jobs of a workflow run (``actions: write``). True on 201."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return False
    try:
        resp = httpx.post(f"{_API}/repos/{repo_full_name}/actions/runs/{int(run_id)}/rerun-failed-jobs",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          timeout=30.0, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return False
    return resp.status_code == 201
