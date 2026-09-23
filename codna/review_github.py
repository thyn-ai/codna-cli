"""Post a codna review to a GitHub PR: ONE review (inline comments + summary) + a check run.

Pure builders (comment body, review payload, existing-fingerprint scan, summary markdown) are
unit-tested offline; only :func:`post_review` touches the network. Each inline comment carries a
hidden ``<!-- codna:review:fp=… -->`` marker so a re-run on the same head does not repost a finding
that Codna already left (the CLI half of the no-spam invariant; the full lifecycle store lives in the
App path).
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import subprocess
from dataclasses import replace

from .review_findings import CATEGORIES, SEVERITIES, CodnaReviewFinding, ReviewResult

_API = "https://api.github.com"
CHECK_NAME = "codna review"
_MARKER_RE = re.compile(r"<!--\s*codna:review:fp=([0-9a-f]{6,40})\s*-->")
# Machine-readable finding payload embedded in every inline comment so the `@codna fix` wedge can
# reconstruct the finding EXACTLY (path/line/severity/category/title/explanation) — the human markdown
# above is product copy (reworded, truncated, non-ASCII separators) and is NOT a reliable parse source.
# The JSON is base64url-encoded so the marker can never be broken by braces / `-->` / quotes in the
# explanation, and the capture is a trivial, unambiguous character class.
_FINDING_MARKER_RE = re.compile(r"<!--\s*codna:finding\s+([A-Za-z0-9_=-]+)\s*-->")
# Stamped in each review body so the NEXT review can diff only the commits pushed since (incremental).
_REVIEWED_HEAD_RE = re.compile(r"<!--\s*codna:reviewed-head=([0-9a-f]{7,40})\s*-->")
# Stamped in each review body the GitHub App posts: the queue row id of the job that produced it
# (``CODNA_REVIEW_RUN_ID``). A second run of the SAME job -- a lease handed to another worker while
# the first is still finishing, a retry after the post but before the row was marked done -- finds
# its own stamp on the PR and does not post a second review. Inert (never emitted) for the CLI.
_RUN_MARKER_RE = re.compile(r"<!--\s*codna:review:run=([A-Za-z0-9_.:-]{1,64})\s*-->")

_SEV_EMOJI = {"high": "🔴", "medium": "🟡", "low": "⚪"}

# Approval policy. A review posted as COMMENT is invisible to everything that counts approvals --
# GitHub's "require approvals" rule and OpenSSF Scorecard's Code-Review check both look for a review
# in state APPROVED by someone other than the author (Scorecard's codeApproved probe; bot reviewers
# count, bot AUTHORS are skipped). Codna therefore APPROVES when its review is clean at the
# severities that mean "do not merge this yet", and only then. Low findings never block approval.
APPROVAL_BLOCKING_SEVERITIES = ("high", "medium")
_MAX_THREADS = 100  # reviewThreads page size; more than that -> unknown -> no approval (fail closed)
_MAX_FILE_PAGES = 30  # GitHub lists at most 3000 files per PR (30 pages of 100); more -> unknown


def review_event(result: ReviewResult, *, unresolved_blocking: int | None, approve: bool = True) -> str:
    """``APPROVE`` when approving is enabled, this pass found nothing at a blocking severity AND no
    earlier Codna finding at a blocking severity is still unresolved on the PR; otherwise ``COMMENT``.
    ``unresolved_blocking=None`` means the thread state could not be read: no approval."""
    if not approve or unresolved_blocking is None or unresolved_blocking > 0:
        return "COMMENT"
    if any(f.severity in APPROVAL_BLOCKING_SEVERITIES for f in result.findings):
        return "COMMENT"
    return "APPROVE"


def approval_note(event: str, result: ReviewResult, unresolved_blocking: int | None, *, approve: bool = True) -> str:
    """One line for the check summary saying whether this review approved the PR, and if not, why."""
    if event == "APPROVE":
        return "✅ Approved: no medium/high findings and no unresolved codna threads."
    if not approve:
        return "Approval is turned off for this repository (`review.approve: false`)."
    blocking = sum(1 for f in result.findings if f.severity in APPROVAL_BLOCKING_SEVERITIES)
    parts = []
    if blocking:
        parts.append(f"{blocking} medium/high finding(s) in this review")
    if unresolved_blocking is None:
        parts.append("the PR's review threads could not be read")
    elif unresolved_blocking:
        parts.append(f"{unresolved_blocking} earlier codna finding thread(s) at medium/high still unresolved -- "
                     "resolve them once addressed")
    return "⏸ Not approved: " + "; ".join(parts) + "." if parts else "⏸ Not approved."


def unresolved_blocking_findings(repo_slug: str, pr_number: int, token: str | None) -> int | None:
    """How many review threads Codna opened on this PR for a medium/high finding are still unresolved.

    The author resolves a thread once the finding is addressed -- that human act, not a re-review of
    the whole PR, is what lets an incremental review approve. Only threads whose first comment is a
    Bot's and carries a well-formed ``codna:finding`` marker count. Returns None when the answer is
    not certain (no token, API error, more than ``_MAX_THREADS`` threads): the caller must then not
    approve."""
    if not token or "/" not in (repo_slug or ""):
        return None
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    owner, name = repo_slug.split("/", 1)
    query = (
        "query($owner:String!,$name:String!,$number:Int!,$first:Int!){"
        " repository(owner:$owner,name:$name){ pullRequest(number:$number){"
        "  reviewThreads(first:$first){ pageInfo{ hasNextPage }"
        "   nodes{ isResolved comments(first:1){ nodes{ body author{ login __typename } } } } } } } }"
    )
    try:
        r = httpx.post(f"{_API}/graphql", headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                       json={"query": query, "variables": {"owner": owner, "name": name, "number": int(pr_number), "first": _MAX_THREADS}},
                       timeout=30.0, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code >= 300:
        return None
    try:
        threads = r.json()["data"]["repository"]["pullRequest"]["reviewThreads"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(threads, dict) or (threads.get("pageInfo") or {}).get("hasNextPage"):
        return None
    unresolved = 0
    for node in threads.get("nodes") or []:
        if not isinstance(node, dict) or node.get("isResolved"):
            continue
        first = ((node.get("comments") or {}).get("nodes") or [None])[0]
        if not isinstance(first, dict) or ((first.get("author") or {}).get("__typename") != "Bot"):
            continue
        finding = parse_codna_finding(str(first.get("body") or ""))
        if finding and finding.get("severity") in APPROVAL_BLOCKING_SEVERITIES:
            unresolved += 1
    return unresolved


def fp_marker(fingerprint: str) -> str:
    return f"<!-- codna:review:fp={fingerprint} -->"


def reviewed_head_marker(head_sha: str | None) -> str:
    return f"<!-- codna:reviewed-head={head_sha} -->" if head_sha else ""


def run_marker(run_id: str | None) -> str:
    """The hidden per-job stamp (see ``_RUN_MARKER_RE``); empty when no run id is known."""
    return f"<!-- codna:review:run={run_id} -->" if run_id else ""


def existing_run_ids(reviews: list[dict]) -> set[str]:
    """Run ids already stamped on this PR's reviews (scanned from review bodies)."""
    ids: set[str] = set()
    for rv in reviews or []:
        for m in _RUN_MARKER_RE.finditer(str(rv.get("body") or "")):
            ids.add(m.group(1))
    return ids


def finding_marker(f: CodnaReviewFinding) -> str:
    """A hidden marker carrying the full finding (base64url JSON), so the fix wedge reconstructs it
    losslessly regardless of what characters the explanation contains."""
    payload = {
        "fp": f.fingerprint,
        "path": f.path,
        "line": f.line,
        "end_line": f.end_line,
        "severity": f.severity,
        "category": f.category,
        "title": f.title,
        "explanation": f.explanation,
    }
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("ascii")
    return f"<!-- codna:finding {b64} -->"


def parse_codna_finding(body: str) -> dict | None:
    """Inverse of :func:`finding_marker`: reconstruct the finding dict from a comment body, or None.

    Only accepts a well-formed marker with an in-domain severity/category and a positive int line, so a
    user pasting arbitrary text can't masquerade as a Codna finding (authorship is verified separately)."""
    # Iterate ALL markers and keep the LAST that decodes to an in-domain finding. Codna appends the
    # real marker last, and a finding's explanation could itself contain marker-like text (e.g. a
    # review OF this marker code) that would otherwise shadow the genuine trailing marker.
    found: dict | None = None
    for m in _FINDING_MARKER_RE.finditer(body or ""):
        try:
            d = json.loads(base64.urlsafe_b64decode(m.group(1).encode("ascii")))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        path = str(d.get("path") or "").strip()
        try:
            line = int(d.get("line"))
        except (TypeError, ValueError):
            continue
        if not path or line < 1 or d.get("severity") not in SEVERITIES or d.get("category") not in CATEGORIES:
            continue
        found = {
            "fp": str(d.get("fp") or "") or None,
            "path": path,
            "line": line,
            "end_line": d.get("end_line") if isinstance(d.get("end_line"), int) else None,
            "severity": d["severity"],
            "category": d["category"],
            "title": str(d.get("title") or "").strip(),
            "explanation": str(d.get("explanation") or "").strip(),
        }
    return found


def comment_body(f: CodnaReviewFinding) -> str:
    """Inline-comment markdown for one finding (+ hidden fingerprint + JSON finding markers)."""
    head = f"**{_SEV_EMOJI.get(f.severity, '•')} {f.severity.upper()} · {f.category}** — {f.title}"
    lines = [head, "", f.explanation.strip()]
    if f.suggested_patch:
        fence = "suggestion" if _looks_like_suggestion(f.suggested_patch) else ""
        lines += ["", f"```{fence}", f.suggested_patch.strip(), "```"]
    lines += ["", "_Reply `@codna fix` and Codna opens a verified, test-green, risk-gated fix PR._"]
    lines += ["", fp_marker(f.fingerprint), finding_marker(f)]
    return "\n".join(lines)


def _looks_like_suggestion(patch: str) -> bool:
    # A GitHub ```suggestion block must be the literal replacement lines (no diff markers).
    return not any(ln[:1] in ("+", "-", "@") for ln in patch.splitlines())


def summary_body(result: ReviewResult, approval: str | None = None, *, run_id: str | None = None) -> str:
    """The review's top-level body: headline + any findings that could not be anchored inline
    (+ the approval line, when the caller decided one; + the per-job run stamp when the App runs
    it -- see :func:`run_marker`)."""
    n = len(result.findings)
    head_mark = reviewed_head_marker(result.head_sha)
    if run_id:
        head_mark = (head_mark + "\n" + run_marker(run_id)) if head_mark else run_marker(run_id)
    tail = (approval + "\n\n") if approval else ""
    if n == 0:
        return "**codna review** — no high-confidence issues found. ✅\n\n" + tail + fp_marker("none") + "\n" + head_mark
    lines = [f"**codna review** — {n} finding(s): "
             f"{len(result.inline_findings)} inline, {len(result.summary_findings)} in summary.", ""]
    for f in result.summary_findings:
        loc = f"`{f.path}:{f.line}`"
        lines.append(f"- {_SEV_EMOJI.get(f.severity, '•')} **{f.severity}/{f.category}** {loc} — "
                     f"{f.title}  {fp_marker(f.fingerprint)}")
        if f.explanation:
            lines.append(f"  <br>{f.explanation.strip()}")
    lines += ["", "_Reply `@codna fix` on a finding to have Codna open a verified, test-green, "
              "risk-gated fix PR._", ""]
    if approval:
        lines += [approval, ""]
    lines.append(head_mark)
    return "\n".join(lines)


def inline_comment_payload(f: CodnaReviewFinding) -> dict:
    """One entry in the review's ``comments[]`` (single-line or multi-line range)."""
    a = f.diff_anchor
    body = {"path": f.path, "line": a.line, "side": a.side, "body": comment_body(f)}
    if a.start_line is not None and a.start_side is not None:
        body["start_line"] = a.start_line
        body["start_side"] = a.start_side
    return body


def existing_fingerprints(comments: list[dict]) -> set[str]:
    """Fingerprints Codna has already posted on this PR (scanned from comment bodies)."""
    fps: set[str] = set()
    for c in comments or []:
        for m in _MARKER_RE.finditer(str(c.get("body") or "")):
            fps.add(m.group(1))
    return fps


def demote_to_summary(result: ReviewResult, *, keep_paths: set[str] | frozenset[str] = frozenset(),
                      drop_fingerprints: set[str] | frozenset[str] = frozenset()) -> ReviewResult:
    """A copy of ``result`` in which every inline finding whose path is not in ``keep_paths`` is
    routed to the summary instead (the default keeps none: every inline finding is demoted). A
    demoted finding loses only its diff anchor -- text, severity and fingerprint are intact and the
    review body lists it -- so a finding is never dropped because GitHub would refuse its anchor.

    ``drop_fingerprints`` are the fingerprints already threaded on the PR (:func:`existing_fingerprints`,
    the ``skip`` set dedup builds). A demoted finding with one of those is left out rather than
    listed: dedup keeps it out of the inline comments precisely because the PR already carries it,
    and the body must not say it a second time. The input is not modified; when no inline finding
    is outside ``keep_paths`` it is returned as is."""
    kept = [f for f in result.inline_findings if f.path in keep_paths]
    outside = [f for f in result.inline_findings if f.path not in keep_paths]
    if not outside:
        return result
    demoted = [replace(f, diff_anchor=None) for f in outside if f.fingerprint not in drop_fingerprints]
    return replace(result, inline_findings=kept, summary_findings=[*result.summary_findings, *demoted])


def build_review_payload(result: ReviewResult, *, skip_fingerprints: set[str] | None = None,
                         event: str = "COMMENT", approval: str | None = None,
                         pr_files: set[str] | None = None, run_id: str | None = None) -> dict:
    """Build the ``POST …/reviews`` body: ``event`` (COMMENT, or APPROVE per :func:`review_event`),
    summary body, inline comments for anchored findings whose fingerprint has not already been
    posted. Never REQUEST_CHANGES — the check conclusion is the gate, the review stays non-blocking.

    ``pr_files`` is the pull request's diff as GitHub lists it (:func:`fetch_pr_files`; None =
    unknown, filters nothing). An inline comment may only address a path in that set -- GitHub
    rejects the WHOLE review otherwise -- so a finding on any other path goes to the body instead,
    unless its fingerprint is already threaded on the PR (``skip_fingerprints``): then it is said
    nowhere, exactly as dedup keeps it out of the inline comments."""
    skip = skip_fingerprints or set()
    if pr_files is not None:
        result = demote_to_summary(result, keep_paths=pr_files, drop_fingerprints=skip)
    comments = [inline_comment_payload(f) for f in result.inline_findings if f.fingerprint not in skip]
    payload: dict = {"event": event, "body": summary_body(result, approval, run_id=run_id), "comments": comments}
    if result.head_sha:
        payload["commit_id"] = result.head_sha
    return payload


def fetch_pr(repo_slug: str, pr_number: int, token: str | None) -> dict:
    """GET the PR metadata codna review needs: base ref, head SHA, head ref, fork flag."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = httpx.get(f"{_API}/repos/{repo_slug}/pulls/{pr_number}", headers=headers, timeout=30.0, follow_redirects=True)
    if r.status_code >= 300:
        raise RuntimeError(f"fetch PR #{pr_number} failed: {r.status_code} {r.text[:200]}")
    d = r.json()
    base, head = d.get("base") or {}, d.get("head") or {}
    return {
        "base_ref": base.get("ref"),
        "head_sha": head.get("sha"),
        "head_ref": head.get("ref"),
        "is_fork": bool((head.get("repo") or {}).get("fork")) or (
            (head.get("repo") or {}).get("full_name") != repo_slug
        ),
    }


def fetch_pr_files(repo_slug: str, pr_number: int, token: str | None) -> set[str] | None:
    """The paths in the pull request's diff as GitHub sees it (``GET /pulls/{n}/files``, every page),
    or None when the answer is not certain: no httpx, a request failed, a page did not parse, or
    the PR has more files than GitHub serves. Best-effort by design -- None means "unknown", and an
    unknown set filters nothing (the 422 retry in :func:`post_review` still covers that case).

    This set, not the diff codna reviewed, is what GitHub resolves inline comment paths against. An
    incremental review's diff can differ from it (an "Update branch" merge in the range), and one
    comment on a path outside it fails the whole review with ``422 Path could not be resolved``."""
    try:
        import httpx
    except Exception:  # httpx is optional at import time; without it the set is simply unknown
        return None
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    paths: set[str] = set()
    for page in range(1, _MAX_FILE_PAGES + 1):
        try:
            r = httpx.get(f"{_API}/repos/{repo_slug}/pulls/{pr_number}/files", headers=headers,
                          params={"per_page": 100, "page": page}, timeout=30.0, follow_redirects=True)
            if r.status_code >= 300:
                return None
            batch = r.json()
        except Exception:  # a transport error or an unparseable page: the set is incomplete -> unknown
            return None
        if not isinstance(batch, list):
            return None
        for entry in batch:
            name = entry.get("filename") if isinstance(entry, dict) else None
            if name:
                paths.add(str(name))
        if len(batch) < 100:
            return paths
    return None  # more pages than GitHub serves: incomplete, so unknown


def fetch_review_context(repo_slug: str, pr_number: int, token: str | None, *, max_pages: int = 3) -> dict:
    """Gather cross-run context for a PR review (best-effort; empty on any failure):

    - ``last_reviewed_head``: the head SHA of Codna's most recent review (from the `codna:reviewed-head`
      marker in review bodies) → lets the caller review only commits pushed SINCE (incremental review).
    - ``prior_feedback``: human + other-bot PR comments (NOT Codna's own findings) so the reviewer can
      avoid duplicating points already raised.
    """
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return {"last_reviewed_head": None, "prior_feedback": None}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def _get(url: str) -> list:
        items: list = []
        for page in range(1, max_pages + 1):
            try:
                r = httpx.get(url, headers=headers, params={"per_page": 100, "page": page},
                              follow_redirects=True, timeout=30.0)
            except Exception:  # noqa: BLE001
                break
            if r.status_code >= 300:
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
        return items

    last_head = None
    for rv in _get(f"{_API}/repos/{repo_slug}/pulls/{pr_number}/reviews"):
        m = _REVIEWED_HEAD_RE.search(str(rv.get("body") or ""))
        if m:
            last_head = m.group(1)  # reviews come oldest-first → last match is the most recent review

    feedback: list[str] = []
    for kind, url in (("inline", f"{_API}/repos/{repo_slug}/pulls/{pr_number}/comments"),
                      ("issue", f"{_API}/repos/{repo_slug}/issues/{pr_number}/comments")):
        for c in _get(url):
            body = str(c.get("body") or "")
            # skip Codna's OWN comments (findings / markers) — only human + other-bot feedback
            if _MARKER_RE.search(body) or "codna:finding" in body or "codna:reviewed-head" in body:
                continue
            txt = body.strip()
            if not txt:
                continue
            author = (c.get("user") or {}).get("login", "?")
            loc = f" ({c.get('path')}:{c.get('line')})" if c.get("path") else ""
            feedback.append(f"- @{author}{loc}: {txt[:400]}")
    return {"last_reviewed_head": last_head, "prior_feedback": "\n".join(feedback[:40]) or None}


_LAST_PAGE_RE = re.compile(r'<([^>]*[?&]page=(\d+)[^>]*)>;\s*rel="last"')


_STAMP_TAIL_PAGES = 4  # the newest ~400 reviews; a same-job re-run is minutes to an hour behind its first run


def list_pr_reviews(repo_slug: str, pr_number: int, token: str | None) -> list[dict]:
    """The reviews that can carry a duplicate-run stamp: the first page and the LAST
    ``_STAMP_TAIL_PAGES`` pages as the ``Link: rel="last"`` header names them -- at most five
    requests however long the PR's history, each bounded at 15 s, because this sits on every App
    review's critical path. A stamp from the same job is among the newest reviews: a second run of
    one job follows its first by a retry backoff (minutes) or a lease expiry, never by weeks, and
    ~400 reviews is far more than a PR accrues in that window. Best-effort: an error returns what
    was read, and a missing stamp means "not seen", the fallback the fingerprint dedup provides."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return []
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{_API}/repos/{repo_slug}/pulls/{pr_number}/reviews"

    def _page(number: int) -> tuple[list[dict], str]:
        r = httpx.get(url, headers=headers, params={"per_page": 100, "page": number}, timeout=15.0, follow_redirects=True)
        if r.status_code >= 300:
            return [], ""
        batch = r.json()
        rows = [b for b in batch if isinstance(b, dict)] if isinstance(batch, list) else []
        return rows, str(r.headers.get("link", "") if hasattr(r, "headers") else "")

    reviews: list[dict] = []
    try:
        first, link = _page(1)
        reviews.extend(first)
        m = _LAST_PAGE_RE.search(link)
        last = int(m.group(2)) if m else 1
        for number in range(max(2, last - _STAMP_TAIL_PAGES + 1), last + 1):
            rows, _ = _page(number)
            reviews.extend(rows)
    except Exception:  # noqa: BLE001 -- best-effort listing
        pass
    return reviews


def repo_slug_from_url(url: str) -> str | None:
    """Derive ``owner/repo`` from a GitHub repo URL (ssh or https, with/without .git)."""
    m = re.search(r"github\.com[:/]+([^/\s]+/[^/\s]+?)(?:\.git)?/?$", (url or "").strip())
    return m.group(1) if m else None


def repo_slug_from_git(repo_dir: str) -> str | None:
    """Derive ``owner/repo`` from the origin remote (ssh or https GitHub URL)."""
    try:
        url = subprocess.run(
            ["git", "-C", repo_dir, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return repo_slug_from_url(url)


def parse_pr_arg(value: str, repo_dir: str) -> tuple[str | None, int | None]:
    """Parse ``--pr`` as ``123``, ``owner/repo#123``, or a PR URL → ``(repo_slug, pr_number)``."""
    value = (value or "").strip()
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", value)
    if m:
        return m.group(1), int(m.group(2))
    if "#" in value:
        slug, _, num = value.partition("#")
        slug = slug.strip() or repo_slug_from_git(repo_dir)
        return slug, (int(num) if num.strip().isdigit() else None)
    if value.isdigit():
        return repo_slug_from_git(repo_dir), int(value)
    return repo_slug_from_git(repo_dir), None


def head_moved_note(reviewed: str, current: str) -> str:
    """The approval line when the pull request's head is no longer the one this review read."""
    return (f"⏸ Not approved: the pull request head moved from {reviewed[:8]} to {current[:8]} while this "
            f"review ran; the current head gets a review of its own.")


# statusCheckRollup context states that mean "this check is red" (a CANCELLED run says nothing).
_RED_CHECK_RUN = frozenset({"FAILURE", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"})
_RED_STATUS = frozenset({"FAILURE", "ERROR"})
_ROLLUP_QUERY = (
    "query($owner:String!,$name:String!,$number:Int!,$after:String){"
    " repository(owner:$owner,name:$name){ pullRequest(number:$number){"
    "  commits(last:1){ nodes{ commit{ oid statusCheckRollup{ contexts(first:100,after:$after){"
    "   pageInfo{ hasNextPage endCursor }"
    "   nodes{ __typename"
    "    ... on CheckRun{ name status conclusion isRequired(pullRequestNumber:$number) }"
    "    ... on StatusContext{ context state isRequired(pullRequestNumber:$number) } } } } } } } } } }"
)
_ROLLUP_PAGES_MAX = 5  # 500 contexts; a rollup beyond that with nothing red on it is unknown, never green


def failing_required_checks(repo_slug: str, pr_number: int, token: str | None) -> list[dict] | None:
    """The REQUIRED checks that are red on the pull request's current head, as
    ``[{"name", "state", "head_sha"}]`` -- one GraphQL request in the normal case (the PR head's
    ``statusCheckRollup``, each context's ``isRequired`` for this PR; a rollup of more than 100
    contexts is walked page by page, at most ``_ROLLUP_PAGES_MAX``). An empty list means no required
    check is failing; None means the answer is not known (no token, no httpx, an error, a rollup
    longer than the pages read with nothing red on them) and the caller says nothing rather than
    guessing.

    Commit statuses (a Vercel deployment reports as one) are visible only to a token with
    ``statuses: read``; a token without it sees the check runs alone. Read right before posting, like
    the head itself: the review's approval means "no medium/high findings", and a reader who sees
    "Approved" next to a red required check must not take it for "safe to merge" (thyn-ai/codna-site#57
    was approved while its required Vercel status was FAILURE)."""
    if not token or "/" not in (repo_slug or ""):
        return None
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return None
    owner, name = repo_slug.split("/", 1)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    red: list[dict] = []
    after: str | None = None
    more = True
    for _ in range(_ROLLUP_PAGES_MAX):
        variables = {"owner": owner, "name": name, "number": int(pr_number), "after": after}
        try:
            r = httpx.post(f"{_API}/graphql", headers=headers, json={"query": _ROLLUP_QUERY, "variables": variables},
                           timeout=20.0, follow_redirects=True)
        except Exception:  # noqa: BLE001 -- best-effort: unknown, never a false "green"
            return None
        if r.status_code >= 300:
            return None
        try:
            commit = r.json()["data"]["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"]
            contexts = (commit.get("statusCheckRollup") or {}).get("contexts") or {}
            nodes = contexts.get("nodes") or []
            page = contexts.get("pageInfo") or {}
            head_sha = str(commit.get("oid") or "")
        except (ValueError, KeyError, TypeError, IndexError):
            return None
        if not isinstance(nodes, list):
            return None
        for node in nodes:
            if not isinstance(node, dict) or not node.get("isRequired"):
                continue
            if node.get("__typename") == "CheckRun":
                state = str(node.get("conclusion") or "").upper()
                if node.get("status") == "COMPLETED" and state in _RED_CHECK_RUN:
                    red.append({"name": str(node.get("name") or ""), "state": state, "head_sha": head_sha})
            elif node.get("__typename") == "StatusContext":
                state = str(node.get("state") or "").upper()
                if state in _RED_STATUS:
                    red.append({"name": str(node.get("context") or ""), "state": state, "head_sha": head_sha})
        more = bool(page.get("hasNextPage"))
        after = page.get("endCursor") if more else None
        if not more or not isinstance(after, str) or not after:
            break
    if more:
        # Pages were left unread (more than the cap, or a cursor GitHub did not hand back). Red ones
        # already seen are still named -- a partial warning beats none -- but nothing red so far is
        # "unknown", never "green".
        return red or None
    return red


def red_head_note(checks: list[dict]) -> str:
    """One line for the review body and the check summary naming the required checks that are red on
    the head at review time, so "Approved" is never read as "safe to merge"."""
    if not checks:
        return ""
    names = ", ".join(f"`{c.get('name') or '?'}` ({str(c.get('state') or 'red').lower()})" for c in checks)
    sha = str((checks[0].get("head_sha") or ""))[:8]
    where = f"head {sha} is red" if sha else "the head is red"
    return (f"🔴 Heads-up: {where} -- required check(s) failing at review time: {names}. "
            "This review's verdict is about the diff, not a merge go-ahead.")


def post_review(
    result: ReviewResult, *, repo_slug: str, pr_number: int, token: str, dedup: bool = True,
    check_run_id: int | None = None, approve: bool = True,
    check_head_sha: str | None = None, pr_head_sha: str | None = None, run_id: str | None = None,
    failing_checks: list[dict] | None = None,
) -> dict:
    """Post ONE review (+ the "codna review" check) to the PR. Returns a small status dict.

    ``failing_checks`` (:func:`failing_required_checks`, read by the caller right before posting, like
    ``pr_head_sha``) are the REQUIRED checks red on the PR head at review time. They change no
    verdict -- the event is still decided by the findings alone -- but when there are any, the
    approval line is followed by one :func:`red_head_note` naming them, in the review body and in the
    check summary, so "Approved" is never mistaken for "safe to merge". None = unknown, nothing said.

    ``approve`` (``review.approve`` in codna.yaml, default on) lets a clean review be posted as an
    APPROVE -- see :func:`review_event` for exactly when. A PR the App itself authored cannot be
    approved by the App (GitHub answers 422); that case falls back to a COMMENT.

    ``pr_head_sha`` is the pull request's head as the caller read it right before posting. When it
    differs from ``result.head_sha`` the head moved while the review ran: the review is still posted
    against the commit it read, but never as an APPROVE -- a pull request approval counts for the
    whole pull request, and nobody reviewed the new head (thyn-ai/codna#569). ``check_head_sha`` is
    the commit the caller's ``check_run_id`` is anchored to. Both are returned as they are (None =
    unknown), next to the reviewed head, so a caller of ``codna review --json`` can see the three
    disagree; when this function creates the check itself, it is anchored to the reviewed head.

    ``dedup`` fetches existing PR review comments and skips fingerprints Codna already posted, so a
    re-run on the same head does not duplicate threads. ``check_run_id`` (the webhook passes it as
    ``CODNA_CHECK_RUN_ID``) is a Check Run the caller already opened for this review: it is
    completed with the findings instead of a second run being created, so the PR shows exactly one
    "codna review" check.

    Inline comments are anchored only to paths in the PR's diff as GitHub lists it
    (:func:`fetch_pr_files`, read when there is an inline finding to check); a finding elsewhere is
    listed in the review body. Should GitHub still answer 422 to a payload that carries inline
    comments, the review is retried ONCE with every inline finding demoted to the body
    (:func:`demote_to_summary`): no finding is lost and the check does not turn red because GitHub
    would not place a thread. That retry composes with the APPROVE -> COMMENT fallback: each fires
    at most once, the approval one first. Re-routing never changes what this pass FOUND: the event
    and the approval note are decided from the full result before any finding moves, and a finding
    already threaded on the PR (dedup's ``skip``) is left out of a demotion rather than repeated.

    The returned counters partition the inline findings this pass found: every one is exactly one of
    ``inline_posted``, ``demoted_to_summary`` (said in the body) or ``skipped_duplicates`` (already
    threaded on the PR, wherever it would otherwise have gone).

    ``run_id`` (the webhook passes the queue row id as ``CODNA_REVIEW_RUN_ID``) is stamped into
    the review body. When the PR ALREADY carries a review with this stamp -- the same job ran
    twice -- no review is created (``posted_review`` False, ``duplicate_run`` True); the check
    run is still completed, because the caller opened it and nothing else will."""
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"posting a review needs httpx: {exc}") from exc

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    duplicate_run = False
    if run_id:
        # Every page of reviews (a long-lived PR can carry hundreds): a stamp on an old review must
        # still be seen, or the guard would silently pass a second review through.
        duplicate_run = str(run_id) in existing_run_ids(list_pr_reviews(repo_slug, pr_number, token))
    skip: set[str] = set()
    if dedup:
        try:
            r = httpx.get(
                f"{_API}/repos/{repo_slug}/pulls/{pr_number}/comments",
                headers=headers, params={"per_page": 100}, timeout=30.0, follow_redirects=True,
            )
            if r.status_code < 300:
                skip = existing_fingerprints(r.json())
        except Exception:  # noqa: BLE001 — dedup is best-effort; never block posting
            skip = set()

    unresolved = unresolved_blocking_findings(repo_slug, pr_number, token) if approve else None
    # What THIS pass found decides the event -- before any finding is re-routed for posting.
    # Demotion changes where a finding is said, never its severity, and a duplicate left out of the
    # body is still a finding of this pass.
    event = review_event(result, unresolved_blocking=unresolved, approve=approve)
    note = approval_note(event, result, unresolved, approve=approve)
    head_moved = bool(pr_head_sha and result.head_sha and pr_head_sha != result.head_sha)
    if head_moved:
        # The verdict is about the commit this review read; the pull request's head is another one.
        # Findings still go up (they are anchored to the reviewed commit), an approval does not.
        moved = head_moved_note(str(result.head_sha), str(pr_head_sha))
        if event == "APPROVE":
            event, note = "COMMENT", moved
        else:
            note = f"{note} {moved}"
    if failing_checks:
        # Said next to the verdict, never instead of it: the findings decide the event; this line
        # only keeps a reader from taking an approval for a merge go-ahead on a red head.
        note = f"{note} {red_head_note(failing_checks)}"

    inline_found = len(result.inline_findings)
    demoted = 0
    if result.inline_findings:
        # Only a path in the PR's diff (as GitHub lists it) can carry an inline comment; one comment
        # elsewhere fails the whole review. Read only when there is an inline finding to check.
        pr_files = fetch_pr_files(repo_slug, pr_number, token)
        if pr_files is not None:
            anchored = demote_to_summary(result, keep_paths=pr_files, drop_fingerprints=skip)
            demoted += len(anchored.summary_findings) - len(result.summary_findings)
            result = anchored
    payload = build_review_payload(result, skip_fingerprints=skip, event=event, approval=note, run_id=run_id)
    posted_review = False

    def _create_review(body: dict):
        return httpx.post(f"{_API}/repos/{repo_slug}/pulls/{pr_number}/reviews", headers=headers,
                          json=body, timeout=60.0, follow_redirects=True)

    # Post the review if there is something new to say (new inline comments), findings exist, or the
    # review is an approval -- an approval with nothing else to say is exactly the review a clean PR
    # needs (a clean review used to post nothing, so no PR ever carried an approval). Never when
    # this very job already posted one (duplicate_run): the PR carries that review.
    if not duplicate_run and (payload["comments"] or result.findings or event == "APPROVE"):
        resp = _create_review(payload)
        if resp.status_code == 422 and event == "APPROVE":
            # The App cannot approve a PR it authored (its own fix PRs). Say the same thing as a
            # comment, when there is anything to say.
            event = "COMMENT"
            note = approval_note(event, result, unresolved, approve=approve) + " (the App cannot approve its own pull request)"
            payload = build_review_payload(result, skip_fingerprints=skip, event=event, approval=note, run_id=run_id)
            resp = _create_review(payload) if (payload["comments"] or result.findings) else None
        if resp is not None and resp.status_code == 422 and payload["comments"]:
            # GitHub refused an anchor this payload carried -- a path or line outside the PR's diff
            # ("Path could not be resolved"); the review's text is not the problem. Say everything
            # in the body instead and retry once: no finding is lost, and the check does not turn
            # red because GitHub would not place a thread. Fires after the approval fallback so a
            # refused approval on a payload with comments is still recognised as such first. A
            # finding dedup already skipped stays skipped: the PR carries its thread.
            in_body = demote_to_summary(result, drop_fingerprints=skip)
            demoted += len(in_body.summary_findings) - len(result.summary_findings)
            result = in_body
            payload = build_review_payload(result, skip_fingerprints=skip, event=event, approval=note, run_id=run_id)
            resp = _create_review(payload)
        if resp is not None and resp.status_code >= 300:
            raise RuntimeError(f"create review failed: {resp.status_code} {resp.text[:300]}")
        posted_review = resp is not None

    check = None
    if check_run_id is not None or result.head_sha:
        from . import webhook_github

        try:
            if check_run_id is not None:
                webhook_github.update_check_run(
                    repo_slug, token, check_run_id, conclusion=result.conclusion,
                    summary=summary_body(result, note, run_id=run_id), name=CHECK_NAME,
                )
                check = {"id": check_run_id, "conclusion": result.conclusion, "updated_existing": True}
            else:
                cid = webhook_github.create_check_run(
                    repo_slug, token, name=CHECK_NAME, head_sha=result.head_sha,
                    summary=f"{len(result.findings)} finding(s)",
                )
                if cid is not None:
                    webhook_github.update_check_run(
                        repo_slug, token, cid, conclusion=result.conclusion,
                        summary=summary_body(result, note), name=CHECK_NAME,
                    )
                check = {"id": cid, "conclusion": result.conclusion, "updated_existing": False}
        except Exception:  # noqa: BLE001 — the check is advisory; a failure must not fail the review
            check = None

    return {
        "posted_review": posted_review,
        "duplicate_run": duplicate_run,
        "event": event,
        "unresolved_blocking": unresolved,
        "inline_posted": len(payload["comments"]),
        # inline_found == inline_posted + skipped_duplicates + demoted_to_summary, whichever
        # fallbacks fired: what was neither posted inline nor said in the body was already on the PR.
        "skipped_duplicates": inline_found - len(payload["comments"]) - demoted,
        "demoted_to_summary": demoted,
        "check": check,
        # The three commits a caller needs to compare: what the check is on, what was reviewed
        # (``head_sha`` in the enclosing result) and where the pull request stands now.
        "check_head_sha": check_head_sha if check_run_id is not None else result.head_sha,
        "pr_head_sha": pr_head_sha,
        # The required checks that were red on the head when this review posted (None = not read).
        "failing_required_checks": failing_checks,
    }
