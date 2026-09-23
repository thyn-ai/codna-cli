"""Codna's GitHub App webhook — the App channel, owned in Codna.

Codna is always local; the GitHub App is the only hosted surface. Anything Codna-specific —
the ``codna-fix`` / ``codna-secure`` labels, deciding what a GitHub event means, running
``codna fix`` / ``codna secure`` — lives HERE in the Codna repo, never in the Algenta SDK or
the engine (which ships standalone to other users).

Architecture (thin ingress → durable queue → worker pool), all inside the one hosted app:
  * this module is the pure core (:func:`verify_signature`, :func:`classify_event`,
    :func:`codna_command`) + a stdlib ``http.server`` ingress that VERIFIES and ENQUEUES only
    (fast ack — GitHub times out at 10s);
  * :mod:`codna.webhook_queue` is a durable, dedup'd, local SQLite queue;
  * :mod:`codna.webhook_worker` runs one job -- the codna CLI in an isolated, scrubbed workspace
    with a scoped installation token, reported via a Check Run;
  * :mod:`codna.webhook_pool` decides which job runs next and for how long (merge-gating classes
    first, a thread reserved for them, a deadline for the whole job);
  * :mod:`codna.webhook_control` is the live registry of what is running, the only thing that can
    cancel it (a superseding head, the job deadline), and where the pool's size and /healthz's
    saturation report come from.

Fail-closed: an unsigned/mis-signed delivery → 401; an event Codna doesn't act on → 204; a
recognized event → 202 (accepted, enqueued); a duplicate delivery → 200 (already queued).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

# Import-light by construction (stdlib only at module level), so the ingress can reach the live
# job registry without dragging the worker's dependencies into this module.
from . import webhook_control

# --- Codna-specific policy (labels that trigger a job) -------------------------------
FIX_LABEL = "codna-fix"
SECURE_LABEL = "codna-secure"
_WEBHOOK_SECRET_ENV = "CODNA_GITHUB_WEBHOOK_SECRET"
_MAX_BODY_BYTES = 25 * 1024 * 1024  # GitHub caps webhook payloads at ~25 MB; reject anything larger
_SENSITIVE_TEXT_RE = re.compile(
    r"(gh[opsu]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{20,}|pypi-[A-Za-z0-9_-]{20,}|"
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----)",
    re.DOTALL,
)


class WebhookError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# --- pure core: verify · classify · command (stdlib only, unit-tested) ----------------
def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """Constant-time verify GitHub's ``X-Hub-Signature-256`` (``sha256=<hex hmac>``)."""
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    provided = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, provided)


@dataclass(frozen=True)
class WebhookJob:
    """A Codna job derived from a verified GitHub event."""

    kind: str  # "fix" | "secure" | "review" | "queue" (a merge-queue group inheriting the PR's review)
    repo_full_name: str
    ref: str | None = None
    installation_id: int | None = None
    issue_number: int | None = None
    pr_number: int | None = None  # review jobs: the PR to review + post to
    # A `@codna fix` reply to an inline review finding carries the finding location + PR refs here so
    # the worker can enrich it (fetch the finding prose) and drive a verified fix PR. JSON-serialized
    # through the queue. Kept None for all other jobs (so WebhookJob stays hashable in practice).
    context: dict | None = None
    reason: str = ""


# A PR/issue comment triggers Codna when a line starts with `@codna <verb>`.
_CODNA_COMMENT_RE = re.compile(r"^\s*@codna\s+(review|fix)\b", re.IGNORECASE)

# `@codna fix` SPENDS a metered run and PUSHES a branch/PR, so — like applying a fix label — it is
# gated to commenters with write-level authority on the repo. GitHub's author_association reflects
# the commenter's standing; a drive-by/external commenter (CONTRIBUTOR / NONE) cannot trigger a fix.
_FIX_WRITE_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Codna pushes its own fix branches as `codna/<plan-id>` (see packaged_git._validated_branch and
# fix_run). Recognising that prefix is what stops a CI-failure fix from fixing its own fix PR.
_CODNA_BRANCH_PREFIX = "codna/"
# ONE Check Run per job, spelled exactly like the CLI command and the PR trigger (`codna review`,
# `@codna review`): "codna review" is also the exact name the org ruleset requires. Until
# 2026-09-19 the worker's wrapper run and the CLI's findings run carried different casings.
CHECK_RUN_NAMES = {"review": "codna review", "fix": "codna fix", "secure": "codna secure"}


def check_run_name(kind: str) -> str:
    """The Check Run name for a job kind: ``codna review`` / ``codna fix`` / ``codna secure``."""
    return CHECK_RUN_NAMES.get(kind, f"codna {kind.replace('_', ' ')}")
# The App's own slug on github.com. A self-hosted install runs under a different slug, so the id in
# GITHUB_APP_ID is checked too (see _is_own_check_suite).
_OWN_APP_SLUG = "codna-ai"


# refs/heads/gh-readonly-queue/<base>/pr-<N>-<base sha> -- the only place a merge group names its PR.
# Parsed by prefix + last path segment, not by a search over the whole ref: an unanchored regex with a
# lazy `.+?` over attacker-shaped input is polynomial (CodeQL py/polynomial-redos on the first cut).
_MERGE_GROUP_PREFIX = "refs/heads/gh-readonly-queue/"
_MERGE_GROUP_TAIL = re.compile(r"pr-(\d+)-[0-9a-f]{40}")


def merge_group_pr_number(head_ref: str) -> int | None:
    """The pull request number a merge-group ref was built for, or None."""
    if not head_ref.startswith(_MERGE_GROUP_PREFIX):
        return None
    tail = head_ref.rsplit("/", 1)[-1]  # "pr-<N>-<40 hex>"; the base branch may itself contain "/"
    m = _MERGE_GROUP_TAIL.fullmatch(tail)
    return int(m.group(1)) if m else None


def _is_own_check_suite(suite: Mapping[str, Any]) -> bool:
    """True when a check_suite belongs to this App itself (its review / fix Check Runs)."""
    app = suite.get("app") or {}
    if str(app.get("slug") or "") == _OWN_APP_SLUG:
        return True
    own_id = os.environ.get("GITHUB_APP_ID")
    return bool(own_id) and str(app.get("id") or "") == str(own_id)


def _parse_codna_command(body: Any) -> str | None:
    """Return the verb ('review' | 'fix') from the first `@codna <verb>` line in a comment, else None."""
    if not isinstance(body, str):
        return None
    for line in body.splitlines():
        m = _CODNA_COMMENT_RE.match(line)
        if m:
            return m.group(1).lower()
    return None


def _fix_context_from_review_comment(pr: Mapping[str, Any], comment: Mapping[str, Any]) -> dict | None:
    """Build the fix context from a `@codna fix` inline REPLY (pure — payload only).

    Requires ``in_reply_to_id`` (the parent Codna finding thread) — a `@codna fix` that is not a reply
    to an existing finding thread is ignored. path/line come from the reply object (same location as
    the finding). head/base refs steer the fix (analyze the PR head, target the PR's own branch)."""
    in_reply_to = comment.get("in_reply_to_id")
    if not isinstance(in_reply_to, int):
        return None
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    head_repo = (head.get("repo") or {}).get("full_name")
    base_repo = (base.get("repo") or {}).get("full_name")
    line = comment.get("line")
    if not isinstance(line, int):
        line = comment.get("original_line") if isinstance(comment.get("original_line"), int) else None
    path = comment.get("path")
    commenter = (comment.get("user") or {}).get("login")
    return {
        "in_reply_to_id": in_reply_to,
        "commenter": commenter if isinstance(commenter, str) and commenter else None,
        "author_association": comment.get("author_association"),
        "path": path if isinstance(path, str) and path else None,
        "line": line,
        "head_sha": _safe_ref(head.get("sha")),
        "head_ref": _safe_ref(head.get("ref")),
        "base_ref": _safe_ref(base.get("ref")),
        "is_fork": bool(head_repo and base_repo and head_repo != base_repo),
    }


def _repo_full_name(payload: Mapping[str, Any]) -> str | None:
    name = (payload.get("repository") or {}).get("full_name")
    return name if isinstance(name, str) and "/" in name else None


def _installation_id(payload: Mapping[str, Any]) -> int | None:
    value = (payload.get("installation") or {}).get("id")
    return value if isinstance(value, int) else None


def _safe_ref(value: Any) -> str | None:
    """A ref is forwarded only if it can't be read by git as an option (no leading '-').

    Defense-in-depth at the trust boundary; packaged_git._validated_ref is the hard sink.
    """
    return value if isinstance(value, str) and value and not value.startswith("-") else None


# --- feedback-repo cross-repo dispatch (thyn-ai/feedback -> a product's own repo) -----------------
# Env, not a constant: any company adopting this must be able to point it at their own public
# intake repo and their own product repos without a code change.
_FEEDBACK_REPO_ENV = "CODNA_FEEDBACK_REPO"
_FEEDBACK_ROUTES_ENV = "CODNA_FEEDBACK_ROUTES"
_PRODUCT_LABEL_RE = re.compile(r"^product:(\S+)$")


def feedback_repo() -> str | None:
    """The public intake repo (e.g. ``thyn-ai/feedback``), or None when unconfigured -- in which
    case feedback-repo events are just ordinary unrecognized events (204), not an error."""
    return os.environ.get(_FEEDBACK_REPO_ENV) or None


def feedback_routes() -> dict[str, str]:
    """``{"algenta": "thyn-ai/algenta", ...}`` from CODNA_FEEDBACK_ROUTES (a JSON object). Malformed
    or absent config yields an empty map -- every report then reports itself unrouted rather than
    raising, since a config typo must never take down the whole webhook."""
    raw = os.environ.get(_FEEDBACK_ROUTES_ENV)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and "/" in v}


def _product_from_labels(labels: Any) -> str | None:
    """The bare product name from a ``product:<name>`` label on the issue, if present. The first
    match wins (an issue should carry exactly one such label)."""
    if not isinstance(labels, list):
        return None
    for entry in labels:
        name = entry.get("name") if isinstance(entry, dict) else None
        m = _PRODUCT_LABEL_RE.match(name or "")
        if m:
            return m.group(1)
    return None


def _classify_feedback_repo_fix(payload: Mapping[str, Any], *, installation: int | None,
                                fallback_repo: str) -> WebhookJob:
    """A ``codna-fix`` label on the public feedback repo: route by the issue's own
    ``product:<name>`` label to that product's real repo, or explain why it couldn't."""
    issue = payload.get("issue") or {}
    number = issue.get("number")
    product = _product_from_labels(issue.get("labels"))
    target = feedback_routes().get(product) if product else None
    ctx: dict[str, Any] = {"feedback_repo": fallback_repo, "feedback_issue": number,
                           "fp": f"feedback#{number}"}
    if target:
        ctx["product"] = product
        return WebhookJob("fix", target, installation_id=installation, context=ctx,
                          reason="feedback_routed_fix")
    reason = (f"no route is configured for product `{product}`" if product
             else "no `product:<name>` label was found on this report")
    ctx["routing_error"] = reason
    # repo_full_name = the FEEDBACK repo itself (there is no target) + issue_number set, so the
    # existing generic issue-outcome reporting in process_job posts the explanation without any
    # feedback-specific plumbing -- this case needs none.
    return WebhookJob("fix", fallback_repo, installation_id=installation, issue_number=number,
                      context=ctx, reason="feedback_unrouted")


def classify_event(event: str, payload: Mapping[str, Any]) -> WebhookJob | None:
    """Map a GitHub ``(event, payload)`` to a Codna job, or ``None`` to ignore it.

    fix    — an issue labeled ``codna-fix``, ``@codna fix`` on a review-comment thread, or a
             ``check_suite`` that FAILED on a pull request.
    secure — an issue labeled ``codna-secure``, or a ``code_scanning_alert`` appearing.
    review — a PR opened/updated, or ``@codna review`` on a PR (conversation or inline thread).

    Still does NOT react to a repo's overall CI health. The original objection stands and is worth
    restating, because it is what shapes the check_suite branch below: reacting to ANY red check on
    a push catches infra/outage flakes that have nothing to do with the code and resolve on their
    own retry, and it is not how Cursor's Bugbot works either — it reacts to a pull request, not to
    a repo's CI health.

    A check_suite that failed ON A PULL REQUEST is the case that objection does not cover. The PR
    is itself the explicit human action this function requires (``opening/updating a PR`` was
    already accepted as one), and the failure is scoped to that person's proposed change rather
    than to the repo at large. So the branch below admits exactly that and nothing else: a
    ``failure`` conclusion, associated with a PR, on a branch Codna did not open.
    """
    action = str(payload.get("action") or "")
    repo = _repo_full_name(payload)
    if not repo:
        return None
    installation = _installation_id(payload)

    # Proactive PR review: on open/update, review the diff (read-only) and post findings. The payload
    # carries the head SHA directly, so no extra API round-trip is needed to start.
    if event == "pull_request" and action in {"opened", "synchronize", "reopened", "ready_for_review"}:
        pr = payload.get("pull_request") or {}
        number = pr.get("number")
        if isinstance(number, int) and not pr.get("draft"):
            head_sha = _safe_ref((pr.get("head") or {}).get("sha"))
            return WebhookJob("review", repo, ref=head_sha, pr_number=number,
                              installation_id=installation, reason=f"pull_request_{action}")
        return None

    # Manual re-review: `@codna review` on a PR — either the conversation tab (issue_comment on an
    # issue that is a PR) or an inline review thread (pull_request_review_comment).
    if event == "issue_comment" and action == "created":
        issue = payload.get("issue") or {}
        number = issue.get("number")
        if issue.get("pull_request") and isinstance(number, int):
            if _parse_codna_command((payload.get("comment") or {}).get("body")) == "review":
                return WebhookJob("review", repo, pr_number=number,
                                  installation_id=installation, reason="comment_codna_review")
        return None

    if event == "pull_request_review_comment" and action == "created":
        pr = payload.get("pull_request") or {}
        comment = payload.get("comment") or {}
        number = pr.get("number")
        verb = _parse_codna_command(comment.get("body"))
        if isinstance(number, int):
            if verb == "review":
                return WebhookJob("review", repo, pr_number=number,
                                  installation_id=installation, reason="review_comment_codna_review")
            if verb == "fix":
                # `@codna fix` reply to a Codna finding → route that finding into a verified fix PR.
                # Authorization happens in the WORKER against the repository's collaborator
                # permission, not here: the payload's author_association is not reliable across
                # repos -- GitHub sent CONTRIBUTOR for an org admin on thyn-ai/algenta-integrations
                # (2026-09-17) while the REST API reported MEMBER for the same comment, and the gate
                # that used to live here dropped the request without a trace.
                ctx = _fix_context_from_review_comment(pr, comment)
                if ctx is not None:
                    return WebhookJob("fix", repo, ref=ctx.get("head_sha"), pr_number=number,
                                      installation_id=installation, context=ctx,
                                      reason="review_comment_codna_fix")
        return None

    if event == "issues" and action == "labeled":
        label = (payload.get("label") or {}).get("name")
        number = (payload.get("issue") or {}).get("number")
        fbrepo = feedback_repo()
        if fbrepo and repo == fbrepo and label == FIX_LABEL and isinstance(number, int):
            # A report on the public intake repo, approved by a maintainer's codna-fix label:
            # route it by product to that product's own (often private) repo. Same installation
            # covers both -- the App is org-wide -- so no separate auth is needed to reach either.
            return _classify_feedback_repo_fix(payload, installation=installation, fallback_repo=fbrepo)
        if label == FIX_LABEL:
            return WebhookJob("fix", repo, installation_id=installation, issue_number=number, reason="labeled_codna_fix")
        if label == SECURE_LABEL:
            return WebhookJob("secure", repo, installation_id=installation, issue_number=number, reason="labeled_codna_secure")
        return None

    if event == "code_scanning_alert" and action in {"created", "reopened", "appeared_in_branch"}:
        return WebhookJob("secure", repo, ref=_safe_ref(payload.get("ref")),
                          installation_id=installation, reason="code_scanning_alert")

    if event == "merge_group" and action == "checks_requested":
        # A merge queue builds a temporary group commit and asks every REQUIRED check to report on
        # it. "codna review" is required on repos that use one (algenta-sdk), and a check that only
        # reacts to pull_request events would never appear there -- the queue would time out and
        # kick the PR. The group is not re-reviewed: the PR head IS what was reviewed (GitHub only
        # admits a PR to the queue once its required checks passed), so the worker copies that
        # verdict onto the group commit. Requires the App to be subscribed to "Merge group" events.
        group = payload.get("merge_group") or {}
        head_sha = _safe_ref(group.get("head_sha"))
        number = merge_group_pr_number(str(group.get("head_ref") or ""))
        if not head_sha or number is None:
            return None
        return WebhookJob("queue", repo, ref=head_sha, pr_number=number,
                          installation_id=installation, reason="merge_group_checks_requested")

    if event == "check_suite" and action == "completed":
        suite = payload.get("check_suite") or {}
        conclusion = str(suite.get("conclusion") or "")
        if conclusion != "failure":
            # `cancelled` / `timed_out` / `stale` / `neutral` / `skipped` are not evidence that the
            # code is broken. Measured on thyn-ai/algenta across a single day: of 25 check suites,
            # 19 succeeded, 5 were CANCELLED by a concurrency group and 1 genuinely failed. Treating
            # anything but `failure` as fixable would spend a metered run on infrastructure noise —
            # precisely the objection in this function's docstring.
            return None
        if _is_own_check_suite(suite):
            # Codna's OWN check suite (the review / fix Check Runs) going red says something about a
            # Codna job, never about the pull request's code. Without this the App fixed itself: a
            # failed `codna fix` re-fired a fix (deduped only by the 24 h rule), and on 2026-09-18
            # GitHub completed the App's suite as `failure` with ZERO check runs two seconds before
            # the review's first run existed (thyn-ai/telys#123), spending a metered run on nothing.
            return None
        if suite.get("latest_check_runs_count") == 0:
            # A suite that "failed" without a single check run has nothing to fix.
            return None
        for pr in suite.get("pull_requests") or []:
            number = pr.get("number")
            if not isinstance(number, int):
                continue
            head = pr.get("head") or {}
            head_ref = str(head.get("ref") or "")
            if head_ref.startswith(_CODNA_BRANCH_PREFIX):
                # Codna's own fix PRs run CI like any other. Without this, a fix PR whose checks go
                # red triggers a fix OF THE FIX, whose checks go red, and so on — an unbounded loop
                # that spends a metered run and opens a PR on every cycle.
                continue
            head_sha = _safe_ref(head.get("sha"))
            # `head_ref` travels in context so the command builder can target the PR's OWN branch:
            # a fix PR opened against the default branch would try to repair code that only exists
            # on this PR's branch.
            ctx: dict[str, Any] = {"head_ref": head_ref} if head_ref else {}
            if isinstance(suite.get("id"), int):
                # The suite id lets the worker ask, before spending anything, whether this suite is
                # still red and what its failed jobs actually died on (webhook_ci_triage).
                ctx["check_suite_id"] = suite["id"]
            return WebhookJob("fix", repo, ref=head_sha, pr_number=number,
                              installation_id=installation, context=ctx or None,
                              reason="check_suite_failure")
        # No associated pull request means this was a push to a branch — the repo-CI-health case
        # the docstring rejects. Ignored on purpose.
        #
        # Two behaviours here are inherited rather than chosen, and are worth knowing:
        #   - fork PRs are reported to arrive with an EMPTY `pull_requests` list, which would make
        #     fork CI unable to trigger a fix. That is the behaviour we want, but it is GitHub's
        #     to change and is NOT verified here — do not rely on it as a security boundary.
        #   - draft PRs are not distinguishable from this payload (the PR objects are minimal), so
        #     unlike the `review` branch above this does not skip them. A draft's red CI will be
        #     fixed. If that turns out to be unwanted, it needs an API round-trip to detect.
        return None

    return None


def webhook_marker(job: WebhookJob) -> str:
    """Deterministic idempotency marker embedded in each fix PR body.

    Keyed on the trigger (repo/kind/ref-or-issue), NOT on the delivery id, so a recover_stale
    re-run after a hard crash produces the SAME marker — the worker then finds the PR the first
    run already opened and reuses it instead of opening a duplicate.
    """
    ctx = job.context or {}
    finding_key = ctx.get("in_reply_to_id") or ctx.get("fp")
    if finding_key:
        # A comment-fix targets ONE finding: key on the finding thread so two different findings on
        # the same head SHA get distinct PRs (not collapsed into one by the idempotency-reuse branch).
        key = f"{job.ref or 'head'}:{finding_key}"
    else:
        key = job.ref or (str(job.pr_number or job.issue_number) if (job.pr_number or job.issue_number) else "default")
    return f"codna-webhook-id: {job.repo_full_name}#{job.kind}#{key}"


# A narrative preamble a model emits when asked for a one-line cause ("Here's a summary of every
# change made:", "Summary of changes:") -- never a usable title. Deliberately NARROW: an earlier
# version matched a bare leading `summary`/`the following`/`i have`, which rejected perfectly good
# human issue titles like "Summary tab shows stale totals after a refund". A "here's/here is" opener
# is prose on its own; the weaker openers only count as prose when the line also ENDS in a colon,
# the way a section header does.
_PROSE_OPENER_RE = re.compile(r"^(here(?:'s|\s+is)\b)", re.IGNORECASE)
_PROSE_HEADER_RE = re.compile(r"^(summary|the following|changes?|overview|details)\b", re.IGNORECASE)

# A GitHub closing keyword + issue reference. Issue text is ATTACKER-INFLUENCED (anyone who can file
# an issue controls its title), and the PR title becomes the branch's commit message
# (packaged_git.py), where GitHub HONORS these on merge. So a title smuggling "Closes #1" could close
# an unrelated issue -- a maintainer's security tracker, say. Defanged by dropping the reference
# marker (`#`/`GH-`/issue URL), which keeps the words readable but inert.
_CLOSING_REF_RE = re.compile(
    r"\b(clos(?:e|es|ed)|fix(?:e[sd])?|resolv(?:e|es|ed))(\s+)"
    r"(?:#|GH-|https?://\S*?/issues/)(\d+)",
    re.IGNORECASE,
)


def pr_title_from_issue_text(issue_text: str | None, *, limit: int = 72) -> str | None:
    """A concise PR title from the FIRST line of ``issue_text``, or None if it isn't usable.

    ``fetch_issue_text`` formats an issue as "<title>\\n\\n<body>", so line 1 IS the issue title --
    the same thing a human would name the PR after. Without this, a label-triggered fix passed no
    ``--pr-title`` at all and fix_run fell back to the model's ``root_cause``, which produced the
    real title "codna: fix Here's a summary of every change made:" on a live fix PR. The
    comment-triggered path already passes an explicit title; this closes that asymmetry.

    FIRST LINE ONLY, and None rather than falling through to later lines: the body is never the
    title, so walking into it just swapped one nonsense title for another (a repro step, a quoted
    log, an @-mention). Returning None is safe -- ``codna_command`` then omits ``--pr-title`` and
    fix_run's own fallback applies.
    """
    first = next((ln for ln in (issue_text or "").splitlines() if ln.strip()), "")
    # Strip markdown heading / bullet / quote decoration, then collapse whitespace.
    line = re.sub(r"^\s*(?:[#>*\-+]+\s*)+", "", first)
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return None
    if _PROSE_OPENER_RE.match(line) or (line.endswith(":") and _PROSE_HEADER_RE.match(line)):
        return None
    line = _CLOSING_REF_RE.sub(r"\1\2\3", line).rstrip(":").strip()
    if not line:
        return None
    if len(line) > limit:
        cut = line[:limit].rsplit(" ", 1)[0].rstrip(",;:.-")
        line = (cut or line[:limit]).rstrip()
    return line


def _fix_issue_text(ctx: Mapping[str, Any]) -> str:
    """Render a localized fix obligation from an (enriched) review-finding context → `codna fix --issue`.

    For a git-URL fix the engine localizes from this text alone (focus-path steering is local-only),
    so it names the file:line + the finding so the fix targets exactly that spot."""
    loc = ctx.get("path") or "(unknown file)"
    if ctx.get("line"):
        loc = f"{loc}:{ctx['line']}"
    parts = ["Fix the issue codna review flagged in this pull request.", f"Location: {loc}"]
    if ctx.get("severity") or ctx.get("category"):
        parts.append(f"Severity: {ctx.get('severity') or '?'} ({ctx.get('category') or '?'})")
    if ctx.get("title"):
        parts.append(f"Problem: {ctx['title']}")
    if ctx.get("explanation"):
        parts.append(str(ctx["explanation"]).strip())
    parts.append("Change only the code needed to fix THIS issue at the location above; do not modify "
                 "tests, CI, or unrelated code.")
    return "\n".join(parts)


def codna_command(
    job: WebhookJob, *, repo_url: str | None = None, sarif_path: str | None = None,
    issue_text: str | None = None,
) -> list[str]:
    """Build the ``codna`` argv for a classified job.

    fix    — a CI-triggered fix auto-discovers its failing tests (``--tests``); an issue-label
             fix needs the issue's own text (``issue_text``, fetched by the caller — see
             ``run_codna_job``) since there is no commit to run tests against. Or the
             comment-triggered variant below.
    secure — ``codna secure <url> [--ref] --from-sarif <sarif>`` (read-only Tier-1
             classification; opening security PRs uses the privilege-separated two-job
             workflow, never a single server process).
    review — ``codna review <url> --pr <n> [--ref <head_sha>] --post --json`` (read-only diff review;
             completes the job's "codna review" check with the findings).
    """
    url = repo_url or f"https://github.com/{job.repo_full_name}.git"
    if job.kind == "review":
        if not job.pr_number:
            raise WebhookError("review_requires_pr", "review jobs need a PR number.")
        args = ["codna", "review", url, "--pr", str(job.pr_number), "--post", "--json"]
        if job.ref:
            args += ["--ref", job.ref]
        return args
    if job.kind == "secure":
        if not sarif_path:
            raise WebhookError("secure_requires_sarif", "secure jobs need a SARIF report (fetched from code scanning).")
        args = ["codna", "secure", url, "--from-sarif", sarif_path]
        if job.ref:
            args += ["--ref", job.ref]
        return args
    if job.kind == "fix":
        ctx = job.context or {}
        # `in_reply_to_id` is set ONLY by _fix_context_from_review_comment, which itself returns
        # None when it's absent -- so it is a precise, structural discriminator for "this is a
        # comment-fix", never a coincidental one. A feedback-routed job has context (product,
        # feedback_repo, ...) but never this key, so it correctly falls through below instead of
        # being force-fit into the comment-fix shape (which needs head_ref -- a PR branch that
        # doesn't exist for an issue filed on a different repo entirely).
        if ctx.get("in_reply_to_id"):
            # Comment-triggered verified fix of a specific review finding. Analyze the PR head (--ref)
            # and target the PR's OWN branch (--base-branch head_ref) so the fix stacks onto the PR.
            if not ctx.get("head_ref"):
                raise WebhookError("fix_requires_head_ref", "comment-fix needs the PR head branch to target.")
            args = ["codna", "fix", url, "--open-pr", "--issue", _fix_issue_text(ctx)]
            if ctx.get("head_sha"):
                args += ["--ref", ctx["head_sha"]]
            args += ["--base-branch", ctx["head_ref"],
                     "--pr-title", f"codna: fix {(ctx.get('title') or 'review finding')[:60]}",
                     "--pr-body", f"Verified fix for a codna review finding.\n\n{webhook_marker(job)}"]
            return args
        args = ["codna", "fix", url, "--open-pr"]
        if job.ref:
            # A fix job with a ref but no comment context and no issue text -- let codna discover
            # what's actually failing by running the repo's own tests against this commit.
            args += ["--ref", job.ref, "--tests"]
            if ctx.get("head_ref"):
                # CI-failure fix on a PR: stack onto the PR's OWN branch. Targeting the default
                # branch instead would try to repair code that does not exist there yet -- the
                # failing change lives only on this PR's branch until it merges.
                args += ["--base-branch", ctx["head_ref"]]
            if ctx.get("ci_failure"):
                # CI evidence gathered by webhook_ci_triage (the failing step and the log around its
                # error): the agent fixes what CI saw, not only what its own sandboxed test run finds.
                args += ["--issue", str(ctx["ci_failure"])]
        elif job.issue_number or issue_text or ctx.get("feedback_issue"):
            # Label-triggered (labeled_codna_fix, or a feedback-routed report -- both have no
            # commit/tests to run): `codna fix` requires one of --issue/--tests/--from-junit or it
            # dies immediately (fix_run.py), so the caller (run_codna_job) must have already
            # fetched the text -- from the issue itself, or (feedback-routed) from the public
            # report, since job.issue_number is None there and there is nothing else to fetch.
            #
            # `ctx.get("feedback_issue")` alone (without job.issue_number or a passed issue_text)
            # must STILL enter this branch: a feedback job with NO issue_text supplied is exactly
            # the fetch-failure case, and skipping this branch entirely would silently build an
            # incomplete `codna fix --open-pr` with no --issue/--tests, which `codna fix` itself
            # would then reject at run time instead of failing here with a catchable error.
            if not issue_text:
                raise WebhookError(
                    "fix_requires_issue_text",
                    f"issue-label fix for #{job.issue_number} needs the issue's own text.",
                )
            args += ["--issue", issue_text]
            # Name the PR after the issue. Without this the label path passed no --pr-title and
            # fix_run fell back to the model's `root_cause` prose (a live PR really was titled
            # "codna: fix Here's a summary of every change made:").
            issue_title = pr_title_from_issue_text(issue_text)
            if issue_title:
                args += ["--pr-title", f"codna: fix {issue_title}"]
        # Embed the idempotency marker in the PR body so a re-run can find + reuse this PR. For an
        # issue trigger also REFERENCE the issue (plain `#N`, deliberately not a closing keyword):
        # the marker is not a GitHub reference, so nothing cross-linked the PR back to the issue and
        # the issue looked untouched. A bot should make the link visible, not decide to close a
        # human's issue -- merging is what proves the fix, and the human owns that call.
        body = f"Automated fix opened by the Codna GitHub App.\n\n{webhook_marker(job)}"
        if job.issue_number and not ctx:
            body = (
                f"Automated fix opened by the Codna GitHub App for #{job.issue_number}.\n\n"
                f"{webhook_marker(job)}"
            )
        elif ctx.get("feedback_repo") and ctx.get("feedback_issue"):
            # Cross-link to the PUBLIC report, not to job.issue_number (there is none -- the issue
            # lives on the feedback repo, not this one).
            body = (
                f"Automated fix opened by the Codna GitHub App for a report filed at "
                f"{ctx['feedback_repo']}#{ctx['feedback_issue']}.\n\n{webhook_marker(job)}"
            )
        args += ["--pr-body", body]
        return args
    raise WebhookError("unknown_job_kind", f"unknown job kind {job.kind!r}")


def _redact_debug_value(value: Any) -> Any:
    if isinstance(value, str):
        return _SENSITIVE_TEXT_RE.sub("<redacted>", value)[:500]
    if isinstance(value, dict):
        return {str(k): _redact_debug_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_debug_value(v) for v in value[:20]]
    return value


def _queue_debug_payload(server: Any) -> dict[str, Any]:
    queue = getattr(server, "queue", None)
    pool = getattr(server, "worker_pool", None)
    payload: dict[str, Any] = {
        "service": "codna-webhook",
        "queue": {"configured": queue is not None},
        "worker_pool": pool.diagnostics() if pool is not None else {"configured": False},
    }
    if queue is not None:
        payload["queue"] = {
            "configured": True,
            "counts": queue.counts(),
            "recent": _redact_debug_value(queue.recent(limit=10)),
        }
    return payload


def default_queue_path() -> str:
    """Where the durable job queue lives (override with CODNA_WEBHOOK_QUEUE, e.g. a Fly volume)."""
    override = os.environ.get("CODNA_WEBHOOK_QUEUE")
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    root = Path(os.environ.get("CODNA_RUNTIME_ROOT") or (Path.home() / ".codna"))
    path = root / "webhook" / "queue.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# --- self-hostable server: verify + enqueue (thin ingress) ----------------------------
def _log_delivery(event: str, payload: Mapping[str, Any], delivery_id: str | None, outcome: str,
                  job: WebhookJob | None = None, **fields: Any) -> None:
    """One stderr JSON line per verified delivery: what arrived and what became of it.

    Until now a delivery that classified as `ignored` or collapsed as a `duplicate` left no trace
    anywhere -- on 2026-09-17 a `@codna fix` reply produced neither a job nor a log line and the
    only way to tell "never delivered" from "delivered and dropped" was to guess. Same shape as the
    worker's lines so one grep covers ingress and processing (`"service": "codna-webhook"`).
    ``fields`` are extra keys a path wants on record (the Check Run the ingress opened, for one).
    """
    repo = (payload.get("repository") or {}).get("full_name")
    line = {
        "service": "codna-webhook",
        "event": "delivery",
        "github_event": event,
        "action": payload.get("action"),
        "delivery_id": delivery_id,
        "repo": repo,
        "outcome": outcome,
    }
    if job is not None:
        line.update({"kind": job.kind, "reason": job.reason, "pr_number": job.pr_number})
    line.update(fields)
    print(json.dumps(line, sort_keys=True), file=sys.stderr, flush=True)


class _Handler(BaseHTTPRequestHandler):
    server_version = "codna-webhook/2"

    def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # keep the server quiet by default
        return

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        # Liveness/readiness for load balancers + Fly health checks. Never leaks config.
        path = self.path.rstrip("/")
        if path in ("/health", "/healthz", ""):
            configured = bool(os.environ.get(_WEBHOOK_SECRET_ENV))
            self._reply(200 if configured else 503,
                        webhook_control.health_payload(self.server, ok=configured))
            return
        if path == "/ready":
            secret_ready = bool(os.environ.get(_WEBHOOK_SECRET_ENV))
            if os.environ.get("GITHUB_TOKEN"):
                app_ready = True
            else:
                from .webhook_github import app_auth_config_ready

                app_ready = app_auth_config_ready(os.environ.get("GITHUB_APP_ID"), os.environ.get("GITHUB_APP_PRIVATE_KEY"))
            pool = getattr(self.server, "worker_pool", None)
            pool_diag = pool.diagnostics() if pool is not None else None
            worker_ready = True if pool_diag is None else bool(pool_diag.get("ready"))
            ready = secret_ready and app_ready and worker_ready
            queue = getattr(self.server, "queue", None)
            try:
                # Depth by status ({"queued": n, "running": n, "retry": n, ...}): 17 pull requests opened
                # within a minute on 2026-09-19 sat behind a 2-thread pool for ~15 minutes and nothing
                # public said why the checks were missing. Counts only -- no job content.
                queue_counts = queue.counts() if queue is not None else None
            except Exception:  # noqa: BLE001 -- readiness must never fail because a diagnostic did
                queue_counts = None
            self._reply(200 if ready else 503, {
                "service": "codna-webhook",
                "ok": ready,
                "checks": {"webhook_secret": secret_ready, "github_app_auth": app_ready,
                           "worker_pool": worker_ready},
                "worker": None if pool_diag is None else {
                    k: pool_diag.get(k) for k in ("alive_threads", "busy_threads", "longest_running_s", "wedged")
                },
                "queue": queue_counts,
            })
            return
        if path == "/debug/queue":
            secret = os.environ.get(_WEBHOOK_SECRET_ENV, "")
            token = self.headers.get("X-Codna-Debug-Token", "")
            if not secret or not hmac.compare_digest(secret, token):
                self._reply(401, {"error": "unauthorized"})
                return
            self._reply(200, _queue_debug_payload(self.server))
            return
        self._reply(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") not in ("/webhooks/github", "/webhook"):
            self._reply(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._reply(400, {"error": "invalid_content_length"})
            return
        if length < 0 or length > _MAX_BODY_BYTES:
            self._reply(400, {"error": "invalid_content_length"})
            return
        body = self.rfile.read(length) if length else b""
        secret = os.environ.get(_WEBHOOK_SECRET_ENV, "")
        if not verify_signature(secret, body, self.headers.get("X-Hub-Signature-256")):
            self._reply(401, {"error": "invalid_signature"})
            return
        event = self.headers.get("X-GitHub-Event", "")
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            self._reply(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):  # valid JSON that isn't an object (array/number/null)
            self._reply(400, {"error": "invalid_json"})
            return
        job = classify_event(event, payload)
        delivery_header = self.headers.get("X-GitHub-Delivery")
        if job is None:
            _log_delivery(event, payload, delivery_header, "ignored")
            self._reply(204, {"status": "ignored", "event": event})
            return
        # Idempotency key: prefer X-GitHub-Delivery; if absent (redelivery via a header-stripping
        # proxy, replay, etc.), synthesize a stable one from event+body so a NULL never reaches the
        # UNIQUE column and identical deliveries still collapse to one row.
        delivery_id = self.headers.get("X-GitHub-Delivery") or (
            "synth-" + hashlib.sha256(event.encode("utf-8") + b"\x00" + body).hexdigest()
        )
        queue = getattr(self.server, "queue", None)
        if queue is None:  # no durable queue (e.g. bare handler in a test) — accept without persisting
            _log_delivery(event, payload, delivery_id, "accepted", job)
            self._reply(202, {"status": "accepted", "kind": job.kind, "repo": job.repo_full_name})
            return
        # The review's Check Run is opened `queued` HERE, before the row exists, so the tenant sees
        # it the moment the delivery is accepted rather than when a worker slot frees (measured
        # 2026-09-20: p95 23 s invisible during a 21-review fan-out). The row is born owning the
        # run; the worker updates it. webhook_queued_check has the whole lifecycle; the service
        # switches this on for the Postgres backend only (webhook_service.serve).
        queued_check = bool(getattr(self.server, "queued_check_runs", False))
        pre = None
        late_gate: threading.Event | None = None
        if queued_check:
            from . import webhook_queued_check

            late_gate = threading.Event()
            pre = webhook_queued_check.precreate(
                job, queue=queue, delivery_id=delivery_id, github=self._github(),
                app_id=os.environ.get("GITHUB_APP_ID"), private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY"),
                late_gate=late_gate)
        try:
            if pre is not None:
                fresh = queue.enqueue(job, delivery_id=delivery_id, check_run_id=pre.check_run_id)
            else:
                fresh = queue.enqueue(job, delivery_id=delivery_id)
        finally:
            if late_gate is not None:
                late_gate.set()  # a create that answers after the budget may now adopt onto this row
        check_run: dict[str, Any] = {}
        if pre is not None:
            bound = webhook_queued_check.bind_or_close(pre, job=job, queue=queue, delivery_id=delivery_id,
                                                      fresh=fresh, github=self._github())
            check_run = {"check_run_id": pre.check_run_id, "check_run": bound}
        # A new head retires the work the old one left behind. enqueue() already collapsed the
        # WAITING rows; this stops the ones already RUNNING, whose verdict would land on a commit
        # that no longer exists (thyn-ai/mojo-kernels#32, 2026-09-19: two reviews ran to completion
        # on ba252b20 and 04e3c8a5 after the head had moved to 21e3610) -- or, for a `codna fix`,
        # would open a pull request built on a base that has moved.
        superseded = 0
        if fresh:
            try:
                superseded = webhook_control.supersede_running(job, queue=queue)
            except Exception:  # noqa: BLE001 -- a delivery must be accepted even if cancelling fails
                superseded = 0
            if queued_check and job.pr_number is not None:
                # ... and the WAITING rows enqueue() just retired had Check Runs of their own, opened
                # queued by this ingress: complete them neutral ("superseded by <sha>") the way the
                # running ones are, off the request thread. The reaper sweeps whatever this misses.
                # The pre-create's token is scoped to this repository and may complete those runs;
                # without one (a `fix` retiring a review's row) a token is minted per row.
                token_for = webhook_queued_check.token_for(
                    github=self._github(), app_id=os.environ.get("GITHUB_APP_ID"),
                    private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY"))
                if pre is not None:
                    def token_for(_row: Any, _token: str = pre.token) -> str:  # one token, one repository
                        return _token
                try:
                    webhook_queued_check.settle_retired_in_background(
                        queue, github=self._github(), repo=job.repo_full_name, pr_number=job.pr_number,
                        token_for=token_for)
                except Exception:  # noqa: BLE001 -- the reaper's sweep is the backstop
                    pass
        _log_delivery(event, payload, delivery_id, "accepted" if fresh else "duplicate", job, **check_run)
        self._reply(
            202 if fresh else 200,
            {"status": "accepted" if fresh else "duplicate", "kind": job.kind,
             "repo": job.repo_full_name, "reason": job.reason, "superseded_running": superseded, **check_run},
        )

    def _github(self) -> Any:
        """The GitHub client the ingress opens and closes Check Runs with: an object the service
        (or a test) hung on the server, else the real module."""
        client = getattr(self.server, "github", None)
        if client is not None:
            return client
        from . import webhook_github

        return webhook_github


def require_webhook_secret(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured webhook secret or fail before opening a listener."""
    source = os.environ if environ is None else environ
    secret = source.get(_WEBHOOK_SECRET_ENV, "").strip()
    if not secret:
        raise WebhookError("webhook_secret_required", f"set {_WEBHOOK_SECRET_ENV} to the GitHub App webhook secret.")
    return secret


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the Codna App webhook (blocking): ingress + durable queue + worker pool.

    Requires ``CODNA_GITHUB_WEBHOOK_SECRET``. App auth is ``GITHUB_APP_ID`` /
    ``GITHUB_APP_PRIVATE_KEY`` (a plain ``GITHUB_TOKEN`` also works for single-repo
    self-hosting). The worker runs the packaged Codna CLI in a scrubbed workspace and forwards
    only the scoped GitHub token plus an optional per-installation ``CODNA_API_KEY`` when the
    install is linked to a Codna account.
    """
    require_webhook_secret()
    import signal
    import threading

    from .webhook_procs import prepare_scratch_root
    from .webhook_queue import WebhookQueue
    from .webhook_resume import RunningJobRegistry, registry_dir_for
    from .webhook_pool import WorkerPool

    prepare_scratch_root()  # before anything touches tempfile: job scratch lives on the volume

    queue_path = default_queue_path()
    queue = WebhookQueue(queue_path)
    pool = WorkerPool(
        queue,
        concurrency=webhook_control.default_concurrency(),
        app_id=os.environ.get("GITHUB_APP_ID"),
        private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY"),
        # Jobs in flight, beside the queue on the same volume: a redeploy mid-job is resumed once on
        # the next boot (or its Check Run closed honestly) instead of spinning forever -- webhook_resume.
        registry=RunningJobRegistry(registry_dir_for(queue_path)),
    )
    pool.start()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.queue = queue  # type: ignore[attr-defined]
    httpd.worker_pool = pool  # type: ignore[attr-defined]
    print(f"codna webhook: listening on {host}:{port}/webhooks/github", file=sys.stderr)

    def _graceful(_sig: int, _frame: Any) -> None:
        # serve_forever() can't shut itself down from its own thread — do it from another.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _graceful)
        except (ValueError, OSError):
            pass  # not the main thread (e.g. under a test harness) — skip handler install
    try:
        httpd.serve_forever()
    finally:
        pool.stop()          # drain in-flight jobs up to the grace window (no abandon on redeploy)
        httpd.server_close()
