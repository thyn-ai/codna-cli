"""``codna review`` — read-only review of a change.

Default: a **findings** review — diff the change and run the read-only Cline agent
(``task_kind="review"``) over it to produce structured, deduped, noise-controlled
:class:`~codna.review_findings.CodnaReviewFinding` records, optionally posted to a PR as one review
(inline comments + summary + a "codna review" check). ``--triage`` keeps the older engine-backed risk
triage (suspect files/symbols + context reduction).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess


class ReviewError(RuntimeError):
    code = "review_error"

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        # Per-INSTANCE. `details` used to be a bare class attribute, so it was one shared dict
        # that nothing ever wrote to: every review failure serialised `"details": {}`. That
        # included the agent-core failures, whose wrapped PackagedRepositoryAdvancedError
        # carries sidecar_url / status / terminal_state / error -- all of it dropped by the
        # stringifying `except` below. The check run then said only "review agent failed:
        # Local agent-core did not complete the packaged fix run successfully." with an empty
        # object, which names the symptom and withholds every field needed to act on it.
        self.details: dict[str, object] = dict(details or {})


class ReviewTimeout(ReviewError):
    """The review turn outran the budget it was granted (review_budget).

    Reported under its own code so the check run reads ``codna review failed (review_timeout): ...``
    with the diff's size, the budget and how to proceed, rather than the generic sentence. The check
    conclusion stays ``failure`` like every other review failure (webhook_worker.process_job marks a
    non-ok job ``failure``; ``neutral`` would let an unreviewed PR through a queue that requires the
    review, cf. ``_inherit_review_for_merge_group``), and no review -- so no approval -- is posted,
    because this is raised before ``review_github.post_review`` runs. The webhook queue retries a
    ``review_timeout`` (webhook_summaries._NON_RETRYABLE_CODES does not list it), and the retry's
    budget grows from the recorded, exhausted turn.
    """

    code = "review_timeout"


def _fail(message: str, details: dict[str, object] | None = None) -> None:
    raise ReviewError(message, details)


def _env_check_run_id() -> int | None:
    """``CODNA_CHECK_RUN_ID``: a Check Run the caller (the webhook worker) already opened for this
    review; ``--post`` completes it instead of creating a second one. Unset or garbage -> None."""
    raw = os.environ.get("CODNA_CHECK_RUN_ID", "").strip()
    return int(raw) if raw.isdigit() else None


_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


def _env_check_run_head_sha() -> str | None:
    """``CODNA_CHECK_RUN_HEAD_SHA``: the commit the caller anchored that Check Run to, reported back
    in ``--json`` (``posted.check_head_sha``) next to the head this review actually read, so a
    caller can see the two disagree (thyn-ai/codna#569). Unset or not a SHA -> None."""
    raw = os.environ.get("CODNA_CHECK_RUN_HEAD_SHA", "").strip().lower()
    return raw if _SHA_RE.fullmatch(raw) else None


def _pr_head_now(repo_slug: str | None, pr_number: int | None, token: str | None) -> str | None:
    """The pull request's head at this moment, or None when it cannot be read. Read right before
    posting: a head that moved while the review ran must not receive the review's approval."""
    if not (repo_slug and pr_number):
        return None
    from . import review_github as rg

    try:
        return rg.fetch_pr(repo_slug, pr_number, token).get("head_sha") or None
    except Exception:  # noqa: BLE001 -- best-effort; an unreadable head is reported as unknown, never as moved
        return None


def _failing_checks_now(repo_slug: str | None, pr_number: int | None, token: str | None) -> list[dict] | None:
    """The required checks red on the pull request's head at this moment (one GraphQL request), or
    None when they cannot be read. Read right before posting, next to the head, so the review's
    approval line can name a red required check instead of reading as a merge go-ahead."""
    if not (repo_slug and pr_number and token):
        return None
    from . import review_github as rg

    try:
        return rg.failing_required_checks(repo_slug, pr_number, token)
    except Exception:  # noqa: BLE001 -- best-effort; unknown is said as nothing, never as green
        return None


def _env_review_run_id() -> str | None:
    """``CODNA_REVIEW_RUN_ID``: the webhook job this review runs for; stamped into the review body
    so the same job running twice posts once (review_github.post_review). Unset -> None."""
    raw = os.environ.get("CODNA_REVIEW_RUN_ID", "").strip()
    return raw if raw and len(raw) <= 64 and all(c.isalnum() or c in "_.:-" for c in raw) else None


def _wrapped_details(exc: BaseException) -> dict[str, object]:
    """Structured detail carried by a wrapped error, if it has any.

    Errors raised inside the runtime already describe themselves precisely -- a stub runtime, a
    run that reached a terminal failure, and an unreachable sidecar are three different problems
    that all stringify to nearly the same sentence. Keeping the fields is what makes them
    distinguishable from the pull request, without leaking a traceback.
    """
    details = getattr(exc, "details", None)
    out: dict[str, object] = dict(details) if isinstance(details, dict) else {}
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        out.setdefault("cause_code", code)
    return out


def changed_files(repo_dir: str, base: str = "HEAD") -> list[str]:
    """Files changed in the working tree relative to `base` (default HEAD = all uncommitted). Best-effort."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "diff", "--name-only", base],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:  # noqa: BLE001 - git absent / not a repo → nothing to review
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def run_review(c, args) -> dict:
    """Register + triage the changed files. Returns a structured review result."""
    from . import cli as _cli
    from .project_rules import read_project_guidance

    local = os.path.abspath(os.path.expanduser(args.repo))
    if not os.path.isdir(local):
        _fail("`codna review` needs a local repo path (it diffs the working tree).")
    base = getattr(args, "base", None) or "HEAD"
    files = changed_files(local, base)
    if not files:
        return {"repository": args.repo, "base": base, "changed_files": [], "note": f"no changed files vs {base}"}
    rid, snap = _cli._register(c, args.repo, getattr(args, "ref", None), focus_paths=files)
    sig = {
        "issue_text": getattr(args, "issue", None) or "Review these changes for bugs, regressions, and risk.",
        "changed_files": files,
    }
    guidance = read_project_guidance(local)
    if guidance:
        sig["project_guidance"] = guidance
    tri = _cli._dump(c.triage_repository(rid, {"snapshot_id": snap["snapshot_id"], "signals": sig}))
    return {
        "repository": args.repo,
        "base": base,
        "changed_files": files,
        "suspect_files": tri.get("suspect_files") or [],
        "suspect_symbols": tri.get("suspect_symbols") or [],
        "reduction_ratio": tri.get("reduction_ratio"),
    }


def _config_overrides(args, cfg):
    """Apply CLI flag overrides onto a loaded ReviewConfig (flags win over codna.yaml)."""
    mc = getattr(args, "min_confidence", None)
    if mc is not None:
        cfg.min_confidence = max(0.0, min(1.0, float(mc)))
    mf = getattr(args, "max_findings", None)
    if mf is not None and mf >= 0:
        cfg.max_findings = int(mf)
    if getattr(args, "blocking", False):
        cfg.blocking_enabled = True
    eff = getattr(args, "effort", None)
    if eff:
        cfg.effort = str(eff).strip().lower()
    return cfg


def _is_git_url(value: str) -> bool:
    """A remote repo spec the review path should clone rather than treat as a local dir."""
    return value.startswith(("http://", "https://", "git@", "ssh://")) or value.endswith(".git")


def _materialize_pr(url: str, pr_number: int, base_ref: str, token: str | None,
                    incremental_base: str | None = None) -> tuple[str, str, str, list[str] | None]:
    """Clone ``url`` and fetch the PR head into a temp dir. Returns ``(local_path, diff_range,
    head_sha, diff_paths)``. Default diff_range is ``origin/<base>...codna-pr-head`` (3-dot merge-base
    diff, the whole PR) with ``diff_paths=None`` (no restriction). If ``incremental_base`` (the
    last-reviewed head SHA) is given and is an ANCESTOR of the current head (i.e. no force-push
    rewrote it) and not already the head, the range narrows to ``<incremental_base>...codna-pr-head``
    — only the commits pushed since the last review — and ``diff_paths`` is the pull request's OWN
    file set (``git diff --name-only origin/<base>...codna-pr-head``), which the diff of that range
    must be confined to (``review_findings.compute_diff``).

    The confinement is what keeps base-branch history out of an incremental review. The narrowed
    range covers everything reachable from the new head and not from the old one -- and that
    includes whatever the base branch gained when the author pressed "Update branch". The new head is
    then a merge commit whose SECOND parent is reachable from ``origin/<base>`` and whose FIRST parent
    is the previously reviewed head, so the ancestor test above passes and the range's diff is the
    base branch's own recent changes: files the pull request does not touch at all. Findings anchored
    to them made GitHub reject the whole review (``422 Path could not be resolved``,
    thyn-ai/codna-action#6, head bc1219d3 = merge of main into the branch). No merge detection is
    needed to handle that case: a file the pull request does not change is never in ``diff_paths``,
    so the merge's contribution is filtered out whether or not the head is a merge commit, while a
    commit pushed after the merge is still reviewed through the files it changes. An empty
    intersection means nothing new to review, and ``compute_diff`` reports it as no changes -- a
    fast, clean result, not a failure. Uses ``pull/<n>/head`` so forked-PR heads materialize too,
    without any repo write."""
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="codna-review-")
    auth_url = url
    if token and url.startswith("https://"):
        auth_url = url.replace("https://", f"https://x-access-token:{token}@", 1)

    def _git(*a: str, allow_fail: bool = False) -> subprocess.CompletedProcess:
        p = subprocess.run(["git", *a], capture_output=True, text=True, timeout=300)
        if p.returncode != 0 and not allow_fail:
            raise RuntimeError(f"git {' '.join(a[:2])} failed: {p.stderr[-200:]}")
        return p

    _git("clone", "--no-tags", "--quiet", auth_url, tmp)
    _git("-C", tmp, "fetch", "--quiet", "origin", f"pull/{pr_number}/head:codna-pr-head")
    _git("-C", tmp, "checkout", "--quiet", "codna-pr-head")
    head_sha = _git("-C", tmp, "rev-parse", "HEAD").stdout.strip()
    diff_range = f"origin/{base_ref}...codna-pr-head"
    diff_paths: list[str] | None = None
    if incremental_base and incremental_base != head_sha:
        # incremental only when the prior head is still in this branch's history (not force-pushed away)
        anc = _git("-C", tmp, "merge-base", "--is-ancestor", incremental_base, "codna-pr-head", allow_fail=True)
        if anc.returncode == 0:
            # The pull request's own files: the incremental diff is confined to these, so an
            # "Update branch" merge in the range contributes nothing (see the docstring). The range
            # narrows only together with its confinement -- if the file set cannot be read, the
            # review stays a whole-PR review rather than an unconfined incremental one.
            names = _git("-C", tmp, "diff", "--name-only", f"origin/{base_ref}...codna-pr-head", allow_fail=True)
            if names.returncode == 0:
                diff_range = f"{incremental_base}...codna-pr-head"
                diff_paths = [ln.strip() for ln in names.stdout.splitlines() if ln.strip()]
    return tmp, diff_range, head_sha, diff_paths


def run_findings_review(args) -> dict:
    """Findings review: diff → read-only agent → normalized findings; optionally post to a PR.

    Local path (default): diffs the working tree. Remote URL (webhook/CI): clones + materializes the
    PR head. Returns the ReviewResult dict (+ a ``posted`` block when ``--post`` succeeds)."""
    from . import review_findings as rf
    from . import review_github as rg
    from .cline_agent import ReviewTurnTimeout
    from .project_rules import read_project_guidance

    diff_range = getattr(args, "diff", None)
    base = None if diff_range else getattr(args, "base", None)
    diff_paths: list[str] | None = None   # set by an incremental remote review: the PR's own files
    head_sha = None
    repo_slug = pr_number = None
    prior_feedback = None
    # Incremental review is on by default; --full forces a whole-PR review, and an explicit --diff/--base
    # range is always taken verbatim (no incremental narrowing).
    incremental = not getattr(args, "full", False) and not diff_range and not base
    token = (getattr(args, "github_token", None) or os.environ.get("CODNA_GITHUB_TOKEN")
             or os.environ.get("GITHUB_TOKEN"))
    pr = getattr(args, "pr", None)
    materialized: str | None = None  # the clone this call made and must remove

    if _is_git_url(str(args.repo)):
        # Remote review (webhook / CI): needs a PR to materialize the diff. The repo slug ALWAYS comes
        # from the URL — never the CWD's git remote — so running `codna review <url> --pr N` from
        # inside an UNRELATED repo can't misroute the review/post to that repo (a real footgun).
        if not pr:
            _fail("reviewing a remote URL needs --pr (the PR to review).")
        repo_slug = rg.repo_slug_from_url(str(args.repo))
        if not repo_slug:
            _fail(f"could not parse owner/repo from {args.repo!r}.")
        _, pr_number = rg.parse_pr_arg(str(pr), str(args.repo))  # take only the number from --pr
        if not pr_number:
            _fail(f"--pr {pr!r}: could not resolve a PR number.")
        try:
            meta = rg.fetch_pr(repo_slug, pr_number, token)
        except Exception as exc:  # noqa: BLE001
            _fail(f"could not fetch PR #{pr_number}: {exc}")
        if not meta.get("base_ref"):
            _fail(f"PR #{pr_number} has no base ref (cannot compute the diff).")
        # Cross-run context: prior human/bot comments (don't duplicate) + the last-reviewed head SHA
        # (review only the commits pushed since → incremental). Best-effort; needs a token.
        incremental_base = None
        if token:
            ctx = rg.fetch_review_context(repo_slug, pr_number, token)
            prior_feedback = ctx.get("prior_feedback")
            if incremental:
                incremental_base = ctx.get("last_reviewed_head")
        try:
            local, diff_range, head_sha, diff_paths = _materialize_pr(
                str(args.repo), pr_number, meta["base_ref"], token, incremental_base=incremental_base)
        except Exception as exc:  # noqa: BLE001
            _fail(f"could not materialize PR #{pr_number}: {exc}")
        materialized = local
        base = None
    else:
        local = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(local):
            _fail("`codna review` needs a local repo path or a git URL (with --pr).")
        if pr:
            repo_slug, pr_number = rg.parse_pr_arg(str(pr), local)
            if not repo_slug or not pr_number:
                _fail(f"--pr {pr!r}: could not resolve owner/repo#number (pass owner/repo#123 or a PR URL).")

    try:
        try:
            cfg = _config_overrides(args, rf.load_review_config(local))
        except Exception as exc:  # noqa: BLE001 — malformed config fails closed with a clear message
            _fail(f"review config error: {exc}")

        guidance = read_project_guidance(local)
        try:
            result = rf.run_diff_review(
                local, repository=args.repo, diff_range=diff_range, base=base, config=cfg,
                guidance=guidance, prior_feedback=prior_feedback, head_sha=head_sha,
                model=getattr(args, "model", None) or None, diff_paths=diff_paths,
            )
        except ReviewTurnTimeout as exc:
            # The turn outran its adaptive budget: its message already names the diff size, the
            # budget granted and how to proceed, so it is the whole error, not a prefix's tail.
            raise ReviewTimeout(str(exc), _wrapped_details(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - agent/runtime failures must not leak tracebacks.
            # Carry the wrapped error's own fields through. Not leaking a traceback is the rule;
            # discarding the structured detail alongside it was an accident, and it is what left
            # every agent-core failure reported as one sentence and an empty object.
            _fail(f"review agent failed: {exc}", _wrapped_details(exc))
        out = result.to_dict()

        if getattr(args, "post", False):
            token = (getattr(args, "github_token", None) or os.environ.get("CODNA_GITHUB_TOKEN")
                     or os.environ.get("GITHUB_TOKEN"))
            if not (repo_slug and pr_number):
                _fail("--post needs --pr owner/repo#123 (or a PR number in a repo with a github origin).")
            if not token:
                _fail("--post needs a write token: pass --github-token or set GITHUB_TOKEN.")
            try:
                out["posted"] = rg.post_review(result, repo_slug=repo_slug, pr_number=pr_number, token=token,
                                               check_run_id=_env_check_run_id(), approve=cfg.approve_clean,
                                               check_head_sha=_env_check_run_head_sha(),
                                               pr_head_sha=_pr_head_now(repo_slug, pr_number, token),
                                               run_id=_env_review_run_id(),
                                               failing_checks=_failing_checks_now(repo_slug, pr_number, token))
            except Exception as exc:  # noqa: BLE001
                _fail(f"posting the review failed: {exc}")
        return out
    finally:
        if materialized is not None:
            # The clone lives in the process's temp dir, NOT in a per-call scratch dir. Without
            # this, every remote review left a full checkout behind on the webhook machine until
            # its root filesystem filled and reviews died with ENOSPC (thyn-ai/algenta#1023).
            shutil.rmtree(materialized, ignore_errors=True)


def dispatch(args, make_client) -> int:
    """Run ``codna review`` and print the result; return the process exit code. ``--triage`` uses the
    engine-backed risk triage (needs a client); otherwise the findings review runs locally."""
    import json

    if getattr(args, "triage", False):
        result = run_review(make_client(), args)
        if getattr(args, "as_json", False):
            print(json.dumps(result, indent=2))
            return 0
        if not result.get("changed_files"):
            print(f"codna: {result.get('note', 'no changes to review')}")
            return 0
        files = result["changed_files"]
        print(f"✓ reviewed {len(files)} changed file(s) vs {result['base']}")
        print(f"  changed      : {', '.join(files[:8])}" + (" …" if len(files) > 8 else ""))
        print(f"  suspect files: {', '.join(result['suspect_files']) or '(none)'}")
        if result.get("suspect_symbols"):
            print(f"  suspect syms : {', '.join(result['suspect_symbols'][:8])}")
        return 0

    result = run_findings_review(args)
    if getattr(args, "as_json", False):
        print(json.dumps(result, indent=2))
    else:
        for line in render_findings(result):
            print(line)
    # Non-zero exit only when the check is a configured failure (blocking) — never for neutral/success.
    return 1 if result.get("conclusion") == "failure" else 0


def render_findings(result: dict) -> list[str]:
    """Human rendering of a findings-review result (the non-JSON output)."""
    if result.get("note") and not result.get("findings"):
        return [f"codna: {result['note']}"]
    findings = result.get("findings") or []
    base = result.get("base")
    head = f" @ {result['head_sha'][:8]}" if result.get("head_sha") else ""
    lines = [f"\n✓ codna reviewed {len(result.get('changed_files') or [])} changed file(s) vs {base}{head}"]
    if not findings:
        lines.append("  no high-confidence issues found ✅")
        return lines
    emoji = {"high": "🔴", "medium": "🟡", "low": "⚪"}
    lines.append(f"  {len(findings)} finding(s)  ·  check: {result.get('conclusion')}")
    for f in findings:
        loc = f"{f['path']}:{f['line']}" + (f"-{f['end_line']}" if f.get("end_line") else "")
        tag = "inline" if f.get("diff_anchor") else "summary"
        lines.append(f"\n  {emoji.get(f['severity'], '•')} [{f['severity']}/{f['category']}] {loc} ({tag})")
        lines.append(f"     {f['title']}")
        if f.get("explanation"):
            lines.append(f"     {f['explanation']}")
    posted = result.get("posted")
    if posted:
        lines.append(f"\n✓ posted {posted.get('inline_posted', 0)} inline comment(s)"
                     + (f", skipped {posted['skipped_duplicates']} duplicate(s)" if posted.get("skipped_duplicates") else ""))
    return lines
