"""A red required check on the PR head at review time is NAMED next to the verdict
(review_github.failing_required_checks / red_head_note / post_review), so "Approved" -- which means
"no medium/high findings" and nothing else -- is never read as "safe to merge". thyn-ai/codna-site#57
was approved while its required `Vercel` status was FAILURE (v2 removed a path the site imports)."""
from __future__ import annotations

import httpx
import pytest

from codna import review_findings as rf
from codna import review_github as rg

HEAD = "615e29cef3d90987fa098a84e8eef1ed1d572816"


def _rollup(nodes, *, more=False, cursor=None):
    """The recorded shape of the statusCheckRollup query on codna-site#57 (2026-09-20)."""
    return {"data": {"repository": {"pullRequest": {"commits": {"nodes": [{"commit": {
        "oid": HEAD, "statusCheckRollup": {"contexts": {"pageInfo": {"hasNextPage": more, "endCursor": cursor}, "nodes": nodes}}}}]}}}}}


def _green(name="ci"):
    return {"__typename": "CheckRun", "name": name, "conclusion": "SUCCESS", "status": "COMPLETED", "isRequired": True}


SITE_57 = [
    {"__typename": "CheckRun", "name": "full / security gate", "conclusion": "SUCCESS", "status": "COMPLETED", "isRequired": True},
    {"__typename": "StatusContext", "context": "Vercel", "state": "FAILURE", "isRequired": True},
    {"__typename": "CheckRun", "name": "codna review", "conclusion": "NEUTRAL", "status": "COMPLETED", "isRequired": True},
    {"__typename": "CheckRun", "name": "Vercel Preview Comments", "conclusion": "SUCCESS", "status": "COMPLETED", "isRequired": False},
]


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, ""

    def json(self):
        return self._d


def _ar(findings=()):
    return rf.ReviewResult(repository="u", base="b", head_sha=HEAD, changed_files=["package.json"],
                           inline_findings=list(findings), summary_findings=[], conclusion="neutral" if findings else "success",
                           dropped={})


def test_failing_required_checks_names_only_the_required_red_contexts(monkeypatch):
    seen = {}
    monkeypatch.setattr(httpx, "post", lambda url, **k: seen.update(k["json"]["variables"]) or _Resp(_rollup(SITE_57)))
    assert rg.failing_required_checks("thyn-ai/codna-site", 57, "tok") == [{"name": "Vercel", "state": "FAILURE", "head_sha": HEAD}]
    assert seen == {"owner": "thyn-ai", "name": "codna-site", "number": 57, "after": None}   # ONE GraphQL request, the PR's own number


def test_failing_required_checks_reads_check_runs_and_statuses_and_ignores_cancelled(monkeypatch):
    nodes = [
        {"__typename": "CheckRun", "name": "ci / cli tests", "conclusion": "FAILURE", "status": "COMPLETED", "isRequired": True},
        {"__typename": "CheckRun", "name": "ci / lint", "conclusion": "TIMED_OUT", "status": "COMPLETED", "isRequired": True},
        {"__typename": "CheckRun", "name": "ci / flaky", "conclusion": "CANCELLED", "status": "COMPLETED", "isRequired": True},
        {"__typename": "CheckRun", "name": "ci / running", "conclusion": None, "status": "IN_PROGRESS", "isRequired": True},
        {"__typename": "CheckRun", "name": "optional / red", "conclusion": "FAILURE", "status": "COMPLETED", "isRequired": False},
        {"__typename": "StatusContext", "context": "Vercel", "state": "ERROR", "isRequired": True},
        {"__typename": "StatusContext", "context": "Vercel – docs", "state": "PENDING", "isRequired": True},
    ]
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup(nodes)))
    assert [(c["name"], c["state"]) for c in rg.failing_required_checks("o/r", 1, "tok")] == [
        ("ci / cli tests", "FAILURE"), ("ci / lint", "TIMED_OUT"), ("Vercel", "ERROR")]


def test_failing_required_checks_is_unknown_never_green_when_the_answer_is_not_certain(monkeypatch):
    assert rg.failing_required_checks("o/r", 1, None) is None                      # no token
    assert rg.failing_required_checks("nonsense", 1, "tok") is None                # no owner/repo
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp({"message": "bad credentials"}, 401))
    assert rg.failing_required_checks("o/r", 1, "tok") is None
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp({"data": {"repository": None}}))
    assert rg.failing_required_checks("o/r", 1, "tok") is None                     # partial / errored payload

    def boom(url, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "post", boom)
    assert rg.failing_required_checks("o/r", 1, "tok") is None
    green = [_green()]
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup(green, more=True)))
    assert rg.failing_required_checks("o/r", 1, "tok") is None                     # more pages, no cursor to read them: unknown
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup(green)))
    assert rg.failing_required_checks("o/r", 1, "tok") == []                       # certain: nothing required is red


def test_failing_required_checks_walks_a_rollup_longer_than_one_page(monkeypatch):
    """A PR with more than 100 contexts: the red required one on page 2 is found, and every page's
    red checks are named (not only the first page's)."""
    red2 = {"__typename": "StatusContext", "context": "Vercel", "state": "FAILURE", "isRequired": True}
    red1 = {"__typename": "CheckRun", "name": "ci / lint", "conclusion": "FAILURE", "status": "COMPLETED", "isRequired": True}
    pages = {None: _rollup([_green(f"ci-{i}") for i in range(99)] + [red1], more=True, cursor="c1"),
             "c1": _rollup([_green("late"), red2])}
    asked = []

    def post(url, **k):
        asked.append(k["json"]["variables"]["after"])
        return _Resp(pages[k["json"]["variables"]["after"]])

    monkeypatch.setattr(httpx, "post", post)
    assert [c["name"] for c in rg.failing_required_checks("o/r", 1, "tok")] == ["ci / lint", "Vercel"]
    assert asked == [None, "c1"]                                                    # one request per page, in order

    # Beyond the page cap: red ones already seen are still named (a partial warning beats none);
    # nothing red so far is unknown, never green.
    def endless_red(url, **k):
        return _Resp(_rollup([red1], more=True, cursor="again"))

    monkeypatch.setattr(httpx, "post", endless_red)
    assert [c["name"] for c in rg.failing_required_checks("o/r", 1, "tok")] == ["ci / lint"] * rg._ROLLUP_PAGES_MAX
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup([_green()], more=True, cursor="again")))
    assert rg.failing_required_checks("o/r", 1, "tok") is None


def test_a_failed_commit_status_alone_is_named_in_the_red_head_note(monkeypatch):
    """The codna-site#57 case end to end: the head's only red required context is a StatusContext
    (Vercel, FAILURE -- a deployment, not a check run). With a token that may read commit statuses the
    rollup carries it, and the note names it as `Vercel` (failure)."""
    nodes = [_green("full / security gate"), _green("codna review"),
             {"__typename": "StatusContext", "context": "Vercel", "state": "FAILURE", "isRequired": True},
             {"__typename": "StatusContext", "context": "Vercel – preview", "state": "SUCCESS", "isRequired": True}]
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup(nodes)))
    red = rg.failing_required_checks("thyn-ai/codna-site", 57, "tok-with-statuses-read")
    assert red == [{"name": "Vercel", "state": "FAILURE", "head_sha": HEAD}]
    note = rg.red_head_note(red)
    assert "`Vercel` (failure)" in note and "head 615e29ce is red" in note
    # an ERROR status is red too; a PENDING or SUCCESS one is not
    nodes[2]["state"] = "ERROR"
    assert rg.red_head_note(rg.failing_required_checks("thyn-ai/codna-site", 57, "tok")).count("`Vercel` (error)") == 1
    nodes[2]["state"] = "PENDING"
    assert rg.failing_required_checks("thyn-ai/codna-site", 57, "tok") == []


def test_a_token_without_statuses_read_sees_no_commit_status_and_the_note_stays_silent(monkeypatch):
    """What the review saw on codna-site#57 before the App had Commit statuses: read: GitHub lists the
    check runs only, the rollup is certain-and-green, and nothing is said -- the reason the review
    token now asks for `statuses: read` (webhook_github.token_permissions_for)."""
    check_runs_only = [n for n in SITE_57 if n["__typename"] == "CheckRun"]
    monkeypatch.setattr(httpx, "post", lambda url, **k: _Resp(_rollup(check_runs_only)))
    assert rg.failing_required_checks("thyn-ai/codna-site", 57, "tok-without-statuses") == []
    assert rg.red_head_note([]) == ""


def test_red_head_note_is_one_line_naming_the_checks_and_the_head():
    note = rg.red_head_note([{"name": "Vercel", "state": "FAILURE", "head_sha": HEAD},
                             {"name": "ci / cli tests", "state": "TIMED_OUT", "head_sha": HEAD}])
    assert "\n" not in note
    assert note.startswith("🔴 Heads-up: head 615e29ce is red") and "`Vercel` (failure)" in note and "`ci / cli tests` (timed_out)" in note
    assert "not a merge go-ahead" in note
    assert rg.red_head_note([]) == ""


@pytest.fixture
def github(monkeypatch):
    from codna import webhook_github

    state = {"posts": [], "checks": []}
    monkeypatch.setattr(httpx, "get", lambda url, **k: _Resp([]))
    monkeypatch.setattr(httpx, "post", lambda url, **k: state["posts"].append((url, k["json"])) or _Resp({}))
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    monkeypatch.setattr(rg, "fetch_pr_files", lambda *a, **k: None)
    monkeypatch.setattr(webhook_github, "update_check_run",
                        lambda repo, token, cid, **k: state["checks"].append((cid, k["conclusion"], k["summary"])))
    monkeypatch.setattr(webhook_github, "create_check_run", lambda *a, **k: 777)
    return state


def test_post_review_still_approves_a_clean_diff_but_says_the_head_is_red_in_body_and_check(github):
    """Approval semantics are untouched -- no medium/high findings = APPROVE -- and the red required
    check is named right after the approval line, in the review body AND the check-run summary."""
    red = [{"name": "Vercel", "state": "FAILURE", "head_sha": HEAD}]
    out = rg.post_review(_ar(), repo_slug="o/r", pr_number=57, token="t", dedup=False, check_run_id=9001, failing_checks=red)
    assert out["event"] == "APPROVE" and out["posted_review"] is True and out["failing_required_checks"] == red
    [(url, payload)] = github["posts"]
    assert url.endswith("/repos/o/r/pulls/57/reviews") and payload["event"] == "APPROVE"
    body = payload["body"]
    assert "✅ Approved: no medium/high findings and no unresolved codna threads." in body
    assert "🔴 Heads-up: head 615e29ce is red -- required check(s) failing at review time: `Vercel` (failure)." in body
    assert body.index("✅ Approved") < body.index("🔴 Heads-up")                       # verdict first, then the warning
    [(cid, conclusion, summary)] = github["checks"]
    assert cid == 9001 and conclusion == "success" and "`Vercel` (failure)" in summary


def test_post_review_says_nothing_extra_when_no_required_check_is_red_or_the_state_is_unknown(github):
    for checks in (None, []):
        github["posts"].clear()
        out = rg.post_review(_ar(), repo_slug="o/r", pr_number=57, token="t", dedup=False, failing_checks=checks)
        assert out["event"] == "APPROVE" and out["failing_required_checks"] == checks
        assert "Heads-up" not in github["posts"][0][1]["body"]


def test_the_red_line_composes_with_a_not_approved_verdict_and_a_moved_head(github):
    f = rf.CodnaReviewFinding(path="package.json", line=1, severity="medium", category="correctness", title="t", explanation="x",
                              confidence=0.9, fingerprint="f1", diff_anchor=rf.DiffAnchor(commit_id=HEAD, side="RIGHT", line=1))
    red = [{"name": "ci / build", "state": "FAILURE", "head_sha": "a" * 40}]
    out = rg.post_review(_ar([f]), repo_slug="o/r", pr_number=57, token="t", dedup=False, pr_head_sha="a" * 40, failing_checks=red)
    assert out["event"] == "COMMENT"
    body = github["posts"][0][1]["body"]
    assert "⏸ Not approved: 1 medium/high finding(s) in this review." in body
    assert "head moved from 615e29ce to aaaaaaaa" in body and "🔴 Heads-up: head aaaaaaaa is red" in body


def test_failing_checks_now_is_best_effort(monkeypatch):
    from codna import review

    assert review._failing_checks_now(None, 5, "t") is None and review._failing_checks_now("o/r", 5, None) is None
    monkeypatch.setattr(rg, "failing_required_checks", lambda *a, **k: [{"name": "Vercel", "state": "FAILURE", "head_sha": HEAD}])
    assert review._failing_checks_now("o/r", 5, "t")[0]["name"] == "Vercel"

    def boom(*a, **k):
        raise RuntimeError("graphql down")

    monkeypatch.setattr(rg, "failing_required_checks", boom)
    assert review._failing_checks_now("o/r", 5, "t") is None
