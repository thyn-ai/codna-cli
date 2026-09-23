"""The per-job run stamp on App-posted reviews (review_github.run_marker): the same webhook job
running twice -- a lease handed to a second worker while the first is finishing, a retry after the
review posted -- must not post two reviews or two approvals. The check run is still completed,
because the caller opened it and nothing else will."""
from __future__ import annotations

import httpx
import pytest

from codna import review_findings as rf
from codna import review_github as rg


def _result(findings=()):
    return rf.ReviewResult(repository="u", base="b", head_sha="h" * 40, changed_files=["a.py"],
                           inline_findings=list(findings), summary_findings=[], conclusion="neutral" if findings else "success",
                           dropped={})


def _finding(fp="f1"):
    return rf.CodnaReviewFinding(path="a.py", line=1, severity="medium", category="correctness", title=f"t-{fp}",
                                 explanation="x", confidence=0.9, fingerprint=fp, end_line=None,
                                 diff_anchor=rf.DiffAnchor(commit_id="h" * 40, side="RIGHT", line=1))


def test_summary_body_stamps_the_run_id_only_when_given():
    assert "codna:review:run=" not in rg.summary_body(_result())
    body = rg.summary_body(_result(), run_id="1234")
    assert rg.run_marker("1234") in body and "codna:reviewed-head=" in body
    assert rg.existing_run_ids([{"body": body}, {"body": "no marker"}]) == {"1234"}
    assert rg.existing_run_ids([{"body": rg.summary_body(_result([_finding()]), run_id="9")}]) == {"9"}
    assert rg.run_marker(None) == ""


def test_env_review_run_id_accepts_a_row_id_and_rejects_garbage(monkeypatch):
    from codna.review import _env_review_run_id

    monkeypatch.setenv("CODNA_REVIEW_RUN_ID", " 42 ")
    assert _env_review_run_id() == "42"
    monkeypatch.setenv("CODNA_REVIEW_RUN_ID", "x" * 100)
    assert _env_review_run_id() is None
    monkeypatch.setenv("CODNA_REVIEW_RUN_ID", "12 --> <script>")
    assert _env_review_run_id() is None
    monkeypatch.delenv("CODNA_REVIEW_RUN_ID")
    assert _env_review_run_id() is None


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text, self.headers = data, status, "", {}

    def json(self):
        return self._d


@pytest.fixture
def github(monkeypatch):
    """A PR whose review list and check-run calls are recorded; ``reviews`` is what GitHub already has."""
    from codna import webhook_github

    state = {"reviews": [], "posts": [], "checks": []}
    monkeypatch.setattr(httpx, "get", lambda url, **k: _Resp(state["reviews"] if url.endswith("/reviews") else []))
    monkeypatch.setattr(httpx, "post", lambda url, **k: state["posts"].append((url, k["json"])) or _Resp({}))
    monkeypatch.setattr(rg, "unresolved_blocking_findings", lambda *a, **k: 0)
    monkeypatch.setattr(rg, "fetch_pr_files", lambda *a, **k: None)
    monkeypatch.setattr(webhook_github, "update_check_run",
                        lambda repo, token, cid, **k: state["checks"].append((cid, k["conclusion"], k["summary"])))
    monkeypatch.setattr(webhook_github, "create_check_run", lambda *a, **k: 777)
    return state


def test_the_same_job_running_twice_posts_one_review_and_completes_the_check_both_times(github):
    first = rg.post_review(_result(), repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9001, run_id="31")
    assert first["posted_review"] is True and first["duplicate_run"] is False and first["event"] == "APPROVE"
    [(url, payload)] = github["posts"]
    assert url.endswith("/repos/o/r/pulls/5/reviews") and rg.run_marker("31") in payload["body"]
    github["reviews"] = [{"body": payload["body"]}]                                # GitHub now carries that review
    second = rg.post_review(_result(), repo_slug="o/r", pr_number=5, token="t", dedup=False, check_run_id=9002, run_id="31")
    assert second["posted_review"] is False and second["duplicate_run"] is True
    assert len(github["posts"]) == 1                                                  # no second review, no second approval
    assert [c[0] for c in github["checks"]] == [9001, 9002]                           # each run's own check is completed
    assert rg.run_marker("31") in github["checks"][1][2]


def test_a_different_job_on_the_same_head_still_posts(github):
    github["reviews"] = [{"body": rg.summary_body(_result(), run_id="31")}]
    out = rg.post_review(_result([_finding()]), repo_slug="o/r", pr_number=5, token="t", dedup=False, run_id="32")
    assert out["posted_review"] is True and out["duplicate_run"] is False
    assert rg.run_marker("32") in github["posts"][0][1]["body"]


def test_without_a_run_id_behaviour_is_unchanged(github):
    github["reviews"] = [{"body": rg.summary_body(_result(), run_id="31")}]
    out = rg.post_review(_result(), repo_slug="o/r", pr_number=5, token="t", dedup=False)   # the standalone CLI
    assert out["posted_review"] is True and out["duplicate_run"] is False
    assert "codna:review:run=" not in github["posts"][0][1]["body"]


def test_a_failed_stamp_lookup_never_blocks_posting(monkeypatch, github):
    def boom(url, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "get", boom)
    out = rg.post_review(_result(), repo_slug="o/r", pr_number=5, token="t", dedup=False, run_id="31")
    assert out["posted_review"] is True and out["duplicate_run"] is False


def test_the_stamp_check_reads_the_first_and_the_newest_pages_only(monkeypatch):
    """A stamp on a recent review of a long-lived PR (page 7 of 9) must be seen, and the check
    must stay off the critical path: never more than five requests however long the history."""
    last = 9
    pages = {n: [{"body": f"review p{n}-{i}"} for i in range(100)] for n in range(1, last + 1)}
    pages[7][40] = {"body": rg.summary_body(_result(), run_id="31")}
    pages[last] = pages[last][:20]
    seen = []

    class _Page:
        status_code = 200

        def __init__(self, page):
            self._p = page
            self.headers = {"link": '<https://api.github.com/repos/o/r/pulls/5/reviews?per_page=100&page=2>; rel="next", '
                                    f'<https://api.github.com/repos/o/r/pulls/5/reviews?per_page=100&page={last}>; rel="last"'}

        def json(self):
            return pages.get(self._p, [])

    def fake_get(url, **k):
        assert url.endswith("/pulls/5/reviews")
        seen.append(k["params"]["page"])
        return _Page(k["params"]["page"])

    monkeypatch.setattr(httpx, "get", fake_get)
    reviews = rg.list_pr_reviews("o/r", 5, "t")
    assert seen == [1] + list(range(last - rg._STAMP_TAIL_PAGES + 1, last + 1))   # first page + the newest four: five requests
    assert len(seen) <= 5 and len(reviews) == 100 + 3 * 100 + 20
    assert rg.existing_run_ids(reviews) == {"31"}
    pages[7][40] = {"body": "no stamp"}
    assert rg.existing_run_ids(rg.list_pr_reviews("o/r", 5, "t")) == set()
    del seen[:]
    pages_short = {1: [{"body": "a"}] * 100, 2: [{"body": "b"}] * 100, 3: [{"body": rg.summary_body(_result(), run_id="8")}]}
    pages.clear()
    pages.update(pages_short)
    last = 3
    assert rg.existing_run_ids(rg.list_pr_reviews("o/r", 5, "t")) == {"8"} and seen == [1, 2, 3]   # never a page below 2 twice


def test_the_stamp_check_makes_one_request_for_a_short_history(monkeypatch):
    seen = []

    class _One:
        status_code = 200
        headers = {}

        def json(self):
            return [{"body": rg.summary_body(_result(), run_id="7")}]

    monkeypatch.setattr(httpx, "get", lambda url, **k: seen.append(k["params"]["page"]) or _One())
    assert rg.existing_run_ids(rg.list_pr_reviews("o/r", 5, "t")) == {"7"} and seen == [1]


def test_the_stamp_check_survives_a_page_error(monkeypatch):
    class _Err:
        status_code = 502
        headers = {}

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "get", lambda url, **k: _Err())
    assert rg.list_pr_reviews("o/r", 5, "t") == []
