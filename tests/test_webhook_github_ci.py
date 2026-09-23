"""webhook_github CI-evidence helpers: pagination can never produce a false 'green'."""
from __future__ import annotations

import httpx

from codna import webhook_github as gh


class _Resp:
    def __init__(self, payload, link=""):
        self.status_code, self._payload, self.headers = 200, payload, {"link": link}

    def json(self):
        return self._payload


def _run(i, conclusion="success"):
    return {"id": i, "name": f"job {i}", "status": "completed", "conclusion": conclusion,
            "html_url": f"https://x/{i}", "app": {"slug": "github-actions"}}


def test_failing_runs_follow_link_pagination(monkeypatch):
    pages = {
        "https://api.github.com/repos/acme/app/check-suites/5/check-runs": _Resp(
            {"total_count": 3, "check_runs": [_run(1), _run(2)]},
            link='<https://api.github.com/p2>; rel="next", <https://api.github.com/p2>; rel="last"'),
        "https://api.github.com/p2": _Resp({"total_count": 3, "check_runs": [_run(3, "failure")]}),
    }
    seen = []
    monkeypatch.setattr(httpx, "get", lambda url, **kw: seen.append(url) or pages[url])
    out = gh.failing_check_runs_in_suite("acme/app", "tok", 5)
    assert [r["id"] for r in out] == [3]                      # the failure sat on page 2
    assert seen == list(pages)


def test_a_short_listing_is_unknown_never_green(monkeypatch):
    """total_count says 150, one page of 100 came back with no next link: answer None (unknown),
    not [] (green) -- a false green skips a real code failure."""
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _Resp(
        {"total_count": 150, "check_runs": [_run(i) for i in range(100)]}))
    assert gh.failing_check_runs_in_suite("acme/app", "tok", 5) is None


def test_next_page_parsing():
    assert gh._next_page('<https://a/b?page=2>; rel="next", <https://a/b?page=9>; rel="last"') == "https://a/b?page=2"
    assert gh._next_page('<https://a/b?page=9>; rel="last"') is None
    assert gh._next_page("") is None


def test_next_page_tolerates_commas_inside_urls():
    link = '<https://a/b?q=x,y&page=2>; rel="next", <https://a/b?q=x,y&page=9>; rel="last"'
    assert gh._next_page(link) == "https://a/b?q=x,y&page=2"
    assert gh._next_page('<https://a/b?page=1>; rel="prev", <https://a/b?page=3>; rel="next"') == "https://a/b?page=3"
