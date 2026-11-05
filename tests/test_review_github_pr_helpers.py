"""The two pull-request helpers a durable run opens its PR with (review_github).

`find_open_pr_for_branch` and `create_pull_request` exist for `durable.activities.open_pr`, whose
idempotency is structural: the head branch is ``codna/run-<run>/task-<n>``, so "already opened" is
exactly "a PR exists whose head is this branch". Two properties are worth testing offline, because
each has a failure mode that is silent in production:

* the ``head`` filter MUST be qualified as ``owner:branch`` — GitHub requires it, and an unqualified
  value matches nothing, so a retry would open a DUPLICATE pull request every time;
* a provider's error body is carried into the raised message, so the caller sees the reason instead
  of a bare status code. GitHub's 422 ("A pull request already exists") is the expected race, and
  "which request failed and why" is the whole diagnosis.

Tested through `httpx.MockTransport` installed as the module's client, so no network and no
monkeypatched internals: the assertions are about the request that would have been sent and the answer
that comes back.
"""
from __future__ import annotations

import json

import httpx
import pytest

from codna import review_github as rg


def _install(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """Route `httpx.get`/`httpx.post` through a mock transport, recording what was sent.

    `review_github` imports httpx INSIDE each function (so importing the module needs no HTTP stack),
    which means patching the `httpx` module itself is what a lazy import resolves to — and is why the
    patch targets `httpx`, not an attribute of `review_github`.
    """
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: client.get(url, **kw))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: client.post(url, **kw))
    return seen


def test_find_open_pr_qualifies_the_head_with_the_owner(monkeypatch):
    """The filter GitHub actually honours: ``owner:branch``.

    An unqualified ``head=branch`` silently matches nothing, which turns every retried `open_pr`
    into a second pull request. That is the precise bug this asserts against.
    """
    seen = _install(
        monkeypatch,
        lambda request: httpx.Response(200, json=[{"number": 12, "html_url": "https://x/12", "head": {"ref": "b"}}]),
    )

    found = rg.find_open_pr_for_branch("thyn-ai/codna", "b", "tok")

    assert found == {"number": 12, "url": "https://x/12", "branch": "b"}
    assert seen[0].url.params["head"] == "thyn-ai:b"
    assert seen[0].url.params["state"] == "open"


def test_find_open_pr_returns_none_when_no_pr_has_that_head(monkeypatch):
    _install(monkeypatch, lambda request: httpx.Response(200, json=[]))
    assert rg.find_open_pr_for_branch("thyn-ai/codna", "b", "tok") is None


def test_find_open_pr_ignores_a_pr_whose_head_is_a_different_branch(monkeypatch):
    """Defence in depth: even if the server ignored the filter, the answer is matched by branch."""
    _install(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=[{"number": 1, "html_url": "https://x/1", "head": {"ref": "other"}}]
        ),
    )
    assert rg.find_open_pr_for_branch("thyn-ai/codna", "b", "tok") is None


def test_find_open_pr_surfaces_the_providers_message(monkeypatch):
    _install(
        monkeypatch,
        lambda request: httpx.Response(500, json={"message": "boom"}),
    )
    with pytest.raises(RuntimeError, match="500"):
        rg.find_open_pr_for_branch("thyn-ai/codna", "b", "tok")


def test_create_pull_request_returns_the_number_and_url(monkeypatch):
    seen = _install(
        monkeypatch,
        lambda request: httpx.Response(201, json={"number": 7, "html_url": "https://gh/7"}),
    )

    created = rg.create_pull_request(
        "thyn-ai/codna", token="tok", title="t", head="b", base="main", body="body"
    )

    assert created == {"number": 7, "url": "https://gh/7", "branch": "b"}
    body = json.loads(seen[0].content)
    assert body == {"title": "t", "head": "b", "base": "main", "body": "body", "draft": False}


def test_create_pull_request_reports_a_422_with_the_providers_text(monkeypatch):
    """The expected race: a retried activity whose first attempt already opened the PR."""
    _install(
        monkeypatch,
        lambda request: httpx.Response(
            422, json={"message": "Validation Failed", "errors": [{"message": "A pull request already exists"}]}
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        rg.create_pull_request("thyn-ai/codna", token="tok", title="t", head="b", base="main")

    message = str(excinfo.value)
    assert "422" in message
    assert "already exists" in message


def test_create_pull_request_refuses_a_response_without_a_number(monkeypatch):
    """A 2xx with no number is not a success: the caller needs an id it can record."""
    _install(monkeypatch, lambda request: httpx.Response(201, json={"html_url": "https://gh/x"}))
    with pytest.raises(RuntimeError, match="no number"):
        rg.create_pull_request("thyn-ai/codna", token="tok", title="t", head="b", base="main")
