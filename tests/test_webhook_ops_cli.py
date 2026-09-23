"""The CLI front of the operator surface, without a database: the autoscaler's token precedence
(``CODNA_WEBHOOK_FLY_TOKEN`` first, ``FLY_API_TOKEN`` as the fallback) and ``codna webhook ops
scaler`` answering for the SERVING process through its loopback ``/ops/scaler`` (``"view": "live"``),
falling back to its own view marked ``"view": "cli-rebuild"`` with the reason. The live path never
opens the queue and the fallback runs against a stand-in, so none of this needs Postgres."""
from __future__ import annotations

import argparse
import json

import httpx
import pytest

from codna import webhook_ops
from codna.webhook_ops import LIVE_VERBS, cli_ops, live_verb
from codna.webhook_scaler import FLY_TOKEN_ENVS, fly_token_from_env, machines_client_from_env

TOKEN = "ops-bearer-for-tests"


# --- the Machines API credential -------------------------------------------------------------------------

def test_fly_token_precedence_is_the_services_own_name_then_flyctls():
    assert FLY_TOKEN_ENVS == ("CODNA_WEBHOOK_FLY_TOKEN", "FLY_API_TOKEN")
    assert fly_token_from_env({}) is None
    assert fly_token_from_env({"FLY_API_TOKEN": "legacy"}) == "legacy"                                   # the fallback still works
    assert fly_token_from_env({"CODNA_WEBHOOK_FLY_TOKEN": "ours", "FLY_API_TOKEN": "legacy"}) == "ours"
    assert fly_token_from_env({"CODNA_WEBHOOK_FLY_TOKEN": " \n", "FLY_API_TOKEN": "legacy"}) == "legacy"  # blank = unset
    assert machines_client_from_env({"CODNA_WEBHOOK_FLY_TOKEN": "", "FLY_API_TOKEN": ""}) is None
    client = machines_client_from_env({"CODNA_WEBHOOK_FLY_TOKEN": "ours", "FLY_API_TOKEN": "legacy", "CODNA_WEBHOOK_WORKER_APP": "w"})
    assert client is not None and (client._token, client._app) == ("ours", "w")


# --- the live view ---------------------------------------------------------------------------------------

def _client(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def _refuse(request):
    raise httpx.ConnectError("connection refused", request=request)


def _never(request):
    raise AssertionError("no request may be made here")


def test_live_verb_asks_the_serving_process_with_the_ops_bearer_and_marks_the_view(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = httpx.URL(str(request.url))
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"configured": True, "paused": False, "scaler_leader": 1})

    monkeypatch.setattr(webhook_ops, "_live_client", _client(handler))
    env = {"CODNA_WEBHOOK_OPS_TOKEN": TOKEN, "CODNA_WEBHOOK_PORT": "8181"}
    out, why = live_verb("scaler", {"action": "status", "limit": 50, "value": None}, actor="github-actions:octocat", environ=env)
    assert why == "" and out == {"configured": True, "paused": False, "scaler_leader": 1, "view": "live"}
    assert seen["auth"] == f"Bearer {TOKEN}"
    assert (seen["url"].host, seen["url"].port, seen["url"].path) == ("127.0.0.1", 8181, "/ops/scaler")
    assert dict(seen["url"].params) == {"action": "status", "limit": "50", "actor": "github-actions:octocat"}  # None dropped


@pytest.mark.parametrize("env, handler, expect", [
    ({}, _never, "CODNA_WEBHOOK_OPS_TOKEN is not set"),
    ({"CODNA_WEBHOOK_OPS_TOKEN": TOKEN}, _refuse, "loopback /ops/scaler unreachable: ConnectError"),
    ({"CODNA_WEBHOOK_OPS_TOKEN": TOKEN, "CODNA_WEBHOOK_PORT": "eighty"}, _never, "unreachable: InvalidURL"),  # not an HTTPError
    ({"CODNA_WEBHOOK_OPS_TOKEN": TOKEN}, lambda r: httpx.Response(401, json={"error": "unauthorized"}), "answered HTTP 401"),
    ({"CODNA_WEBHOOK_OPS_TOKEN": TOKEN}, lambda r: httpx.Response(200, text="not json"), "non-JSON body"),
    ({"CODNA_WEBHOOK_OPS_TOKEN": TOKEN}, lambda r: httpx.Response(200, json=[1, 2]), "non-object body"),
])
def test_live_verb_says_why_it_could_not_ask_and_never_the_token(monkeypatch, env, handler, expect):
    monkeypatch.setattr(webhook_ops, "_live_client", _client(handler))
    out, why = live_verb("scaler", {}, actor="t", environ=env)
    assert out is None and expect in why
    assert TOKEN not in why


# --- `codna webhook ops` -------------------------------------------------------------------------------

def _ns(verb, **overrides):
    ns = dict(verb=verb, id=None, priority=None, status=None, kind=None, repo=None, installation=None,
              action=None, value=None, owner=None, reason="operator", limit=50, actor="github-actions:octocat")
    ns.update(overrides)
    return argparse.Namespace(**ns)


class _NeverBuilt:
    def __init__(self, *args, **kwargs):
        raise AssertionError("the live view must not open the queue")


class _StandIn:
    """Stands in for the PostgresQueue the CLI opens; the verbs exercised here never touch it."""

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


def test_cli_ops_scaler_prints_the_live_view_without_a_database(monkeypatch, capsys):
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", TOKEN)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CODNA_WEBHOOK_DATABASE_URL", raising=False)
    monkeypatch.setattr(webhook_ops, "_live_client",
                        _client(lambda r: httpx.Response(200, json={"configured": True, "cap": 4, "paused": False})))
    monkeypatch.setattr(webhook_ops, "PostgresQueue", _NeverBuilt)
    assert cli_ops(_ns("scaler", action="status")) == 0
    assert json.loads(capsys.readouterr().out) == {"cap": 4, "configured": True, "paused": False, "view": "live"}


def test_cli_ops_scaler_falls_back_to_its_own_view_marked_cli_rebuild(monkeypatch, capsys):
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", TOKEN)
    monkeypatch.setenv("DATABASE_URL", "postgresql://stand-in/db")
    monkeypatch.setattr(webhook_ops, "_live_client", _client(_refuse))
    monkeypatch.setattr(webhook_ops, "PostgresQueue", _StandIn)
    assert cli_ops(_ns("scaler")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["view"] == "cli-rebuild" and out["configured"] is False
    assert out["live_unavailable"] == "loopback /ops/scaler unreachable: ConnectError"
    assert "GET /ops/scaler" in out["hint"]
    assert TOKEN not in json.dumps(out)


def _no_loopback():
    raise AssertionError("a table read must not go to loopback")


def test_cli_ops_table_verbs_never_go_to_loopback(monkeypatch, capsys):
    assert LIVE_VERBS == ("scaler",)
    monkeypatch.setenv("CODNA_WEBHOOK_OPS_TOKEN", TOKEN)
    monkeypatch.setenv("DATABASE_URL", "postgresql://stand-in/db")
    monkeypatch.setattr(webhook_ops, "_live_client", _no_loopback)
    monkeypatch.setattr(webhook_ops, "PostgresQueue", _StandIn)
    seen = []
    monkeypatch.setattr(webhook_ops, "run_verb",
                        lambda queue, verb, params, **kw: seen.append((verb, params, kw["actor"])) or {"jobs": []})
    assert cli_ops(_ns("jobs", status="dead")) == 0
    assert seen == [("jobs", {"status": "dead", "reason": "operator", "limit": 50}, "ops-cli:github-actions:octocat")]
    assert json.loads(capsys.readouterr().out) == {"jobs": []}                       # no view marker on a table read
