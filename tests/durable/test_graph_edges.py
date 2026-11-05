"""The graph's conditional edges must stay async — a static guard for spike finding 5.

WHY THIS IS A STATIC TEST AND NOT AN INTEGRATION ONE. LangGraph runs a *synchronous* conditional
edge through ``loop.run_in_executor``, and Temporal's deterministic event loop does not implement
``run_in_executor``. The failure is a bare ``NotImplementedError`` raised inside the branch, so it
only fires once a graph actually takes that branch — i.e. it would surface in D2.3 as a workflow-task
failure with no obvious cause, and only under the `durable` extra with a live test server.

`graph.py` records the finding in a docstring next to `_route`. A docstring does not fail a build.
Someone "simplifying" `async def decide` into `def decide` — or passing a raw router to
`add_conditional_edges` instead of through `_route` — would reintroduce the bug and every cheap test
would stay green. So the rule is asserted structurally instead:

  1. every `add_conditional_edges(...)` passes its router through `_route(...)`, and
  2. the function `_route` returns is declared `async`.

Parsing the SOURCE rather than importing the module keeps this test in the base suite: `build_task_graph`
imports `langgraph`, and `codna[durable]` is an opt-in extra that the default install does not carry.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[2]
GRAPH_MODULE = CLI_DIR / "codna" / "durable" / "graph.py"


@pytest.fixture(scope="module")
def graph_tree() -> ast.Module:
    return ast.parse(GRAPH_MODULE.read_text(encoding="utf-8"), filename=str(GRAPH_MODULE))


def _conditional_edge_calls(tree: ast.Module) -> list[ast.Call]:
    """Every `<something>.add_conditional_edges(origin, router, mapping)` in the module."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_conditional_edges":
            calls.append(node)
    return calls


def _route_returned_definitions(tree: ast.Module) -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    """The `decide` functions defined inside `_route`, whichever way they are declared."""
    found: list[ast.AsyncFunctionDef | ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_route":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and inner is not node:
                found.append(inner)
    return found


def test_the_module_really_has_conditional_edges(graph_tree: ast.Module):
    """The guard is tested: a rename that emptied this list would make the assertions vacuous."""
    calls = _conditional_edge_calls(graph_tree)
    assert len(calls) >= 3, [ast.unparse(call.func) for call in calls]


def test_every_conditional_edge_router_goes_through_the_async_adapter(graph_tree: ast.Module):
    """A raw router passed straight to `add_conditional_edges` is a sync edge, which is the bug."""
    for call in _conditional_edge_calls(graph_tree):
        assert len(call.args) >= 2, ast.unparse(call)
        router = call.args[1]
        assert isinstance(router, ast.Call) and isinstance(router.func, ast.Name) and router.func.id == "_route", (
            "conditional edge router must be wrapped by `_route(...)`, got "
            f"{ast.unparse(router)} — a synchronous conditional edge dies with NotImplementedError "
            "inside Temporal's event loop (spike finding 5)"
        )


def test_the_router_adapter_is_async(graph_tree: ast.Module):
    """`_route` must RETURN an `async def`: that is the whole point of the adapter."""
    definitions = _route_returned_definitions(graph_tree)
    assert definitions, "`_route` defines no inner function — the shape of the adapter changed"
    for definition in definitions:
        assert isinstance(definition, ast.AsyncFunctionDef), (
            f"`_route` returns a synchronous `def {definition.name}`; LangGraph would run it through "
            "`loop.run_in_executor`, which Temporal's event loop does not implement (spike finding 5)"
        )


def test_the_routers_themselves_stay_pure_and_sync():
    """The adapter is async; the policy is not. Routers must not become coroutines.

    Keeping the routing decision pure and synchronous is what lets it be unit-tested by calling it,
    and it is where the plan puts the branch logic. If a router grew into an `async def`, the adapter
    would have to `await` it inside the workflow, and a router that awaits is a router that can do
    I/O mid-replay.
    """
    routing_source = (CLI_DIR / "codna" / "durable" / "routing.py").read_text(encoding="utf-8")
    tree = ast.parse(routing_source, filename="routing.py")
    routers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("after_")]
    assert {router.name for router in routers} == {"after_plan", "after_verify", "after_review"}
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            raise AssertionError(f"`{node.name}` in routing.py is async; routers must stay pure and sync")
