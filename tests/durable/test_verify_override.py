"""Verification is never an unexplained `undetectable`, and a user's command is honoured.

── THE MEASURED DEFECT ────────────────────────────────────────────────────────────────────────────

Run `run-20261112T230312Z` failed with:

    ActivityError: no verification command was supplied and none could be detected for this repository

on a repository whose tests were a working `node --test` suite. Two separate gaps produced that
outcome, and both are pinned here:

1. **Detection** excluded JavaScript runners by design, so the common case was never recognised
   (`codna.testrun.detect_test_command`). Fixed there; `cli/tests/test_testrun.py` covers it.
2. **The failure said nothing a user could act on.** It named no signal it had looked for and offered
   no command, so the only way forward was a hand-written `codna.yaml` — which a user driving the
   product from the New task sheet cannot do. This file pins the actionable failure and the
   user-supplied command that now reaches verify from the sheet.

The user-supplied path already existed below the surface (`POST /runs` accepts `verify_commands`,
`codna run --verify`), and `verify_task` preferred it. What is asserted here is that this precedence
holds at the step boundary AND that the empty case now explains itself.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_steps import (  # noqa: E402
    _verify_plan,
    checkpoint_turn,
    run_cline_turn,
    verify_task,
)
from codna.durable.activity_support import ActivityError  # noqa: E402
from codna.durable.contracts import (  # noqa: E402
    CheckpointRequest,
    TaskContext,
    TaskSpec,
    VerifyRequest,
)
from tests.durable import fakes  # noqa: E402


def _spec(repo, *, verify_commands=()) -> TaskSpec:
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="build a calculator",
        repo_dir=str(repo),
        verify_commands=verify_commands,
        engine="plain",
    )
    spec.validate()
    return spec


def _run_to_checkpoint(harness: fakes.Harness, spec: TaskSpec):
    """One real turn + checkpoint, so the verify step runs against a tree the turn actually edited."""
    context = TaskContext.from_spec(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    return context, checkpoint_turn(CheckpointRequest(state=moved.state, context=context))


def _plan_commands(harness: fakes.Harness, request: VerifyRequest, *, tree: str) -> tuple[str, ...]:
    """The commands `_verify_plan` DECIDES for this request, with the test seam unbound.

    `_verify_plan` answers ``("injected",)`` whenever a `run_tests` seam is bound — that seam is how the
    durable suite avoids a subprocess, and it deliberately bypasses the decision. The decision is what
    these tests are about, so it is exercised with the seam unbound and read directly.
    """
    harness.runtime.run_tests = None
    return _verify_plan(request, harness.runtime, tree=tree).commands


def test_a_supplied_command_is_what_verify_runs(tmp_path):
    """The user's word wins over detection, and the sandbox sees exactly that command.

    This is the path the sheet's Verify command field reaches. A detect-first step would run something
    else and report it as the user's verification.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    # A repository detection WOULD recognise, so "the supplied command won" is a real assertion.
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "calc.test.js").write_text("// t\n", encoding="utf-8")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo, verify_commands=("node --test tests/calc.test.js",))
    context, checkpoint = _run_to_checkpoint(harness, spec)

    request = VerifyRequest(state=checkpoint.state, context=context, commands=spec.verify_commands)
    assert _plan_commands(harness, request, tree=str(repo)) == ("node --test tests/calc.test.js",)

    # And the step runs that plan: with the seam bound, verify reports the one command it was given.
    harness.runtime.run_tests = harness.run_tests
    outcome = verify_task(request)
    assert outcome.details["passed"] is True
    assert harness.verify_calls == 1


def test_an_undetectable_repo_reports_what_was_checked_and_what_could_be_typed(tmp_path):
    """The failure a user can act on: the signals tried, and candidate commands to confirm.

    Before this, the message was one sentence naming neither, which is why the measured run's only
    route forward was editing a file the sheet cannot edit.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "package.json").write_text('{"scripts": {"test": "jest --ci"}}\n', encoding="utf-8")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo)
    context, checkpoint = _run_to_checkpoint(harness, spec)

    harness.runtime.run_tests = None
    with pytest.raises(ActivityError) as caught:
        verify_task(VerifyRequest(state=checkpoint.state, context=context, commands=()))

    error = caught.value
    assert error.code == "activity_verify_undetectable", error.as_dict()
    # WHAT it looked for, so the verdict is not opaque.
    assert error.details["checked"], error.as_dict()
    assert any("node --test" in entry for entry in error.details["checked"])
    # WHAT the repository declares, offered for a person to CONFIRM — never applied on its own.
    assert error.details["suggested"] == ["jest --ci"], error.as_dict()
    # And the message names both product routes, because the sheet is where a user actually is.
    assert "Verify command" in str(error)
    assert "--verify" in str(error)
    assert "fix.test_command" in str(error)
    assert "jest --ci" in str(error)


def test_a_node_repo_verifies_with_no_configuration_at_all(tmp_path, monkeypatch):
    """The common case: a plain JS repo with test files needs no codna.yaml and no override.

    This is the measured repository's shape — and the reason the failure was a product defect rather
    than a user error.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "calc.test.js").write_text("import test from 'node:test';\n", encoding="utf-8")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo)
    context, checkpoint = _run_to_checkpoint(harness, spec)

    request = VerifyRequest(state=checkpoint.state, context=context, commands=())
    # No codna.yaml, no override: the detected plan is the repository's own test file.
    assert _plan_commands(harness, request, tree=str(repo)) == (
        "node --test tests/calc.test.js",
    )

    # End to end through the step, with the sandbox scripted so no subprocess runs: the plan decide the
    # actual command, and verify reports it as having run.
    harness.runtime.run_tests = None
    executed: list[tuple[str, ...]] = []

    def fake_sandbox(_request: VerifyRequest, plan: Any, *, tree: str) -> Any:
        executed.append(tuple(plan.commands))
        return fakes.FakeVerify(passed=True, returncode=0)

    monkeypatch.setattr("codna.durable.activity_steps._run_in_sandbox", fake_sandbox)
    outcome = verify_task(request)

    assert outcome.details["passed"] is True
    assert executed == [("node --test tests/calc.test.js",)], executed
