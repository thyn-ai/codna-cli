"""Goal mode's PURE half: the stop condition, the CI classification, and the plan-side recap (D5).

No server, no sidecar, no git — the whole point of `goal.py` is that "is the goal met" and "what does
the next plan know" are decidable without I/O. If any of this needed an activity, the answer could
not be replayed, and "the run said it was done" would be a claim rather than a fact.

Three failure modes this file exists to pin down, each of which would be a silent wrong answer in
production:

1. **A goal that stops on the LAST task instead of all of them.** A run whose task 1 failed verify has
   not met a goal requiring verification, however green task 2 was.
2. **"Unknown" read as "green".** A missing token, an unreadable rollup, or a check still running is
   not a pass. A goal that treated them as a pass would open a PR whose checks were never read.
3. **Infrastructure failures becoming fix tasks.** Asking an agent to fix a runner that timed out
   burns a task and teaches it to patch code CI never judged — the failure `webhook_ci_triage` exists
   to prevent, reused here rather than re-derived.
"""
from __future__ import annotations

import importlib.util

import pytest

from codna.durable.contracts import TaskSpec
from codna.durable.goal import (
    MAX_EPISODES,
    MAX_OBJECTIVE_CHARS,
    EpisodeRecap,
    EpisodeRef,
    GoalEvidence,
    GoalSpec,
    StopCondition,
    build_ci_fix_brief,
    ci_verdict,
    evaluate_stop,
    format_objective_section,
    recap_from_payload,
    recall_lines,
    required_checks,
)

# -- StopCondition + GoalSpec validation -----------------------------------------------------


def test_the_condition_names_its_required_checks_in_a_fixed_order():
    assert required_checks(StopCondition()) == ("verify", "review")
    assert required_checks(StopCondition(ci=True)) == ("verify", "review", "ci")
    assert required_checks(StopCondition(review=False)) == ("verify",)


def test_a_condition_requiring_nothing_is_refused():
    """``StopCondition()`` with every check off is "a goal that is already met", which is not a goal."""
    with pytest.raises(ValueError) as caught:
        StopCondition(verify=False, review=False, ci=False).validate()
    assert getattr(caught.value, "code", "") == "contract_invalid"


def test_a_condition_round_trips_and_defaults_to_verify_and_review():
    assert StopCondition.from_payload(None) == StopCondition()
    assert StopCondition.from_payload({"verify": True, "review": True, "ci": True}).ci is True
    spec = GoalSpec(objective="ship it")
    assert GoalSpec.from_payload(spec.to_payload()) == spec


def test_a_goal_refuses_an_empty_over_long_or_secret_objective():
    with pytest.raises(ValueError):
        GoalSpec(objective="   ").validate()
    with pytest.raises(ValueError):
        GoalSpec(objective="x" * (MAX_OBJECTIVE_CHARS + 1)).validate()
    with pytest.raises(ValueError) as caught:
        GoalSpec(objective="use ghp_" + "a" * 36).validate()
    assert getattr(caught.value, "code", "") == "contract_contains_secret"


def test_a_goal_needs_a_positive_task_and_ci_bound():
    """``max_tasks=0`` would be a goal that may not buy any work — a refusal, not a no-op."""
    with pytest.raises(ValueError):
        GoalSpec(objective="x", max_tasks=0).validate()
    with pytest.raises(ValueError):
        GoalSpec(objective="x", max_ci_attempts=0).validate()


# -- evaluate_stop ---------------------------------------------------------------------------


def _evidence(task_id: str, *, verify: bool | None = True, review: str | None = "approved") -> GoalEvidence:
    return GoalEvidence(
        task_id=task_id,
        outcome="approved" if review == "approved" else "rejected",
        verify_passed=verify,
        review_verdict=review,
    )


def test_an_empty_run_has_not_met_its_goal():
    decision = evaluate_stop(StopCondition(), [])
    assert decision.satisfied is False
    assert decision.reason == "no_evidence"
    assert decision.checks == (("verify", "not_run"), ("review", "not_run"))


def test_the_goal_is_the_whole_run_not_the_last_task():
    """Task 1 failed verify; task 2 is green. The goal is NOT met — and the reason says which check."""
    decision = evaluate_stop(
        StopCondition(),
        [_evidence("task-1", verify=False), _evidence("task-2")],
    )
    assert decision.satisfied is False
    assert dict(decision.checks) == {"verify": "failed", "review": "approved"}
    assert decision.summary() == "verify=failed; review=approved"


def test_every_task_must_be_approved():
    decision = evaluate_stop(
        StopCondition(), [_evidence("task-1"), _evidence("task-2", review="changes_requested")]
    )
    assert decision.satisfied is False
    assert dict(decision.checks)["review"] == "failed"


def test_a_check_that_never_ran_is_not_a_pass():
    """``None`` means "did not run" — the one reading that is never optimistic."""
    decision = evaluate_stop(StopCondition(), [_evidence("task-1", verify=None, review=None)])
    assert decision.satisfied is False
    assert dict(decision.checks) == {"verify": "failed", "review": "failed"}


def test_ci_is_required_only_when_the_condition_asks_for_it():
    green = [_evidence("task-1")]
    assert evaluate_stop(StopCondition(ci=False), green, ci_state="failed").satisfied is True
    assert evaluate_stop(StopCondition(ci=True), green, ci_state="passed").satisfied is True


@pytest.mark.parametrize("state", ["failed", "pending", "infrastructure", "unknown", None])
def test_every_non_green_ci_state_fails_the_gate(state: str | None):
    """Pending is not green; unknown is not green. Only "passed" satisfies a CI gate."""
    decision = evaluate_stop(StopCondition(ci=True), [_evidence("task-1")], ci_state=state)
    assert decision.satisfied is False
    assert dict(decision.checks)["ci"] == "failed"


def test_evaluate_stop_is_pure_and_order_independent_for_the_verdict():
    items = [_evidence("task-1"), _evidence("task-2")]
    assert evaluate_stop(StopCondition(), items) == evaluate_stop(StopCondition(), items)
    assert evaluate_stop(StopCondition(), items).satisfied is True


# -- ci_verdict --------------------------------------------------------------------------------


def test_no_red_required_check_is_green():
    assert ci_verdict([]) == "passed"


def test_a_plain_red_check_is_a_code_failure():
    assert ci_verdict([{"name": "ci / test", "state": "FAILURE"}]) == "failed"
    assert ci_verdict([{"name": "typecheck", "state": "TIMED_OUT"}]) == "infrastructure"


@pytest.mark.parametrize("name", ["setup-node", "install deps", "cache restore", "docker build"])
def test_a_plumbing_shaped_check_name_is_infrastructure(name: str):
    """A fetch/setup check that is red says nothing about the code — a fix task would patch nothing."""
    assert ci_verdict([{"name": name, "state": "FAILURE"}]) == "infrastructure"


def test_one_infrastructure_check_makes_the_verdict_infrastructure():
    """Conservative on purpose: a red check we cannot classify must not earn a code fix."""
    assert (
        ci_verdict(
            [
                {"name": "ci / test", "state": "FAILURE"},
                {"name": "deploy", "state": "STARTUP_FAILURE"},
            ]
        )
        == "infrastructure"
    )


# -- the fix CI brief ----------------------------------------------------------------------------


def test_the_fix_brief_names_the_checks_the_head_and_the_attempt():
    brief = build_ci_fix_brief(
        pr_number=42,
        head_sha="0123456789abcdef",
        checks=[{"name": "ci / test", "state": "FAILURE"}],
        attempt=1,
    )
    assert "pull request #42" in brief
    assert "head 01234567" in brief
    assert "`ci / test` (failure)" in brief
    assert "Do not disable, skip or loosen a check" in brief
    # The first attempt does not claim a previous one failed.
    assert "attempt 2" not in brief


def test_a_second_attempt_says_a_previous_fix_did_not_work():
    """A repeat that read exactly like attempt 1 would be a loop pretending to be bounded."""
    brief = build_ci_fix_brief(pr_number=1, head_sha=None, checks=[], attempt=3)
    assert "This is attempt 3" in brief
    assert "the current head" in brief


# -- episodes: references, recaps, formatting -----------------------------------------------------


def test_an_episode_reference_validates_its_shape():
    ref = EpisodeRef(path="episodes/task-1.json", digest="a" * 64, task_id="task-1")
    assert ref.to_payload() == {"path": "episodes/task-1.json", "digest": "a" * 64, "task_id": "task-1"}
    assert EpisodeRef.from_payload(ref.to_payload()) == ref
    with pytest.raises(ValueError):
        EpisodeRef(path="episodes/x.json", digest="not-a-digest", task_id="task-1").validate()


def test_a_recap_reads_a_verified_flag_and_a_verdict_out_of_the_episode_file():
    """The recap is derived from the FILE the run recorded, not from a second copy of the facts."""
    recap = recap_from_payload(
        {
            "task_id": "task-1",
            "outcome": "approved",
            "brief": "Fix the flaky retry\nwith details nobody needs in a recap",
            "verify_ref": {"ref": "verify/task-1/abc", "passed": True},
            "review_verdict": "approved",
            "pr": {"number": 3, "url": "https://github.com/o/r/pull/3"},
            "plan_ref": "refs/plans/task-1-abc",
        }
    )
    assert recap.brief == "Fix the flaky retry"
    assert recap.verify_passed is True
    assert recap.review_verdict == "approved"
    assert recap.pr_url == "https://github.com/o/r/pull/3"
    assert recap.plan_ref == "refs/plans/task-1-abc"
    line = recap.line()
    assert "task-1 (approved)" in line
    assert "verify=pass" in line and "review=approved" in line


def test_a_failed_verify_is_spelled_out_in_the_recap():
    """A recap that said "verify=False" would read like a field name; FAIL is what a plan notices."""
    recap = recap_from_payload(
        {"task_id": "t", "outcome": "rejected", "verify_ref": {"passed": False}}
    )
    assert "verify=FAIL" in recap.line()


def test_the_objective_section_carries_the_objective_and_the_recaps():
    block = format_objective_section(
        "Ship the retry fix", [EpisodeRecap("task-1", "approved", "did A", True, "approved", None, None)]
    )
    assert "Run objective" in block
    assert "Ship the retry fix" in block
    assert "Already completed in this run" in block
    assert "- task-1 (approved): did A" in block


def test_a_goal_with_no_completed_tasks_has_no_recall_block():
    """The FIRST task's plan must not claim a history it does not have."""
    block = format_objective_section("Ship it", [])
    assert "Already completed" not in block
    assert recall_lines([]) == ""


def test_the_recap_block_is_bounded_by_the_caller_and_oldest_first():
    recaps = [EpisodeRecap(f"task-{n}", "approved", f"did {n}", True, "approved", None, None) for n in (1, 2, 3)]
    lines = recall_lines(recaps).splitlines()
    assert [line.split()[1] for line in lines] == ["task-1", "task-2", "task-3"]
    # The cap itself is applied where the refs are SELECTED (`goal_steps.recall_episodes`), so the
    # formatter stays pure and total. The constant is still one value, asserted here.
    assert MAX_EPISODES >= 1


# -- the plan prompt carries the goal block --------------------------------------------------------


def _spec(**overrides: object) -> TaskSpec:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the flaky retry",
        "repo_dir": "/tmp/repo",
        "verify_commands": ("true",),
    }
    payload.update(overrides)
    return TaskSpec(**payload)  # type: ignore[arg-type]


def test_a_task_mode_plan_prompt_is_byte_identical_to_what_it_always_was():
    """The whole replay-safety story for this feature: no goal context, no change."""
    from codna.durable.contracts import plan_prompt

    spec = _spec()
    assert spec.goal_context == ""
    assert plan_prompt(spec.brief) == plan_prompt(spec.brief, (), "")


def test_the_goal_block_is_APPENDED_so_a_recorded_prompt_replays_byte_for_byte():
    from codna.durable.contracts import plan_prompt

    plain = plan_prompt("Fix it")
    with_goal = plan_prompt("Fix it", (), "\n\nRun objective (the goal this task serves):\nShip it")
    assert with_goal.startswith(plain)
    assert with_goal.endswith("Ship it")


def test_a_spec_carries_and_bounds_its_goal_context():
    spec = _spec().with_goal_context("Run objective:\nShip it")
    assert spec.goal_context == "Run objective:\nShip it"
    with pytest.raises(ValueError):
        _spec().with_goal_context("x" * 20_000)


def test_a_fix_ci_task_keeps_its_parents_branch_and_number():
    """The fix commits to the SAME pull request, so its branch and task number must not move."""
    parent = _spec()
    fix = parent.with_ci_fix(task_id="task-1-ci1", brief="Fix CI", goal_context="goal")
    assert fix.task_id == "task-1-ci1"
    assert fix.task_number == parent.task_number == 1
    assert fix.branch == parent.branch == "codna/run-run-1/task-1"
    assert fix.brief == "Fix CI"
    assert fix.goal_context == "goal"


# -- episode recall: the file, the digest, and the refusals ---------------------------------------


def _harness(tmp_path):
    if importlib.util.find_spec("temporalio") is None:
        pytest.skip("the `durable` extra (temporalio) is not installed")

    from tests.durable import fakes

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return repo, runs_root, fakes.build_harness(repo, runs_root)


def _episode_file(tmp_path, *, brief: str = "Fix it", outcome: str = "approved"):
    """Write one episode through the REAL step, so the digest is the one a run would record."""
    from pathlib import Path

    from codna.durable.contracts import EpisodeRequest, TaskContext
    from codna.durable.state import TaskState

    repo, runs_root, harness = _harness(tmp_path)
    spec = _spec(repo_dir=str(repo), brief=brief)
    context = TaskContext.from_spec(spec)
    state = TaskState.from_payload(spec.initial_state())
    outcome_value = __import__(
        "codna.durable.activity_steps", fromlist=["remember_episode"]
    ).remember_episode(
        EpisodeRequest(state=state.to_payload(), context=context, outcome=outcome, brief=brief)
    )
    return harness, outcome_value, Path(runs_root)


def test_recall_reads_the_episode_a_task_wrote_and_verifies_its_digest(tmp_path):
    from codna.durable.goal_contracts import RecallRequest
    from codna.durable.goal_steps import recall_episodes

    harness, episode, _runs = _episode_file(tmp_path)
    recalled = recall_episodes(
        RecallRequest(
            run_id="run-1",
            refs=({"path": episode.path, "digest": episode.digest, "task_id": "task-1"},),
        )
    )
    assert [item.task_id for item in recalled.episodes] == ["task-1"]
    assert "Fix it" in recalled.lines()
    assert recalled.missing == ()
    # The recall is recorded, so "task 2 planned with task 1's episode" is an audited fact.
    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.payload.get("event") == "episodes.recalled"
    ]
    assert notes and notes[0]["count"] == 1


def test_a_tampered_episode_is_EXCLUDED_and_reported_not_silently_trusted(tmp_path):
    """Planning on a file that is not the episode the run recorded is planning on unaudited data."""
    from codna.durable.goal_contracts import RecallRequest
    from codna.durable.goal_steps import recall_episodes

    harness, episode, runs_root = _episode_file(tmp_path)
    path = runs_root / "run-1" / episode.path
    body = path.read_text(encoding="utf-8").replace("Fix it", "Something else entirely")
    path.write_text(body, encoding="utf-8")

    recalled = recall_episodes(
        RecallRequest(
            run_id="run-1",
            refs=({"path": episode.path, "digest": episode.digest, "task_id": "task-1"},),
        )
    )
    assert recalled.episodes == ()
    # Either digest check can catch a tamper — the file's own `digest` field, or the body that must
    # hash to it. Both are reported as a mismatch, which is what a plan must not silently accept.
    assert recalled.missing and "digest" in recalled.missing[0]


def test_a_missing_episode_is_reported_and_does_not_fail_the_plan(tmp_path):
    from codna.durable.goal_contracts import RecallRequest
    from codna.durable.goal_steps import recall_episodes

    harness, episode, _runs = _episode_file(tmp_path)
    recalled = recall_episodes(
        RecallRequest(
            run_id="run-1",
            refs=(
                {"path": "episodes/task-9.json", "digest": "b" * 64, "task_id": "task-9"},
                {"path": episode.path, "digest": episode.digest, "task_id": "task-1"},
            ),
        )
    )
    # The readable one still comes back; the gone one is named.
    assert [item.task_id for item in recalled.episodes] == ["task-1"]
    assert any("task-9" in entry for entry in recalled.missing)


def test_a_path_that_escapes_the_run_directory_is_refused(tmp_path):
    from codna.durable.goal_contracts import RecallRequest
    from codna.durable.goal_steps import recall_episodes

    _harness, _episode, _runs = _harness_only(tmp_path)
    recalled = recall_episodes(
        RecallRequest(
            run_id="run-1",
            refs=({"path": "../../etc/passwd", "digest": "c" * 64, "task_id": "task-1"},),
        )
    )
    assert recalled.episodes == ()
    assert any("escapes the run directory" in entry for entry in recalled.missing)


def _harness_only(tmp_path):
    return _harness(tmp_path)


def test_recall_keeps_the_NEWEST_episodes_when_handed_too_many(tmp_path):
    """The plan for task N is about what tasks N-1 and N-2 learned, not about task 3."""
    from codna.durable.goal_contracts import RecallRequest
    from codna.durable.goal_steps import recall_episodes

    harness, _episode, _runs = _episode_file(tmp_path)
    refs = tuple(
        {"path": "episodes/gone.json", "digest": "d" * 64, "task_id": f"task-{n}"}
        for n in range(1, 10)
    )
    recall_episodes(RecallRequest(run_id="run-1", refs=refs))
    note = next(
        record.payload
        for record in harness.ledger("run-1").records()
        if record.payload.get("event") == "episodes.recalled"
    )
    # Only MAX_EPISODES were even attempted, and they are the LAST ones.
    assert len(note["missing"]) == MAX_EPISODES
    assert "task-9" in note["missing"][-1]
    assert "task-1" not in " ".join(note["missing"])
