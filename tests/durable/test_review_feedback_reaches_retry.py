"""A rejected review must FEED ITS FINDINGS into the next implement attempt.

THE DEFECT THESE TESTS COVER, measured on the installed build `6f7a904d` (`run-20261116T005743Z`):

    verify passed      true, rc 0, three times (13 tests OK, independently re-confirmed)
    artifact           correct and complete, with full browser evidence
    review             REJECTED three times on an UNCHANGED SHA 8cce0fa1
    then               parked at `approval.requested` (`review_not_approved`), unable to complete
    the 3rd and 4th implement prompts in the ledger are BYTE-IDENTICAL

The mechanism was in the code, and each half is asserted here rather than described:

* `routing.after_review` returns ``implement`` on a rejection while retries remain — CORRECT, and
  unchanged by this fix (a rejection must still reject);
* `prompts.implement_prompt` took a brief and human STEER only — there was no parameter through
  which a finding could reach the retry;
* `review_task` recorded the review's verdict and a findings COUNT, and dropped the findings
  themselves, so `TaskState` had nothing to carry;
* therefore the retry was asked to "try again" with no new information, and a `changes_requested`
  loop could never converge. Parking was honest, but the work was unrecoverable without a human.

What is asserted, in the order it matters:

1. a rejected review followed by an implement retry produces a prompt CONTAINING the findings;
2. an APPROVED review emits no feedback block (and CLEARS one from an earlier rejection);
3. the bound holds for a large findings list, and the number omitted is STATED rather than dropped;
4. a replay of the retry produces a byte-identical prompt;
5. the review gate is NOT weakened — the rejection still routes back to `implement`.

The whole file is skipped without the `durable` extra. The module-level ``pytest.importorskip`` is
not optional: a module-level import of `temporalio` raises at COLLECTION time, and that aborts the
entire offline CI job rather than skipping one file (it broke CI twice in the session that produced
this fix).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from datetime import timedelta  # noqa: E402

from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from codna.durable import prompts as prompts_module  # noqa: E402
from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: E402
from codna.durable.contracts import (  # noqa: E402
    ContractError,
    TaskState,
    implement_prompt,
    implement_prompt_for_state,
)
from codna.durable.graph import build_task_graph  # noqa: E402
from codna.durable.review_feedback import (  # noqa: E402
    MAX_FINDINGS_RECORDED,
    REVIEW_SEVERITIES,
    ReviewFeedback,
    review_findings_from_report,
)
from codna.durable.workflows import (  # noqa: E402
    TaskWorkflow,
    TaskWorkflowPlain,
    registered_workflows,
)
from tests.durable import fakes  # noqa: E402

TASK_QUEUE = "codna-durable-workflow-tests"
RUN_TIMEOUT = timedelta(hours=24)
RESULT_TIMEOUT_S = 60.0

# The two findings the fake reviewer returns, with the fields a prompt has to be able to render. The
# values are chosen to be IMPOSSIBLE to trip over by accident: a prompt containing them cannot have
# produced them from the brief, which is what makes "the prompt carries the findings" a real claim.
REJECTING_FINDINGS: list[dict[str, Any]] = [
    {
        "path": "cli/codna/durable/routing.py",
        "line": 101,
        "severity": "medium",
        "category": "correctness",
        "title": "The rejection path does not carry the reviewer's findings",
        "explanation": "after_review returns 'implement' but nothing tells the retry what to fix.",
        "fingerprint": "0123456789abcdef",
    },
    {
        "path": "cli/codna/exit_codes.py",
        "line": 42,
        "severity": "low",
        "category": "correctness",
        "title": "An unhandled exit code is reported as success",
        "explanation": "The caller reads only 0 and 1, so 2 reads as success.",
        "fingerprint": "fedcba9876543210",
    },
]


def _feedback_payload(**overrides: Any) -> dict[str, Any]:
    """A valid `review_feedback` payload, for the pure prompt tests."""
    payload: dict[str, Any] = {
        "verdict": "changes_requested",
        "sha": "8cce0fa15bab741add8a6fd1ae8eec9a343b40e9",
        "omitted": 0,
        "findings": [dict(item) for item in REJECTING_FINDINGS],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------------------
# 1. The mechanism, asserted at the seam where it was missing: the RETRY's prompt.
# ---------------------------------------------------------------------------------------


def test_a_retry_after_a_rejection_carries_the_findings_into_the_prompt():
    """THE REGRESSION TEST. Before the fix this prompt was byte-identical to the rejected attempt's.

    Asserted from the RENDERED TEXT, not from a field: the defect was that the text was the same,
    so a test that stopped at `state.review_feedback is not None` would pass on a fix that recorded
    the findings and still never showed them to the model.
    """
    prompt = implement_prompt_for_state(
        {"review_feedback": _feedback_payload()}, "Fix the flaky retry in the client"
    )

    # The block announces itself as a RETRY for a specific attempt.
    assert prompts_module._REVIEW_FEEDBACK_HEADING in prompt
    assert "8cce0fa15bab" in prompt  # the rejected SHA, so it is clear WHICH attempt is meant

    # ...and every finding is RENDERED: path, line, severity, and the actionable statement.
    for finding in REJECTING_FINDINGS:
        assert finding["path"] in prompt, finding["path"]
        assert f":{finding['line']}" in prompt, finding
        assert f"[{finding['severity']}/" in prompt, finding
        assert finding["title"] in prompt, finding["title"]
    # The order is severity-first, because a bounded list must keep the most urgent finding.
    assert prompt.index("routing.py") < prompt.index("exit_codes.py")

    # It is NOT a JSON dump: the raw finding schema would be a second, unstated contract.
    assert "{\"path\"" not in prompt
    assert "\"fingerprint\"" not in prompt


def test_the_same_state_renders_the_same_prompt_twice():
    """Determinism at the prompt: no clock, no I/O, no dict-ordering dependence."""
    payload = _feedback_payload()
    assert implement_prompt_for_state({"review_feedback": payload}, "brief") == (
        implement_prompt_for_state({"review_feedback": payload}, "brief")
    )


def test_a_state_without_feedback_renders_the_prompt_it_always_did():
    """Replay safety for every run recorded before this field existed: no field, no change."""
    plain = implement_prompt("Fix the flaky retry in the client", ("note",))
    assert implement_prompt_for_state(
        {"review_feedback": None}, "Fix the flaky retry in the client", ("note",)
    ) == plain
    # A state payload that predates the field entirely (the key is absent rather than None).
    assert implement_prompt_for_state({}, "Fix the flaky retry in the client", ("note",)) == plain


def test_the_findings_are_appended_so_the_rest_of_the_prompt_is_unchanged():
    """The append-only rule the module docstring promises, asserted rather than assumed."""
    base = implement_prompt("Fix it", ("steer note",))
    with_feedback = implement_prompt_for_state(
        {"review_feedback": _feedback_payload()}, "Fix it", ("steer note",)
    )
    assert with_feedback.startswith(base)
    # The block is at the END: everything before it is the prompt a first attempt would have received.
    assert with_feedback[len(base):].lstrip().startswith(prompts_module._REVIEW_FEEDBACK_HEADING)


# ---------------------------------------------------------------------------------------
# 2. An APPROVED review is unchanged, and CLEARS an earlier rejection.
# ---------------------------------------------------------------------------------------


def test_an_approved_review_produces_no_feedback_record():
    """An approval is not a finding. Recording one would emit a rejection block for a passing review."""
    report = fakes.FakeReviewReport(conclusion="success", findings=[dict(REJECTING_FINDINGS[0])])
    assert review_findings_from_report(
        report, verdict="approved", sha="a" * 40, fail=ContractError
    ) is None


def test_the_review_activity_clears_feedback_when_the_retry_passes(tmp_path: Path):
    """The whole loop, on the REAL activity: reject with findings, retry, then approve and CLEAR.

    Driven directly rather than through a workflow because that is where the state transition is
    decided, and because the assertion is about `TaskState` — which is what a replayed workflow
    rebuilds. A stale record surviving an approval would put a rejection block in front of a model
    whose work had just been accepted.
    """
    from codna.durable.activities import review_task
    from codna.durable.contracts import TaskContext
    from tests.durable.workflow_specs import task_spec

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    # First call rejects WITH findings, second approves (and, per the test above, carries none).
    harness.review_conclusions = ["neutral", "success"]
    harness.review_findings_script = [[dict(item) for item in REJECTING_FINDINGS], []]
    spec = task_spec(repo, engine="plain")
    context = TaskContext.from_spec(spec)

    model = TaskState(run_id="run-1", task_id="task-1")
    rejected = review_task(_review_request(context, model, sha="b" * 40))
    rejected_state = TaskState.from_payload(rejected.state)

    assert rejected_state.review_verdict == "changes_requested"
    assert rejected_state.review_feedback is not None
    assert len(rejected_state.review_feedback.findings) == len(REJECTING_FINDINGS)
    # The findings are IN THE LEDGER, so a replay reconstructs the same record (and the same prompt).
    ledger_reviews = [record for record in harness.ledger().records() if record.kind == "review"]
    assert ledger_reviews[-1].payload["feedback"]["findings"][0]["path"] == REJECTING_FINDINGS[0]["path"]
    assert ledger_reviews[-1].payload["feedback_digest"] == rejected_state.review_feedback.digest

    approved = review_task(_review_request(context, rejected_state, sha="c" * 40))
    approved_state = TaskState.from_payload(approved.state)
    assert approved_state.review_verdict == "approved"
    assert approved_state.review_feedback is None, "an approval must CLEAR an earlier rejection"


def _review_request(context: Any, state: TaskState, *, sha: str) -> Any:
    """A `ReviewRequest` whose checkpoint is `sha`, for calling the activity directly."""
    from dataclasses import replace as dataclass_replace

    from codna.durable.contracts import ReviewRequest
    from codna.durable.state import CheckpointRef

    checked = dataclass_replace(
        state,
        checkpoint_ref=CheckpointRef(ref="refs/codna/runs/run-1/turn-1", sha=sha),
    )
    return ReviewRequest(state=checked.to_payload(), context=context, base_sha="a" * 40)


# ---------------------------------------------------------------------------------------
# 3. The bound: a large findings list is truncated, and the omission is STATED.
# ---------------------------------------------------------------------------------------


def _many_findings(count: int) -> list[dict[str, Any]]:
    """A reviewer that returned `count` findings, with severities cycling so the sort does real work."""
    severities = ("low", "medium", "high")
    return [
        {
            "path": f"src/module_{index:03d}.py",
            "line": index + 1,
            # Cycling severities means the kept set is not simply the input prefix (which is what a
            # naive `[:limit]` would keep), so a severity-ordered truncation is observable.
            "severity": severities[index % len(severities)],
            "category": "correctness",
            "title": f"Finding number {index}",
            "explanation": "x" * 50,
        }
        for index in range(count)
    ]


def test_a_large_findings_list_is_truncated_and_says_how_many_were_omitted():
    """The prompt must stay bounded WITHOUT pretending the truncated list is complete."""
    total = MAX_FINDINGS_RECORDED * 4
    report = fakes.FakeReviewReport(conclusion="neutral", findings=_many_findings(total))
    feedback = review_findings_from_report(
        report, verdict="changes_requested", sha="d" * 40, fail=ContractError
    )
    assert feedback is not None
    assert len(feedback.findings) == MAX_FINDINGS_RECORDED
    assert feedback.omitted == total - MAX_FINDINGS_RECORDED

    prompt = implement_prompt_for_state({"review_feedback": feedback.to_payload()}, "Fix it")
    # The omission is STATED, with the number, rather than the list simply ending.
    assert f"{feedback.omitted} further finding(s) were omitted" in prompt
    # ...and the bound is real: the block cannot grow with the reviewer's output.
    assert prompt.count("Finding number") == MAX_FINDINGS_RECORDED
    # What survived is the most SEVERE set, not an arbitrary prefix of the input.
    assert "[high/" in prompt
    assert len(prompt) < 20_000, "a bounded record must produce a bounded prompt"


def test_the_record_itself_refuses_more_findings_than_it_may_carry():
    """The bound is enforced at the DECODE boundary, so a hand-written payload cannot exceed it.

    Distinct from the truncation test above, and both matter: the BUILDER truncates (a reviewer that
    returns a thousand findings must not fail the run), while the DECODER refuses (a state payload
    claiming to carry more than the record allows is malformed, and coercing it would mean a
    hand-edited history could put an unbounded block into the next prompt).
    """
    from codna.durable.state import TaskStateError

    oversize = _feedback_payload(
        findings=[dict(item) for item in _many_findings(MAX_FINDINGS_RECORDED + 1)]
    )
    with pytest.raises(TaskStateError) as excinfo:
        ReviewFeedback.from_payload(oversize, fail=TaskStateError)
    assert excinfo.value.code == "invalid_review_feedback"
    assert excinfo.value.details["limit"] == MAX_FINDINGS_RECORDED


def test_a_rejection_whose_findings_cannot_be_anchored_still_tells_the_retry_something():
    """A rejection with nothing renderable must not produce a BLIND retry — that is the defect."""
    feedback = review_findings_from_report(
        fakes.FakeReviewReport(conclusion="failure", findings=[{"severity": "high"}]),
        verdict="rejected",
        sha="f" * 40,
        fail=ContractError,
    )
    assert feedback is not None
    assert len(feedback.findings) == 1
    assert feedback.findings[0].severity in REVIEW_SEVERITIES
    assert "without a finding it could anchor" in feedback.findings[0].title


# ---------------------------------------------------------------------------------------
# 4. Replay: the retry's prompt is a pure function of the STATE, which is what history rebuilds.
# ---------------------------------------------------------------------------------------


def test_a_replay_of_the_retry_renders_a_byte_identical_prompt():
    """The property a recorded history depends on: decoding the payload back yields the same text.

    Round-trips the payload through JSON (what Temporal stores) before rendering the second time, so
    an ordering or type dependence in the decoder would show up as a different prompt.
    """
    payload = _feedback_payload()
    stored = json.loads(json.dumps(payload, sort_keys=True))
    first = implement_prompt_for_state({"review_feedback": payload}, "Fix it", ("note",))
    second = implement_prompt_for_state({"review_feedback": stored}, "Fix it", ("note",))
    assert first == second
    # ...and a state that round-trips through TaskState (the workflow's own encode/decode) agrees too.
    state = TaskState(
        run_id="run-1",
        task_id="task-1",
        review_verdict="changes_requested",
        review_feedback=ReviewFeedback.from_payload(payload, fail=ContractError),
    )
    decoded = TaskState.from_payload(json.loads(json.dumps(state.to_payload())))
    assert implement_prompt_for_state(decoded.to_payload(), "Fix it", ("note",)) == first


def test_a_malformed_feedback_record_is_a_NAMED_contract_failure():
    """A half-decoded record must fail loudly, not render a prompt from whatever parsed."""
    with pytest.raises(ContractError) as excinfo:
        implement_prompt_for_state({"review_feedback": {"verdict": "approved"}}, "Fix it")
    assert excinfo.value.code == "invalid_review_feedback"


def test_the_record_rejects_an_approved_verdict_outright():
    """`approved` is not a reason to change anything, so it is refused rather than rendered."""
    with pytest.raises(ContractError) as excinfo:
        ReviewFeedback.from_payload(_feedback_payload(verdict="approved"), fail=ContractError)
    assert "REJECTED review" in str(excinfo.value)


def test_the_severity_vocabulary_matches_the_review_pipeline():
    """The tuple is duplicated to keep this module a leaf; the copies must not drift."""
    from codna.review_findings import SEVERITIES

    assert set(REVIEW_SEVERITIES) == set(SEVERITIES)


def test_a_finding_that_quotes_a_credential_is_redacted_rather_than_breaking_the_run():
    """A reviewer that finds a HARDCODED TOKEN is doing its job; that must not fail the workflow.

    The state refuses secret-shaped data outright, so a raw finding quoting the token would either
    make `TaskState.to_payload()` raise mid-run — turning a correct finding into a broken run — or
    put the credential into workflow history. Redaction at record time is the only answer that keeps
    both promises, and the token must not survive anywhere in the record or the prompt.
    """
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    report = fakes.FakeReviewReport(
        conclusion="failure",
        findings=[
            {
                "path": "cli/codna/config.py",
                "line": 7,
                "severity": "high",
                "category": "security",
                # The shape a real security finding has: it QUOTES the credential it found.
                "title": f"A GitHub token ({token}) is committed in the config module",
                "explanation": f"The literal {token} is used as the API credential.",
            }
        ],
    )
    feedback = review_findings_from_report(
        report, verdict="rejected", sha="a" * 40, fail=ContractError
    )
    assert feedback is not None

    # It records, and it is the FINDING that survives — a redaction is not a dropped finding.
    assert len(feedback.findings) == 1
    assert "GitHub token" in feedback.findings[0].title
    assert "committed in the config module" in feedback.findings[0].title
    # ...and the credential is nowhere in the record or the prompt it renders.
    payload_json = json.dumps(feedback.to_payload())
    prompt = implement_prompt_for_state({"review_feedback": feedback.to_payload()}, "Fix it")
    for text in (payload_json, prompt):
        assert token not in text
        assert "[redacted" in text

    # The state it produces is ENCODABLE — the run continues instead of dying on its own review.
    state = TaskState(
        run_id="run-1",
        task_id="task-1",
        review_verdict="rejected",
        review_feedback=feedback,
    )
    assert state.to_payload()["review_feedback"] is not None


def test_a_constructed_finding_carrying_a_credential_is_refused():
    """The other half: a record built in code cannot smuggle a token past the guard."""
    from codna.durable.review_feedback import ReviewFinding

    token = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz0123456789"
    with pytest.raises(ContractError) as excinfo:
        ReviewFinding(
            path="cli/codna/x.py", line=1, severity="high", title=f"Key {token}", explanation=""
        ).validate(fail=ContractError)
    assert excinfo.value.code == "invalid_review_feedback"
    assert token not in json.dumps(excinfo.value.details)


# ---------------------------------------------------------------------------------------
# 5. THE GATE: a rejection still REJECTS. Nothing here weakens the review.
# ---------------------------------------------------------------------------------------


def test_a_rejection_still_routes_back_to_implement_and_an_approval_still_opens_the_pr():
    """The fix adds information to the retry. It must not turn a rejection into a pass."""
    from codna.durable.routing import after_review
    from codna.durable.state import Attempts

    rejecting = TaskState(
        run_id="run-1",
        task_id="task-1",
        review_verdict="changes_requested",
        review_feedback=ReviewFeedback.from_payload(_feedback_payload(), fail=ContractError),
        attempts=Attempts(implement=1),
    )
    assert after_review(rejecting, max_attempts=3) == "implement"
    # An exhausted attempt budget still PARKS rather than looping with the findings in hand.
    exhausted = TaskState(
        run_id="run-1",
        task_id="task-1",
        review_verdict="changes_requested",
        review_feedback=ReviewFeedback.from_payload(_feedback_payload(), fail=ContractError),
        attempts=Attempts(implement=3),
    )
    assert after_review(exhausted, max_attempts=3) == "wait_for_human"
    approving = TaskState(run_id="run-1", task_id="task-1", review_verdict="approved")
    assert after_review(approving, max_attempts=3) == "open_pr"


# ---------------------------------------------------------------------------------------
# 6. End to end under BOTH engines: the retry's PROMPT (what the sidecar received) changed.
#
# The assertions above are on the prompt BUILDER. This section is on the value that reached the
# engine, which is the only place that can prove the wiring: `fakes.FakeSidecar.prompts` records
# every prompt the fake sidecar was started with, in order.
# ---------------------------------------------------------------------------------------


def _scripted_rejection_harness(repo: Path) -> fakes.Harness:
    """One review rejection (with findings), then an approval."""
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    harness.verify_passes = [True, True]
    harness.review_conclusions = ["neutral", "success"]
    harness.review_findings_script = [[dict(item) for item in REJECTING_FINDINGS], []]
    return harness


def _implement_prompts(harness: fakes.Harness) -> list[str]:
    """The prompts the sidecar was asked to IMPLEMENT with, in order.

    Read off the fake sidecar rather than rebuilt, so this asserts what the ENGINE received. The plan
    turn is excluded by its own marker, which is the prompt text the plan builder emits.
    """
    return [
        prompt
        for prompt in harness.sidecar.prompts
        if "Implement this task in the repository" in prompt
    ]


async def _drive_to_approval(workflow_cls: Any, spec: Any, workflow_id: str, client: Any) -> Any:
    """Start a task, wait for its gate, approve it, and return the completion."""
    handle = await client.start_workflow(
        workflow_cls.run,
        args=[spec.to_payload(), None],
        id=workflow_id,
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    deadline = asyncio.get_running_loop().time() + RESULT_TIMEOUT_S
    status: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        status = await handle.query(workflow_cls.status)
        if status.get("awaiting_approval"):
            break
        await asyncio.sleep(0.02)
    else:  # pragma: no cover - a workflow that never reaches its gate
        raise AssertionError(f"{workflow_id} never parked; last status={status!r}")
    await handle.execute_update(workflow_cls.approve, {"identity": "reviewer@codna.ai"})
    return await asyncio.wait_for(handle.result(), timeout=RESULT_TIMEOUT_S)


def _with_worker(engine: str, scenario: Any) -> Any:
    async def run() -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            plugins = [build_task_graph().plugin] if engine == "langgraph" else []
            with activity_executor() as executor:
                async with Worker(
                    env.client,
                    task_queue=TASK_QUEUE,
                    workflows=list(registered_workflows()),
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                    plugins=plugins,
                ):
                    with env.auto_time_skipping_disabled():
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    return asyncio.run(run())


@pytest.mark.parametrize(
    "engine,workflow_cls",
    [("plain", TaskWorkflowPlain), ("langgraph", TaskWorkflow)],
    ids=["plain", "langgraph"],
)
def test_the_engine_that_RAN_the_retry_received_the_findings_in_its_prompt(
    tmp_path: Path, engine: str, workflow_cls: Any
):
    """The end-to-end claim, on both engines: the retry prompt the SIDECAR got is not the old one.

    This is the assertion that would have failed on `run-20261116T005743Z`: with two implement
    prompts recorded and the rejection in between, the second must contain the reviewer's words.
    Before the fix the two prompts were byte-identical.
    """
    from tests.durable.workflow_specs import task_spec

    repo = fakes.make_repo(tmp_path / "repo")
    harness = _scripted_rejection_harness(repo)
    spec = task_spec(repo, engine=engine)

    async def scenario(client: Any) -> Any:
        return await _drive_to_approval(workflow_cls, spec, f"{engine}-feedback", client)

    completion = _with_worker(engine, scenario)
    assert completion.outcome == "approved"
    assert completion.state["review_verdict"] == "approved"

    implement_prompts = _implement_prompts(harness)
    assert len(implement_prompts) == 2, implement_prompts
    first, retry = implement_prompts

    # THE DEFECT: the retry was byte-identical to the attempt that was rejected.
    assert retry != first
    # ...and the difference is the reviewer's findings, rendered.
    assert prompts_module._REVIEW_FEEDBACK_HEADING not in first
    assert prompts_module._REVIEW_FEEDBACK_HEADING in retry
    for finding in REJECTING_FINDINGS:
        assert finding["title"] in retry
        assert f"{finding['path']}:{finding['line']}" in retry
    # The task itself is still stated: the block ADDS to the prompt, it does not replace it.
    assert "Fix the flaky retry in the client" in retry

    # The gate held throughout: a rejection was routed back, and only the APPROVAL published.
    assert harness.push_calls == 1
