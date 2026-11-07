"""Publishing: skipped when there is nothing to publish to, refused when a token is missing.

`open_pr` is the unconditional last doing-node of every durable run, so "cannot publish" used to have
nowhere to land: the activity raised, the task failed, and a run whose work was committed, verified and
approved was reported as broken. These tests pin the contract that replaced that:

* a repository with **no origin** SKIPS — the task finishes ``approved_unpublished``, the ledger says
  ``publish.skipped`` with a named reason, and no pull request is fabricated;
* a repository whose origin is **not GitHub** skips for its OWN reason (the user's next step differs:
  "add a remote" versus "this remote is not one codna publishes to");
* a repository **with a GitHub origin and no token REFUSES** — the activity raises the named
  ``activity_pr_token_missing``, writes an actionable ``publish.blocked`` note naming the variables,
  and the task does NOT finish as if publishing had been unnecessary;
* a **transient git failure is not a skip**: a checkout git cannot read stays a hard failure;
* a normal run **with a token still publishes**, and the observable outcomes match on BOTH engines.

The runs go through the REAL workflow classes on a real (time-skipping) Temporal server with the real
activities — a fake sidecar, the harness's scripted push and REAL git — because the defect was in the
composition of the steps, not in any one of them. `publish_probe` and the step are also tested
directly, where the distinction "skip versus failure" is decided.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

import asyncio  # noqa: E402

from codna.durable.activity_steps import checkpoint_turn, open_pr, run_cline_turn  # noqa: E402
from codna.durable.activity_support import ActivityError  # noqa: E402
from codna.durable.contracts import (  # noqa: E402
    CheckpointRequest,
    PullRequestRequest,
    TaskSpec,
)
from codna.durable.publish import (  # noqa: E402
    PUBLISH_SKIP_NO_REMOTE,
    PUBLISH_SKIP_REASON_VALUES,
    PUBLISH_SKIP_REMOTE_NOT_GITHUB,
    PublishSkipReason,
)
from codna.durable.publish_steps import publish_probe  # noqa: E402
from codna.durable.state import TaskState, TaskStateError  # noqa: E402
from codna.durable.workflows import RunWorkflow, TaskWorkflow, TaskWorkflowPlain  # noqa: E402
from tests.durable import fakes  # noqa: E402
from tests.durable.approval_driver import approve_if_parked  # noqa: E402
from tests.durable.test_task_workflow_time_skipping import (  # noqa: E402
    RUN_TIMEOUT,
    TASK_QUEUE,
    _result,
    _run,
    _update,
    _wait_parked,
    _with_worker,
)

GITLAB_ORIGIN = "https://gitlab.com/thyn-ai/codna.git"
LOCAL_ORIGIN = "/tmp/codna-local-origin.git"


def _spec(repo: Path, *, engine: str, task_id: str = "task-1") -> TaskSpec:
    return TaskSpec(
        run_id="run-1",
        task_id=task_id,
        task_number=1,
        brief="Fix the flaky retry in the client",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine=engine,
    )


def _harness(repo: Path) -> fakes.Harness:
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    return fakes.build_harness(repo, runs_root)


def _implemented(repo: Path, *, engine: str = "plain") -> tuple[fakes.Harness, Any, Any]:
    """A task driven as far as a committed, approved change — the state `open_pr` is handed.

    Returns the harness, the task context and the checkpoint outcome, so each test can call the
    publish step with exactly what a workflow would hand it (and assert on the ledger it wrote).
    """
    harness = _harness(repo)
    spec = _spec(repo, engine=engine)
    from codna.durable.contracts import TaskContext

    context = TaskContext.from_spec(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))
    return harness, context, checkpoint


# -- the probe: three answers, deliberately distinguished ---------------------------------


def test_a_repository_with_no_origin_is_no_remote(tmp_path: Path) -> None:
    repo = fakes.make_repo(tmp_path / "repo", origin=None)
    harness = _harness(repo)
    assert publish_probe(harness.runtime, str(repo)) == ("", PUBLISH_SKIP_NO_REMOTE)


def test_a_non_github_origin_is_its_own_reason(tmp_path: Path) -> None:
    repo = fakes.make_repo(tmp_path / "repo", origin=GITLAB_ORIGIN)
    harness = _harness(repo)
    # The URL comes BACK with the reason: a user needs to read which remote was found, and the
    # ledger note redacts it before writing it anywhere.
    assert publish_probe(harness.runtime, str(repo)) == (GITLAB_ORIGIN, PUBLISH_SKIP_REMOTE_NOT_GITHUB)


def test_a_local_path_origin_is_not_publishable(tmp_path: Path) -> None:
    """An origin can exist and still be unpublishable: a local path is not a GitHub repository."""
    repo = fakes.make_repo(tmp_path / "repo", origin=LOCAL_ORIGIN)
    harness = _harness(repo)
    assert publish_probe(harness.runtime, str(repo)) == (LOCAL_ORIGIN, PUBLISH_SKIP_REMOTE_NOT_GITHUB)


def test_a_github_origin_is_publishable(tmp_path: Path) -> None:
    repo = fakes.make_repo(tmp_path / "repo")
    harness = _harness(repo)
    assert publish_probe(harness.runtime, str(repo)) == (fakes.GITHUB_ORIGIN, None)


def test_a_checkout_git_cannot_read_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    """A transient git problem must never be reported as "this repository has no remote".

    `publish_probe` asks whether the tree is a repository at all FIRST. A tree that is not one (or a
    git that cannot run there) has no answer to "what is its origin", so the honest outcome is the
    hard failure — silence would turn a broken worker into a run that looks finished.
    """
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    harness = _harness(fakes.make_repo(tmp_path / "repo"))

    with pytest.raises(ActivityError) as caught:
        publish_probe(harness.runtime, str(not_a_repo))
    assert caught.value.code == "activity_git_failed", caught.value.as_dict()


# -- the step: a skip finishes; a refusal does not -----------------------------------------


def test_no_remote_skips_without_a_pull_request(tmp_path: Path) -> None:
    repo = fakes.make_repo(tmp_path / "repo", origin=None)
    harness, context, checkpoint = _implemented(repo)

    outcome = open_pr(
        PullRequestRequest(state=checkpoint.state, context=context, title="Fix the retry")
    )

    assert harness.push_calls == 0, "nothing should have been pushed"
    assert outcome.state["pr"] is None
    assert outcome.state["publish_skip"] == PUBLISH_SKIP_NO_REMOTE
    assert outcome.details["skipped"] is True
    assert outcome.details["reason"] == PUBLISH_SKIP_NO_REMOTE
    # The attempt counter still advances: the step RAN, which must be visible in the state an
    # auditor reads.
    assert outcome.state["attempts"]["open_pr"] == 1
    # And the ledger says what was skipped, that the branch exists, and where it points. "Nothing
    # happened" is not the fact to record; "the commit is there and there is nowhere to send it" is.
    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.kind == "note" and record.payload.get("event") == "publish.skipped"
    ]
    assert len(notes) == 1, notes
    assert notes[0]["reason"] == PUBLISH_SKIP_NO_REMOTE
    assert notes[0]["branch"] == context.branch
    assert notes[0]["head_sha"] == checkpoint.sha
    assert notes[0]["origin"] is None
    assert "no 'origin' remote" in notes[0]["message"]


def test_a_non_github_origin_skips_and_redacts_the_url(tmp_path: Path) -> None:
    """The note names the remote it found, with any embedded credential removed."""
    tokenized = "https://x-access-token:ghp_secretvaluehere@gitlab.com/thyn-ai/codna.git"
    repo = fakes.make_repo(tmp_path / "repo", origin=tokenized)
    harness, context, checkpoint = _implemented(repo)

    outcome = open_pr(
        PullRequestRequest(state=checkpoint.state, context=context, title="Fix the retry")
    )

    assert outcome.state["publish_skip"] == PUBLISH_SKIP_REMOTE_NOT_GITHUB
    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.kind == "note" and record.payload.get("event") == "publish.skipped"
    ]
    assert len(notes) == 1, notes
    assert notes[0]["reason"] == PUBLISH_SKIP_REMOTE_NOT_GITHUB
    # REDACTED in the ledger: a ledger is exported, and the origin legitimately embedded a token.
    assert "ghp_secretvaluehere" not in str(notes[0])
    assert notes[0]["origin"] == "https://gitlab.com/thyn-ai/codna.git"


def test_a_github_origin_without_a_token_refuses_with_a_named_code(tmp_path: Path) -> None:
    """The case that is NOT a skip: a GitHub remote was configured, so a PR was expected.

    A skip here would leave the user believing a pull request exists. The refusal keeps the code
    callers already handle (`activity_pr_token_missing`), names the variable to set, and records it
    in the ledger — the run's own audit trail — so the fix does not require opening a history.
    """
    repo = fakes.make_repo(tmp_path / "repo")  # a GitHub origin, by default
    harness = _harness(repo)
    # The refuser under test is the REAL resolution rule: no provider injected, and no token in the
    # process environment. Both are cleared rather than assumed empty.
    harness.runtime.github_token = None
    spec = _spec(repo, engine="plain")
    from codna.durable.contracts import TaskContext

    context = TaskContext.from_spec(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))

    import os

    saved = {name: os.environ.pop(name) for name in ("CODNA_GITHUB_TOKEN", "GITHUB_TOKEN") if name in os.environ}
    try:
        with pytest.raises(ActivityError) as caught:
            open_pr(PullRequestRequest(state=checkpoint.state, context=context, title="Fix it"))
    finally:
        os.environ.update(saved)

    assert caught.value.code == "activity_pr_token_missing", caught.value.as_dict()
    # The message tells a user what to DO, and the details name the variables without a value.
    assert "CODNA_GITHUB_TOKEN" in str(caught.value)
    assert caught.value.details["variable_names"] == ["CODNA_GITHUB_TOKEN", "GITHUB_TOKEN"]
    assert "token" not in {key for key in caught.value.details if key != "variable_names"}, (
        "a details map is printed: it must not name a value"
    )
    assert harness.push_calls == 0
    # The ledger carries the actionable half, so the refusal is auditable from the run.
    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.kind == "note" and record.payload.get("event") == "publish.blocked"
    ]
    assert len(notes) == 1, notes
    assert notes[0]["reason"] == "no_credential"
    assert notes[0]["variable_names"] == ["CODNA_GITHUB_TOKEN", "GITHUB_TOKEN"]
    assert "not skipped" in notes[0]["message"]
    assert notes[0]["origin"] == fakes.GITHUB_ORIGIN


def test_a_retried_refusal_records_the_note_once(tmp_path: Path) -> None:
    """`ACTIVITY_RETRIES` attempts of the SAME refusal must leave ONE record, not one per attempt.

    MEASURED on the live acceptance run: the refusal path wrote three identical `publish.blocked`
    notes, because a bookkeeping note carries neither `session_id` nor `event_index` and so is outside
    the ledger's own dedupe key. "One record per fact" is what the ledger is read for, so the step
    checks before appending — and this pins it, because the bug is invisible in a happy-path test.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = _harness(repo)
    harness.runtime.github_token = None
    spec = _spec(repo, engine="plain")
    from codna.durable.contracts import TaskContext

    context = TaskContext.from_spec(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    checkpoint = checkpoint_turn(CheckpointRequest(state=moved.state, context=context))
    request = PullRequestRequest(state=checkpoint.state, context=context, title="Fix it")

    import os

    saved = {name: os.environ.pop(name) for name in ("CODNA_GITHUB_TOKEN", "GITHUB_TOKEN") if name in os.environ}
    try:
        for _attempt in range(3):  # exactly what Temporal does after a retryable activity error
            with pytest.raises(ActivityError):
                open_pr(request)
    finally:
        os.environ.update(saved)

    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.kind == "note" and record.payload.get("event") == "publish.blocked"
    ]
    assert len(notes) == 1, notes


def test_a_publishable_repository_with_a_token_still_publishes(tmp_path: Path) -> None:
    """The normal path unchanged: a token and a GitHub origin open the pull request."""
    repo = fakes.make_repo(tmp_path / "repo")
    harness, context, checkpoint = _implemented(repo)

    outcome = open_pr(
        PullRequestRequest(state=checkpoint.state, context=context, title="Fix the retry")
    )

    assert harness.push_calls == 1
    assert outcome.state["pr"] is not None
    assert outcome.state["pr"]["branch"] == context.branch
    assert outcome.state["publish_skip"] is None
    events = [
        record.payload.get("event")
        for record in harness.ledger("run-1").records()
        if record.kind == "note"
    ]
    assert "pull_request.opened" in events
    assert "publish.skipped" not in events


# -- state: the skip is a closed, validated fact --------------------------------------------


def test_a_skip_reason_is_the_closed_set() -> None:
    assert set(PUBLISH_SKIP_REASON_VALUES) == {"no_remote", "remote_not_github"}
    # `PublishSkipReason` is the same closed set, and this is what keeps a new reason from being
    # added in prose only: the type, the tuple and the state's validation all read one place.
    assert set(PublishSkipReason.__args__) == set(PUBLISH_SKIP_REASON_VALUES)


def test_state_refuses_an_unknown_publish_skip() -> None:
    payload = TaskState(run_id="r", task_id="t").to_payload()
    payload["publish_skip"] = "maybe_later"
    with pytest.raises(TaskStateError) as caught:
        TaskState.from_payload(payload)
    assert caught.value.code == "invalid_publish_skip"
    assert "no_remote" in str(caught.value)


def test_state_refuses_a_skip_beside_a_pull_request() -> None:
    """The two facts contradict each other, so a state carrying both is refused, not preferred."""
    payload = TaskState(
        run_id="r",
        task_id="t",
        pr=None,
    ).to_payload()
    payload["pr"] = {
        "number": 1,
        "url": "https://github.com/thyn-ai/codna/pull/1",
        "branch": "codna/run-r/task-1",
        "head_sha": None,
    }
    payload["publish_skip"] = PUBLISH_SKIP_NO_REMOTE
    with pytest.raises(TaskStateError) as caught:
        TaskState.from_payload(payload)
    assert caught.value.code == "invalid_publish_skip"
    assert "both" in str(caught.value)


# -- the workflows: a run completes, and says what it published ---------------------------


def _drive_task(tmp_path: Path, engine: str, workflow_cls: Any, *, origin: str | None) -> dict[str, Any]:
    repo = fakes.make_repo(tmp_path / "repo", origin=origin)
    harness = _harness(repo)
    spec = _spec(repo, engine=engine)
    workflow_id = f"publish-{engine}-{'local' if origin is None else 'github'}"

    async def scenario(client: Any) -> dict[str, Any]:
        handle = await client.start_workflow(
            workflow_cls.run,
            args=[spec.to_payload(), None],
            id=workflow_id,
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        await _wait_parked(handle, query=workflow_cls.status)
        await _update(handle, workflow_cls.approve, {"identity": "reviewer@codna.ai"})
        completion = await _result(handle)
        return {"completion": completion, "route": harness.route(), "pushes": harness.push_calls}

    return _run(_with_worker(engine, scenario))


@pytest.mark.parametrize(
    ("engine", "workflow_cls"),
    [("plain", TaskWorkflowPlain), ("langgraph", TaskWorkflow)],
    ids=["plain", "langgraph"],
)
def test_a_local_only_repository_completes_without_a_pull_request(
    tmp_path: Path, engine: str, workflow_cls: Any
) -> None:
    """The acceptance case: no origin, and the task still SUCCEEDS — `approved_unpublished`."""
    result = _drive_task(tmp_path, engine, workflow_cls, origin=None)
    completion = result["completion"]

    assert completion.outcome == "approved_unpublished", completion
    assert result["pushes"] == 0
    assert completion.state["pr"] is None
    assert completion.state["publish_skip"] == PUBLISH_SKIP_NO_REMOTE
    # The route shows the publish step RAN (and skipped) rather than never reaching it.
    assert result["route"][-1] == "open_pr_skipped", result["route"]
    assert "approval" not in result["route"]


@pytest.mark.parametrize(
    ("engine", "workflow_cls"),
    [("plain", TaskWorkflowPlain), ("langgraph", TaskWorkflow)],
    ids=["plain", "langgraph"],
)
def test_a_publishable_repository_still_finishes_approved(
    tmp_path: Path, engine: str, workflow_cls: Any
) -> None:
    """The normal case, unchanged on both engines: `approved`, one push, a pull request."""
    result = _drive_task(tmp_path, engine, workflow_cls, origin=fakes.GITHUB_ORIGIN)
    completion = result["completion"]

    assert completion.outcome == "approved", completion
    assert result["pushes"] == 1
    assert completion.state["pr"] is not None
    assert completion.state["publish_skip"] is None
    assert result["route"][-1] == "open_pr"


def test_a_run_of_a_local_only_repository_completes_and_counts_unpublished_tasks(
    tmp_path: Path,
) -> None:
    """At RUN level: `completed` (the work is done), with the skips counted and named in the ledger.

    The run-level outcome is deliberately NOT a new value: `completed` means every task finished its
    work, which is true. What a reader must be able to see is that nothing was published, so the
    completion counts the unpublished tasks and the ledger records which they were and why.
    """
    repo = fakes.make_repo(tmp_path / "repo", origin=None)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    tasks = tuple(_spec(repo, engine="plain", task_id=f"task-{n}") for n in (1,))
    payload = {
        "run_id": "run-1",
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": "plain",
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
    }

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id="run-local-only",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        for _ in range(600):
            await approve_if_parked(
                client, "task-run-1-task-1", TaskWorkflowPlain, identity="soak@codna.ai"
            )
            described = await handle.describe()
            if int(described.status) != 1:
                break
            await asyncio.sleep(0.02)
        return await asyncio.wait_for(handle.result(), timeout=60)

    completion = _run(_with_worker("plain", scenario))

    assert completion.outcome == "completed", completion
    assert [summary.task_id for summary in completion.tasks] == ["task-1"]
    assert completion.tasks[0].outcome == "approved_unpublished"
    assert completion.unpublished_tasks == 1
    assert harness.push_calls == 0
    # ONE run-level note naming what was not published, so "did this run open anything?" is
    # answerable from the run's own ledger without walking every task.
    notes = [
        record.payload
        for record in harness.ledger("run-1").records()
        if record.kind == "note" and "without a pull request" in str(record.payload.get("message"))
    ]
    assert len(notes) == 1, notes
    assert "task-1" in notes[0]["message"]