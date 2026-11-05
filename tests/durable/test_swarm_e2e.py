"""END-TO-END: a durable run that spawns a SWARM and multitask-parallelises, under real Temporal.

This is the acceptance test the swarm claim needs, and it is written to be one that CANNOT pass by
accident. Concretely, every assertion below fails on the pre-change code:

* the two delegation fields are asserted on the payload the ENGINE was handed — read from the fake
  sidecar's recorded create-session specs, which is the boundary a real sidecar would see. Before this
  change the durable path sent neither field, so these assertions were unreachable, not merely wrong.
* concurrency is asserted as INTERVAL OVERLAP on a monotonic clock, not as "both tasks finished". A
  strictly sequential run finishes both tasks too; it simply never has two turns open at once, and the
  overlap assertion is exactly what distinguishes the two.
* the graph engine's history is REPLAYED with a `Replayer` against the recorded run, so "LangGraph is
  driving it" is shown by replay-safety rather than by inspecting a node list.
* the ledger's hash chain is verified at the end, so the run's own audit trail is proved consistent.

What is NOT real here, and why: the MODEL. A turn is served by `tests/durable/fakes.FakeSidecar`, the
same boundary every other durable scenario uses, because a real provider call would make this test
non-deterministic, slow and credential-dependent. What IS real: the Temporal server (time-skipping
test server), both workflow engines, the LangGraph plugin and graph, every activity body, the ledger,
and git. The engine's own team lifecycle (actually running teammates) is covered by the engine's suite
(`bun test .../tools/team`), and its refusal of the flags under a forbidden profile by
`algenta/policy.test.ts`; this test covers the SEAM the durable path was missing — the flags reaching
the engine at all.

Run with the extra installed: ``CODNA_TEST_TEMPORAL=1 python3 -m pytest
tests/durable/test_swarm_e2e.py``.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

from temporalio.exceptions import TemporalError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402

from codna.durable.activities import ACTIVITIES, activity_executor  # noqa: E402
from codna.durable.contracts import TaskSpec  # noqa: E402
from codna.durable.graph import build_task_graph  # noqa: E402
from codna.durable.swarm import (  # noqa: E402
    PROFILE_SWARM_PERMISSIONS,
    SwarmConfigError,
    normalize_swarm,
    swarm_session_spec,
)
from codna.durable.workflows import (  # noqa: E402
    RunWorkflow,
    TaskWorkflow,
    TaskWorkflowPlain,
    registered_workflows,
)
from tests.durable import fakes  # noqa: E402

TASK_QUEUE = "codna-durable-swarm-tests"
RUN_ID = "run-swarm"
RUN_TIMEOUT = timedelta(hours=24)
# Long enough that two turns MUST overlap to be observed at all, short enough that the whole file stays
# a few seconds. Two turns per task, four tasks: ~8 held turns.
TURN_HOLD_S = 0.05


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _spec(repo: Path, number: int, *, engine: str, spawn: bool, teams: bool) -> TaskSpec:
    return TaskSpec(
        run_id=RUN_ID,
        task_id=f"task-{number}",
        task_number=number,
        brief=f"Parallelise area {number}",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=("true",),
        engine=engine,  # type: ignore[arg-type]
        worktree=True,
        enable_spawn_agent=spawn,
        enable_agent_teams=teams,
    )


def _payload(
    repo: Path, tasks: tuple[TaskSpec, ...], *, engine: str, parallel: int
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task.to_payload() for task in tasks],
        "engine": engine,
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
        "worktree": True,
        "max_parallel_tasks": parallel,
    }


async def _approve_child(client: Any, handle_id: str, approval_cls: Any) -> None:
    """Approve a parked child, if it is parked. The two BENIGN races are the only ones tolerated.

    **``approval_cls`` is a required class for a reason, and getting it wrong is a HANG, not a red
    assertion.** A parameterised test supplied ``None`` here for the graph engine; ``None.status``
    raised ``AttributeError`` inside the driver loop, which the helper's blanket ``except Exception``
    swallowed — so no approval was ever sent, every child stayed parked at its ``interrupt()``, the run
    stayed RUNNING forever, and the caller died on a 120s timeout with no message pointing at the
    cause. A hang can never satisfy a caller and never explains itself, so the preconditions are
    asserted rather than guarded: a wrong class now fails at the call that got it wrong.

    The ``except`` clauses are narrowed to `TemporalError` for the same reason. Both races here are
    genuine Temporal answers — a child that has not started yet (``RPCError``: execution not found) and
    a double approval (``WorkflowUpdateFailedError``: the validator refused the second) — so both are
    subclasses of it. Anything else is a bug in this harness, and must propagate instead of silently
    turning a test into a timeout.
    """
    if not isinstance(approval_cls, type):
        raise TypeError(
            f"_approve_child needs the child workflow class, got {approval_cls!r}: a None/live object "
            "here cannot query or update, and swallowing that failure leaves every child parked"
        )
    if not hasattr(approval_cls, "status") or not hasattr(approval_cls, "approve"):
        raise TypeError(
            f"{approval_cls!r} cannot drive an approval: it needs the 'status' query and the 'approve' update"
        )
    child = client.get_workflow_handle(handle_id)
    try:
        status = await child.query(approval_cls.status)
    except TemporalError:  # not started yet, or already closed
        return
    if not status.get("awaiting_approval"):
        return
    try:
        await child.execute_update(approval_cls.approve, {"identity": "swarm@codna.ai"})
    except TemporalError:  # a concurrent iteration already approved it
        return


async def _drive(
    client: Any,
    run_payload: dict[str, Any],
    count: int,
    *,
    workflow_cls: Any,
    approval_cls: Any,
) -> Any:
    handle = await client.start_workflow(
        RunWorkflow.run,
        args=[run_payload, None],
        id=f"run-{RUN_ID}",
        task_queue=TASK_QUEUE,
        run_timeout=RUN_TIMEOUT,
    )
    for _ in range(1200):
        for number in range(1, count + 1):
            await _approve_child(client, f"task-{RUN_ID}-task-{number}", approval_cls)
        described = await handle.describe()
        if int(described.status) != 1:  # 1 == RUNNING
            break
        await asyncio.sleep(0.01)
    return await asyncio.wait_for(handle.result(), timeout=120)


def _with_worker(engine: str, scenario: Any) -> Any:
    """A fresh time-skipping server + the REAL activities, the graph plugin for the graph engine."""

    async def wrapper() -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            plugins = [build_task_graph().plugin] if engine == "langgraph" else []
            with activity_executor(max_workers=4) as executor:
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

    return _run(wrapper())


# ---------------------------------------------------------------------------------------
# 1. The flags reach the ENGINE's create-session payload (the thing that was missing).
# ---------------------------------------------------------------------------------------


def test_the_swarm_flags_are_in_every_create_session_payload(tmp_path: Path):
    """THE regression: the durable path used to send neither field, so the engine never saw them.

    Read from the payloads the fake sidecar RECORDED (`create_session(spec=…)`), which is byte-for-byte
    what a real sidecar receives. A version of this test that asserted on the TaskSpec instead would
    pass while the flags were dropped at the boundary — which is precisely the bug being fixed.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    tasks = tuple(
        _spec(repo, number, engine="plain", spawn=True, teams=True) for number in (1, 2)
    )
    completion = _with_worker(
        "plain",
        lambda client: _drive(
            client,
            _payload(repo, tasks, engine="plain", parallel=1),
            2,
            workflow_cls=RunWorkflow,
            approval_cls=TaskWorkflowPlain,
        ),
    )
    assert completion.outcome == "completed", completion

    # Every session the run created — plan and implement turns alike — carried both fields.
    sessions = [entry for entry in harness.sidecar.sessions if entry.get("spec")]
    assert sessions, "no session was created, so nothing about delegation was exercised"
    for entry in sessions:
        assert entry["spec"].get("enable_spawn_agent") is True, entry["spec"]
        assert entry["spec"].get("enable_agent_teams") is True, entry["spec"]


def test_a_run_that_never_asked_sends_neither_field(tmp_path: Path):
    """The other half, and what keeps the pre-feature request byte-identical.

    An absent key is the answer for a run that did not ask; sending ``false`` explicitly would be a
    different request from the one every recorded history replays.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    tasks = (_spec(repo, 1, engine="plain", spawn=False, teams=False),)
    _with_worker(
        "plain",
        lambda client: _drive(
            client,
            _payload(repo, tasks, engine="plain", parallel=1),
            1,
            workflow_cls=RunWorkflow,
            approval_cls=TaskWorkflowPlain,
        ),
    )
    sessions = [entry for entry in harness.sidecar.sessions if entry.get("spec")]
    assert sessions
    for entry in sessions:
        assert "enable_spawn_agent" not in entry["spec"], entry["spec"]
        assert "enable_agent_teams" not in entry["spec"], entry["spec"]


# ---------------------------------------------------------------------------------------
# 2. MULTITASK: multiple turns REALLY run concurrently (interval overlap, not "both finished").
# ---------------------------------------------------------------------------------------


def _overlaps(intervals: list[tuple[str, float, float]]) -> list[tuple[str, str]]:
    """Every pair of turn intervals that genuinely overlapped in time.

    Strictly ``start_a < finish_b and start_b < finish_a``: touching endpoints (finish == start) is NOT
    overlap, so a run that hands one slot to the next task the instant it frees it cannot satisfy this.
    """
    pairs: list[tuple[str, str]] = []
    for left in range(len(intervals)):
        for right in range(left + 1, len(intervals)):
            a_id, a_start, a_end = intervals[left]
            b_id, b_start, b_end = intervals[right]
            if a_start < b_end and b_start < a_end:
                pairs.append((a_id, b_id))
    return pairs


@pytest.mark.parametrize(
    ("engine", "workflow_cls"),
    [
        ("plain", TaskWorkflowPlain),
        ("langgraph", TaskWorkflow),
    ],
    ids=["plain", "langgraph"],
)
def test_multiple_turns_overlap_in_time(tmp_path: Path, engine: str, workflow_cls) -> None:
    """Two tasks' paid turns are open SIMULTANEOUSLY — measured, not assumed, on BOTH engines.

    The failure this catches: `RunWorkflow` awaiting each child before starting the next. That run
    finishes every task and passes a "both completed" assertion, so the assertion has to be about the
    intervals the turns actually occupied.

    Parameterised because this was first proven only on `plain`, while the owner's requirement was
    concurrency under LangGraph too. A graph engine that serialises its children while the plain engine
    does not is a real and separate regression, and one engine passing proves nothing about the other.

    **The ``[langgraph]`` case once hung, and it was this harness, not the engine.** The parameter
    supplied ``None`` as the approval class, ``None.status`` raised ``AttributeError``, the old blanket
    ``except Exception`` in `_approve_child` swallowed it, no child was ever approved, and the run sat
    RUNNING until the 120s timeout. The engine was concurrency-correct the whole time: with the class
    supplied it starts both children before awaiting either, their turns overlap across tasks, and the
    parent and both child histories replay clean. That is why `_approve_child` now asserts its class and
    catches only `TemporalError`.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(
        repo, runs_root, measure_overlap=True, turn_hold_s=TURN_HOLD_S
    )
    tasks = tuple(
        _spec(repo, number, engine=engine, spawn=True, teams=True) for number in (1, 3)
    )
    completion = _with_worker(
        engine,
        lambda client: _drive(
            client,
            _payload(repo, tasks, engine=engine, parallel=2),
            3,
            workflow_cls=RunWorkflow,
            approval_cls=workflow_cls,
        ),
    )
    assert completion.outcome == "completed", completion

    intervals = harness.sidecar.intervals
    assert len(intervals) >= 4, intervals
    overlaps = _overlaps(intervals)
    assert overlaps, (
        "no two turns overlapped: the run serialised its tasks. "
        f"engine={engine} intervals={intervals}"
    )
    # And the overlap is across TASKS, not merely between two turns of one task: the run's task count
    # is what was parallelised. `turn_key` is `<run>:<task>:<seq>`, so the task is its middle field.
    cross_task = [
        pair
        for pair in overlaps
        if pair[0].split(":")[1] != pair[1].split(":")[1]
    ]
    assert cross_task, f"engine={engine} turns overlapped only within one task: {overlaps}"


def test_a_wrong_approval_class_fails_at_once_instead_of_hanging():
    """The guard that turns the hang this file actually produced into an immediate, named failure.

    ``test_multiple_turns_overlap_in_time[langgraph]`` used to pass ``None`` as the approval class.
    ``None.status`` raised ``AttributeError``, the driver's blanket ``except Exception`` swallowed it, no
    child was ever approved, and the run sat RUNNING until the 120s timeout — a hang that named neither
    the cause nor the engine. Two properties are pinned here, both of which the old helper failed:

    * a class that cannot query or update is REFUSED, with the offender in the message;
    * only `TemporalError` (the two benign races: not started yet, already approved) is absorbed, so a
      harness bug propagates instead of becoming a timeout.

    No Temporal server is needed: this is the helper's own contract, and the point is that it fails
    before any await.
    """

    async def scenario() -> None:
        with pytest.raises(TypeError) as caught:
            await _approve_child(None, f"task-{RUN_ID}-task-1", None)
        assert "child workflow class" in str(caught.value)

        # A live object is refused too: it is neither None nor a class, and has no classmethods.
        with pytest.raises(TypeError):
            await _approve_child(None, f"task-{RUN_ID}-task-1", object())

        # A class that is NOT a workflow (no `status`/`approve`) is refused rather than queried.
        class NotAWorkflow:
            pass

        with pytest.raises(TypeError) as caught_missing:
            await _approve_child(None, f"task-{RUN_ID}-task-1", NotAWorkflow)
        assert "cannot drive an approval" in str(caught_missing.value)

    _run(scenario())


def test_parallel_1_is_strictly_sequential_so_the_overlap_assertion_is_not_vacuous(tmp_path: Path):
    """The CONTROL for the test above: with ``max_parallel_tasks=1`` nothing overlaps.

    Without this, an overlap assertion could be satisfied by the harness holding every turn open at
    once regardless of the workflow's behaviour, and the "concurrency" claim would be about the fake.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(
        repo, runs_root, measure_overlap=True, turn_hold_s=TURN_HOLD_S
    )
    tasks = tuple(
        _spec(repo, number, engine="plain", spawn=False, teams=False) for number in (1, 3)
    )
    completion = _with_worker(
        "plain",
        lambda client: _drive(
            client,
            _payload(repo, tasks, engine="plain", parallel=1),
            3,
            workflow_cls=RunWorkflow,
            approval_cls=TaskWorkflowPlain,
        ),
    )
    assert completion.outcome == "completed", completion
    cross_task = [
        pair
        for pair in _overlaps(harness.sidecar.intervals)
        if pair[0].split(":")[1] != pair[1].split(":")[1]
    ]
    assert cross_task == [], f"sequential run overlapped across tasks: {cross_task}"


# ---------------------------------------------------------------------------------------
# 3. LangGraph drives it, and the history REPLAYS.
# ---------------------------------------------------------------------------------------


def test_the_graph_engine_runs_the_swarm_and_its_history_replays(tmp_path: Path):
    """The graph engine executes the same swarm run, and a `Replayer` accepts its history.

    Replay-safety is the claim "LangGraph is the thing driving it" in its strongest form: the plugin
    rewrites graph nodes into activity calls, so a history recorded under the graph can only replay if
    the graph produced exactly those commands. ``replay_failure is None`` is the verdict.

    The Replayer gets a FRESH plugin — constructing `LangGraphPlugin` mutates the graph it is given
    (the D2.0 spike's finding 2), so a shared one works once and raises afterwards.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    tasks = tuple(
        _spec(repo, number, engine="langgraph", spawn=True, teams=True) for number in (1, 2)
    )

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[_payload(repo, tasks, engine="langgraph", parallel=1), None],
            id=f"run-{RUN_ID}",
            task_queue=TASK_QUEUE,
            run_timeout=RUN_TIMEOUT,
        )
        for _ in range(1200):
            for number in (1, 2):
                await _approve_child(client, f"task-{RUN_ID}-task-{number}", TaskWorkflow)
            if int((await handle.describe()).status) != 1:
                break
            await asyncio.sleep(0.01)
        completion = await asyncio.wait_for(handle.result(), timeout=120)
        # The RUN's history schedules child workflows; the graph's node activities are in each CHILD's
        # own history, so the graph evidence has to be read there.
        child_histories = []
        for number in (1, 2):
            child = client.get_workflow_handle(f"task-{RUN_ID}-task-{number}")
            child_histories.append(await child.fetch_history())
        return completion, child_histories

    completion, child_histories = _with_worker("langgraph", scenario)
    assert completion.outcome == "completed", completion
    # The graph's own node activities are in the child histories, which is what makes these GRAPH runs
    # rather than the plain engine wearing the same name.
    scheduled: list[str] = []
    for history in child_histories:
        scheduled.extend(
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
        )
    graph_nodes = [name for name in scheduled if name.startswith("codna.task.")]
    assert "codna.task.plan" in graph_nodes, graph_nodes
    assert "codna.task.implement" in graph_nodes, graph_nodes
    # ...and the delegation fields reached the engine on THIS engine too.
    for entry in harness.sidecar.sessions:
        if entry.get("spec"):
            assert entry["spec"].get("enable_agent_teams") is True, entry["spec"]

    async def replay() -> Any:
        # One fresh graph/plugin pair per child history: constructing `LangGraphPlugin` mutates the
        # graph it is given (spike finding 2), so a plugin cannot be reused across replays.
        failures = []
        for history in child_histories:
            replayer = Replayer(workflows=[TaskWorkflow], plugins=[build_task_graph().plugin])
            failures.append((await replayer.replay_workflow(history)).replay_failure)
        return failures

    failures = _run(replay())
    assert failures == [None, None], failures


# ---------------------------------------------------------------------------------------
# 4. The ledger verifies clean, and the run is deterministic.
# ---------------------------------------------------------------------------------------


def test_the_run_ledger_verifies_and_the_outcome_is_deterministic(tmp_path: Path):
    """The audit trail is a valid hash chain, and two identical runs reach identical outcomes.

    Determinism is asserted on the OBSERVABLE outcome (per-task outcome, turn count, checkpoint refs),
    not on history: the two runs are two executions with their own run ids and timestamps, so their
    histories must differ. What must not differ is what the run DID.
    """

    async def one(repo: Path) -> tuple[Any, Any]:
        runs_root = repo / "runs"
        runs_root.mkdir()
        harness = fakes.build_harness(repo, runs_root)
        tasks = tuple(
            _spec(repo, number, engine="plain", spawn=True, teams=True) for number in (1, 2)
        )
        completion = await _with_worker_async(
            "plain",
            lambda client: _drive(
                client,
                _payload(repo, tasks, engine="plain", parallel=1),
                2,
                workflow_cls=RunWorkflow,
                approval_cls=TaskWorkflowPlain,
            ),
        )
        return completion, harness

    async def _with_worker_async(engine: str, scenario: Any) -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            plugins = [build_task_graph().plugin] if engine == "langgraph" else []
            with activity_executor(max_workers=4) as executor:
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

    async def both() -> tuple[Any, Any]:
        left = await one(fakes.make_repo(tmp_path / "repo-a"))
        right = await one(fakes.make_repo(tmp_path / "repo-b"))
        return left, right

    (left_completion, left_harness), (right_completion, right_harness) = _run(both())

    # The ledger's hash chain, contiguity and per-session event indices all verify.
    for harness in (left_harness, right_harness):
        ledger = harness.ledger(RUN_ID)
        assert ledger.verify() == []
        assert ledger.check_contiguity() == []
        for session_id in {
            record.session_id for record in ledger.records() if record.session_id is not None
        }:
            assert ledger.check_session_contiguity(session_id) == [], session_id

    # Same work, same outcome — the observable shape, which is what a replay or a re-run must preserve.
    # `TaskSummary` carries five scalars and two optional strings (it must stay small enough for a
    # continue-as-new round trip), so the comparison is over THOSE, and the per-task audit refs are
    # compared from the LEDGER — the record a human audits, and the thing a divergent re-run would show
    # up in.
    assert left_completion.outcome == right_completion.outcome == "completed"
    assert [summary.task_id for summary in left_completion.tasks] == [
        summary.task_id for summary in right_completion.tasks
    ]
    for field in ("outcome", "turn_seq", "approved_by", "verify_passed", "review_verdict"):
        assert [getattr(summary, field) for summary in left_completion.tasks] == [
            getattr(summary, field) for summary in right_completion.tasks
        ], field
    # The audit refs are a deterministic function of `(run, task, turn_seq)`, so they must agree. Taken
    # from the ledger rather than from the summary, which deliberately does not carry them.
    assert [entry["ref"] for entry in left_harness.checkpoints(RUN_ID)] == [
        entry["ref"] for entry in right_harness.checkpoints(RUN_ID)
    ]
    assert [entry["turn_seq"] for entry in left_harness.checkpoints(RUN_ID)] == [
        entry["turn_seq"] for entry in right_harness.checkpoints(RUN_ID)
    ]
    # The delegation actually happened in BOTH runs — otherwise "deterministic" would be a statement
    # about two runs that both failed to delegate.
    for harness in (left_harness, right_harness):
        specs = [entry["spec"] for entry in harness.sidecar.sessions if entry.get("spec")]
        assert specs and all(spec.get("enable_agent_teams") is True for spec in specs)


# ---------------------------------------------------------------------------------------
# 5. The gate itself: a forbidden capability is REFUSED, not sent.
# ---------------------------------------------------------------------------------------


def test_the_profile_gate_refuses_a_forbidden_capability():
    """Fail closed in front of the engine's own refusal, and in the engine's own words.

    The engine refuses these in `resolveSessionPolicy`; the durable side refuses them one step earlier
    so an operator learns at the CLI rather than three activities into a run. Asserting the CODE and
    the profile named in the details means a refusal nobody can act on fails this test.
    """
    # Every profile the engine grants currently permits both, so the refusal is exercised against a
    # profile the table does NOT list — the future-proofing case, and the one that must never default
    # to "allowed".
    with pytest.raises(SwarmConfigError) as caught:
        normalize_swarm(
            enable_spawn_agent=True,
            enable_agent_teams=False,
            approval_profile="some_future_profile",
        )
    assert caught.value.code == "swarm_config_invalid"

    # And a non-boolean is refused rather than coerced: the value enters workflow history and then a
    # request body, so "yes" recorded but `true` sent would be a spec that lies about its request.
    with pytest.raises(SwarmConfigError) as caught_bool:
        normalize_swarm(enable_spawn_agent="yes", enable_agent_teams=False)  # type: ignore[arg-type]
    assert caught_bool.value.code == "swarm_config_invalid"

    # Every profile in the mirrored table permits both, which is the engine's current state — asserted
    # so a future narrowing on the engine side shows up here as a table that no longer matches.
    for profile, permissions in PROFILE_SWARM_PERMISSIONS.items():
        assert normalize_swarm(
            enable_spawn_agent=True, enable_agent_teams=True, approval_profile=profile
        ) == (True, True), profile
        assert permissions["enable_spawn_agent"] and permissions["enable_agent_teams"], profile


def test_a_task_spec_cannot_bypass_the_profile_gate(tmp_path: Path):
    """The refusal is at the SPEC boundary, so no code path can build a forbidden spec silently."""
    from codna.durable.contract_primitives import ContractError

    spec = TaskSpec(
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        brief="delegate",
        repo_dir=str(tmp_path),
        enable_spawn_agent="yes",  # type: ignore[arg-type]
    )
    with pytest.raises(ContractError) as caught:
        spec.validate()
    assert caught.value.code == "swarm_config_invalid"


def test_the_session_spec_is_empty_when_nothing_was_asked():
    """`{}`, not `{"enable_spawn_agent": False}`: an absent key is what keeps the request identical."""

    class Ctx:
        enable_spawn_agent = False
        enable_agent_teams = False

    assert swarm_session_spec(Ctx()) == {}

    class Swarming:
        enable_spawn_agent = True
        enable_agent_teams = True

    assert swarm_session_spec(Swarming()) == {
        "enable_spawn_agent": True,
        "enable_agent_teams": True,
    }
