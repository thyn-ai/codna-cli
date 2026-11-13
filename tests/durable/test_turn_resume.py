"""One paid turn per ``(run, task, turn_seq)`` — asserted from the LEDGER, across two attempts.

The ``turn_key`` design promises that a retried activity replays instead of buying a second turn.
That promise only holds while the ENGINE still remembers the session: ``state.turnKeyIndex`` is a map
of session → key → record (``server/state.ts``), so a session re-created after a sidecar restart gives
the same key a fresh, empty index and a genuinely new paid turn.

What is asserted here is the durable side of the guarantee, which must not depend on the engine:

* two attempts at the same ``(run, task, turn_seq)`` leave exactly ONE ``turn.started`` record in
  the ledger and exactly ONE paid turn in the engine;
* the second attempt does not POST at all once a turn id has been recorded for that turn key;
* if the recorded session is gone, the attempt fails loudly with a distinct code rather than
  silently starting a second paid turn;
* the soak detector keeps its teeth: a healthy retry is not a finding, a genuine re-scope still is.

Run WITHOUT the durable extra this module SKIPs (it drives the real activity bodies, which import
``temporalio``); the durable CI job runs it with the extra so a skip is not how it passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

from codna.durable import soak_report as sr  # noqa: E402
from codna.durable.activities import ActivityError, run_cline_turn  # noqa: E402
from codna.durable.contracts import TaskSpec, TaskContext  # noqa: E402
from codna.durable.ledger import LedgerRecord  # noqa: E402
from codna.durable.state import TaskState  # noqa: E402
from tests.durable import fakes  # noqa: E402


def _context(repo: Path):
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="make hello return goodbye",
        repo_dir=str(repo),
        verify_commands=("true",),
        engine="plain",
    )
    spec.validate()
    return TaskContext.from_spec(spec), spec


def _turn_starts(harness: fakes.Harness) -> list[LedgerRecord]:
    return [record for record in harness.ledger().records() if record.kind == "turn.started"]


def test_two_attempts_at_one_turn_seq_are_one_ledger_record_and_one_paid_turn(tmp_path: Path):
    """The property, read from the ledger a human audits — not from a call counter."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    # The SAME request object shape a Temporal retry replays: the pre-turn state, untouched.
    second = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    starts = _turn_starts(harness)
    assert len(starts) == 1, [record.payload for record in starts]
    assert len(harness.sidecar.turns) == 1, harness.sidecar.turns
    assert first.details["turn_id"] == second.details["turn_id"]
    # One turn, one key, one session — so the detector has nothing to say about it.
    assert sr.check_duplicate_turn_key(harness.ledger().records()) == []


def test_a_retried_turn_does_not_post_once_a_turn_id_is_recorded(tmp_path: Path):
    """The second attempt must ATTACH to the recorded turn, not ask the engine for another one."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    posts_after_first = len(harness.sidecar.prompts)
    second = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert len(harness.sidecar.prompts) == posts_after_first, harness.sidecar.prompts
    assert second.details["turn_id"] == first.details["turn_id"]
    assert second.details["resumed"] is True


def _interrupted_turn(harness: fakes.Harness, context, spec) -> str:
    """Start a turn that gets recorded but never finishes — the crash-a-worker shape.

    The activity raises ``activity_turn_failed`` on the terminal failure, so no ``turn.finished`` is
    written: the ledger holds a ``turn.started`` and nothing else, which is exactly the state a
    retry after a mid-turn crash sees.
    """
    harness.sidecar.terminal = fakes.TERMINAL_FAILED
    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    assert excinfo.value.code == "activity_turn_failed", excinfo.value.as_dict()
    starts = _turn_starts(harness)
    assert len(starts) == 1, [record.payload for record in starts]
    return str(starts[0].payload["turn_id"])


def test_a_kill_between_the_post_and_the_ledger_record_still_bills_once(tmp_path: Path):
    """The one window the ledger cannot cover: the process dies after the POST, before the append.

    Nothing durable recorded the turn, so the retry MUST POST again — and the guarantee then rests
    entirely on the engine's own replay index. This test therefore asserts the narrower, honest
    claim: within one session the second POST is answered as a replay (200 + the existing turn id),
    so exactly one turn is paid for even in the window where the durable side has no record yet.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    # Simulate the kill: the engine accepted a turn, then the process died before the ledger append.
    first_session = context.turn_request("implement it", spec.initial_state())
    state = TaskState.from_payload(first_session.state)
    session_id = harness.sidecar.create_session(working_dir=str(repo), spec={"owner": "codna-run-1"}).session_id
    turn_key = f"{state.run_id}:{state.task_id}:{state.turn_seq + 1}"
    start = harness.sidecar.start_turn(session_id, "implement it", turn_key=turn_key)
    assert len(harness.sidecar.turns) == 1
    assert harness.ledger().records() == [], "the crash left no durable record, by construction"

    # The retry carries the same pre-turn state, so it re-derives the same key and session.
    resumed_state = TaskState.from_payload(first_session.state).started_turn(
        session_id=session_id, turn_id=start.turn_id
    )
    replay = harness.sidecar.start_turn(
        session_id, "implement it", turn_key=f"{resumed_state.run_id}:{resumed_state.task_id}:1"
    )
    assert replay.replayed is True
    assert replay.turn_id == start.turn_id
    assert len(harness.sidecar.turns) == 1, "the replay must not have started a second turn"


def test_an_unfinished_turn_is_reattached_by_its_recorded_id(tmp_path: Path):
    """The in-flight retry: same turn, no second POST, one start record.

    This is the case the engine's replay index cannot cover after a restart, so the durable side
    must re-attach by the turn id the ledger recorded before the first stream was read.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    turn_id = _interrupted_turn(harness, context, spec)
    posts_after_crash = len(harness.sidecar.prompts)

    # The turn recovers (a transient engine failure), and the retry finishes it.
    harness.sidecar.terminal = fakes.TERMINAL_OK
    recovered = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert recovered.details["turn_id"] == turn_id
    assert recovered.details["resumed"] is True
    assert len(harness.sidecar.prompts) == posts_after_crash, harness.sidecar.prompts
    assert len(_turn_starts(harness)) == 1
    assert len(harness.sidecar.turns) == 1, harness.sidecar.turns
    assert sr.check_duplicate_turn_key(harness.ledger().records()) == []


def test_an_unfinished_turn_whose_session_is_gone_fails_instead_of_buying_a_new_turn(tmp_path: Path):
    """A lost session must not turn a retry into a second paid turn.

    The engine's replay index is per session, so a re-created session would accept the same
    ``turn_key`` as new work. That is exactly the case ``turn_key_rescoped`` exists to report, and
    the durable side must refuse it rather than produce it.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    turn_id = _interrupted_turn(harness, context, spec)
    posts_after_crash = len(harness.sidecar.prompts)
    # The sidecar restarted: it no longer holds the session, so `resume_from` cannot be honoured.
    harness.sidecar.live_sessions.clear()
    harness.sidecar.terminal = fakes.TERMINAL_OK

    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert excinfo.value.code == "activity_turn_unobservable", excinfo.value.as_dict()
    assert excinfo.value.details["turn_id"] == turn_id
    # Nothing was bought: no new POST, no second turn, no second start record.
    assert len(harness.sidecar.prompts) == posts_after_crash, harness.sidecar.prompts
    assert len(harness.sidecar.turns) == 1, harness.sidecar.turns
    assert len(_turn_starts(harness)) == 1


def test_the_second_turn_runs_when_the_engine_tore_the_session_down(tmp_path: Path):
    """A durable task's session must SURVIVE its own first turn, and the second turn must actually run.

    The defect this pins, measured against the real sidecar: the SDK finalizes a non-interactive session
    at the end of turn 1 (`finalizeSingleRun` → `shutdownSession` → `sessions.delete`), so the `plan`
    turn completed and the `implement` turn's POST answered
    ``SessionNotFoundError: session not found`` — with NO restart involved. That is the error the soak
    reported and read as a restart symptom.

    Two assertions, and the second is the one that matters:

    * ``resume_from`` was honoured (the same id, not a silently new one), and
    * the second turn actually RAN — read from the ledger and the engine, not from "create returned 200".

    The fake models the teardown itself (``FakeSidecar._finalize_if_one_shot``), so this test is not
    passing because the fake is permissive: with ``multi_turn`` absent from the create spec the second
    ``start_turn`` raises exactly what the real engine raises.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    session_id = first.state["session_id"]

    # THE ASSERTION THAT FAILS TODAY. Before the fix the create carried no `multi_turn`, the SDK tore the
    # session down at the end of turn 1, and this raised
    # `SidecarClientError: 404 session_not_found: session not found: …` — the soak's exact error, reached
    # here with no restart and no network.
    second = run_cline_turn(context.turn_request("implement it again", first.state))

    # …and it ran because the create DECLARED the session multi-turn, at the one place one is created.
    assert harness.sidecar.sessions[0]["spec"]["multi_turn"] is True
    assert session_id not in harness.sidecar.finalized

    # The second turn RAN: a new turn id, in the SAME session, and both turns are in the engine's record.
    assert second.details["turn_id"] != first.details["turn_id"]
    assert second.details["session_id"] == session_id
    assert second.state["session_id"] == session_id
    assert second.state["turn_seq"] == 2
    assert [turn.session_id for turn in harness.sidecar.turns] == [session_id, session_id]

    # …and the ledger agrees: two starts, two finishes, one session, distinct keys.
    starts = _turn_starts(harness)
    assert len(starts) == 2, [record.payload for record in starts]
    assert {record.payload["turn_key"] for record in starts} == {
        "run-1:task-1:1",
        "run-1:task-1:2",
    }
    assert {record.payload["session_id"] for record in starts} == {session_id}
    assert sr.check_duplicate_turn_key(harness.ledger().records()) == []


def test_a_restart_moves_the_conversation_to_a_new_session_and_still_runs(tmp_path: Path):
    """After a REAL sidecar restart the run continues on a NEW session, and says so in the ledger.

    The engine cannot honour `resume_from` once its process is gone, but it does not lose the run either:
    it recovers the persisted transcript and seeds a fresh session from it, answering `resumed: true` with
    a DIFFERENT id and `resume_from_ignored: false`. Keying the "the conversation moved" notice off
    `resume_from_ignored` therefore missed precisely the case a restart produces, so the test is written
    against the id instead — plus an assertion that the turn really ran, since a silently empty
    conversation would otherwise look identical.
    """
    import json

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    old_session = first.state["session_id"]

    # The sidecar is killed and comes back empty-handed: it has never heard of this id.
    harness.sidecar.forget_session(old_session)

    second = run_cline_turn(context.turn_request("implement it again", first.state))

    new_session = second.state["session_id"]
    assert new_session != old_session
    # The turn RAN in the new session — the engine recorded it there, not nowhere.
    assert second.details["session_id"] == new_session
    assert harness.sidecar.turns[-1].session_id == new_session
    assert [turn.session_id for turn in harness.sidecar.turns] == [old_session, new_session]

    # …and the move is REPORTED, not left for a reader to infer from two different session ids. The event
    # names both ids and says whether the transcript actually carried over, because "continued elsewhere"
    # and "the conversation was lost" must not look the same in a run's record.
    notices = [
        json.loads(line)
        for line in harness.notifications
        if json.loads(line).get("event") == "turn.session_restarted"
    ]
    assert len(notices) == 1, harness.notifications
    assert notices[0]["previous_session_id"] == old_session
    assert notices[0]["session_id"] == new_session
    assert notices[0]["resumed_from_history"] is True
    assert notices[0]["resume_from_ignored"] is False


def test_multi_turn_keeps_turn_keys_per_session_and_never_reuses_an_id(tmp_path: Path):
    """The interaction the fix must NOT break: two turns, one session, two DISTINCT keys and turn ids.

    `turn_key` idempotency is scoped PER SESSION on the engine side (`turnKeyIndex: Map<sessionId, …>`).
    Keeping a session alive across turns must therefore not let the second turn land on the first one's key:
    that would replay turn 1 instead of running turn 2, and the run would silently do no work while
    reporting success twice. Asserted from the engine's own record, not from the return values.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    first = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    second = run_cline_turn(context.turn_request("implement it again", first.state))
    # The third turn exists to prove the keys keep advancing PAST two; its return value is not the subject.
    third = run_cline_turn(context.turn_request("implement once more", second.state))
    assert third.state["turn_seq"] == 3

    turns = harness.sidecar.turns
    assert len(turns) == 3, turns
    # ONE session for all three: this is the continuity the fix buys.
    assert {turn.session_id for turn in turns} == {first.state["session_id"]}
    # …and THREE distinct keys/ids, so no turn replayed another.
    assert len({turn.turn_key for turn in turns}) == 3, [turn.turn_key for turn in turns]
    assert len({turn.turn_id for turn in turns}) == 3, [turn.turn_id for turn in turns]
    # Each POST reached the engine exactly once — no key was retried, so nothing is double-charged.
    assert len(harness.sidecar.prompts) == 3, harness.sidecar.prompts
    assert sr.check_duplicate_turn_key(harness.ledger().records()) == []


def test_a_one_shot_session_is_torn_down_so_the_fake_cannot_hide_the_defect(tmp_path: Path):
    """The negative control: WITHOUT ``multi_turn`` the same flow fails the way the real engine fails.

    Without this, the test above could pass because the fake accepts every session id forever — which is
    exactly how 741 durable tests stayed green while a real soak could not get past turn one. This asserts
    the fake's rule is real, and names the engine's own error code so a regression reads as itself.
    """
    from codna.durable.sidecar_client import SidecarClientError

    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, repo / "runs")
    harness.runs_root.mkdir(exist_ok=True)

    # A session created the way a ONE-SHOT caller creates it: no multi_turn.
    session = harness.sidecar.create_session(
        working_dir=str(repo), spec={"owner": "codna-run-1", "task_kind": "fix"}
    )
    first = harness.sidecar.start_turn(session.session_id, "implement it")
    list(harness.sidecar.stream_turn(session.session_id, first.turn_id))

    assert session.session_id in harness.sidecar.finalized
    with pytest.raises(SidecarClientError) as excinfo:
        harness.sidecar.start_turn(session.session_id, "again")
    assert excinfo.value.code == "session_not_found"
    assert excinfo.value.status == 404


def test_a_real_resumed_turn_leaves_a_clean_report(tmp_path: Path):
    """The detector over the LEDGER a real retry produced: no findings, and the resume is named.

    Both halves matter. Findings would mean the fix merely moved the symptom; no trace of the resume
    would mean an auditor reading the run could not tell that two attempts happened at all.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    _interrupted_turn(harness, context, spec)
    harness.sidecar.terminal = fakes.TERMINAL_OK
    run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    records = harness.ledger().records()
    # `allow_open_turns` because the first attempt deliberately never finished; the point here is the
    # duplicate/rescope checks, which must stay silent about a healthy resume.
    report = sr.build_report(run_id="run-1", records=records, allow_open_turns=True)
    assert list(report.findings) == [], [finding.render() for finding in report.findings]

    resumes = [
        record.payload
        for record in records
        if record.kind == "note" and record.payload.get("event") == "turn.resumed_from_ledger"
    ]
    assert len(resumes) == 1, [record.payload for record in records if record.kind == "note"]
    assert resumes[0]["source"] == "turn.started"
    assert resumes[0]["turn_id"] == _turn_starts(harness)[0].payload["turn_id"]


def test_the_resume_floor_is_scoped_to_the_recorded_session(tmp_path: Path):
    """A resume must ask from the recorded SESSION's high-water mark, never across sessions.

    Event indices are per session. A floor computed across every session in the ledger could sit
    above the resumed turn's real position and silently skip events — the one error a resume must
    never make — so this pins the scoping with a ledger that holds a LATER index in another session.
    """
    from codna.durable.activity_support import _scan_ledger

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    ledger = harness.ledger()

    ledger.append(
        "turn.started",
        task_id="task-1",
        payload={"turn_id": "turn-a", "turn_key": "run-1:task-1:1", "session_id": "session-a"},
    )
    ledger.append_events(
        [{"event_index": 3, "event_type": "agent.text", "payload": {}}],
        task_id="task-1",
        session_id="session-a",
    )
    # A LATER index in a DIFFERENT session: it must not raise the floor for session-a.
    ledger.append_events(
        [{"event_index": 99, "event_type": "agent.text", "payload": {}}],
        task_id="task-1",
        session_id="session-b",
    )

    recorded, completion = _scan_ledger(ledger, "run-1:task-1:1")
    assert recorded is not None
    assert completion is None
    assert recorded.session_id == "session-a"
    assert recorded.high_water == 3, recorded


def test_the_loop_verdict_is_reported_from_the_single_scan(tmp_path: Path):
    """The verdict the retry reports is the one `_scan_ledger` carried out of its own pass.

    Asserts the CONTRACT directly, so a future refactor that reintroduces a second scan cannot
    silently drop the verdict: the fields a caller branches on survive the fold into one traversal.
    """
    from codna.durable.activity_support import _scan_ledger
    from codna.durable.turn_guard import TOOL_LOOP_EVENT

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    ledger = harness.ledger()

    ledger.append(
        "turn.started",
        task_id="task-1",
        payload={"turn_id": "turn-a", "turn_key": "run-1:task-1:1", "session_id": "session-a"},
    )
    ledger.append_events(
        [{"event_index": 3, "event_type": "agent.text", "payload": {}}],
        task_id="task-1",
        session_id="session-a",
    )
    # Another turn's verdict must not be adopted: only this turn's is reported.
    ledger.append(
        "note",
        task_id="task-1",
        payload={"event": TOOL_LOOP_EVENT, "turn_id": "turn-other", "tool": "other", "count": 9},
    )
    ledger.append(
        "note",
        task_id="task-1",
        payload={
            "event": TOOL_LOOP_EVENT,
            "turn_id": "turn-a",
            "tool": "editor",
            "error": "old_text required",
            "count": 7,
        },
    )

    recorded, completion = _scan_ledger(ledger, "run-1:task-1:1")
    assert recorded is not None and completion is None
    assert recorded.loop_failure is not None, "the verdict must survive the single pass"
    assert recorded.loop_failure["tool"] == "editor", recorded.loop_failure
    assert recorded.loop_failure["count"] == 7, recorded.loop_failure


def test_a_turn_with_no_verdict_reports_none(tmp_path: Path):
    """The common case: a recorded turn that was never judged is not given a fabricated verdict."""
    from codna.durable.activity_support import _scan_ledger
    from codna.durable.turn_guard import TOOL_LOOP_EVENT

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    ledger = harness.ledger()

    ledger.append(
        "turn.started",
        task_id="task-1",
        payload={"turn_id": "turn-a", "turn_key": "run-1:task-1:1", "session_id": "session-a"},
    )
    ledger.append(
        "note",
        task_id="task-1",
        payload={"event": TOOL_LOOP_EVENT, "turn_id": "turn-other", "tool": "editor", "count": 7},
    )

    recorded, _ = _scan_ledger(ledger, "run-1:task-1:1")
    assert recorded is not None
    assert recorded.loop_failure is None, recorded.loop_failure


def test_the_detector_still_fires_for_a_genuinely_rescoped_key() -> None:
    """The healthy retry is exempt; a different session with a new turn id is NOT.

    Weakening the detector to make the retry pass would hide the failure it exists for, so the
    negative case is asserted next to the positive one.
    """
    healthy = [
        {"seq": 1, "kind": "turn.started", "payload": {"turn_id": "turn-a", "turn_key": "k", "session_id": "s1"}},
        {"seq": 2, "kind": "turn.started", "payload": {"turn_id": "turn-a", "turn_key": "k", "session_id": "s1"}},
    ]
    assert sr.check_duplicate_turn_key(healthy) == []

    rescoped = [
        {"seq": 1, "kind": "turn.started", "payload": {"turn_id": "turn-a", "turn_key": "k", "session_id": "s1"}},
        {"seq": 2, "kind": "turn.started", "payload": {"turn_id": "turn-b", "turn_key": "k", "session_id": "s2"}},
    ]
    findings = sr.check_duplicate_turn_key(rescoped)
    assert [finding.code for finding in findings] == [sr.TURN_KEY_RESCOPED]


def test_the_resume_floor_steps_below_a_trailing_terminal_event(tmp_path: Path):
    """`?after=` is EXCLUSIVE, so a floor sitting ON the terminal would skip the ending forever.

    MEASURED failure this pins: an implement turn appended `session.turn_aborted` (its terminal) and
    then died one line before writing `turn.finished`. The retry re-attached with
    `?after=<terminal index>` — asking for everything AFTER the event that ends the turn. The sidecar
    had nothing more to send, so the stream replayed nothing, followed no live events, and blocked
    until the 60-minute activity ceiling while heartbeating happily: the run looked alive forever and
    its ledger never grew.

    A floor of `index - 1` replays exactly that one terminal, so `_stream` sees the ending and the
    retry fails by name instead of hanging.
    """
    from codna.durable.activity_support import _scan_ledger

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    ledger = harness.ledger()

    ledger.append(
        "turn.started",
        task_id="task-1",
        payload={"turn_id": "turn-a", "turn_key": "run-1:task-1:1", "session_id": "session-a"},
    )
    ledger.append_events(
        [
            {"event_index": 3, "event_type": "agent.text", "payload": {}},
            {"event_index": 4, "event_type": "session.turn_aborted", "payload": {}},
        ],
        task_id="task-1",
        session_id="session-a",
    )

    recorded, completion = _scan_ledger(ledger, "run-1:task-1:1")
    assert recorded is not None
    # No `turn.finished`: this is the crash window, not a completed turn.
    assert completion is None
    assert recorded.high_water == 3, (
        "the floor must sit BELOW the trailing terminal (4) so the replay can deliver it; "
        f"got {recorded.high_water}"
    )


def test_the_resume_floor_is_untouched_when_the_terminal_is_not_last(tmp_path: Path):
    """A terminal with later events is not the floor: the stream was already reading past it."""
    from codna.durable.activity_support import _scan_ledger

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    ledger = harness.ledger()

    ledger.append(
        "turn.started",
        task_id="task-1",
        payload={"turn_id": "turn-a", "turn_key": "run-1:task-1:1", "session_id": "session-a"},
    )
    ledger.append_events(
        [
            {"event_index": 4, "event_type": "session.turn_aborted", "payload": {}},
            {"event_index": 5, "event_type": "status", "payload": {}},
        ],
        task_id="task-1",
        session_id="session-a",
    )

    recorded, completion = _scan_ledger(ledger, "run-1:task-1:1")
    assert recorded is not None and completion is None
    assert recorded.high_water == 5, recorded


# ── THE ABORTED TURN MUST SAY WHY TO THE DURABLE CALLER ──────────────────────────────────────────
#
# The Python client is the last consumer in the chain the defect was measured along. Its terminal
# branch raised `activity_turn_failed` with only the terminal NAME, so a write-scope guard refusing a
# write (the fail-closed arm) and a user cancellation produced an identical failure: `terminal:
# "session.turn_aborted"` and nothing else. The engine now carries `abort: {kind, reason}`; these pin
# that it reaches the error a durable caller actually receives.
def test_an_aborted_turn_reports_the_guard_reason_to_the_durable_caller(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    harness.sidecar.terminal = fakes.TERMINAL_ABORTED
    harness.sidecar.abort_on_terminal = {
        "kind": "control_stop",
        "reason": "write-scope violation: this turn is read-only, so 'editor' is refused",
    }

    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    details = excinfo.value.details
    assert excinfo.value.code == "activity_turn_failed"
    assert details["terminal"] == "session.turn_aborted"
    # The refusal is now legible without replaying the stream: kind, quoted rule, and the promoted
    # `reason` field every ActivityError consumer already reads.
    assert details["kind"] == "control_stop"
    assert "write-scope violation" in details["reason"]
    assert "write-scope violation" in details["abort"]["reason"]


def test_a_cancelled_turn_is_not_reported_as_a_policy_refusal(tmp_path: Path):
    """The distinction the whole fix exists for: a user pressing stop must not read as a guard refusal."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    harness.sidecar.terminal = fakes.TERMINAL_ABORTED
    harness.sidecar.abort_on_terminal = {"kind": "caller_abort", "reason": "user cancelled"}

    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    details = excinfo.value.details
    assert details["kind"] == "caller_abort"
    assert details["reason"] == "user cancelled"
    assert "control_stop" not in json.dumps(details)
    assert "write-scope" not in json.dumps(details)


def test_an_aborted_turn_with_no_abort_block_still_reports_the_terminal(tmp_path: Path):
    """An older engine sends no `abort` block. The failure keeps its old shape and gains no keys."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)

    harness.sidecar.terminal = fakes.TERMINAL_ABORTED

    with pytest.raises(ActivityError) as excinfo:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    details = excinfo.value.details
    assert details["terminal"] == "session.turn_aborted"
    assert "kind" not in details
    assert "reason" not in details
