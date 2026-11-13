"""A turn that repeats one deterministic tool failure FAILS FAST with a coded reason.

── THE MEASURED DEFECT ────────────────────────────────────────────────────────────────────────────

Run `run-20261112T232923Z` (a small calculator app, started from the packaged app's New task sheet)
died with the deadline rather than with its cause:

    seq 3131  session.turn_failed  {"error": "Error: turn exceeded timeout budget of 600000ms",
                                    "turn_id": "turn_session_..._2"}

`verify` and `review` never ran and the run's outcome was `failed`. The ledger shows what the ten
minutes bought: 11 iterations, 10 tool calls, and EVERY call was the same `editor` call failing the
same way —

    seq  527  agent.content_start  {"contentType": "tool", "toolName": "editor",
                                    "input": {"new_text": "<!DOCTYPE html>…"}}
    seq  529  agent.content_end    {"contentType": "tool", "toolName": "editor",
                                    "output": {"error": "Editor operation failed: Parameter
                                              `old_text` is required when editing an existing file
                                              without `insert_line`", "success": false}}

— repeated verbatim at 567, 625, 661, 743, 797, 861, 937, 1027 and 1123.

The control is the FIRST app, which completed: `run-20261112T230703Z`'s implement turn ran 11
iterations in 54 s (seq 274 → seq 888) and verified clean. So the budget was not too small for the
work; one turn was repeating a call that could not succeed. Waiting out the deadline replaces the
real cause (`old_text` required) with a number, and costs the run its verify and review steps.

These tests drive the REAL `run_cline_turn` body over the harness's sidecar seam, with the events
scripted from the shape above, so what is asserted is the durable side's own behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")
pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")

from codna.durable.activities import ActivityError, run_cline_turn  # noqa: E402
from codna.durable.contracts import TaskContext, TaskSpec  # noqa: E402
from codna.durable.turn_guard import (  # noqa: E402
    MAX_IDENTICAL_TOOL_FAILURES,
    TurnGuard,
)
from tests.durable import fakes  # noqa: E402

#: The error the measured run repeated, verbatim from ledger seq 529.
EDITOR_ERROR = (
    "Editor operation failed: Parameter `old_text` is required when editing an existing file "
    "without `insert_line`"
)


def _tool_failure(tool: str = "editor", error: str = EDITOR_ERROR) -> tuple[str, dict[str, object]]:
    """One finished FAILED tool call, in the shape the engine emits (`execution.ts` → `artifacts.ts`)."""
    return (
        "agent.content_end",
        {
            "contentType": "tool",
            "toolName": tool,
            "output": {"error": error, "query": f"edit:{tool}", "success": False},
        },
    )


def _tool_success(tool: str = "editor") -> tuple[str, dict[str, object]]:
    """One finished SUCCESSFUL tool call, which must clear any failure streak."""
    return (
        "agent.content_end",
        {
            "contentType": "tool",
            "toolName": tool,
            "output": {"query": f"edit:{tool}", "result": "File created successfully", "success": True},
        },
    )


def _batch_failure(query: str, error: str) -> tuple[str, dict[str, object]]:
    """A `read_files`/`search_codebase` result: the engine's output is a LIST of per-path dicts.

    Transcribed from `run-20261112T222825Z` seq 316. A reader that only understood a MAPPING output
    saw no failure at all here, which is one of the ways the shipped guard was blind to a loop.
    """
    return (
        "agent.content_end",
        {
            "contentType": "tool",
            "toolName": "read_files",
            "output": [{"error": error, "query": query, "result": "", "success": False}],
        },
    )


def _mcp_failure(text: str) -> tuple[str, dict[str, object]]:
    """An MCP tool result: `{isError: true, content: [{type, text}]}` and no `success` key.

    Transcribed from `run-20261112T223954Z` seq 1683 (`browser__browser_run_code_unsafe`). This shape
    carries its error in NEITHER `success` NOR a string `error`, so it too was invisible.
    """
    return (
        "agent.content_end",
        {
            "contentType": "tool",
            "toolName": "browser__browser_run_code_unsafe",
            "output": {"content": [{"text": text, "type": "text"}], "isError": True},
        },
    )


def _context(repo: Path) -> tuple[TaskContext, TaskSpec]:
    spec = TaskSpec(
        run_id="run-1",
        task_id="task-1",
        task_number=1,
        brief="build a calculator",
        repo_dir=str(repo),
        verify_commands=("true",),
        engine="plain",
    )
    spec.validate()
    return TaskContext.from_spec(spec), spec


def _harness(tmp_path: Path) -> tuple[fakes.Harness, TaskContext, TaskSpec]:
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = repo / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    context, spec = _context(repo)
    return harness, context, spec


# -- the measured defect, end to end through the real turn activity -----------------------


def test_a_repeated_identical_tool_failure_fails_the_turn_by_name(tmp_path: Path):
    """The measured run's event stream must END the turn with `activity_turn_tool_loop`.

    Before the guard existed this turn ran to its terminal and reported success (the fake's default),
    which is exactly the durable side's contribution to the 600 s death: nothing here noticed ten
    identical failures, so only the deadline could end the turn.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]

    with pytest.raises(ActivityError) as caught:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    error = caught.value
    assert error.code == "activity_turn_tool_loop", error.as_dict()
    # The FAILURE names what actually went wrong, which is the whole point: the deadline named a
    # number, and the reason `verify` never ran was this editor parameter.
    assert error.details["tool"] == "editor", error.as_dict()
    assert error.details["count"] == MAX_IDENTICAL_TOOL_FAILURES, error.as_dict()
    assert "old_text" in str(error.details["error"])
    assert "old_text" in str(error)
    # The engine is TOLD to stop: judging a turn unsalvageable and letting it keep generating is
    # paying for tokens nobody will read.
    assert harness.sidecar.aborts == [harness.session_ids()[0]], harness.sidecar.aborts


def test_a_loop_shorter_than_the_threshold_is_left_alone(tmp_path: Path):
    """Two identical failures is a plausible retry, not proof. The turn completes."""
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [_tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES - 1)]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details
    assert harness.sidecar.aborts == [], harness.sidecar.aborts


def test_a_success_between_failures_clears_the_streak(tmp_path: Path):
    """`fail, succeed, fail, succeed` is an ordinary edit sequence, and must not fail a turn.

    This is the shape the FIRST app's run actually had (a tool that failed once and then worked), and
    the reason the counter clears on success rather than counting failures in total.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure(),
        _tool_success(),
        _tool_failure(),
        _tool_success(),
        _tool_failure(),
        _tool_success(),
        _tool_failure(),
    ]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details


def test_a_toolset_refusal_is_not_counted_as_a_loop(tmp_path: Path):
    """A policy refusal is deterministic but says the TOOLSET is wrong, and a model may recover.

    MEASURED in the first app's run: `graph_query` was refused with
    `Tool "graph_query" is disabled by policy`, the model chose another tool, and the turn completed
    and verified clean. Counting these would fail runs that are making progress.
    """
    refusal = 'Tool "graph_query" is disabled by policy'
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure("graph_query", refusal) for _ in range(MAX_IDENTICAL_TOOL_FAILURES + 2)
    ]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details


def test_a_different_error_restarts_the_count(tmp_path: Path):
    """Distinct errors are distinct information; the model deserves the iteration to react."""
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure(error=f"failure {index}") for index in range(MAX_IDENTICAL_TOOL_FAILURES + 2)
    ]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details


# -- the guard as a pure unit, so its rule is pinned rather than only exercised ------------


def test_the_guard_trips_on_the_threshold_exactly_and_reports_the_count():
    guard = TurnGuard()
    flagged = [
        guard.observe(event_type, payload)
        for event_type, payload in [_tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES)]
    ]
    assert flagged[: MAX_IDENTICAL_TOOL_FAILURES - 1] == [None] * (MAX_IDENTICAL_TOOL_FAILURES - 1)
    assert flagged[-1] is not None
    assert flagged[-1].tool_name == "editor"
    assert flagged[-1].count == MAX_IDENTICAL_TOOL_FAILURES


def test_the_guard_ignores_events_that_are_not_finished_tool_calls():
    """Total on every other stream event: a partial frame must not be read as a failure."""
    guard = TurnGuard()
    for event_type, payload in (
        ("agent.text", {"n": 1}),
        ("agent.content_start", {"contentType": "tool", "toolName": "editor"}),
        ("agent.content_end", {"contentType": "text", "text": "thinking"}),
        ("agent.content_end", {"contentType": "tool"}),
        ("agent.usage", {"inputTokens": 1}),
        ("session.turn_completed", {"usage": {}}),
    ):
        assert guard.observe(event_type, payload) is None


def test_the_guard_refuses_an_unusable_threshold():
    for value in (0, -1):
        with pytest.raises(ValueError):
            TurnGuard(max_identical_failures=value)


def test_a_retried_activity_reports_the_LOOP_not_the_abort_it_caused(tmp_path: Path):
    """The verdict survives the retry, because the abort would otherwise become the reported cause.

    An activity retry re-reads the ledger and re-attaches to the turn it finds. This turn was ABORTED
    by the guard, so without the recorded verdict the retry would fail with `activity_turn_failed`
    ("did not complete: session.turn_aborted") — replacing "the editor call was wrong" with "the turn
    stopped", which is the very substitution this fix exists to remove. Asserted end to end: the
    FIRST attempt aborts and raises, the SECOND (a genuine replay of the same request) reports the
    same code and buys no second turn.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]
    request = context.turn_request("implement it", spec.initial_state())

    with pytest.raises(ActivityError) as first:
        run_cline_turn(request)
    assert first.value.code == "activity_turn_tool_loop", first.value.as_dict()
    turns_after_first = len(harness.sidecar.turns)

    # The state a REAL aborted turn is left in: the engine emits its ABORTED terminal and nothing else.
    # The scripted failing calls are cleared because they already happened — re-emitting them would let
    # the guard re-trip on the replay and hide whether the LEDGER is what preserved the verdict.
    harness.sidecar.scripted_events = []
    harness.sidecar.terminal = "session.turn_aborted"
    harness.sidecar.abort_on_terminal = {"kind": "control_stop", "reason": "aborted by the guard"}

    with pytest.raises(ActivityError) as second:
        run_cline_turn(request)

    assert second.value.code == "activity_turn_tool_loop", second.value.as_dict()
    assert second.value.details["tool"] == "editor", second.value.as_dict()
    assert second.value.details["count"] == MAX_IDENTICAL_TOOL_FAILURES
    # And it did NOT buy a second turn: the loop would simply repeat.
    assert len(harness.sidecar.turns) == turns_after_first, harness.sidecar.turns


# -- A LOOP MUST NOT ESCAPE, AND PROGRESS MUST NOT BE MISTAKEN FOR ONE --------------------
#
# The guard decides when a turn is KILLED, so both directions are correctness bugs. Each test below
# is one of them, and each was verified FAILING against the shipped detector before the fix.


def test_a_loop_that_alternates_between_two_bad_calls_is_still_caught(tmp_path: Path):
    """`A, B, A, B, …` is a runaway even though no two consecutive failures are equal.

    MEASURED escape in the shipped detector: it held ONE "last failure" slot, so every interleaved
    failure reset it. `run-20261108T222614Z-stall` alternated a failing `editor` call with a
    policy-refused `graph_query` and never tripped, however long it went on. The count is per
    (tool, error) now, so interleaving cannot launder it.
    """
    harness, context, spec = _harness(tmp_path)
    other = _tool_failure("read_files", "Error reading file: ENOENT: no such file or directory")
    harness.sidecar.scripted_events = [
        event
        for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
        for event in (_tool_failure(), other)
    ]

    with pytest.raises(ActivityError) as caught:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert caught.value.code == "activity_turn_tool_loop", caught.value.as_dict()
    assert caught.value.details["tool"] == "editor", caught.value.as_dict()


def test_a_changing_suffix_on_a_constant_error_does_not_reset_the_count(tmp_path: Path):
    """The same failing call whose message gains a volatile tail is still the same failing call.

    The editor reports the directory it could not create, which is none of the model's business and
    changes on every retry (`EEXIST … mkdir '<path>'`, measured at seq 1385/1415/1441/1481 of
    `run-20261112T223954Z`). Comparing raw text let that call escape detection forever.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure(
            error=(
                "Editor operation failed: EEXIST: file already exists, mkdir "
                f"'/private/tmp/run/tests-{index}' (request 6f1c{index:04d}-aa)"
            )
        )
        for index in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]

    with pytest.raises(ActivityError) as caught:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert caught.value.code == "activity_turn_tool_loop", caught.value.as_dict()
    assert caught.value.details["count"] == MAX_IDENTICAL_TOOL_FAILURES


def test_a_model_that_recovers_after_the_threshold_minus_one_finishes(tmp_path: Path):
    """The whole point of the threshold: a model that learns from repeated failures must not be killed.

    MEASURED, and the reason the threshold is not three: `run-20261112T223954Z`'s implement turn
    failed the same edit three times, then READ the file, learned the argument it was missing and
    edited it successfully (seq 1315). This is that shape, one failure short of the guard.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        *[_tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES - 1)],
        _tool_success(),
    ]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details
    assert harness.sidecar.aborts == [], harness.sidecar.aborts


def test_a_list_shaped_batch_failure_is_counted(tmp_path: Path):
    """`read_files` reports per-path results as a LIST; an all-failed batch is a failed call.

    Before this was read, the shipped detector saw `output` as a non-Mapping and called it "not a
    failure" — so a loop of batch reads was invisible however long it ran.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _batch_failure("src/app.py", "Error reading file: ENOENT: no such file or directory")
        for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]

    with pytest.raises(ActivityError) as caught:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert caught.value.code == "activity_turn_tool_loop", caught.value.as_dict()
    assert caught.value.details["tool"] == "read_files", caught.value.as_dict()


def test_a_batch_with_one_success_is_progress_and_never_counts(tmp_path: Path):
    """A batch that read some of what it asked for is progress, so the count must not accrue."""
    partial = (
        "agent.content_end",
        {
            "contentType": "tool",
            "toolName": "read_files",
            "output": [
                {"error": "Error reading file: ENOENT", "query": "missing.py", "result": "", "success": False},
                {"query": "present.py", "result": "1 | x = 1", "success": True},
            ],
        },
    )
    guard = TurnGuard()
    assert [guard.observe(*partial) for _ in range(20)] == [None] * 20


def test_an_mcp_isError_result_is_counted(tmp_path: Path):
    """An MCP tool reports failure as `{isError: true, content: [...]}`, with no `success` key.

    MEASURED shape (`run-20261112T223954Z` seq 1683). Unread, it made every MCP loop invisible.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _mcp_failure("### Error\nReferenceError: require is not defined")
        for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]

    with pytest.raises(ActivityError) as caught:
        run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert caught.value.code == "activity_turn_tool_loop", caught.value.as_dict()
    assert "require is not defined" in str(caught.value.details["error"])


def test_a_toolset_refusal_between_failures_neither_extends_nor_clears_the_count():
    """A policy refusal is not evidence about the call, in either direction.

    It must not be counted (it is the toolset that is wrong) and must not be treated as a success
    (nothing about the tree changed). The measured todo run recovered from one by switching tools,
    so counting it would fail runs that are making progress.
    """
    refusal = _tool_failure("graph_query", 'Tool "graph_query" is disabled by policy')
    guard = TurnGuard()
    flagged = []
    for _ in range(MAX_IDENTICAL_TOOL_FAILURES - 1):
        flagged.append(guard.observe(*_tool_failure()))
        flagged.append(guard.observe(*refusal))
    assert flagged == [None] * (2 * (MAX_IDENTICAL_TOOL_FAILURES - 1))

    tripped = guard.observe(*_tool_failure())
    assert tripped is not None, "the refusal must not have reset the editor count"
    assert tripped.count == MAX_IDENTICAL_TOOL_FAILURES


def test_the_unavailable_tool_refusal_is_not_a_loop_either():
    """The engine's other refusal literal, measured for the team/`Command`/`write_file` tools."""
    refusal = (
        "Tool call team_broadcast was rejected before execution: Model tried to call unavailable "
        "tool 'team_broadcast'."
    )
    guard = TurnGuard()
    assert [guard.observe(*_tool_failure("team_broadcast", refusal)) for _ in range(20)] == [None] * 20


def test_a_success_on_ANOTHER_tool_does_not_clear_the_loop_count():
    """Only a success of the SAME tool is progress for that tool's calls."""
    guard = TurnGuard()
    flagged = []
    for _ in range(MAX_IDENTICAL_TOOL_FAILURES - 1):
        flagged.append(guard.observe(*_tool_failure()))
        flagged.append(guard.observe(*_tool_success("read_files")))
    assert flagged == [None] * (2 * (MAX_IDENTICAL_TOOL_FAILURES - 1))

    tripped = guard.observe(*_tool_failure())
    assert tripped is not None, "an unrelated tool's success must not hide this loop"
    assert tripped.tool_name == "editor"


def test_a_retry_reads_the_ledger_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The retry path must not re-parse the whole NDJSON file a second time.

    MEASURED cost this pins: `_scan_ledger` exists to make ONE pass, because a run writes one ledger
    line per streamed event (the measured runs had 3131+). The tool-loop verdict used to be read by a
    separate helper that called `ledger.records()` again, so every activity retry re-parsed the whole
    file twice. On a long run that is a second O(n) read of the largest file the run owns.
    """
    from codna.durable.ledger import RunLedger

    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        _tool_failure() for _ in range(MAX_IDENTICAL_TOOL_FAILURES)
    ]
    request = context.turn_request("implement it", spec.initial_state())

    with pytest.raises(ActivityError) as first:
        run_cline_turn(request)
    assert first.value.code == "activity_turn_tool_loop", first.value.as_dict()

    # A genuine replay of the same request: it finds the recorded turn and its verdict.
    harness.sidecar.scripted_events = []
    harness.sidecar.terminal = "session.turn_aborted"
    harness.sidecar.abort_on_terminal = {"kind": "control_stop", "reason": "aborted by the guard"}

    scans: list[int] = []
    original = RunLedger.records

    def counting_records(self):  # type: ignore[no-untyped-def]
        scans.append(1)
        return original(self)

    monkeypatch.setattr(RunLedger, "records", counting_records)
    try:
        with pytest.raises(ActivityError) as second:
            run_cline_turn(request)
    finally:
        monkeypatch.setattr(RunLedger, "records", original)

    assert second.value.code == "activity_turn_tool_loop", second.value.as_dict()
    assert second.value.details["tool"] == "editor", second.value.as_dict()
    # ONE traversal for the whole retry, INCLUDING the loop verdict it reports.
    assert len(scans) == 1, f"the retry scanned the ledger {len(scans)} times"


def test_the_same_error_against_a_different_path_is_progress_not_a_loop(tmp_path: Path):
    """A model iterating over FILES is making progress, and killing it destroys real work.

    MEASURED false positive of the SHIPPED detector, and the reason the guard clears on a success:
    `run-20261112T223954Z`'s implement turn failed `old_text` required, then READ styles.css, learned
    the argument it was missing and edited it successfully at seq 1315. A guard keyed on arguments or
    on the raw error alone stops that turn before the read that fixes it.
    """
    harness, context, spec = _harness(tmp_path)
    harness.sidecar.scripted_events = [
        *[_tool_failure(error=EDITOR_ERROR) for _ in range(MAX_IDENTICAL_TOOL_FAILURES - 1)],
        _tool_success(),
        _tool_failure(),
        _tool_success(),
    ]

    outcome = run_cline_turn(context.turn_request("implement it", spec.initial_state()))

    assert outcome.details["finish_reason"] == "completed", outcome.details
    assert harness.sidecar.aborts == [], harness.sidecar.aborts


def test_reading_DIFFERENT_missing_files_is_exploration_not_a_loop(tmp_path: Path):
    """A plan turn exploring a tree that does not have the files yet must not be killed.

    MEASURED false positive, and the reason an ABSENCE error keeps its path. `run-20261113T051836Z`
    (this branch's own app-driven run) was a read-only PLAN turn: `read_files` on `package.json`,
    `./package.json`, `.`, `index.html`, `tests/counter.test.js`, `./tests`, `./tests/counter.test.js`
    and `./index.html` — seven DIFFERENT paths, every one `ENOENT`, nothing written yet. Collapsing the
    path made those one key, the count reached seven, and the turn was aborted before it produced a
    single file. Each path here is a distinct call, so the count must never reach the threshold.

    The paths are the ones from that ledger, transcribed rather than invented.
    """
    harness, context, spec = _harness(tmp_path)
    measured = [
        "Error reading file: ENOENT: no such file or directory, stat '/repo/package.json'",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/package.json'",
        "Error reading file: Path is not a file: /repo",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/index.html'",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/index.html'",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/tests/counter.test.js'",
        "Error reading file: Path is not a file: /repo/tests",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/tests/counter.test.js'",
        "Error reading file: ENOENT: no such file or directory, stat '/repo/index.html'",
    ]
    guard = TurnGuard()
    verdicts = [guard.observe(*_batch_failure(f"q{index}", error)) for index, error in enumerate(measured)]

    assert verdicts == [None] * len(measured), [
        (index, verdict.as_details()) for index, verdict in enumerate(verdicts) if verdict is not None
    ]


def test_an_editor_ENOTDIR_loop_against_different_outputs_still_trips():
    """A missing-DIRECTORY error is NOT exempt from path collapsing, so this loop is still caught.

    MEASURED, and the reason `_ABSENCE_RE` is narrow. The corpus has this exact shape from `editor`
    (`ENOTDIR: not a directory, open '<path>'`, `run-20261112T222825Z` seq 3115), where the path is
    the OUTPUT the call chose rather than a source it was looking for. Exempting `ENOTDIR` meant the
    same editor failure against several outputs produced several canonical keys and never tripped;
    the review reproduced it and was right. Distinct outputs are one failing call, not progress.
    """
    guard = TurnGuard()
    verdicts = [
        guard.observe(
            *(
                "agent.content_end",
                {
                    "contentType": "tool",
                    "toolName": "editor",
                    "output": {
                        "error": "Cannot write: target path is not a directory, /repo/outputs/" + name,
                        "query": f"edit:outputs/{name}",
                        "success": False,
                    },
                },
            )
        )
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta")
    ]

    tripped = [verdict for verdict in verdicts if verdict is not None]
    assert len(tripped) == 1, [(i, v.as_details()) for i, v in enumerate(verdicts) if v]
    assert tripped[0].tool_name == "editor", tripped[0].as_details()
    assert tripped[0].count == MAX_IDENTICAL_TOOL_FAILURES, tripped[0].as_details()


def test_a_genuine_ENOENT_runaway_on_ONE_path_still_trips():
    """The half that must not regress: keeping the path must not make an absence loop invisible.

    A runaway re-reads the SAME path it already knows is missing, so its key is stable and the count
    still reaches the threshold. Without this the false-positive fix above would be a hole.
    """
    same = "Error reading file: ENOENT: no such file or directory, stat '/repo/index.html'"
    guard = TurnGuard()
    verdicts = [guard.observe(*_batch_failure("index.html", same)) for _ in range(MAX_IDENTICAL_TOOL_FAILURES)]

    tripped = [verdict for verdict in verdicts if verdict is not None]
    assert len(tripped) == 1, f"expected exactly one trip, got {len(tripped)}"
    assert tripped[0].tool_name == "read_files", tripped[0].as_details()
    assert tripped[0].count == MAX_IDENTICAL_TOOL_FAILURES, tripped[0].as_details()


def test_the_guard_reads_unfamiliar_shapes_are_not_failures():
    """Total on message SHAPES: an unfamiliar payload is "no failure", never a manufactured one."""
    guard = TurnGuard()
    for payload in (
        {"contentType": "tool", "toolName": "x"},                       # no output at all
        {"contentType": "tool", "toolName": "x", "output": 7},          # a scalar
        {"contentType": "tool", "toolName": "x", "output": {"ok": 1}},  # a mapping with no error
        {"contentType": "tool", "toolName": "x", "output": []},         # an empty batch
        {"contentType": "tool", "toolName": "", "output": {"error": "boom"}},  # unattributed
    ):
        assert guard.observe("agent.content_end", payload) is None
