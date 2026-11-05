"""Tests for the soak invariant checker and its seeded kill schedule.

The point of this file is that every invariant the soak relies on fails *loudly and specifically*:
one named code, naming the offending id. A checker that reported "something is wrong" would pass a
smoke test and fail an operator at 3am, so each case asserts the code AND the subject.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codna.durable import ledger as ledger_module
from codna.durable import soak_report as sr


# -- record builders -----------------------------------------------------------------------------


def _record(
    seq: int,
    kind: str,
    *,
    prev_hash: str = sr.GENESIS_HASH,
    payload: dict[str, Any] | None = None,
    task_id: str | None = "task-1",
    run_id: str = "run-soak",
    session_id: str | None = "session-soak",
) -> dict[str, Any]:
    """One ledger record with a correct hash for its content.

    Built as a dict rather than a ``LedgerRecord`` so the tests exercise the same shape a soak gets
    from reading JSONL, and so a test can corrupt one field without fighting a frozen dataclass.
    """
    body: dict[str, Any] = {
        "seq": seq,
        "kind": kind,
        "ts": 1_700_000_000.0 + seq,
        "run_id": run_id,
        "task_id": task_id,
        "session_id": session_id,
        "event_index": None,
        "payload": dict(payload or {}),
        "prev_hash": prev_hash,
    }
    body["hash"] = sr.compute_hash(body)
    return body


def _chain(specs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """A valid hash chain over ``(kind, payload)`` pairs — the healthy baseline each test breaks."""
    records: list[dict[str, Any]] = []
    prev = sr.GENESIS_HASH
    for index, (kind, payload) in enumerate(specs, start=1):
        record = _record(index, kind, prev_hash=prev, payload=payload)
        records.append(record)
        prev = record["hash"]
    return records


def _rechain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute ``prev_hash``/``hash`` in order, leaving ``seq`` values untouched.

    Used to isolate one invariant: a test that drops or renumbers a line while leaving stale hashes
    would trip ``hash_chain_broken`` too, and then a failure could not be attributed to the invariant
    under test. Rechaining keeps the chain internally consistent so only the targeted check fires.
    """
    previous = sr.GENESIS_HASH
    for record in records:
        record["prev_hash"] = previous
        record["hash"] = sr.compute_hash(record)
        previous = record["hash"]
    return records


def _healthy() -> list[dict[str, Any]]:
    """A short but complete run: start, one paired turn, its checkpoint, finish."""
    return _chain(
        [
            ("run.started", {}),
            ("turn.started", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1"}),
            ("turn.finished", {"turn_id": "turn-a", "terminal": "end_turn"}),
            ("checkpoint", {"turn_seq": 1, "ref": "refs/codna/runs/run-soak/turn-1", "sha": "abc"}),
            ("run.finished", {"outcome": "success"}),
        ]
    )


def _codes(findings: Any) -> list[str]:
    return [finding.code for finding in findings]


# -- baseline ------------------------------------------------------------------------------------


def test_healthy_chain_produces_no_findings() -> None:
    """The baseline must be clean, or every other assertion in this file is meaningless."""
    report = sr.build_report("run-soak", _healthy(), checked_at=1_700_000_100.0)
    assert report.ok is True, report.render()
    assert report.findings == ()
    assert report.warnings == ()
    assert report.records == 5
    assert report.turns_started == 1
    assert report.turns_finished == 1
    assert report.checkpoints == 1
    assert report.checked_at == 1_700_000_100.0


def test_report_is_deterministic_and_pure() -> None:
    """Same records twice, same payload — and the input list is not mutated."""
    records = _healthy()
    snapshot = json.dumps(records, sort_keys=True)
    first = sr.build_report("run-soak", records, checked_at=1.5).to_payload()
    second = sr.build_report("run-soak", records, checked_at=1.5).to_payload()
    assert first == second
    assert json.dumps(records, sort_keys=True) == snapshot


def test_compute_hash_matches_the_ledger_implementation() -> None:
    """The checker's independent hash must agree with the writer's, byte for byte.

    A disagreement here would make every soak report a false ``hash_chain_broken``, so it is pinned
    explicitly rather than discovered in CI.
    """
    record = _record(1, "run.started", payload={"nested": {"b": 2, "a": [1, None]}})
    assert sr.compute_hash(record) == ledger_module.compute_hash(record)
    assert sr._canonical({"b": 1, "a": 2}) == ledger_module._canonical({"b": 1, "a": 2})


# -- invariant 1: duplicated turn_seq ------------------------------------------------------------


def test_duplicate_turn_seq_names_the_turn_triple() -> None:
    """Two checkpoints for one turn_seq: one named finding carrying run:task:seq."""
    records = _chain(
        [
            ("run.started", {}),
            ("turn.started", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1"}),
            ("turn.finished", {"turn_id": "turn-a"}),
            ("checkpoint", {"turn_seq": 1, "sha": "first"}),
            ("checkpoint", {"turn_seq": 1, "sha": "second"}),
        ]
    )
    findings = sr.check_duplicate_turn_seq(records)
    assert _codes(findings) == [sr.DUPLICATE_TURN_SEQ]
    assert findings[0].subject == "run-soak:task-1:1"
    assert findings[0].seq == 5
    assert "seq 4" in findings[0].detail


def test_duplicate_turn_seq_is_scoped_per_task() -> None:
    """Two tasks each on turn 1 is healthy — the invariant must not fire on it."""
    records = _chain(
        [
            ("checkpoint", {"turn_seq": 1}),
            ("checkpoint", {"turn_seq": 1}),
        ]
    )
    records[0]["task_id"] = "task-1"
    records[1]["task_id"] = "task-2"
    records[1]["prev_hash"] = records[0]["hash"]
    records[1]["hash"] = sr.compute_hash(records[1])
    assert sr.check_duplicate_turn_seq(records) == []


def test_duplicate_turn_seq_ignores_records_without_an_integer_seq() -> None:
    """A malformed checkpoint payload is a writer bug, not a duplicate — skipped, not mislabelled."""
    records = _chain(
        [
            ("checkpoint", {"turn_seq": "1"}),
            ("checkpoint", {"turn_seq": None}),
        ]
    )
    assert sr.check_duplicate_turn_seq(records) == []


# -- invariant 2: duplicated turn_key ------------------------------------------------------------


def test_duplicate_turn_key_names_the_key_and_both_turn_ids() -> None:
    """Same key, different turn_id: the replay path did not engage. Both ids in the detail."""
    records = _chain(
        [
            ("turn.started", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1"}),
            ("turn.finished", {"turn_id": "turn-a"}),
            ("turn.started", {"turn_id": "turn-b", "turn_key": "run-soak:task-1:1"}),
            ("turn.finished", {"turn_id": "turn-b"}),
        ]
    )
    findings = sr.check_duplicate_turn_key(records)
    assert _codes(findings) == [sr.DUPLICATE_TURN_KEY]
    assert findings[0].subject == "run-soak:task-1:1"
    assert findings[0].seq == 3
    assert "turn-a" in findings[0].detail and "turn-b" in findings[0].detail


def test_duplicate_turn_key_allows_a_replayed_same_turn_id() -> None:
    """The healthy replay: same key AND same turn_id — the engine replayed, nothing was billed twice."""
    records = _chain(
        [
            ("turn.started", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1", "replayed": False}),
            ("turn.finished", {"turn_id": "turn-a"}),
            ("turn.started", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1", "replayed": True}),
            ("turn.finished", {"turn_id": "turn-a"}),
        ]
    )
    assert sr.check_duplicate_turn_key(records) == []


def test_duplicate_turn_key_across_sessions_is_reported_as_rescoped() -> None:
    """Same key, different turn_id AND a different session: continuity was lost, not replay broken.

    The engine's turn-key index is per session, so a new session gives the same key a fresh index and
    the "replay" becomes a second paid turn. Reporting that as ``duplicate_turn_key`` would send an
    operator hunting an engine idempotency bug when the cause was a dropped session — so the code
    must differ, and the detail must name BOTH sessions.
    """
    records: list[dict[str, Any]] = []
    prev = sr.GENESIS_HASH
    specs: list[tuple[str, str, dict[str, Any]]] = [
        ("run.started", "session-a", {}),
        ("turn.started", "session-a", {"turn_id": "turn-a", "turn_key": "run-soak:task-1:1"}),
        ("turn.finished", "session-a", {"turn_id": "turn-a"}),
        # The worker died and the session could not be resumed: a NEW session, same key.
        ("turn.started", "session-b", {"turn_id": "turn-b", "turn_key": "run-soak:task-1:1"}),
        ("turn.finished", "session-b", {"turn_id": "turn-b"}),
    ]
    for index, (kind, session_id, payload) in enumerate(specs, start=1):
        record = _record(index, kind, prev_hash=prev, payload=payload, session_id=session_id)
        records.append(record)
        prev = record["hash"]

    findings = sr.check_duplicate_turn_key(records)
    assert _codes(findings) == [sr.TURN_KEY_RESCOPED]
    assert findings[0].subject == "run-soak:task-1:1"
    assert findings[0].seq == 4
    detail = findings[0].detail
    assert "session-a" in detail and "session-b" in detail
    assert "turn-a" in detail and "turn-b" in detail


def test_duplicate_turn_key_reads_the_session_from_the_payload() -> None:
    """The REAL ledger shape: ``turn.started`` carries the session in its payload, not the column.

    ``RunLedger.append`` refuses a ``session_id`` without an ``event_index``, so a turn's own record
    cannot put the session in the column — it goes in the payload. If the checker only looked at the
    column, every real ledger would degrade to "same key, different turn" and the resco
    diagnosis would never fire on production data. This pins the payload path so that cannot regress.
    """
    records = _chain(
        [
            ("turn.started", {"turn_id": "turn-a", "turn_key": "k", "session_id": "session-a"}),
            ("turn.finished", {"turn_id": "turn-a"}),
            ("turn.started", {"turn_id": "turn-b", "turn_key": "k", "session_id": "session-b"}),
            ("turn.finished", {"turn_id": "turn-b"}),
        ]
    )
    for record in records:
        record["session_id"] = None  # only the payload names the session, as in the real ledger
    _rechain(records)
    findings = sr.check_duplicate_turn_key(records)
    assert _codes(findings) == [sr.TURN_KEY_RESCOPED]
    assert "session-a" in findings[0].detail and "session-b" in findings[0].detail


# -- invariant 3: broken hash chain --------------------------------------------------------------


def test_broken_hash_chain_on_an_edited_record_names_its_seq() -> None:
    """Editing a payload after the fact leaves the stored hash stale — reported at that seq."""
    records = _healthy()
    records[2]["payload"]["terminal"] = "tampered"
    findings = sr.check_hash_chain(records)
    assert _codes(findings) == [sr.HASH_CHAIN_BROKEN]
    assert findings[0].subject == "3"
    assert "does not match recomputed" in findings[0].detail


def test_broken_hash_chain_on_a_removed_record_names_the_successor() -> None:
    """Dropping a middle line breaks the link at the next record, not the missing one."""
    records = _healthy()
    del records[2]
    findings = sr.check_hash_chain(records)
    assert _codes(findings) == [sr.HASH_CHAIN_BROKEN]
    assert findings[0].subject == "4"  # the old seq 4 now sits where seq 3 belonged
    assert "prev_hash" in findings[0].detail


def test_broken_hash_chain_reports_both_halves_separately() -> None:
    """A record with a wrong prev_hash AND a wrong hash yields two findings, not one.

    Corrupting the *last* record is deliberate: a forged ``hash`` also breaks the link to whatever
    follows it, so corrupting a middle record legitimately yields three findings. The last record
    isolates "both halves of one record" without that knock-on.
    """
    records = _healthy()
    records[-1]["prev_hash"] = "f" * 64
    records[-1]["hash"] = "e" * 64
    findings = sr.check_hash_chain(records)
    assert _codes(findings) == [sr.HASH_CHAIN_BROKEN, sr.HASH_CHAIN_BROKEN]
    assert {finding.subject for finding in findings} == {"5"}
    details = " | ".join(finding.detail for finding in findings)
    assert "prev_hash" in details and "does not match recomputed" in details


def test_broken_hash_chain_does_not_cascade_after_the_first_break() -> None:
    """Only the edited record is blamed: the chain resumes from its stored hash, not the recomputed."""
    records = _healthy()
    records[1]["payload"]["terminal"] = "tampered"
    findings = sr.check_hash_chain(records)
    assert len(findings) == 1, [finding.render() for finding in findings]


# -- invariant 4: seq gap ------------------------------------------------------------------------


def test_seq_gap_names_the_missing_position() -> None:
    """A ledger that skips seq 3 says so at the record that arrived in its place."""
    records = _healthy()
    del records[2]  # seqs become 1, 2, 4, 5 — exactly one position went missing
    _rechain(records)
    findings = sr.check_seq_contiguity(records)
    assert _codes(findings) == [sr.SEQ_GAP]
    assert findings[0].subject == "4"
    assert "where 3 was expected" in findings[0].detail


def test_seq_gap_on_a_repeated_position() -> None:
    """A duplicate seq is also a gap — the position was claimed twice.

    Two findings are expected and correct: seq 2 arrives where 3 belonged (the duplicate), and the
    record after it then sits one position ahead of where the resync landed. Collapsing those into
    one would under-report how far the ledger drifted.
    """
    records = _healthy()
    records[2]["seq"] = 2
    _rechain(records)
    findings = sr.check_seq_contiguity(records)
    assert _codes(findings) == [sr.SEQ_GAP, sr.SEQ_GAP]
    assert findings[0].subject == "2"
    assert "where 3 was expected" in findings[0].detail


def test_seq_gap_reports_the_first_position_only_when_the_rest_realigns() -> None:
    """One dropped line yields exactly one gap; the following seqs continue correctly."""
    records = _healthy()
    del records[1]
    _rechain(records)
    findings = sr.check_seq_contiguity(records)
    assert _codes(findings) == [sr.SEQ_GAP]
    assert findings[0].subject == "3"


# -- invariant 5: unpaired turns -----------------------------------------------------------------


def test_unpaired_turn_names_the_turn_id() -> None:
    """A finished run with a started turn that never finished is a hole, named by turn id."""
    records = _chain(
        [
            ("run.started", {}),
            ("turn.started", {"turn_id": "turn-orphan", "turn_key": "run-soak:task-1:1"}),
            ("run.finished", {"outcome": "crashed"}),
        ]
    )
    findings = sr.check_turn_pairing(records)
    assert _codes(findings) == [sr.TURN_UNPAIRED]
    assert findings[0].subject == "turn-orphan"
    assert findings[0].seq == 2


def test_unpaired_turn_is_tolerated_while_the_run_is_live() -> None:
    """Mid-run snapshots see open turns; ``allow_open_turns`` is why the checker stays usable there."""
    records = _chain(
        [
            ("run.started", {}),
            ("turn.started", {"turn_id": "turn-live", "turn_key": "run-soak:task-1:1"}),
        ]
    )
    assert sr.check_turn_pairing(records, allow_open_turns=True) == []
    report = sr.build_report("run-soak", records, allow_open_turns=True)
    assert report.ok is True


# -- report aggregation --------------------------------------------------------------------------


def test_report_aggregates_every_distinct_failure_in_a_stable_order() -> None:
    """All five invariants at once: five findings, one per code, sorted deterministically."""
    records = _healthy()
    # seq gap: drop the checkpoint line entirely. Seqs become 1,2,3,5 — one position lost, and
    # rechaining keeps the hash invariant out of this finding so each code has one clean subject.
    del records[3]
    _rechain(records)
    # duplicate turn_key + unpaired turn: a second start under the same key with a different turn id.
    duplicate = _record(
        6,
        "turn.started",
        prev_hash=records[-1]["hash"],
        payload={"turn_id": "turn-b", "turn_key": "run-soak:task-1:1"},
    )
    records.append(duplicate)
    # duplicate turn_seq: two checkpoints claiming turn 1.
    first_checkpoint = _record(
        7, "checkpoint", prev_hash=duplicate["hash"], payload={"turn_seq": 1, "sha": "first"}
    )
    records.append(first_checkpoint)
    records.append(
        _record(
            8,
            "checkpoint",
            prev_hash=first_checkpoint["hash"],
            payload={"turn_seq": 1, "sha": "second"},
        )
    )
    # hash chain: break the final link only, so exactly one hash finding appears.
    records[-1]["prev_hash"] = "a" * 64
    records[-1]["hash"] = sr.compute_hash(records[-1])

    report = sr.build_report("run-soak", records, checked_at=2.0)
    codes = [finding.code for finding in report.findings]
    assert codes == sorted(codes), codes  # _sort_findings groups by code
    assert set(codes) == {
        sr.DUPLICATE_TURN_SEQ,
        sr.DUPLICATE_TURN_KEY,
        sr.HASH_CHAIN_BROKEN,
        sr.SEQ_GAP,
        sr.TURN_UNPAIRED,
    }
    assert len(codes) == 5, [finding.render() for finding in report.findings]
    subjects = {finding.code: finding.subject for finding in report.findings}
    assert subjects[sr.DUPLICATE_TURN_SEQ] == "run-soak:task-1:1"
    assert subjects[sr.DUPLICATE_TURN_KEY] == "run-soak:task-1:1"
    assert subjects[sr.HASH_CHAIN_BROKEN] == "8"
    assert subjects[sr.SEQ_GAP] == "5"
    assert subjects[sr.TURN_UNPAIRED] == "turn-b"
    assert "FAILED" in report.render()


def test_report_accepts_ledger_records_as_well_as_dicts() -> None:
    """``load_ledger`` hands back frozen dataclasses; the checker must accept both shapes."""
    records = _healthy()
    decoded = [ledger_module.LedgerRecord.from_payload(record) for record in records]
    from_dicts = sr.build_report("run-soak", records).to_payload()
    from_records = sr.build_report("run-soak", decoded).to_payload()
    assert from_dicts == from_records


def test_malformed_records_fail_instead_of_becoming_findings() -> None:
    """A record with no seq is a decoding bug; reporting it as a gap would hide the real cause."""
    records = _healthy()
    del records[0]["seq"]
    with pytest.raises(sr.SoakInvariantError) as caught:
        sr.build_report("run-soak", records)
    assert caught.value.code == "soak_record_invalid"


def test_build_report_rejects_a_missing_run_id() -> None:
    with pytest.raises(sr.SoakInvariantError) as caught:
        sr.build_report("", _healthy())
    assert caught.value.code == "soak_input_invalid"


def test_build_report_rejects_a_non_numeric_checked_at() -> None:
    with pytest.raises(sr.SoakInvariantError) as caught:
        sr.build_report("run-soak", _healthy(), checked_at="now")  # type: ignore[arg-type]
    assert caught.value.code == "soak_input_invalid"


# -- seeded kill schedule ------------------------------------------------------------------------


def test_kill_schedule_is_deterministic_for_a_seed() -> None:
    """The whole point of the seed: a failure is reproducible from the number in the log."""
    first = sr.generate_kill_points(7, count=5)
    second = sr.generate_kill_points(7, count=5)
    assert [point.to_payload() for point in first] == [point.to_payload() for point in second]
    assert sr.describe_schedule(first) == sr.describe_schedule(second)


def test_kill_schedule_differs_across_seeds() -> None:
    """Deterministic must not mean identical — otherwise the seed buys nothing."""
    first = sr.generate_kill_points(1, count=5)
    second = sr.generate_kill_points(2, count=5)
    assert [p.delay_seconds for p in first] != [p.delay_seconds for p in second]


def test_kill_schedule_delays_are_bounded_and_ordered() -> None:
    """Delays stay within ``total_seconds`` plus the jitter ceiling, and increase with the index."""
    points = sr.generate_kill_points(3, count=5, total_seconds=100.0, max_jitter_seconds=2.0)
    delays = [point.delay_seconds for point in points]
    assert delays == sorted(delays)
    assert all(0.0 <= delay <= 102.0 for delay in delays)
    assert [point.index for point in points] == [0, 1, 2, 3, 4]


def test_kill_schedule_resamples_fractions_for_an_unusual_count() -> None:
    """Asking for 12 kills gets 12 evenly spread points, not a silent truncation to five."""
    points = sr.generate_kill_points(11, count=12, total_seconds=120.0, max_jitter_seconds=0.0)
    assert len(points) == 12
    assert points[0].delay_seconds == pytest.approx(120.0 / 13)
    assert points[-1].delay_seconds == pytest.approx(120.0 * 12 / 13)


def test_kill_schedule_cycles_the_requested_actions() -> None:
    """A mixed schedule alternates the actions it was given, in order, deterministically."""
    points = sr.generate_kill_points(
        5, count=4, total_seconds=10.0, actions=("sigkill", "sigterm"), max_jitter_seconds=0.0
    )
    assert [point.action for point in points] == ["sigkill", "sigterm", "sigkill", "sigterm"]


def test_kill_schedule_zero_count_is_empty() -> None:
    assert sr.generate_kill_points(1, count=0) == []


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"total_seconds": 0.0}, "soak_schedule_invalid"),
        ({"total_seconds": -5.0}, "soak_schedule_invalid"),
        ({"max_jitter_seconds": -1.0}, "soak_schedule_invalid"),
        ({"fractions": ()}, "soak_schedule_invalid"),
        ({"fractions": (0.5, 1.5)}, "soak_schedule_invalid"),
        ({"actions": ()}, "soak_schedule_invalid"),
        ({"actions": ("sigstop",)}, "soak_schedule_invalid"),
    ],
)
def test_kill_schedule_rejects_bad_input(kwargs: dict[str, Any], code: str) -> None:
    """Refusing to start beats starting a soak that runs for an hour or kills nothing."""
    with pytest.raises(sr.SoakInvariantError) as caught:
        sr.generate_kill_points(1, count=5, **kwargs)
    assert caught.value.code == code


@pytest.mark.parametrize("count", [-1, True])
def test_kill_schedule_rejects_a_bad_count(count: Any) -> None:
    """``count`` gets its own case because it is positional-ish: ``True`` must not read as 1."""
    with pytest.raises(sr.SoakInvariantError) as caught:
        sr.generate_kill_points(1, count=count)
    assert caught.value.code == "soak_schedule_invalid"


# -- the chaos script's pure helpers --------------------------------------------------------------
#
# `scripts/durable_soak.py` is not a package module (it is the driver, not library code), so it is
# loaded by PATH. Only its PURE helpers are tested here: settings validation and the worker-readiness
# predicate. Everything else in that file starts processes, and that is what the soak itself covers.


def _soak_script() -> Any:
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[3] / "scripts" / "durable_soak.py"
    spec = importlib.util.spec_from_file_location("codna_durable_soak_script", script)
    assert spec is not None and spec.loader is not None, script
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


# A provider is REQUIRED by `SoakSettings` (see the field's comment): these tests carry one so the
# cases below test the setting they name rather than tripping over the provider check.
_BASE_SETTINGS: dict[str, Any] = {
    "kills": 3,
    "seed": 2,
    "duration_s": 600.0,
    "spread_s": 300.0,
    "provider": "openai",
    "provider_key_env": "OPENAI_API_KEY",
}


def test_soak_settings_accept_a_coherent_configuration() -> None:
    script = _soak_script()
    settings = script.SoakSettings(**_BASE_SETTINGS)
    assert settings.validate() is settings


@pytest.mark.parametrize(
    "overrides",
    [
        {"kills": -1},
        {"kills": True},
        {"duration_s": 0.0},
        {"spread_s": 0.0},
        {"spread_s": 900.0},  # a planned kill beyond the run's own budget can never be reached
        {"engine": "celery"},
        {"brief": "   "},
        {"task_id": "task/1"},
        {"task_id": ""},
        # A soak with no provider cannot reach a checkpoint, so it must not start at all: this is the
        # guard that stops a ledger-only run from being produced and read as a durability result.
        {"provider": None},
        {"provider": "   "},
        {"provider": ""},
    ],
)
def test_soak_settings_refuse_an_unusable_configuration(overrides: dict[str, Any]) -> None:
    """Every refusal names its problems — a soak must not start and then misreport the reason."""
    script = _soak_script()
    base: dict[str, Any] = dict(_BASE_SETTINGS)
    base.update(overrides)
    with pytest.raises(script.SoakError) as caught:
        script.SoakSettings(**base).validate()
    assert caught.value.code == "soak_settings_invalid"
    assert caught.value.details["problems"]


def test_worker_readiness_requires_the_registration_banner() -> None:
    """A worker that merely started is not registered — and `run --pin` refuses an unregistered build."""
    script = _soak_script()
    ready = '{"event": "worker.starting", "registration": {"ready": true}, "build_id": "codna-1"}'
    not_ready = '{"event": "worker.starting", "registration": {"ready": false}}'
    assert script._line_reports_registration(ready) is True
    assert script._line_reports_registration(not_ready) is False
    assert script._line_reports_registration("worker.starting (plain text)") is False
    assert script._line_reports_registration('{"event": "worker.stopped"}') is False


def test_closed_workflow_detection_is_narrow() -> None:
    """A closed workflow is "finished"; a missing dependency is not, and must not look like one."""
    script = _soak_script()
    assert script._mentions_closed_workflow('{"error": {"code": "durable_query_failed"}}') is True
    assert script._mentions_closed_workflow('{"error": {"code": "durable_workflow_not_found"}}') is True
    assert script._mentions_closed_workflow("ModuleNotFoundError: No module named 'temporalio'") is False
    assert script._mentions_closed_workflow("the dev server never started") is False


def test_ledger_finished_requires_the_runs_own_terminal_notification() -> None:
    """"Finished" is a ledger fact. A closed workflow with no ``run.finished`` is a run that died."""
    script = _soak_script()
    finished = _chain(
        [
            ("run.started", {}),
            ("note", {"event": "notification", "kind": "run.finished", "message": "done"}),
        ]
    )
    died = _chain([("run.started", {}), ("note", {"event": "notification", "kind": "task.started"})])
    assert script._ledger_shows_finished(finished) is True
    assert script._ledger_shows_finished(died) is False
    assert script._ledger_shows_finished([]) is False
    # A checkpoint payload that merely mentions "finished" is not the run's terminal.
    decoy = _chain([("checkpoint", {"turn_seq": 1, "sha": "finished"})])
    assert script._ledger_shows_finished(decoy) is False


def test_worker_errors_picks_the_cause_without_the_stack() -> None:
    """A crashed activity's last line names the cause; the report keeps that and drops the stack."""
    script = _soak_script()
    lines = [
        '{"event": "worker.starting", "registration": {"ready": true}}',
        "Completing activity as failed ({'activity_id': '2', 'attempt': 1})",
        "Traceback (most recent call last):",
        '  File "/x/activity.py", line 1, in plan_task',
        "codna.durable.sidecar_client.SidecarClientError: start_turn failed: 403 owner mismatch",
        "Completing activity as failed ({'activity_id': '2', 'attempt': 2})",
        "codna.durable.sidecar_client.SidecarClientError: start_turn failed: 403 owner mismatch",
        "some unrelated line",
    ]
    picked = script._worker_errors(lines)
    assert picked, picked
    assert any("403 owner mismatch" in line for line in picked)
    assert len(picked) == len(set(picked)), "the same failure retried must appear once"
    assert all("File \"" not in line for line in picked), picked


def test_worker_errors_caps_the_list() -> None:
    script = _soak_script()
    picked = script._worker_errors([f"Error: failure {index}" for index in range(40)], limit=5)
    assert len(picked) == 5
    assert picked[-1] == "Error: failure 39"


def test_soak_task_queue_is_unique_per_workdir() -> None:
    """Two soaks must not share a queue, or a leftover worker could execute the next run's activities."""
    script = _soak_script()
    first = script._task_queue_for(Path("/tmp/soak-a"))
    second = script._task_queue_for(Path("/tmp/soak-b"))
    assert first != second
    assert first.startswith("codna-soak-")
    assert first.isascii() and " " not in first
    assert len(first) <= 64  # the config's queue-name validator caps at 64 characters


# -- module hygiene ------------------------------------------------------------------------------


def test_soak_report_has_no_io_or_clock_imports() -> None:
    """The checker is pure by construction: no time, no os, no subprocess, no network.

    A regression here would make the verdict depend on when and where it ran, which is exactly the
    property the durable package refuses to allow.
    """
    source = Path(sr.__file__).read_text(encoding="utf-8")
    for forbidden in ("import time", "import os", "import socket", "import subprocess", "import pathlib"):
        assert forbidden not in source, forbidden
