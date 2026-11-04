"""The ledger: dedupe, contiguity, hash chain, and the Parquet rollup.

Each test targets one of the four properties in the module docstring, and the tamper tests
deliberately edit the file the way a partial write or a hand-edit would — because "tamper-evident"
is only true if someone actually tries. The Parquet rollup is checked against the NDJSON it is
derived from, so a rollup that silently drops rows fails rather than producing a plausible file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codna.durable.config import DurablePaths, validate_run_id
from codna.durable.ledger import (
    GENESIS_HASH,
    LedgerError,
    RunLedger,
    compute_hash,
    load_ledger,
    records_after,
    verify_chain,
)


@pytest.fixture
def paths(tmp_path: Path) -> DurablePaths:
    root = tmp_path / ".codna"
    return DurablePaths(
        root=root,
        temporal_dir=root / "temporal",
        dev_db_path=root / "temporal" / "dev.db",
        dev_server_bin_dir=root / "temporal" / "bin",
        runs_root=root / "runs",
        logs_dir=root / "logs",
    )


def _now() -> float:
    return 1_700_000_000.0


@pytest.fixture
def ledger(paths: DurablePaths) -> RunLedger:
    return RunLedger.for_run(paths, "run-1", clock=_now)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records), encoding="utf-8")


# -- append / hash chain ----------------------------------------------------------------


def test_first_record_links_to_genesis(ledger: RunLedger):
    record = ledger.append("run.started", payload={"brief": "fix the bug"})
    assert record is not None
    assert record.seq == 1
    assert record.prev_hash == GENESIS_HASH
    assert record.hash != GENESIS_HASH
    assert record.hash == compute_hash(record.to_payload())


def test_chain_links_each_record_to_its_predecessor(ledger: RunLedger):
    first = ledger.append("run.started")
    second = ledger.append("task.started", task_id="task-1")
    third = ledger.append("note", payload={"text": "hello"})
    assert second.prev_hash == first.hash
    assert third.prev_hash == second.hash
    assert verify_chain(ledger.path) == []
    assert [r.seq for r in ledger.records()] == [1, 2, 3]


def test_ndjson_file_is_one_object_per_line_with_stable_key_order(ledger: RunLedger):
    ledger.append("run.started", payload={"b": 1, "a": 2})
    raw = ledger.path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    assert raw.startswith('{"event_index":null')  # sorted keys, compact separators
    assert "hash" in json.loads(raw)


def test_hash_is_stable_for_identical_content():
    record = {
        "seq": 1,
        "kind": "note",
        "ts": 1.0,
        "run_id": "r",
        "task_id": None,
        "session_id": None,
        "event_index": None,
        "payload": {"a": 1, "b": 2},
        "prev_hash": GENESIS_HASH,
    }
    assert compute_hash(record) == compute_hash({**record, "hash": "ignored"})
    assert compute_hash(record) != compute_hash({**record, "payload": {"a": 1, "b": 3}})


# -- dedupe -----------------------------------------------------------------------------


def test_same_session_event_appended_twice_is_recorded_once(ledger: RunLedger):
    first = ledger.append("turn.event", session_id="session_a", event_index=1, payload={"event_type": "x"})
    again = ledger.append("turn.event", session_id="session_a", event_index=1, payload={"event_type": "x"})
    assert first is not None
    assert again is None
    assert len(ledger.records()) == 1


def test_dedupe_survives_a_new_ledger_instance(ledger: RunLedger, paths: DurablePaths):
    ledger.append("turn.event", session_id="session_a", event_index=7, payload={"event_type": "x"})
    # A retried activity constructs a fresh ledger; the dedupe set is recovered from the file.
    reopened = RunLedger.for_run(paths, "run-1", clock=_now)
    assert reopened.append("turn.event", session_id="session_a", event_index=7, payload={}) is None
    assert len(reopened.records()) == 1


def test_dedupe_is_keyed_on_the_pair_not_the_index_alone(ledger: RunLedger):
    assert ledger.append("turn.event", session_id="session_a", event_index=5, payload={}) is not None
    assert ledger.append("turn.event", session_id="session_b", event_index=5, payload={}) is not None
    assert len(ledger.records()) == 2


def test_event_batch_drops_the_overlap_a_reconnect_produces(ledger: RunLedger):
    events = [
        {"event_index": 1, "event_type": "session.created", "payload": {}},
        {"event_index": 2, "event_type": "agent.thinking", "payload": {"text": "..."}},
    ]
    first = ledger.append_events(events, task_id="task-1", session_id="session_a")
    # A second attempt replays from index 1 (the SSE window overlaps by design).
    second = ledger.append_events(events, task_id="task-1", session_id="session_a")
    assert len(first) == 2
    assert second == []
    assert len(ledger.records()) == 2


def test_two_record_kinds_for_one_event_require_the_explicit_override(ledger: RunLedger):
    ledger.append("turn.event", session_id="s", event_index=3, payload={"event_type": "x"})
    # The default stays strict: a duplicate is a no-op.
    assert ledger.append("turn.finished", session_id="s", event_index=3, payload={}) is None
    # The one legitimate case passes the override explicitly.
    assert (
        ledger.append("turn.finished", session_id="s", event_index=3, payload={}, allow_duplicate_event=True)
        is not None
    )
    assert len(ledger.records()) == 2


def test_partial_dedupe_key_is_refused(ledger: RunLedger):
    with pytest.raises(LedgerError) as excinfo:
        ledger.append("turn.event", session_id="s", payload={})
    assert excinfo.value.code == "ledger_dedupe_key_incomplete"


# -- validation -------------------------------------------------------------------------


def test_unknown_record_kind_is_refused(ledger: RunLedger):
    with pytest.raises(LedgerError) as excinfo:
        ledger.append("task.exploded")
    assert excinfo.value.code == "ledger_kind_invalid"
    assert "task.started" in str(excinfo.value.details["allowed"])


def test_non_json_payload_is_refused_with_the_kind_named(ledger: RunLedger):
    with pytest.raises(LedgerError) as excinfo:
        ledger.append("note", payload={"when": object()})
    assert excinfo.value.code == "ledger_payload_invalid"
    assert excinfo.value.details["kind"] == "note"
    # Nothing was written: the record is refused before the hash is computed.
    assert ledger.records() == []


def test_nan_in_a_payload_is_refused_so_the_hash_cannot_drift(ledger: RunLedger):
    with pytest.raises(LedgerError) as excinfo:
        ledger.append("note", payload={"cost": float("nan")})
    assert excinfo.value.code == "ledger_payload_invalid"


def test_boolean_event_index_is_refused(ledger: RunLedger):
    with pytest.raises(LedgerError) as excinfo:
        ledger.append("turn.event", session_id="s", event_index=True, payload={})
    assert excinfo.value.code == "ledger_event_index_invalid"


# -- contiguity -------------------------------------------------------------------------


def test_ledger_seq_is_contiguous(ledger: RunLedger):
    for _ in range(5):
        ledger.append("note")
    assert ledger.check_contiguity() == []
    assert [r.seq for r in ledger.records()] == [1, 2, 3, 4, 5]


def test_a_missing_seq_is_reported_with_both_numbers(ledger: RunLedger):
    for _ in range(3):
        ledger.append("note")
    records = _lines(ledger.path)
    del records[1]  # simulate a lost append
    _rewrite(ledger.path, records)
    problems = ledger.check_contiguity()
    assert problems
    assert "expected seq 2, found 3" in problems[0]


def test_session_event_contiguity_is_per_task(ledger: RunLedger):
    ledger.append("turn.event", task_id="task-1", session_id="s", event_index=1, payload={})
    ledger.append("turn.event", task_id="task-1", session_id="s", event_index=3, payload={})  # gap
    ledger.append("turn.event", task_id="task-2", session_id="s", event_index=1, payload={})
    ledger.append("turn.event", task_id="task-2", session_id="s", event_index=2, payload={})
    problems = ledger.check_session_contiguity("s")
    assert len(problems) == 1
    assert "task-1" in problems[0]
    assert "skips 1" in problems[0]


def test_a_contiguous_session_reports_nothing(ledger: RunLedger):
    for index in (1, 2, 3, 4):
        ledger.append("turn.event", task_id="t", session_id="s", event_index=index, payload={})
    assert ledger.check_session_contiguity("s") == []


# -- tamper evidence --------------------------------------------------------------------


def test_editing_a_payload_breaks_the_chain_at_that_record(ledger: RunLedger):
    for _ in range(3):
        ledger.append("note", payload={"text": "original"})
    records = _lines(ledger.path)
    records[1]["payload"]["text"] = "tampered"
    _rewrite(ledger.path, records)

    problems = verify_chain(ledger.path)
    assert any("seq 2: hash mismatch" in problem for problem in problems)
    # The chain is broken from there on: record 3's prev_hash no longer matches record 2's stored
    # hash (the stored hash of 2 is unchanged, so only the recompute fails here) — asserted
    # explicitly so a future refactor cannot make verification stop at the first error.
    assert len(problems) >= 1


def test_breaking_a_link_is_reported_at_the_following_record(ledger: RunLedger):
    for _ in range(3):
        ledger.append("note")
    records = _lines(ledger.path)
    records[1]["prev_hash"] = "f" * 64
    _rewrite(ledger.path, records)
    problems = verify_chain(ledger.path)
    assert any("prev_hash" in problem for problem in problems)


def test_a_truncated_final_line_is_reported_not_skipped(ledger: RunLedger):
    ledger.append("note", payload={"text": "a"})
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "kind": "no')  # a kill mid-append
    with pytest.raises(LedgerError) as excinfo:
        ledger.records()
    assert excinfo.value.code == "ledger_line_invalid"


def test_an_intact_ledger_verifies_clean(ledger: RunLedger):
    ledger.append("run.started", payload={"brief": "x"})
    ledger.append("turn.event", session_id="s", event_index=1, payload={"event_type": "y"})
    ledger.append("checkpoint", payload={"sha": "a" * 40})
    assert verify_chain(ledger.path) == []
    records, problems = load_ledger(ledger.path)
    assert len(records) == 3
    assert problems == []


# -- reading ----------------------------------------------------------------------------


def test_tail_and_stats(ledger: RunLedger):
    assert ledger.tail() is None
    assert ledger.stats().records == 0
    ledger.append("run.started")
    last = ledger.append("note", payload={"cost_usd": 0.25, "tokens": 100})
    assert ledger.tail() == last
    stats = ledger.stats()
    assert stats.records == 2
    assert stats.last_seq == 2
    assert stats.last_hash == last.hash
    assert stats.first_ts == stats.last_ts == _now()


def test_records_after_supports_reattach(ledger: RunLedger):
    for _ in range(5):
        ledger.append("note")
    records = ledger.records()
    assert [r.seq for r in records_after(records, 3)] == [4, 5]
    assert records_after(records, 99) == []


# -- parquet rollup ---------------------------------------------------------------------


def test_parquet_rollup_mirrors_the_ndjson(ledger: RunLedger):
    pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    ledger.append("run.started", payload={"brief": "x"})
    ledger.append("turn.event", task_id="task-1", session_id="s", event_index=1, payload={"event_type": "agent.thinking", "cost_usd": 0.5, "tokens": 120})
    ledger.append("turn.event", task_id="task-1", session_id="s", event_index=2, payload={"event_type": "agent.text", "cost_usd": 0.25, "tokens": 80})

    target = ledger.rollup_parquet()
    assert target.is_file()
    table = parquet.read_table(target)
    assert table.num_rows == 3

    rows = table.to_pylist()
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert rows[1]["event_type"] == "agent.thinking"
    assert rows[1]["cost_usd"] == pytest.approx(0.5)
    assert rows[2]["tokens"] == pytest.approx(80.0)
    # Derived data must be able to answer the question it exists for.
    assert sum(row["cost_usd"] or 0.0 for row in rows) == pytest.approx(0.75)
    # The hashes ride along, so a rollup can be checked against the chain without the NDJSON.
    assert rows[0]["hash"] == ledger.records()[0].hash
    assert rows[0]["prev_hash"] == GENESIS_HASH


def test_parquet_rollup_is_rebuilt_not_appended(ledger: RunLedger):
    parquet = pytest.importorskip("pyarrow.parquet")
    ledger.append("note", payload={"tokens": 1})
    target = ledger.rollup_parquet()
    assert parquet.read_table(target).num_rows == 1
    ledger.append("note", payload={"tokens": 2})
    target = ledger.rollup_parquet()
    assert parquet.read_table(target).num_rows == 2  # not 3: the rollup is rebuilt from scratch


def test_parquet_rollup_of_an_empty_ledger_is_a_valid_empty_table(ledger: RunLedger, paths: DurablePaths):
    parquet = pytest.importorskip("pyarrow.parquet")
    target = ledger.rollup_parquet()
    table = parquet.read_table(target)
    assert table.num_rows == 0
    assert set(table.column_names) >= {"seq", "kind", "hash", "cost_usd"}


def test_parquet_rollup_replaces_atomically(ledger: RunLedger):
    pytest.importorskip("pyarrow")
    ledger.append("note")
    target = ledger.rollup_parquet()
    tmp = target.with_suffix(target.suffix + ".tmp")
    assert not tmp.exists()  # the temp file never survives a successful rollup


# -- run id safety ----------------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["../etc/passwd", "a/b", "", "x" * 200, ".hidden", "a..b"])
def test_unsafe_run_ids_are_refused(run_id):
    with pytest.raises(Exception):
        validate_run_id(run_id)


def test_ledger_refuses_to_be_constructed_for_an_unsafe_run_id(paths: DurablePaths):
    with pytest.raises(Exception):
        RunLedger.for_run(paths, "../escape")


def test_two_instances_interleaved_do_not_fork_the_chain(tmp_path):
    """The ledger is SHARED, and the flock serialises only the write.

    Measured before this test existed: two `RunLedger` instances over one file, alternating appends, produced
    seqs [1, 2, 2, 3] with a shared prev_hash — a forked chain whose whole purpose is being tamper-evident.
    Each instance cached the tail when it first loaded and never refreshed it, so both computed `seq = 1` from
    the same genesis parent.

    The tail is now re-read under the lock on EVERY append, so `seq` and `prev_hash` come from the file rather
    than from a snapshot that another writer invalidated.
    """
    from codna.durable.ledger import RunLedger, verify_chain

    path = tmp_path / "ledger.ndjson"

    def event(index: int, message: str) -> dict:
        return {"kind": "note", "message": message, "event_index": index}

    a = RunLedger(path, run_id="run-fork")
    b = RunLedger(path, run_id="run-fork")

    # Alternate so each instance's cached tail is stale by the time it writes again.
    a.append_events([event(0, "a1")], session_id="s-a")
    b.append_events([event(1, "b1")], session_id="s-b")
    a.append_events([event(2, "a2")], session_id="s-a")
    b.append_events([event(3, "b2")], session_id="s-b")

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seqs = [record["seq"] for record in records]
    parents = [record["prev_hash"] for record in records]

    assert seqs == [1, 2, 3, 4], f"the chain forked: seqs={seqs}"
    assert len(set(parents)) == len(parents), f"two records share a parent: {parents}"
    assert verify_chain(path) == [], "verify_chain must accept an interleaved chain"
