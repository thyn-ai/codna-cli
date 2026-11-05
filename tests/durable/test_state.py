"""`TaskState` — the size ceiling and the secret guard.

These are the two invariants that keep a durable run safe rather than merely working: state that is
too big overflows Temporal's payload limit mid-run, and state that carries a key publishes it into
workflow history, where it outlives every other copy of that secret. Both are tested on the way in
(decode), on the way out (encode), and through a field added by a future change — the last case is
what stops the guards from being cargo-cult: they run over the *payload*, so a new field cannot
bypass them.
"""
from __future__ import annotations

import json

import pytest

from codna.durable import state as state_module
from codna.durable.state import (
    MAX_STATE_BYTES,
    Attempts,
    Budget,
    CheckpointRef,
    PendingApproval,
    PullRequestRef,
    TaskState,
    TaskStateError,
    VerifyRef,
    assert_state_encodable,
    find_secret_like_strings,
)


def _state(**overrides) -> TaskState:
    base = {"run_id": "run-1", "task_id": "task-1"}
    base.update(overrides)
    return TaskState(**base)


def _github_token() -> str:
    # Split so the literal never appears in the file: a scanner (and this repo's own secret gate)
    # would otherwise flag the fixture itself. `ghp_` + 36 alnum chars is the classic PAT shape.
    return "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _openai_key() -> str:
    return "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz0123456789"


# -- size guard -------------------------------------------------------------------------


def test_payload_roundtrips_and_reports_its_size():
    original = _state()
    payload = original.to_payload()
    size = assert_state_encodable(payload)
    assert size > 0
    assert TaskState.from_payload(payload) == original


def test_oversized_state_is_refused_on_the_way_out():
    # A state whose *serialized* form is too large. Built through the payload path rather than by
    # putting a blob in a reference field, because the whole point is that the bound applies to the
    # encoding, not to a chosen field.
    payload = _state().to_payload()
    payload["payload"] = "z" * MAX_STATE_BYTES
    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable(payload)
    assert excinfo.value.code == "state_too_large"


def test_a_reference_field_cannot_carry_a_blob():
    # Independently of size: a reference field is a reference, so a 2 MB "plan_ref" is refused by
    # the charset/length rule even when it is under the byte ceiling.
    with pytest.raises(TaskStateError) as excinfo:
        _state(plan_ref="p" * 5000).to_payload()
    assert excinfo.value.code == "invalid_plan_ref"


def test_oversized_state_is_refused_on_the_way_in():
    payload = _state().to_payload()
    payload["note"] = "x" * (MAX_STATE_BYTES + 1)  # an extra field still counts: the guard is on the payload
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.code == "state_too_large"
    assert excinfo.value.details["limit"] == MAX_STATE_BYTES
    assert excinfo.value.details["bytes"] > MAX_STATE_BYTES


def test_a_field_added_later_cannot_bypass_the_size_guard(monkeypatch):
    # The invariant is "the serialized payload is bounded", not "these fifteen fields are bounded".
    # Simulate a future field by shrinking the limit and adding an unknown key: the guard counts the
    # whole encoding, so a field it has never heard of still trips it.
    payload = _state().to_payload()  # serialized BEFORE the limit is shrunk, so it is valid
    payload["future_field"] = "y" * 500
    monkeypatch.setattr(state_module, "MAX_STATE_BYTES", len(json.dumps(_state().to_payload())))
    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable(payload)
    assert excinfo.value.code == "state_too_large"


def test_exactly_at_the_limit_is_accepted_and_one_byte_over_is_not(monkeypatch):
    payload = _state().to_payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=None)
    monkeypatch.setattr(state_module, "MAX_STATE_BYTES", len(encoded.encode("utf-8")))
    assert assert_state_encodable(payload) == len(encoded.encode("utf-8"))
    monkeypatch.setattr(state_module, "MAX_STATE_BYTES", len(encoded.encode("utf-8")) - 1)
    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable(payload)
    assert excinfo.value.code == "state_too_large"


def test_non_json_values_are_refused_rather_than_stringified():
    from pathlib import Path

    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable({"path": Path("/tmp/x")})
    assert excinfo.value.code == "state_not_serializable"


# -- secret guard -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_ref", _github_token()),
        ("plan_ref", _openai_key()),
        ("plan_ref", "AIza" + "SyA1234567890abcdefghijklmnopq"),
        ("plan_ref", "hf_" + "abcdefghijklmnopqrstuvwxyz"),
        ("plan_ref", "glpat-" + "abcdefghijklmnopqrst"),
        ("plan_ref", "AKIA" + "ABCDEFGHIJKLMNOP"),
        ("plan_ref", "Bearer " + "abcdefghijklmnopqrstuvwxyz0123"),
    ],
)
def test_secret_shaped_values_are_refused(field, value):
    payload = _state().to_payload()
    payload[field] = value
    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable(payload)
    assert excinfo.value.code == "state_contains_secret"
    # The report names the path and NEVER repeats the value: the error itself goes to a log.
    assert value not in json.dumps(excinfo.value.details)
    assert excinfo.value.details["findings"][0]["path"] == field


@pytest.mark.parametrize("name", ["api_key", "provider_api_key", "authorization", "password", "GITHUB_TOKEN"])
def test_forbidden_field_names_are_refused_whatever_the_value(name):
    with pytest.raises(TaskStateError) as excinfo:
        assert_state_encodable({"run_id": "r", name: "not-even-secret-looking"})
    assert excinfo.value.code == "state_contains_secret"
    assert excinfo.value.details["findings"][0]["reason"] == "forbidden_field_name"


def test_secret_nested_in_a_list_is_found_with_its_path():
    findings = find_secret_like_strings({"budget": {"notes": ["fine", _openai_key()]}})
    assert findings == [{"path": "budget.notes[1]", "reason": "secret_value_pattern"}]


def test_a_realistic_state_with_refs_contains_no_secret():
    state = _state(
        plan_ref="refs/plans/run-1/task-1.json",
        session_id="session_abc123",
        checkpoint_ref=CheckpointRef(ref="refs/codna/runs/run-1/turn-1", sha="a" * 40),
        verify_ref=VerifyRef(ref="verify/run-1/1", passed=True, command_digest="b" * 64),
        pr=PullRequestRef(number=7, url="https://github.com/o/r/pull/7", branch="codna/run-1/task-1"),
        budget=Budget(max_cost_usd=5.0, spent_usd=1.25, max_tokens=100_000, spent_tokens=2_000),
        pending_approval=PendingApproval(kind="approve", reason="ready to open PR", requested_at_index=42),
        attempts=Attempts(implement=2),
        review_verdict="approved",
    )
    assert assert_state_encodable(state.to_payload()) > 0
    assert find_secret_like_strings(state.to_payload()) == []


# -- structural validation --------------------------------------------------------------


def test_schema_version_mismatch_is_refused():
    payload = _state().to_payload()
    payload["schema_version"] = 999
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.code == "unsupported_state_schema"


def test_checkpoint_ref_requires_both_ref_and_sha():
    payload = _state().to_payload()
    payload["checkpoint_ref"] = {"ref": "refs/x/y"}
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.code == "invalid_task_state"
    assert excinfo.value.details["field"] == "checkpoint_ref.sha"


def test_checkpoint_ref_rejects_a_non_hex_sha():
    payload = _state().to_payload()
    payload["checkpoint_ref"] = {"ref": "refs/x/y", "sha": "zzzz"}
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.code == "invalid_checkpoint_sha"


def test_pull_request_ref_requires_https_and_a_positive_number():
    with pytest.raises(TaskStateError) as excinfo:
        PullRequestRef(number=0, url="https://x/y", branch="b").validate()
    assert excinfo.value.code == "invalid_pr_number"
    with pytest.raises(TaskStateError) as excinfo:
        PullRequestRef(number=1, url="http://insecure", branch="b").validate()
    assert excinfo.value.code == "invalid_pr_url"


def test_pending_approval_kind_is_a_closed_set():
    payload = _state().to_payload()
    payload["pending_approval"] = {"kind": "whatever", "reason": "x", "requested_at_index": 0}
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.code == "invalid_pending_approval"


def test_turn_seq_and_event_index_must_be_non_negative():
    with pytest.raises(TaskStateError) as excinfo:
        _state(turn_seq=-1).validate()
    assert excinfo.value.code == "invalid_turn_seq"
    with pytest.raises(TaskStateError) as excinfo:
        _state(last_event_index=-5).validate()
    assert excinfo.value.code == "invalid_last_event_index"


def test_event_index_cannot_go_backwards():
    state = _state().with_events(10)
    with pytest.raises(TaskStateError) as excinfo:
        state.with_events(9)
    assert excinfo.value.code == "event_index_regressed"
    assert state.with_events(10).last_event_index == 10


def test_bool_is_not_accepted_as_an_integer():
    # `True` is an `int` in Python; a boolean where a sequence number belongs is a bug upstream and
    # must be named rather than silently interpreted as 1.
    payload = _state().to_payload()
    payload["turn_seq"] = True
    with pytest.raises(TaskStateError) as excinfo:
        TaskState.from_payload(payload)
    assert excinfo.value.details["field"] == "turn_seq"


# -- budget / attempts / turn key -------------------------------------------------------


def test_budget_exhaustion_is_deterministic_on_both_caps():
    assert Budget(5.0, 4.99, None, 0).exhausted is False
    assert Budget(5.0, 5.0, None, 0).exhausted is True  # crossing the cap counts as exhausted
    assert Budget(None, 0.0, 100, 99).exhausted is False
    assert Budget(None, 0.0, 100, 100).exhausted is True
    assert Budget(None, 1e9, None, 10**9).exhausted is False  # no caps at all


def test_unpriced_spend_counts_tokens_but_not_dollars():
    budget = Budget(1.0, 0.0, None, 0).with_spend(cost_usd=0.5, tokens=100, priced=False)
    assert budget.spent_usd == 0.0
    assert budget.spent_tokens == 100
    assert budget.unpriced_tokens == 100  # never silently treated as $0

    priced = budget.with_spend(cost_usd=0.5, tokens=50)
    assert priced.spent_usd == 0.5
    assert priced.spent_tokens == 150
    assert priced.unpriced_tokens == 100


def test_negative_spend_is_refused():
    with pytest.raises(TaskStateError) as excinfo:
        Budget(None, 0.0, None, 0).with_spend(cost_usd=-1.0, tokens=0)
    assert excinfo.value.code == "invalid_budget"


def test_turn_key_is_stable_for_the_next_turn_and_advances_once():
    state = _state(turn_seq=2)
    assert state.turn_key == "run-1:task-1:3"
    started = state.started_turn(session_id="session_x", turn_id="turn_session_x_1")
    assert started.turn_seq == 3
    assert started.last_turn_key == "run-1:task-1:3"
    # The engine's id for the turn is recorded WITH the key: the pair is what lets a later reader
    # match a key to a turn without asking an engine that may have forgotten the session.
    assert started.last_turn_id == "turn_session_x_1"
    assert started.session_id == "session_x"
    # Re-deriving after the start does NOT re-issue the same key: the next turn is seq 4.
    assert started.turn_key == "run-1:task-1:4"
    # And a retry of the same transition (from the pre-start state) is the same key again.
    assert state.started_turn(session_id="session_x", turn_id="turn_session_x_1").last_turn_key == "run-1:task-1:3"


def test_a_started_turn_records_the_turn_id_and_refuses_a_missing_one():
    with pytest.raises(TaskStateError) as excinfo:
        _state().started_turn(session_id="session_x", turn_id="")
    assert excinfo.value.code == "invalid_turn_id"


def test_a_turn_id_without_its_key_or_session_is_refused():
    """The triple is what a retry matches on, so a partial record is a bug, not a value to coerce.

    Checked on the way OUT (and therefore on the way in, through the same `validate`), which is the
    only path that matters: a state reaching history is a state that was serialized.
    """
    with pytest.raises(TaskStateError) as excinfo:
        _state(last_turn_id="turn_session_x_1").to_payload()
    assert excinfo.value.code == "invalid_task_state"


def test_attempts_increment_one_stage_only():
    attempts = Attempts().incremented("verify").incremented("verify")
    assert attempts.verify == 2
    assert attempts.implement == 0
    assert attempts.total == 2


def test_state_digest_is_stable_and_changes_with_the_references():
    first = _state(plan_ref="refs/a").state_digest()
    assert first == _state(plan_ref="refs/a").state_digest()
    assert first != _state(plan_ref="refs/b").state_digest()


def test_pending_approval_clears_only_via_resume():
    state = _state().awaiting(PendingApproval(kind="approve", reason="r", requested_at_index=1))
    assert state.pending_approval is not None
    assert state.resumed().pending_approval is None
