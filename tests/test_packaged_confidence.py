"""The packaged decision-plan confidence must be a REAL per-run signal, not a fixed number."""
from __future__ import annotations

from codna.packaged_agent_runner import PackagedAgentRunResult
from codna.packaged_repository_advanced import _confidence


def _r(terminal_state="succeeded", status="succeeded", patch="--- a/x\n+++ b/x\n+fix\n"):
    return PackagedAgentRunResult(
        status=status, terminal_state=terminal_state, agent_run_id="a", session_id="s",
        text="", files_changed=[], telemetry={}, artifacts={}, runtime={}, patch_diff=patch,
    )


def test_no_change_is_zero_confidence():
    assert _confidence(_r(patch=""), []) == 0.0
    assert _confidence(_r(), []) == 0.0            # succeeded but nothing changed


def test_confidence_varies_with_blast_radius():
    one = _confidence(_r(), ["a.py"])
    two = _confidence(_r(), ["a.py", "b.py"])
    many = _confidence(_r(), ["a.py", "b.py", "c.py", "d.py", "e.py"])
    assert one > two > many                        # a tighter fix is more trustworthy


def test_clean_success_beats_hitting_a_limit():
    clean = _confidence(_r(terminal_state="succeeded", status="succeeded"), ["a.py"])
    limited = _confidence(_r(terminal_state="max_iterations", status="max_iterations"), ["a.py"])
    errored = _confidence(_r(terminal_state="error", status="error"), ["a.py"])
    assert clean > limited > errored


def test_never_fabricates_the_old_static_value_or_certainty():
    # the removed hardcode was a flat 0.78 for any single success; now it derives from signals
    vals = {_confidence(_r(), fs) for fs in (["a.py"], ["a.py", "b.py"], ["a.py", "b.py", "c.py"])}
    assert 0.78 not in vals                         # not the old constant
    assert len(vals) > 1                            # genuinely varies
    assert all(0.0 <= v <= 0.9 for v in vals)       # never claims certainty pre-verification
