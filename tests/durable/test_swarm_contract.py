"""The swarm contract's own rules, and the one cross-language assertion that keeps them honest.

Two things are tested here and neither needs Temporal:

1. **The approval-profile table matches the ENGINE's.** `swarm.PROFILE_SWARM_PERMISSIONS` mirrors
   `algenta/server/contracts.ts::DEFAULT_APPROVAL_PROFILES` for exactly the two delegation flags. A
   mirror that is merely "believed" is a drift waiting to happen — the engine gaining a profile that
   forbids delegation would leave this side silently permitting what the engine will refuse at session
   create. So the TypeScript is PARSED and compared field by field. If the vendored engine moves, this
   test fails and names the profile.
2. **The validator's rules**, one case each, because each is a distinct way a spec could lie about the
   request it produces.

Why a parser rather than a generated file: the table it mirrors is nine lines of a TypeScript object
literal, and a code generator for it would be more moving parts than the risk it removes. A failure
here is a one-line edit to `swarm.py` with the reason in front of the reader.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from codna.durable.swarm import (
    DURABLE_APPROVAL_PROFILE,
    PROFILE_SWARM_PERMISSIONS,
    SwarmAttachment,
    SwarmConfigError,
    normalize_swarm,
    swarm_attachment_state,
    swarm_session_spec,
)

#: `agent-core/vendor/cline/algenta/server/contracts.ts`, resolved from this file. Skipped rather than
#: failed when the vendored engine is absent: this suite is about the CLI's contract, and a checkout
#: without the engine submodule should still be able to run it.
_ENGINE_CONTRACTS = (
    Path(__file__).resolve().parents[3]
    / "agent-core"
    / "vendor"
    / "cline"
    / "algenta"
    / "server"
    / "contracts.ts"
)

# `repository_review: {` … then `allowSpawnAgent: true,` / `allowAgentTeams: true,` inside that block.
_PROFILE = re.compile(r"^\t(?P<name>[a-z_]+): \{$", re.MULTILINE)
_FLAG = re.compile(r"^\t\t(?P<flag>allowSpawnAgent|allowAgentTeams): (?P<value>true|false),$", re.MULTILINE)


def _engine_profile_table() -> dict[str, dict[str, bool]]:
    """Parse the engine's `DEFAULT_APPROVAL_PROFILES` into ``{profile: {python_field: bool}}``.

    Walks the object literal's top-level entries and, inside each, the two flags. Deliberately narrow:
    it matches the exact shape the file has today, so a restructuring of that literal fails this test
    loudly (with the parse result attached) rather than quietly comparing empty dicts.
    """
    source = _ENGINE_CONTRACTS.read_text(encoding="utf-8")
    start = source.index("export const DEFAULT_APPROVAL_PROFILES")
    # The literal ends at the first `};` at column 0 after the declaration.
    end = source.index("\n};", start)
    body = source[start:end]

    table: dict[str, dict[str, bool]] = {}
    entries = list(_PROFILE.finditer(body))
    for position, match in enumerate(entries):
        chunk_end = entries[position + 1].start() if position + 1 < len(entries) else len(body)
        chunk = body[match.start() : chunk_end]
        flags: dict[str, bool] = {}
        for flag_match in _FLAG.finditer(chunk):
            # The engine spells them allowSpawnAgent/allowAgentTeams; the request fields are the
            # snake_case names the wire uses. One mapping, stated once.
            field = {
                "allowSpawnAgent": "enable_spawn_agent",
                "allowAgentTeams": "enable_agent_teams",
            }[flag_match.group("flag")]
            flags[field] = flag_match.group("value") == "true"
        if flags:
            table[match.group("name")] = flags
    return table


@pytest.mark.skipif(
    not _ENGINE_CONTRACTS.is_file(), reason="the vendored engine is not present in this checkout"
)
def test_the_profile_table_matches_the_engines():
    """The mirror is asserted against the TypeScript, not merely believed to be right."""
    engine = _engine_profile_table()
    # A parse that found nothing would make every other assertion in this test vacuous.
    assert engine, f"parsed no profiles from {_ENGINE_CONTRACTS}"
    assert set(engine) == set(PROFILE_SWARM_PERMISSIONS), (
        "the engine's profiles and the durable mirror disagree. "
        f"engine={sorted(engine)} durable={sorted(PROFILE_SWARM_PERMISSIONS)}"
    )
    for profile, flags in engine.items():
        assert PROFILE_SWARM_PERMISSIONS[profile] == flags, (
            f"profile {profile!r} disagrees: engine={flags} durable={PROFILE_SWARM_PERMISSIONS[profile]}"
        )


def test_the_durable_profile_is_the_engine_default():
    """The durable path sends no `approval_profile`, so it gets the engine's DEFAULT — and the gate
    must be evaluated against that same profile or it would be checking a session that cannot exist."""
    source = _ENGINE_CONTRACTS.read_text(encoding="utf-8") if _ENGINE_CONTRACTS.is_file() else ""
    match = re.search(r'export const DEFAULT_APPROVAL_PROFILE: ApprovalProfile = "([a-z_]+)";', source)
    if not match:
        pytest.skip("the engine is not present, so the default cannot be read")
    assert DURABLE_APPROVAL_PROFILE == match.group(1), (
        f"the durable path assumes {DURABLE_APPROVAL_PROFILE!r} but the engine's default is "
        f"{match.group(1)!r}; a session created without an explicit profile would be governed by a "
        "different table than this side validated against"
    )


# -- the validator -------------------------------------------------------------------------


def test_each_flag_is_independent_and_carried_verbatim():
    """`teams` without `spawn` is a LEGAL session (the engine gates them separately), so it is not
    silently upgraded — deriving one from the other would change what the run asked for."""
    assert normalize_swarm(enable_spawn_agent=False, enable_agent_teams=True) == (False, True)
    assert normalize_swarm(enable_spawn_agent=True, enable_agent_teams=False) == (True, False)
    assert normalize_swarm(enable_spawn_agent=True, enable_agent_teams=True) == (True, True)
    assert normalize_swarm(enable_spawn_agent=False, enable_agent_teams=False) == (False, False)


@pytest.mark.parametrize("value", ["yes", 1, "true", None, [], {}])
def test_a_non_boolean_flag_is_refused_not_coerced(value: Any):
    """The value enters workflow history and then a request body. A truthy string the engine reads as
    ``true`` while this side recorded ``"yes"`` is a spec that does not describe its own request."""
    with pytest.raises(SwarmConfigError) as caught:
        normalize_swarm(enable_spawn_agent=value, enable_agent_teams=False)
    assert caught.value.code == "swarm_config_invalid"
    assert caught.value.details["field"] == "enable_spawn_agent"


def test_an_unknown_profile_fails_closed():
    """A profile this side has not learned about forfeits the gate, so nothing is permitted under it.

    Refused even when neither flag is set, and deliberately: the profile is not a request field the
    caller chooses — the durable path never sends one, so it always resolves to
    `DURABLE_APPROVAL_PROFILE`. An unknown value therefore means the two tables disagree (or the
    constant drifted), which is a configuration bug that must surface at validation rather than become
    "no profile matched, so carry on".
    """
    with pytest.raises(SwarmConfigError) as caught:
        normalize_swarm(enable_spawn_agent=True, enable_agent_teams=False, approval_profile="future")
    assert caught.value.code == "swarm_config_invalid"
    assert caught.value.details["field"] == "approval_profile"
    # ...and asking for NOTHING does not sneak past it either.
    with pytest.raises(SwarmConfigError):
        normalize_swarm(
            enable_spawn_agent=False, enable_agent_teams=False, approval_profile="future"
        )


def test_a_forbidden_capability_names_the_profiles_that_would_allow_it(monkeypatch: pytest.MonkeyPatch):
    """The refusal must be actionable: it names the profile AND where the capability IS available."""
    monkeypatch.setitem(
        PROFILE_SWARM_PERMISSIONS, "restricted", {"enable_spawn_agent": False, "enable_agent_teams": False}
    )
    try:
        with pytest.raises(SwarmConfigError) as caught:
            normalize_swarm(
                enable_spawn_agent=True, enable_agent_teams=False, approval_profile="restricted"
            )
        assert caught.value.code == "swarm_not_permitted"
        assert caught.value.details["approval_profile"] == "restricted"
        assert "repository_fix_gated" in caught.value.details["allowed_profiles"]
    finally:
        PROFILE_SWARM_PERMISSIONS.pop("restricted", None)


# -- the request fragment ------------------------------------------------------------------


def test_the_session_spec_is_empty_when_nothing_was_asked():
    """`{}` rather than two `False`s: an absent key is what keeps a pre-feature request identical."""

    class Silent:
        enable_spawn_agent = False
        enable_agent_teams = False

    assert swarm_session_spec(Silent()) == {}


def test_the_session_spec_carries_both_flags_when_asked():
    class Swarm:
        enable_spawn_agent = True
        enable_agent_teams = True

    assert swarm_session_spec(Swarm()) == {
        "enable_spawn_agent": True,
        "enable_agent_teams": True,
    }


def test_the_attachment_state_tells_apart_not_asked_from_told_apart():
    """Three answers, because "never asked" and "asked and dropped" are different facts."""

    class Silent:
        enable_spawn_agent = False
        enable_agent_teams = False

    silent = swarm_attachment_state(Silent())
    assert silent.attached is False
    assert "does not delegate" in silent.reason
    assert swarm_attachment_state(None).attached is False

    class Half:
        enable_spawn_agent = True
        enable_agent_teams = False

    half = swarm_attachment_state(Half())
    assert half.attached is True and half.spawn is True and half.teams is False

    assert swarm_attachment_state(Silent()).to_payload()["attached"] is False


def test_the_attachment_payload_is_json_safe():
    """It goes into a ledger row, so every value must survive a JSON round trip."""
    import json

    class Swarm:
        enable_spawn_agent = True
        enable_agent_teams = True

    payload = swarm_attachment_state(Swarm()).to_payload()
    assert json.loads(json.dumps(payload)) == payload


def test_the_attachment_is_comparable():
    """`__eq__` exists so a test can assert on the pair as a value rather than field by field."""
    left = SwarmAttachment(True, "", spawn=True, teams=True)
    right = SwarmAttachment(True, "", spawn=True, teams=True)
    assert left == right
    assert left != SwarmAttachment(True, "", spawn=True, teams=False)
    assert (left == "not-an-attachment") is False
