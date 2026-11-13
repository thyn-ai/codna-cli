"""The repository POLICY is a second narrowing the delegation flags cannot override.

THE DEFECT THESE PIN, from `run-20261112T223954Z` (a real 898-record ledger):

* the run's note said ``delegation is attached — enable_spawn_agent, enable_agent_teams``;
* its session snapshot said ``{"enableSpawn": true, "enableTeams": true}``;
* its approval profile (``repository_fix_gated``) grants all 18 ``team_*`` names;
* its ``.codna/policy.yaml`` said ``tools: {allow: [read_files, editor, run_commands]}``.

The model called ``team_run_task`` and got
``Model tried to call unavailable tool 'team_run_task'``, then the run recorded ZERO
``team_progress`` events, ZERO subagent sessions and ZERO dispatches.

``enable_agent_teams`` is a GATE; the tools are a GRANT, narrowed twice — by the profile's
``allowedTools`` and again by the policy at session create
(``run-server.ts`` → ``applyPolicyToToolPolicies`` → ``narrowToolPolicies``, which sets
``enabled: false`` on every name the policy does not permit). This suite asserts the CLI's half of
the invariant: a run whose policy strips the team tools SAYS SO in its ledger instead of letting an
operator discover it from a mid-run tool failure.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

# Both are imported at MODULE scope, and that is the point: `delegation_policy` carries no workflow
# runtime, so this file also runs in the lane that does NOT install the `durable` extra (this
# directory's `conftest.py` states the rule). Importing `delegation_policy_note` from
# `capability_report` — which schedules an activity through `workflow_support` → `temporalio` — is
# what turned a pure string builder into a collection error there.
from codna.durable.contracts import DelegationPolicyRequest
from codna.durable.delegation_policy import delegation_policy, delegation_policy_note
from codna.durable.swarm import (
    PROMPT_NAMED_TEAM_TOOLS,
    policy_stripped_team_tools,
)

#: The policy that produced the measured failure, verbatim in shape.
_TODO_APP_POLICY = """\
version: 1
tools:
  allow: [read_files, editor, run_commands]
  deny: []
write_scope:
  paths: ["**"]
command_allow_list:
  commands: [pytest, npm, node, python3]
budget:
  max_cost_usd: 5.0
"""

#: A policy that permits the orchestration surface, so a run under it CAN form a team.
_SWARM_POLICY = """\
version: 1
tools:
  allow: [read_files, editor, run_commands, spawn_agent, team_run_task, team_send_message,
          team_broadcast, team_status, team_await_runs, team_mission_log,
          team_review_outcome_fragment]
  deny: []
write_scope:
  paths: ["**"]
command_allow_list:
  commands: [pytest]
"""


def _repo(tmp_path: Path, policy: str | None) -> Path:
    """A repository directory, with the policy at the ONE path the reader consults."""
    if policy is not None:
        codna_dir = tmp_path / ".codna"
        codna_dir.mkdir(parents=True, exist_ok=True)
        (codna_dir / "policy.yaml").write_text(policy, encoding="utf-8")
    return tmp_path


def test_a_policy_that_permits_three_tools_strips_every_prompt_named_team_tool(tmp_path: Path):
    """The measured case: seven names the prompt spells, none of them permitted."""
    stripped = policy_stripped_team_tools(
        _repo(tmp_path, _TODO_APP_POLICY), enable_agent_teams=True
    )
    # EVERY name, not a sample: the prompt's two sentences together spell all seven, so a subset would
    # still let the prompt name a tool the model does not have.
    assert stripped == PROMPT_NAMED_TEAM_TOOLS


def test_a_policy_that_permits_the_team_tools_strips_nothing(tmp_path: Path):
    """The other direction, so the report cannot be unconditional noise."""
    assert (
        policy_stripped_team_tools(
            _repo(tmp_path, _SWARM_POLICY), enable_agent_teams=True
        )
        == ()
    )


def test_a_repository_with_no_policy_strips_nothing(tmp_path: Path):
    """No policy means the approval profile decides alone, and every profile the durable path can
    resolve grants all seven — so there is nothing to report."""
    assert (
        policy_stripped_team_tools(_repo(tmp_path, None), enable_agent_teams=True) == ()
    )


def test_delegation_not_requested_is_never_reported(tmp_path: Path):
    """A run that never asked to delegate cannot be missing delegation: reporting the policy there
    would make the ordinary case noisy and the note meaningless."""
    assert (
        policy_stripped_team_tools(
            _repo(tmp_path, _TODO_APP_POLICY), enable_agent_teams=False
        )
        == ()
    )


def test_a_partial_grant_names_only_what_is_missing(tmp_path: Path):
    """A policy may keep the mailbox and drop the dispatch tool. The report must name exactly the
    absent names, because that is the operator's edit list."""
    policy = """\
version: 1
tools:
  allow: [team_run_task, team_send_message, team_status]
write_scope:
  paths: ["**"]
command_allow_list:
  commands: [pytest]
"""
    stripped = policy_stripped_team_tools(
        _repo(tmp_path, policy), enable_agent_teams=True
    )
    # The three permitted are NOT reported…
    assert "team_run_task" not in stripped
    assert "team_send_message" not in stripped
    assert "team_status" not in stripped
    # …and every other prompt-named tool is.
    assert set(stripped) == set(PROMPT_NAMED_TEAM_TOOLS) - {
        "team_run_task",
        "team_send_message",
        "team_status",
    }


def test_a_deny_entry_wins_over_an_allow_entry(tmp_path: Path):
    """`deny` beats `allow` in the policy reader, so a tool cannot be granted back by listing it
    twice. A report that read only `allow` would claim a tool the engine will not enable."""
    policy = """\
version: 1
tools:
  allow: [team_run_task, team_send_message, team_status, team_await_runs,
          team_mission_log, team_broadcast, team_review_outcome_fragment]
  deny: [team_run_task]
write_scope:
  paths: ["**"]
command_allow_list:
  commands: [pytest]
"""
    stripped = policy_stripped_team_tools(
        _repo(tmp_path, policy), enable_agent_teams=True
    )
    assert stripped == ("team_run_task",)


def test_the_note_names_the_policy_file_and_every_stripped_tool():
    """The sentence an operator acts on: it must say WHICH file and WHICH names.

    ``None`` for a run with nothing stripped is the other half — the ordinary case records no extra
    row, so the first note an operator has always seen is unchanged.

    The note is imported at module scope from `codna.durable.delegation_policy`, the runtime-free
    module, so this test — and this whole file — runs in the lane WITHOUT the extra.
    """
    message = delegation_policy_note(
        "run-20260101T000000Z", PROMPT_NAMED_TEAM_TOOLS, "/repo/.codna/policy.yaml"
    )
    assert message is not None
    assert "/repo/.codna/policy.yaml" in message
    for name in PROMPT_NAMED_TEAM_TOOLS:
        assert name in message
    # It names the fix rather than only the symptom.
    assert "tools.allow" in message

    assert delegation_policy_note("run-x", (), "/repo/.codna/policy.yaml") is None


def test_the_activity_is_the_one_that_reads_the_policy(tmp_path: Path):
    """The report runs in an ACTIVITY, and that is load-bearing rather than incidental.

    Temporal's workflow sandbox refuses filesystem access from workflow code — measured as
    ``RestrictedWorkflowAccessError: Cannot access os.path from inside a workflow`` when this logic
    was first written inline in ``RunWorkflow._report_capabilities``. A regression that moved it back
    into the workflow would fail ``test_swarm_e2e.py`` with a sandbox violation rather than with a
    wrong answer, which is why the seam is asserted here as well: this calls the ACTIVITY BODY and
    proves the answer is the same one the workflow records.

    Called from `codna.durable.delegation_policy`, so it runs in the lane without the extra. The
    registration is asserted from SOURCE (`activity_steps` binds this body, `activities` registers
    that binding under the activity's name) — which is what "the activity is the one that reads the
    policy" means — and by IDENTITY when the extra is installed, which source text cannot replace.
    """
    repo = _repo(tmp_path, _TODO_APP_POLICY)
    outcome = delegation_policy(
        DelegationPolicyRequest(
            run_id="run-20260101T000000Z", repo_dir=str(repo), enable_agent_teams=True
        )
    )
    assert outcome.stripped == PROMPT_NAMED_TEAM_TOOLS
    assert outcome.policy_path.endswith(".codna/policy.yaml")

    # And a repository that permits them reports nothing, so the ordinary case is silent.
    quiet = delegation_policy(
        DelegationPolicyRequest(
            run_id="run-20260101T000001Z",
            repo_dir=str(_repo(tmp_path / "ok", _SWARM_POLICY)),
            enable_agent_teams=True,
        )
    )
    assert quiet.stripped == ()

    _assert_the_registered_activity_is_this_body()


def _assert_the_registered_activity_is_this_body() -> None:
    """`activity_steps.delegation_policy` IS this body, and `activities` registers it under the name."""
    durable_dir = Path(__file__).resolve().parents[2] / "codna" / "durable"
    steps = ast.parse(
        (durable_dir / "activity_steps.py").read_text(encoding="utf-8"),
        filename="activity_steps.py",
    )
    # The alias the body is imported under, then the plain module-level assignment re-exporting it:
    # `from .delegation_policy import delegation_policy as _read_delegation_policy` and
    # `delegation_policy = _read_delegation_policy`. A second definition HERE would be the copy
    # nothing asserts, so its absence is part of the property.
    imported = {
        alias.asname
        for node in steps.body
        if isinstance(node, ast.ImportFrom) and node.module == "delegation_policy"
        for alias in node.names
        if alias.name == "delegation_policy"
    }
    assert imported == {"_read_delegation_policy"}, (
        f"`activity_steps` does not import the runtime-free body: {imported}"
    )
    rebound = [
        node.value.id
        for node in steps.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and any(
            isinstance(target, ast.Name) and target.id == "delegation_policy"
            for target in node.targets
        )
    ]
    assert rebound == ["_read_delegation_policy"], (
        f"`activity_steps.delegation_policy` is no longer the imported body: {rebound}"
    )
    assert not [
        node
        for node in steps.body
        if isinstance(node, ast.FunctionDef) and node.name == "delegation_policy"
    ], "`activity_steps` grew a second `delegation_policy` implementation; the activity body is the copy nothing asserts"

    activities = (durable_dir / "activities.py").read_text(encoding="utf-8")
    assert "activity.defn(name=NAME_DELEGATION_POLICY)(" in activities, (
        "the policy read is no longer registered under NAME_DELEGATION_POLICY"
    )
    assert "NAME_DELEGATION_POLICY: delegation_policy_activity" in activities, (
        "the registered policy read is not in the activity dispatch table"
    )

    if importlib.util.find_spec("temporalio") is None:
        return
    # Only meaningful with the extra installed: the registered callable and the function this test
    # called must be ONE object, which is the property `activities.body_for` relies on.
    from codna.durable import activity_steps

    assert activity_steps.delegation_policy is delegation_policy, (
        "the registered activity body is not the function this test asserted against"
    )


def test_the_prompt_named_list_is_the_one_the_prompt_spells():
    """The seven names are asserted against the TypeScript, so a prompt that starts naming an eighth
    tool fails here rather than in a run.

    Parsed rather than imported because the engine is TypeScript: the two arrays the sentences are
    built from are the source of truth, and this reads their literal contents.
    """
    adapter = (
        Path(__file__).resolve().parents[3]
        / "agent-core"
        / "vendor"
        / "cline"
        / "algenta"
        / "adapter.ts"
    )
    if not adapter.is_file():
        pytest.skip("the vendored engine is not present in this checkout")
    source = adapter.read_text(encoding="utf-8")

    def block(name: str) -> tuple[str, ...]:
        # The two literals are formatted differently on purpose by their authors (one name per line vs.
        # all on one), so this reads the QUOTED STRINGS in the literal rather than its line shape.
        start = source.index(f"const {name} = [")
        end = source.index("] as const;", start)
        return tuple(re.findall(r'"([^"]+)"', source[start:end]))

    mailbox = block("TEAM_MAILBOX_TOOLS")
    critique = block("TEAM_CRITIQUE_TOOLS")
    assert (*mailbox, *critique) == PROMPT_NAMED_TEAM_TOOLS, (
        "the prompt's tool names and the report's list disagree; a run would either report a tool "
        "the prompt never named or stay silent about one it does"
    )
