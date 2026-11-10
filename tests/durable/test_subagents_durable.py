"""The durable path's subagent roster: what travels, what does not, and what the ledger says.

The engine and the reader are tested elsewhere. This covers the two claims the DURABLE path makes:

  1. A ``TaskSpec`` carries the POLICY (``attach_subagents``) and never the roster. The roster is file
     contents — a delegated agent's system prompt is the markdown body — so it is the same class of data
     that ``TaskState`` refuses for rules, and the same rule applies.
  2. The roster is read from the TREE THE TURN EDITS, with the USER layer OFF. A worker's
     ``~/.cursor/agents`` is not the repository's roster, and two workers running the same recorded task
     must build the same session.

Both are asserted, plus the failure modes: no delegation, no opt-in, no tree, no agents — each its own
answer, because "we did not ask" and "we could not" are different facts.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codna.durable.subagents import (
    MAX_ROSTER_BYTES,
    RosterRead,
    SubagentAttachment,
    read_subagents,
    subagent_attachment_state,
    subagent_session_spec,
)
from codna.durable.task_context import TaskContext

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "agents_repo"


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _Context:
    """The minimum a durable context must expose. A real `TaskContext` is exercised at the end."""

    def __init__(self, *, attach: bool = True, spawn: bool = True) -> None:
        self.attach_subagents = attach
        self.enable_spawn_agent = spawn


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(FIXTURE, destination)
    return destination


# ---------------------------------------------------------------------------------------------
# what travels
# ---------------------------------------------------------------------------------------------

def test_the_spec_carries_the_roster_with_bodies():
    """Each entry carries its BODY: the engine builds a delegated-agent tool from it."""
    spec = subagent_session_spec(_Context(), str(FIXTURE))
    assert "subagents" in spec
    verifier = next(entry for entry in spec["subagents"] if entry["name"] == "verifier")
    assert verifier["body"].startswith("You are a verifier.")
    assert verifier["readonly"] is True
    assert verifier["description"].startswith("Verifies a change")


def test_the_spec_carries_refusals_and_warnings_SEPARATELY():
    """A shadowed file is a WARNING (it loaded), never a refusal (it did not)."""
    spec = subagent_session_spec(_Context(), str(FIXTURE))
    assert "subagent_refusals" not in spec
    assert [entry["code"] for entry in spec["subagent_warnings"]] == ["agent_name_shadowed"]


def test_a_refusal_travels_with_its_code(tmp_path: Path):
    write(tmp_path, ".cursor/agents/backgrounder.md", (
        "---\ndescription: x\nis_background: true\n---\nbody\n"
    ))
    write(tmp_path, ".cursor/agents/good.md", "---\ndescription: fine\n---\nbody\n")
    spec = subagent_session_spec(_Context(), str(tmp_path))
    assert [entry["name"] for entry in spec["subagents"]] == ["good"]
    assert [entry["code"] for entry in spec["subagent_refusals"]] == ["agent_background_unsupported"]


# ---------------------------------------------------------------------------------------------
# what does NOT travel (the byte-identical contract)
# ---------------------------------------------------------------------------------------------

def test_no_opt_in_sends_nothing():
    assert subagent_session_spec(_Context(attach=False), str(FIXTURE)) == {}


def test_no_delegation_sends_nothing():
    """A roster the engine would refuse to delegate from is not attached at all."""
    assert subagent_session_spec(_Context(spawn=False), str(FIXTURE)) == {}


def test_no_tree_sends_nothing():
    """No working tree named: reading the shared checkout would be ANOTHER task's roster."""
    assert subagent_session_spec(_Context(), "") == {}


def test_a_tree_with_no_agents_sends_nothing(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert subagent_session_spec(_Context(), str(tmp_path / "empty")) == {}


# ---------------------------------------------------------------------------------------------
# the worker's home is not the repository's roster
# ---------------------------------------------------------------------------------------------

def test_read_never_includes_the_user_layer(tmp_path: Path, monkeypatch):
    """A worker's `~/.cursor/agents` must not reach a repository's turn.

    Set HOME to a directory that DOES define an agent: if `read_subagents` read the user layer, that
    agent would appear. This is the assertion behind "two workers running one recorded task build the
    same session".
    """
    home = tmp_path / "home"
    write(home, ".cursor/agents/machine-only.md", "---\ndescription: private\n---\nbody\n")
    repo = tmp_path / "repo"
    write(repo, ".cursor/agents/project.md", "---\ndescription: shared\n---\nbody\n")
    monkeypatch.setenv("HOME", str(home))

    read = read_subagents(str(repo))
    assert [entry["name"] for entry in read.roster] == ["project"]
    assert all("machine-only" not in entry["source_path"] for entry in read.roster)


# ---------------------------------------------------------------------------------------------
# the ledger answer — four distinct facts
# ---------------------------------------------------------------------------------------------

def test_attachment_reports_attached():
    attachment = subagent_attachment_state(_Context(), str(FIXTURE))
    assert attachment.attached is True
    assert attachment.agents == ("planner", "verifier")
    assert attachment.reason == ""


def test_attachment_reports_not_requested():
    attachment = subagent_attachment_state(_Context(attach=False), str(FIXTURE))
    assert attachment.attached is False
    assert "attach_subagents" in attachment.reason


def test_attachment_reports_delegation_disabled():
    attachment = subagent_attachment_state(_Context(spawn=False), str(FIXTURE))
    assert attachment.attached is False
    assert "enable_spawn_agent" in attachment.reason


def test_attachment_reports_no_roster(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    attachment = subagent_attachment_state(_Context(), str(tmp_path / "empty"))
    assert attachment.attached is False
    assert "defines no subagents" in attachment.reason


def test_attachment_reports_no_context():
    attachment = subagent_attachment_state(None)
    assert attachment.attached is False
    assert attachment.reason


def test_attachment_round_trips_through_its_payload():
    attachment = SubagentAttachment(True, agents=("a", "b"), refused=("agent_x",))
    payload = attachment.to_payload()
    assert payload == {"attached": True, "reason": "", "agents": ["a", "b"], "refused": ["agent_x"]}


def test_attachment_equality_is_structural():
    assert SubagentAttachment(True, agents=("a",)) == SubagentAttachment(True, agents=("a",))
    assert SubagentAttachment(True, agents=("a",)) != SubagentAttachment(True, agents=("b",))


# ---------------------------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------------------------

def test_an_agent_over_the_roster_budget_is_reported_not_dropped(tmp_path: Path):
    write(tmp_path, ".cursor/agents/huge.md", "---\ndescription: big\n---\n" + "x" * 2048)
    write(tmp_path, ".cursor/agents/small.md", "---\ndescription: small\n---\nsmall\n")

    import codna.durable.subagents as module

    original = module.MAX_ROSTER_BYTES
    module.MAX_ROSTER_BYTES = 256
    try:
        read = read_subagents(str(tmp_path))
    finally:
        module.MAX_ROSTER_BYTES = original
    assert [entry["name"] for entry in read.roster] == ["small"]
    assert [entry["code"] for entry in read.refusals] == ["agent_body_over_roster_budget"]


def test_the_budget_is_a_real_bound():
    assert MAX_ROSTER_BYTES >= 64 * 1024


# ---------------------------------------------------------------------------------------------
# the plumbing: the spec field must survive a payload round trip and default conservatively
# ---------------------------------------------------------------------------------------------

def test_the_spec_field_defaults_to_true_on_a_fresh_spec():
    """A run that asks for nothing gets the sensible default: honour what the repository wrote."""
    from codna.durable.contracts import TaskSpec

    spec = TaskSpec(
        run_id="r", task_id="t", task_number=1, brief="b", repo_dir="/tmp", base_branch="main",
    )
    assert spec.attach_subagents is True


def test_a_recorded_payload_without_the_field_decodes_to_FALSE():
    """A run recorded BEFORE this field existed sent no roster, so it must replay with none.

    `True` here would change an in-flight run's sessions on replay — the exact class of history bug the
    other `attach_*` fields document.
    """
    # DELIBERATELY `task_config`, not `workflow_support`: the latter imports `temporalio` at module
    # scope, and the offline `cli secure subsystem` job installs neither `temporalio` nor `langgraph` on
    # purpose (see `tests/durable/test_ci_workflow.py`). `task_spec_from_config` is the SAME decoder
    # without the optional import, so this case runs in the job that proves the CLI works offline
    # instead of skipping there.
    from codna.durable.task_config import CONFIG_TASK_SPEC, task_spec_from_config

    payload = {
        "run_id": "r", "task_id": "t", "task_number": 1, "brief": "b",
        "repo_dir": "/tmp", "base_branch": "main",
    }
    spec = task_spec_from_config({"metadata": {CONFIG_TASK_SPEC: payload}})
    assert spec.attach_subagents is False


def test_the_context_carries_the_policy_not_the_roster():
    """The context is what a request carries; it must not carry file contents."""
    from codna.durable.contracts import TaskSpec

    spec = TaskSpec(
        run_id="r", task_id="t", task_number=1, brief="b", repo_dir="/tmp", base_branch="main",
    )
    context = TaskContext.from_spec(spec)
    assert context.attach_subagents is True
    # No roster field on the context at all: the activity reads the tree.
    assert not hasattr(context, "subagents")


def test_the_spec_validates_the_field_is_a_boolean():
    from codna.durable.contracts import TaskSpec

    spec = TaskSpec(
        run_id="r", task_id="t", task_number=1, brief="b", repo_dir="/tmp", base_branch="main",
        attach_subagents="yes",  # type: ignore[arg-type]
    )
    with pytest.raises(Exception) as caught:
        spec.validate()
    assert "attach_subagents" in str(caught.value)


def test_roster_read_is_an_immutable_answer():
    read = RosterRead()
    assert read.roster == [] and read.refusals == [] and read.warnings == []
    assert read == RosterRead()
