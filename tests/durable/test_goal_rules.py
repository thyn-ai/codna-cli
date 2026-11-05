"""Project rules on the ENGINE path, and the MCP gap reported rather than hidden (D5).

The bug this closes: a human running `codna fix` had their repository's conventions honoured
(`fix_run` read `AGENTS.md`/`.codna/rules` and attached them as `project_guidance`), while a durable
run — the path that works unattended for hours — sent a bare prompt. Nothing read the rules at all.

So the assertions are about the SESSION the engine is actually asked to create: the text must arrive
in the field the engine folds into its system prompt, and it must be read from the tree the turn is
about to edit (a task in its own worktree may be on a branch that changed the rules).

The MCP half is the opposite kind of test: the engine's create-session contract has no MCP field, so
the requirement is that the gap is REPORTED — in the ledger, with the reason — rather than silently
dropped or falsely claimed.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_support import _project_rules  # noqa: E402
from codna.durable.contracts import TaskContext, TaskSpec  # noqa: E402
from codna.durable.rules import (  # noqa: E402
    MAX_RULES_CHARS,
    mcp_attachment_state,
    project_rules_for,
)
from tests.durable import fakes  # noqa: E402


def _spec(repo: Path, **overrides: Any) -> TaskSpec:
    payload: dict[str, Any] = {
        "run_id": "run-rules",
        "task_id": "task-1",
        "task_number": 1,
        "brief": "Fix the retry loop",
        "repo_dir": str(repo),
        "verify_commands": ("true",),
        "engine": "plain",
    }
    payload.update(overrides)
    return TaskSpec(**payload)  # type: ignore[arg-type]


# -- the reader ----------------------------------------------------------------------------------


def test_a_repository_with_no_rules_yields_an_empty_answer(tmp_path: Path):
    """Empty is a real answer: most repositories have no rules file, and a run must still work."""
    repo = fakes.make_repo(tmp_path / "repo")
    assert project_rules_for(str(repo)) == ""
    assert _project_rules(TaskContext.from_spec(_spec(repo)), str(repo)) == ""


def test_agents_md_reaches_the_turn(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("# Rules\n- Use pytest\n- Never touch generated files\n", encoding="utf-8")
    text = project_rules_for(str(repo))
    assert "Use pytest" in text
    assert "AGENTS.md" in text  # the reader labels which file each block came from


def test_codna_rules_reaches_the_turn(tmp_path: Path):
    """`.codna/rules` is the plan's name for this; the reader knows it and the turn gets it."""
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / ".codna").mkdir()
    (repo / ".codna" / "rules").write_text("Prefer small diffs.\n", encoding="utf-8")
    assert "Prefer small diffs." in project_rules_for(str(repo))


def test_the_rules_are_bounded_for_the_PROMPT_not_just_the_file(tmp_path: Path):
    """A 16 KB rules file would be a larger system prompt on every turn of every task of every run."""
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("x" * 40_000, encoding="utf-8")
    text = project_rules_for(str(repo))
    assert text
    assert len(text) <= MAX_RULES_CHARS


def test_the_policy_can_switch_attachment_off(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    context = TaskContext.from_spec(_spec(repo, attach_project_rules=False))
    assert _project_rules(context, str(repo)) == ""


def test_the_rules_are_read_from_the_tree_the_TURN_EDITS(tmp_path: Path):
    """A task in its own worktree may be on a branch that CHANGED the rules.

    Reading them from the shared checkout would give the turn yesterday's conventions — the same class
    of bug as committing in the wrong tree, and the reason this is a parameter.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("SHARED CHECKOUT RULES\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "AGENTS.md").write_text("BRANCH RULES\n", encoding="utf-8")
    context = TaskContext.from_spec(_spec(repo))
    assert "BRANCH RULES" in _project_rules(context, str(worktree))
    assert "SHARED CHECKOUT RULES" not in _project_rules(context, str(worktree))


def test_no_tree_named_means_no_rules_rather_than_the_shared_checkout(tmp_path: Path):
    """An unnamed tree falls back to nothing: guessing `repo_dir` is the bug this exists to prevent."""
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("SHARED\n", encoding="utf-8")
    context = TaskContext.from_spec(_spec(repo))
    assert _project_rules(context, "") == ""


# -- the session the engine is asked to create ---------------------------------------------------


def _create_session_spec(
    harness: fakes.Harness,
    spec: TaskSpec,
    *,
    tree: str,
    env: dict[str, str] | None = None,
    unset: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Drive ONE real plan turn and return the create-session spec the fake engine received.

    ``env``/``unset`` exist because an MCP server's credential references are resolved from the
    WORKER's environment at session-create time, so a test that asserts expansion must control that
    environment rather than depend on the machine it runs on.
    """
    from codna.durable.activity_steps import plan_task

    asyncio.run  # noqa: B018 - the steps are sync; this documents that no loop is involved
    spec.validate()
    harness.runtime  # the harness is already installed as the process default
    with patch.dict(os.environ, env or {}, clear=False):
        for name in unset:
            os.environ.pop(name, None)
        plan_task(TaskContext.from_spec(spec).plan_request(spec.brief, spec.initial_state()))
    assert harness.sidecar.sessions, "the plan turn created no session"
    return dict(harness.sidecar.sessions[-1]["spec"])


def test_the_plan_turn_carries_the_rules_in_the_field_the_engine_reads(tmp_path: Path):
    """`injected_context` is what the engine folds into the system prompt — asserted, not assumed."""
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("# Rules\n- Use the sandbox\n", encoding="utf-8")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo)

    sent = _create_session_spec(harness, spec, tree=str(repo))

    assert "Use the sandbox" in sent.get("injected_context", ""), sent
    # ...and the identity this feature must not disturb.
    assert sent["multi_turn"] is True
    assert sent["owner"] == "codna-run-run-rules"


def test_a_task_that_opts_out_sends_no_injected_context(tmp_path: Path):
    repo = fakes.make_repo(tmp_path / "repo")
    (repo / "AGENTS.md").write_text("# Rules\n- Use the sandbox\n", encoding="utf-8")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo, attach_project_rules=False)

    sent = _create_session_spec(harness, spec, tree=str(repo))

    assert "injected_context" not in sent, sent


def test_a_payload_recorded_before_rules_existed_decodes_to_sending_none():
    """Replay safety: an old history must produce the prompts it recorded, not new ones."""
    from codna.durable.workflow_support import spec_from

    payload = _spec(Path("/tmp/repo")).to_payload()
    del payload["attach_project_rules"]
    decoded = spec_from(payload)
    assert decoded.attach_project_rules is False


def test_the_policy_travels_on_the_context_and_the_spec(tmp_path: Path):
    spec = _spec(tmp_path / "repo")
    assert spec.attach_project_rules is True  # the default: honour what the repo wrote
    assert spec.to_payload()["attach_project_rules"] is True
    assert TaskContext.from_spec(spec).attach_project_rules is True
    with pytest.raises(ValueError):
        _spec(tmp_path / "repo", attach_project_rules="yes").validate()


# -- MCP: carried, reported, and refused when malformed -------------------------------------------


def test_mcp_with_no_servers_configured_says_so_and_names_the_fix():
    """The ordinary case. `not attached` with a reason that says WHY and WHAT to set — the honest
    answer, and the one a run with no MCP config must produce."""
    attachment = mcp_attachment_state()
    assert attachment.attached is False
    assert attachment.servers == ()
    assert "no mcp servers are configured" in attachment.reason.lower()
    # The reason names the FIX, not a vague "unsupported": this is the field that carries it.
    assert "mcp_servers" in attachment.reason


def test_a_run_that_CONFIGURED_servers_reports_them_as_attached(tmp_path: Path):
    """The gap this closed: a configured server is no longer refused. It is CARRIED, and the run's
    ledger says which ones — so an operator can tell "attached" from "configured and dropped"."""
    repo = fakes.make_repo(tmp_path / "repo")
    context = TaskContext.from_spec(
        _spec(
            repo,
            mcp_servers=[
                {"name": "docs", "transport": {"type": "stdio", "command": "npx"}},
                {"name": "gh", "transport": {"type": "sse", "url": "https://example.com/sse"}},
            ],
        )
    )
    attachment = mcp_attachment_state(context)
    assert attachment.attached is True
    assert attachment.reason == ""
    assert attachment.servers == ("docs", "gh")


def test_every_server_DISABLED_is_not_the_same_answer_as_none_configured(tmp_path: Path):
    """Two different facts, two different sentences: configuration that exists and is switched off is
    not the absence of configuration, and a ledger that blurred them would hide a misconfiguration."""
    repo = fakes.make_repo(tmp_path / "repo")
    context = TaskContext.from_spec(
        _spec(
            repo,
            mcp_servers=[
                {"name": "docs", "disabled": True, "transport": {"type": "stdio", "command": "npx"}},
            ],
        )
    )
    attachment = mcp_attachment_state(context)
    assert attachment.attached is False
    assert "disabled" in attachment.reason.lower()
    assert "no mcp servers are configured" not in attachment.reason.lower()


def test_a_malformed_mcp_config_is_REFUSED_at_the_boundary_not_dropped(tmp_path: Path):
    """Fail closed: the whole point is that a bad block never becomes a session that runs without the
    server it was told to use. Each defect is refused, and refused as its own coded error."""
    repo = fakes.make_repo(tmp_path / "repo")
    bad_configs = [
        # A transport with no command.
        [{"name": "docs", "transport": {"type": "stdio"}}],
        # An unknown transport type.
        [{"name": "docs", "transport": {"type": "grpc", "url": "https://e.com"}}],
        # A name that would produce an illegal tool name.
        [{"name": "bad name", "transport": {"type": "stdio", "command": "x"}}],
        # A url on a stdio transport: the caller believed they configured a remote server.
        [{"name": "docs", "transport": {"type": "stdio", "command": "x", "url": "https://e.com"}}],
        # A LITERAL secret. A spec is history, so this may only ever be a reference.
        [{"name": "docs", "transport": {"type": "stdio", "command": "x", "env": {"TOKEN": "ghp_live"}}}],
        # Duplicate names.
        [
            {"name": "docs", "transport": {"type": "stdio", "command": "x"}},
            {"name": "docs", "transport": {"type": "stdio", "command": "y"}},
        ],
    ]
    for config in bad_configs:
        with pytest.raises(ValueError):
            _spec(repo, mcp_servers=config).validate()


def test_a_secret_NEVER_travels_in_the_spec_only_its_variable_name(tmp_path: Path):
    """The provider rule, applied to MCP: the spec names the SOURCE, the activity resolves the value."""
    repo = fakes.make_repo(tmp_path / "repo")
    spec = _spec(
        repo,
        mcp_servers=[
            {
                "name": "docs",
                "transport": {
                    "type": "stdio",
                    "command": "npx",
                    "env": {"DOCS_TOKEN": "${DOCS_TOKEN}"},
                },
            },
        ],
    )
    payload = spec.to_payload()
    encoded = json.dumps(payload)
    # The NAME is there; the shorthand was canonicalised to a reference.
    assert payload["mcp_servers"][0]["transport"]["env"]["DOCS_TOKEN"] == {"from_env": "DOCS_TOKEN"}
    # …and no value. The spec is history, and history is what an operator reads.
    assert "DOCS_TOKEN" in encoded


def test_the_configured_servers_reach_the_CREATE_SESSION_request(tmp_path: Path):
    """The join that makes it a feature rather than a field: what the engine is actually POSTed."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(
        repo,
        mcp_servers=[
            {"name": "docs", "transport": {"type": "stdio", "command": "npx", "args": ["-y", "docs-mcp"]}},
        ],
    )

    sent = _create_session_spec(harness, spec, tree=str(repo), env={})

    assert sent["mcp_servers"] == [
        {"name": "docs", "transport": {"type": "stdio", "command": "npx", "args": ["-y", "docs-mcp"]}},
    ]


def test_an_MCP_env_reference_is_expanded_from_the_worker_and_never_serialized(tmp_path: Path):
    """The value is read at session-create time from the WORKER's environment; the request carries it
    because the engine needs it to launch the server, and the SPEC does not because that is history."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(
        repo,
        mcp_servers=[
            {
                "name": "docs",
                "transport": {
                    "type": "stdio",
                    "command": "npx",
                    "env": {"DOCS_TOKEN": "${DOCS_TOKEN}"},
                },
            },
        ],
    )

    sent = _create_session_spec(harness, spec, tree=str(repo), env={"DOCS_TOKEN": "value-from-worker"})

    assert sent["mcp_servers"][0]["transport"]["env"] == {"DOCS_TOKEN": "value-from-worker"}
    # The SPEC still carries only the reference.
    assert spec.to_payload()["mcp_servers"][0]["transport"]["env"]["DOCS_TOKEN"] == {
        "from_env": "DOCS_TOKEN"
    }


def test_an_UNSET_mcp_variable_refuses_the_turn_and_names_the_variable(tmp_path: Path):
    """Not a silent omission: a server whose credential is missing from this worker must fail loudly,
    because a server that starts without it will fail at the first authenticated call instead."""
    from codna.durable.mcp import McpConfigError

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(
        repo,
        mcp_servers=[
            {
                "name": "docs",
                "transport": {"type": "stdio", "command": "npx", "env": {"DOCS_TOKEN": "${DOCS_TOKEN}"}},
            },
        ],
    )

    with pytest.raises(McpConfigError) as raised:
        _create_session_spec(harness, spec, tree=str(repo), env={})
    assert raised.value.code == "mcp_env_missing"
    assert raised.value.details.get("variable") == "DOCS_TOKEN"
    # NOTHING was created: the refusal happens before the POST, so no session exists to clean up.
    assert harness.sidecar.sessions == []


def test_a_payload_recorded_before_mcp_existed_decodes_to_sending_none():
    """Replay safety: an in-flight run must create the same sessions it recorded, not gain a field."""
    from codna.durable.workflow_support import spec_from

    payload = _spec(Path("/tmp/repo")).to_payload()
    del payload["mcp_servers"]
    decoded = spec_from(payload)
    assert decoded.mcp_servers == ()
    assert mcp_attachment_state(TaskContext.from_spec(decoded)).attached is False


def test_the_worker_wide_default_from_the_environment_is_used_when_the_spec_names_none(tmp_path: Path):
    """One export configures every run: `CODNA_MCP_SERVERS` is the default, the spec is the override."""
    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    spec = _spec(repo)
    assert spec.mcp_servers == ()

    sent = _create_session_spec(
        harness,
        spec,
        tree=str(repo),
        env={"CODNA_MCP_SERVERS": json.dumps({"docs": {"transport": {"type": "stdio", "command": "npx"}}})},
    )
    assert sent["mcp_servers"][0]["name"] == "docs"


def test_a_malformed_CODNA_MCP_SERVERS_is_refused_rather_than_ignored():
    """An operator who exported broken JSON must learn at start-up, not from a run that quietly had
    no tools. Never a default, never a silent `()`."""
    from codna.durable.mcp import McpConfigError, mcp_servers_from_env

    with pytest.raises(McpConfigError):
        mcp_servers_from_env({"CODNA_MCP_SERVERS": "{not json"})
    with pytest.raises(McpConfigError):
        mcp_servers_from_env({"CODNA_MCP_SERVERS": json.dumps({"a": {"transport": {"type": "nope"}}})})
    # Absent is the ordinary case and is `()` — not an error.
    assert mcp_servers_from_env({}) == ()


def test_a_run_records_the_mcp_state_in_its_ledger(tmp_path: Path):
    """An operator reads from the RUN what the engine path was given — either direction.

    A run with no MCP config says so (and names the field that would carry one); a run with servers
    says WHICH. That is what turns "we did not" into a fact an operator can act on, and it is the
    assertion a one-sided note could not make.
    """
    pytest.importorskip("langgraph", reason="the `durable` extra (langgraph) is not installed")
    pytest.importorskip("codna.durable.graph", reason="the `durable` extra cannot build the task graph")

    from datetime import timedelta

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from codna.durable.activities import ACTIVITIES, activity_executor
    from codna.durable.workflows import RunWorkflow, registered_workflows

    repo = fakes.make_repo(tmp_path / "repo")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    harness = fakes.build_harness(repo, runs_root)
    task_payload = _spec(repo).to_payload()
    task_payload["mcp_servers"] = [
        {"name": "docs", "transport": {"type": "stdio", "command": "npx"}},
    ]
    payload = {
        "run_id": "run-rules",
        "repo_dir": str(repo),
        "base_branch": "main",
        "tasks": [task_payload],
        "engine": "plain",
        "continue_as_new_tasks": 10,
        "continue_as_new_history_events": 100_000,
    }

    async def scenario(client: Any) -> Any:
        handle = await client.start_workflow(
            RunWorkflow.run,
            args=[payload, None],
            id="run-run-rules",
            task_queue="codna-rules-tests",
            run_timeout=timedelta(hours=24),
        )
        child = client.get_workflow_handle("task-run-rules-task-1")
        for _ in range(600):
            try:
                status = await child.query("status")
            except Exception:  # noqa: BLE001 - not started yet
                await asyncio.sleep(0.02)
                continue
            if status.get("awaiting_approval"):
                break
            await asyncio.sleep(0.02)
        return await handle.query(RunWorkflow.status)

    async def wrapper() -> Any:
        env = await WorkflowEnvironment.start_time_skipping()
        try:
            with activity_executor() as executor:
                async with Worker(
                    env.client,
                    task_queue="codna-rules-tests",
                    workflows=list(registered_workflows()),
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                ):
                    with env.auto_time_skipping_disabled():
                        return await scenario(env.client)
        finally:
            await env.shutdown()

    asyncio.run(wrapper())

    notes = [
        record.payload
        for record in harness.ledger("run-rules").records()
        if record.kind == "note" and record.payload.get("event") == "notification"
    ]
    mcp_notes = [note for note in notes if "MCP is" in note.get("message", "")]
    assert len(mcp_notes) == 1, notes
    message = mcp_notes[0]["message"]
    # The DIRECTION is right: a configured server is reported as attached, by name.
    assert "attached" in message.lower()
    assert "not attached" not in message.lower()
    assert "docs" in message


def test_a_LITERAL_env_value_may_not_enter_history_at_all(tmp_path: Path):
    """The negative that matters most here: a secret cannot be written into the ledger by accident.

    A `TaskSpec` is workflow history and therefore the ledger, and the ledger is an audit artifact
    operators read. So the ONLY representable stdio env entry is a reference. A literal — whether it
    looks like a key or not — is refused outright rather than serialized, because "we only drop
    key-shaped values" is a rule that fails on the first credential that does not look like one.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    with pytest.raises(ValueError) as raised:
        _spec(
            repo,
            mcp_servers=[
                {
                    "name": "docs",
                    "transport": {"type": "stdio", "command": "npx", "env": {"TOKEN": "plain-looking-value"}},
                },
            ],
        ).validate()
    assert "from_env" in str(raised.value)
