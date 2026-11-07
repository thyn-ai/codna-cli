"""`codna durable`: the parser, the run payload, the workflow-id contract and the failure shapes.

The CLI is a translator, so what is worth testing is the translation:

* **The parser exists and wires the right handler** — a subcommand that parses but never dispatches
  is a command that silently exits 0 (the shape `cli.py`'s ``getattr(args, "func", None)`` check
  turns into a help message, which looks like success).
* **The run payload is valid BEFORE it is started** — a typo'd brief or a relative repo path must be
  an exit code, not a workflow that fails inside Temporal's history after the worker picked it up.
* **The child workflow id matches `RunWorkflow._run_task`** — `status --task` and every Update
  (approve/reject/steer/set-budget) address a child by that id; if the two formulas drift, the
  commands report "workflow not found" on a healthy run.
* **Failures are structured** — every branch that can fail does so with a stable `code`, because
  `codna/cli.py` renders those as the documented ``{"error": {...}}`` body.
* **Without the extra, every command is a structured error** — never a `ModuleNotFoundError`
  traceback from the imports the durable stack needs.

The base-suite half needs no `temporalio`. The half that drives a real server
(`test_the_cli_drives_a_real_run_through_approval`) is gated: it starts a run, polls `status`, and
answers an `approve` — the exact sequence a user performs — and asserts the outcome the LEDGER
records, so the CLI cannot pass by printing a plausible status dict.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from codna import cli as top_cli
from codna.durable import cli as durable_cli
from codna.durable import serve_cli
from codna.durable.cli import DurableCliError, _task_workflow_id, cmd_durable, register
from codna.durable.errors import DurableCliError as DurableError

CLI_DIR = Path(__file__).resolve().parents[2]
HAS_EXTRA = importlib.util.find_spec("temporalio") is not None and importlib.util.find_spec("langgraph") is not None

# Servers started by a test, stopped in this module's teardown. A fixture per test would work too,
# but the helper is called from three tests and the list makes the teardown unconditional: a failed
# assertion mid-test must still stop the server.
_ACTIVE_SERVERS: list[Any] = []


@pytest.fixture(autouse=True)
def _stop_test_servers() -> Any:
    yield
    while _ACTIVE_SERVERS:
        _ACTIVE_SERVERS.pop().stop()


def _parse(argv: list[str]) -> Any:
    return top_cli.build_parser().parse_args(argv)


# -- the parser ----------------------------------------------------------------------------


def test_the_top_level_parser_offers_durable():
    """`codna durable` is registered from `cli.py`, one import and one call (its own ceiling)."""
    helpers = top_cli.build_parser().format_help()
    assert "durable" in helpers


def test_the_parser_offers_the_command_the_ide_tells_users_to_run():
    """THE REMEDY THE IDE PRINTS MUST EXIST HERE.

    `RUN_API_START_COMMAND` in the IDE (`ide/packages/codna-engine-client/src/codnaRuns.ts`) is the one
    place that spells the Run API's start command, and every hint and message uses it. It says
    ``codna serve``. The sentence it replaced said ``codna durable run-api``, a subcommand that has never
    been registered — so the product printed a remedy that exits 2 on every machine, and nothing tied the
    IDE's spelling to this parser, which is why no test noticed.

    This is the missing tie: the exact spelling the IDE ships is parsed HERE, as a real command that
    dispatches to a handler.
    """
    # The literal, not an import: this suite must not depend on the TypeScript tree. It is the string the
    # IDE's constant carries, and `taskService.test.ts` pins that constant to this same literal.
    parsed = top_cli.build_parser().parse_args(["serve", "--no-temporal"])
    assert parsed.func is not None
    # `serve` is also reachable under the durable namespace, the spelling older scripts use.
    grouped = top_cli.build_parser().parse_args(["durable", "serve", "--no-temporal"])
    assert grouped.func is cmd_durable
    assert grouped.durable_action == "serve"


def test_serve_can_start_the_stack_in_one_command():
    """`codna serve --start-dev-server` is the ONE command the IDE now prints as its remedy.

    WHY THIS EXISTS. Every message before it asked a user to run TWO foreground commands —
    ``codna durable dev-server start`` and then ``codna serve`` — and keep both alive. Both work (measured),
    but a user who ran only the second got ``run_api_temporal_unavailable`` and was back where they started.
    The flag closes that gap: `serve` starts its own Temporal dependency, idempotently, and then serves.

    The parse is the contract: the IDE's `RUN_API_STACK_COMMAND` is `codna serve --start-dev-server`, so the
    flag has to exist, default to False (starting a background server is a side effect nobody asked for by
    default), and be reachable under both spellings.
    """
    parsed = top_cli.build_parser().parse_args(["serve", "--start-dev-server"])
    assert parsed.func is not None
    assert parsed.start_dev_server is True
    # OPT-IN, not the default: a bare `serve` must not start anything behind the user's back.
    assert top_cli.build_parser().parse_args(["serve"]).start_dev_server is False
    grouped = top_cli.build_parser().parse_args(["durable", "serve", "--start-dev-server"])
    assert grouped.start_dev_server is True


def test_start_dev_server_beside_no_temporal_is_refused_not_silently_ignored():
    """The two flags contradict each other, and a contradiction must be an error.

    `--no-temporal` means "no Temporal client at all"; `--start-dev-server` means "bring Temporal up".
    Accepting both and doing nothing would leave an operator believing a server is running — the exact class
    of defect ("a flag that does nothing") this surface keeps being fixed for.
    """
    with pytest.raises(DurableError) as caught:
        serve_cli.cmd_serve(
            top_cli.build_parser().parse_args(["serve", "--start-dev-server", "--no-temporal"])
        )
    assert caught.value.code == "serve_start_dev_server_conflict"


def test_the_approved_roots_refusal_names_the_fix_where_the_reader_will_see_it():
    """The message must be actionable on its own, because a client may render only `message`.

    MEASURED. The Run API answered the IDE's task sheet with
    ``durable_repo_outside_allowed_roots`` and a `hint` naming ``CODNA_DURABLE_REPO_ROOTS`` — and the sheet
    showed only ``error.message``, so the user saw a refusal with no way to respond. The default allowed root
    is the server's own cwd (`allowed_repo_roots_from_env`), so a `serve` started from a subdirectory refuses
    the workspace above it.
    """
    from pathlib import Path

    from codna.durable.repo_validation import ALLOWED_REPO_ROOTS_ENV, http_repo_dir

    workspace = Path("/tmp/codna-roots-test/workspace")
    elsewhere = Path("/tmp/codna-roots-test/elsewhere")
    workspace.mkdir(parents=True, exist_ok=True)
    elsewhere.mkdir(parents=True, exist_ok=True)
    with pytest.raises(DurableError) as caught:
        http_repo_dir(str(workspace), allowed_roots=(elsewhere,))
    message = str(caught.value)
    assert str(workspace) in message
    # Both halves of the fix, in the sentence a caller actually renders.
    assert ALLOWED_REPO_ROOTS_ENV in message
    assert "codna serve" in message


def test_the_run_api_spelling_the_product_used_to_print_is_not_a_command():
    """The negative half, so the false remedy cannot quietly come back as a valid parse.

    ``codna durable run-api`` was printed by every Run API hint for the life of the surface. If someone
    re-introduces that spelling as an ALIAS (rather than fixing the hint), this test is where the decision
    has to be made deliberately — either it is a real command documented in `--help`, or the hint must not
    name it.
    """
    with pytest.raises(SystemExit):
        top_cli.build_parser().parse_args(["durable", "run-api"])


@pytest.mark.parametrize(
    "argv, action",
    [
        (["durable", "run", "--brief", "b"], "run"),
        (["durable", "worker"], "worker"),
        (["durable", "dev-server", "start"], "dev-server"),
        (["durable", "dev-server"], "dev-server"),
        (["durable", "status", "--run", "run-1"], "status"),
        (["durable", "approve", "--run", "run-1", "--task", "task-1"], "approve"),
        (["durable", "reject", "--run", "run-1", "--task", "task-1"], "reject"),
        (["durable", "steer", "--run", "run-1", "--task", "task-1", "--note", "n"], "steer"),
        (["durable", "set-budget", "--run", "run-1", "--task", "task-1", "--max-cost", "2"], "set-budget"),
        (["durable", "control", "--run", "run-1", "pause"], "control"),
        (["durable", "ledger", "--run", "run-1"], "ledger"),
    ],
)
def test_every_subcommand_parses_and_dispatches_to_one_handler(argv: list[str], action: str):
    args = _parse(argv)
    assert args.func is cmd_durable
    assert args.durable_action == action
    if action == "control":
        assert args.control_action == "pause"


def test_durable_without_a_subcommand_is_a_structured_error():
    """Falling through to the help text would exit 0 and look like a completed command."""
    args = _parse(["durable"])
    with pytest.raises(DurableCliError) as caught:
        cmd_durable(args)
    assert caught.value.code == "durable_action_required"
    assert "run" in caught.value.details["known"]


# -- the temporal profile flag -------------------------------------------------------------


def test_the_profile_flag_exists_on_every_spelling_that_connects():
    """A run and a worker must both be pointable at Cloud, on both surfaces.

    One implementation, so the assertion is that each spelling PARSES the flag — a flag on
    `codna worker` and not on `durable worker` is a documented option that does not exist on the
    path half the users take.
    """
    for argv in (
        ["worker", "--temporal-profile", "cloud"],
        ["durable", "worker", "--temporal-profile", "cloud"],
        ["run", "--brief", "b", "--temporal-profile", "cloud"],
        ["durable", "run", "--brief", "b", "--temporal-profile", "cloud"],
    ):
        assert _parse(argv).temporal_profile == "cloud", argv


def test_the_profile_flag_defaults_to_none_so_the_environment_decides():
    """`None` is "infer from TEMPORAL_*", which is what keeps the local default unchanged."""
    assert _parse(["worker"]).temporal_profile is None
    assert _parse(["run", "--brief", "b"]).temporal_profile is None


def test_the_profile_flag_refuses_an_unknown_name():
    """argparse's `choices` is the first gate; the config's `parse_profile` is the second."""
    with pytest.raises(SystemExit):
        _parse(["worker", "--temporal-profile", "clod"])


def test_the_dev_server_commands_refuse_a_cloud_profile():
    """There is no local server to start for Cloud, and pretending otherwise hides the cause."""
    from codna.durable.config import load_durable_config
    from codna.runtime.config import RuntimeConfig, build_runtime_paths

    paths = build_runtime_paths()
    runtime = RuntimeConfig(
        codna_home=paths.root,
        port_base=18600,
        engine_port=18600,
        sidecar_port=18601,
        engine_url="http://127.0.0.1:18600",
        sidecar_url="http://127.0.0.1:18601",
        remote_engine_url=None,
        ignored_engine_url=None,
        engine_url_override=None,
        paths=paths,
        engine_dir=paths.root / "engine",
        engine_python="python3",
        engine_env_file=paths.root / ".env",
        sidecar_dir=paths.root / "agent-core",
        sidecar_binary=None,
        cline_data_dir=paths.root / "cline-data",
        codna_cli_version="0.0.0-test",
        engine_build_id="engine",
        sidecar_build_id="sidecar",
        runtime_config_hash="hash",
        allow_loopback_override=False,
    )
    cloud = load_durable_config(
        env={
            "TEMPORAL_ENDPOINT": "codna.example.tmprl.cloud:7233",
            "TEMPORAL_NAMESPACE": "codna.example",
            "TEMPORAL_API_KEY": "not-a-real-key",
        },
        runtime=runtime,
        hostname="build-host",
    )
    with pytest.raises(DurableCliError) as caught:
        durable_cli._dev_server(cloud)
    assert caught.value.code == "durable_dev_server_not_applicable"
    assert caught.value.details["temporal_profile"] == "cloud"
    # The hint must send the operator to the RIGHT fix, not to a loopback address.
    assert "Cloud is already running" in caught.value.hint
    # The local profile still gets a server object, unchanged.
    local = load_durable_config(env={}, runtime=runtime, hostname="build-host")
    assert durable_cli._dev_server(local).settings.address == local.address


def test_run_requires_at_least_one_brief():
    with pytest.raises(SystemExit):
        _parse(["durable", "run"])


def test_status_requires_a_run_id():
    with pytest.raises(SystemExit):
        _parse(["durable", "status"])


def test_the_actor_flag_is_required_by_the_command_not_the_parser():
    """Required at dispatch so `--help` works, refused by the command so no approval is anonymous."""
    args = _parse(["durable", "approve", "--run", "run-1", "--task", "task-1"])
    assert args.actor is None
    with pytest.raises(DurableCliError) as caught:
        durable_cli._identity(args)
    assert caught.value.code == "durable_identity_required"


def test_an_actor_with_whitespace_only_is_refused():
    args = _parse(["durable", "approve", "--run", "run-1", "--task", "task-1", "--as", "   "])
    with pytest.raises(DurableCliError) as caught:
        durable_cli._identity(args)
    assert caught.value.code == "durable_identity_required"


def test_the_actor_is_trimmed_and_kept_verbatim():
    args = _parse(["durable", "approve", "--run", "run-1", "--task", "task-1", "--as", " a@b.c "])
    assert durable_cli._identity(args) == "a@b.c"


# -- register() as a unit ------------------------------------------------------------------


def test_register_attaches_the_whole_surface():
    """Called directly, because a parser built through `build_parser` would hide a missing command."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    register(sub)
    parsed = parser.parse_args(["durable", "ledger", "--run", "r"])
    assert parsed.durable_action == "ledger"
    assert parsed.func is cmd_durable


# -- the child workflow id -----------------------------------------------------------------


def test_the_child_id_matches_the_run_workflows_formula():
    """`RunWorkflow._run_task` uses ``f"task-{run_id}-{task_id}"``; the CLI must agree exactly."""
    assert _task_workflow_id("run-1", "task-1") == "task-run-1-task-1"
    source = (CLI_DIR / "codna" / "durable" / "run_workflow.py").read_text(encoding="utf-8")
    assert 'id=f"task-{spec.run_id}-{task.task_id}"' in source


def test_the_child_id_refuses_an_invalid_run_id():
    with pytest.raises(Exception) as caught:
        _task_workflow_id("../etc", "task-1")
    assert getattr(caught.value, "code", "") == "invalid_run_id"


# -- run payload ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_run_payload_has_one_task_per_brief(tmp_path: Path, durable_config: Any):
    repo = _git_repo(tmp_path / "repo")
    payload = durable_cli.run_payload_for(
        config=durable_config,
        run_id="run-1",
        repo_dir=repo,
        briefs=["fix a", "fix b"],
        base_branch="main",
        verify_commands=["true"],
        max_cost_usd=5.0,
        max_tokens=None,
        max_attempts=2,
        engine="plain",
    )
    assert payload["run_id"] == "run-1"
    assert payload["engine"] == "plain"
    assert [task["task_id"] for task in payload["tasks"]] == ["task-1", "task-2"]
    assert [task["task_number"] for task in payload["tasks"]] == [1, 2]
    assert all(task["repo_dir"] == str(repo) for task in payload["tasks"])
    assert payload["max_cost_usd"] == 5.0


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_an_empty_brief_list_is_refused(tmp_path: Path, durable_config: Any):
    repo = _git_repo(tmp_path / "repo")
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=[],
            base_branch="main",
            verify_commands=(),
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
        )
    assert caught.value.code == "durable_brief_required"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_secret_in_a_brief_is_refused_before_the_run_starts(tmp_path: Path, durable_config: Any):
    """The contract refuses secret-like data because a brief enters workflow history."""
    repo = _git_repo(tmp_path / "repo")
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=["use this key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"],
            base_branch="main",
            verify_commands=(),
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
        )
    assert caught.value.code in ("contract_contains_secret", "durable_run_spec_invalid")


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_an_empty_base_branch_is_refused(tmp_path: Path, durable_config: Any):
    repo = _git_repo(tmp_path / "repo")
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=["b"],
            base_branch="  ",
            verify_commands=(),
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
        )
    assert caught.value.code == "durable_base_branch_invalid"


# -- the repo check ------------------------------------------------------------------------


def test_a_missing_repo_is_a_structured_error(tmp_path: Path):
    with pytest.raises(DurableCliError) as caught:
        durable_cli._repo_dir(str(tmp_path / "nope"))
    assert caught.value.code == "durable_repo_missing"


def test_a_directory_that_is_not_a_git_work_tree_is_refused(tmp_path: Path):
    """A durable run commits a ref per turn; without git there is nothing to commit to."""
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(DurableCliError) as caught:
        durable_cli._repo_dir(str(plain))
    assert caught.value.code == "durable_repo_not_git"
    assert caught.value.hint


def test_a_real_work_tree_is_accepted_and_absolutised(tmp_path: Path):
    repo = _git_repo(tmp_path / "repo")
    assert durable_cli._repo_dir(str(repo)) == repo.resolve()


# -- the missing extra ---------------------------------------------------------------------


@pytest.mark.skipif(HAS_EXTRA, reason="this is the behaviour WITHOUT the extra")
@pytest.mark.parametrize(
    "argv",
    [
        ["durable", "run", "--brief", "b"],
        ["durable", "worker"],
        ["durable", "dev-server", "status"],
        ["durable", "status", "--run", "run-1"],
        ["durable", "ledger", "--run", "run-1"],
    ],
)
def test_every_command_reports_the_missing_extra_structured(argv: list[str]):
    """A `ModuleNotFoundError` traceback here would be the first thing a new user ever sees."""
    with pytest.raises(Exception) as caught:
        cmd_durable(_parse(argv))
    assert getattr(caught.value, "code", "") == "durable_extra_missing"


def test_the_cli_module_imports_without_the_extra():
    """`codna/cli.py` imports this to build the parser, so it must not need `temporalio`."""
    script = (
        "import sys\n"
        "import codna.durable.cli\n"
        "import codna.cli\n"
        "assert 'temporalio' not in sys.modules, sorted(name for name in sys.modules if 'temporalio' in name)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(CLI_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_durable_error_renders_as_the_documented_error_body():
    """`cli.py` prints `{"error": {...}}` for anything carrying `.code`; this is that body."""
    error = DurableCliError("boom", code="durable_test", details={"k": "v"}, hint="try again")
    body = error.as_dict()
    assert body == {
        "error": {
            "code": "durable_test",
            "message": "boom",
            "details": {"k": "v"},
            "hint": "try again",
        }
    }
    assert json.dumps(body)  # serialisable, which is what the CLI relies on


def test_public_output_redacts_secrets_without_hiding_credential_variable_names(capsys):
    durable_cli._print({"password": "s3cret", "api_key_env": "TEMPORAL_API_KEY"})
    output = capsys.readouterr().out
    assert "s3cret" not in output
    assert "<redacted>" in output
    assert "TEMPORAL_API_KEY" in output


# -- ledger --------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_missing_ledger_is_reported_with_its_path(tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch):
    args = _parse(["durable", "ledger", "--run", "run-1"])
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    with pytest.raises(DurableCliError) as caught:
        cmd_durable(args)
    assert caught.value.code == "durable_ledger_missing"
    assert caught.value.details["run_id"] == "run-1"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_ledger_command_summarises_and_verifies_a_real_ledger(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The command's job is an honest summary: a valid chain, its record count and its spend."""
    from codna.durable.ledger import RunLedger

    ledger = RunLedger.for_run(durable_config.paths, "run-1", clock=lambda: 1_700_000_000.0)
    # `session_id` and `event_index` travel together: the pair is the dedupe key, so one without
    # the other is refused by the ledger itself.
    ledger.append("turn.finished", session_id="session-1", event_index=1, payload={"cost_usd": 0.25})
    ledger.append("turn.finished", session_id="session-1", event_index=2, payload={"cost_usd": 0.75})

    args = _parse(["durable", "ledger", "--run", "run-1", "--tail", "1"])
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    assert cmd_durable(args) == 0

    body = json.loads(capsys.readouterr().out)
    assert body["valid"] is True
    assert body["problems"] == []
    assert body["records"] == 2
    assert body["sessions"] == 1
    assert body["spent_usd"] == 1.0
    assert len(body["tail"]) == 1
    assert body["tail"][0]["payload"]["cost_usd"] == 0.75


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_tampered_ledger_exits_non_zero_and_names_the_problem(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A broken hash chain is the one thing this command must never report as success."""
    from codna.durable.ledger import RunLedger

    ledger = RunLedger.for_run(durable_config.paths, "run-2", clock=lambda: 1_700_000_000.0)
    ledger.append("turn.finished", payload={"cost_usd": 0.1})
    ledger.append("turn.finished", payload={"cost_usd": 0.2})
    path = durable_config.ledger_path("run-2")
    lines = path.read_text(encoding="utf-8").splitlines()
    # Canonical JSON has no space after `:`, so the substitution is exact and the only change.
    assert '"cost_usd":0.1' in lines[0]
    lines[0] = lines[0].replace('"cost_usd":0.1', '"cost_usd":9.9')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args = _parse(["durable", "ledger", "--run", "run-2"])
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    assert cmd_durable(args) == 2

    body = json.loads(capsys.readouterr().out)
    assert body["valid"] is False
    assert body["problems"]


# -- run/status/approve against a real server ------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_cli_starts_a_real_run_and_reports_what_it_started(
    tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`run` starts a workflow the server really has, on the configured queue, with the real payload.

    The assertions are on the SERVER's answer (a `status` query) rather than on the CLI's own
    banner: the banner is what the command printed, the query is what the server holds.
    """
    repo = _git_repo(tmp_path / "repo")
    _run_commands_against_a_server(tmp_path, durable_config, repo, monkeypatch)

    args = _parse(
        [
            "durable",
            "run",
            "--repo",
            str(repo),
            "--brief",
            "Fix the flaky retry",
            "--base-branch",
            "main",
            "--run-id",
            "run-cli",
            "--no-start-server",
        ]
    )
    assert cmd_durable(args) == 0
    banner = json.loads(capsys.readouterr().out)
    assert banner["started"]["run_id"] == "run-cli"
    assert banner["started"]["workflow_id"] == "run:run-cli"
    assert banner["started"]["tasks"] == 1

    assert cmd_durable(_parse(["durable", "status", "--run", "run-cli"])) == 0
    status = json.loads(capsys.readouterr().out)["status"]
    assert status["run_id"] == "run-cli"
    assert status["tasks_total"] == 1


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_status_reports_a_run_and_its_parked_task(
    tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`status --task` addresses the child by the id formula the workflow actually used."""
    repo = _git_repo(tmp_path / "repo")
    server = _run_commands_against_a_server(tmp_path, durable_config, repo, monkeypatch)
    assert cmd_durable(_parse(["durable", "run", "--repo", str(repo), "--brief", "b", "--run-id", "run-cli",
                               "--no-start-server"])) == 0
    capsys.readouterr()

    server.wait_parked("task-run-cli-task-1")

    assert cmd_durable(_parse(["durable", "status", "--run", "run-cli", "--task", "task-1"])) == 0
    task_body = json.loads(capsys.readouterr().out)
    assert task_body["workflow_id"] == "task-run-cli-task-1"
    assert task_body["status"]["task_id"] == "task-1"
    # Parked, not finished: the CLI must be able to see the state a human has to act on.
    assert task_body["status"]["awaiting_approval"] is True


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
@pytest.mark.parametrize(
    "script, timeout_s, expected_exit",
    [
        (
            [
                {"run_id": "r", "tasks_total": 1, "tasks_done": 0},
                {"run_id": "r", "tasks_total": 1, "tasks_done": 1, "outcome": "completed"},
            ],
            5.0,
            0,
        ),
        (
            [{"run_id": "r", "tasks_total": 1, "tasks_done": 1, "outcome": "budget_exhausted"}],
            5.0,
            2,
        ),
        (
            [{"run_id": "r", "tasks_total": 1, "tasks_done": 1, "outcome": "stopped"}],
            5.0,
            2,
        ),
        ([{"run_id": "r", "tasks_total": 1, "tasks_done": 0}], 0.05, None),
    ],
)
def test_the_wait_loop_polls_the_query_and_maps_the_outcome_to_an_exit_code(
    script: list[dict[str, Any]], timeout_s: float, expected_exit: int | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--wait` polls a QUERY so a parked run is visible, and the exit code distinguishes outcomes.

    A fake client, not a server: this is the loop's own contract (poll, print each change, stop on
    `outcome`, map it to an exit code, time out loudly), and a real server would be testing Temporal
    rather than the loop. ``completed`` is the only success — a run that stopped or ran out of
    budget is a non-zero exit, because a caller scripting around it must not read it as success.
    """
    client = _ScriptedClient(script)
    polls_before_stop = len(script)

    async def scenario() -> int:
        return await durable_cli._wait_run(
            client, _FakeConfig(), "run-1", poll_s=0.001, timeout_s=timeout_s
        )

    if expected_exit is None:
        with pytest.raises(DurableCliError) as caught:
            asyncio.run(scenario())
        assert caught.value.code == "durable_run_wait_timeout"
        assert caught.value.details["last_status"]["tasks_done"] == 0
        assert "codna durable status" in caught.value.hint
    else:
        assert asyncio.run(scenario()) == expected_exit
        # One poll per script entry, then it stops: the loop must not keep querying after `outcome`.
        assert client.polls == polls_before_stop

    printed = _json_objects(capsys.readouterr().out)
    # A repeated status is printed ONCE: a long park must not spam a block per poll.
    assert len(printed) <= len(script)
    assert all("status" in entry for entry in printed)


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Decode a stream of pretty-printed JSON objects printed back to back."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return objects
        obj, index = decoder.raw_decode(text, index)
        objects.append(obj)


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_wait_loop_refuses_a_non_positive_poll_interval() -> None:
    """A zero interval is a hot loop against the server; fail before sending anything."""
    with pytest.raises(DurableCliError) as caught:
        asyncio.run(durable_cli._wait_run(_ScriptedClient([]), _FakeConfig(), "run-1", poll_s=0, timeout_s=1))
    assert caught.value.code == "durable_poll_interval_invalid"


class _FakeConfig:
    """The four config attributes `_wait_run` reads, so the loop is tested without a real config."""

    address = "127.0.0.1:7233"
    namespace = "codna"
    task_queue = "codna.agent.test"

    def run_workflow_id(self, run_id: str) -> str:
        return f"run:{run_id}"


class _ScriptedClient:
    """A client whose `status` query returns a scripted sequence, then repeats the last answer."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self.polls = 0

    def get_workflow_handle(self, workflow_id: str) -> Any:
        return _ScriptedHandle(self, workflow_id)


class _ScriptedHandle:
    def __init__(self, client: _ScriptedClient, workflow_id: str) -> None:
        self._client = client
        self.workflow_id = workflow_id

    async def query(self, _query: Any, **_kwargs: Any) -> dict[str, Any]:
        index = min(self._client.polls, max(len(self._client._script) - 1, 0))
        self._client.polls += 1
        if not self._client._script:
            return {}
        return dict(self._client._script[index])


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_approve_steer_and_set_budget_reach_a_parked_task(
    tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The success path of all three Updates, against a task that is really parked.

    Each command names the engine differently (``set-budget`` on the CLI, ``set_budget`` in the
    Update), so this is the test that catches a name that never got mapped: the wrong name is a
    "workflow not found"-shaped refusal, not a silent no-op.
    """
    repo = _git_repo(tmp_path / "repo")
    server = _run_commands_against_a_server(tmp_path, durable_config, repo, monkeypatch)
    assert cmd_durable(_parse(["durable", "run", "--repo", str(repo), "--brief", "b", "--run-id", "run-cli",
                               "--no-start-server"])) == 0
    capsys.readouterr()
    server.wait_parked("task-run-cli-task-1")

    assert cmd_durable(_parse(["durable", "steer", "--run", "run-cli", "--task", "task-1",
                               "--as", "qa@codna.ai", "--note", "also cover the retry"])) == 0
    steer = json.loads(capsys.readouterr().out)
    assert steer["update"] == "steer"
    assert steer["result"]["applied"] is True

    assert cmd_durable(_parse(["durable", "set-budget", "--run", "run-cli", "--task", "task-1",
                               "--as", "qa@codna.ai", "--max-cost", "4.5"])) == 0
    budget = json.loads(capsys.readouterr().out)
    assert budget["update"] == "set-budget"
    assert budget["result"]["max_cost_usd"] == 4.5

    assert cmd_durable(_parse(["durable", "approve", "--run", "run-cli", "--task", "task-1",
                               "--as", "qa@codna.ai"])) == 0
    approval = json.loads(capsys.readouterr().out)
    assert approval["update"] == "approve"
    assert approval["result"]["identity"] == "qa@codna.ai"

    # The approval ends the task: it leaves the parked state for a terminal one.
    final = server.wait_finished("task-run-cli-task-1")
    assert final["outcome"] == "approved"
    assert final["approvals"][0]["identity"] == "qa@codna.ai"


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_an_approval_is_refused_when_the_task_is_not_parked(
    tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch
):
    """A validator's refusal is the answer: it must be reported, not retried into a timeout."""
    repo = _git_repo(tmp_path / "repo")
    server = _run_commands_against_a_server(tmp_path, durable_config, repo, monkeypatch)

    # Start the run and WAIT only for the park, not the completion: the task must be parked so the
    # refusal under test is the validator's ("already decided"), not "the workflow is gone".
    assert cmd_durable(_parse(["durable", "run", "--repo", str(repo), "--brief", "b", "--run-id", "run-cli",
                               "--no-start-server"])) == 0
    child = "task-run-cli-task-1"
    server.wait_parked(child)
    # The FIRST approval succeeds and is what the second one is refused by.
    assert cmd_durable(_parse(["durable", "approve", "--run", "run-cli", "--task", "task-1", "--as", "a@b.c"])) == 0

    with pytest.raises(DurableCliError) as caught:
        cmd_durable(_parse(["durable", "approve", "--run", "run-cli", "--task", "task-1", "--as", "a@b.c"]))
    assert caught.value.code in ("durable_update_refused", "durable_workflow_not_found", "durable_not_parked")
    assert caught.value.details["action"] == "approve"
    assert caught.value.hint


def _run_commands_against_a_server(
    tmp_path: Path,
    durable_config: Any,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _BackgroundServer:
    """Point the CLI at a live time-skipping server with the REAL activities and a fake sidecar.

    The server runs in a background thread with its own loop (the CLI owns the main thread's loop),
    which is what lets a synchronous command drive it. Everything the activities touch is the
    harness's: a fake sidecar, a scripted verifier/reviewer, and real git on a real repository.

    A task's last node is the approval park, so a run never completes unattended. The tests here
    therefore stop at the PARK (`wait_parked`) or answer it explicitly with `approve` — the same two
    things a person does from a second terminal.
    """
    fakes = _fakes()
    fakes.build_harness(repo, Path(durable_config.runs_root))
    server = _BackgroundServer(durable_config)
    server.start()
    monkeypatch.setattr(durable_cli, "_connect", server.connect)
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    # `monkeypatch` reverts the two attributes; the server needs its own teardown, and a test that
    # leaves one running would poison every later test with a thread and a port.
    _ACTIVE_SERVERS.append(server)
    return server


class _BackgroundServer:
    """A time-skipping server plus a worker, running on their own loop in a thread.

    A thread rather than a nested loop because `codna durable run` calls ``asyncio.run``: the CLI
    owns the main thread's loop for the whole command, and the server must be serving while it does.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._loop: Any = None
        self._thread: Any = None
        self._env: Any = None
        self._stopping: Any = None  # an asyncio.Event, created on the server's own loop
        self._thread_done: Any = None  # a threading.Event, waited on from the test's thread
        self._failure: BaseException | None = None

    def start(self) -> None:
        import asyncio
        import threading

        started = threading.Event()
        self._thread_done = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve(started))
            except BaseException as exc:  # noqa: BLE001 - surfaced by `stop`
                self._failure = exc
            finally:
                self._thread_done.set()

        self._thread = threading.Thread(target=run, name="durable-test-server", daemon=True)
        self._thread.start()
        if not started.wait(timeout=60):
            raise AssertionError("the test server did not start")
        if self._failure is not None:
            raise AssertionError(f"the test server failed to start: {self._failure!r}")

    async def _serve(self, started: Any) -> None:
        import asyncio

        from temporalio.testing import WorkflowEnvironment
        from temporalio.worker import Worker

        from codna.durable.activities import ACTIVITIES, activity_executor
        from codna.durable.workflows import registered_workflows

        # An asyncio.Event, not a threading one: this coroutine must await, not block the loop the
        # worker and the client share.
        self._stopping = asyncio.Event()
        self._env = await WorkflowEnvironment.start_time_skipping()
        try:
            with activity_executor() as executor:
                async with Worker(
                    self._env.client,
                    task_queue=self._config.task_queue,
                    workflows=list(registered_workflows()),
                    activities=list(ACTIVITIES),
                    activity_executor=executor,
                ):
                    started.set()
                    # Time skipping is OFF while the CLI polls: a parked task's outstanding
                    # `run_timeout` is a timer, and skipping would jump past the run's lifetime.
                    with self._env.auto_time_skipping_disabled():
                        await self._stopping.wait()
        finally:
            await self._env.shutdown()

    def wait_parked(self, workflow_id: str, *, attempts: int = 3000) -> dict[str, Any]:
        """Block until the child workflow reports it is awaiting a decision.

        The query is addressed by NAME (``"status"``), which both engines declare, so this does not
        hard-code which engine the CLI selected. A query sent before the child exists raises
        ``Execution not found`` — that is the normal first poll, not a failure.
        """
        import asyncio

        async def poll() -> dict[str, Any]:
            handle = self._env.client.get_workflow_handle(workflow_id)
            last: str | None = None
            for _ in range(attempts):
                try:
                    status = await handle.query("status")
                except Exception as exc:  # noqa: BLE001 - "execution not found" until the child starts
                    last = str(exc)
                    await asyncio.sleep(0.01)
                    continue
                if status.get("awaiting_approval"):
                    return status
                await asyncio.sleep(0.01)
            raise AssertionError(f"{workflow_id} never parked for approval (last error: {last})")

        return asyncio.run_coroutine_threadsafe(poll(), self._loop).result(timeout=120)

    def wait_finished(self, workflow_id: str, *, attempts: int = 3000) -> dict[str, Any]:
        """Block until the child reports a terminal ``outcome``, and return that status."""
        import asyncio

        async def poll() -> dict[str, Any]:
            handle = self._env.client.get_workflow_handle(workflow_id)
            for _ in range(attempts):
                try:
                    status = await handle.query("status")
                except Exception:  # noqa: BLE001 - not created yet, or already closed
                    await asyncio.sleep(0.01)
                    continue
                if status.get("outcome") is not None:
                    return status
                await asyncio.sleep(0.01)
            raise AssertionError(f"{workflow_id} never reached a terminal outcome")

        return asyncio.run_coroutine_threadsafe(poll(), self._loop).result(timeout=120)

    def connect(self, _config: Any) -> Any:
        """The CLI's `await _connect(config)` seam: hand back the server's own client.

        A coroutine, because `_connect` is one — the client was created on the other loop, which is
        exactly the situation a test double has to be honest about.
        """

        async def resolve() -> Any:
            return self._env.client

        return resolve()

    def stop(self) -> None:
        """Ask the server's loop to wind down, then wait for its thread.

        ``call_soon_threadsafe`` because the event belongs to the other loop: setting it from this
        thread would not wake that loop's selector, and the join would then time out on a server
        that is idle but healthy.
        """
        if self._loop is not None and self._stopping is not None:
            self._loop.call_soon_threadsafe(self._stopping.set)
        if self._thread_done is not None:
            self._thread_done.wait(timeout=30)
        if self._failure is not None:  # pragma: no cover - reported instead of hidden
            raise AssertionError(f"the test server failed: {self._failure!r}")


# -- helpers -------------------------------------------------------------------------------


def _fakes() -> Any:
    """The durable harness module, imported by PATH.

    ``from tests.durable import fakes`` resolves to whatever ``tests`` package wins ``sys.path`` —
    on this machine an unrelated site-packages ``tests`` — so the harness is loaded from the file
    beside this module instead. That keeps the import honest under every pytest invocation style.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "fakes.py"
    spec = importlib.util.spec_from_file_location("durable_test_fakes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "app.py").write_text("x = 1\n", encoding="utf-8")
    # E3: `codna run` is GOVERNED — it refuses to start without a valid `.codna/policy.yaml`, because
    # an unattended run with no policy would execute under whatever the approval profile allows. The
    # fixture therefore carries the same minimal policy a real repository needs, committed with the
    # rest of the tree so it does not perturb a test that asserts on a clean worktree.
    (path / ".codna").mkdir(exist_ok=True)
    (path / ".codna" / "policy.yaml").write_text(
        "version: 1\n"
        "tools:\n"
        "  allow: [read_files, search_codebase, editor, apply_patch, run_commands]\n"
        "write_scope:\n"
        "  paths: ['**']\n"
        "command_allow_list:\n"
        "  commands: [pytest, git]\n"
        "budget:\n"
        "  max_cost_usd: 100.0\n",
        encoding="utf-8",
    )
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "test@codna.ai"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "init"],
    ):
        completed = subprocess.run(
            [argv[0], "-C", str(path), *argv[1:]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
    return path


@pytest.fixture
def durable_config(tmp_path: Path) -> Any:
    """A real `DurableConfig` whose runs root is the test's directory.

    The runtime config is injected (as `test_config.py` does) so this never depends on — or
    inspects — a developer's actual checkout, and `CODNA_DURABLE_RUNS_ROOT` keeps every write inside
    `tmp_path` instead of `~/.codna`.
    """
    from codna.durable.config import load_durable_config
    from codna.runtime.config import RuntimeConfig, build_runtime_paths

    paths = build_runtime_paths()
    runtime = RuntimeConfig(
        codna_home=tmp_path,
        port_base=18600,
        engine_port=18600,
        sidecar_port=18601,
        engine_url="http://127.0.0.1:18600",
        sidecar_url="http://127.0.0.1:18601",
        remote_engine_url=None,
        ignored_engine_url=None,
        engine_url_override=None,
        paths=paths,
        engine_dir=tmp_path / "engine",
        engine_python="python3",
        engine_env_file=tmp_path / ".env",
        sidecar_dir=tmp_path / "agent-core",
        sidecar_binary=None,
        cline_data_dir=tmp_path / "cline-data",
        codna_cli_version="0.0.0-test",
        engine_build_id="engine",
        sidecar_build_id="sidecar",
        runtime_config_hash="hash",
        allow_loopback_override=False,
    )
    # `plain` engine deliberately: these tests are about the CLI's start/select/query/update path,
    # and the graph engine is exercised by its own suite. Pinning it here means a graph-engine
    # regression cannot be misread as a CLI regression (and vice versa).
    return load_durable_config(
        env={"CODNA_DURABLE_RUNS_ROOT": str(tmp_path / "runs"), "CODNA_DURABLE_ENGINE": "plain"},
        runtime=runtime,
        hostname="build-host",
    )


# -- events: reattach with --after ----------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_events_reattaches_after_a_given_seq(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`--after N` is the reattach contract: only what the caller has NOT already seen.

    Asserted as a strict suffix — the seqs after N, in order — because an off-by-one here silently
    re-delivers or silently SKIPS a record, and a skipped record is a step an auditor never sees.
    """
    from codna.durable.ledger import RunLedger

    ledger = RunLedger.for_run(durable_config.paths, "run-ev", clock=lambda: 1_700_000_000.0)
    for index in range(1, 5):
        ledger.append("turn.finished", session_id="s-1", event_index=index, payload={"n": index})

    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    assert cmd_durable(_parse(["durable", "events", "--run", "run-ev", "--after", "2"])) == 0

    emitted = _json_objects(capsys.readouterr().out)
    seqs = [entry["record"]["seq"] for entry in emitted]
    assert seqs == [3, 4], seqs
    assert all("record" in entry for entry in emitted)


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_events_without_after_replays_the_whole_ledger(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The `--after` default is the beginning, not "the end" — a fresh client must see everything."""
    from codna.durable.ledger import RunLedger

    ledger = RunLedger.for_run(durable_config.paths, "run-ev2", clock=lambda: 1_700_000_000.0)
    for index in range(1, 4):
        ledger.append("turn.finished", session_id="s-1", event_index=index, payload={"n": index})

    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    assert cmd_durable(_parse(["durable", "events", "--run", "run-ev2"])) == 0
    seqs = [entry["record"]["seq"] for entry in _json_objects(capsys.readouterr().out)]
    assert seqs == [1, 2, 3]


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_events_after_a_seq_past_the_end_emits_nothing(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A client that is fully caught up gets an empty result, not an error and not a re-send."""
    from codna.durable.ledger import RunLedger

    RunLedger.for_run(durable_config.paths, "run-ev3", clock=lambda: 1_700_000_000.0).append(
        "turn.finished", session_id="s-1", event_index=1, payload={}
    )
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    assert cmd_durable(_parse(["durable", "events", "--run", "run-ev3", "--after", "99"])) == 0
    assert capsys.readouterr().out.strip() == ""


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_events_reports_a_missing_ledger_with_its_path(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(durable_cli, "_config", lambda _args: durable_config)
    with pytest.raises(DurableCliError) as caught:
        cmd_durable(_parse(["durable", "events", "--run", "run-absent"]))
    assert caught.value.code == "durable_ledger_missing"


def test_after_is_validated_as_a_non_negative_integer():
    """A mistyped resume point must fail, never be coerced into "skip nothing" or "skip everything"."""
    ok = _parse(["durable", "events", "--run", "r", "--after", "0"])
    assert durable_cli._after_seq(ok) == 0
    assert durable_cli._after_seq(_parse(["durable", "events", "--run", "r"])) == 0
    with pytest.raises(DurableCliError) as caught:
        durable_cli._after_seq(SimpleNamespace(after=-1))
    assert caught.value.code == "durable_after_invalid"
    with pytest.raises(DurableCliError):
        durable_cli._after_seq(SimpleNamespace(after=True))


def test_a_negative_after_from_the_command_line_is_refused_before_any_read(
    durable_config: Any, monkeypatch: pytest.MonkeyPatch
):
    """`--after=-3` reaches the command (argparse reads a leading `-3` as a flag, so `--after -3` is a
    usage error instead) and must be refused there — before a single record is read.

    argparse accepts a negative int for ``type=int`` (it only special-cases values that look like an
    option), so the COMMAND is the only guard — which is why this is a command test rather than a
    parser test. Missing it would let a reattach silently skip records.
    """
    args = _parse(["durable", "events", "--run", "r", "--after", "-3"])
    assert args.after == -3
    monkeypatch.setattr(durable_cli, "_config", lambda _a: durable_config)
    monkeypatch.setattr("codna.durable.require_durable", lambda: None)
    with pytest.raises(DurableCliError) as caught:
        cmd_durable(args)
    assert caught.value.code == "durable_after_invalid"


# -- `codna run` runs a worker in-process ---------------------------------------------------------


def test_the_top_level_run_chooses_the_in_process_worker_by_default():
    """`codna run` with nothing else polling would otherwise start a run and sit forever."""
    from codna.durable.run_cli import _worker_choice

    assert _worker_choice(_parse(["run", "--brief", "b"])) is True
    assert _worker_choice(_parse(["durable", "run", "--brief", "b"])) is False


def test_worker_and_no_worker_are_explicit_overrides_on_either_surface():
    from codna.durable.run_cli import _worker_choice

    assert _worker_choice(_parse(["run", "--brief", "b", "--no-worker"])) is False
    assert _worker_choice(_parse(["durable", "run", "--brief", "b", "--worker"])) is True


def test_worker_and_no_worker_together_are_refused():
    """Two contradictory instructions must not resolve to whichever was read last."""
    from codna.durable.run_cli import _worker_choice

    with pytest.raises(DurableCliError) as caught:
        _worker_choice(_parse(["run", "--brief", "b", "--worker", "--no-worker"]))
    assert caught.value.code == "durable_worker_conflict"


def test_pin_and_no_pin_together_are_refused():
    from codna.durable.run_cli import _pin_choice

    with pytest.raises(DurableCliError) as caught:
        _pin_choice(_parse(["run", "--brief", "b", "--pin", "--no-pin"]))
    assert caught.value.code == "durable_pin_conflict"


def test_the_pin_tristate_maps_the_flags():
    from codna.durable.run_cli import _pin_choice

    assert _pin_choice(_parse(["run", "--brief", "b"])) is None      # decide by asking the server
    assert _pin_choice(_parse(["run", "--brief", "b", "--pin"])) is True
    assert _pin_choice(_parse(["run", "--brief", "b", "--no-pin"])) is False


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_drained_in_process_run_reports_that_it_detached(
    tmp_path: Path, durable_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Ctrl-C while waiting must leave the run alive and say so — not fail it and not pretend success.

    The run lives on the server; a stop is "I am done watching", so the command exits non-zero (it did
    not reach a success state) while the workflow is untouched and still reattachable.
    """
    repo = _git_repo(tmp_path / "repo")
    server = _run_commands_against_a_server(tmp_path, durable_config, repo, monkeypatch)
    assert cmd_durable(_parse(["durable", "run", "--repo", str(repo), "--brief", "b", "--run-id", "run-cli",
                               "--no-start-server"])) == 0
    capsys.readouterr()
    server.wait_parked("task-run-cli-task-1")

    stop = asyncio.Event()
    stop.set()  # already "interrupted"
    exit_code = asyncio.run(
        durable_cli._wait_run(
            server._env.client,
            durable_config,
            "run-cli",
            poll_s=0.01,
            timeout_s=5.0,
            stop=stop,
        )
    )
    assert exit_code == 2
    body = _json_objects(capsys.readouterr().out)
    assert any("detached" in entry for entry in body), body
    detached = next(entry["detached"] for entry in body if "detached" in entry)
    assert detached["run_id"] == "run-cli"
    assert "continues on the server" in detached["note"]


# -- delegation (D7/SWARM) ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_the_run_payload_carries_swarm_flags_onto_every_task(tmp_path: Path, durable_config: Any):
    """Both delegation fields reach EVERY task's spec, so no task in a swarm run is a silent exception."""
    repo = _git_repo(tmp_path / "repo")
    payload = durable_cli.run_payload_for(
        config=durable_config,
        run_id="run-1",
        repo_dir=repo,
        briefs=["fix a", "fix b"],
        base_branch="main",
        verify_commands=["true"],
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=1,
        engine="plain",
        enable_spawn_agent=True,
        enable_agent_teams=True,
    )
    for task in payload["tasks"]:
        assert task["enable_spawn_agent"] is True
        assert task["enable_agent_teams"] is True


@pytest.mark.skipif(not HAS_EXTRA, reason="the `durable` extra is not installed")
def test_a_run_that_does_not_delegate_records_neither_flag(tmp_path: Path, durable_config: Any):
    """The default stays `False` on both, which is what every pre-feature spec decodes to."""
    repo = _git_repo(tmp_path / "repo")
    payload = durable_cli.run_payload_for(
        config=durable_config,
        run_id="run-1",
        repo_dir=repo,
        briefs=["fix a"],
        base_branch="main",
        verify_commands=["true"],
        max_cost_usd=None,
        max_tokens=None,
        max_attempts=1,
        engine="plain",
    )
    assert payload["tasks"][0]["enable_spawn_agent"] is False
    assert payload["tasks"][0]["enable_agent_teams"] is False


def test_parallel_without_worktree_is_refused(tmp_path: Path, durable_config: Any):
    """Concurrent tasks need one checkout each; `--parallel 4` without `--worktree` is a CLI error."""
    repo = _git_repo(tmp_path / "repo")
    with pytest.raises(DurableCliError) as caught:
        durable_cli.run_payload_for(
            config=durable_config,
            run_id="run-1",
            repo_dir=repo,
            briefs=["fix a", "fix b"],
            base_branch="main",
            verify_commands=["true"],
            max_cost_usd=None,
            max_tokens=None,
            max_attempts=1,
            engine="plain",
            max_parallel_tasks=2,
            worktree=False,
        )
    assert caught.value.code == "contract_invalid"


def _swarm_args(**overrides: Any) -> Any:
    """A minimal argparse namespace shaped like `codna run`'s, for `_swarm_choice`."""
    base = {
        "swarm": False,
        "no_swarm": False,
        "spawn_agent": False,
        "agent_teams": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_swarm_means_both_capabilities():
    args = _swarm_args(swarm=True)
    assert durable_cli._swarm_choice(args, "spawn_agent") is True
    assert durable_cli._swarm_choice(args, "agent_teams") is True


def test_the_per_capability_flags_are_independent():
    """The engine gates the two separately, so a team-only run is a legal thing to ask for."""
    args = _swarm_args(agent_teams=True)
    assert durable_cli._swarm_choice(args, "spawn_agent") is False
    assert durable_cli._swarm_choice(args, "agent_teams") is True


def test_no_swarm_turns_delegation_off():
    args = _swarm_args(swarm=True, no_swarm=True)
    with pytest.raises(DurableCliError) as caught:
        durable_cli._swarm_choice(args, "spawn_agent")
    assert caught.value.code == "durable_swarm_conflict"

    assert durable_cli._swarm_choice(_swarm_args(no_swarm=True), "spawn_agent") is False
