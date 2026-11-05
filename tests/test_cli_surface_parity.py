from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from codna import cli
from codna.fix_inputs import FixInputError, resolve_issue


# ---- parser wiring ----------------------------------------------------------------------------

def _parse(argv):
    return cli.build_parser().parse_args(argv)


def test_triage_json_flag():
    assert _parse(["triage", ".", "--json"]).as_json is True
    assert _parse(["triage", "."]).as_json is False


def test_fix_tests_flags():
    a = _parse(["fix", ".", "--issue", "x", "--tests", "--test-cmd", "make t"])
    assert a.tests is True and a.test_cmd == "make t"


def test_mcp_start_and_repo():
    a = _parse(["mcp", "start", "--repo", "/x"])
    assert a.action == "start" and a.repo == "/x"
    b = _parse(["mcp"])
    assert b.action == "start" and b.repo is None


def test_root_config_flag():
    assert _parse(["--config", "/c.yaml", "triage", "."]).config == "/c.yaml"


# ---- fix input resolution ---------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.issue = None
        self.failing_test = None
        self.from_junit = None
        self.tests = False
        self.test_cmd = None
        self.repo = "."
        self.__dict__.update(kw)


def test_resolve_issue_plain():
    issue, failing = resolve_issue(_Args(issue="broken"))
    assert issue == "broken" and failing == []


def test_resolve_issue_failing_tests_merge():
    issue, failing = resolve_issue(_Args(issue="x", failing_test=["a::b"]))
    assert failing == ["a::b"]


def test_resolve_issue_tests_discovers(monkeypatch):
    monkeypatch.setattr("codna.testrun.discover_failing_tests",
                        lambda repo, cmd: ("2 failing test(s)", ["p::t1", "p::t2"]))
    issue, failing = resolve_issue(_Args(tests=True))
    assert failing == ["p::t1", "p::t2"] and "failing" in issue


def test_resolve_issue_tests_materializes_remote_repo_for_discovery(monkeypatch):
    captured = {}

    def fake_remote_discovery(args, github_token):
        captured["repo"] = args.repo
        captured["github_token"] = github_token
        return "1 failing test(s)", ["remote::test"]

    monkeypatch.setattr("codna.fix_inputs._discover_remote_failing_tests_for_fix", fake_remote_discovery)

    issue, failing = resolve_issue(
        _Args(tests=True, repo="https://github.com/owner/repo.git"),
        github_token="ghs_test",
    )

    assert issue == "1 failing test(s)"
    assert failing == ["remote::test"]
    assert captured == {"repo": "https://github.com/owner/repo.git", "github_token": "ghs_test"}


def test_resolve_issue_tests_no_failures_and_no_issue_raises(monkeypatch):
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd: (None, []))
    with pytest.raises(FixInputError):
        resolve_issue(_Args(tests=True))


def test_resolve_issue_explicit_issue_overrides_tests(monkeypatch):
    monkeypatch.setattr("codna.testrun.discover_failing_tests",
                        lambda repo, cmd: ("synth", ["p::t1"]))
    issue, failing = resolve_issue(_Args(tests=True, issue="my issue"))
    assert issue == "my issue" and failing == ["p::t1"]


# ---- triage --json output ---------------------------------------------------------------------

def test_cmd_triage_json_output(monkeypatch, capsys):
    class _FakeClient:
        def triage_repository(self, rid, body):
            return {
                "suspect_files": ["a.py"],
                "suspect_symbols": ["a.f"],
                "raw_repo_token_estimate": 100,
                "evidence_bundle_token_count": 20,
                "reduction_ratio": 5.0,
                "workspace_evidence_bundle_ref": "bundle-1",
            }

    monkeypatch.setattr(cli, "_client", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(cli, "_register", lambda c, repo, ref, focus_paths=None: ("rid-1", {"snapshot_id": "snap-1", "snapshot_file_count": 3}))
    monkeypatch.setattr(cli, "_issue_focus_paths", lambda local, issue: [])
    monkeypatch.setattr(cli, "_dump", lambda x: x)

    args = cli.build_parser().parse_args(["triage", ".", "--json"])
    cli.cmd_triage(args)
    out = capsys.readouterr().out
    assert "codna: understanding" not in out  # status line suppressed under --json
    data = json.loads(out)
    assert data["repository_id"] == "rid-1"
    assert data["snapshot_id"] == "snap-1"
    assert data["suspect_files"] == ["a.py"]
    assert data["reduction_ratio"] == 5.0  # raw float, not preformatted "×"
    assert "elapsed_s" in data


def test_only_engine_commands_consume_config():
    # config is applied (and can fail-close) ONLY for fix/triage/secure; not doctor/mcp/login/key
    for cmd in (["triage", "."], ["fix", ".", "--issue", "x"], ["secure", ".", "--from-sarif", "s.json"]):
        assert getattr(_parse(cmd), "uses_config", False) is True, cmd
    for cmd in (["doctor"], ["mcp"], ["login"], ["key", "list"]):
        assert getattr(_parse(cmd), "uses_config", False) is False, cmd


def test_secure_json_flag_parses():
    # secure --json must be assertable in the minimal-dep 'cli secure subsystem' CI job (no mcp).
    assert _parse(["secure", ".", "--from-sarif", "s.sarif", "--json"]).as_json is True


def test_secure_json_is_read_only_for_action_flags(capsys):
    rc = cli.main([
        "secure",
        ".",
        "--from-sarif",
        "missing.sarif",
        "--engine",
        "local",
        "--fix",
        "--json",
    ])

    assert rc == 1
    captured = capsys.readouterr()
    assert "read-only" in captured.err
    assert "--fix/--open-pr" in captured.err


def test_resolve_issue_tests_uses_temp_worktree_for_clean_git_repo(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "runner.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "from calc import add\n"
        "Path('__pycache__').mkdir(exist_ok=True)\n"
        "Path('__pycache__/generated.pyc').write_text('dirty')\n"
        "junit = Path(os.environ['CODNA_JUNIT'])\n"
        "if add(2, 2) == 4:\n"
        "    junit.write_text('<testsuite tests=\"1\" failures=\"0\"><testcase classname=\"test_calc\" name=\"test_add\" /></testsuite>')\n"
        "    raise SystemExit(0)\n"
        "junit.write_text('<testsuite tests=\"1\" failures=\"1\"><testcase classname=\"test_calc\" name=\"test_add\"><failure /></testcase></testsuite>')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)

    args = _Args(tests=True, repo=str(repo), test_cmd="python3 runner.py")
    issue, failing = resolve_issue(args)

    assert issue == "1 failing test(s): test_calc::test_add"
    assert failing == ["test_calc::test_add"]
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    assert status.stdout == b""
    assert not (repo / "__pycache__").exists()

# ---- capabilities doc parity ------------------------------------------------------------------

def test_capabilities_doc_matches_mcp_tool_contracts():
    from codna import mcp_server

    root = Path(__file__).resolve().parents[2]
    doc = (root / "CAPABILITIES.md").read_text(encoding="utf-8")

    for name in ("codna_triage", "codna_fix", "codna_secure", "codna_recall"):
        assert name in doc

    assert "codna://capabilities" in doc
    assert 'fix_bug(issue, repo=".")' in doc
    assert "open_pr=false" not in doc
    assert "### `codna webhook serve`" in doc
    assert "mode                 fix or secure" in doc
    assert "The current `action.yml` does not expose secure mode" not in doc
    assert "secure mode read-only" in doc
    assert "CODNA_DISABLE_KEYCHAIN" in doc
    assert "do not trigger OS authorization prompts" in doc

    parser = cli.build_parser()
    subparser_action = next(action for action in parser._actions if isinstance(getattr(action, "choices", None), dict))
    assert "webhook" in subparser_action.choices

    assert str(inspect.signature(mcp_server.recall_json)) == (
        "(repo: 'str' = '.', query: 'str' = '', service: 'str' = '', "
        "language: 'str' = '', final_k: 'int' = 8) -> 'str'"
    )
    assert '`codna_fix` | `repo="."`, `issue=""`, `ref=""`, `model="repository.verified_agentic_v1"`' in doc


def test_resolve_issue_tests_lets_an_environment_gap_keep_its_code(monkeypatch):
    """TestEnvironmentUnavailable must not be flattened into a FixInputError (cli_error): the CLI
    prints its own code and the GitHub App ends the check neutral on it (mojo-kernels#1)."""
    from codna.testrun import TestEnvironmentUnavailable

    def _boom(repo, cmd):
        raise TestEnvironmentUnavailable("no pixi here.")

    monkeypatch.setattr("codna.testrun.discover_failing_tests", _boom)
    with pytest.raises(TestEnvironmentUnavailable):
        resolve_issue(_Args(tests=True))


# ---- the durable surface's two entrances ------------------------------------------------------

def test_the_top_level_parser_exposes_run_and_worker():
    """The plan names `codna run` / `codna worker` as first-class commands, not only `durable` verbs.

    Asserted on the parser rather than on prose: a command that exists in the docs and not in
    `build_parser` is a command a user cannot type.
    """
    parser = cli.build_parser()
    subcommands = next(a for a in parser._actions if getattr(a, "choices", None))
    assert {"run", "worker", "durable"} <= set(subcommands.choices)


def test_both_run_spellings_accept_the_same_flags():
    """One flag definition, two entrances: a flag on one and not the other is a documented lie."""
    top = _parse(["run", "--brief", "b", "--pin", "--no-worker"])
    grouped = _parse(["durable", "run", "--brief", "b", "--pin", "--no-worker"])
    for attribute in ("brief", "pin", "no_worker", "engine", "run_id", "max_cost", "poll_interval"):
        assert getattr(top, attribute) == getattr(grouped, attribute), attribute


def test_the_two_run_spellings_differ_only_in_whether_they_run_a_worker():
    """`codna run` executes what it starts; `durable run` is the lower-level start-only surface."""
    assert _parse(["run", "--brief", "b"]).run_worker_by_default is True
    assert _parse(["durable", "run", "--brief", "b"]).run_worker_by_default is False
    # ...and --worker/--no-worker override it on either surface.
    assert _parse(["run", "--brief", "b", "--no-worker"]).no_worker is True
    assert _parse(["durable", "run", "--brief", "b", "--worker"]).worker is True


def test_worker_flags_exist_on_both_spellings():
    top = _parse(["worker"])
    grouped = _parse(["durable", "worker"])
    for attribute in ("max_concurrent_activities", "grace_period", "no_promote", "engine"):
        assert getattr(top, attribute) == getattr(grouped, attribute), attribute


def test_durable_commands_do_not_consume_codna_yaml():
    """`codna.yaml` is model/privacy config; a durable command must not fail because it is absent."""
    for argv in (["durable", "status", "--run", "r"], ["worker"], ["durable", "events", "--run", "r"]):
        assert getattr(_parse(argv), "uses_config", False) is False, argv


# ---- the Run API's two entrances --------------------------------------------------------------

def test_the_top_level_parser_exposes_serve():
    """`codna serve` is a product surface, not only a `durable` verb: the IDE connects to it.

    Asserted on the parser rather than on prose, like `run`/`worker` above: a command that exists in
    the documentation and not in `build_parser` is a command a user cannot type.
    """
    parser = cli.build_parser()
    subcommands = next(a for a in parser._actions if getattr(a, "choices", None))
    assert {"run", "worker", "serve", "durable"} <= set(subcommands.choices)


def test_both_serve_spellings_accept_the_same_flags():
    """One flag definition, two entrances — a flag on one spelling alone is a documented lie."""
    top = _parse(["serve", "--port", "0", "--max-streams", "3", "--no-temporal"])
    grouped = _parse(["durable", "serve", "--port", "0", "--max-streams", "3", "--no-temporal"])
    for attribute in ("host", "port", "heartbeat_interval", "max_streams", "no_temporal"):
        assert getattr(top, attribute) == getattr(grouped, attribute), attribute


def test_serve_defaults_to_loopback_and_a_fixed_port():
    """A stable address so the IDE can be configured once; loopback because it holds a token."""
    args = _parse(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8790
    assert args.no_temporal is False
    assert args.max_streams > 0 and args.heartbeat_interval > 0


def test_serve_resolves_the_durable_config():
    """The Run API reads the same config `run` does (address, namespace, queue, runs root)."""
    assert getattr(_parse(["serve"]), "uses_config", False) is True


def test_serve_dispatches_through_the_durable_group():
    """The grouped spelling routes through `cmd_durable`, like every other `durable` verb."""
    from codna.durable.cli import cmd_durable

    args = _parse(["durable", "serve"])
    assert args.func is cmd_durable
    assert args.durable_action == "serve"


def test_there_is_no_flag_that_turns_off_the_loopback_rule():
    """The refusal is not configurable: a flag to disable it is a flag somebody would disable."""
    parser = cli.build_parser()
    options = {
        option
        for action in parser._actions
        for option in getattr(action, "option_strings", ())
    }
    assert not any("allow-remote" in option or "allow-public" in option for option in options)
    serve = next(
        action for action in parser._actions if getattr(action, "choices", None)
    ).choices["serve"]
    serve_options = {
        option for action in serve._actions for option in getattr(action, "option_strings", ())
    }
    assert not any("allow" in option for option in serve_options), serve_options
