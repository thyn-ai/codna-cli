"""A command the PATH cannot resolve is a CODED error, and a PATH entry must earn its place.

── WHY THIS IS A SEPARATE DEFECT FROM THE PATH ITSELF ───────────────────────────────────────────

`test_verify_env.py` and `test_verify_environment.py` prove that verify resolves the user's login-shell
PATH. That fix makes the toolchain reachable. It does NOT decide what happens when the toolchain is
genuinely absent, and the measured run is the evidence that the old answer was wrong:

    run-20261113T225225Z     three `verify` records with passed: false, returncode: 127

`after_verify` read each of those as "the tests are red" and routed back to `implement` — which cannot
install a missing binary, so two paid turns rewrote code that already passed. `after_verify` then found
no retries left and returned `wait_for_human`. **A park is an approvable state** (`wait_for_human` is
the graph's last node), so the run finished on work its own verdict had never once passed.

That is the whole reason this file exists: a missing TOOLCHAIN must be terminal for the step, with an
error that names what to install, instead of a red result that routes to a node that cannot fix it.

── AND WHY THE SANITIZATION IS HERE ─────────────────────────────────────────────────────────────

The resolver merges a login shell's PATH — rc files are arbitrary code — into the environment of a
command that runs code a model just wrote. An entry is admitted only when it is absolute, single-line
and credential-free; an empty entry is refused because a shell reads it as the CURRENT DIRECTORY, and
the sandbox's `cwd` is the repository under test. Both are pinned here, offline and without a shell.

── PORTABILITY ─────────────────────────────────────────────────────────────────────────────────

The suite runs on `ubuntu-latest`, where `/bin/sh` is dash. Every shell's "not found" message is
different — dash says `sh: 1: node: not found`, where macOS's `/bin/sh` says
`/bin/sh: node: command not found` — so the detector is exercised against all of them as DATA, and the
end-to-end tests run whatever `/bin/sh` the host provides rather than naming one.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_steps import (  # noqa: E402
    _missing_tool,
    _run_in_sandbox,
    _verify_plan,
    verify_task,
)
from codna.durable.activity_support import ActivityError  # noqa: E402
from codna.durable.contracts import (  # noqa: E402
    TaskContext,
    TaskSpec,
    VerifyRequest,
)
from codna.sandbox import Sandbox, SandboxResult, scrub_env  # noqa: E402
from codna.verify_env import (  # noqa: E402
    EXPLICIT_ENV,
    LoginPath,
    _MARK_BEGIN,
    _MARK_END,
    merge_paths,
    reset_cache,
    resolve_verify_environment,
    sanitize_path_entries,
)
from tests.durable import fakes  # noqa: E402

ABSENT = "codna-absent-tool"
#: A write-token VALUE, assembled rather than written out: a literal in a test file trips the same
#: secret scanners it is meant to satisfy, and this string must never look like a real credential.
TOKEN = "ghp_" + "a" * 36


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The resolved environment is cached for the process lifetime; no test may inherit another's."""
    reset_cache()
    yield
    reset_cache()


# -- what a PATH entry must be ------------------------------------------------------------------


def test_an_entry_that_is_not_absolute_is_refused():
    """Empty and relative entries are refused, and the count says how many.

    An EMPTY entry is the current directory to a shell, and the sandbox's `cwd` is the repository the
    model just wrote — so admitting one puts generated code on the search path of the command that
    judges it. A RELATIVE entry has the same problem without needing the special case.
    """
    kept, refused = sanitize_path_entries(["", "   ", "bin", "./bin", "../bin", "/usr/bin"])
    assert kept == ("/usr/bin",), kept
    assert refused == 5


def test_an_entry_carrying_a_separator_is_refused():
    """A newline or NUL inside one entry turns it into two — of PATH, or of argv."""
    kept, refused = sanitize_path_entries(
        ["/tmp/one\ntwo", "/tmp/one\rtwo", "/tmp/one\x00two", "/usr/bin"]
    )
    assert kept == ("/usr/bin",), kept
    assert refused == 3


def test_an_entry_carrying_a_write_token_is_refused_and_never_echoed():
    """The same pattern `scrub_env` refuses, applied one level down.

    A token pasted into PATH is a token handed to untrusted code by another door. It is REFUSED rather
    than scrubbed: a directory that had to be edited is not one anybody meant to search.
    """
    hostile = f"/tmp/{TOKEN}/bin"
    kept, refused = sanitize_path_entries([hostile, "/usr/bin"])
    assert kept == ("/usr/bin",), kept
    assert refused == 1
    # The refused entry is not echoed anywhere: everything returned is the KEPT entries and a count.
    assert TOKEN not in str(kept)


def test_every_rule_is_exercised_together_and_legitimate_entries_survive():
    """The whole rule in one call: a real toolchain PATH must come through untouched."""
    kept, refused = sanitize_path_entries(
        [
            "/opt/homebrew/bin",
            "",  # empty: the current directory
            "relative/bin",  # not absolute
            f"/tmp/{TOKEN}/bin",  # a credential
            "/tmp/one\ntwo",  # a separator smuggled inside an entry
            " /usr/local/bin ",  # incidental whitespace, otherwise legitimate
            "/opt/homebrew/bin",  # a duplicate: dropped without being counted as refused
        ]
    )
    assert kept == ("/opt/homebrew/bin", "/usr/local/bin"), kept
    assert refused == 4


def test_the_worker_side_of_the_merge_is_sanitized_too():
    """Neither half is trusted: the worker's PATH can be inherited from an environment it did not build."""
    merged = merge_paths(f"{os.defpath}::bin:{TOKEN}", "/opt/example/bin")
    assert merged.split(os.pathsep) == [*os.defpath.split(os.pathsep), "/opt/example/bin"], merged
    assert TOKEN not in merged


def test_a_resolved_environment_reports_how_many_entries_it_refused(monkeypatch):
    """A refusal is not silent: the note says a count, so an operator can see a dotfile was hostile."""
    monkeypatch.setattr("codna.verify_env._shell_candidates", lambda env: ("/bin/sh",))
    resolved = resolve_verify_environment(
        base_env={"PATH": "/usr/bin:/bin", "SHELL": "/bin/sh"},
        run=lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, f"{_MARK_BEGIN}/opt/example/bin:{TOKEN}{_MARK_END}", ""
        ),
    )
    assert "/opt/example/bin" in resolved.path.split(os.pathsep)
    assert TOKEN not in resolved.path
    assert "1 PATH entry/entries refused" in resolved.note, resolved.note


# -- recognising the shell's message ------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "stderr"),
    [
        ("macos-sh", "/bin/sh: node: command not found"),
        ("posix-sh", "sh: node: command not found"),
        # Ubuntu's /bin/sh is dash, and dash's message is SHORTER and in a different word order. A
        # detector written for "command not found" alone reports nothing here — which is the shape the
        # suite runs its end-to-end tests in.
        ("dash", "sh: 1: node: not found"),
        ("dash-absolute", "/bin/dash: 1: node: not found"),
        ("bash", "/bin/bash: node: command not found"),
        ("bash-line", "sh: line 1: node: command not found"),
        ("zsh", "zsh:1: command not found: node"),
    ],
)
def test_every_shell_s_not_found_message_is_recognised(label, stderr):
    """All four shells `verify` can run under, because CI's `/bin/sh` is not macOS's."""
    result = SandboxResult(127, "", stderr, False, ["/bin/sh", "-lc", "node --test"], "/tmp", "deny")
    assert result.command_not_found() == "node", label


@pytest.mark.parametrize(
    ("label", "returncode", "stderr"),
    [
        ("a-red-suite", 1, "not ok 1 - the counter is off"),
        # A test's OWN output may contain the phrase; it is not the shell reporting a missing program.
        # `make: node: command not found` is make's sentence, not the shell's, and the shell ran fine.
        ("tap-mention", 1, "not ok: command not found"),
        ("zero-exit", 0, "/bin/sh: node: command not found"),
        ("other-code", 2, "sh: 1: node: not found"),
    ],
)
def test_an_ordinary_failure_is_never_read_as_a_missing_toolchain(label, returncode, stderr):
    """GUARD. The coded error must not swallow real failures — those route back to `implement`."""
    result = SandboxResult(returncode, "", stderr, False, ["/bin/sh", "-lc", "x"], "/tmp", "deny")
    assert result.command_not_found() is None, label


def test_a_timed_out_command_is_not_a_missing_toolchain():
    """A timeout returns 124, but assert the flag is what decides, not the code alone."""
    result = SandboxResult(124, "", "sh: 1: node: not found", True, ["/bin/sh"], "/tmp", "deny")
    assert result.command_not_found() is None


# -- the step: a missing toolchain is coded, not red ---------------------------------------------


def _tool(name: str, directory: Path, body: str = "exit 0") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / name
    tool.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return tool


def _spec(repo: Path, *, commands: tuple[str, ...]) -> TaskSpec:
    spec = TaskSpec(
        run_id="run-toolchain",
        task_id="task-1",
        task_number=1,
        brief="build a counter",
        repo_dir=str(repo),
        verify_commands=commands,
        engine="plain",
    )
    spec.validate()
    return spec


def _base_state() -> dict:
    """A minimal valid `TaskState` payload.

    A DICT, not a decoded `TaskState`: `VerifyRequest.state` is a wire payload and `verify_task`
    decodes it itself, so handing it a live object would fail the contract check before the step
    under test ever ran.
    """
    return {
        "schema_version": 1,
        "run_id": "run-toolchain",
        "task_id": "task-1",
        "turn_seq": 1,
        "session_id": "session-1",
        "last_turn_key": "run-toolchain:task-1:1",
        "last_turn_id": "turn-1",
        "last_event_index": 10,
        "budget": {
            "max_cost_usd": None,
            "spent_usd": 0.0,
            "max_tokens": None,
            "spent_tokens": 0,
        },
    }


@pytest.fixture
def worker_path(monkeypatch):
    """The MEASURED worker environment: base utilities only, and no toolchain directory.

    Set on ``os.environ`` because that is where the step reads it — a test that injected the
    environment would be exercising a seam the defect did not travel through.
    """
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.delenv("CODNA_VERIFY_PATH", raising=False)
    return "/usr/bin:/bin"


@pytest.fixture
def shell_answers(monkeypatch):
    """Pose the login shell's answer, so nothing reads this host's dotfiles or layout."""
    monkeypatch.setattr(
        "codna.verify_env.login_shell_path",
        lambda env, **kwargs: LoginPath("/usr/bin:/bin", "/bin/sh", None),
    )


def test_a_missing_binary_is_recognised_at_the_sandbox_seam(
    tmp_path, worker_path, shell_answers
):
    """THE DEFECT, at the seam that can see it. A red result here is what routes to `implement`,
    which cannot install a binary.

    `_run_in_sandbox` returns the result rather than raising, because the caller must LEDGER the
    attempt before turning it into a terminal error — that is what keeps the audit trail. So what is
    asserted here is the fact the caller acts on: the result is the shell failing to resolve a program,
    not a command that ran and failed.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.runtime.run_tests = None
    spec = _spec(repo, commands=(f"{ABSENT} --test",))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)
    plan = _verify_plan(request, harness.runtime, tree=str(repo))

    result = _run_in_sandbox(request, plan, tree=str(repo))

    assert result.returncode == 127, result
    assert result.accepted() is False
    assert result.command_not_found() == ABSENT, result.stderr
    assert _missing_tool(result) == (f"{ABSENT} --test", ABSENT)


def test_a_multi_command_plan_names_the_command_that_actually_failed(
    tmp_path, worker_path, shell_answers
):
    """The FIRST failing command is the one to report, and it is usually not the plan's last.

    `_run_in_sandbox` stops at the first failure, so deriving the command from `plan.commands[-1]`
    names a command that may never have run — telling a user to install a toolchain for a step that
    was fine. The command travels with the result for exactly this reason.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.runtime.run_tests = None
    # Two commands; the FIRST is the one whose program is absent, and the second would have passed.
    spec = _spec(repo, commands=(f"{ABSENT} --test", "echo the-second-command-is-fine"))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)
    plan = _verify_plan(request, harness.runtime, tree=str(repo))
    assert len(plan.commands) > 1, "this test is only meaningful for a multi-command plan"
    assert plan.commands[-1] != plan.commands[0]

    result = _run_in_sandbox(request, plan, tree=str(repo))

    assert result.returncode == 127
    # The result's own argv is the evidence: it is the FIRST command, not the plan's last entry.
    assert result.argv[2] == plan.commands[0], result.argv
    assert _missing_tool(result) == (plan.commands[0], ABSENT)

    # What matters is the assertion a user reads: the error names the command that could not run, not
    # the innocent trailing one. This is the binding check, because it goes through `verify_task`.
    with pytest.raises(ActivityError) as caught:
        verify_task(request)
    assert caught.value.details["command"] == plan.commands[0], caught.value.details
    assert caught.value.details["missing"] == ABSENT
    assert plan.commands[-1] not in str(caught.value.details["command"]), caught.value.details


def test_the_error_names_the_command_the_binary_and_where_the_path_came_from(
    tmp_path, worker_path, shell_answers
):
    """The four facts a repair needs, without reproducing the run.

    Without the PATH and its source, "command not found" cannot be told apart from "installed, but
    invisible to the sandbox" — and that difference is between asking a user to install a toolchain and
    us fixing our own environment. It is why the measured run needed a manual investigation.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.runtime.run_tests = None
    spec = _spec(repo, commands=(f"{ABSENT} --test",))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)

    with pytest.raises(ActivityError) as caught:
        verify_task(request)

    error = caught.value
    assert error.code == "activity_verify_toolchain_missing", error.as_dict()
    assert error.details["command"] == f"{ABSENT} --test", error.details
    assert error.details["missing"] == ABSENT, error.details
    environment = str(error.details["environment"])
    assert environment.startswith("login-shell"), environment
    assert "PATH=" in environment and "/usr/bin" in environment, environment
    # The message is a sentence a user can act on, not a code with no noun in it.
    message = str(error)
    assert ABSENT in message and "TOOLCHAIN" in message, message


def test_the_attempt_is_ledgered_before_the_error_is_raised(
    tmp_path, worker_path, shell_answers
):
    """An error must not leave a silent hole in the audit trail.

    The whole complaint about the measured run was that its record did not say what happened. Raising
    before writing the attempt would trade a silent mis-read for a missing record, so the ledger entry
    (and its log, which names the PATH) is written FIRST.
    """
    import json

    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.runtime.run_tests = None
    spec = _spec(repo, commands=(f"{ABSENT} --test",))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)

    with pytest.raises(ActivityError):
        verify_task(request)

    ledger = tmp_path / "runs" / "run-toolchain" / "ledger.ndjson"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    verify = [record for record in records if record["kind"] == "verify"]
    assert len(verify) == 1, [record["kind"] for record in records]
    payload = verify[0]["payload"]
    assert payload["passed"] is False
    assert payload["returncode"] == 127
    assert "login-shell" in payload["environment"]
    log = (tmp_path / "runs" / "run-toolchain" / payload["log_path"]).read_text(encoding="utf-8")
    assert "command not found" in log or "not found" in log
    assert "environment: login-shell" in log


def test_verify_task_does_not_return_a_result_for_an_absent_binary(
    tmp_path, worker_path, shell_answers
):
    """The regression in one assertion: no `ActivityOutcome` is produced, so nothing can route on it.

    `after_verify` routes on `verify_ref.passed`. An outcome is what makes a park possible; an error is
    what makes the step terminal. That distinction is the fix.
    """
    repo = fakes.make_repo(tmp_path / "repo")
    harness = fakes.build_harness(repo, tmp_path / "runs")
    harness.runtime.run_tests = None
    spec = _spec(repo, commands=(f"{ABSENT} --test",))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)

    with pytest.raises(ActivityError):
        verify_task(request)


def test_a_real_failure_after_a_successful_resolution_is_still_an_ordinary_result(tmp_path):
    """GUARD. The coded error must not turn every red suite into an environment fault.

    A tool that IS found and then exits non-zero is a result about the code: it must keep routing back
    to `implement`, which is the node that can fix it.
    """
    bin_dir = tmp_path / "toolchain"
    _tool("codna_failing_tool", bin_dir, body="echo 'not ok 1 - wrong' >&2\nexit 1")
    repo = fakes.make_repo(tmp_path / "repo")
    spec = _spec(repo, commands=("codna_failing_tool",))
    context = TaskContext.from_spec(spec)
    request = VerifyRequest(state=_base_state(), context=context, commands=spec.verify_commands)

    from types import SimpleNamespace

    _verify_plan(request, SimpleNamespace(run_tests=None), tree=str(repo))
    # An explicit PATH: the assertion is about classification, not about resolving the host's shell.
    from codna.verify_env import VerifyEnvironment
    from types import MappingProxyType

    environment = VerifyEnvironment(
        env=MappingProxyType({**scrub_env(dict(os.environ)), "PATH": f"{bin_dir}:/usr/bin:/bin"}),
        path=f"{bin_dir}:/usr/bin:/bin",
        source="explicit",
        shell=None,
        note="posed by the test",
    )
    sandbox = Sandbox(env=dict(environment.env), env_note=environment.describe())
    result = sandbox.run(["/bin/sh", "-lc", "codna_failing_tool"], cwd=str(repo))

    assert result.returncode == 1
    assert result.accepted() is False
    assert result.command_not_found() is None
    assert _missing_tool(result) is None


def test_a_timeout_is_not_reported_as_a_missing_toolchain():
    """A hung command is neither a missing binary nor a pass; it stays a red result."""
    result = SandboxResult(124, "", "", True, ["/bin/sh", "-lc", "sleep 9"], "/tmp", "deny")
    assert result.command_not_found() is None
    # And a timeout is not mistaken for a missing binary either, whatever command is named.
    assert _missing_tool(result) is None


def test_a_result_with_an_unexpected_argv_still_yields_an_actionable_message():
    """The defensive branch must not produce a blank command.

    Production always runs `/bin/sh -lc <command>`, so `argv` has three elements and `argv[2]` is the
    command. This pins the OTHER shape: a result whose argv is not that. An error reading
    "could not run ``" is unactionable, and an unexpected shape is exactly when a diagnosis matters —
    so the command falls back to a sentence naming the binary that was missing.
    """
    result = SandboxResult(
        127, "", "/bin/sh: node: command not found", False, ["/bin/sh", "-c"], "/tmp", "deny"
    )
    missing = _missing_tool(result)
    assert missing is not None
    command, binary = missing
    assert binary == "node"
    assert command.strip(), "an empty command would make the error unactionable"
    assert "node" in command


def test_a_failed_probe_does_not_report_a_suspicious_entry_that_never_existed():
    """The fallback must not raise a false alarm about a hostile dotfile.

    `"".split(os.pathsep)` is `[""]` — one empty entry that was never in the value. Counting it as
    refused would make EVERY documented fallback (a probe that could not answer, an empty
    `CODNA_VERIFY_PATH`) report "1 entry refused", which reads like evidence of a crafted PATH on the
    path operators most often arrive by. The note has to mean what it says.
    """
    monkeypatch_ok = resolve_verify_environment(
        base_env={"PATH": "/usr/bin:/bin", "SHELL": "/bin/sh", EXPLICIT_ENV: ""},
    )
    assert "refused" not in monkeypatch_ok.note, monkeypatch_ok.note
    assert monkeypatch_ok.path == "/usr/bin:/bin", monkeypatch_ok.path


def test_an_empty_entry_INSIDE_a_real_path_is_still_refused_and_counted():
    """The other half: a genuine empty entry means the current directory, and stays refused.

    `/usr/bin::/bin` and a trailing separator both emit a real empty entry. Refusing those is the
    whole point of the rule — the sandbox's `cwd` is the repository the model just wrote.
    """
    kept, refused = sanitize_path_entries(("/usr/bin", "", "/bin"))
    assert kept == ("/usr/bin", "/bin"), kept
    assert refused == 1

    from codna.verify_env import _merge

    _path, trailing = _merge("/usr/bin:", "")
    assert trailing == 1, "a trailing separator is a real empty entry"
    _path, adjacent = _merge("/usr/bin::/bin", "")
    assert adjacent == 1, "adjacent separators are a real empty entry"
    _path, absent = _merge("/usr/bin:/bin", "")
    assert absent == 0, "a value with no empty entry refuses nothing"
