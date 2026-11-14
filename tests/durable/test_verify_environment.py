"""`verify_task` runs with a PATH that reaches the user's tools — proved through the REAL step.

── WHY THIS FILE EXISTS BESIDE `test_verify_env.py` ────────────────────────────────────────────────

`test_verify_env.py` proves the resolver's decisions. This file proves the DECISION IS APPLIED: that
`verify_task`'s sandbox path calls it, that the resulting `PATH` is what a real `/bin/sh -lc` sees, and
that the attempt records which `PATH` it used. A resolver that was correct and unwired would leave the
measured failure exactly where it was, which is the way this class of fix usually fails — silently, in
the join between two modules.

── THE MEASURED FAILURE ────────────────────────────────────────────────────────────────────────────

Run ``run-20261113T225225Z``: a correct three-file app, ``node --test`` passing 3/3 in a terminal, and
three verify attempts each reporting ``exit 127  /bin/sh: node: command not found`` because the worker's
launcher gave it ``/usr/bin:/bin`` while ``node`` lives in ``/opt/homebrew/bin``.

`node` is not required to prove it: a fixture executable in a directory that only the resolved PATH
contains is the same mechanism, and it runs on a machine with no Node at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("temporalio", reason="the `durable` extra (temporalio) is not installed")

from codna.durable.activity_steps import (  # noqa: E402
    checkpoint_turn,
    run_cline_turn,
    verify_task,
)
from codna.durable.activity_support import ActivityError  # noqa: E402
from codna.durable.contracts import (  # noqa: E402
    CheckpointRequest,
    TaskContext,
    TaskSpec,
    VerifyRequest,
)
from codna.verify_env import LoginPath, reset_cache  # noqa: E402
from tests.durable import fakes  # noqa: E402

RUN_ID = "run-verify-env"

#: Fixture data for the PATCHED probe, not a path this suite expects on the host: `login_shell_path`
#: is replaced below in every test that uses it, so nothing stats this string or executes it. It is
#: named once so a reader cannot mistake it for a claim about the runner's login shell — the defect
#: `test_verify_env.py` was rewritten to stop making.
FAKE_LOGIN_SHELL = "/opt/example-toolchain/bin/login-shell"


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The resolved environment lives for the process lifetime; a test must not inherit another's."""
    reset_cache()
    yield
    reset_cache()


def _toolchain(tmp_path: Path) -> Path:
    """A directory holding one executable, in the shape of a user's toolchain directory."""
    bin_dir = tmp_path / "toolchain" / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / "codna_fixture_tool"
    tool.write_text("#!/bin/sh\necho fixture-tool-ran\n", encoding="utf-8")
    tool.chmod(0o755)
    return bin_dir


def _to_checkpoint(repo: Path, tmp_path: Path):
    """One real turn + checkpoint, so verify runs against a tree that exists."""
    runs_root = repo.parent / "runs"
    runs_root.mkdir(exist_ok=True)
    harness = fakes.build_harness(repo, runs_root)
    # The seam this file is about is the SANDBOX, so the scripted runner must be unbound.
    harness.runtime.run_tests = None
    spec = TaskSpec(
        run_id=RUN_ID,
        task_id="task-1",
        task_number=1,
        brief="build it",
        repo_dir=str(repo),
        base_branch="main",
        verify_commands=(),
        engine="plain",
    )
    spec.validate()
    context = TaskContext.from_spec(spec)
    moved = run_cline_turn(context.turn_request("implement it", spec.initial_state()))
    return harness, context, checkpoint_turn(CheckpointRequest(state=moved.state, context=context))


def _ledger_verify_record(repo: Path) -> dict:
    """The `verify` record the run wrote — read from the file, not from the step's return value."""
    path = repo.parent / "runs" / RUN_ID / "ledger.ndjson"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    verify = [record for record in records if record["kind"] == "verify"]
    assert len(verify) == 1, [record["kind"] for record in records]
    return verify[0]["payload"]


def test_verify_runs_a_command_that_only_the_login_shell_path_can_find(
    tmp_path: Path, monkeypatch
):
    """THE DEFECT, END TO END THROUGH THE STEP.

    The worker's PATH cannot find the tool. The login shell's PATH can. `verify_task` must report a PASS
    and the run's ledger must name the PATH it used — the two things the measured run got wrong (it
    reported a failure and recorded nothing about where the failure came from).
    """
    bin_dir = _toolchain(tmp_path)
    repo = fakes.make_repo(tmp_path / "repo")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SHELL", FAKE_LOGIN_SHELL)
    monkeypatch.delenv("CODNA_VERIFY_PATH", raising=False)
    # The probe is injected because a test must not depend on the dotfiles of the machine running CI;
    # what is under test is that the RESULT of a successful probe reaches the subprocess.
    monkeypatch.setattr(
        "codna.verify_env.login_shell_path",
        lambda env, **kwargs: LoginPath(f"{bin_dir}:/usr/bin:/bin", FAKE_LOGIN_SHELL, None),
    )

    harness, context, checkpoint = _to_checkpoint(repo, tmp_path)
    outcome = verify_task(
        VerifyRequest(state=checkpoint.state, context=context, commands=("codna_fixture_tool",))
    )

    assert outcome.details["passed"] is True, outcome.details
    assert outcome.details["returncode"] == 0
    assert "login-shell" in outcome.details["environment"], outcome.details
    assert str(bin_dir) in outcome.details["environment"]

    payload = _ledger_verify_record(repo)
    assert payload["passed"] is True
    assert payload["environment"] == outcome.details["environment"]
    log = (repo.parent / "runs" / RUN_ID / payload["log_path"]).read_text(encoding="utf-8")
    assert "environment:" in log and str(bin_dir) in log
    assert harness.verify_calls == 0, "the sandbox seam must have been the one that ran"


def test_verify_records_the_fallback_when_the_login_shell_cannot_be_asked(
    tmp_path: Path, monkeypatch
):
    """A probe that fails is a recorded fallback, not a silent minimal PATH — and the attempt is kept.

    Before this, a `127` said only "command not found" — nothing about the PATH that had been searched,
    which is why the measured run needed a manual reproduction to diagnose.

    The verdict is still a fail, and it is now a CODED one: a binary that is not on the searched PATH
    cannot be installed by another `implement` turn, so the step raises
    `activity_verify_toolchain_missing` naming the command, the binary and where the PATH came from.
    The attempt is LEDGERED FIRST, so the property this test was written for — the fallback reason is
    on the record — still holds, and is asserted from the ledger rather than from a return value that
    the raise necessarily replaces.
    """
    _toolchain(tmp_path)
    repo = fakes.make_repo(tmp_path / "repo")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SHELL", FAKE_LOGIN_SHELL)
    monkeypatch.delenv("CODNA_VERIFY_PATH", raising=False)
    monkeypatch.setattr(
        "codna.verify_env.login_shell_path",
        lambda env, **kwargs: LoginPath(None, None, f"{FAKE_LOGIN_SHELL} did not answer within 5s"),
    )

    harness, context, checkpoint = _to_checkpoint(repo, tmp_path)
    with pytest.raises(ActivityError) as caught:
        verify_task(
            VerifyRequest(state=checkpoint.state, context=context, commands=("codna_fixture_tool",))
        )

    error = caught.value
    assert error.code == "activity_verify_toolchain_missing", error.as_dict()
    assert error.details["missing"] == "codna_fixture_tool"
    # The reason the probe fell back is carried in the error, so the repair is readable from it alone.
    environment = str(error.details["environment"])
    assert environment.startswith("process:"), environment
    assert "did not answer within 5s" in environment
    assert "PATH=/usr/bin:/bin" in environment

    payload = _ledger_verify_record(repo)
    assert payload["passed"] is False and payload["returncode"] == 127
    assert payload["environment"] == environment
    log = (repo.parent / "runs" / RUN_ID / payload["log_path"]).read_text(encoding="utf-8")
    assert "did not answer within 5s" in log


def test_the_explicit_override_reaches_the_sandbox(tmp_path: Path, monkeypatch):
    """`CODNA_VERIFY_PATH` is the operator's route where no login shell exists (containers, CI)."""
    bin_dir = _toolchain(tmp_path)
    repo = fakes.make_repo(tmp_path / "repo")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("CODNA_VERIFY_PATH", str(bin_dir))
    # If the probe ran, the tool would still be found — so the assertion that the probe was SKIPPED is
    # separate: a runner that fails the test if it is called at all.
    monkeypatch.setattr(
        "codna.verify_env.login_shell_path",
        lambda env, **kwargs: pytest.fail("the login-shell probe ran despite CODNA_VERIFY_PATH"),
    )

    harness, context, checkpoint = _to_checkpoint(repo, tmp_path)
    outcome = verify_task(
        VerifyRequest(state=checkpoint.state, context=context, commands=("codna_fixture_tool",))
    )

    assert outcome.details["passed"] is True, outcome.details
    assert outcome.details["environment"].startswith("explicit:"), outcome.details


def test_a_genuinely_failing_command_is_still_a_failure(tmp_path: Path, monkeypatch):
    """GUARD. Making the toolchain reachable must not make a red suite green.

    The whole risk of a fix that prepares a better environment is that it converts "we could not run it"
    into "it passed". A command that runs and exits non-zero must remain a failed verification.
    """
    bin_dir = _toolchain(tmp_path)
    repo = fakes.make_repo(tmp_path / "repo")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("CODNA_VERIFY_PATH", raising=False)
    monkeypatch.setattr(
        "codna.verify_env.login_shell_path",
        lambda env, **kwargs: LoginPath(f"{bin_dir}:/usr/bin:/bin", FAKE_LOGIN_SHELL, None),
    )

    harness, context, checkpoint = _to_checkpoint(repo, tmp_path)
    outcome = verify_task(
        VerifyRequest(state=checkpoint.state, context=context, commands=("exit 3",))
    )

    assert outcome.details["passed"] is False
    assert outcome.details["returncode"] == 3
    # The PATH was still resolved and still recorded: a real failure is diagnosable too.
    assert "login-shell" in outcome.details["environment"]