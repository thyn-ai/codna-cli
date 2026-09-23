from __future__ import annotations

import json
import os
import subprocess


from codna import cli, fix_run


class _Args:
    def __init__(self, **kw):
        self.repo = "/repo"
        self.ref = None
        self.issue = "fix it"
        self.failing_test = None
        self.from_junit = None
        self.tests = False
        self.test_cmd = None
        self.model = "m"
        self.apply = False
        self.open_pr = False
        self.github_token = None
        self.base_branch = None
        self.pr_title = None
        self.pr_body = None
        self.max_iterations = 1
        self.as_json = False
        self.__dict__.update(kw)


def _plan(root="rc", patch="p1", usage=None, model="m"):
    return {
        "runtime_model": model,
        "planner_usage": usage or {},
        "decision_plan": {
            "repository_analysis": {
                "root_cause": root,
                "generated_patch_ref": patch,
                "impacted_symbols": ["f"],
            },
            "confidence": 0.9,
        },
    }


def test_report_only_no_apply(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    result = fix_run.run_fix(object(), _Args())
    assert result["applied"] is None
    assert result["iterations"][0]["summary"]["root_cause"] == "rc"
    assert result["iterations"][0]["summary"]["patch_ref"] == "p1"
    # human render never crashes
    assert any("root cause" in ln for ln in fix_run.render_human(result))


def test_fix_json_exposes_top_level_usage_totals(monkeypatch):
    usage = {
        "input_tokens": 3239,
        "output_tokens": 256,
        "cache_read_tokens": 1024,
        "cache_write_tokens": 0,
        "cost_usd": 0.019267,
    }
    monkeypatch.setattr(
        fix_run,
        "_plan_once",
        lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan(usage=usage, model="gpt-5.5")),
    )

    result = fix_run.run_fix(object(), _Args())

    assert result["usage"] == {
        "input_tokens": 3239,
        "output_tokens": 256,
        "total_tokens": 3495,
        "cache_read_tokens": 1024,
        "cache_write_tokens": 0,
    }
    assert result["tokens"] == {
        "input": 3239,
        "output": 256,
        "total": 3495,
        "cache_read": 1024,
        "cache_write": 0,
    }
    assert result["input_tokens"] == 3239
    assert result["output_tokens"] == 256
    assert result["total_tokens"] == 3495
    assert result["cache_read_tokens"] == 1024
    assert result["cache_write_tokens"] == 0
    assert result["cost_usd"] == 0.019267
    assert result["runtime_model"] == "gpt-5.5"
    assert result["iterations"][0]["summary"]["input_tokens"] == 3239
    assert result["iterations"][0]["summary"]["total_tokens"] == 3495


def test_plan_once_routes_provider_model_to_sidecar_agent_env(monkeypatch, tmp_path):
    captured = {}

    class Client:
        def triage_repository(self, _repository_id, _request):
            return {"workspace_evidence_bundle_ref": "bundle-1"}

        def create_repository_decision_plan(self, _repository_id, request):
            captured["request"] = dict(request)
            captured["agent_provider"] = os.environ.get("ALGENTA_AGENT_PROVIDER")
            captured["agent_model"] = os.environ.get("ALGENTA_AGENT_MODEL")
            return _plan(model="gpt-5.5")

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("ALGENTA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("ALGENTA_AGENT_MODEL", raising=False)
    monkeypatch.setattr(cli, "_register", lambda *_args, **_kwargs: ("rid", {"snapshot_id": "snap-1"}))
    monkeypatch.setattr(cli, "_issue_focus_paths", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_dump", lambda value: value)

    fix_run._plan_once(
        Client(),
        repo=str(repo),
        ref=None,
        issue="broken",
        failing=[],
        model="openai-native/gpt-5.5",
        open_pr=False,
        gh_token=None,
    )

    assert captured["request"]["model"] == "repository.verified_agentic_v1"
    assert captured["agent_provider"] == "openai-native"
    assert captured["agent_model"] == "gpt-5.5"
    assert "ALGENTA_AGENT_PROVIDER" not in os.environ
    assert "ALGENTA_AGENT_MODEL" not in os.environ


def test_plan_once_preserves_repository_planner_model(monkeypatch, tmp_path):
    captured = {}

    class Client:
        def triage_repository(self, _repository_id, _request):
            return {"workspace_evidence_bundle_ref": "bundle-1"}

        def create_repository_decision_plan(self, _repository_id, request):
            captured["request"] = dict(request)
            captured["agent_provider"] = os.environ.get("ALGENTA_AGENT_PROVIDER")
            captured["agent_model"] = os.environ.get("ALGENTA_AGENT_MODEL")
            return _plan(model="repository.deterministic_local_v1")

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("ALGENTA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("ALGENTA_AGENT_MODEL", raising=False)
    monkeypatch.setattr(cli, "_register", lambda *_args, **_kwargs: ("rid", {"snapshot_id": "snap-1"}))
    monkeypatch.setattr(cli, "_issue_focus_paths", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cli, "_dump", lambda value: value)

    fix_run._plan_once(
        Client(),
        repo=str(repo),
        ref=None,
        issue="broken",
        failing=[],
        model="repository.deterministic_local_v1",
        open_pr=False,
        gh_token=None,
    )

    assert captured["request"]["model"] == "repository.deterministic_local_v1"
    assert captured["agent_provider"] is None
    assert captured["agent_model"] is None


def test_apply_local_branch(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once",
                        lambda *a, **k: {"mode": "local_branch", "branch_name": "b", "local_checkout_path": "/co"})
    result = fix_run.run_fix(object(), _Args(apply=True))
    assert result["applied"]["branch_name"] == "b"
    assert result["verified"] is None  # no --tests → no verification


def test_open_pr_with_tests_keeps_remote_repo_as_fix_target(monkeypatch):
    captured = {}

    def fake_resolve_issue(args, github_token=None):
        captured["resolve_token"] = github_token
        return "1 failing test(s)", ["remote::test"]

    def fake_plan_once(_client, **kwargs):
        captured["plan"] = kwargs
        return "rid", {"snapshot_id": "s"}, _plan()

    monkeypatch.setattr(fix_run, "resolve_issue", fake_resolve_issue)
    monkeypatch.setattr(fix_run, "_plan_once", fake_plan_once)
    monkeypatch.setattr(
        fix_run,
        "_apply_once",
        lambda *a, **k: {
            "mode": "remote_pr",
            "pull_request_url": "https://github.com/owner/repo/pull/1",
            "status": "created",
        },
    )

    result = fix_run.run_fix(
        object(),
        _Args(
            repo="https://github.com/owner/repo.git",
            open_pr=True,
            tests=True,
            issue=None,
            github_token="ghs_test",
        ),
    )

    assert captured["resolve_token"] == "ghs_test"
    assert captured["plan"]["repo"] == "https://github.com/owner/repo.git"
    assert captured["plan"]["open_pr"] is True
    assert captured["plan"]["gh_token"] == "ghs_test"
    assert captured["plan"]["failing"] == ["remote::test"]
    assert result["pull_request_url"] == "https://github.com/owner/repo/pull/1"
    assert result["verified"] is None


# discover_failing_tests contract: GREEN is (None, []); a failure is (issue_text, [ids]) OR
# (issue_text, []) when the run failed but produced no per-test ids.
_FAIL = ("1 failing test(s): pkg::t1", ["t1"])
_GREEN = (None, [])


def _apply(*a, **k):
    return {"mode": "local_branch", "branch_name": "b", "local_checkout_path": "/co"}


def test_verify_green_first_try(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    seq = iter([_FAIL, _GREEN])  # resolve finds a failure; verify finds green
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd=None: next(seq))
    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=3))
    assert result["verified"] is True
    assert len(result["iterations"]) == 1


def test_test_cmd_with_apply_verifies_without_tests_flag(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    calls = []

    def fake_discover(repo, cmd=None):
        calls.append((repo, cmd))
        return _GREEN

    monkeypatch.setattr("codna.testrun.discover_failing_tests", fake_discover)

    result = fix_run.run_fix(object(), _Args(apply=True, tests=False, test_cmd="python -m unittest"))

    assert result["verified"] is True
    assert result["remaining_failing_tests"] == []
    assert calls == [("/co", "python -m unittest")]


def test_verify_refixes_until_green(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr(fix_run, "_rollback_failed_local_apply", lambda applied, before_head: {"local_checkout_path": "/co"})
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    # resolve → fail; verify#1 → still failing (re-fix); verify#2 → green
    seq = iter([_FAIL, _FAIL, _GREEN])
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd=None: next(seq))
    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=3))
    assert result["verified"] is True
    assert len(result["iterations"]) == 2  # took a re-fix


def test_retry_preserves_verification_output_with_explicit_issue(monkeypatch):
    """Regression: an explicit --issue must not hide traceback evidence on retry."""
    captured: list[dict] = []

    def fake_plan_once(_client, **kwargs):
        captured.append(dict(kwargs))
        return "rid", {"snapshot_id": "s"}, _plan(root=f"attempt-{len(captured)}")

    verify = iter([
        (
            "tests are failing (no per-test JUnit ids were produced):\n"
            "src/itsdangerous/encoding.py:34: binascii.Error: Incorrect padding",
            [],
        ),
        _GREEN,
    ])
    monkeypatch.setattr(fix_run, "resolve_issue", lambda args, github_token=None: (args.issue, []))
    monkeypatch.setattr(fix_run, "_plan_once", fake_plan_once)
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr(fix_run, "_rollback_failed_local_apply", lambda applied, before_head: {"local_checkout_path": "/co"})
    monkeypatch.setattr(fix_run, "_verify_applied_checkout", lambda applied, test_cmd=None: next(verify))
    monkeypatch.setattr("os.path.isdir", lambda p: True)

    result = fix_run.run_fix(
        object(),
        _Args(
            apply=True,
            tests=True,
            issue="base64_decode fails for stripped URL-safe padding",
            max_iterations=2,
        ),
    )

    assert result["verified"] is True
    assert len(captured) == 2
    retry_issue = captured[1]["issue"]
    assert "Original issue:" in retry_issue
    assert "base64_decode fails for stripped URL-safe padding" in retry_issue
    assert "Verification after the previous fix is still failing:" in retry_issue
    assert "src/itsdangerous/encoding.py:34" in retry_issue


def test_usage_totals_sum_multiple_fix_iterations(monkeypatch):
    plans = iter([
        _plan(usage={"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 5, "cost_usd": 0.01}, model="m1"),
        _plan(usage={"input_tokens": 200, "output_tokens": 20, "cache_write_tokens": 7, "cost_usd": 0.02}, model="m2"),
    ])
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, next(plans)))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr(fix_run, "_rollback_failed_local_apply", lambda applied, before_head: {"local_checkout_path": "/co"})
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    seq = iter([_FAIL, _FAIL, _GREEN])
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd=None: next(seq))

    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=2))

    assert result["usage"] == {
        "input_tokens": 300,
        "output_tokens": 30,
        "total_tokens": 330,
        "cache_read_tokens": 5,
        "cache_write_tokens": 7,
    }
    assert result["tokens"]["total"] == 330
    assert result["input_tokens"] == 300
    assert result["output_tokens"] == 30
    assert result["total_tokens"] == 330
    assert result["cache_read_tokens"] == 5
    assert result["cache_write_tokens"] == 7
    assert result["cost_usd"] == 0.03
    assert result["runtime_models"] == ["m1", "m2"]


def test_verify_gives_up_after_max_iterations(monkeypatch):
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr(fix_run, "_rollback_failed_local_apply", lambda applied, before_head: {"local_checkout_path": "/co"})
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd=None: _FAIL)  # never green
    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=2))
    assert result["verified"] is False
    assert result["remaining_failing_tests"] == ["t1"]
    assert len(result["iterations"]) == 2  # bounded


def test_verify_failing_without_ids_is_not_green(monkeypatch):
    """Regression: tests fail but produce NO per-test ids -> must NOT be reported verified/green."""
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once", _apply)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    # non-None issue text (failed) but EMPTY id list — the false-green trap
    monkeypatch.setattr("codna.testrun.discover_failing_tests",
                        lambda repo, cmd=None: ("tests are failing (no per-test JUnit ids)", []))
    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=1))
    assert result["verified"] is False  # NOT a false green


def test_verify_no_checkout_fails_closed(monkeypatch):
    """Regression: verification requested but no checkout to run against -> verified False (not None/0)."""
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    monkeypatch.setattr(fix_run, "_apply_once",
                        lambda *a, **k: {"mode": "local_branch", "branch_name": "b", "local_checkout_path": None})
    seq = iter([_FAIL])  # resolve finds a failure; apply returns no checkout
    monkeypatch.setattr("codna.testrun.discover_failing_tests", lambda repo, cmd=None: next(seq))
    result = fix_run.run_fix(object(), _Args(apply=True, tests=True, max_iterations=2))
    assert result["verified"] is False  # fail closed, not a silent pass


def test_cmd_fix_json_output_and_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda **_kwargs: object())

    def noisy_run_fix(c, args):
        print("internal engine log that must not corrupt stdout")
        return {"repository": "/r", "verified": False, "remaining_failing_tests": ["t1"], "iterations": []}

    monkeypatch.setattr(fix_run, "run_fix", noisy_run_fix)
    rc = cli.cmd_fix(_Args(as_json=True))
    captured = capsys.readouterr()
    assert rc == 1  # verified is False → non-zero (scriptable signal)
    assert json.loads(captured.out)["verified"] is False
    assert "internal engine log" in captured.err


def test_apply_codna_error_keeps_patch_ref_hint(monkeypatch):
    """Remote-engine apply failures (CodnaError) must still get the 'apply failed: … (patch ref)' hint."""
    import pytest
    from codna.cli import CodnaError
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan(patch="patch://abc")))

    def boom(*a, **k):
        raise CodnaError("engine POST /apply failed: status=422 message=repository_apply_gate_failed")

    monkeypatch.setattr(fix_run, "_apply_once", boom)
    with pytest.raises(CodnaError) as exc:
        fix_run.run_fix(object(), _Args(apply=True))
    msg = str(exc.value)
    assert "apply failed:" in msg and "patch://abc" in msg  # framing + recovery hint preserved


def test_verify_uses_applied_checkout_when_commit_sha_available(monkeypatch, tmp_path):
    """Regression: clean git worktrees omit untracked dependency caches such as node_modules.

    A local --apply fix should be verified in the checkout the user will continue using, not in
    a detached worktree that can fail only because install artifacts are untracked.
    """
    monkeypatch.setattr(fix_run, "_plan_once", lambda c, **k: ("rid", {"snapshot_id": "s"}, _plan()))
    checkout = tmp_path / "repo"
    checkout.mkdir()
    monkeypatch.setattr(
        fix_run,
        "_apply_once",
        lambda *a, **k: {
            "mode": "local_branch",
            "branch_name": "b",
            "local_checkout_path": str(checkout),
            "commit_sha": "a" * 40,
        },
    )
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    seen = []

    def fake_discover(repo, cmd=None):
        seen.append(repo)
        return _FAIL if len(seen) == 1 else _GREEN

    monkeypatch.setattr("codna.testrun.discover_failing_tests", fake_discover)

    result = fix_run.run_fix(object(), _Args(apply=True, tests=True))

    assert result["verified"] is True
    assert seen == ["/repo", str(checkout)]


def test_cleanup_generated_apply_artifacts_removes_new_untracked_patch_file(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    before = fix_run._snapshot_generated_apply_artifacts(str(repo))
    (repo / ".algenta.patch.diff").write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")

    fix_run._cleanup_generated_apply_artifacts(str(repo), before)

    assert not (repo / ".algenta.patch.diff").exists()


def test_apply_once_recovers_when_agent_edit_already_matches_plan(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    target = repo / "src" / "lib.rs"
    target.parent.mkdir()
    target.write_text("fn main() {\n    broken()\n}\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/lib.rs"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    target.write_text("fn main() {\n    fixed()\n}\n", encoding="utf-8")

    class AlreadyAppliedClient:
        def simulate_repository(self, *_a, **_k):
            return {"validated_inputs": {"simulation_id": "sim-1"}}

        def apply_repository(self, *_a, **_k):
            raise RuntimeError(
                "git apply --index .algenta.patch.diff failed: patch does not apply"
            )

    plan = {
        "decision_plan_id": "plan_1234567890abcdef",
        "changed_files": ["src/lib.rs"],
        "decision_plan": {"repository_analysis": {"generated_patch_ref": "patch-1"}},
    }

    result = fix_run._apply_once(
        AlreadyAppliedClient(),
        "rid",
        {"snapshot_id": "snap-1"},
        plan,
        open_pr=False,
        args=_Args(repo=str(repo)),
        issue="fix rust parse error",
        summary=fix_run.summarize_plan(plan),
        repo=str(repo),
    )

    assert result["recovered_after_apply_error"] is True
    assert result["branch_name"] == "codna/1234567890abcdef"
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True) == ""
    assert subprocess.check_output(["git", "show", "--format=", "--name-only", "HEAD"], cwd=repo, text=True).strip() == "src/lib.rs"


def test_apply_recovery_rejects_unplanned_dirty_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    target = repo / "src" / "lib.rs"
    extra = repo / "src" / "extra.rs"
    target.parent.mkdir()
    target.write_text("a\n", encoding="utf-8")
    extra.write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/lib.rs", "src/extra.rs"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    target.write_text("b\n", encoding="utf-8")
    extra.write_text("y\n", encoding="utf-8")

    recovered = fix_run._recover_already_applied_local_branch(
        str(repo),
        {"decision_plan_id": "plan_abcdef", "changed_files": ["src/lib.rs"]},
        RuntimeError("git apply .algenta.patch.diff failed: patch does not apply"),
    )

    assert recovered is None
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).splitlines() == [
        " M src/extra.rs",
        " M src/lib.rs",
    ]


def test_apply_recovery_applies_generated_patch_when_index_apply_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    target = repo / "src" / "lib.rs"
    target.parent.mkdir()
    target.write_text("bad();\nok();\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/lib.rs"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    (repo / ".algenta.patch.diff").write_text(
        "--- a/src/lib.rs\n"
        "+++ b/src/lib.rs\n"
        "@@ -1,2 +1 @@\n"
        "-bad();\n"
        " ok();\n",
        encoding="utf-8",
    )

    recovered = fix_run._recover_already_applied_local_branch(
        str(repo),
        {"decision_plan_id": "plan_abcdef1234567890"},
        RuntimeError("git apply --index .algenta.patch.diff failed: patch does not apply"),
    )

    assert recovered is not None
    assert recovered["recovered_after_apply_error"] is True
    assert target.read_text(encoding="utf-8") == "ok();\n"
    assert not (repo / ".algenta.patch.diff").exists()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True) == ""


def test_rollback_failed_local_apply_resets_only_expected_codna_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    target = repo / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    before = fix_run._git_head(str(repo))

    target.write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "codna bad attempt", "-q"], cwd=repo, check=True)
    applied_commit = fix_run._git_head(str(repo))

    rollback = fix_run._rollback_failed_local_apply(
        {"mode": "local_branch", "local_checkout_path": str(repo), "commit_sha": applied_commit},
        before,
    )

    assert rollback["reset_to"] == before
    assert fix_run._git_head(str(repo)) == before
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_rollback_failed_local_apply_rejects_unknown_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "codna-test@example.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Codna Test"], cwd=repo, check=True)
    target = repo / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    before = fix_run._git_head(str(repo))
    target.write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "user commit", "-q"], cwd=repo, check=True)

    try:
        fix_run._rollback_failed_local_apply(
            {"mode": "local_branch", "local_checkout_path": str(repo), "commit_sha": "0" * 40},
            before,
        )
    except RuntimeError as exc:
        assert "current HEAD does not match" in str(exc)
    else:
        raise AssertionError("rollback must reject unproven HEAD ownership")
