"""`codna fix --tests` remote discovery: the clone must be shallow and get a size-scaled timeout.

Regression guard for a live failure (thyn-ai/algenta, 2026-09-17): with a ref given, the clone ran
WITHOUT --depth=1 (full history) under a flat 120 s timeout and timed out on every attempt, so no
CI-failure autofix on that repo ever got as far as discovering a test.
"""

from __future__ import annotations

import types

from codna import fix_inputs


def _capture(monkeypatch):
    calls = []

    def fake_run(args, *, cwd, env, timeout=fix_inputs._REMOTE_GIT_TIMEOUT_S):
        calls.append({"args": list(args), "timeout": timeout})

    monkeypatch.setattr(fix_inputs, "_run_git_for_remote_tests", fake_run)
    monkeypatch.setattr(
        fix_inputs,
        "_checkout_remote_ref",
        lambda *a, **k: calls.append({"checkout_ref": a[1]}),
    )
    monkeypatch.setattr(
        fix_inputs,
        "_discover_failing_tests_for_fix",
        lambda checkout, test_cmd: ("issue", []),
    )
    monkeypatch.setattr(
        "codna.packaged_git._git_auth_env",
        lambda url, token: {"GIT_TERMINAL_PROMPT": "0"},
    )
    monkeypatch.setattr(
        "codna.packaged_git._reject_embedded_credentials", lambda url: None
    )
    return calls


def _args(**over):
    base = {
        "repo": "https://github.com/acme/app.git",
        "ref": None,
        "base_branch": None,
        "test_cmd": None,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def test_clone_is_shallow_even_when_a_ref_is_given(monkeypatch):
    calls = _capture(monkeypatch)
    fix_inputs._discover_remote_failing_tests_for_fix(
        _args(ref="cc26ddd6" + "0" * 32), github_token="t"
    )
    clone = calls[0]["args"]
    assert clone[0] == "clone" and "--depth=1" in clone, clone
    # the exact ref is still checked out afterwards (shallow fetch lives in _checkout_remote_ref)
    assert {"checkout_ref": "cc26ddd6" + "0" * 32} in calls


def test_clone_gets_the_size_scaled_timeout_not_the_flat_git_default(monkeypatch):
    calls = _capture(monkeypatch)
    fix_inputs._discover_remote_failing_tests_for_fix(
        _args(ref="cc26ddd6" + "0" * 32), github_token="t"
    )
    assert calls[0]["timeout"] == fix_inputs._REMOTE_CLONE_TIMEOUT_S
    assert fix_inputs._REMOTE_CLONE_TIMEOUT_S > fix_inputs._REMOTE_GIT_TIMEOUT_S


def test_single_branch_narrowing_only_applies_without_a_ref(monkeypatch):
    calls = _capture(monkeypatch)
    fix_inputs._discover_remote_failing_tests_for_fix(
        _args(base_branch="main"), github_token="t"
    )
    assert ["--single-branch", "--branch", "main"] == [
        a for a in calls[0]["args"] if a in ("--single-branch", "--branch", "main")
    ]
    calls.clear()
    fix_inputs._discover_remote_failing_tests_for_fix(
        _args(ref="abc123", base_branch="main"), github_token="t"
    )
    assert (
        "--single-branch" not in calls[0]["args"]
    )  # the ref, not the branch tip, is what gets checked out
    assert "--depth=1" in calls[0]["args"]



def test_abbreviated_sha_ref_is_rejected_before_cloning(monkeypatch):
    import types

    called = []
    monkeypatch.setattr(fix_inputs, "_run_git_for_remote_tests", lambda *a, **k: called.append(a))
    args = types.SimpleNamespace(repo="https://github.com/acme/app.git", ref="abc1234", base_branch=None, test_cmd=None)
    try:
        fix_inputs._discover_remote_failing_tests_for_fix(args, None)
    except fix_inputs.FixInputError as exc:
        assert "abbreviated" in str(exc) and "40-character" in str(exc)
    else:
        raise AssertionError("expected FixInputError")
    assert called == []


def test_checkout_remote_ref_surfaces_the_fetch_error(monkeypatch, tmp_path):
    def _git(args, *, cwd, env, timeout=None):
        if args[0] == "fetch":
            raise fix_inputs.FixInputError("could not prepare remote checkout for --tests: fatal: couldn't find remote ref nope")
        raise AssertionError(f"unexpected git call after a failed fetch: {args}")

    monkeypatch.setattr(fix_inputs, "_run_git_for_remote_tests", _git)
    try:
        fix_inputs._checkout_remote_ref(str(tmp_path), "nope", repository_url="https://github.com/acme/app.git", token=None)
    except fix_inputs.FixInputError as exc:
        assert "couldn't find remote ref nope" in str(exc)
    else:
        raise AssertionError("expected FixInputError")
