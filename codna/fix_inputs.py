"""Fix-input helpers for ``codna fix`` — resolve ``(issue_text, failing_test_ids)`` from
``--from-junit``, ``--failing-test``, or ``--tests`` (sandboxed auto-discovery), plus simulation-id
extraction and PR-body rendering. Split out of ``cli.py`` to keep it under the modularity ceiling.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from .testrun import TestEnvironmentUnavailable


class FixInputError(Exception):
    """Bad/unusable fix input, surfaced to the user (the CLI maps it to a clean error)."""


def from_junit(path: str):
    """``(issue_text, [failing_ids])`` from a JUnit/pytest XML report (ids as ``classname::name``)."""
    from .testrun import parse_junit_failures

    try:
        failing = parse_junit_failures(path)
    except Exception as exc:  # noqa: BLE001
        raise FixInputError(f"could not read JUnit report {path}: {exc}") from exc
    if not failing:
        return None, []
    return f"{len(failing)} failing test(s): " + "; ".join(failing[:8]), failing


def _clean_git_head(repo_dir: str) -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repo_dir,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if status.returncode != 0 or status.stdout:
        return None
    revision = head.stdout.strip()
    return revision if len(revision) == 40 else None


def _discover_failing_tests_for_fix(repo_dir: str, test_cmd: str | None):
    from .testrun import discover_failing_tests

    head = _clean_git_head(repo_dir)
    if head is None:
        return discover_failing_tests(repo_dir, test_cmd)

    from .worktree import EphemeralWorktree

    with EphemeralWorktree(repo_dir, head, prefix="codna-test-discovery-") as worktree:
        return discover_failing_tests(worktree.path, test_cmd)


def _is_remote_repo_spec(repo: str) -> bool:
    clean = (repo or "").strip()
    return clean.startswith(("http://", "https://", "git@")) or clean.endswith(".git")


_REMOTE_GIT_TIMEOUT_S = 120
# The initial clone is the one step that scales with repository size, and 120 s was measured too
# small for a real repo: thyn-ai/algenta (~81 MB packed) hit exactly 120 s on a 2-vCPU Fly machine
# on 2026-09-17, every time, so every CI-failure autofix there died before discovering a single
# test. Shallow (below) plus five minutes.
_REMOTE_CLONE_TIMEOUT_S = 300
# A shallow clone can only fetch what GitHub serves by name; an abbreviated SHA is not that.
_ABBREVIATED_SHA = re.compile(r"[0-9a-fA-F]{7,39}")


def _run_git_for_remote_tests(
    args: list[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float = _REMOTE_GIT_TIMEOUT_S,
) -> None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FixInputError(f"could not prepare remote checkout for --tests: {exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise FixInputError(f"could not prepare remote checkout for --tests: {stderr}")


def _checkout_remote_ref(checkout: str, ref: str, *, repository_url: str, token: str | None) -> None:
    from .packaged_git import _git_auth_env, _validated_ref

    clean_ref = _validated_ref(ref)
    env = _git_auth_env(repository_url, token)
    # A shallow clone holds nothing but the tip, so `checkout <ref>` after a failed fetch could
    # never work -- surface the fetch error itself instead of a misleading checkout error.
    _run_git_for_remote_tests(["fetch", "--depth=1", "origin", clean_ref], cwd=checkout, env=env)
    _run_git_for_remote_tests(["checkout", "--detach", "FETCH_HEAD"], cwd=checkout, env=env)
    _run_git_for_remote_tests(["reset", "--hard", "HEAD"], cwd=checkout, env=env)
    _run_git_for_remote_tests(["clean", "-fdx"], cwd=checkout, env=env)


def _discover_remote_failing_tests_for_fix(args, github_token: str | None):
    from .packaged_git import PackagedGitError, _git_auth_env, _reject_embedded_credentials

    repo_url = getattr(args, "repo", "") or ""
    try:
        _reject_embedded_credentials(repo_url)
        env = _git_auth_env(repo_url, github_token)
    except PackagedGitError as exc:
        raise FixInputError(str(exc)) from exc

    root = tempfile.mkdtemp(prefix="codna-remote-test-discovery-")
    checkout = os.path.join(root, "repo")
    try:
        # ALWAYS shallow. This used to skip --depth=1 whenever a ref was given, on the theory that
        # the ref might not be the branch tip -- but _checkout_remote_ref fetches the exact ref
        # shallowly right after (`git fetch --depth=1 origin <ref>`), so the full history was
        # downloaded and then never used. On a large repo that full clone is precisely what blew
        # the timeout (thyn-ai/algenta, 2026-09-17: 120 s, every attempt).
        clone_args = ["clone", "--no-tags", "--depth=1"]
        ref = getattr(args, "ref", None)
        if ref and _ABBREVIATED_SHA.fullmatch(str(ref)):
            raise FixInputError(
                "--ref looks like an abbreviated commit SHA, which a shallow clone cannot resolve; "
                "pass the full 40-character SHA, a branch, or a tag"
            )
        base_branch = getattr(args, "base_branch", None)
        if not ref and base_branch:
            clone_args.extend(["--single-branch", "--branch", base_branch])
        clone_args.extend([repo_url, checkout])
        _run_git_for_remote_tests(
            clone_args, cwd=None, env=env, timeout=_REMOTE_CLONE_TIMEOUT_S
        )
        if ref:
            _checkout_remote_ref(checkout, ref, repository_url=repo_url, token=github_token)
        return _discover_failing_tests_for_fix(checkout, getattr(args, "test_cmd", None))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def resolve_issue(args, github_token: str | None = None):
    """Resolve ``(issue_text, failing_test_ids)`` from the fix args. Raises FixInputError on bad input.

    Precedence: ``--from-junit`` and ``--failing-test`` are explicit inputs; ``--tests`` auto-discovers
    by running the repo's tests (sandboxed). An explicit ``--issue`` always overrides the synthesized
    text; explicit ``--failing-test`` ids are merged in either way.
    """
    explicit_failing = list(getattr(args, "failing_test", None) or [])

    if getattr(args, "from_junit", None):
        issue, failing = from_junit(args.from_junit)
        merged = list(dict.fromkeys([*explicit_failing, *failing]))
        return (args.issue or issue), merged

    if getattr(args, "tests", False):
        try:
            repo = getattr(args, "repo", ".") or "."
            if _is_remote_repo_spec(repo):
                issue, failing = _discover_remote_failing_tests_for_fix(args, github_token)
            else:
                repo_dir = os.path.abspath(os.path.expanduser(repo))
                issue, failing = _discover_failing_tests_for_fix(repo_dir, getattr(args, "test_cmd", None))
        except TestEnvironmentUnavailable:
            # Keeps its own error code: the CLI prints it structured and the GitHub App ends the
            # check neutral -- flattening it into a cli_error made the check red (mojo-kernels#1).
            raise
        except Exception as exc:  # noqa: BLE001 - surface any discovery failure as a clean CLI error
            raise FixInputError(str(exc)) from exc
        merged = list(dict.fromkeys([*explicit_failing, *failing]))
        issue = args.issue or issue
        if not issue and not merged:
            raise FixInputError(
                "`codna fix --tests` found no failing tests — nothing to fix "
                "(pass --issue to describe a different problem)."
            )
        return issue, merged

    return getattr(args, "issue", None), explicit_failing


def sim_id(sim: dict):
    # /simulate returns a DecisionEnvelope; the simulation id is surfaced in
    # validated_inputs.simulation_id (see engine execution_workflow ~L499).
    vi = sim.get("validated_inputs") or {}
    if vi.get("simulation_id"):
        return vi["simulation_id"]
    for k in ("simulation_id", "id"):
        if sim.get(k):
            return sim[k]
    for nest in ("envelope", "decision_plan", "result"):
        v = sim.get(nest) or {}
        if isinstance(v, dict) and v.get("simulation_id"):
            return v["simulation_id"]
    return None


def pr_body(issue, ra, plan):
    dp = plan.get("decision_plan") or {}
    lines = ["Automated fix by **codna**.", "", f"**Issue:** {issue}"]
    if ra.get("root_cause"):
        lines.append(f"**Root cause:** {ra['root_cause']}")
    if ra.get("impacted_symbols"):
        lines.append(f"**Symbols:** {', '.join(ra['impacted_symbols'])}")
    if dp.get("confidence") is not None:
        lines.append(f"**Confidence:** {int(dp['confidence'] * 100)}%")
    lines += ["", "_Review before merging._"]
    return "\n".join(lines)
