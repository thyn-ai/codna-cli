"""``codna fix`` orchestration: one fix attempt + the verification loop.

Kept out of ``cli.py`` for the modularity ceiling. Engine calls go through the client ``c``; the cli
helpers (``_register`` / ``_dump`` / ``_sim_id`` / ``_pr_body`` / ``_issue_focus_paths`` / ``_resolve_issue``)
are imported lazily to avoid an import cycle.

The verification loop (``--tests --apply``): plan → apply (local branch) → re-run the repo's tests at the
applied checkout → if still failing and iterations remain, re-fix *on the applied branch* with the new
failing tests as the signal, until green or ``--max-iterations`` is hit. Test runs use the hardened
:mod:`~codna.testrun` sandbox (scrubbed env, enforced egress denial). ``--open-pr`` is one-shot (no loop —
you don't iterate a pushed PR from here).
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
from pathlib import Path

from .fix_inputs import FixInputError, pr_body, resolve_issue, sim_id

_REPOSITORY_AGENTIC_MODEL = "repository.verified_agentic_v1"


def summarize_plan(plan: dict) -> dict:
    """Extract the stable fields the CLI reports (human + JSON) from a decision plan."""
    dp = plan.get("decision_plan") or {}
    ra = dp.get("repository_analysis") or {}
    usage = _normalized_usage(plan.get("planner_usage") or {})
    tr = ra.get("token_reduction_metrics") or {}
    risk = dp.get("risk") or {}
    return {
        "root_cause": ra.get("root_cause"),
        "impacted_symbols": ra.get("impacted_symbols") or [],
        "blast_radius": ra.get("blast_radius"),
        "confidence": dp.get("confidence"),
        "regression_risk": risk.get("probability_of_loss"),
        "patch_ref": ra.get("generated_patch_ref"),
        "runtime_model": plan.get("runtime_model"),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cache_read_tokens": usage["cache_read_tokens"],
        "cache_write_tokens": usage["cache_write_tokens"],
        "tokens": {
            "input": usage["input_tokens"],
            "output": usage["output_tokens"],
            "total": usage["total_tokens"],
            "cache_read": usage["cache_read_tokens"],
            "cache_write": usage["cache_write_tokens"],
        },
        "cost_usd": usage["cost_usd"],
        "reduction_ratio": tr.get("reduction_ratio"),
        "raw_repo_tokens": tr.get("raw_repo_token_estimate"),
        "evidence_bundle_tokens": tr.get("evidence_bundle_token_count"),
    }


def _normalized_usage(raw: dict) -> dict:
    input_tokens = _int_or_zero(raw.get("input_tokens"))
    output_tokens = _int_or_zero(raw.get("output_tokens"))
    cache_read_tokens = _int_or_zero(raw.get("cache_read_tokens"))
    cache_write_tokens = _int_or_zero(raw.get("cache_write_tokens"))
    cost = raw.get("cost_usd")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cost_usd": cost if isinstance(cost, (int, float)) else None,
    }


def _int_or_zero(value) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _decision_plan_model_and_agent_env(model: str | None) -> tuple[str, dict[str, str]]:
    requested = (model or "").strip()
    if not requested:
        return _REPOSITORY_AGENTIC_MODEL, {}
    if requested.startswith("repository."):
        return requested, {}
    if "/" not in requested:
        return _REPOSITORY_AGENTIC_MODEL, {"ALGENTA_AGENT_MODEL": requested}
    provider, model_id = requested.split("/", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        return _REPOSITORY_AGENTIC_MODEL, {"ALGENTA_AGENT_MODEL": requested}
    return _REPOSITORY_AGENTIC_MODEL, {
        "ALGENTA_AGENT_PROVIDER": provider,
        "ALGENTA_AGENT_MODEL": model_id,
    }


@contextlib.contextmanager
def _scoped_agent_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def attach_usage_totals(result: dict) -> dict:
    """Expose aggregate model usage at the top level for scriptable ``codna fix --json`` users."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    total_cost = 0.0
    saw_cost = False
    runtime_models: list[str] = []
    for iteration in result.get("iterations") or []:
        summary = iteration.get("summary") if isinstance(iteration, dict) else None
        if not isinstance(summary, dict):
            continue
        usage = _normalized_usage(summary)
        for key in totals:
            totals[key] += usage[key]
        if usage["cost_usd"] is not None:
            total_cost += float(usage["cost_usd"])
            saw_cost = True
        model = summary.get("runtime_model")
        if isinstance(model, str) and model and model not in runtime_models:
            runtime_models.append(model)
    result["usage"] = totals
    result["tokens"] = {
        "input": totals["input_tokens"],
        "output": totals["output_tokens"],
        "total": totals["total_tokens"],
        "cache_read": totals["cache_read_tokens"],
        "cache_write": totals["cache_write_tokens"],
    }
    # Keep the historical flat fields for scripts/benchmarks that consume
    # `codna fix --json` directly, while also exposing the structured groups.
    result.update(totals)
    result["cost_usd"] = total_cost if saw_cost else None
    if len(runtime_models) == 1:
        result["runtime_model"] = runtime_models[0]
    elif runtime_models:
        result["runtime_models"] = runtime_models
    return result


def _plan_once(c, *, repo, ref, issue, failing, model, open_pr, gh_token):
    from . import cli as _cli

    local = os.path.abspath(os.path.expanduser(repo)) if repo else "."
    focus = _cli._issue_focus_paths(local, issue) if os.path.isdir(local) else []
    rid, snap = _cli._register(c, repo, ref, github_token=gh_token if open_pr else None, focus_paths=focus)
    sig = {"issue_text": issue, "failing_tests": failing}
    if focus:
        sig["changed_files"] = focus
    # Surface project guidance (AGENTS.md / .codna/rules) so the engine/agent can honor conventions.
    if os.path.isdir(local):
        from .project_rules import read_project_guidance
        guidance = read_project_guidance(local)
        if guidance:
            sig["project_guidance"] = guidance
    tri = _cli._dump(c.triage_repository(rid, {"snapshot_id": snap["snapshot_id"], "signals": sig}))
    plan_model, agent_env = _decision_plan_model_and_agent_env(model)
    with _scoped_agent_env(agent_env):
        plan = _cli._dump(c.create_repository_decision_plan(rid, {
            "snapshot_id": snap["snapshot_id"],
            "workspace_evidence_bundle_ref": tri.get("workspace_evidence_bundle_ref"),
            "signals": sig,
            "model": plan_model,
        }))
    return rid, snap, plan


_GENERATED_APPLY_ARTIFACTS = (".algenta.patch.diff", ".codna-packaged.patch.diff")


def _snapshot_generated_apply_artifacts(repo: str) -> dict[str, bool]:
    root = Path(os.path.abspath(os.path.expanduser(repo)))
    return {name: (root / name).exists() for name in _GENERATED_APPLY_ARTIFACTS}


def _is_untracked_file(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard", "--", relative],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result.returncode == 0 and relative in {line.strip() for line in result.stdout.splitlines()}


def _cleanup_generated_apply_artifacts(checkout: str | None, before: dict[str, bool]) -> None:
    if not checkout:
        return
    repo_root = Path(os.path.abspath(os.path.expanduser(checkout)))
    for name in _GENERATED_APPLY_ARTIFACTS:
        if before.get(name):
            continue
        path = repo_root / name
        if path.is_file() and _is_untracked_file(repo_root, path):
            path.unlink()


def _git_head(repo: str | None) -> str | None:
    if not repo:
        return None
    result = subprocess.run(
        ["git", "-C", os.path.abspath(os.path.expanduser(repo)), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _rollback_failed_local_apply(applied: dict, before_head: str | None) -> dict:
    checkout = applied.get("local_checkout_path")
    applied_commit = applied.get("commit_sha")
    if not checkout or not before_head or not applied_commit:
        raise RuntimeError("cannot rollback failed attempt: missing checkout, base commit, or applied commit")

    repo = os.path.abspath(os.path.expanduser(str(checkout)))
    current_head = _git_head(repo)
    if current_head != applied_commit:
        raise RuntimeError(
            "cannot rollback failed attempt: current HEAD does not match Codna-applied commit "
            f"(current={current_head or 'unknown'}, applied={applied_commit})"
        )

    reset = subprocess.run(
        ["git", "-C", repo, "reset", "--hard", "--quiet", before_head],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if reset.returncode != 0:
        detail = (reset.stderr or reset.stdout or "").strip()
        raise RuntimeError(f"cannot rollback failed attempt: git reset failed ({detail})")
    return {"mode": "local_branch", "local_checkout_path": repo, "reset_to": before_head}


def _verify_applied_checkout(applied: dict, test_cmd: str | None):
    from .testrun import discover_failing_tests

    checkout = applied.get("local_checkout_path")
    return discover_failing_tests(checkout, test_cmd)


def _retry_issue_text(original_issue: str | None, verify_issue: str | None, still_failing: list[str]) -> str:
    """Build the next attempt's issue text from the user request plus live verification output."""
    sections: list[str] = []
    if original_issue and original_issue.strip():
        sections.append("Original issue:\n" + original_issue.strip())
    if verify_issue and verify_issue.strip():
        sections.append("Verification after the previous fix is still failing:\n" + verify_issue.strip())
    elif still_failing:
        sections.append(
            "Verification after the previous fix still reports failing tests:\n"
            + "\n".join(f"- {test_id}" for test_id in still_failing[:12])
        )
    else:
        sections.append("Verification after the previous fix is still failing.")
    return "\n\n".join(sections)


def _fallback_pr_title(issue: str | None, summary: dict) -> str:
    """PR title when the caller passed no --pr-title.

    Prefers the request's OWN first line (for a webhook fix that is the issue title) over the model's
    ``root_cause``, because root_cause is free text and models fill it with narrative: a live fix PR
    shipped titled "codna: fix Here's a summary of every change made:". The webhook now always passes
    --pr-title, so this is the safety net for every other caller (`codna fix` run by hand, the Action).
    """
    from .webhook import pr_title_from_issue_text  # local import: avoids a module-level cycle

    for candidate in (issue, summary.get("root_cause")):
        title = pr_title_from_issue_text(candidate, limit=60)
        if title:
            return f"codna: fix {title}"
    return "codna: fix bug"


def _apply_once(c, rid, snap, plan, *, open_pr, args, issue, summary, repo):
    from . import cli as _cli

    sim = _cli._dump(c.simulate_repository(rid, {"snapshot_id": snap["snapshot_id"],
                                                 "decision_plan_id": plan.get("decision_plan_id")}))
    sid = sim_id(sim)
    if not sid:
        raise _cli.CodnaError(f"no simulation id in simulate() response (keys: {list(sim)[:8]})")
    if open_pr:
        res = _cli._dump(c.apply_repository(rid, {
            "mode": "remote_pr", "write_permission": True,
            "decision_plan_id": plan.get("decision_plan_id"), "simulation_id": sid,
            **({"base_branch": args.base_branch} if args.base_branch else {}),
            "pull_request_title": args.pr_title or _fallback_pr_title(issue, summary),
            "pull_request_body": args.pr_body or pr_body(issue, {"root_cause": summary.get("root_cause"),
                                                                       "impacted_symbols": summary.get("impacted_symbols")}, plan),
        }))
        return {"mode": "remote_pr", "pull_request_url": res.get("pull_request_url"), "status": res.get("status")}
    try:
        res = _cli._dump(c.apply_repository(rid, {
            "mode": "local_branch", "write_permission": True,
            "decision_plan_id": plan.get("decision_plan_id"), "simulation_id": sid}))
    except Exception as exc:  # noqa: BLE001
        recovered = _recover_already_applied_local_branch(repo, plan, exc)
        if recovered is None:
            raise
        return recovered
    return {"mode": "local_branch", "branch_name": res.get("branch_name"),
            "local_checkout_path": res.get("local_checkout_path"), "commit_sha": res.get("commit_sha")}


def _recover_already_applied_local_branch(repo: str, plan: dict, exc: BaseException) -> dict | None:
    """Commit an already-applied agent edit only when it exactly matches plan evidence.

    Some local SDK apply paths call the agent first, then apply the generated diff with
    git. If the agent already changed the worktree, the second git apply can reject the
    patch even though the checkout is correct. Recovery is intentionally narrow: the
    failure must be a git-apply rejection, the dirty tracked paths must exactly match the
    plan's changed files, and generated patch artifacts are the only ignored untracked files.
    """
    if not _is_git_apply_rejection(exc):
        return None
    repo_path = os.path.abspath(os.path.expanduser(repo))
    if not os.path.isdir(repo_path):
        return None
    plan_id = plan.get("decision_plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        return None
    patch_path = _generated_apply_patch_path(repo_path)
    expected = _plan_changed_files(plan)
    if not expected and patch_path is not None:
        expected = _patch_changed_files(patch_path.read_text(encoding="utf-8", errors="surrogateescape"))
    if not expected:
        return None
    state = _worktree_recovery_state(repo_path)
    if state is None:
        return None
    tracked_dirty, untracked = state
    ignored_untracked = {name for name in _GENERATED_APPLY_ARTIFACTS}
    unexpected_untracked = [path for path in untracked if path not in ignored_untracked]
    if unexpected_untracked:
        return None
    branch = "codna/" + plan_id.removeprefix("plan_")[:16]
    _git_checked(repo_path, ["checkout", "-B", branch])
    if tracked_dirty:
        if sorted(tracked_dirty) != sorted(expected):
            return None
    elif patch_path is not None:
        if not _git_apply_check(repo_path, patch_path):
            return None
        _git_checked(repo_path, ["apply", "--whitespace=nowarn", str(patch_path)])
    else:
        return None
    _git_checked(repo_path, ["add", "--", *expected])
    staged = _git_checked(repo_path, ["diff", "--cached", "--name-only", "-z"]).stdout
    staged_files = sorted(path for path in staged.split("\0") if path)
    if staged_files != sorted(expected):
        return None
    _git_checked(
        repo_path,
        [
            "-c",
            "user.name=Codna",
            "-c",
            "user.email=codna@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-m",
            f"codna: apply {plan_id}",
        ],
    )
    for artifact in _GENERATED_APPLY_ARTIFACTS:
        Path(repo_path, artifact).unlink(missing_ok=True)
    return {
        "mode": "local_branch",
        "branch_name": branch,
        "local_checkout_path": repo_path,
        "commit_sha": _git_checked(repo_path, ["rev-parse", "HEAD"]).stdout.strip(),
        "recovered_after_apply_error": True,
    }


def _is_git_apply_rejection(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "git apply" in text
        or "patch does not apply" in text
        or "git_command_failed" in text
    ) and (
        ".algenta.patch.diff" in text
        or ".codna-packaged.patch.diff" in text
        or "patch does not apply" in text
    )


def _plan_changed_files(plan: dict) -> list[str]:
    candidates = [
        plan.get("changed_files"),
        (plan.get("decision_plan") or {}).get("changed_files"),
        ((plan.get("decision_plan") or {}).get("repository_analysis") or {}).get("changed_files"),
    ]
    for value in candidates:
        if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
            return sorted(set(value))
    return []


def _generated_apply_patch_path(repo: str) -> Path | None:
    for artifact in _GENERATED_APPLY_ARTIFACTS:
        path = Path(repo, artifact)
        if path.is_file():
            return path
    return None


def _patch_changed_files(patch_diff: str) -> list[str]:
    changed: set[str] = set()
    for line in patch_diff.splitlines():
        if not line.startswith("+++ "):
            continue
        path = line[4:].strip()
        if path == "/dev/null":
            continue
        if path.startswith("b/"):
            path = path[2:]
        if path:
            changed.add(path)
    return sorted(changed)


def _git_apply_check(repo: str, patch_path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", repo, "apply", "--check", "--whitespace=nowarn", str(patch_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.returncode == 0


def _worktree_recovery_state(repo: str) -> tuple[list[str], list[str]] | None:
    status = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if status.returncode != 0:
        return None
    tracked: list[str] = []
    untracked: list[str] = []
    entries = [entry for entry in status.stdout.split("\0") if entry]
    i = 0
    while i < len(entries):
        entry = entries[i]
        code = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        if code == "??":
            untracked.append(path)
        elif code[:1] in {"R", "C"} or code[1:2] in {"R", "C"}:
            return None
        elif path:
            tracked.append(path)
        i += 1
    return sorted(set(tracked)), sorted(set(untracked))


def _git_checked(repo: str, args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed during apply recovery: {detail}")
    return result


def run_fix(c, args) -> dict:
    """Run the fix (with optional verification loop) and return a structured result dict.

    The CLI renders this as human text or ``--json``. Raises CodnaError on unrecoverable failures.
    """
    from . import cli as _cli
    from .cli_errors import format_cli_error

    gh_token = (args.github_token or os.environ.get("CODNA_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    open_pr = bool(getattr(args, "open_pr", False))
    apply = bool(getattr(args, "apply", False))
    if open_pr:
        if os.path.isdir(os.path.abspath(os.path.expanduser(args.repo))):
            _cli._die("--open-pr needs a git URL (a local path has no remote to push to).")
        if not gh_token:
            _cli._die("--open-pr needs a write token: pass --github-token or set GITHUB_TOKEN.")

    try:
        issue, failing = resolve_issue(args, github_token=gh_token)
    except FixInputError as exc:
        _cli._die(str(exc))
    if not issue and not failing:
        _cli._die("`codna fix` needs --issue (or --from-junit / --tests) describing what's broken.")

    verify = (bool(getattr(args, "tests", False)) or bool(getattr(args, "test_cmd", None))) and apply and not open_pr
    max_iterations = max(1, int(getattr(args, "max_iterations", 1) or 1)) if verify else 1

    t0 = time.perf_counter()
    repo = args.repo
    original_issue = issue
    result: dict = {"repository": repo, "issue": issue, "iterations": [], "applied": None,
                    "verified": None, "remaining_failing_tests": None, "pull_request_url": None}

    for i in range(max_iterations):
        try:
            rid, snap, plan = _plan_once(c, repo=repo, ref=args.ref, issue=issue, failing=failing,
                                         model=args.model, open_pr=open_pr, gh_token=gh_token)
        except _cli.CodnaError:
            raise
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__module__.startswith("codna."):
                raise
            raise _cli.CodnaError(f"fix failed: {format_cli_error(exc)}") from exc

        summary = summarize_plan(plan)
        iteration = {"n": i + 1, "summary": summary}

        if not (apply or open_pr):
            result["iterations"].append(iteration)
            result["elapsed_s"] = round(time.perf_counter() - t0, 1)
            return attach_usage_totals(result)

        pre_apply_head = _git_head(repo) if apply and not open_pr else None
        apply_artifacts_before = _snapshot_generated_apply_artifacts(repo) if apply and not open_pr else {}
        try:
            applied = _apply_once(c, rid, snap, plan, open_pr=open_pr, args=args, issue=issue, summary=summary, repo=repo)
            if applied.get("mode") == "local_branch":
                _cleanup_generated_apply_artifacts(applied.get("local_checkout_path"), apply_artifacts_before)
        except Exception as exc:  # noqa: BLE001 — mirror the old cmd_fix: ALL apply errors (incl. a
            # CodnaError from the remote engine's risk gate, or a missing sim id) get the 'apply failed:'
            # framing + the patch-ref recovery hint, so the user can still apply the generated patch.
            raise _cli.CodnaError(
                f"apply failed: {format_cli_error(exc)}"
                + (f"  (patch ref: {summary.get('patch_ref')})" if summary.get("patch_ref") else "")
            ) from exc
        iteration["applied"] = applied
        result["applied"] = applied
        # Surface the PR URL at the top level so `--json` consumers (the GitHub Action) read one
        # stable field instead of scraping human output with a regex.
        if applied.get("mode") == "remote_pr" and applied.get("pull_request_url"):
            result["pull_request_url"] = applied["pull_request_url"]

        if not verify:
            result["iterations"].append(iteration)
            break

        # Verify: re-run the repo's tests at the applied checkout.
        checkout = applied.get("local_checkout_path")
        if not checkout or not os.path.isdir(checkout):
            # Verification was requested but cannot run — fail closed (never a false "verified").
            iteration["verify_error"] = "no local checkout to verify against"
            result["verified"] = False
            result["iterations"].append(iteration)
            break
        verify_issue, still_failing = _verify_applied_checkout(applied, getattr(args, "test_cmd", None))
        # GREEN iff discover reports no failure at all. `still_failing` can be [] even when the tests
        # FAILED (custom --test-cmd with no JUnit / a collection error) — in that case verify_issue is a
        # non-None "tests are failing …" string, so keying greenness on the id-list alone would falsely
        # pass. Green is verify_issue is None.
        green = verify_issue is None
        iteration["still_failing"] = still_failing
        iteration["tests_pass"] = green
        if not green and not still_failing:
            iteration["verify_note"] = verify_issue  # failure with no per-test ids
        result["iterations"].append(iteration)
        if green:
            result["verified"] = True
            result["remaining_failing_tests"] = []
            break
        # Still red — re-fix ON the applied branch with the new failures, if iterations remain.
        result["verified"] = False
        result["remaining_failing_tests"] = still_failing
        if i + 1 >= max_iterations:
            break
        if applied.get("mode") == "local_branch":
            try:
                iteration["rollback"] = _rollback_failed_local_apply(applied, pre_apply_head)
            except Exception as exc:  # noqa: BLE001
                raise _cli.CodnaError(f"retry rollback failed: {format_cli_error(exc)}") from exc
        failing = still_failing
        issue = _retry_issue_text(args.issue or original_issue, verify_issue, still_failing)
        repo = checkout

    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return attach_usage_totals(result)


def render_human(result: dict) -> list[str]:
    """Render a run_fix() result as human-readable lines (the non-JSON output)."""
    lines: list[str] = []
    iters = result.get("iterations") or []
    multi = len(iters) > 1
    for it in iters:
        s = it.get("summary") or {}
        lines.append(f"\n[attempt {it.get('n')}]" if multi else f"\n✓ codna analyzed {result.get('repository')}")
        lines.append(f"  root cause   : {s.get('root_cause', '?')}")
        if s.get("impacted_symbols"):
            lines.append(f"  symbol       : {', '.join(s['impacted_symbols'])}  (blast radius: {s.get('blast_radius', '?')})")
        conf = int((s.get("confidence") or 0) * 100)
        rr = s.get("regression_risk")
        lines.append(f"  confidence   : {conf}%" + (f"  ·  regression risk: {int(rr * 100)}%" if rr is not None else ""))
        if s.get("reduction_ratio"):
            lines.append(f"  context      : {int(s.get('raw_repo_tokens', 0)):,} → "
                         f"{int(s.get('evidence_bundle_tokens', 0)):,} tokens  ({s['reduction_ratio']:.0f}× smaller)")
        tok = s.get("tokens") or {}
        itk, otk = tok.get("input"), tok.get("output")
        have = isinstance(itk, (int, float)) and isinstance(otk, (int, float))
        lines.append(f"  agent        : {s.get('runtime_model') or '?'} via codna"
                     + (f"  ·  tokens: {int(itk):,} in / {int(otk):,} out" if have else "")
                     + (f"  ·  cost: ${s['cost_usd']:.3f}" if s.get("cost_usd") is not None else ""))
        applied = it.get("applied")
        if applied and applied.get("mode") == "remote_pr":
            url = applied.get("pull_request_url")
            lines.append(f"\n✓ opened pull request: {url}" if url else f"  applied (no PR url): {applied.get('status')}")
        elif applied:
            lines.append(f"  applied      : branch {applied.get('branch_name')} "
                         f"@ {applied.get('local_checkout_path') or applied.get('commit_sha')}")
        if it.get("verify_error"):
            lines.append(f"  verify       : ✗ could not verify ({it['verify_error']})")
        elif "tests_pass" in it:
            if it["tests_pass"]:
                lines.append("  verify       : ✓ tests pass")
            else:
                sf = it.get("still_failing") or []
                lines.append(f"  verify       : ✗ {len(sf)} still failing" if sf else "  verify       : ✗ tests failing")
    if result.get("verified") is True:
        lines.append(f"\n✓ verified — tests pass after {len(iters)} attempt(s) in {result.get('elapsed_s')}s")
    elif result.get("verified") is False:
        rem = result.get("remaining_failing_tests") or []
        lines.append(f"\n✗ not fully green after {len(iters)} attempt(s): {len(rem)} test(s) still failing")
    last = iters[-1] if iters else {}
    if not result.get("applied") and (last.get("summary") or {}).get("patch_ref"):
        lines.append(f"  patch        : {last['summary']['patch_ref']}   "
                     "(--apply for a local branch · --open-pr to open a PR)")
    return lines
