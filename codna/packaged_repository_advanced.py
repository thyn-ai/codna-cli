from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .packaged_git import _git_identity_args, open_remote_pr
from .patch_text import GIT_APPLY_FLAG_SETS, normalize_unified_diff

SCHEMA_VERSION = 1
BACKEND_MODE = "packaged_local_fix"


class PackagedRepositoryAdvancedError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class PackagedAgentRunRequest:
    repository_id: str
    snapshot_id: str
    repo_root: Path
    issue_text: str
    model: str
    signals: dict[str, Any]
    evidence_bundle: dict[str, Any]
    snapshot: dict[str, Any]
    task_kind: str = "fix"


@dataclass(frozen=True)
class PackagedAgentRunResult:
    status: str
    terminal_state: str | None
    agent_run_id: str | None
    session_id: str | None
    text: str
    files_changed: list[str]
    telemetry: dict[str, Any]
    artifacts: dict[str, Any]
    runtime: dict[str, Any]
    patch_diff: str


PackagedAgentRunner = Callable[[PackagedAgentRunRequest], PackagedAgentRunResult]


class PackagedRepositoryAdvanced:
    def __init__(self, *, runtime_root: Path, agent_runner: PackagedAgentRunner | None) -> None:
        root = runtime_root.expanduser().resolve() / "repository-intelligence" / "packaged"
        self._plan_dir = root / "decision-plans"
        self._patch_dir = root / "patches"
        self._simulation_dir = root / "simulations"
        self._apply_dir = root / "applies"
        self._agent_runner = agent_runner

    def create_repository_decision_plan(
        self,
        *,
        repository_id: str,
        request: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self._agent_runner is None:
            raise PackagedRepositoryAdvancedError(
                "local_repository_agent_runner_required",
                "Packaged local fix requires the bundled Codna agent-core runner.",
                {"repository_id": repository_id, "backend_mode": BACKEND_MODE},
            )
        self._validate_repository(repository_id, snapshot, request)
        issue_text = _issue_text(request)
        repo_root = _repo_root(snapshot)
        run_request = PackagedAgentRunRequest(
            repository_id=repository_id,
            snapshot_id=str(snapshot["snapshot_id"]),
            repo_root=repo_root,
            issue_text=issue_text,
            model=str(request.get("model") or "repository.verified_agentic_v1"),
            signals=dict(request.get("signals") if isinstance(request.get("signals"), dict) else {}),
            evidence_bundle=dict(evidence_bundle),
            snapshot=dict(snapshot),
        )
        result = self._agent_runner(run_request)
        patch_diff = normalize_unified_diff(result.patch_diff).strip()
        if not patch_diff:
            raise PackagedRepositoryAdvancedError(
                "local_repository_agent_no_patch",
                "The local agent completed without producing a patch.",
                {
                    "repository_id": repository_id,
                    "snapshot_id": snapshot["snapshot_id"],
                    "status": result.status,
                    "terminal_state": result.terminal_state,
                    "agent_run_id": result.agent_run_id,
                    "session_id": result.session_id,
                },
            )
        changed_files = _changed_files_from_patch(patch_diff)
        _validate_patch_scope_for_test_driven_fix(
            signals=run_request.signals,
            changed_files=changed_files,
            repository_id=repository_id,
            snapshot_id=str(snapshot["snapshot_id"]),
            agent_run_id=result.agent_run_id,
            session_id=result.session_id,
        )
        created_at = _utc_now()
        patch_id = "patch_" + _sha256_text(
            f"{repository_id}|{snapshot['snapshot_id']}|{patch_diff}"
        )[:24]
        patch_path = self._patch_dir / f"{patch_id}.diff"
        _atomic_write_text(patch_path, patch_diff + "\n")
        plan_id = "plan_" + _sha256_text(
            f"{repository_id}|{snapshot['snapshot_id']}|{patch_id}|{issue_text}"
        )[:24]
        usage = _usage_from_telemetry(result.telemetry)
        token_metrics = _token_reduction_metrics(evidence_bundle)
        plan_payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "snapshot_id": snapshot["snapshot_id"],
            "decision_plan_id": plan_id,
            "created_at": created_at,
            "workspace_evidence_bundle_ref": request.get("workspace_evidence_bundle_ref"),
            "_repo_root_path": str(repo_root),
            "patch_id": patch_id,
            "patch_path": str(patch_path),
            "patch_sha256": _sha256_text(patch_diff),
            "changed_files": changed_files,
            "agent": _agent_summary(result),
            "planner_usage": usage,
            "runtime_model": result.telemetry.get("model") or request.get("model"),
            "decision_plan": {
                "plan_id": plan_id,
                "confidence": _confidence(result, changed_files),
                "repository_analysis": {
                    "root_cause": _summary_text(result.text, issue_text),
                    "impacted_symbols": _suspect_symbols(evidence_bundle),
                    "blast_radius": "localized" if len(changed_files) <= 2 else "multi-file",
                    "generated_patch_ref": f"packaged-local-patch://{patch_id}",
                    "changed_files": changed_files,
                    "token_reduction_metrics": token_metrics,
                },
            },
        }
        _atomic_write_json(self._plan_dir / f"{plan_id}.json", plan_payload)
        public = dict(plan_payload)
        public.pop("_repo_root_path", None)
        public.pop("patch_path", None)
        return public

    def simulate_repository(
        self,
        *,
        repository_id: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan_id = _required_string(request, "decision_plan_id")
        plan = self._load_plan(repository_id, plan_id)
        patch_diff = self._read_patch(plan)
        changed = _changed_files_from_patch(patch_diff)
        changed_lines = _changed_line_count(patch_diff)
        risk = _risk_score(changed, changed_lines)
        action = "apply_patch" if changed and risk < 0.95 else "hold_patch"
        simulation_id = "sim_" + _sha256_text(
            f"{repository_id}|{plan_id}|{plan.get('patch_sha256')}|{risk:.6f}"
        )[:24]
        created_at = _utc_now()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "snapshot_id": plan["snapshot_id"],
            "decision_plan_id": plan_id,
            "created_at": created_at,
            "validated_inputs": {"simulation_id": simulation_id},
            "recommended_action": action,
            "metrics": {
                "probability_of_loss": risk,
                "var_95": -float(changed_lines),
                "changed_file_count": len(changed),
                "changed_line_count": changed_lines,
            },
            "score_breakdown": {
                "apply_gate": {
                    "passed": action == "apply_patch",
                    "policy": "packaged_local_deterministic_patch_gate_v1",
                }
            },
        }
        _atomic_write_json(self._simulation_dir / f"{simulation_id}.json", payload)
        return payload

    def apply_repository(
        self,
        *,
        repository_id: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan_id = _required_string(request, "decision_plan_id")
        simulation_id = _required_string(request, "simulation_id")
        mode = str(request.get("mode") or "patch_only")
        plan = self._load_plan(repository_id, plan_id)
        simulation = self._load_simulation(repository_id, simulation_id)
        if simulation.get("decision_plan_id") != plan_id:
            raise PackagedRepositoryAdvancedError(
                "repository_apply_simulation_mismatch",
                "Simulation does not belong to the requested decision plan.",
                {"repository_id": repository_id, "decision_plan_id": plan_id, "simulation_id": simulation_id},
            )
        patch_diff = self._read_patch(plan)
        if mode == "patch_only":
            return self._patch_only(repository_id, plan, simulation, patch_diff)
        if mode == "remote_pr":
            return self._apply_remote_pr(repository_id, request, plan, simulation, patch_diff)
        if mode != "local_branch":
            raise PackagedRepositoryAdvancedError(
                "packaged_local_apply_mode_unsupported",
                "Packaged local apply supports patch_only, local_branch, and remote_pr.",
                {"repository_id": repository_id, "mode": mode},
            )
        if request.get("write_permission") is not True:
            raise PackagedRepositoryAdvancedError(
                "repository_apply_write_permission_required",
                "local_branch apply requires write_permission=true.",
                {"repository_id": repository_id, "mode": mode},
            )
        repo_root = _repo_root(plan)
        applied = self._apply_local_branch(repo_root, plan_id, patch_diff)
        apply_id = "apply_" + _sha256_text(
            f"{repository_id}|{plan_id}|{simulation_id}|{applied['branch_name']}|{applied['commit_sha']}"
        )[:24]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "decision_plan_id": plan_id,
            "simulation_id": simulation_id,
            "apply_result_id": apply_id,
            "mode": mode,
            "status": "applied",
            "branch_name": applied["branch_name"],
            "commit_sha": applied["commit_sha"],
            "local_checkout_path": str(repo_root),
            "changed_files": plan.get("changed_files") or [],
            "created_at": _utc_now(),
        }
        _atomic_write_json(self._apply_dir / f"{apply_id}.json", payload)
        return payload

    def _apply_remote_pr(
        self,
        repository_id: str,
        request: Mapping[str, Any],
        plan: Mapping[str, Any],
        simulation: Mapping[str, Any],
        patch_diff: str,
    ) -> dict[str, Any]:
        if request.get("write_permission") is not True:
            raise PackagedRepositoryAdvancedError(
                "repository_apply_write_permission_required",
                "remote_pr apply requires write_permission=true.",
                {"repository_id": repository_id, "mode": "remote_pr"},
            )
        repo_root = _repo_root(plan)
        remote = open_remote_pr(
            repo_root=repo_root,
            repository_url=_required_string(request, "repository_url"),
            repository_slug=_optional_string(request, "repository_slug"),
            access_token=_optional_string(request, "access_token"),
            plan_id=str(plan["decision_plan_id"]),
            patch_diff=patch_diff,
            base_branch=_optional_string(request, "base_branch"),
            title=_optional_string(request, "pull_request_title") or "codna: fix repository issue",
            body=_optional_string(request, "pull_request_body") or "Automated fix by codna.",
        )
        apply_id = "apply_" + _sha256_text(
            f"{repository_id}|{plan['decision_plan_id']}|"
            f"{simulation['validated_inputs']['simulation_id']}|{remote['branch_name']}|remote_pr"
        )[:24]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "decision_plan_id": plan["decision_plan_id"],
            "simulation_id": simulation["validated_inputs"]["simulation_id"],
            "apply_result_id": apply_id,
            "mode": "remote_pr",
            "status": "opened_pull_request",
            "branch_name": remote["branch_name"],
            "base_branch": remote["base_branch"],
            "repository_slug": remote["repository_slug"],
            "pull_request_url": remote["pull_request_url"],
            "changed_files": plan.get("changed_files") or [],
            "created_at": _utc_now(),
        }
        _atomic_write_json(self._apply_dir / f"{apply_id}.json", payload)
        return payload

    def _patch_only(
        self,
        repository_id: str,
        plan: Mapping[str, Any],
        simulation: Mapping[str, Any],
        patch_diff: str,
    ) -> dict[str, Any]:
        apply_id = "apply_" + _sha256_text(
            f"{repository_id}|{plan['decision_plan_id']}|{simulation['validated_inputs']['simulation_id']}|patch_only"
        )[:24]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "backend_mode": BACKEND_MODE,
            "repository_id": repository_id,
            "decision_plan_id": plan["decision_plan_id"],
            "simulation_id": simulation["validated_inputs"]["simulation_id"],
            "apply_result_id": apply_id,
            "mode": "patch_only",
            "status": "patch_ready",
            "applied": False,
            "patch": patch_diff,
            "patch_ref": f"packaged-local-patch://{plan['patch_id']}",
            "changed_files": plan.get("changed_files") or [],
            "created_at": _utc_now(),
        }
        _atomic_write_json(self._apply_dir / f"{apply_id}.json", payload)
        return payload

    def _apply_local_branch(self, repo_root: Path, plan_id: str, patch_diff: str) -> dict[str, str]:
        _require_git_repo(repo_root)
        blockers = _apply_blockers(repo_root, patch_diff)
        if blockers:
            raise PackagedRepositoryAdvancedError(
                "repository_apply_dirty_worktree",
                "local_branch apply requires no tracked changes and no untracked files that conflict with the patch.",
                {"repo_root": str(repo_root), **blockers},
            )
        branch = "codna/" + plan_id.removeprefix("plan_")[:16]
        _run_git(repo_root, ["checkout", "-b", branch])
        patch_path = repo_root / ".codna-packaged.patch.diff"
        _atomic_write_text(patch_path, normalize_unified_diff(patch_diff))
        try:
            flags = _applicable_apply_flags(repo_root, patch_path)
            _run_git(repo_root, ["apply", *flags, str(patch_path)])
            _commit_applied_patch(repo_root, plan_id, _changed_files_from_patch(patch_diff))
        finally:
            patch_path.unlink(missing_ok=True)
        commit_sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
        return {"branch_name": branch, "commit_sha": commit_sha}

    def _validate_repository(
        self,
        repository_id: str,
        snapshot: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> None:
        if snapshot.get("repository_id") != repository_id:
            raise PackagedRepositoryAdvancedError(
                "unknown_repository_snapshot",
                "Snapshot does not belong to the requested repository.",
                {"repository_id": repository_id, "snapshot_id": snapshot.get("snapshot_id")},
            )
        requested = request.get("snapshot_id")
        if requested and requested != snapshot.get("snapshot_id"):
            raise PackagedRepositoryAdvancedError(
                "repository_decision_plan_snapshot_mismatch",
                "Decision plan snapshot_id does not match the loaded evidence bundle.",
                {"requested_snapshot_id": requested, "snapshot_id": snapshot.get("snapshot_id")},
            )

    def _load_plan(self, repository_id: str, plan_id: str) -> dict[str, Any]:
        payload = _read_json(self._plan_dir / f"{plan_id}.json", "decision_plan")
        if payload.get("repository_id") != repository_id:
            raise PackagedRepositoryAdvancedError(
                "unknown_repository_decision_plan",
                "Decision plan is not registered for this repository.",
                {"repository_id": repository_id, "decision_plan_id": plan_id},
            )
        return payload

    def _load_simulation(self, repository_id: str, simulation_id: str) -> dict[str, Any]:
        payload = _read_json(self._simulation_dir / f"{simulation_id}.json", "simulation")
        if payload.get("repository_id") != repository_id:
            raise PackagedRepositoryAdvancedError(
                "unknown_repository_simulation",
                "Simulation is not registered for this repository.",
                {"repository_id": repository_id, "simulation_id": simulation_id},
            )
        return payload

    @staticmethod
    def _read_patch(plan: Mapping[str, Any]) -> str:
        patch_path = Path(str(plan.get("patch_path") or ""))
        try:
            return patch_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PackagedRepositoryAdvancedError(
                "repository_patch_artifact_missing",
                "Decision plan patch artifact is missing or unreadable.",
                {"decision_plan_id": plan.get("decision_plan_id"), "patch_path": str(patch_path)},
            ) from exc


def _issue_text(request: Mapping[str, Any]) -> str:
    signals = request.get("signals") if isinstance(request.get("signals"), dict) else {}
    issue = signals.get("issue_text")
    if isinstance(issue, str) and issue.strip():
        return issue.strip()
    raise PackagedRepositoryAdvancedError(
        "invalid_repository_request",
        "Decision planning requires signals.issue_text.",
        {"field": "signals.issue_text"},
    )


def _repo_root(payload: Mapping[str, Any]) -> Path:
    raw = payload.get("_repo_root_path") or payload.get("repo_root_path")
    if not isinstance(raw, str) or not raw:
        raise PackagedRepositoryAdvancedError(
            "repository_root_missing",
            "Packaged local fix requires a stored local repository path.",
            {},
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise PackagedRepositoryAdvancedError(
            "repository_root_missing",
            "Stored local repository path does not exist.",
            {"repo_root": str(path)},
        )
    return path


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PackagedRepositoryAdvancedError(
            "invalid_repository_request",
            f"Repository request requires a non-empty {key}.",
            {"field": key},
        )
    return value.strip()


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _token_reduction_metrics(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_repo_token_estimate": int(bundle.get("raw_repo_token_estimate") or 0),
        "evidence_bundle_token_count": int(bundle.get("evidence_bundle_token_count") or 0),
        "reduction_ratio": float(bundle.get("reduction_ratio") or 0.0),
    }


def _suspect_symbols(bundle: Mapping[str, Any]) -> list[str]:
    symbols = bundle.get("suspect_symbols")
    if isinstance(symbols, list):
        return [item for item in symbols if isinstance(item, str) and item]
    output: list[str] = []
    for item in bundle.get("evidence_items") or []:
        if isinstance(item, dict) and isinstance(item.get("symbol_name"), str):
            output.append(item["symbol_name"])
    return list(dict.fromkeys(output))


def _usage_from_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": int(telemetry.get("tokens_in_uncached") or 0),
        "output_tokens": int(telemetry.get("tokens_out") or 0),
        "cache_read_tokens": int(telemetry.get("cache_read_tokens") or 0),
        "cache_write_tokens": int(telemetry.get("cache_write_tokens") or 0),
        "cost_usd": telemetry.get("total_cost") if isinstance(telemetry.get("total_cost"), (int, float)) else None,
    }


def _agent_summary(result: PackagedAgentRunResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "terminal_state": result.terminal_state,
        "agent_run_id": result.agent_run_id,
        "session_id": result.session_id,
        "files_changed": result.files_changed,
        "artifacts": result.artifacts,
        "runtime": result.runtime,
    }


def _confidence(result: PackagedAgentRunResult, changed_files: list[str]) -> float:
    """A REAL, per-run PRE-verification confidence derived from the agent-run signals — not a fixed
    number. It reflects how the run actually went (a clean terminal success vs hitting an iteration or
    mistake limit) tempered by blast radius (a one-file fix is more trustworthy than a sprawling one).
    This is an ESTIMATE; the actual proof is the test-loop / engine risk gate, so it is kept coarse and
    capped below certainty rather than presented as a precise probability."""
    if not changed_files or not (result.patch_diff or "").strip():
        return 0.0  # no change produced → no confidence to report
    success = {"succeeded", "success", "completed"}
    ts = (result.terminal_state or "").strip().lower()
    st = (result.status or "").strip().lower()
    if ts in success or st in success:
        base = 0.70
    elif ts in {"max_iterations", "mistake_limit"} or st in {"max_iterations", "mistake_limit"}:
        base = 0.40  # ran out of room / kept erroring — less decisive
    else:
        base = 0.25  # aborted / error / unknown terminal state
    n = len(changed_files)
    if n == 1:
        base += 0.15
    elif n == 2:
        base += 0.05
    elif n >= 5:
        base -= 0.15
    return round(max(0.0, min(0.9, base)), 2)


def _summary_text(agent_text: str, issue_text: str) -> str:
    for line in agent_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:500]
    return f"Patch generated for reported issue: {issue_text[:420]}"


def _applicable_apply_flags(repo_root: Path, patch_path: Path) -> list[str]:
    first_error = ""
    for flags in GIT_APPLY_FLAG_SETS:
        check = _run_git(repo_root, ["apply", "--check", *flags, str(patch_path)], check=False)
        if check.returncode == 0:
            return list(flags)
        first_error = first_error or check.stderr.strip()
    raise PackagedRepositoryAdvancedError(
        "repository_patch_rejected",
        "The generated patch does not apply to the repository: "
        + (first_error.splitlines()[0] if first_error else "git apply --check failed"),
        {"repo_root": str(repo_root), "stderr": first_error[-2000:]},
    )


def _changed_files_from_patch(patch_diff: str) -> list[str]:
    files: list[str] = []
    for line in patch_diff.splitlines():
        if not line.startswith("+++ b/"):
            continue
        path = line.removeprefix("+++ b/").strip()
        if path and path != "/dev/null" and path not in files:
            files.append(path)
    return files


def _validate_patch_scope_for_test_driven_fix(
    *,
    signals: Mapping[str, Any],
    changed_files: list[str],
    repository_id: str,
    snapshot_id: str,
    agent_run_id: str | None,
    session_id: str | None,
) -> None:
    if not _is_test_driven_fix(signals) or not changed_files:
        return
    if _explicit_test_edit_request(signals):
        return
    if any(not _is_test_path(path) for path in changed_files):
        return
    raise PackagedRepositoryAdvancedError(
        "local_repository_agent_test_only_patch",
        "Codna refused a test-only patch for a failing-test fix. Fix source code, not the test, "
        "or pass an explicit --issue stating that the test expectation itself is wrong.",
        {
            "repository_id": repository_id,
            "snapshot_id": snapshot_id,
            "changed_files": changed_files,
            "agent_run_id": agent_run_id,
            "session_id": session_id,
        },
    )


def _is_test_driven_fix(signals: Mapping[str, Any]) -> bool:
    if _string_list(signals.get("failing_tests")):
        return True
    issue = signals.get("issue_text")
    return isinstance(issue, str) and bool(
        re.search(r"\b(failing test|tests? are failing|unittest|pytest|junit|assertionerror)\b", issue, re.I)
    )


def _explicit_test_edit_request(signals: Mapping[str, Any]) -> bool:
    issue = signals.get("issue_text")
    if not isinstance(issue, str):
        return False
    return bool(
        re.search(r"\b(update|change|rewrite|correct)\s+(the\s+)?(test|tests|assertion|expectation)s?\b", issue, re.I)
        or re.search(r"\b(test|tests|assertion|expectation)s?\s+(is|are)\s+(wrong|incorrect|outdated)\b", issue, re.I)
    )


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = parts[-1] if parts else normalized
    return (
        any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", "_spec.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _changed_line_count(patch_diff: str) -> int:
    count = 0
    for line in patch_diff.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            count += 1
    return count


def _risk_score(changed_files: list[str], changed_lines: int) -> float:
    if not changed_files:
        return 1.0
    return min(0.94, round(0.04 + len(changed_files) * 0.04 + changed_lines * 0.002, 6))


def _read_json(path: Path, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackagedRepositoryAdvancedError(
            f"unknown_repository_{kind}",
            f"Repository {kind} artifact is not registered.",
            {"path": str(path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise PackagedRepositoryAdvancedError(
            f"corrupt_repository_{kind}",
            f"Repository {kind} artifact is not valid JSON.",
            {"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise PackagedRepositoryAdvancedError(
            f"corrupt_repository_{kind}",
            f"Repository {kind} artifact must be a JSON object.",
            {"path": str(path)},
        )
    return payload


def _require_git_repo(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise PackagedRepositoryAdvancedError(
            "repository_apply_requires_git",
            "local_branch apply requires a git repository. Use patch_only for non-git directories.",
            {"repo_root": str(repo_root)},
        )


def _apply_blockers(repo_root: Path, patch_diff: str) -> dict[str, Any]:
    tracked = _git_status(repo_root, include_untracked=False)
    if tracked:
        return {"tracked_status": tracked}
    patch_targets = set(_changed_files_from_patch(patch_diff))
    if not patch_targets:
        return {"patch_targets": []}
    conflicts = _conflicting_untracked_files(repo_root, patch_targets)
    if conflicts:
        return {"conflicting_untracked_files": conflicts}
    return {}


def _commit_applied_patch(repo_root: Path, plan_id: str, changed_files: list[str]) -> None:
    if not changed_files:
        raise PackagedRepositoryAdvancedError(
            "repository_apply_empty_patch",
            "local_branch apply requires a patch with at least one changed file.",
            {"repo_root": str(repo_root), "decision_plan_id": plan_id},
        )
    _run_git(repo_root, ["add", "--", *changed_files])
    staged = _run_git(repo_root, ["diff", "--cached", "--name-only", "-z"])
    staged_files = [entry for entry in staged.stdout.split("\0") if entry]
    if not staged_files:
        raise PackagedRepositoryAdvancedError(
            "repository_apply_empty_patch",
            "The packaged local patch applied without staged changes.",
            {"repo_root": str(repo_root), "decision_plan_id": plan_id, "changed_files": changed_files},
        )
    _run_git(
        repo_root,
        [
            *_git_identity_args(),
            "commit",
            "--no-gpg-sign",
            "-m",
            f"codna: apply {plan_id}",
        ],
    )


def _git_status(repo_root: Path, *, include_untracked: bool) -> list[str]:
    untracked = "all" if include_untracked else "no"
    result = _run_git(repo_root, ["status", "--porcelain=v1", "-z", f"--untracked-files={untracked}"])
    return [entry for entry in result.stdout.split("\0") if entry]


def _conflicting_untracked_files(repo_root: Path, patch_targets: set[str]) -> list[str]:
    result = _run_git(repo_root, ["ls-files", "--others", "--exclude-standard", "-z"])
    conflicts: list[str] = []
    for raw in result.stdout.split("\0"):
        path = raw.strip()
        if not path:
            continue
        if path in patch_targets:
            conflicts.append(path)
    return conflicts


def _run_git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if check and result.returncode != 0:
        raise PackagedRepositoryAdvancedError(
            "git_command_failed",
            "Git command failed while applying the packaged local patch.",
            {"repo_root": str(repo_root), "args": args, "stderr": result.stderr[-2000:]},
        )
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
