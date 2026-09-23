from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import httpx

from .agent_core_runtime import ensure_agent_core_running
from .packaged_repository_advanced import (
    PackagedAgentRunRequest,
    PackagedAgentRunResult,
    PackagedRepositoryAdvancedError,
)
from .review_budget import review_turn_budget_ms, size_budget_ms
from .runtime.config import ConfigValue, RuntimeConfig

_VERIFIED_AGENTIC_MODEL = "repository.verified_agentic_v1"
_WORKSPACE_SKIP_DIRS = {
    ".codna",
    ".codna-memory",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_PATCH_CAPTURE_SKIP_DIRS = _WORKSPACE_SKIP_DIRS | {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
}
_PATCH_CAPTURE_SKIP_FILES = {
    ".coverage",
    ".DS_Store",
    "coverage.xml",
}
_PATCH_CAPTURE_SKIP_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True)
class _Workspace:
    path: Path
    kind: str


class SidecarPackagedAgentRunner:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        keys: Mapping[str, ConfigValue] | None = None,
    ) -> None:
        self._config = config
        self._keys = keys

    def run(self, request: PackagedAgentRunRequest) -> PackagedAgentRunResult:
        workspace = _prepare_workspace(
            repo_root=request.repo_root,
            runtime_root=self._config.paths.root,
            repository_id=request.repository_id,
            snapshot_id=request.snapshot_id,
        )
        try:
            final = self._run_sidecar(request, workspace.path)
            patch_diff = _capture_patch(workspace.path)
            return PackagedAgentRunResult(
                status=str(final.get("status") or "failed"),
                terminal_state=_optional_string(final.get("terminal_state")),
                agent_run_id=_optional_string(final.get("agent_run_id")),
                session_id=_optional_string(final.get("session_id")),
                text=str(final.get("text") or ""),
                files_changed=_string_list(final.get("files_changed")),
                telemetry=dict(final.get("telemetry") if isinstance(final.get("telemetry"), dict) else {}),
                artifacts=dict(final.get("artifacts") if isinstance(final.get("artifacts"), dict) else {}),
                runtime=dict(final.get("runtime") if isinstance(final.get("runtime"), dict) else {}),
                patch_diff=patch_diff,
            )
        finally:
            _cleanup_workspace(request.repo_root, workspace)

    def _run_sidecar(self, request: PackagedAgentRunRequest, workspace: Path) -> dict[str, Any]:
        endpoint = ensure_agent_core_running(keys=self._keys)
        payload = _sidecar_payload(request, workspace)
        timeout_s = max(10.0, float(payload["limits"]["timeoutMs"]) / 1000.0 + 30.0)
        try:
            response = httpx.post(
                f"{endpoint.url.rstrip('/')}/run",
                json=payload,
                timeout=timeout_s,
            )
        except httpx.HTTPError as exc:
            raise PackagedRepositoryAdvancedError(
                "agent_core_transport_failed",
                "Codna could not reach the local agent-core sidecar.",
                {"sidecar_url": endpoint.url, "error": str(exc)},
            ) from exc
        if response.status_code != 200:
            raise PackagedRepositoryAdvancedError(
                "agent_core_run_failed",
                "Local agent-core rejected the packaged fix run.",
                {
                    "sidecar_url": endpoint.url,
                    "status_code": response.status_code,
                    "body": response.text[:2000],
                },
            )
        return _parse_final_frame(
            response.text,
            sidecar_url=endpoint.url,
            task_kind=_validated_task_kind(request.task_kind),
        )


def _sidecar_payload(request: PackagedAgentRunRequest, workspace: Path) -> dict[str, Any]:
    task_kind = _validated_task_kind(request.task_kind)
    timeout_ms = _timeout_ms_for_request(request)
    max_turns = _max_turns_for_request(request)
    requested_provider, requested_model = _provider_model_from_request(request.model)
    provider = (
        os.environ.get("ALGENTA_AGENT_PROVIDER")
        or os.environ.get("CODNA_AGENT_PROVIDER")
        or requested_provider
    )
    model = (
        os.environ.get("ALGENTA_AGENT_MODEL")
        or os.environ.get("CODNA_AGENT_MODEL")
        or requested_model
    )
    payload: dict[str, Any] = {
        "working_dir": str(workspace),
        "channel": "cli",
        "repository_scope": request.repository_id,
        "task_kind": task_kind,
        "tool_profile": _tool_profile_for_task(task_kind),
        "limits": {"maxIterations": max_turns, "timeoutMs": timeout_ms},
        "task_spec": {
            "task_kind": task_kind,
            "engine": {"enabled": False},
        },
        "injected_context": {},
        "allowed_write_paths": [] if task_kind == "review" else _allowed_write_paths(request),
    }
    if task_kind == "review":
        # A review is delivered VERBATIM as the user turn (`prompt`), never as `task_spec.issue_text`.
        # The sidecar's issue_text path is the FIX path: buildPrompt (server/utils.ts) wraps it in
        # "Fix this repository issue: ... Make the minimal correct change with your file-editing
        # tools, then stop.", mines TARGET FILES out of it, and buildSystemPrompt (adapter.ts) then
        # tells the model to "edit that file directly" -- on top of the review contract's "Do not
        # modify files ... FINAL message MUST be a single JSON object". Given both, the model
        # announced the edit it could not make (writes are filtered out of its tools) and ended the
        # turn in prose, and `codna review` failed with "did not return parseable findings JSON"
        # (thyn-ai/algenta-sdk#16 2026-08-27, thyn-ai/test-codna-app-e2e#15/#16 2026-09-19).
        # No issue_text also means no Telys enrichment and no fix-worded project guidance: the
        # review prompt already carries the diff and the project guidance (review_findings.py).
        payload["prompt"] = request.issue_text
        payload["approval_profile"] = "repository_review"
        payload["allowed_write_scope"] = []
    else:
        payload["task_spec"]["issue_text"] = _issue_text_with_guidance(request)
        payload["injected_context"] = _injected_context(request)
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    return payload


def _validated_task_kind(value: str) -> str:
    task_kind = (value or "fix").strip()
    if task_kind not in {"fix", "triage", "review"}:
        raise PackagedRepositoryAdvancedError(
            "agent_task_kind_invalid",
            "Unsupported local agent task kind.",
            {"task_kind": value, "allowed": ["fix", "triage", "review"]},
        )
    return task_kind


def _tool_profile_for_task(task_kind: str) -> str | None:
    if task_kind == "review":
        return None
    return os.environ.get(
        "ALGENTA_REPOSITORY_AGENT_TOOL_PROFILE",
        "no_exploration_local_validation",
    )


def _timeout_ms_for_request(request: PackagedAgentRunRequest) -> int:
    """The per-turn budget the sidecar enforces (``limits.timeoutMs``).

    A fix gets the size-derived ``_dynamic_timeout_ms``. A review used to get one fixed number
    instead -- ``review_timeout_s`` (240 s, cline_agent.run_cline_review's default) -- which is
    what failed thyn-ai/algenta-sdk#71 (+1008/-15, 5 files) with ``turn exceeded timeout budget of
    240000ms`` while the fix path next to it would have granted minutes more. A review now gets
    ``review_budget.review_turn_budget_ms``: the same size formula fed with the diff's own lines,
    files and prompt tokens, raised by the admission gate's Monte-Carlo forecast over observed review
    turns, bounded by [240 s, 20 min], ``ALGENTA_AGENT_TURN_TIMEOUT_MS`` winning as for fixes. An
    explicit ``review_timeout_s`` signal remains a caller's pin (tests, an embedder), never a default.
    """
    if request.task_kind != "review":
        return _dynamic_timeout_ms(request.snapshot)
    value = request.signals.get("review_timeout_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _env_int_value(str(int(value * 1000)), minimum=30_000, maximum=3_600_000)
    return review_turn_budget_ms(request.signals, request.snapshot)


def _max_turns_for_request(request: PackagedAgentRunRequest) -> int:
    value = request.signals.get("review_max_iterations") if request.task_kind == "review" else None
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, min(50, value))
    return _env_int("ALGENTA_AGENT_MAX_TURNS", default=4, minimum=1, maximum=50)


def _provider_model_from_request(model: str) -> tuple[str | None, str | None]:
    requested = (model or "").strip()
    if not requested or requested == _VERIFIED_AGENTIC_MODEL:
        return None, None
    if "/" not in requested:
        return None, requested
    provider, model_id = requested.split("/", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        return None, requested
    return provider, model_id


def _project_guidance(request: PackagedAgentRunRequest) -> str | None:
    """Project rules (AGENTS.md / .codna/rules) for the executing Cline agent to honor.

    Prefer the guidance already surfaced onto the signal by the CLI; fall back to reading the repo
    root directly. Codna is the integrator — it makes the Cline SDK agent honor project conventions
    here, without any Algenta (repo-intelligence) change.
    """
    signalled = (request.signals or {}).get("project_guidance") if getattr(request, "signals", None) else None
    if isinstance(signalled, str) and signalled.strip():
        return signalled.strip()
    from .project_rules import read_project_guidance
    return read_project_guidance(str(getattr(request, "repo_root", "")) or None)


def _issue_text_with_guidance(request: PackagedAgentRunRequest) -> str:
    """The Cline agent's task, with project guidance appended so the agent honors it while editing."""
    allowed = _allowed_write_paths(request)
    test_guidance = ""
    if _is_test_driven_fix(request.signals) and not _explicit_test_edit_request(request.signals):
        test_guidance = (
            "\n\n[CODNA FAILING-TEST FIX CONTRACT]\n"
            "The failing tests are verification signals, not the target fix. "
            "Do not modify test/spec files unless the user explicitly states that the test expectation is wrong. "
            "Fix the production/source code that makes the tests fail."
        )
    path_guidance = ""
    if allowed:
        path_guidance = (
            "\n\n[CODNA WRITE SCOPE]\n"
            "Edit only these repository-relative files unless the run escalates to exploration fallback:\n"
            + "\n".join(f"- {path}" for path in allowed)
            + "\nFailing test ids such as package.module::test_name are test identifiers, not file paths."
            + "\nWhen fixing incorrect code, replace the incorrect logic; do not leave duplicate unreachable code behind."
        )
    guidance = _project_guidance(request)
    if not guidance:
        return request.issue_text + test_guidance + path_guidance
    return (f"{request.issue_text}\n\n"
            "[PROJECT GUIDANCE — honor these project rules while making changes]\n"
            f"{guidance}{test_guidance}{path_guidance}")


def _injected_context(request: PackagedAgentRunRequest) -> dict[str, Any]:
    evidence_items = [
        item for item in request.evidence_bundle.get("evidence_items", []) if isinstance(item, dict)
    ]
    context_bundle = []
    for item in evidence_items[:8]:
        file_path = item.get("file_path")
        snippet = item.get("snippet")
        if isinstance(file_path, str) and isinstance(snippet, str):
            context_bundle.append(
                {
                    "file_path": file_path,
                    "symbol_name": item.get("symbol_name"),
                    "numbered_source": snippet,
                }
            )
    ctx: dict[str, Any] = {
        "localization_tier": "packaged_local_triage",
        "suspect_files": _string_list(request.evidence_bundle.get("suspect_files")),
        "suspect_symbols": _string_list(request.evidence_bundle.get("suspect_symbols")),
        "evidence_items": evidence_items,
        "context_bundle": context_bundle,
    }
    guidance = _project_guidance(request)
    if guidance:
        ctx["project_guidance"] = guidance  # first-class context for the executing agent
    return ctx


def _allowed_write_paths(request: PackagedAgentRunRequest) -> list[str]:
    """Localized fast path may edit only repo-evidence files; Tier-2 fallback can relax this."""
    output: list[str] = []
    for value in _string_list(request.evidence_bundle.get("suspect_files")):
        _append_path(output, value)
    for item in request.evidence_bundle.get("evidence_items", []):
        if isinstance(item, dict):
            _append_path(output, item.get("file_path"))
    if _is_test_driven_fix(request.signals) and not _explicit_test_edit_request(request.signals):
        non_test_paths = [path for path in output if not _is_test_path(path)]
        return non_test_paths
    return output


def _append_path(output: list[str], value: Any) -> None:
    if not isinstance(value, str):
        return
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return
    normalized = path.as_posix()
    if normalized and normalized != "." and normalized not in output:
        output.append(normalized)


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


def _dynamic_timeout_ms(snapshot: Mapping[str, Any]) -> int:
    """A fix turn's budget: 300 s base + 2 ms per estimated repo token (cap 900 s) + 1 s per snapshot
    file (cap 600 s), capped at 1800 s -- ``review_budget.size_budget_ms`` is the shared formula
    (the review path feeds it the diff's lines/files/tokens with its own coefficients)."""
    raw_tokens = _int_value(snapshot.get("raw_repo_token_estimate"))
    files = _int_value(snapshot.get("snapshot_file_count") or snapshot.get("file_count"))
    configured = os.environ.get("ALGENTA_AGENT_TURN_TIMEOUT_MS") or os.environ.get("FIX_TIMEOUT_MS")
    if configured:
        return _env_int_value(configured, minimum=30_000, maximum=3_600_000)
    return size_budget_ms(
        base_ms=300_000, cap_ms=1_800_000,
        tokens=raw_tokens, token_ms=2, token_cap_ms=900_000,
        files=files, file_ms=1_000, file_cap_ms=600_000,
    )


def _prepare_workspace(
    *,
    repo_root: Path,
    runtime_root: Path,
    repository_id: str,
    snapshot_id: str,
) -> _Workspace:
    workspace = runtime_root / "repository-intelligence" / "packaged" / "workspaces" / (
        f"{repository_id}_{snapshot_id}"
    )
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if _is_clean_git_repo(repo_root) and _git_worktree_add(repo_root, workspace):
        return _Workspace(path=workspace, kind="git_worktree")
    shutil.copytree(repo_root, workspace, ignore=_copy_ignore_for(runtime_root))
    _init_baseline_git_repo(workspace)
    return _Workspace(path=workspace, kind="copy")


def _cleanup_workspace(repo_root: Path, workspace: _Workspace) -> None:
    if workspace.kind == "git_worktree":
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(workspace.path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    shutil.rmtree(workspace.path, ignore_errors=True)


def _capture_patch(workspace: Path) -> str:
    _add_untracked_patch_candidates(workspace)
    result = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", "--no-ext-diff", "--"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise PackagedRepositoryAdvancedError(
            "agent_patch_capture_failed",
            "Codna could not capture the local agent patch.",
            {"workspace": str(workspace), "stderr": result.stderr[-2000:]},
        )
    return result.stdout


def _add_untracked_patch_candidates(workspace: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise PackagedRepositoryAdvancedError(
            "agent_patch_capture_failed",
            "Codna could not inspect untracked local agent patch files.",
            {"workspace": str(workspace), "stderr": result.stderr.decode("utf-8", "replace")[-2000:]},
        )
    candidates = [
        path
        for path in _decode_git_z_paths(result.stdout)
        if _should_capture_untracked_patch_file(path)
    ]
    for start in range(0, len(candidates), 100):
        _git_add_intent_to_add(workspace, candidates[start : start + 100])


def _decode_git_z_paths(raw: bytes) -> list[str]:
    return [item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item]


def _should_capture_untracked_patch_file(path: str) -> bool:
    parsed = PurePosixPath(path)
    if any(part in _PATCH_CAPTURE_SKIP_DIRS for part in parsed.parts):
        return False
    name = parsed.name
    if name in _PATCH_CAPTURE_SKIP_FILES:
        return False
    return not name.endswith(_PATCH_CAPTURE_SKIP_SUFFIXES)


def _git_add_intent_to_add(workspace: Path, paths: list[str]) -> None:
    if not paths:
        return
    result = subprocess.run(
        ["git", "-C", str(workspace), "add", "-N", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise PackagedRepositoryAdvancedError(
            "agent_patch_capture_failed",
            "Codna could not stage local agent patch candidates for capture.",
            {"workspace": str(workspace), "stderr": result.stderr[-2000:]},
        )


def _init_baseline_git_repo(workspace: Path) -> None:
    commands = [
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=Codna", "-c", "user.email=codna@example.invalid", "commit", "-q", "-m", "codna baseline"],
    ]
    for args in commands:
        result = subprocess.run(args, cwd=workspace, capture_output=True, text=True, check=False, timeout=120)
        if result.returncode != 0:
            raise PackagedRepositoryAdvancedError(
                "agent_workspace_baseline_failed",
                "Codna could not create an isolated baseline workspace for the local agent.",
                {"workspace": str(workspace), "args": args, "stderr": result.stderr[-2000:]},
            )


def _is_clean_git_repo(repo_root: Path) -> bool:
    inside = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return False
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1", "-z", "--untracked-files=no"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    return status.returncode == 0 and not status.stdout


def _git_worktree_add(repo_root: Path, workspace: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", "--force", str(workspace), "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode == 0:
        return True
    shutil.rmtree(workspace, ignore_errors=True)
    return False


def _copy_ignore_for(runtime_root: Path):
    runtime_root = runtime_root.expanduser().resolve()

    def _copy_ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory).resolve()
        ignored = {name for name in names if name in _WORKSPACE_SKIP_DIRS}
        for name in names:
            child = (directory_path / name).resolve()
            if child == runtime_root or _is_relative_to(child, runtime_root):
                ignored.add(name)
        return ignored

    return _copy_ignore


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _terminal_states_meaning_success(task_kind: str) -> set[str]:
    """Which terminal states count as a completed run, for THIS kind of run.

    A review is dispatched with NO write scope -- `_sidecar_payload` sets
    `allowed_write_paths` and `allowed_write_scope` to `[]` for task_kind == "review", on
    purpose, because a review analyses and must not modify the repository. agent-core emits
    `apply_blocked` in exactly that situation (artifacts.ts: allowedWriteScope.length === 0
    and some tool call succeeded).

    So a review that works PERFECTLY always terminates in `apply_blocked`. It is the success
    signal for a read-only run, not a failure. Demanding "succeeded" made `codna review`
    impossible to pass -- not flaky, categorically impossible -- which is why it failed on
    every pull request across the org with an identical message.

    codna already encodes this judgement elsewhere: cline_parity_report.py:423 accepts
    {"engine_evidence_missing", "succeeded", "apply_blocked"} as a proved engine gate. This
    brings the runner into line with it.

    For a FIX, `apply_blocked` stays a genuine failure: a fix that was not allowed to write
    has not fixed anything.
    """
    if task_kind == "review":
        return {"succeeded", "apply_blocked"}
    return {"succeeded"}


_MISSING_PROVIDER_KEY_RE = re.compile(r"no API key in env for '([^']+)' \(tried: ([^)]+)\)")


def _run_failed_message(error: object, *, task_kind: str) -> str:
    """Message for `agent_core_run_failed` — actionable when the cause is known.

    The common fresh-machine failure is a missing BYOK provider key: agent-core's provider
    adapter aborts the run and the final frame carries
    "no API key in env for '<provider>' (tried: <ENV_NAMES>)". Name the fix instead of hiding
    it behind the generic line. The code and details payload are unchanged either way, and any
    other failure keeps the original message verbatim.
    """
    if isinstance(error, str):
        missing_key = _MISSING_PROVIDER_KEY_RE.search(error)
        if missing_key:
            provider, env_names = missing_key.group(1), missing_key.group(2)
            return (
                f"codna {task_kind} needs a provider key — set {env_names} (or store one with "
                f"`codna key set {provider}`, or configure another provider). The local engine "
                "itself needs no login."
            )
    return "Local agent-core did not complete the packaged fix run successfully."


def _parse_final_frame(body: str, *, sidecar_url: str, task_kind: str = "fix") -> dict[str, Any]:
    final: dict[str, Any] | None = None
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackagedRepositoryAdvancedError(
                "agent_core_invalid_ndjson",
                "Local agent-core returned invalid NDJSON.",
                {"sidecar_url": sidecar_url, "line": line[:500]},
            ) from exc
        if isinstance(frame, dict) and frame.get("type") == "final":
            final = frame
    if final is None:
        raise PackagedRepositoryAdvancedError(
            "agent_core_final_frame_missing",
            "Local agent-core did not return a final run frame.",
            {"sidecar_url": sidecar_url},
        )
    terminal_state = final.get("terminal_state")
    status = final.get("status")
    accepted = _terminal_states_meaning_success(task_kind)
    if terminal_state not in accepted and status not in {"succeeded", "success", "completed"}:
        raise PackagedRepositoryAdvancedError(
            "agent_core_run_failed",
            _run_failed_message(final.get("error"), task_kind=task_kind),
            {
                "sidecar_url": sidecar_url,
                "status": status,
                "terminal_state": terminal_state,
                "task_kind": task_kind,
                "accepted_terminal_states": sorted(accepted),
                "error": final.get("error"),
            },
        )
    runtime = final.get("runtime")
    if isinstance(runtime, dict) and runtime.get("is_stub") is True:
        raise PackagedRepositoryAdvancedError(
            "agent_core_stub_runtime_rejected",
            "Codna refuses to treat a stub agent-core runtime as a real fix.",
            {"sidecar_url": sidecar_url, "runtime": runtime},
        )
    return final


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return _env_int_value(raw, minimum=minimum, maximum=maximum)


def _env_int_value(raw: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except ValueError:
        return minimum
    return max(minimum, min(maximum, value))
