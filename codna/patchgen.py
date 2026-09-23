"""Engine-backed patch generator — the remediation seam the worker invokes.

`LocalWorker.remediate` calls a `patch_generator(finding, cwd) -> diff`. The real generator is
codna's existing decision-plan agent: it turns a security finding into a remediation obligation
("fix this source→sink without changing behavior, don't touch tests/CI/suppressions"), asks the
engine to produce a fix, and returns the unified diff for the worker to apply in its sandboxed
worktree.

Transport is the same injected `post(path, body) -> dict` callable the `EngineAdapter` uses, so
this is testable offline with a stub. The generated diff is then subject to the worker's G4
anti-evasion check before it is ever applied.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Callable, Mapping

from .findings import NormalizedFinding
from .runtime.config import ConfigValue

PostFn = Callable[[str, dict], dict]


class PatchGenError(Exception):
    pass


def finding_issue(finding: NormalizedFinding) -> str:
    """Render a finding as a remediation obligation for the agent."""
    loc = finding.primary_location
    where = f"{loc.uri}:{loc.start_line}" if loc and loc.start_line else (loc.uri if loc else "(unknown location)")
    flow = ""
    if finding.code_flows:
        seq = finding.code_flows[0]
        if seq:
            src, sink = seq[0], seq[-1]
            flow = f" Dataflow: {src.uri}:{src.start_line} -> {sink.uri}:{sink.start_line}."
    return (
        f"Security finding {finding.rule_id} ({finding.severity.value}, kind={finding.finding_kind.value}) "
        f"at {where}: {finding.message}.{flow} "
        "Remediate the vulnerability without changing behavior. "
        "Do NOT modify tests, CI, scanner configuration, or suppressions."
    )


class EnginePatchGenerator:
    def __init__(
        self,
        post: PostFn,
        *,
        repository_id: str,
        snapshot_id: str,
        model: str = "repository.verified_agentic_v1",
    ):
        self._post = post
        self.repository_id = repository_id
        self.snapshot_id = snapshot_id
        self.model = model

    def __call__(self, finding: NormalizedFinding, cwd: str) -> str:
        resp = self._post(
            f"/v1/repositories/{self.repository_id}/decision-plan",
            {
                "snapshot_id": self.snapshot_id,
                "model": self.model,
                "signals": {
                    "issue_text": finding_issue(finding),
                    "security_finding": {
                        "canonical_id": finding.canonical_id,
                        "rule_id": finding.rule_id,
                        "finding_kind": finding.finding_kind.value,
                    },
                },
            },
        )
        diff = resp.get("patch_diff") or (resp.get("decision_plan") or {}).get("patch_diff")
        if not diff or not str(diff).strip():
            raise PatchGenError(f"engine returned no patch for {finding.rule_id} ({finding.canonical_id})")
        return diff


class ClinePatchGenerator:
    """`patch_generator(finding, cwd) -> diff` backed by the agentic Cline SDK (agent-core).

    This is the self-hosted remediation seam: instead of asking a remote engine for a patch, it
    drives codna's bundled sidecar agent against an isolated workspace and returns the captured
    unified diff for the caller to subject to the G4 anti-evasion check and then apply.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_iterations: int = 20,
        timeout_s: int = 360,
        keys: Mapping[str, ConfigValue] | None = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s
        self.keys = keys

    def __call__(self, finding: NormalizedFinding, cwd: str) -> str:
        from .packaged_agent_runner import SidecarPackagedAgentRunner
        from .packaged_repository_advanced import PackagedAgentRunRequest
        from .runtime.config import resolve_runtime_config

        prompt = (
            finding_issue(finding)
            + " Edit only the vulnerable source file(s); use REPOSITORY-RELATIVE paths."
        )
        snapshot_id = _secure_snapshot_id(cwd, finding)
        request = PackagedAgentRunRequest(
            repository_id=_secure_repository_id(cwd),
            snapshot_id=snapshot_id,
            repo_root=Path(cwd).resolve(strict=True),
            issue_text=prompt,
            model=_packaged_model(self.model),
            signals={
                "issue_text": prompt,
                "security_finding": {
                    "canonical_id": finding.canonical_id,
                    "rule_id": finding.rule_id,
                    "finding_kind": finding.finding_kind.value,
                },
            },
            evidence_bundle=_evidence_bundle(cwd, finding),
            snapshot={
                "snapshot_id": snapshot_id,
                "snapshot_file_count": _count_repo_files(cwd),
                "raw_repo_token_estimate": _estimate_repo_tokens(cwd),
            },
        )
        runner = SidecarPackagedAgentRunner(config=resolve_runtime_config(keys=self.keys), keys=self.keys)
        with _scoped_agent_limits(self.max_iterations, self.timeout_s):
            diff = runner.run(request).patch_diff
        if not diff.strip():
            raise PatchGenError(f"the agent produced no changes for {finding.rule_id} ({finding.canonical_id})")
        return diff


def _packaged_model(model: str | None) -> str:
    requested = (model or "").strip()
    if not requested:
        return "repository.verified_agentic_v1"
    if requested.startswith("repository.") or "/" in requested:
        return requested
    return f"anthropic/{requested}"


@contextlib.contextmanager
def _scoped_agent_limits(max_iterations: int, timeout_s: int):
    updates = {
        "ALGENTA_AGENT_MAX_TURNS": str(max_iterations),
        "ALGENTA_AGENT_TURN_TIMEOUT_MS": str(max(1, timeout_s) * 1000),
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _secure_repository_id(cwd: str) -> str:
    digest = hashlib.sha256(str(Path(cwd).resolve()).encode("utf-8")).hexdigest()[:16]
    return f"secure_local_{digest}"


def _secure_snapshot_id(cwd: str, finding: NormalizedFinding) -> str:
    raw = f"{Path(cwd).resolve()}|{finding.canonical_id}"
    return "secure_snapshot_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _evidence_bundle(cwd: str, finding: NormalizedFinding) -> dict:
    path = finding.primary_location.uri if finding.primary_location else ""
    items = []
    if path:
        items.append(
            {
                "file_path": path,
                "symbol_name": None,
                "snippet": _source_snippet(cwd, path, finding.primary_location.start_line),
            }
        )
    return {
        "suspect_files": [path] if path else [],
        "suspect_symbols": [],
        "evidence_items": items,
    }


def _source_snippet(cwd: str, relative_path: str, start_line: int | None) -> str:
    path = Path(cwd) / relative_path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    line = max(1, int(start_line or 1))
    begin = max(1, line - 8)
    end = min(len(lines), line + 8)
    return "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(begin, end + 1))


def _count_repo_files(cwd: str) -> int:
    return sum(1 for path in Path(cwd).rglob("*") if path.is_file() and ".git" not in path.parts)


def _estimate_repo_tokens(cwd: str) -> int:
    total = 0
    for path in Path(cwd).rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            total += max(1, path.stat().st_size // 4)
        except OSError:
            continue
    return total
