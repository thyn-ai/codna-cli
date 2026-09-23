"""`codna mcp` — run codna as an MCP server for Cursor / Claude Desktop.

Exposes codna's local repo-intelligence tools (triage, fix, secure, recall) backed by
the same engine client the CLI uses. Add to your MCP config:

    { "mcpServers": { "codna": { "command": "codna", "args": ["mcp"],
        "env": { "CODNA_API_KEY": "..." } } } }
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from .cli import _client, _dump, _register


def _default_repo(repo: str) -> str:
    """Resolve the repo for an MCP tool call: an explicit non-'.' value wins, else the default set by
    `codna mcp start --repo X` (env CODNA_MCP_DEFAULT_REPO), else the current directory."""
    if repo and repo != ".":
        return repo
    return os.environ.get("CODNA_MCP_DEFAULT_REPO") or "."


def _mcp_memory_root() -> Path:
    override = (os.environ.get("CODNA_TELYS_MEMORY_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return (Path.home() / ".codna" / "telys-memory").resolve(strict=False)


def _mcp_memory_db_path(repo: str) -> str:
    """Store background MCP recall indexes outside the user's git checkout."""
    from .memory import repo_id_for

    repo_id = repo_id_for(repo)
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:24]
    return str(_mcp_memory_root() / "mcp" / digest)


def recall_json(repo: str = ".", query: str = "", service: str = "", language: str = "",
                final_k: int = 8) -> str:
    """Return local Telys memory recall as JSON without crashing the MCP server."""
    if not query:
        return "codna_recall error: query is required"
    repo = _default_repo(repo)
    try:
        from .memory import CodeMemory

        mem = CodeMemory(repo, db_path=_mcp_memory_db_path(repo), service=service or None)
        if mem.is_empty():
            mem.index(languages=(language,) if language else None)
        res = mem.recall(query, service=service or None, language=language or None, final_k=final_k)
        return json.dumps({
            "symbols": res["symbols"],
            "explain": res["explain"],
            "candidate_count": res["candidate_count"],
        }, indent=2)
    except Exception as exc:  # noqa: BLE001
        return f"codna_recall error: {exc}"


def report_bug_json(
    title: str, body: str = "", product: str = "codna", include_diagnostics: bool = False,
) -> str:
    """File a report to thyn-ai/feedback and return the result as JSON, without crashing the MCP
    server. Same submission path `codna report` (the CLI) uses, so an agent and a human land in
    the same place with the same fields."""
    if not title:
        return "codna_report_bug error: title is required"
    try:
        from .report_cli import submit_report

        result = submit_report(title=title, product=product, body=body,
                               attach_diagnostics=include_diagnostics)
        return json.dumps({
            "submitted": result.submitted,
            "url": result.url,
            "local_path": result.local_path,
        }, indent=2)
    except Exception as exc:  # noqa: BLE001 - surface as tool text, never crash the MCP server.
        return f"codna_report_bug error: {exc}"


def secure_json(repo: str = ".", sarif_path: str = "", ref: str = "") -> str:
    """Return local SARIF reachability classification as JSON for MCP.

    MCP secure is read-only and local by default: it must not start or depend on the agent sidecar.
    """
    del ref  # local reference classification is SARIF-only; repo is included for caller traceability.
    repo = _default_repo(repo)
    try:
        from .policy import Policy
        from .refengine import LocalReferenceEngine
        from .sarif import ingest_sarif
        from .secure import classify_only

        if not sarif_path:
            return "codna_secure error: sarif_path is required"
        ingest = ingest_sarif(sarif_path)
        if not ingest.provenance_valid:
            return "codna_secure error: SARIF provenance incomplete: " + "; ".join(ingest.provenance_errors)
        report = classify_only(ingest, engine=LocalReferenceEngine(), policy=Policy())
        return json.dumps({
            "repository": repo,
            "sarif": sarif_path,
            "counts": report.counts(),
            "autofix_eligible": report.eligible,
            "findings": [
                {
                    "rule_id": r.rule_id,
                    "kind": r.finding_kind,
                    "classification": str(r.classification),
                    "eligible": bool(r.eligible),
                    "reason": r.reason,
                }
                for r in report.rows
            ],
        }, indent=2)
    except Exception as exc:  # noqa: BLE001 - surface as tool text, never crash the MCP server.
        return f"codna_secure error: {exc}"


def fix_json(
    repo: str = ".", issue: str = "", ref: str = "", open_pr: bool = False,
    model: str = "repository.verified_agentic_v1",
) -> str:
    """Return a Codna fix plan/PR result as JSON text without crashing the MCP server.

    This is deliberately synchronous because it reuses the CLI/local-client path. The MCP tool
    runs it in a worker thread so local repository-intelligence calls can safely use their own
    event loop internally.
    """
    repo = _default_repo(repo)
    try:
        c = _client(include_keychain=True)
        if open_pr:
            # Same path the CLI's `codna fix --open-pr` uses (engine apply -> real PR), so the
            # MCP tool and CLI are one behavior. run_fix() would sys.exit on bad input, which
            # must never kill the server - so we pre-validate and also trap SystemExit.
            import types

            from . import fix_run

            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("CODNA_GITHUB_TOKEN")
            if not token:
                return "codna_fix error: open_pr=true needs a GITHUB_TOKEN with write access."
            if os.path.isdir(os.path.abspath(os.path.expanduser(repo))):
                return "codna_fix error: open_pr=true needs a git URL (a local path has no remote to push to)."
            if not issue:
                return "codna_fix error: open_pr=true needs an `issue` describing what's broken."
            ns = types.SimpleNamespace(
                repo=repo, ref=ref or None, issue=issue, from_junit=None,
                tests=False, apply=False, open_pr=True, github_token=token,
                model=model or "repository.verified_agentic_v1", max_iterations=1,
                pr_title=None, pr_body=None, base_branch=None, test_cmd=None, as_json=True,
            )
            try:
                result = fix_run.run_fix(c, ns)
            except SystemExit as exc:  # _die() inside run_fix - surface, don't crash the server
                return f"codna_fix error: {exc}"
            it = (result.get("iterations") or [{}])[-1]
            s = it.get("summary") or {}
            applied = result.get("applied") or {}
            return json.dumps({
                "root_cause": s.get("root_cause"),
                "confidence": s.get("confidence"),
                "pull_request_url": result.get("pull_request_url"),
                "status": applied.get("status"),
                "model": s.get("runtime_model"),
            }, indent=2)
        # plan-only (default, read-only): locate + plan the fix, change nothing. Use the same
        # model/provider routing as `codna fix --model` so MCP users in Cursor/Claude get parity
        # with CLI and GitHub Action channels.
        from . import fix_run

        _rid, _snap, plan = fix_run._plan_once(
            c, repo=repo, ref=ref or None, issue=issue, failing=[],
            model=model or "repository.verified_agentic_v1", open_pr=False, gh_token=None,
        )
        dp = plan.get("decision_plan") or {}
        ra = dp.get("repository_analysis") or {}
        usage = plan.get("planner_usage") or {}
        return json.dumps({
            "root_cause": ra.get("root_cause"),
            "impacted_symbols": ra.get("impacted_symbols"),
            "blast_radius": ra.get("blast_radius"),
            "confidence": dp.get("confidence"),
            "patch_ref": ra.get("generated_patch_ref"),
            "model": plan.get("runtime_model"),
            "cost_usd": usage.get("cost_usd"),
        }, indent=2)
    except Exception as exc:  # noqa: BLE001
        return f"codna_fix error: {exc}"


def _build_server():
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP("codna")

    @server.tool(annotations=ToolAnnotations(title="Triage a repository", readOnlyHint=True))
    def codna_triage(repo: str = ".", issue: str = "") -> str:
        """Understand a repository and locate the code relevant to an issue.

        repo: a local path or a git URL (default: current directory).
        issue: optional description of what you're looking for.
        Returns suspect files + the context-reduction the engine achieved (0 LLM tokens).
        """
        repo = _default_repo(repo)
        try:
            c = _client()
            rid, snap = _register(c, repo, None)
            sig = {"issue_text": issue or "Map this repository and locate its most relevant code."}
            tri = _dump(c.triage_repository(rid, {"snapshot_id": snap["snapshot_id"], "signals": sig}))
            return json.dumps({
                "suspect_files": tri.get("suspect_files"),
                "reduction_ratio": tri.get("reduction_ratio"),
                "raw_repo_tokens": tri.get("raw_repo_token_estimate"),
                "evidence_bundle_tokens": tri.get("evidence_bundle_token_count"),
            }, indent=2)
        except Exception as exc:  # noqa: BLE001 — surface as tool error, never crash the server
            return f"codna_triage error: {exc}"

    @server.tool(annotations=ToolAnnotations(title="Fix a bug (plan, or open a PR)", readOnlyHint=False))
    async def codna_fix(
        repo: str = ".", issue: str = "", ref: str = "", open_pr: bool = False,
        model: str = "repository.verified_agentic_v1",
    ) -> str:
        """Find and fix a bug. Runs the full Codna agent + engine + risk simulation.

        repo: a local path or a git URL. issue: what's broken (e.g. the failing test).
        model: optional provider-qualified model such as openai/gpt-5.
        open_pr: when false (default) this is READ-ONLY — returns the plan (root cause, confidence,
          patch reference) and changes nothing. When true, codna pushes a fix branch and OPENS A
          PULL REQUEST — this needs a git URL for `repo` and a GITHUB_TOKEN with write access.
        Returns the root cause, confidence, model, and (when open_pr) the pull_request_url.
        """
        return await asyncio.to_thread(fix_json, repo, issue, ref, open_pr, model)

    @server.tool(annotations=ToolAnnotations(title="Prove scanner-finding reachability", readOnlyHint=True))
    def codna_secure(repo: str = ".", sarif_path: str = "", ref: str = "") -> str:
        """Prove which scanner findings are reachable. Ingests a SARIF report (CodeQL /
        Semgrep / Snyk / Trivy), classifies each finding (exploitable / production-reachable /
        unreachable / unknown) via the engine, and reports which are autofix-eligible.
        Read-only and 0 LLM tokens. `sarif_path` is required.
        """
        return secure_json(repo, sarif_path, ref)

    @server.tool(annotations=ToolAnnotations(title="Recall code from local memory", readOnlyHint=True))
    def codna_recall(repo: str = ".", query: str = "", service: str = "", language: str = "",
                     final_k: int = 8) -> str:
        """Recall relevant code from the local Telys memory, if installed."""
        return recall_json(repo, query, service, language, final_k)

    @server.tool(annotations=ToolAnnotations(title="Report a bug, feature, or question", readOnlyHint=False))
    async def codna_report_bug(
        title: str, body: str = "", product: str = "codna", include_diagnostics: bool = False,
    ) -> str:
        """File a report to thyn-ai/feedback — the public front door for algenta, codna, telys,
        and sqai. Files the same shape of issue `codna report` (the CLI) does.

        title: a short summary, e.g. "codna fix hangs on a monorepo".
        body: what happened, or what you want — as much detail as you have.
        product: one of algenta, codna, telys, sqai, accounts, docs (default: codna).
        include_diagnostics: attach the same redacted output `codna doctor` prints — no secret
          values, only which config sources are set. Off by default.
        Returns the filed issue's URL, or — if it couldn't file automatically (no GitHub token
        reachable, offline) — a pre-filled URL for a human to finish submitting.
        """
        return await asyncio.to_thread(report_bug_json, title, body, product, include_diagnostics)

    @server.resource("codna://capabilities")
    def capabilities() -> str:
        """codna's MCP capabilities — @-mentionable so the client knows what codna offers."""
        return json.dumps({
            "product": "codna",
            "tools": {
                "codna_triage": "understand a repo + locate relevant code (read-only, 0 LLM tokens)",
                "codna_fix": "find + fix a bug (root cause, patch ref, confidence, optional model)",
                "codna_secure": "prove which scanner findings are reachable (read-only)",
                "codna_recall": "recall code from local on-device memory",
                "codna_report_bug": "file a bug/feature/question to thyn-ai/feedback",
            },
            "default_repo": os.environ.get("CODNA_MCP_DEFAULT_REPO", "."),
        }, indent=2)

    @server.prompt(title="Fix a bug with codna")
    def fix_bug(issue: str, repo: str = ".") -> str:
        """Templated prompt: ask codna to locate then fix a bug."""
        return (f"Use the codna tools to fix this in repo `{repo}`:\n\n{issue}\n\n"
                "First call codna_triage to locate the relevant code, then codna_fix with the issue and model when specified. "
                "Report the root cause, the patch reference, and codna's confidence.")

    return server


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
