"""`codna mcp` — run codna as an MCP server for Cursor / Claude Desktop.

Exposes codna's local repo-intelligence tools (triage, fix, secure, recall) backed by
the same engine client the CLI uses. Add to your MCP config:

    { "mcpServers": { "codna": { "command": "codna", "args": ["mcp"],
        "env": { "CODNA_API_KEY": "..." } } } }
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

from .cli import _api_key, _client, _dump, _register, _runtime_keys

# FastMCP builds each tool's input schema by evaluating the Annotated[str, Field(...)]
# annotations below against THIS module's globals (inspect.signature(eval_str=True)), so
# pydantic's Field must resolve here whenever a server can be built. But this module must also
# import without pydantic: the offline CI lane installs neither pydantic nor the mcp extra and
# exercises the *_json handlers directly. Keying the import to pydantic's presence satisfies
# both — a server can only be built when the mcp extra (which hard-requires pydantic) exists,
# so the binding is never absent when FastMCP evaluates the annotations.
if importlib.util.find_spec("pydantic") is not None:
    from pydantic import Field

LOGIN_REQUIRED_MESSAGE = (
    "requires the one-time free `codna login` device authorization (free community license); "
    "run `codna login` once — fully offline thereafter"
)


def _login_required_error(tool: str) -> str | None:
    """Uniform execution gate: every MCP tool call requires the free community login.

    All codna tools execute under the same licensing model (algenta house pattern — sqai returns
    `login_required`, algenta-mcp an auth error; introspection stays credential-free, execution
    needs the login). The `codna login` artifact is the codna account key, resolved through the
    same key layer the CLI/status paths use: env first, then the OS keychain via the non-secret
    name index (a clean machine is never prompted), never the network — offline after the one
    login. The key's presence proves the login state; it is not validated against the control
    plane (offline by design). Returns the structured error line, or None when authorized.
    """
    if _api_key(_runtime_keys(include_keychain=True)):
        return None
    return f"{tool} error: {LOGIN_REQUIRED_MESSAGE}"


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


def load_skill_json(repo: str = ".", name: str = "") -> str:
    """The body of ``codna_load_skill``, as a module-level function.

    Defined HERE rather than only inside ``_build_server`` for the same reason ``fix_json`` /
    ``secure_json`` / ``recall_json`` are: the minimal-dependency CI lane installs no ``mcp`` extra, so a
    tool whose only test is behind that extra is a tool that is never tested there. The registered tool
    below is a thin wrapper over this, so there is one implementation and two ways to reach it.

    Returns the skill as JSON (body + bundled file paths), or a structured ``skill_not_found`` — never a
    raise, because a tool must answer.
    """
    if not name or not name.strip():
        return "codna_load_skill error: name is required"
    try:
        from .rules import load_skill

        skill = load_skill(repo, name)
    except Exception as exc:  # noqa: BLE001 — surface as tool text, never crash the server
        return f"codna_load_skill error: {exc}"
    if skill is None:
        return json.dumps({
            "error": {
                "code": "skill_not_found",
                "message": f"no skill named {name!r}",
                "details": {"requested": name, "repo": repo},
            },
        }, indent=2)
    return json.dumps({
        "name": skill.name,
        "scope": skill.scope.value,
        "mode": skill.mode.value,
        "source_path": skill.source_path,
        "description": skill.description,
        "paths": list(skill.paths),
        # Paths RELATIVE TO THE SKILL FOLDER, so the body's references resolve without the reader
        # having to know where the repository is checked out.
        "bundled": list(skill.bundled),
        "body": skill.body,
    }, indent=2)


def recall_json(repo: str = ".", query: str = "", service: str = "", language: str = "",
                final_k: int = 8) -> str:
    """Return local Telys memory recall as JSON without crashing the MCP server."""
    gate = _login_required_error("codna_recall")
    if gate:
        return gate
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
    gate = _login_required_error("codna_report_bug")
    if gate:
        return gate
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
    gate = _login_required_error("codna_secure")
    if gate:
        return gate
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
    gate = _login_required_error("codna_fix")
    if gate:
        return gate
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


def _browser_tools_enabled() -> bool:
    """Whether ``CODNA_BROWSER_TOOL`` is on. A bad value is off for REGISTRATION, and said so.

    Registration must not raise: `codna mcp` failing to list its own tools because of a typo in one
    optional switch turns a browser setting into a broken server. The typo is reported instead — by
    ``codna mcp browser``, by the ``codna://capabilities`` resource, and by every browser tool body when
    called. Off is the correct answer for "this tool must not exist yet".
    """
    from .browser_tool import BrowserToolConfigError, browser_tool_enabled

    try:
        return browser_tool_enabled()
    except BrowserToolConfigError:
        return False


def _browser_capability_state() -> dict:
    """The honest browser-tool state for the capabilities resource (never raises)."""
    from .browser_tool import browser_tool_state

    return browser_tool_state()


def _browser_image(content_factory) -> list:
    """Run a browser body that returns ``(text, images)`` and build MCP content, image included.

    The result is a LIST of content blocks rather than a string, because a screenshot that arrived as a
    base64 string in a text block is a screenshot the model cannot see. A failure still answers with text —
    the tool body already rendered it.
    """
    from mcp.types import ImageContent, TextContent

    text, images = content_factory()
    blocks: list = [TextContent(type="text", text=text)]
    for image in images:
        blocks.append(ImageContent(
            type="image", data=image["data"],
            mimeType=image.get("mime_type") or "image/png",
        ))
    return blocks


def _register_browser_tools(server) -> None:
    """Register the ``codna_browser_*`` tools, but ONLY when the switch is on.

    Gated at REGISTRATION rather than at call time so an agent never sees a browser tool it cannot use: a
    tool listed and then refusing every call is a worse contract than a tool that is absent, and the
    absence is explained by ``codna mcp browser`` and by the ``codna://capabilities`` resource.

    Each body is the module-level function in ``browser_bridge`` — the same code the smoke test and the
    durable path use — so there is one implementation behind the registration, not two.
    """
    from . import browser_bridge

    if not _browser_tools_enabled():
        return

    from mcp.types import ToolAnnotations

    @server.tool(annotations=ToolAnnotations(
        title="Open a URL in a browser and read the page",
        readOnlyHint=False,
        # Same URL twice opens the same page, but the session is stateful (cookies, history).
        idempotentHint=False,
        # A browser loads whatever the network serves. This is the tool whose whole job is open-world.
        openWorldHint=True,
    ))
    def codna_browser_navigate(url: str = "") -> str:
        """Navigate the browser to `url` and return the page's accessible text.

        Requires `CODNA_BROWSER_TOOL=1` in the server's environment (default off) and Node.js 18+ on PATH:
        Codna drives Playwright's own MCP server and does not bundle Node. No `codna login` is needed —
        browsing is local.

        url: an absolute http(s) URL, a `data:` URL, or a `file:` URL (the last needs
          `CODNA_BROWSER_FILE_ACCESS=unrestricted`).
        Returns the page's accessible tree as text (the same snapshot `codna_browser_snapshot` reads), or
          `codna_browser_navigate error: ...` — the tool never raises.
        """
        return browser_bridge.browser_navigate_json(url)

    @server.tool(annotations=ToolAnnotations(
        title="Read the current page as accessible text",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ))
    def codna_browser_snapshot(depth: float = 0) -> str:
        """Read the current page's accessibility tree — the readable form of what is on screen.

        This is the assertion primitive: after navigating, read the snapshot and check that the text,
        heading or button you expected is present. It runs no page JavaScript.

        depth: limit the tree depth; 0 (default) means the whole tree.
        Returns the snapshot as text, or `codna_browser_snapshot error: ...`.
        """
        return browser_bridge.browser_snapshot_json(depth)

    @server.tool(annotations=ToolAnnotations(
        title="Click an element",
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ))
    def codna_browser_click(target: str = "", element: str = "") -> str:
        """Click an element on the current page.

        target: the element reference from `codna_browser_snapshot` (e.g. `e3`), a CSS selector (e.g.
          `#submit`), or another Playwright locator.
        element: optional human-readable description of the element, used by the browser's permission
          prompt.
        Returns the page snapshot after the click, or `codna_browser_click error: ...`.
        """
        return browser_bridge.browser_click_json(target, element)

    @server.tool(annotations=ToolAnnotations(
        title="Type into an element",
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ))
    def codna_browser_type(target: str = "", text: str = "", element: str = "",
                           submit: bool = False) -> str:
        """Type text into an editable element (an input, a textarea, a contenteditable).

        target: the element reference from `codna_browser_snapshot`, or a CSS selector.
        text: what to type.
        element: optional human-readable element description.
        submit: press Enter after typing (default false).
        Returns the page snapshot after typing, or `codna_browser_type error: ...`.
        """
        return browser_bridge.browser_type_json(target, text, element, submit)

    @server.tool(annotations=ToolAnnotations(
        title="Screenshot the page",
        readOnlyHint=False,
        # Writes a new image file each call; two calls are two files.
        idempotentHint=False,
        openWorldHint=False,
    ))
    async def codna_browser_screenshot(filename: str = "", full_page: bool = False):
        """Screenshot the page and return the IMAGE, plus where it was written.

        The image is returned as real image content, not as a path in text, so the model actually sees it.

        filename: a name relative to the browser output directory (`$CODNA_RUNTIME_ROOT/browser`); an
          absolute path or `..` is refused. Default: a timestamped `page-<time>.png`.
        full_page: capture the whole scrollable page instead of the viewport (default false).
        Returns the image content plus a JSON block with the path, size and Playwright's own output, or
          `codna_browser_screenshot error: ...`.
        """
        return await asyncio.to_thread(
            _browser_image,
            lambda: browser_bridge.browser_screenshot_content(filename, full_page),
        )

    @server.tool(annotations=ToolAnnotations(
        title="Wait for text, or for a pause",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ))
    def codna_browser_wait(text: str = "", text_gone: str = "", time_s: float = 0.0) -> str:
        """Wait until text appears (or disappears), or until a number of seconds have passed.

        Use this instead of guessing: a page that fetches data after load needs the text to be waited for
        before its snapshot is asserted.

        text: wait for this text to appear.
        textGone: wait for this text to disappear.
        time_s: seconds to wait, at most 30. With `text`/`text_gone` it is their timeout instead.
        Returns the page snapshot after the wait, or `codna_browser_wait error: ...`.
        """
        return browser_bridge.browser_wait_json(text, text_gone, time_s)

    @server.tool(annotations=ToolAnnotations(
        title="Close the browser session",
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))
    def codna_browser_close() -> str:
        """Close the browser session and its browser process.

        The session is reused across calls, so close it when a verification is done rather than leaving
        Chromium resident. The next browser call starts a fresh session.
        """
        return browser_bridge.browser_close_json()


def _build_server():
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP("codna")

    @server.tool(annotations=ToolAnnotations(
        title="Triage a repository",
        readOnlyHint=True,
        # Same repo + same issue -> same suspect files; re-triage changes nothing.
        idempotentHint=True,
    ))
    def codna_triage(repo: str = ".", issue: str = "") -> str:
        """Understand a repository and locate the code relevant to an issue.

        The sibling tools act, this one maps: codna_fix plans/acts on one bug and
        codna_recall retrieves stored code memory — triage is the zero-token map +
        suspect-file list you run first.

        Requires the one-time free `codna login` device authorization (free community
        license); fully offline thereafter.

        repo: a local path or a git URL (default: current directory).
        issue: optional description of what you're looking for.
        Returns suspect files + the context-reduction the engine achieved (0 LLM tokens).
        """
        gate = _login_required_error("codna_triage")
        if gate:
            return gate
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

    @server.tool(annotations=ToolAnnotations(
        title="Fix a bug (plan, or open a PR)",
        readOnlyHint=False,
        # Default (open_pr=false) only reads + plans; with open_pr=true it pushes a fix branch —
        # additive, never destructive.
        destructiveHint=False,
        # open_pr=true writes outside this machine: it opens a real pull request on GitHub.
        openWorldHint=True,
    ))
    async def codna_fix(
        repo: Annotated[str, Field(
            description="A local path or a git URL. open_pr=true requires a git URL (a local "
                        "checkout has no remote to push to). Default '.': the current "
                        "directory, or the server default set by `codna mcp start --repo` "
                        "(env CODNA_MCP_DEFAULT_REPO).",
        )] = ".",
        issue: Annotated[str, Field(
            description="What's broken — e.g. the failing test or the observed behavior. "
                        "Required when open_pr=true; otherwise optional but strongly "
                        "recommended (an empty issue yields a generic repo-wide plan).",
        )] = "",
        ref: Annotated[str, Field(
            description="Branch, tag, or commit to analyze: the planner's repository "
                        "snapshot is taken at this ref (registration forwards it to "
                        "create_repository_snapshot). Default '': the repo's current state — "
                        "the checkout as-is for a local path, the default branch for a git URL.",
        )] = "",
        open_pr: Annotated[bool, Field(
            description="false (default): READ-ONLY plan — returns root cause, confidence and "
                        "a patch reference, and changes nothing. true: pushes a fix branch "
                        "and OPENS A PULL REQUEST — needs a git URL for repo and GITHUB_TOKEN "
                        "(or CODNA_GITHUB_TOKEN) with write access; returns pull_request_url.",
        )] = False,
        model: Annotated[str, Field(
            description="Provider-qualified planner model, e.g. openai/gpt-5. Default "
                        "'repository.verified_agentic_v1' (codna's verified agentic runtime).",
        )] = "repository.verified_agentic_v1",
    ) -> str:
        """Find and fix a bug. Runs the full Codna agent + engine + risk simulation.

        Requires the one-time free `codna login` device authorization (free community
        license) plus a provider key (BYOK — e.g. ANTHROPIC_API_KEY or `codna key set
        anthropic`). Offline truth: planning calls the provider API (network), and only
        open_pr=true writes to GitHub — codna_triage, codna_secure and codna_recall are
        the fully-offline tools.

        Returns the root cause, confidence, model, and (when open_pr) the pull_request_url.
        Failures come back as 'codna_fix error: ...' text — the tool never raises.
        """
        return await asyncio.to_thread(fix_json, repo, issue, ref, open_pr, model)

    @server.tool(annotations=ToolAnnotations(
        title="Prove scanner-finding reachability",
        readOnlyHint=True,
        # Same SARIF in -> same classification out; purely deterministic and local.
        idempotentHint=True,
        # Reads one local SARIF file; never starts the sidecar or touches the network.
        openWorldHint=False,
    ))
    def codna_secure(
        sarif_path: Annotated[str, Field(
            description="Path to the scanner's SARIF 2.1.0 report (CodeQL, Semgrep, Snyk, "
                        "Trivy, or any other scanner). REQUIRED — there is no auto-discovery: "
                        "omitting it fails input validation, and an empty string returns "
                        "'codna_secure error: sarif_path is required'.",
        )],
        repo: Annotated[str, Field(
            description="Repository the findings belong to (a local path or a git URL), echoed "
                        "as 'repository' in the output for traceability. The classification "
                        "reads ONLY the SARIF file — repo is never scanned. Default '.': the "
                        "current directory, or the server default set by `codna mcp start "
                        "--repo` (env CODNA_MCP_DEFAULT_REPO).",
        )] = ".",
        ref: Annotated[str, Field(
            description="Branch, tag, or commit the SARIF was produced from. Accepted for "
                        "parity with `codna secure --ref`; informational only in the MCP path — "
                        "the local SARIF-only classifier does not resolve it. Default ''.",
        )] = "",
    ) -> str:
        """Prove which scanner findings are reachable. Ingests a SARIF 2.1.0 report (CodeQL /
        Semgrep / Snyk / Trivy), classifies each finding (exploitable / production-reachable /
        unreachable / unknown) via the local engine, and reports which are autofix-eligible.
        Read-only, deterministic, 0 LLM tokens — the SARIF file is the only input read; the
        repo is never scanned and the agent sidecar never starts.

        `sarif_path` is REQUIRED (see the input schema) — there is no workspace scan or
        default-path fallback. SARIF provenance (tool name + revision) must be complete or the
        report is rejected. All failures come back as 'codna_secure error: ...' text — the
        tool never raises.

        Requires the one-time free `codna login` device authorization (free community
        license); fully offline thereafter.
        """
        return secure_json(repo=repo, sarif_path=sarif_path, ref=ref)

    @server.tool(annotations=ToolAnnotations(
        title="Recall code from local memory",
        readOnlyHint=True,
        # Same query over the same index -> same symbols; repeated calls change nothing
        # (first use may build the local index, once).
        idempotentHint=True,
        # On-device Telys memory only: no network, no sidecar, 0 LLM tokens.
        openWorldHint=False,
    ))
    def codna_recall(repo: str = ".", query: str = "", service: str = "", language: str = "",
                     final_k: int = 8) -> str:
        """Recall relevant code from the local on-device Telys memory — semantic + lexical
        search over the symbols this machine has already indexed, with zero LLM tokens and
        zero network calls.

        Use when you need the code behind a concept ("where is SARIF provenance validated?")
        rather than a whole-repo map (codna_triage) — and when you want it offline. The index
        is built on first use (an empty index auto-indexes the repo, once) and is stored
        outside the git checkout. On a fresh machine, first use requires the one-time free
        `codna login` device authorization (free community license); the on-device memory
        runtime arrives via the login-gated runtime install. After that, recall is fully
        offline with no key and no network calls.

        repo: a local path or git URL (default: current directory, or the server default set
          by `codna mcp start --repo`).
        query: what to recall (required — an empty query returns an error string).
        service: optional service name to scope recall to one service in a monorepo
          (default: all services).
        language: optional language filter such as "python" (default: all indexed languages).
        final_k: maximum number of symbols to return (default: 8).
        Returns matching symbols, the ranking explanation, and the candidate count as JSON.
        Failure modes (missing query, Telys memory not installed, unreadable repo) come back
        as "codna_recall error: ..." text — the tool never raises.
        """
        return recall_json(repo, query, service, language, final_k)

    @server.tool(annotations=ToolAnnotations(
        title="Report a bug, feature, or question",
        readOnlyHint=False,
        # Only ever CREATES a new issue in thyn-ai/feedback; never modifies or deletes
        # existing state.
        destructiveHint=False,
        # Writes outside this machine: files a real GitHub issue (or, offline, returns a
        # pre-filled URL for a human to submit).
        openWorldHint=True,
    ))
    async def codna_report_bug(
        title: str, body: str = "", product: str = "codna", include_diagnostics: bool = False,
    ) -> str:
        """File a report to thyn-ai/feedback — the public front door for algenta, codna, telys,
        and sqai. Files the same shape of issue `codna report` (the CLI) does.
        It only files the issue — codna_fix is the one that can open a PR with the fix.

        Requires the one-time free `codna login` device authorization (free community
        license); auto-filing additionally needs a GitHub token (else a pre-filled URL).

        title: a short summary, e.g. "codna fix hangs on a monorepo".
        body: what happened, or what you want — as much detail as you have.
        product: one of algenta, codna, telys, sqai, accounts, docs (default: codna).
        include_diagnostics: attach the same redacted output `codna doctor` prints — no secret
          values, only which config sources are set. Off by default.
        Returns the filed issue's URL, or — if it couldn't file automatically (no GitHub token
        reachable, offline) — a pre-filled URL for a human to finish submitting.
        """
        return await asyncio.to_thread(report_bug_json, title, body, product, include_diagnostics)

    @server.tool(annotations=ToolAnnotations(
        title="Pull one project rule by name",
        readOnlyHint=True,
        idempotentHint=True,
        # It only reads markdown inside the repository; it opens no connection and writes nothing.
        openWorldHint=False,
    ))
    def codna_request_rule(repo: str = ".", name: str = "") -> str:
        """Fetch the FULL text of one project rule, by the name the prompt catalogued.

        Codna attaches `alwaysApply` and glob-matched rules in full, but a rule with only a
        `description` is catalogued as a single line so it costs a token, not a kilobyte, on the turns
        it has nothing to do with. Call this when that line looks relevant: `name` is the identifier
        the catalog printed, with or without a leading `@`.

        Requires the one-time free `codna login` device authorization (free community
        license); fully offline thereafter.

        repo: a local path (default: current directory).
        name: the rule name from the catalog, e.g. "python" or "@python".
        Returns the rule's scope, source path and full body, or a structured `rule_not_found`.
        """
        gate = _login_required_error("codna_request_rule")
        if gate:
            return gate
        if not name or not name.strip():
            return "codna_request_rule error: name is required"
        try:
            from .rules import request_rule

            rule = request_rule(_default_repo(repo), name)
        except Exception as exc:  # noqa: BLE001 — surface as tool text, never crash the server
            return f"codna_request_rule error: {exc}"
        if rule is None:
            return json.dumps({
                "error": {
                    "code": "rule_not_found",
                    "message": f"no rule named {name!r}",
                    "details": {"requested": name, "repo": _default_repo(repo)},
                },
            }, indent=2)
        return json.dumps({
            "name": rule.name,
            "scope": rule.scope.value,
            "mode": rule.mode.value,
            "source_path": rule.source_path,
            "base_dir": rule.base_dir,
            "description": rule.description,
            "globs": list(rule.globs),
            "body": rule.body,
        }, indent=2)

    @server.tool(annotations=ToolAnnotations(
        title="Pull one skill's instructions by name",
        readOnlyHint=True,
        idempotentHint=True,
        # It only reads markdown inside the repository; it opens no connection and writes nothing.
        openWorldHint=False,
    ))
    def codna_load_skill(repo: str = ".", name: str = "") -> str:
        """Fetch one skill's FULL instructions, by the name the prompt catalogued.

        The prompt carries a CATALOGUE of skills — one line each with the name, the description and
        the `paths` they apply to — and never their bodies. That is what keeps a repository with
        dozens of skills from costing dozens of full prompts on every turn. Call this when a catalogue
        line looks relevant: `name` is the identifier it printed, with or without a leading `/` or `@`.

        The SAME pull serves `/skill-name`: a skill the user invokes and a skill you ask for are the
        same body through the same tool, so the two can never diverge.

        Requires the one-time free `codna login` device authorization (free community
        license); fully offline thereafter.

        repo: a local path (default: current directory).
        name: the skill name from the catalogue, e.g. "pdf-processing" or "/pdf-processing".
        Returns the skill's scope, source path, body and the files it ships beside its SKILL.md
        (`scripts/`, `references/`, `assets/`), or a structured `skill_not_found`.
        """
        gate = _login_required_error("codna_load_skill")
        if gate:
            return gate
        # The SAME function the module-level body exposes, so the minimal-dependency lane tests the real
        # implementation rather than a second copy that could drift from it.
        return load_skill_json(_default_repo(repo), name)

    @server.resource("codna://capabilities")
    def capabilities() -> str:
        """codna's MCP capabilities — @-mentionable so the client knows what codna offers."""
        from .browser_tool import BROWSER_TOOL_ENV

        return json.dumps({
            "product": "codna",
            "tools": {
                "codna_triage": "understand a repo + locate relevant code (read-only, 0 LLM tokens)",
                "codna_fix": "find + fix a bug (root cause, patch ref, confidence, optional model)",
                "codna_secure": "prove which scanner findings are reachable (read-only)",
                "codna_recall": "recall code from local on-device memory",
                "codna_report_bug": "file a bug/feature/question to thyn-ai/feedback",
                "codna_request_rule": "pull one project rule's full text by name (the description-only "
                                      "rules the prompt only catalogues)",
                "codna_load_skill": "pull one skill's full instructions by name (the skills the prompt "
                                    "only catalogues), including the files it ships",
                **(
                    {
                        "codna_browser_navigate": "open a URL and read the page (needs "
                                                  f"{BROWSER_TOOL_ENV}=1)",
                        "codna_browser_snapshot": "read the current page as accessible text",
                        "codna_browser_click": "click an element on the page",
                        "codna_browser_type": "type into an editable element",
                        "codna_browser_screenshot": "screenshot the page (returns the image)",
                        "codna_browser_wait": "wait for text to appear/disappear, or a pause",
                        "codna_browser_close": "close the browser session",
                    }
                    if _browser_tools_enabled()
                    else {}
                ),
            },
            # Reported even when OFF: the switch is the one thing a user needs to know about this tool,
            # and "which export turns it on" is not discoverable from its absence.
            "browser": _browser_capability_state(),
            "default_repo": os.environ.get("CODNA_MCP_DEFAULT_REPO", "."),
        }, indent=2)

    @server.prompt(title="Fix a bug with codna")
    def fix_bug(issue: str, repo: str = ".") -> str:
        """Templated prompt: ask codna to locate then fix a bug."""
        return (f"Use the codna tools to fix this in repo `{repo}`:\n\n{issue}\n\n"
                "First call codna_triage to locate the relevant code, then codna_fix with the issue and model when specified. "
                "Report the root cause, the patch reference, and codna's confidence.")

    # Registered LAST so the eight core tools keep their order in `tools/list`, and only when the browser
    # switch is on: an agent must never see a browser tool the server would refuse to run.
    _register_browser_tools(server)

    return server


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
