"""`codna report` — file a report to thyn-ai's public feedback repo (thyn-ai/feedback).

One of three doors into the same intake: humans use the issue forms in a browser, scripts use
`gh issue create` against the same forms, and this is the door for a terminal or an AI agent
driving the CLI directly. The MCP tool `codna_report_bug` (mcp_server.py) wraps this module's
`submit_report()`, so all three paths produce an identically-shaped issue.

Never fails the caller's session, by design: any submission error — no token, no network, an
air-gapped machine — falls back to writing the report to a local file and printing a pre-filled
`github.com/.../issues/new` URL instead of raising. Filing a report is the point; losing what the
user already typed because GitHub was unreachable would defeat it.

Diagnostics are attached only when the caller opts in (`--attach-diagnostics`), and only the
redacted payload `codna doctor` itself would print — see `doctor.build_report` /
`doctor.format_public_output`. Nothing here reads or sends anything the user didn't ask for; see
CLAUDE.md's no-phone-home policy — this command is the explicitly-user-initiated exception it
carves out, not a channel that runs on its own.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import __version__

FEEDBACK_REPO = os.environ.get("CODNA_FEEDBACK_REPO", "thyn-ai/feedback")
PRODUCTS = ("algenta", "codna", "telys", "sqai", "accounts", "docs")
_REQUEST_TIMEOUT_S = 15.0


def _local_fallback_dir() -> Path:
    """Honors CODNA_RUNTIME_ROOT like every other piece of codna's local state (the webhook
    queue, the local-stack state file) — a hardcoded ~/.codna here would be the one path in the
    runtime that can't be sandboxed or redirected the way everything else can."""
    root = os.environ.get("CODNA_RUNTIME_ROOT")
    base = Path(root).expanduser() if root else Path.home() / ".codna"
    return base / "reports"


@dataclass(frozen=True)
class ReportResult:
    ok: bool
    submitted: bool  # True: a real issue was filed. False: prefill URL / local file fallback.
    url: str
    local_path: str | None = None


def _github_token() -> str | None:
    """A token to file the issue AS THE USER — never the App's installation credentials, which
    have nothing to do with this command. `GITHUB_TOKEN`/`GH_TOKEN` first (CI/script-friendly),
    then `gh auth token` if the `gh` CLI is on PATH and already logged in (the common dev-machine
    case, so most developers need to type nothing extra)."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    if not shutil.which("gh"):
        return None
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False, timeout=10,
        )
    except OSError:
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


def _redacted_diagnostics() -> str:
    """Exactly what `codna doctor --json` prints — never a raw env dump, never a secret value."""
    try:
        from .doctor import build_report, format_public_output

        return format_public_output(build_report(), json_output=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics are a nice-to-have, never fatal to the report
        return json.dumps({"diagnostics_unavailable": str(exc)})


def normalize_product(product: str | None) -> str:
    value = (product or "").strip().lower()
    return value if value in PRODUCTS else "not sure"


def build_report_body(*, body: str, product: str, attach_diagnostics: bool) -> str:
    """The rendered body, shaped to match bug_report.yml's fields so a human reading it in
    thyn-ai/feedback sees the same structure whichever door the report came through."""
    parts = [
        f"### Which product?\n\n{normalize_product(product)}",
        f"### Version\n\ncodna {__version__}",
        f"### Platform\n\n{platform.platform()} / Python {platform.python_version()}",
        f"### What happened?\n\n{body.strip() or '(no description given)'}",
    ]
    if attach_diagnostics:
        parts.append(
            "### Diagnostics (redacted — presence/source only, never a secret value)\n\n"
            f"```json\n{_redacted_diagnostics()}\n```"
        )
    return "\n\n".join(parts)


def prefill_url(*, title: str, body: str) -> str:
    """Zero-auth fallback: GitHub issue forms accept field values via query params, so this works
    even for someone with no GITHUB_TOKEN and no `gh` login — they just have to click Submit."""
    query = urllib.parse.urlencode({"template": "bug_report.yml", "title": title, "what-happened": body})
    return f"https://github.com/{FEEDBACK_REPO}/issues/new?{query}"


def _write_local_fallback(*, title: str, body: str) -> Path:
    directory = _local_fallback_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{int(time.time())}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def submit_report(*, title: str, product: str, body: str, attach_diagnostics: bool) -> ReportResult:
    """Try to file a real issue; on ANY failure (no token, no network, GitHub unreachable), write
    the report locally and return a pre-filled URL instead. Always returns; never raises — the
    worst case for the caller is "paste this URL yourself", never "the report is gone"."""
    full_body = build_report_body(body=body, product=product, attach_diagnostics=attach_diagnostics)
    fallback_url = prefill_url(title=title, body=full_body)
    token = _github_token()

    if token:
        try:
            response = httpx.post(
                f"https://api.github.com/repos/{FEEDBACK_REPO}/issues",
                json={"title": title, "body": full_body},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "User-Agent": f"codna-cli/{__version__}",
                },
                timeout=_REQUEST_TIMEOUT_S,
            )
            if response.status_code == 201:
                return ReportResult(ok=True, submitted=True, url=response.json()["html_url"])
        except httpx.HTTPError:
            pass  # network/DNS/TLS failure (e.g. air-gapped) -- fall through to the local bundle

    local_path = None
    try:
        local_path = str(_write_local_fallback(title=title, body=full_body))
    except OSError:
        pass  # even the local write failing must not stop the URL from being printed
    return ReportResult(ok=True, submitted=False, url=fallback_url, local_path=local_path)


# ── the `codna report` command ───────────────────────────────────────────────────────────────────
def _prompt_for_body() -> str:
    print("Describe what happened (empty line to finish):")
    lines: list[str] = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def cmd_report(args) -> int:
    import sys

    product = normalize_product(getattr(args, "product", None))
    body = getattr(args, "body", None) or ""
    attach = bool(getattr(args, "attach_diagnostics", False))
    # Only prompt when nothing was given AND there is an actual human at the other end of stdin —
    # a script or an agent piping this command must never block waiting for input that never comes.
    if not body and sys.stdin.isatty():
        body = _prompt_for_body()

    if getattr(args, "dry_run", False):
        print(f"product: {product}")
        print(f"title  : {args.title}")
        print("body:")
        print(build_report_body(body=body, product=product, attach_diagnostics=attach))
        return 0

    result = submit_report(title=args.title, product=product, body=body, attach_diagnostics=attach)
    if result.submitted:
        print(f"filed: {result.url}")
        return 0

    print("could not file this automatically — open the link below to finish submitting it:")
    print(f"  {result.url}")
    if result.local_path:
        print(f"(also saved a copy at {result.local_path} in case the link expires or you're offline)")
    return 0


def register(sub) -> None:
    """Attach `codna report` to the top-level subparsers. Lives here, not in cli.py, for the same
    reason `byok_cli`/`ci_cli` do — cli.py is held under a hard line ceiling by `test_modularity`."""
    pr = sub.add_parser(
        "report",
        help="File a bug/feature/question to thyn-ai/feedback — the public front door for every product.",
    )
    pr.add_argument("title", help="short title, e.g. \"codna fix hangs on a monorepo\"")
    pr.add_argument("--product", choices=PRODUCTS, default="codna",
                    help="which product this is about (default: codna)")
    pr.add_argument("--body", default="", help="the description; omit to be prompted interactively")
    pr.add_argument("--attach-diagnostics", action="store_true",
                    help="attach the same redacted output as `codna doctor` (no secret values)")
    pr.add_argument("--dry-run", action="store_true",
                    help="print exactly what would be sent; submits nothing")
    pr.set_defaults(func=cmd_report)
