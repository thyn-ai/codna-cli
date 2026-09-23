#!/usr/bin/env python3
"""Cursor baseline — the agentic competitor, run SEPARATELY (not through codna's infra).

For each issue, copies the repo to a throwaway dir and runs the headless Cursor CLI agent
(`cursor-agent -p … --force`) — Cursor browses, localizes, and edits on its own, exactly as a
user would. Captures, per repo: the files Cursor changed (from `git diff`), whether it touched the
same file codna localized, Cursor's input/output tokens + wall-clock, and patch size.

This is the apples-to-apples agentic comparison (codna's Cline+localization vs Cursor's agent),
distinct from the raw single-call GPT/Gemini baselines.

Env: CURSOR_API_KEY. Reads codna's localization from .fix-results-claude-sonnet-4-6.json.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from bench_env import model_id, resolve_bin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, ".fix-results-cursor.json")
PER_REPO_TIMEOUT = int(os.environ.get("CURSOR_TIMEOUT_S", "900"))

CASES = [
    ("requests", "Session.send does not retry when the underlying connection times out; add a retry-on-timeout path"),
    ("click", "a required option in a command group does not raise a clear error when omitted"),
    ("flask", "send_file returns the wrong content-type for .webp files"),
    ("httpx", "the connection pool is not released when a streaming response is closed early"),
    ("rich", "Table column width is miscalculated when a cell contains wide (CJK) unicode characters"),
]


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout


def _cursor_bin() -> str:
    """Absolute path of the cursor-agent executable: ``$CURSOR_BIN`` or ``~/.local/bin/cursor-agent``.

    Validated when the benchmark runs (allowlisted name on PATH, or an absolute existing
    executable), never at import, so the module stays importable on a machine without Cursor;
    a missing or unvetted binary still fails clearly before any repo is copied.
    """
    return resolve_bin("CURSOR_BIN", "~/.local/bin/cursor-agent", allowed=("cursor-agent",))


def main() -> int:
    cursor = _cursor_bin()
    model = model_id("CURSOR_MODEL")  # validated identifier; empty = Cursor's default
    base = os.environ.get("BENCH_REPOS", "/tmp/bench-repos")
    # cursor-agent reads CURSOR_API_KEY from its environment (inherited below); it is never
    # placed on the command line, where `ps` would expose it.
    if not os.environ.get("CURSOR_API_KEY"):
        print("CURSOR_API_KEY not set")
        return 2
    rows = []
    for repo, issue in CASES:
        rdir = os.path.join(base, repo)
        if not os.path.isdir(rdir):
            continue
        work = tempfile.mkdtemp(prefix=f"cursor-{repo}-")
        dst = os.path.join(work, repo)
        shutil.copytree(rdir, dst, symlinks=True)
        _git(dst, "stash", "-u")              # clean tree baseline (drop any prior edits)
        prompt = (f"Fix this bug with the minimal code change. Do not ask questions — locate the "
                  f"responsible function and make the edit. BUG: {issue}")
        cmd = [cursor, "-p", prompt, "--output-format", "json", "--force"]
        if model:
            cmd += ["--model", model]
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd, cwd=dst, capture_output=True, text=True, timeout=PER_REPO_TIMEOUT)
            out = p.stdout.strip()
            status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            out, status = "", "timeout"
        wall = time.perf_counter() - t
        # parse cursor JSON (last JSON object on stdout)
        meta = {}
        m = re.findall(r"\{.*\}", out, re.DOTALL)
        if m:
            try:
                meta = json.loads(m[-1])
            except json.JSONDecodeError:
                meta = {}
        usage = meta.get("usage", {})
        diff = _git(dst, "diff")
        changed = [ln.split("/", 1)[-1] for ln in _git(dst, "diff", "--name-only").splitlines()]
        rows.append({
            "repo": repo,
            "files": ", ".join(changed) or "—",
            "n_files": len(changed),
            "patch_lines": len(
                [
                    line
                    for line in diff.splitlines()
                    if line[:1] in "+-" and line[:3] not in ("+++", "---")
                ]
            ),
            "in_tokens": usage.get("inputTokens", 0),
            "out_tokens": usage.get("outputTokens", 0),
            "cache_read": usage.get("cacheReadTokens", 0),
            "time": f"{wall:.0f}s",
            "cursor_duration_ms": meta.get("duration_ms"),
            "is_error": meta.get("is_error"),
            "status": status if status != "ok" else ("changed" if changed else "no-change"),
        })
        print(f"  {repo}: {rows[-1]['status']}  files={changed or '—'}  in={usage.get('inputTokens',0)}  {wall:.0f}s")
        shutil.rmtree(work, ignore_errors=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"model_label": f"cursor ({model or 'default'})",
                       "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "rows": rows}, f, indent=2)
    print(f"done -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
