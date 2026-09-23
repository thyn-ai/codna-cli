#!/usr/bin/env python3
"""Codex baseline — OpenAI's agentic coding CLI, run SEPARATELY (not through codna's infra).

Mirror of benchmark_cursor.py for `codex exec` (headless). For each issue, copies the repo to a
throwaway dir and runs `codex exec "<issue>" --full-auto` — Codex browses, localizes, and edits on
its own (default model gpt-5.4). Captures, per repo: the files Codex changed (from `git diff`),
tokens used, wall-clock, and patch size. Apples-to-apples with codna + Cursor (all agentic CLIs).

Env: codex must be logged in (~/.codex/auth.json) or OPENAI_API_KEY set. Reads codna's localization
from .fix-results-claude-sonnet-4-6.json for the comparison.
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
OUT_JSON = os.path.join(HERE, ".fix-results-codex.json")
PER_REPO_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT_S", "900"))

CASES = [
    ("requests", "Session.send does not retry when the underlying connection times out; add a retry-on-timeout path"),
    ("click", "a required option in a command group does not raise a clear error when omitted"),
    ("flask", "send_file returns the wrong content-type for .webp files"),
    ("httpx", "the connection pool is not released when a streaming response is closed early"),
    ("rich", "Table column width is miscalculated when a cell contains wide (CJK) unicode characters"),
]


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout


def _codex_bin() -> str:
    """Absolute path of the codex executable: ``$CODEX_BIN`` or ``codex`` on PATH.

    Validated when the benchmark runs (allowlisted name on PATH, or an absolute existing
    executable), never at import, so the module stays importable on a machine without codex;
    a missing or unvetted binary still fails clearly before any repo is copied.
    """
    return resolve_bin("CODEX_BIN", "codex", allowed=("codex",))


def main() -> int:
    codex = _codex_bin()
    model = model_id("CODEX_MODEL")  # validated identifier; empty = Codex default (gpt-5.4)
    base = os.environ.get("BENCH_REPOS", "/tmp/bench-repos")
    rows = []
    for repo, issue in CASES:
        rdir = os.path.join(base, repo)
        if not os.path.isdir(rdir):
            continue
        work = tempfile.mkdtemp(prefix=f"codex-{repo}-")
        dst = os.path.join(work, repo)
        shutil.copytree(rdir, dst, symlinks=True)
        _git(dst, "stash", "-u")
        prompt = (f"Fix this bug with the minimal code change. Do not ask questions — locate the "
                  f"responsible function and make the edit. BUG: {issue}")
        cmd = [codex, "exec", prompt, "--full-auto", "-C", dst]
        if model:
            cmd += ["--model", model]
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_REPO_TIMEOUT)
            out = p.stdout + p.stderr
            status_run = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            out, status_run = "", "timeout"
        wall = time.perf_counter() - t
        tok = re.search(r"tokens used\s+([\d,]+)", out)
        model_used = re.search(r"^model:\s*(\S+)", out, re.MULTILINE)
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
            "tokens": int(tok.group(1).replace(",", "")) if tok else 0,
            "model": model_used.group(1) if model_used else (model or "default"),
            "time": f"{wall:.0f}s",
            "status": status_run if status_run != "ok" else ("changed" if changed else "no-change"),
        })
        print(f"  {repo}: {rows[-1]['status']}  files={changed or '—'}  tokens={rows[-1]['tokens']}  {wall:.0f}s")
        shutil.rmtree(work, ignore_errors=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"model_label": f"codex ({model or 'gpt-5.4 default'})",
                       "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "rows": rows}, f, indent=2)
    print(f"done -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
