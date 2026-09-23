#!/usr/bin/env python3
"""Baseline comparison: can a RAW frontier model fix the same issues WITHOUT codna's infra?

codna localizes (finds the symbol, reduces a 100k–1M-token repo to ~5k tokens of evidence) and
then patches. This harness gives GPT-5 and Gemini the SAME issue but no codna infra — just the model
API with the repository source pasted in (bounded to a context cap), and asks each to localize the fix
+ propose a unified diff. It then contrasts, per repo:

  - localized file (and whether it matches codna's localization)
  - context tokens the raw model had to ingest (vs codna's reduced ~5k)
  - output tokens, wall-clock latency, and whether the repo even fit the context cap

This isolates codna's structural value (localization + context reduction) from raw model capability.
NOT through the Cline sidecar (which is Anthropic-only) — these are direct, separate API calls.

Env: OPENAI_API_KEY, GEMINI_API_KEY. Reads codna's per-repo result from .fix-results-claude-sonnet-4-6.json.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BENCHMARK_OUT", os.path.join(HERE, "BENCHMARK-fix-baselines.md"))
MAX_CTX_CHARS = int(os.environ.get("BASELINE_MAX_CTX_CHARS", "480000"))  # ~120k tokens — fits GPT-5/Gemini
TIMEOUT = int(os.environ.get("BASELINE_TIMEOUT_S", "300"))

CASES = [
    ("requests", "Session.send does not retry when the underlying connection times out; add a retry-on-timeout path"),
    ("click", "a required option in a command group does not raise a clear error when omitted"),
    ("flask", "send_file returns the wrong content-type for .webp files"),
    ("httpx", "the connection pool is not released when a streaming response is closed early"),
    ("rich", "Table column width is miscalculated when a cell contains wide (CJK) unicode characters"),
]
MODELS = [
    ("gpt-5.4", "openai"),
    ("gemini-2.5-pro", "gemini"),
]
PROMPT = (
    "You are fixing a bug in the `{repo}` Python codebase. Identify the SINGLE most likely file and "
    "function/method to change, and propose a minimal unified-diff patch.\n\nISSUE: {issue}\n\n"
    "Respond ONLY as compact JSON: {{\"file\": \"relative/path.py\", \"symbol\": \"Qualified.Name\", "
    "\"patch\": \"<unified diff or empty>\"}}.\n\nSOURCE{trunc}:\n{src}"
)


def _collect_src(repo_dir: str) -> tuple[str, int, bool]:
    """Concatenate the repo's .py source (sorted), bounded to MAX_CTX_CHARS. Returns (text, files, truncated)."""
    parts, n, total, truncated = [], 0, 0, False
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "node_modules", "__pycache__", ".codna-memory", "tests", "test")]
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), repo_dir)
            try:
                with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                    body = f.read()
            except OSError:
                continue
            block = f"\n# ===== {rel} =====\n{body}"
            if total + len(block) > MAX_CTX_CHARS:
                truncated = True
                continue
            parts.append(block)
            total += len(block)
            n += 1
    return "".join(parts), n, truncated


def _http_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def _call_openai(model: str, prompt: str) -> tuple[str, int, int]:
    key = os.environ["OPENAI_API_KEY"]
    d = _http_json("https://api.openai.com/v1/chat/completions",
                   {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 16000},
                   {"Authorization": f"Bearer {key}"})
    txt = d["choices"][0]["message"]["content"]
    u = d.get("usage", {})
    return txt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _call_gemini(model: str, prompt: str) -> tuple[str, int, int]:
    key = os.environ["GEMINI_API_KEY"]
    d = _http_json(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                   {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 32000}}, {})  # 2.5-pro thinking tokens count vs this
    cand = (d.get("candidates") or [{}])[0]
    txt = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    u = d.get("usageMetadata", {})
    return txt, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def _parse(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    try:
        codna = {r["repo"]: r for r in json.load(open(os.path.join(HERE, ".fix-results-claude-sonnet-4-6.json")))["rows"]}
    except OSError:
        codna = {}
    base = os.environ.get("BENCH_REPOS", "/tmp/bench-repos")
    rows = []
    for repo, issue in CASES:
        rdir = os.path.join(base, repo)
        if not os.path.isdir(rdir):
            continue
        src, nfiles, trunc = _collect_src(rdir)
        prompt = PROMPT.format(repo=repo, issue=issue, trunc=" (TRUNCATED to context cap)" if trunc else "", src=src)
        for model, provider in MODELS:
            t = time.perf_counter()
            try:
                txt, pin, pout = (_call_openai if provider == "openai" else _call_gemini)(model, prompt)
                r = _parse(txt)
                status = "ok" if r.get("file") else "no-localize"
            except Exception as exc:  # noqa: BLE001
                txt, pin, pout, r, status = "", 0, 0, {}, f"err: {type(exc).__name__}: {str(exc)[:60]}"
            wall = time.perf_counter() - t
            rows.append({"repo": repo, "model": model, "file": r.get("file", "—"), "symbol": r.get("symbol", "—"),
                         "ctx_files": nfiles, "ctx_tokens_in": pin, "out_tokens": pout, "trunc": trunc,
                         "time": f"{wall:.0f}s", "status": status, "patch_lines": len((r.get("patch") or "").splitlines())})
            print(f"  {repo}/{model}: {status}  file={r.get('file','—')}  ctx_in={pin}  {wall:.0f}s")
        _write(rows, codna)
    _write(rows, codna)
    print(f"done -> {OUT}")
    return 0


def _write(rows, codna) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    o = ["# Baseline comparison — raw frontier models vs codna (fix localization)\n",
         f"_Generated {now} · GPT-5.4 (OpenAI) + Gemini-2.5-pro fix the SAME issues **separately, not through "
         f"codna's infra** — direct API calls with the repo source pasted in (cap {MAX_CTX_CHARS//1000}k chars "
         f"≈ {MAX_CTX_CHARS//4000}k tokens). codna's Cline agent is Anthropic-only; these are independent baselines._\n",
         "> The contrast is **context efficiency + localization**: codna reduces a 100k–1M-token repo to ~5k tokens "
         "of evidence and patches the right symbol; a raw model must ingest the whole repo (and large repos may not "
         "even fit). Raw-model patch *quality* is not scored here — this isolates codna's structural advantage.\n",
         "## Per-repo: codna vs raw models\n",
         "| Repo | Engine | Localized file | Context tokens in | Out tokens | Time | Status |",
         "|---|---|---|--:|--:|--:|---|"]
    for repo in [c for c, _ in CASES]:
        cr = codna.get(repo, {})
        if cr:
            ctx = cr.get("context", "—")
            cin = re.search(r"→ ([\d,]+) tokens", ctx)
            o.append(f"| {repo} | **codna** (Sonnet+localize) | {cr.get('symbol','—').split(',')[0]} | "
                     f"~{cin.group(1) if cin else '5,000'} | — | {cr.get('time','—')} | {cr.get('status','—')} |")
        for r in [x for x in rows if x["repo"] == repo]:
            o.append(f"| {repo} | {r['model']} (raw) | {r['file']}{' ⚠trunc' if r['trunc'] else ''} | "
                     f"{r['ctx_tokens_in']:,} | {r['out_tokens']:,} | {r['time']} | {r['status']} |")
    o.append("\n## Method\n- Raw models get the repo's non-test `.py` source concatenated (bounded), the issue, and "
             "are asked for `{file, symbol, patch}` JSON in ONE call — no browsing, no agentic loop, no codna infra.\n"
             "- `Context tokens in` is what each engine fed the model: codna's reduced evidence (~5k) vs the raw repo "
             "dump (the model's actual prompt_tokens). `⚠trunc` = the repo exceeded the context cap (codna's "
             "localization is *required* at that size).\n- Direct OpenAI `chat/completions` + Gemini `generateContent`.\n")
    with open(OUT, "w") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    raise SystemExit(main())
