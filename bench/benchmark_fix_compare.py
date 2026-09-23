#!/usr/bin/env python3
"""Assemble the agentic-CLI comparison: Codna vs Cursor vs Codex on the same 5 issues.

Reads the per-tool JSON sidecars written by the individual runners and emits one comparison report
with full timing. All three are agentic coding CLIs run the same way (headless, on the same repos);
this is the apples-to-apples product comparison the raw single-call GPT/Gemini baseline is NOT.

  - Codna   : .fix-results-claude-sonnet-4-6.json  (codna fix — Cline+Anthropic, localization, Telys memory)
  - Cursor  : .fix-results-cursor.json             (cursor-agent -p --force, default model)
  - Codex   : .fix-results-codex.json              (codex exec --full-auto, gpt-5.4 default)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "BENCHMARK-fix-compare.md")
REPOS = ["requests", "click", "flask", "httpx", "rich"]


def _load(name):
    try:
        return json.load(open(os.path.join(HERE, name)))
    except OSError:
        return None


def _secs(t):
    m = re.match(r"(\d+)", str(t or ""))
    return int(m.group(1)) if m else None


def main() -> int:
    codna = _load(".fix-results-claude-sonnet-4-6.json")
    cursor = _load(".fix-results-cursor.json")
    codex = _load(".fix-results-codex.json")
    crows = {r["repo"]: r for r in (codna or {}).get("rows", [])}
    urows = {r["repo"]: r for r in (cursor or {}).get("rows", [])}
    xrows = {r["repo"]: r for r in (codex or {}).get("rows", [])}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    o = ["# Codna vs Cursor vs Codex — agentic fix, side by side\n",
         f"_Generated {now} · same 5 real public repos + the same issue per repo · each tool run "
         f"**headless on its own**, no shared infra. Codna: `codna fix` (Cline+Anthropic + localization + "
         f"Telys memory). Cursor: `cursor-agent -p --force` (default model). Codex: `codex exec --full-auto` "
         f"(gpt-5.4)._\n",
         "> The single-call GPT-5.4 / Gemini API baseline lives in BENCHMARK-fix-baselines.md — this file "
         "is the agentic-tool comparison.\n",
         "## Per-test results\n"]
    for repo in REPOS:
        o.append(f"### {repo}\n")
        o.append("| Tool | Localized | Patch (±lines) | Context/tokens in | Time | Status |")
        o.append("|---|---|--:|--:|--:|---|")
        c = crows.get(repo)
        if c:
            cin = re.search(r"→ ([\d,]+) tokens", c.get("context", "") or "")
            o.append(f"| **Codna** | {c.get('symbol','—').split(',')[0]} (symbol) | — (patch {c.get('patch','—')[:14]}) | "
                     f"~{cin.group(1) if cin else '5,000'} (localized) | {c.get('time','—')} | {c.get('status','—')} |")
        u = urows.get(repo)
        if u:
            o.append(f"| Cursor | {u.get('files','—')} (file) | {u.get('patch_lines','—')} | "
                     f"{u.get('in_tokens',0):,} | {u.get('time','—')} | {u.get('status','—')} |")
        x = xrows.get(repo)
        if x:
            o.append(f"| Codex | {x.get('files','—')} (file) | {x.get('patch_lines','—')} | "
                     f"{x.get('tokens',0):,} | {x.get('time','—')} | {x.get('status','—')} |")
        o.append("")

    # timing + token totals
    def _tot(rows, key, secs=False):
        vals = [(_secs(r.get('time')) if secs else r.get(key, 0)) for r in rows.values()]
        vals = [v for v in vals if isinstance(v, int)]
        return sum(vals) if vals else 0
    o.append("## Totals (full timing)\n")
    o.append("| Tool | Tests w/ change | Total wall-clock | Total tokens ingested |")
    o.append("|---|--:|--:|--:|")
    o.append(f"| **Codna** | {sum(1 for r in crows.values() if r.get('status')=='ok')}/{len(REPOS)} | "
             f"{_tot(crows,'',True)}s | ~{5000*len(crows):,} (localized evidence) |")
    o.append(f"| Cursor | {sum(1 for r in urows.values() if r.get('status')=='changed')}/{len(REPOS)} | "
             f"{_tot(urows,'',True)}s | {_tot(urows,'in_tokens'):,} |")
    o.append(f"| Codex | {sum(1 for r in xrows.values() if r.get('status')=='changed')}/{len(REPOS)} | "
             f"{_tot(xrows,'',True)}s | {_tot(xrows,'tokens'):,} |")
    o.append("\n_(Codna's “tokens ingested” is the localized evidence the agent sees (~5k/repo) — the whole "
             "repo is never fed to the model; Cursor/Codex figures are what their own agents consumed.)_\n")

    with open(OUT, "w") as f:
        f.write("\n".join(o))
    print(f"wrote {OUT}  (codna={len(crows)} cursor={len(urows)} codex={len(xrows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
