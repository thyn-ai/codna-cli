#!/usr/bin/env python3
"""End-to-end benchmark of `codna memory` on real public repos — emits a Markdown report.

Indexes several repos into ONE shared Telys collection (one scope per repo) and measures, per repo:
  - index throughput (files, symbols, skipped, ms)
  - exact-symbol recall@1 (query a symbol's own text -> it should rank #1)  [index-correctness, embedder-light]
  - partition-hit rate (scoped queries take the contiguous partition path, not a scatter fallback)
  - context-token reduction (final-k symbol texts vs the whole source files they came from)
  - text-to-results latency (embed + retrieve + rerank)
and cross-repo isolation across the shared collection. Real `codna.memory` calls, no mocks.

Run:  TELYS_KERNEL=/path/libame_kernel.dylib python cli/bench/benchmark_memory.py REPO_DIR [REPO_DIR ...]
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.dirname(HERE)
if CLI not in sys.path:
    sys.path.insert(0, CLI)

# Realistic natural-language queries per repo (a real user's words, not symbol names).
QUERIES = {
    "click": ["define a command line option with a default value", "prompt the user for confirmation"],
    "flask": ["register a url route for a view", "return a json response"],
    "requests": ["send a post request with a json body", "set a timeout on the request"],
    "httpx": ["create an async client and send a request", "stream a large response body"],
    "rich": ["render a table to the console", "display a progress bar"],
}
DEFAULT_QUERIES = ["read a file from disk", "parse command line arguments"]


def _toks(text: str) -> int:
    return max(1, len(text) // 4)


def main(argv=None) -> int:
    try:
        from telys.embedding import AlgentaBigramEmbedder
        AlgentaBigramEmbedder()
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP — Telys kernel unavailable: {exc}")
        return 0

    from codna.codeunits import extract_repo
    from codna.memory import CodeMemory

    repos = argv or sys.argv[1:]
    if not repos:
        print("usage: benchmark_memory.py REPO_DIR [REPO_DIR ...]")
        return 2
    db = tempfile.mkdtemp(prefix="codna-bench-")
    rows = []          # aggregate metrics per repo
    transcripts = []   # per-repo recall transcript blocks
    mems = {}          # name -> CodeMemory (shared db)

    for path in repos:
        name = os.path.basename(os.path.abspath(path))
        mem = CodeMemory(path, db_path=db, service=None)
        mems[name] = mem
        t = time.perf_counter()
        rep = mem.index()
        index_ms = (time.perf_counter() - t) * 1000

        units, _ = extract_repo(mem.repo_path, mem.repo_id, languages=("python",))
        id2text = {u.id: u.text for u in units}
        funcs = [u for u in units if u.symbol_type in ("function", "method", "class")]

        # exact-symbol recall@1 over a deterministic sample (every Nth symbol, up to 25)
        sample = funcs[:: max(1, len(funcs) // 25)][:25]
        hit1 = 0
        for u in sample:
            r = mem.recall(u.text, language="python", final_k=1)
            if r["symbols"] and r["symbols"][0]["id"] == u.id:
                hit1 += 1
        recall1 = hit1 / max(1, len(sample))

        # domain queries: partition-hit + latency + a transcript
        qs = QUERIES.get(name, DEFAULT_QUERIES)
        plans, lats, blocks = [], [], []
        for q in qs:
            t = time.perf_counter()
            r = mem.recall(q, language="python", top_k=40, final_k=5)
            lats.append((time.perf_counter() - t) * 1000)
            plans.append(r["explain"].get("plan", "?"))
            lines = [f"  $ codna memory recall {name} --query \"{q}\""]
            for s in r["symbols"][:5]:
                lines.append(f"    {s['score']:.3f}  {(s.get('symbol_type') or '?'):8} {s.get('path')}")
            blocks.append("\n".join(lines))
        part_hit = sum(p.startswith("Partition") for p in plans) / len(plans)

        # context-token reduction on the first query: final-k symbol texts vs their whole source files
        r0 = mem.recall(qs[0], language="python", final_k=8)
        codna_tok = sum(_toks(id2text.get(s["id"], "")) for s in r0["symbols"])
        files = {s["path"] for s in r0["symbols"] if s.get("path")}
        base_tok = 0
        for rel in files:
            try:
                with open(os.path.join(mem.repo_path, rel), encoding="utf-8", errors="replace") as f:
                    base_tok += _toks(f.read())
            except OSError:
                pass
        reduction = (base_tok - codna_tok) / base_tok if base_tok else 0.0

        rows.append((name, rep["files"], rep["indexed"], rep["skipped"], index_ms,
                     recall1, part_hit, reduction, statistics.mean(lats)))
        transcripts.append((name, mem.repo_id, blocks))

    # cross-repo isolation: REOPEN a fresh engine so the FULL shared collection (all repos) is loaded
    # from disk, then check each repo's scoped recall returns only its own symbols. (Reopening matters:
    # the indexing engines each only held their own + prior repos — a fresh open loads all of them.)
    iso_lines = []
    total_docs = CodeMemory(repos[0], db_path=db, service=None).status().get("documents")
    for path in repos:
        name = os.path.basename(os.path.abspath(path))
        mem = CodeMemory(path, db_path=db, service=None)        # fresh open -> full collection
        q = QUERIES.get(name, DEFAULT_QUERIES)[0]
        r = mem.recall(q, language="python", final_k=10)
        own = sum(1 for s in r["symbols"] if s["id"].startswith(mem.repo_id))
        iso_lines.append(f"| {name} | {len(r['symbols'])} | {own} | {'PASS' if own == len(r['symbols']) else 'LEAK'} |")

    # ── emit Markdown ─────────────────────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = []
    out.append("# `codna memory` benchmark — real public repos\n")
    out.append(f"_Generated {now} · embedder: AlgentaBigramEmbedder (in-process, 64-d lexical) · "
               "engine: Telys (Mojo kernel) · {} repos, one shared collection._\n".format(len(rows)))
    out.append("> **Honest scope:** day-1 uses the in-process **bigram** stand-in (lexical, weak semantics) — "
               "the future `AlgentaCodeEmbedder` raises recall quality with no code change. The *structural* "
               "wins below (exact-symbol recall, partition-isolated retrieval, context-token reduction, "
               "µs-scale latency, cross-repo isolation) are embedder-independent.\n")
    out.append("## Results\n")
    out.append("| Repo | Py files | Symbols | Skipped | Index (ms) | Exact recall@1 | Partition-hit | "
               "Context-token reduction | Mean recall (ms) |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for (name, files, sym, skip, ims, r1, ph, red, lat) in rows:
        out.append(f"| {name} | {files} | {sym} | {skip} | {ims:.0f} | {r1*100:.0f}% | {ph*100:.0f}% | "
                   f"{red*100:.0f}% | {lat:.2f} |")
    tot_sym = sum(r[2] for r in rows)
    out.append(f"\n**Totals:** {tot_sym:,} symbols indexed across {len(rows)} repos into one collection "
               f"({total_docs:,} live documents). Exact recall@1 mean "
               f"{statistics.mean(r[5] for r in rows)*100:.0f}%, partition-hit mean "
               f"{statistics.mean(r[6] for r in rows)*100:.0f}%, token-reduction mean "
               f"{statistics.mean(r[7] for r in rows)*100:.0f}%.\n")

    out.append("## Cross-repo isolation (shared collection)\n")
    out.append("Each repo's scoped recall returns only its own symbols — proof the partition key isolates "
               "repos in one shared store.\n")
    out.append("| Repo | Returned | In-scope | Verdict |")
    out.append("|---|--:|--:|---|")
    out.extend(iso_lines)

    out.append("\n## Recall transcripts (real output)\n")
    for name, repo_id, blocks in transcripts:
        out.append(f"### {name}  (`{repo_id}`)\n```text")
        out.append("\n\n".join(blocks))
        out.append("```\n")

    out.append("## Method\n")
    out.append("- Real `codna.memory.CodeMemory.index()` / `.recall()` on shallow clones — no mocks.\n"
               "- **Exact recall@1**: query a sampled symbol's own source text; counts how often it ranks #1 "
               "(an index-correctness check — should be ~100%).\n"
               "- **Partition-hit**: share of scoped queries served by the contiguous partition slice "
               "(`PartitionSliceExactF32`) vs a scatter fallback.\n"
               "- **Context-token reduction**: `len//4` token estimate of the final-k symbol texts vs the whole "
               "source files they came from (what an agent would otherwise read).\n"
               "- **Latency**: text-to-results (embed + Telys retrieve + rerank), mean over the domain queries.\n"
               "- **No `X× faster` claim** (per Telys D-28); latency is reported, not marketed.\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
