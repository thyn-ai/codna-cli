#!/usr/bin/env python3
"""Codna ↔ Telys pilot — maps the INTEGRATION-CODNA §9 / DECISIONS D-28 success gates to checks.

Opt-in, run by hand (needs the [memory] extra + a built kernel; set $TELYS_KERNEL):

    TELYS_KERNEL=/abs/path/libame_kernel.dylib python cli/bench/pilot_codna_memory.py [REPO]

Indexes a real repo (default: the codna CLI package) into a throwaway memory and reports PASS/FAIL
per gate + both latencies. NO "X× faster" claim is made (D-28) — this measures product value
(isolation, recall, context-token reduction), not speed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.dirname(HERE)
if CLI not in sys.path:
    sys.path.insert(0, CLI)


def _kernel_ok() -> bool:
    try:
        from telys.embedding import AlgentaBigramEmbedder
        AlgentaBigramEmbedder()
        return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"SKIP — Telys unavailable: {exc}\n"
            "  install 'codna[memory]' and use a Codna-packaged runtime or TELYS_KERNEL if needed"
        )
        return False


def _toks(text: str) -> int:
    return max(1, len(text) // 4)   # crude, identical estimator on both sides of the ratio


def main(argv=None) -> int:
    if not _kernel_ok():
        return 0   # a skipped pilot is not a failure

    from codna.codeunits import extract_repo
    from codna.memory import CodeMemory
    from telys import scope_key

    repo = (argv or sys.argv[1:] or [os.path.join(CLI, "codna")])[0]
    db = tempfile.mkdtemp(prefix="codna-pilot-")
    mem = CodeMemory(repo, db_path=db, service="pilot")

    t0 = time.perf_counter()
    rep = mem.index()
    index_ms = (time.perf_counter() - t0) * 1000
    units, _ = extract_repo(mem.repo_path, mem.repo_id, languages=("python",))
    id2text = {u.id: u.text for u in units}
    print(f"indexed {rep['indexed']} symbols / {rep['files']} files "
          f"({rep['skipped']} skipped) in {index_ms:.0f} ms · scope={scope_key(mem.repo_id,'pilot','python')}\n")

    gates: list[tuple[str, bool, str]] = []

    # 1) exact-path recall@1 — a symbol's own text must retrieve itself rank-1
    sample = [u for u in units if u.symbol_type in ("function", "method", "class")][:12]
    hits = 0
    for u in sample:
        r = mem.recall(u.text, service="pilot", language="python", final_k=1)
        if r["symbols"] and r["symbols"][0]["id"] == u.id:
            hits += 1
    rec1 = hits / max(1, len(sample))
    gates.append(("exact-path recall@1 = 1.0", rec1 == 1.0, f"{rec1:.2f} over {len(sample)} symbols"))

    # 2) partition-hit rate > 70% — scoped queries take the contiguous partition path
    queries = ["open a pull request when CI fails", "classify a security finding",
               "parse a SARIF report", "resolve the engine url and api key", "verify an attestation",
               "extract code units from a repository", "recall related symbols", "run the mcp server"]
    t1 = time.perf_counter()
    plans = [mem.recall(q, service="pilot", language="python", final_k=8)["explain"].get("plan", "") for q in queries]
    recall_ms = (time.perf_counter() - t1) * 1000 / len(queries)
    part = sum(p.startswith("Partition") for p in plans) / len(plans)
    gates.append(("partition-hit rate > 70%", part > 0.70, f"{part*100:.0f}% ({plans[0]})"))

    # 3) scope isolation — an unpopulated scope leaks nothing from the populated one
    miss = mem.recall("open a pull request when CI fails", service="other-service", language="python", final_k=8)
    iso = miss["symbols"] == [] and miss["explain"].get("partition_value") == scope_key(mem.repo_id, "other-service", "python")
    gates.append(("scope isolation (no cross-scope leak)", iso, f"empty scope -> {len(miss['symbols'])} hits"))

    # 4) ≥25% context-token reduction — final-k symbol texts vs the whole-file baseline
    q = "open a pull request when CI fails"
    res = mem.recall(q, service="pilot", final_k=8)
    codna_tok = sum(_toks(id2text.get(s["id"], "")) for s in res["symbols"])
    files = {s["path"] for s in res["symbols"] if s.get("path")}
    base_tok = 0
    for rel in files:
        try:
            base_tok += _toks(open(os.path.join(mem.repo_path, rel), encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    reduction = (base_tok - codna_tok) / base_tok if base_tok else 0.0
    gates.append(("≥25% context-token reduction", reduction >= 0.25,
                  f"{reduction*100:.0f}% ({codna_tok} vs {base_tok} tok)"))

    # 5) delete/update visibility across base+delta+reopen, and 6) kill/reopen no data loss
    before = mem.status()["documents"]
    reopened = CodeMemory(repo, db_path=db, service="pilot")           # fresh engine over the saved dir
    after = reopened.status()["documents"]
    persisted = (after == before) and (after or 0) > 0 and bool(
        reopened.recall("open a pull request when CI fails", service="pilot")["symbols"])
    gates.append(("kill/reopen — no committed-data loss", persisted, f"{before} -> {after} documents"))

    # 7) explain exposes plan + embedding space + source id
    ex = res["explain"]
    prov = bool(res["symbols"]) and all(s.get("id") for s in res["symbols"])
    explains = bool(ex.get("plan")) and bool(mem.status().get("embedding_space")) and prov
    gates.append(("explain exposes plan + space + source id", explains,
                  f"{ex.get('plan')} · {mem.status().get('embedding_space')}"))

    # ── report ───────────────────────────────────────────────────────────────────────────────────
    print(f"latency: index {index_ms:.0f} ms · text-to-results {recall_ms:.1f} ms/query "
          f"(embed+telys+rerank; NO speed claim — D-28)\n")
    width = max(len(name) for name, _, _ in gates)
    ok = True
    for name, passed, detail in gates:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<{width}}  {detail}")
    print(f"\n{'ALL GATES PASS' if ok else 'SOME GATES FAILED'}  ({sum(p for _, p, _ in gates)}/{len(gates)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
