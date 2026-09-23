#!/usr/bin/env python3
"""A–Z fix benchmark across the top languages: Codna (engine-behind) vs public Cline (no engine) vs
Cursor. Same scoped bug per repo for every engine; PER-REPO streaming + a final summary.

Flexible: pick competitors with BENCH_ENGINES (default "codna,cline,cursor") — drop any to disable it,
e.g. BENCH_ENGINES=codna,cursor. Each engine also auto-skips (noted, not crashing) if its creds are
absent. Auth is read from the environment (source your .env before running); no keys are stored here.

  set -a; . /path/to/.env; set +a
  BENCH_ENGINES=codna,cline,cursor python cli/bench/bench_langs.py
"""
import os
import re
import sys
import tempfile
import datetime
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_suite as B  # noqa: E402

ENGINES = [e.strip() for e in os.environ.get("BENCH_ENGINES", "codna,cline,cursor").split(",")
           if e.strip() in B.RUNNERS]
REF = os.environ.get("BENCH_REF", "")  # "" = each repo's default branch

# 10 well-known public repos across the top languages; one scoped, plausible bug each (identical task
# for every engine). Override the set by editing here — repo-agnostic, no hardcoded language logic.
MULTILANG = [
    ("python", "https://github.com/psf/requests",
     "Session.send does not retry when the underlying connection times out; add a retry-on-timeout path"),
    ("typescript", "https://github.com/date-fns/date-fns",
     "format() with the SSS millisecond token drops a leading zero for sub-100ms values"),
    ("go", "https://github.com/gin-gonic/gin",
     "Context.ShouldBindQuery ignores the `default` struct tag for a missing query parameter"),
    ("rust", "https://github.com/BurntSushi/ripgrep",
     "--count-matches double-counts overlapping matches on a single line"),
    ("java", "https://github.com/google/gson",
     "the default Date type adapter fails to parse an ISO-8601 timestamp with a trailing Z zone"),
    ("cpp", "https://github.com/nlohmann/json",
     "parse() rejects an otherwise-valid document that begins with a UTF-8 BOM"),
    ("csharp", "https://github.com/JamesNK/Newtonsoft.Json",
     "serializing a DateTimeOffset with a negative UTC offset emits the wrong offset sign"),
    ("ruby", "https://github.com/sinatra/sinatra",
     "a route with an optional named parameter fails to match when the parameter is omitted"),
    ("php", "https://github.com/guzzle/guzzle",
     "query params with array values are encoded without the [] suffix"),
    ("swift", "https://github.com/Alamofire/Alamofire",
     "URLEncoding does not percent-encode a literal + character in query values"),
]
# BENCH_SKIP: comma-separated substrings matched against the repo URL — drop those repos for THIS run
# (default empty → run the full fixed set; keeps reproducibility, lets you exclude a repo you own).
_SKIP = [s.strip() for s in os.environ.get("BENCH_SKIP", "").split(",") if s.strip()]
if _SKIP:
    MULTILANG = [m for m in MULTILANG if not any(s in m[1] for s in _SKIP)]
MULTILANG = MULTILANG[:int(os.environ.get("BENCH_LIMIT", str(len(MULTILANG))))]  # smoke a subset first

# ── Anomaly log: durable record of failures + suspicious-but-ok signals across runs ──────────────
# Every non-ok engine row, plus Codna localization misses (Telys "0 recalled") and latency outliers,
# get appended here so a flake/regression is tracked, not just scrolled past in the stream.
ANOMALY_LOG = os.environ.get(
    "BENCH_ANOMALY_LOG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ANOMALIES.md"),
)
SLOW_S = int(os.environ.get("BENCH_SLOW_S", "240"))  # wall-clock outlier threshold


def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _anomalies_for(engine, r):
    """List of (severity, kind, detail) anomalies for one engine's result row."""
    out = []
    status = r.get("status")
    if status not in ("ok", "skipped"):
        note = (r.get("notes") or "").strip().replace("\n", " ")[:140]
        out.append(("FAIL", f"{engine} status={status}",
                    f"time={r.get('time_s')}s · {note or 'no captured reason'}"))
    mem = str(r.get("memory") or "")
    m = re.search(r"(\d+)\s+recalled", mem)  # exact count — avoid "10 recalled" matching "0 recalled"
    if engine.startswith("codna") and m and int(m.group(1)) == 0:
        out.append(("WARN", f"{engine} zero-recall",
                    f"Telys[{mem}] — indexed symbols but recalled 0 for the issue (localization miss → "
                    f"agent runs context-starved)"))
    t = r.get("time_s")
    if isinstance(t, (int, float)) and t >= SLOW_S:
        out.append(("WARN", f"{engine} slow", f"{t}s wall-clock (>= {SLOW_S}s)"))
    return out


def _init_anomaly_log(engines):
    new = not os.path.exists(ANOMALY_LOG)
    with open(ANOMALY_LOG, "a") as fh:
        if new:
            fh.write("# Codna benchmark — anomaly log\n\n"
                     "Auto-appended by `bench_langs.py`. FAIL = engine produced no patch / errored; "
                     "WARN = produced a patch but with a degraded signal (zero localization recall, "
                     "latency outlier).\n")
        fh.write(f"\n## Run {_ts()} · engines: {','.join(engines)} · {len(MULTILANG)} repos\n\n"
                 "| time | repo# | lang | repo | severity | kind | detail |\n"
                 "|---|---|---|---|---|---|---|\n")


def _log_anomalies(i, lang, name, engines, rows):
    entries = [(e, a) for e, r in zip(engines, rows) for a in _anomalies_for(e, r)]
    if not entries:
        return 0
    with open(ANOMALY_LOG, "a") as fh:
        for _e, (sev, kind, detail) in entries:
            fh.write(f"| {_ts()} | {i}/{len(MULTILANG)} | {lang} | {name} | {sev} | {kind} | {detail} |\n")
    return len(entries)


def _kt(x):
    return f"{x/1000:.1f}k" if isinstance(x, (int, float)) else "—"


def _cell(r):
    c = r.get("cost_usd")
    cstr = f"${c:.3f}~" if isinstance(c, (int, float)) else "—"          # trailing ~ = ESTIMATED ($)
    mem = r.get("memory")                                                # Telys index/recall status (codna)
    memstr = f"  Telys[{mem}]" if mem else ""
    return (f"{r.get('status', '?'):>7} {str(r.get('time_s', '?')):>4}s  "
            f"in {_kt(r.get('context_in')):>6}/out {_kt(r.get('out_tokens')):>5}  {cstr:>9}{memstr}")


def main():
    if not ENGINES:
        sys.exit("no valid engines (choose from: " + ", ".join(B.RUNNERS) + ")")
    print(f"=== A-Z fix benchmark · {len(MULTILANG)} repos · engines: {ENGINES} ===", flush=True)
    print("    (per-repo as it lands; status 'ok' = produced a patch)\n", flush=True)
    _init_anomaly_log(ENGINES)
    acc = {e: [] for e in ENGINES}
    total_anom = 0
    with tempfile.TemporaryDirectory(prefix="benchlang-") as wd:
        for i, (lang, repo, issue) in enumerate(MULTILANG, 1):
            try:
                name, rows = B._bench_one(repo, issue, ENGINES, REF, wd)
            except Exception as exc:  # never let one repo kill the run
                print(f"[{i:2}/{len(MULTILANG)}] {lang:11}{repo.split('/')[-1]:18} ERROR {str(exc)[:50]}", flush=True)
                continue
            line = "  |  ".join(f"{e} {_cell(r)}" for e, r in zip(ENGINES, rows))
            print(f"[{i:2}/{len(MULTILANG)}] {lang:11}{name:18}  {line}", flush=True)
            n_anom = _log_anomalies(i, lang, name, ENGINES, rows)
            total_anom += n_anom
            if n_anom:
                print(f"         ⚠ {n_anom} anomaly(ies) logged → {ANOMALY_LOG}", flush=True)
            for e, r in zip(ENGINES, rows):
                acc[e].append(r)

    print("\n=== SUMMARY ===", flush=True)
    for e in ENGINES:
        rows = acc[e]
        ok = [r for r in rows if r.get("status") == "ok"]
        skipped = [r for r in rows if r.get("status") == "skipped"]
        times = [r["time_s"] for r in ok if isinstance(r.get("time_s"), (int, float))]
        costs = [r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float))]
        avg_t = round(st.mean(times)) if times else "—"
        tot_c = round(sum(costs), 3) if costs else "—"
        note = f" ({len(skipped)} skipped — creds?)" if skipped else ""
        print(f"  {e:8} patched {len(ok)}/{len(rows)}  avg {avg_t}s  total ${tot_c}{note}", flush=True)
    print(f"\n  anomalies this run: {total_anom}"
          + (f"  →  {ANOMALY_LOG}" if total_anom else "  (none)"), flush=True)


if __name__ == "__main__":
    main()
