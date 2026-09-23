# Codna fix benchmark — Codna vs Cline vs Cursor (vs Codex)

A reusable, **reproducible** side-by-side: the *same* repos, the *same* issues, the *same* reporting,
every run. Any user with the credentials and installed CLIs can reproduce identical results.

- Harness: `cli/bench/benchmark_suite.py` (per-engine runners → one common result row)
- Driver: `cli/bench/bench_langs.py` (the fixed 10-language test set + per-repo streaming + summary)

## What it tests (the FIXED set — do not edit for a comparable run)

`bench_langs.py:MULTILANG` — 10 well-known public repos, one **scoped, plausible bug each**, identical
prompt for every engine:

| # | lang | repo | bug |
|---|---|---|---|
| 1 | python | psf/requests | Session.send no retry on connection timeout |
| 2 | typescript | date-fns | format() SSS drops leading zero <100ms |
| 3 | go | gin-gonic/gin | ShouldBindQuery ignores `default` tag |
| 4 | rust | BurntSushi/ripgrep | --count-matches double-counts overlaps |
| 5 | java | google/gson | Date adapter fails ISO-8601 trailing Z |
| 6 | cpp | nlohmann/json | parse() rejects valid UTF-8 BOM |
| 7 | csharp | JamesNK/Newtonsoft.Json | DateTimeOffset negative-offset sign |
| 8 | ruby | sinatra/sinatra | optional named param fails to match |
| 9 | php | guzzle/guzzle | array query params miss `[]` suffix |
| 10 | swift | Alamofire/Alamofire | URLEncoding doesn't encode literal `+` |

## The three engines (apples-to-apples on gpt-5.4)

| Engine | How it's run | Layers exercised |
|---|---|---|
| **Codna** | `codna fix` **engine-behind** (default) | repo-intelligence localization + Monte-Carlo govern + **Telys reranking** + the vendored-Cline agent. Set `CODNA_BENCH_MODE=local` for the bare execution head (no engine) — NOT the product. |
| **Cline** | public `cline` CLI (`-P openai -m gpt-5.4`) | the agent only (no engine) |
| **Cursor** | `cursor-agent -p` | Cursor's agent (Composer/gpt-5.4) |

## Prerequisites (any user)

1. **`keys.txt`** at repo root (git-ignored), `KEY=value` per line:
   - `CODNA_API_KEY`
   - `OPENAI_API_KEY` (Codna fix model + Cline + Cursor), optionally `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
   - `CURSOR_API_KEY`
2. **Codna runtime:** local runs use Codna's owned packaged runtime. The harness calls
   `codna doctor --start-stack` automatically when needed; run it yourself only when you want
   explicit startup diagnostics.
3. **Binaries**: `bun`, `cline`, `cursor-agent`; `codna` (point `CODNA_BIN` at it).

## Run

```bash
cd <repo>/cli/bench/../..            # repo root
set -a; while IFS= read -r l; do case "$l" in ''|\#*) continue;; esac; export "$l"; done < keys.txt; set +a
export CODNA_BIN=$(command -v codna) CODNA_MEMORY_EMBED=local       # local embedder → no per-call auth
export CLINE_PROVIDER=openai CLINE_MODEL=gpt-5.4 CLINE_API_KEY="$OPENAI_API_KEY"
export CODNA_FIX_MODEL=gpt-5.4 CODNA_TIMEOUT_S=900 CURSOR_TIMEOUT_S=600 CLINE_TIMEOUT_S=600

BENCH_ENGINES=codna,cline,cursor python cli/bench/bench_langs.py        # all 10, streamed per-repo
# knobs: BENCH_LIMIT=1 (first repo only) · BENCH_ENGINES=codna,cursor (drop one) · add ,codex
```

Any engine missing creds/CLI **auto-skips** with a noted reason — it never crashes the run.

## Reporting (identical every run)

Per repo, one row per engine, then an averaged summary:

```
[ 1/10] python  requests   codna ok 11s  in 5.9k/out 0.9k  $0.059~  Telys[726 idx · 10 recalled]  |  cline ok 28s in 145.4k/out 2.6k $0.074~  |  cursor ok 217s in 47.4k/out 5.1k —
```

| column | meaning | trust |
|---|---|---|
| status | `ok` = produced a patch (git diff / patch_ref) | **measured** |
| time | wall-clock seconds | **measured** |
| in / out | input / output tokens | **measured** — for Codna, `in` is the **localized evidence-bundle** token count (post repo-intelligence reduction, ~27× smaller than raw), and `out` is the engine-reported `planner_usage.output_tokens` (the agent's real output tokens). `codna fix --json` now surfaces the full `planner_usage` block (input/output/cache), so out is a measured count, not `—`. |
| `$…~` | cost — **estimated** (tokens × published gpt-5.4 rate; Codna's is engine-reported). `~` = estimate, don't over-trust | estimated |
| `Telys[…]` | symbols auto-indexed + recalled (Codna only) | measured |

## Honesty notes (so results aren't oversold)

- **`ok` = a patch was produced, NOT verified-correct.** A true correctness number needs a test pass
  (fail-before/pass-after) — not yet wired; add a verification stage to claim "fixed."
- **Cost is estimated** and from mixed sources (Codna engine-reported vs others tokens×rate) — lead with
  **time + tokens** (both measured), treat `$` as indicative.
- **Codna must be engine-behind** to be "Codna" — `CODNA_BENCH_MODE=local` is the bare Cline-fork head and
  is not a product comparison.
- **Languages auto-detect** — `codna memory index <repo>` (no `--language`) indexes every present language;
  the benchmark relies on this so Telys works across all 10 repos, not just Python.
