# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 03:21 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-jinja

_repo: https://github.com/pallets/jinja · issue: autoescape misses an edge case in HTML attribute context_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | TestLRUCache, autoescape, select_autoescape, test_clear, tes | 273,825→5,645 (49× smaller) | 1683 sym idx (14.8s) · 10 recalled | 12s | $0.052 | ok |
| Cursor | composer-2.5-fast | src/jinja2/utils.py | 101,219 | — | 175s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/jinja2/filters.py, tests/test_filters.py | 1,471,711 | — | 156s | $0.480 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
