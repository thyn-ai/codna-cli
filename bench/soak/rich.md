# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 04:24 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-rich

_repo: https://github.com/Textualize/rich · issue: Table column width is computed incorrectly with multi-line cells_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | test_init_append_column, test_not_renderable, test_rich_meas | 1,010,097→5,918 (171× smaller) | 1893 sym idx (14.5s) · 10 recalled | 9s | $0.052 | ok |
| Cursor | composer-2.5-fast | — | 323,685 | — | 503s | Composer (subscription — no per-token rate) | success |
| Codex | gpt-5.4 | rich/table.py, tests/test_table.py | 6,328,332 | — | 346s | $1.378 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
