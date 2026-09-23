# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 03:23 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-jsonschema

_repo: https://github.com/python-jsonschema/jsonschema · issue: date-time format validation accepts an invalid string_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | FormatChecker, __repr__, checks, is_css21_color, is_json_poi | 832,541→5,967 (140× smaller) | 718 sym idx (6.1s) · 10 recalled | 13s | $0.058 | ok |
| Cursor | composer-2.5-fast | jsonschema/_format.py | 114,235 | — | 147s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | jsonschema/_format.py, jsonschema/tests/test_format.py | 948,340 | — | 98s | $0.283 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
