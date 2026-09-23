# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 04:55 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-typer

_repo: https://github.com/fastapi/typer · issue: an Optional argument with a default is treated as required_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | MixedAnnotatedAndDefaultStyleError, MultipleTyperAnnotations | 445,467→5,363 (83× smaller) | 2143 sym idx (18.2s) · 10 recalled | 19s | $0.092 | ok |
| Cursor | composer-2.5-fast | typer/main.py | 162,416 | — | 776s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | tests/test_types.py, typer/main.py | 11,560,690 | — | 401s | $2.282 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
