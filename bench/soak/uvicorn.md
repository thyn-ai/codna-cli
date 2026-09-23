# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 09:41 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-uvicorn

_repo: https://github.com/encode/uvicorn · issue: There is a subtle edge-case bug in uvicorn causing incorrect behavior in a common code path; locate the responsible function and fix it with a minimal change._

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | get_remote_addr | 173,293→5,905 (29× smaller) | 767 sym idx (7.2s) · 10 recalled | 7s | $0.040 | ok |
| Cursor | composer-2.5-fast | uvicorn/protocols/utils.py | 120,653 | — | 412s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | tests/protocols/test_utils.py, uvicorn/protocols/utils.py | 2,122,142 | — | 142s | $0.599 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
