# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 04:16 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-requests

_repo: https://github.com/psf/requests · issue: session does not persist cookies set during a redirect_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | SessionRedirectMixin, resolve_redirects | 172,422→5,682 (30× smaller) | 726 sym idx (6.0s) · 10 recalled | 7s | $0.050 | ok |
| Cursor | composer-2.5-fast | src/requests/sessions.py | 77,939 | — | 175s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/requests/sessions.py, tests/test_requests.py | 1,555,827 | — | 168s | $0.478 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
