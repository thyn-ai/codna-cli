# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 07:22 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-requests-oauthlib

_repo: https://github.com/requests/requests-oauthlib · issue: There is a subtle edge-case bug in requests-oauthlib causing incorrect behavior in a common code path; locate the responsible function and fix it with a minimal change._

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | _non_compliant_param_name, instagram_compliance_fix | 58,221→4,682 (12× smaller) | 164 sym idx (2.1s) · 10 recalled | 8s | $0.035 | ok |
| Cursor | composer-2.5-fast | requests_oauthlib/oauth2_session.py | 89,116 | — | 104s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | requests_oauthlib/oauth2_session.py, tests/test_oauth2_sessi | 1,865,885 | — | 190s | $0.564 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
