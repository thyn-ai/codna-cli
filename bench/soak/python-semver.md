# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-28 04:29 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-python-semver

_repo: https://github.com/python-semver/python-semver · issue: prerelease comparison orders rc before beta incorrectly_

| Engine | Model | Localized | Context tokens in | Memory (Telys) | Time | Cost | Status |
|---|---|---|--:|---|--:|--:|---|
| Codna +Telys | gpt-5.4 | Version, _cmp, test_should_compare_prerelease_and_build_with | 61,697→2,944 (21× smaller) | 196 sym idx (2.6s) · 10 recalled | 13s | $0.066 | ok |
| Cursor | composer-2.5-fast | src/semver/version.py | 116,571 | — | 266s | Composer (subscription — no per-token rate) | ok |
| Codex | gpt-5.4 | src/semver/version.py, tests/test_compare.py | 1,630,805 | — | 163s | $0.529 est | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna +Telys vs −Telys** is a controlled A/B: identical engine-behind run, the only difference is whether Telys code memory recalls `related_symbols` into the engine signals. Telys indexes with the **real semantic embedder** (key-gated engine `/v1/embeddings`, model auto-tracking the fix provider) — not a lexical stand-in. The two Codna runs are serialized (one local engine) so the timing delta is clean.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
- **Cost**: Codna's `$` is REAL (engine planner usage). Cursor/Codex CLIs report detailed **tokens** (input/output/cache) but NO USD — so their `$ est` is derived from those exact reported tokens × published per-1M rates (Codex/gpt-5.4 computable; Cursor's Composer is subscription-billed → tokens only). Rates overridable via `CODNA_BENCH_RATES_JSON`.
