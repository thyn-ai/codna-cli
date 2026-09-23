# Codna vs Cursor vs Codex — fix benchmark

_Generated 2026-06-27 19:01 UTC · 1 repo(s) · each engine runs the same issue headless on its own checkout, no shared infra._

## clone-flask

_repo: https://github.com/pallets/flask · issue: send_file returns the wrong content-type for .webp files_

| Engine | Model | Localized | Context tokens in | Time | Cost | Status |
|---|---|---|--:|--:|--:|---|
| Codna (engine-behind) | claude-sonnet-4-6 | get_debug_flag | 306,568→5,769 (53× smaller) | 46s | $0.850 | ok |
| Cursor | default | src/flask/helpers.py | 112,456 | 95s | — | ok |
| Codex | gpt-5.4 | src/flask/helpers.py, tests/test_helpers.py | 94,188 | 93s | — | ok |

## Notes
- Each engine runs the SAME issue headless on its own checkout — no shared infra.
- **Codna** runs engine-behind (Algenta localization + Monte-Carlo govern gate + Cline) and reports the reduced evidence the agent sees (~5k tokens); Cursor/Codex figures are their own agents' consumption. Codna is inspect-mode (patch ref, no apply); Cursor/Codex edit a throwaway copy.
- Missing engine CLI/creds → that engine is skipped with a reason, never a crash.
